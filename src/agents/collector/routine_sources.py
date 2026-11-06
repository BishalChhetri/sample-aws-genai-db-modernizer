# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The exact CREATE statement of each collected routine, and how far each was read (#1652).

Routine migration carries each PostgreSQL and MySQL function, procedure, trigger and
event to Aurora (ADR-034), so the collection keeps the statement that creates it:

- PostgreSQL: a row's ``definition`` is the server's own ``pg_get_functiondef`` /
  ``pg_get_triggerdef`` text;
- MySQL: the CREATE is rebuilt from the catalog columns the rows carry
  (:mod:`src.agents.collector.mysql_routine_create`): the offline script cannot run
  SHOW CREATE.

The statements keep their string literals: they are code the target runs as written.
So they never go in the collector output, a checkpoint or the raw file's stored copy,
which are redacted (#1625). They go in the job's routine sources (ADR-034 Decision 4):
gzip chunks of at most :data:`~src.contracts.routine_sources.MAX_CHUNK_SOURCES`
routines under ``collector/routine_sources/``, listed by ``collector/routine_sources.json``
(:class:`~src.contracts.routine_sources.RoutineSourcesContract`), which no report, model
input or API response reads. :func:`write_routine_sources` holds one chunk at a time and
:func:`read_routine_sources` loads one chunk at a time. A live collector hands its
statements to the collector handler on the contract's private ``_routine_sources``.
Pure, except :func:`write_routine_sources` and :func:`read_routine_sources`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.agents.collector import mysql_routine_create as mysql_create
from src.contracts.collector_output import RoutineCapture
from src.contracts.routine_sources import (
    MAX_CHUNK_SOURCES,
    RoutineSession,
    RoutineSource,
    RoutineSourcesChunk,
    RoutineSourcesContract,
    RoutineSourcesPart,
)

logger = logging.getLogger(__name__)

ROUTINE_SOURCES_FILE = "collector/routine_sources.json"
ROUTINE_SOURCES_DIR = "collector/routine_sources"
SOURCE_ENGINES = ("postgresql", "mysql")
_BODY_STATUSES = ("collected", "not_readable", "opted_out", "not_supported")
_DEPENDENCY_KINDS = ("table", "view", "sequence", "function", "type", "trigger")
_REASON_CAP = 500
_OPTED_OUT = "the collection ran with include_routine_bodies = 0: listed by name and size only"
_REASONS: dict[tuple[str, str], str] = {
    ("mysql", "not_readable"): (
        "the collection user cannot read this body: grant SHOW_ROUTINE (MySQL 8.0.20+, "
        "else global SELECT), or collect as the routine's DEFINER"
    ),
    ("mysql", "not_supported"): (
        "no loadable CREATE can be rebuilt from the catalog for this routine (a language "
        "other than SQL, or a catalog row without its parameters)"
    ),
    ("postgresql", "not_readable"): "the collection user cannot read this body",
    ("postgresql", "not_supported"): "the catalog returned no CREATE for this routine",
}

RoutineKey = tuple[str, str, str]  # (kind, schema, specific name), exact case


@dataclass(frozen=True)
class RoutineSourcesBuild:
    """The CREATE statements of one collection, and the status of each routine whose
    collected body could not be made into one (``not_supported``)."""

    sources: list[RoutineSource] = field(default_factory=list)
    statuses: dict[RoutineKey, str] = field(default_factory=dict)


def _rows(items: object) -> list[dict[str, Any]]:
    return [r for r in items if isinstance(r, dict)] if isinstance(items, list | tuple) else []


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _schema(row: Mapping[str, Any]) -> str:
    return str(row.get("schema_name") or "").strip()


def _setting(row: Mapping[str, Any], key: str) -> str | None:
    """A session setting as collected: '' (MySQL's empty sql_mode) is a value."""
    value = row.get(key)
    if key == "sql_mode":
        return mysql_create.sql_mode_text(value)
    return value if isinstance(value, str) else None


def _count(value: object, *, minimum: int) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        number = int(str(value))
    except ValueError:
        return None
    return number if number >= minimum else None


def _flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in ("t", "true", "1", "yes"):
        return True
    if text in ("f", "false", "0", "no"):
        return False
    return None


def procedure_kind(row: Mapping[str, Any]) -> str | None:
    return {"FUNCTION": "function", "PROCEDURE": "procedure"}.get(
        str(row.get("routine_type") or "").upper()
    )


def routine_key(kind: str, row: Mapping[str, Any]) -> RoutineKey:
    """A routine's identity in one collection: a trigger by table and name (PostgreSQL
    names are unique per table), an event by name, a routine by its specific name
    (PostgreSQL overloads differ there; a MySQL procedure and function sharing a name
    differ by kind)."""
    if kind == "trigger":
        name = f"{row.get('table_name') or ''}.{row.get('trigger_name') or ''}"
    elif kind == "event":
        name = str(row.get("event_name") or "")
    else:
        name = str(row.get("specific_name") or row.get("routine_name") or "")
    return (kind, _schema(row), name)


def body_status_reason(engine: str, status: object) -> str | None:
    if status == "opted_out":
        return _OPTED_OUT
    return _REASONS.get((engine, str(status)))


def _session(row: Mapping[str, Any]) -> RoutineSession:
    return RoutineSession(
        sql_mode=_setting(row, "sql_mode"),
        character_set_client=_setting(row, "character_set_client"),
        collation_connection=_setting(row, "collation_connection"),
        database_collation=_setting(row, "database_collation"),
        time_zone=_setting(row, "time_zone"),
    )


def _parameters_by_routine(rows: object) -> dict[RoutineKey, list[dict[str, Any]]]:
    out: dict[RoutineKey, list[dict[str, Any]]] = {}
    for r in _rows(rows):
        kind = procedure_kind(r)
        if kind and r.get("routine_name"):
            out.setdefault((kind, _schema(r), str(r["routine_name"])), []).append(r)
    return out


def _body(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def iter_routine_sources(
    engine: str,
    *,
    procedures: object,
    triggers: object,
    events: object = (),
    parameters: object = (),
    statuses: dict[RoutineKey, str] | None = None,
) -> Iterator[RoutineSource]:
    """The CREATE statement of each row whose ``body_status`` is ``collected``, one at a
    time. A row with any other status, or none (a collection before #1652), has no
    source. A collected row that makes no CREATE is recorded ``not_supported`` in
    ``statuses``. Only PostgreSQL and MySQL are carried; any other engine has none."""
    if engine not in SOURCE_ENGINES:
        return
    found = statuses if statuses is not None else {}
    seen: set[RoutineKey] = set()

    def source(
        key: RoutineKey, row: Mapping[str, Any], text: str | None, origin: str, rid: str
    ) -> RoutineSource | None:
        if key in seen:
            return None
        seen.add(key)
        if not text or not rid or not key[2]:
            found[key] = "not_supported"
            return None
        return RoutineSource(
            kind=key[0],  # type: ignore[arg-type]
            routine_id=rid,
            schema_name=_text(row.get("schema_name")),
            specific_name=key[2],
            create_statement=text,
            origin=origin,  # type: ignore[arg-type]
            session=_session(row),
        )

    params = _parameters_by_routine(parameters) if engine == "mysql" else {}
    for row in _rows(procedures):
        kind = procedure_kind(row)
        if kind is None or row.get("body_status") != "collected":
            continue
        key = routine_key(kind, row)
        if engine == "postgresql":
            text, origin = _body(row.get("definition")), "pg_get_functiondef"
        else:
            text = mysql_create.routine_create(row, params.get(key, []))
            origin = "information_schema"
        made = source(key, row, text, origin, str(row.get("routine_name") or ""))
        if made is not None:
            yield made

    trigger_rows = _rows(triggers)
    follows = mysql_create.trigger_follows(trigger_rows) if engine == "mysql" else {}
    for row in trigger_rows:
        if row.get("body_status") != "collected":
            continue
        key = routine_key("trigger", row)
        if engine == "postgresql":
            text, origin = _body(row.get("definition")), "pg_get_triggerdef"
        else:
            previous = follows.get(
                (_schema(row), str(row.get("table_name") or ""), str(row.get("trigger_name") or ""))
            )
            text, origin = mysql_create.trigger_create(row, previous), "information_schema"
        made = source(key, row, text, origin, str(row.get("trigger_name") or ""))
        if made is not None:
            yield made

    if engine == "mysql":
        for row in _rows(events):
            if row.get("body_status") != "collected":
                continue
            made = source(
                routine_key("event", row),
                row,
                mysql_create.event_create(row),
                "information_schema",
                str(row.get("event_name") or ""),
            )
            if made is not None:
                yield made


def build_routine_sources(
    engine: str,
    *,
    procedures: object,
    triggers: object,
    events: object = (),
    parameters: object = (),
) -> RoutineSourcesBuild:
    """:func:`iter_routine_sources`, gathered: every source and every status."""
    statuses: dict[RoutineKey, str] = {}
    sources = list(
        iter_routine_sources(
            engine,
            procedures=procedures,
            triggers=triggers,
            events=events,
            parameters=parameters,
            statuses=statuses,
        )
    )
    return RoutineSourcesBuild(sources=sources, statuses=statuses)


def collection_engine(raw: object) -> str:
    """The source engine a raw collection's metadata names (the one detector, #1177)."""
    if not isinstance(raw, Mapping):
        return ""
    from src.tools.database.offline_parser import detect_source_engine

    metadata = raw.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = None
    return detect_source_engine(metadata if isinstance(metadata, dict) else None)[0]


def iter_collection_sources(raw: object) -> Iterator[RoutineSource]:
    """:func:`iter_routine_sources` for a raw collection file, read as given (before any
    redaction), in the engine its metadata names."""
    if not isinstance(raw, Mapping):
        return iter(())
    return iter_routine_sources(
        collection_engine(raw),
        procedures=raw.get("procedures") or [],
        triggers=raw.get("triggers") or [],
        events=raw.get("events") or [],
        parameters=raw.get("routine_parameters") or [],
    )


def routine_sources_from_collection(raw: object) -> RoutineSourcesBuild:
    """:func:`build_routine_sources` for a raw collection file, read as given."""
    if not isinstance(raw, Mapping):
        return RoutineSourcesBuild()
    return build_routine_sources(
        collection_engine(raw),
        procedures=raw.get("procedures") or [],
        triggers=raw.get("triggers") or [],
        events=raw.get("events") or [],
        parameters=raw.get("routine_parameters") or [],
    )


def apply_body_statuses(
    build: RoutineSourcesBuild,
    engine: str,
    *,
    procedures: object,
    triggers: object,
    events: object = (),
) -> None:
    """Set each row's ``body_status`` (a failed rebuild becomes ``not_supported``) and
    ``body_status_reason``, in place. A row without a status (a collection before
    #1652) is left as it is."""
    groups: tuple[tuple[Callable[[Mapping[str, Any]], str | None], object], ...] = (
        (procedure_kind, procedures),
        (lambda _row: "trigger", triggers),
        (lambda _row: "event", events),
    )
    for kind_of, rows in groups:
        for row in _rows(rows):
            kind = kind_of(row)
            if kind is None:
                continue
            status = build.statuses.get(routine_key(kind, row), row.get("body_status"))
            if status not in _BODY_STATUSES:
                continue
            row["body_status"] = status
            reason = body_status_reason(engine, status)
            if reason:
                row["body_status_reason"] = reason[:_REASON_CAP]


def routine_capture(raw: object, *, listed: int) -> dict[str, Any] | None:
    """``RoutineCapture`` fields from what the collection recorded (``metadata.
    routine_capture``, or a live collector's), with ``listed`` the routines, triggers
    and events it holds; None for a collection that records none. A list shorter than
    the server's count is logged: a list came back cut."""
    if not isinstance(raw, Mapping):
        return None
    counted = _count(raw.get("routines_counted"), minimum=0)
    if counted is not None and listed < counted:
        logger.warning(
            "The collection lists %d routines, triggers and events of the %d the server "
            "counts: a list came back empty. Raise the server's max_allowed_packet and "
            "collect again.",
            listed,
            counted,
        )
    privileges = raw.get("privileges_not_seen")
    return {
        "include_bodies": raw.get("include_bodies") is not False,
        "routines_counted": counted,
        "routines_listed": listed,
        "privileges_not_seen": (
            [p for p in privileges if isinstance(p, str)] if isinstance(privileges, list) else []
        ),
    }


def routine_capture_model(raw: object) -> RoutineCapture | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return RoutineCapture.model_validate(raw)
    except ValidationError:
        logger.warning("Ignored a routine capture the contract cannot read")
        return None


def _dependencies(raw: object) -> list[dict[str, str]] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, list):
        return None
    return [
        {"kind": d["kind"], "name": str(d["name"])[:1024]}
        for d in raw
        if isinstance(d, Mapping) and d.get("kind") in _DEPENDENCY_KINDS and d.get("name")
    ]


def body_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    status = row.get("body_status")
    reason = row.get("body_status_reason")
    return {
        "body_status": status if status in _BODY_STATUSES else None,
        "body_status_reason": str(reason)[:_REASON_CAP] if reason else None,
        "body_chars": _count(row.get("body_chars"), minimum=0),
    }


def procedure_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """``Procedure`` keyword arguments for the routine-migration fields of a raw row."""
    security = row.get("security_definer")
    if security is None and row.get("security_type") is not None:
        security = str(row["security_type"]).upper() == "DEFINER"
    return {
        "specific_name": _text(row.get("specific_name")),
        "owner": _text(row.get("owner") or row.get("definer")),
        "security_definer": _flag(security),
        "depends_on": _dependencies(row.get("depends_on")),
        **body_fields(row),
    }


def trigger_fields(row: Mapping[str, Any], follows: str | None = None) -> dict[str, Any]:
    """``Trigger`` keyword arguments for the routine-migration fields of a raw row:
    PostgreSQL's function, or the MySQL trigger it follows, as its dependency."""
    depends: list[dict[str, str]] | None = None
    if row.get("function_signature"):
        depends = [{"kind": "function", "name": str(row["function_signature"])[:1024]}]
    elif follows:
        depends = [{"kind": "trigger", "name": follows}]
    return {
        "owner": _text(row.get("definer") or row.get("owner")),
        "action_order": _count(row.get("action_order"), minimum=1),
        "depends_on": depends,
        **body_fields(row),
    }


def event_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """``ScheduledEvent`` keyword arguments for a raw MySQL event row."""
    status = str(row.get("status") or "").upper()
    schedule = mysql_create.event_schedule(row)
    return {
        "event_id": str(row.get("event_name") or ""),
        "event_name": str(row.get("event_name") or ""),
        "schema_name": _text(row.get("schema_name")),
        "definition": row.get("definition") if isinstance(row.get("definition"), str) else None,
        "owner": _text(row.get("definer")),
        "status": None if not status else ("enabled" if status == "ENABLED" else "disabled"),
        "schedule": schedule[:200] if schedule else None,
        "sql_mode": _setting(row, "sql_mode"),
        **body_fields(row),
    }


def attach_to_contract(contract: Any, sources: Sequence[RoutineSource]) -> None:
    """Hand a live collector's CREATE statements to the collector handler."""
    contract._routine_sources = list(sources)


def routine_sources_of(contract: Any) -> list[RoutineSource]:
    return list(getattr(contract, "_routine_sources", None) or [])


def _part_path(number: int) -> str:
    return f"routine_sources/part-{number:04d}.json.gz"


def _job_prefix(database_name: str, job_id: str) -> str:
    return f"{database_name}/{job_id}/collector"


def write_routine_sources(
    store: Any,
    database_name: str,
    job_id: str,
    collector_output: Mapping[str, Any],
    sources: Iterable[RoutineSource],
    *,
    chunk_size: int = MAX_CHUNK_SOURCES,
) -> str | None:
    """Write the job's routine sources, one gzip chunk of at most ``chunk_size``
    routines at a time, then their index ``<db>/<job>/collector/routine_sources.json``,
    and return the index's key. ``sources`` is read once, as it comes: only one chunk is
    held. Nothing is written (None) without sources, or for an engine routine migration
    does not carry. A part left by an earlier write of the same job is deleted.

    Raises:
        ValueError: two sources name the same routine (kind, schema, specific name).
    """
    from src.tools.sql.redact import collector_output_engine

    engine = collector_output_engine(collector_output)
    if engine not in SOURCE_ENGINES:
        return None
    if not 1 <= chunk_size <= MAX_CHUNK_SOURCES:
        raise ValueError(f"chunk_size must be 1 to {MAX_CHUNK_SOURCES}, got {chunk_size}")
    prefix = _job_prefix(database_name, job_id)
    parts: list[RoutineSourcesPart] = []
    seen: set[bytes] = set()
    buffer: list[RoutineSource] = []

    def flush() -> None:
        number = len(parts) + 1
        chunk = RoutineSourcesChunk(part=number, sources=buffer)
        data = gzip.compress(chunk.model_dump_json().encode("utf-8"), mtime=0)
        path = _part_path(number)
        store.write_bytes(f"{prefix}/{path}", data)
        parts.append(
            RoutineSourcesPart(
                path=path, count=len(buffer), sha256=hashlib.sha256(data).hexdigest()
            )
        )
        buffer.clear()

    for source in sources:
        # A short digest of each key, not the key: the check stays small at any size.
        mark = hashlib.blake2b(repr(source.key()).encode("utf-8"), digest_size=12).digest()
        if mark in seen:
            raise ValueError(
                f"two routine sources for {source.kind} {source.schema_name}.{source.specific_name}"
            )
        seen.add(mark)
        buffer.append(source)
        if len(buffer) >= chunk_size:
            flush()
    if buffer:
        flush()
    stale = len(parts) + 1
    while store.exists(f"{prefix}/{_part_path(stale)}"):
        store.delete(f"{prefix}/{_part_path(stale)}")
        stale += 1
    if not parts:
        return None
    metadata = collector_output.get("metadata")
    source_db = metadata.get("source_database") if isinstance(metadata, Mapping) else None
    version = source_db.get("version") if isinstance(source_db, Mapping) else None
    index = RoutineSourcesContract(
        job_id=job_id,
        source_engine=engine,  # type: ignore[arg-type]
        source_version=str(version) if version else None,
        total=sum(p.count for p in parts),
        parts=parts,
    )
    key = f"{database_name}/{job_id}/collector/routine_sources.json"  # ROUTINE_SOURCES_FILE
    store.write_json(key, index.model_dump(mode="json"))
    return key


def read_routine_sources(store: Any, database_name: str, job_id: str) -> Iterator[RoutineSource]:
    """The job's routine sources in the order written, reading one chunk at a time;
    nothing when the job has none.

    Raises:
        ValueError: a chunk's bytes do not match the index, or it holds another count.
    """
    from src.shared.contract_versioning import load_contract

    key = f"{database_name}/{job_id}/{ROUTINE_SOURCES_FILE}"
    if not store.exists(key):
        return
    index = load_contract(RoutineSourcesContract, store.read_json(key), artifact=key)
    prefix = _job_prefix(database_name, job_id)
    for part in index.parts:
        data = store.read_bytes(f"{prefix}/{part.path}")
        if hashlib.sha256(data).hexdigest() != part.sha256:
            raise ValueError(f"{part.path} does not match its index entry")
        chunk = load_contract(
            RoutineSourcesChunk, json.loads(gzip.decompress(data)), artifact=f"{key} {part.path}"
        )
        del data
        if len(chunk.sources) != part.count:
            raise ValueError(f"{part.path} holds {len(chunk.sources)} routines, not {part.count}")
        yield from chunk.sources
        del chunk  # before the next chunk is read: one chunk is held at a time
