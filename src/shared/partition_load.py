# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""DynamoDB partition load, estimated per partition key value (#1989).

DynamoDB's partition throughput limits (3,000 read units and 1,000 write units per
second, ``DynamoDBPartitionLimits``) apply to one partition key value. A design's
``hot_partition_analysis`` entry is the load on a whole table or index: the total over
all of its key values. Compared with one key value's limit it is an upper bound, reached
only if every request used the same key value, so it never sets a risk's severity on its
own. This module estimates the load on one key value from what the collection holds:

* ``single_key``: every item has the same key value (the key attribute is a constant,
  ``ConstantCalculation`` (#1531), an entity's ``pk_template`` has no column token, or
  the key's source columns hold one distinct value), so every request on the table or
  index uses that value: the busiest key value carries the whole total;
* ``distinct_values``: the key's source columns have a collected number of distinct
  values (a primary key or unique index: the table's row count; else the collector's
  column cardinality), and the total spread evenly over them is the estimate. The
  collection holds no per-value counts, so skew across values is not measured;
* ``not_measured``: the collection holds neither (for example a non-unique column of a
  MySQL source, which collects no column cardinality): the output says "per-key load
  not measured" and the total stays the only figure, as an upper bound.

A key over several attributes (a multi-attribute GSI key, or an item collection whose
entities spell the key from different columns) has at least as many distinct values as
its most selective known part, so that count is the estimate's divisor: a lower bound on
the key's distinct values, which can only raise the per-key estimate.

The percentages are never capped: 303% of one key value's limit is printed as 303%.

No model, network or Strands dependency; the facts layer and the schema-design tools
both import it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from src.contracts.design_metrics import DynamoDBPartitionLimits
from src.shared.engine_names import canonical_engine_id
from src.shared.row_counts import known_row_count
from src.shared.table_keys import normalize_table_name

KeyBasis = Literal["single_key", "distinct_values", "not_measured"]
SINGLE_KEY: KeyBasis = "single_key"
DISTINCT_VALUES: KeyBasis = "distinct_values"
NOT_MEASURED: KeyBasis = "not_measured"

#: How a column set's distinct-value count was collected.
UNIQUE_KEY = "unique_key"
COLUMN_STATISTICS = "column_statistics"

_TOKEN = re.compile(r"\{([^{}]+)\}")


def _limits() -> DynamoDBPartitionLimits:
    return DynamoDBPartitionLimits()


def partition_limit(operation: object) -> float:
    """One key value's limit for a ``read`` or ``write`` entry."""
    limits = _limits()
    return limits.read_units_per_second if operation == "read" else limits.write_units_per_second


def pct_of_limit(units: float, limit: float) -> float:
    """``units / limit × 100``, never capped, to four decimals."""
    return round(units / limit * 100, 4) if limit else 0.0


# ---------------------------------------------------------------------------
# Distinct values of a source table's columns
# ---------------------------------------------------------------------------


def _lower(columns: Iterable[object]) -> list[str]:
    return [str(c).strip().lower() for c in columns or [] if str(c or "").strip()]


def key_distinct_values(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The column sets of a collector table whose number of distinct values is collected.

    A primary key or unique index has as many distinct values as the table has rows (the
    collector's row count, at least 1: a key that takes writes holds a value); a column
    with a collected ``cardinality`` has that many. Sorted, one entry per column set,
    as ``{"columns", "distinct_values", "basis"}``. Empty when the table has no row
    count and no column statistics.
    """
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    rows = known_row_count(table)
    if rows is not None:
        unique: list[list[str]] = []
        if table.get("primary_key"):
            unique.append(_lower(table.get("primary_key") or []))
        for index in table.get("indexes") or []:
            if isinstance(index, Mapping) and (index.get("is_unique") or index.get("is_primary")):
                unique.append(_lower(index.get("columns") or []))
        for columns in unique:
            if columns:
                key = tuple(sorted(set(columns)))
                out[key] = {
                    "columns": list(key),
                    "distinct_values": max(rows, 1),
                    "basis": UNIQUE_KEY,
                }
    for column in table.get("columns") or []:
        if not isinstance(column, Mapping):
            continue
        card = column.get("cardinality")
        name = _lower([column.get("column_name")])
        if not name or isinstance(card, bool) or not isinstance(card, int) or card < 0:
            continue
        key = (name[0],)
        if key not in out:
            out[key] = {
                "columns": [name[0]],
                "distinct_values": max(card, 1),
                "basis": COLUMN_STATISTICS,
            }
    return [out[k] for k in sorted(out)]


class DistinctValues:
    """Look up the distinct values of a source table's columns, by table id or name."""

    def __init__(self, by_table: Mapping[str, Iterable[Mapping[str, Any]]] | None = None):
        self._sets: dict[str, list[tuple[frozenset[str], int]]] = {}
        for table, entries in (by_table or {}).items():
            rows = [
                (frozenset(_lower(e.get("columns") or [])), int(e.get("distinct_values") or 0))
                for e in entries or []
                if isinstance(e, Mapping) and e.get("columns")
            ]
            self._sets.setdefault(normalize_table_name(str(table)), []).extend(rows)

    @classmethod
    def from_collector_tables(cls, tables: Iterable[Mapping[str, Any]] | None) -> DistinctValues:
        """From collector tables (full-contract ``database_schema.tables`` or ``Agent*``)."""
        return cls(
            {
                str(t.get("table_id") or t.get("table_name") or ""): key_distinct_values(t)
                for t in tables or []
                if isinstance(t, Mapping)
            }
        )

    @classmethod
    def from_design_metrics(cls, metrics: Mapping[str, Any] | None) -> DistinctValues:
        """From a ``DesignMetricsContract``'s ``tables[].key_distinct_values`` (JSON)."""
        return cls(
            {
                str(t.get("table_id") or ""): t.get("key_distinct_values") or []
                for t in (metrics or {}).get("tables") or []
                if isinstance(t, Mapping)
            }
        )

    def of(self, table: str, columns: Iterable[str]) -> int | None:
        """Distinct values of ``columns`` together in ``table``, or None when not collected.

        Columns that contain a unique key have the key's count; otherwise the most
        selective single column known is a lower bound on the combination's count."""
        wanted = frozenset(_lower(columns))
        if not wanted:
            return None
        sets = self._sets.get(normalize_table_name(table), [])
        unique = [n for cols, n in sets if cols and cols <= wanted]
        if unique:
            return max(unique)
        single = [n for cols, n in sets if len(cols) == 1 and cols <= wanted]
        return max(single) if single else None


# ---------------------------------------------------------------------------
# A design key's distinct values
# ---------------------------------------------------------------------------

#: A key attribute's distinct values: an int, or None when not collected.
_Count = int | None


def _attributes(table: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out = [a for a in table.get("attributes") or [] if isinstance(a, Mapping)]
    for entity in table.get("entities") or []:
        if isinstance(entity, Mapping):
            out += [a for a in entity.get("attributes") or [] if isinstance(a, Mapping)]
    return out


def _attribute_count(attribute: Mapping[str, Any], values: DistinctValues) -> _Count:
    calc = attribute.get("calculation")
    if isinstance(calc, Mapping):
        return 1 if calc.get("type") == "constant" else None
    if attribute.get("join"):
        return None
    source = attribute.get("source_column")
    columns = [source] if isinstance(source, str) else list(source or [])
    if not columns:
        return None
    return values.of(str(attribute.get("source_table") or ""), columns)


def _template_count(entity: Mapping[str, Any], values: DistinctValues) -> _Count:
    tokens = _TOKEN.findall(str(entity.get("pk_template") or ""))
    if not tokens:
        return 1  # a static template: every item of the entity has the same key value
    return values.of(str(entity.get("source_table") or ""), tokens)


def _combine(counts: list[_Count]) -> _Count:
    """A key spelled by several parts: all constant is one value; otherwise the most
    selective known part is a lower bound; nothing known beyond constants is None."""
    if not counts:
        return None
    if all(c == 1 for c in counts):
        return 1
    known = [c for c in counts if c is not None and c > 1]
    return max(known) if known else None


def _key_attribute_count(
    table: Mapping[str, Any], name: str, values: DistinctValues, base_key: bool
) -> _Count:
    """Distinct values of one key attribute over every item that carries it (the
    entities' parts combine as :func:`_combine` says)."""
    found = [a for a in _attributes(table) if str(a.get("name") or "") == name]
    counts = [_attribute_count(a, values) for a in found]
    if base_key and table.get("entities"):
        # An item collection's key is spelled per entity by its pk_template.
        counts += [
            _template_count(e, values)
            for e in table.get("entities") or []
            if isinstance(e, Mapping)
        ]
    return _combine(counts)


def _key_names(keys: object) -> list[str]:
    items = [keys] if isinstance(keys, Mapping) else keys if isinstance(keys, list) else []
    return [
        str(k.get("attribute_name"))
        for k in items
        if isinstance(k, Mapping) and k.get("attribute_name")
    ]


def key_distinct_count(
    design: Mapping[str, Any], table_name: str, gsi_name: str | None, values: DistinctValues
) -> _Count:
    """Distinct partition key values of a design table (``gsi_name`` None) or its GSI."""
    table = next(
        (
            t
            for t in design.get("table_definitions") or []
            if isinstance(t, Mapping) and str(t.get("table_name") or "") == table_name
        ),
        None,
    )
    if table is None:
        return None
    if gsi_name:
        gsi = next(
            (
                g
                for g in table.get("gsis") or []
                if isinstance(g, Mapping) and str(g.get("gsi_name") or "") == gsi_name
            ),
            None,
        )
        if gsi is None:
            return None
        names, base = _key_names(gsi.get("partition_key")), False
    else:
        names, base = _key_names(table.get("partition_key")), True
    return _combine([_key_attribute_count(table, n, values, base) for n in names])


# ---------------------------------------------------------------------------
# One entry's load per key value
# ---------------------------------------------------------------------------


def _units(entry: Mapping[str, Any], limit: float) -> float:
    try:
        return float(entry.get("rcu_or_wcu_per_second"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # A design without the units: its own (possibly capped) utilization.
        return float(entry.get("utilization_pct") or 0.0) * limit / 100


def partition_load(
    entry: Mapping[str, Any],
    design: Mapping[str, Any],
    values: DistinctValues,
    engine: str = "",
) -> dict[str, Any]:
    """One ``hot_partition_analysis`` entry, with its load per key value estimated.

    ``total_*`` is the entry's own load (the table or index total, an upper bound for one
    key value) and ``per_key_*`` the estimate, None when the key's distinct values are not
    collected (``per_key_basis`` ``not_measured``)."""
    operation = "read" if entry.get("operation") == "read" else "write"
    limit = partition_limit(operation)
    total = round(_units(entry, limit), 4)
    gsi = entry.get("gsi_name") or None
    count = key_distinct_count(design, str(entry.get("table_name") or ""), gsi, values)
    if count is None:
        basis, per_key = NOT_MEASURED, None
    else:
        basis = SINGLE_KEY if count == 1 else DISTINCT_VALUES
        per_key = round(total / count, 4)
    return {
        "engine": canonical_engine_id(engine),
        "table_name": str(entry.get("table_name") or ""),
        "gsi_name": str(gsi) if gsi else None,
        "operation": operation,
        "partition_limit": limit,
        "total_units_per_second": total,
        "total_pct_of_key_limit": pct_of_limit(total, limit),
        "distinct_key_values": count,
        "per_key_units_per_second": per_key,
        "per_key_pct_of_limit": None if per_key is None else pct_of_limit(per_key, limit),
        "per_key_basis": basis,
        "query_ids": sorted({str(q) for q in entry.get("contributing_patterns") or [] if q}),
    }


def over_key_limit(row: Mapping[str, Any]) -> bool:
    """The estimated busiest key value is above its limit: the one HIGH case."""
    pct = row.get("per_key_pct_of_limit")
    return isinstance(pct, (int, float)) and pct > 100


def hot_partition_rows(
    schema_designs: Mapping[str, Mapping[str, Any]] | None,
    source_tables: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Every design entry whose total is above the at-risk share of one key value's
    limit, with its per-key estimate (:func:`partition_load`), by engine then table.

    ``source_tables`` are the collector's tables (``database_schema.tables``); without
    them every key reads as not measured unless the design makes it a single key."""
    threshold = _limits().at_risk_threshold_pct
    values = DistinctValues.from_collector_tables(source_tables)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for engine in sorted(schema_designs or {}, key=canonical_engine_id):
        canonical = canonical_engine_id(engine)
        if canonical in seen:
            continue
        seen.add(canonical)
        design = (schema_designs or {})[engine] or {}
        if not isinstance(design, Mapping):
            continue
        for entry in design.get("hot_partition_analysis") or []:
            if not isinstance(entry, Mapping):
                continue
            row = partition_load(entry, design, values, canonical)
            if row["total_pct_of_key_limit"] > threshold:
                rows.append(row)
    return sorted(
        rows,
        key=lambda r: (r["engine"], r["table_name"], r["gsi_name"] or "", r["operation"]),
    )


__all__ = [
    "COLUMN_STATISTICS",
    "DISTINCT_VALUES",
    "NOT_MEASURED",
    "SINGLE_KEY",
    "UNIQUE_KEY",
    "DistinctValues",
    "KeyBasis",
    "hot_partition_rows",
    "key_distinct_count",
    "key_distinct_values",
    "over_key_limit",
    "partition_limit",
    "partition_load",
    "pct_of_limit",
]
