"""Read-only, deterministic inputs for gold-SQL schema-linking annotation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


GROUPS = (
    "bird_dev",
    "bird_interact_full",
    "bird_interact_lite",
    "spider_dev",
    "spider_test",
)
GROUP_SIZES = {
    "bird_dev": 1534,
    "bird_interact_full": 410,
    "bird_interact_lite": 195,
    "spider_dev": 1034,
    "spider_test": 2147,
}
TOTAL_TASKS = 5320
FEATURES = (
    "cte_or_subquery",
    "set_operation",
    "wildcard",
    "json",
    "lateral_or_table_function",
    "using_or_natural_join",
    "schema_qualified",
    "sqlite_double_quote_ambiguity",
    "ordinary",
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnnotationTask:
    """One frozen question and its blind-annotation inputs and comparison data."""

    task_key: str
    group: str
    partition: str
    external_id: str | int
    database_id: str
    dialect: str
    gold_sql: str
    sql_sha256: str
    schema: dict[str, Any]
    schema_sha256: str
    source_schema_sha256: str
    source_hash: str
    reuse_key: tuple[str, str, str]
    features: tuple[str, ...]
    native_linked_schema: dict[str, tuple[str, ...]]
    rc_linked_schema: dict[str, tuple[str, ...]]
    conservative_reference: dict[str, Any]
    source_reference: dict[str, Any]

    @property
    def canonical_schema(self) -> dict[str, Any]:
        """Alias that makes the model-facing catalog explicit to consumers."""
        return self.schema


def canonical_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Assign stable closed IDs to a frozen physical table/column catalog."""
    tables = schema.get("tables", schema)
    if not isinstance(tables, dict):
        raise ValueError("Schema tables must be an object")

    output_tables = []
    table_by_id, table_id_by_name = {}, {}
    column_by_id, column_id_by_name = {}, {}
    column_number = 1
    for table_number, (table_key, table) in enumerate(
            sorted(tables.items(), key=lambda item: (str(item[0]).casefold(), str(item[0]))), start=1):
        if not isinstance(table, dict):
            raise ValueError(f"Schema table {table_key!r} is not an object")
        table_name = str(table.get("table_name", table_key))
        columns = table.get("columns", {})
        if not isinstance(columns, dict):
            raise ValueError(f"Schema columns for {table_name!r} must be an object")
        table_id = f"T{table_number}"
        canonical_columns = []
        column_id_by_name[table_name] = {}
        for column_key, column in sorted(columns.items(), key=lambda item: (str(item[0]).casefold(), str(item[0]))):
            if not isinstance(column, dict):
                raise ValueError(f"Schema column {table_name}.{column_key} is not an object")
            column_name = str(column.get("column_name", column_key))
            column_id = f"C{column_number}"
            column_number += 1
            canonical_columns.append({"id": column_id, "name": column_name,
                                      "type": str(column.get("column_type", ""))})
            column_by_id[column_id] = {"table": table_name, "column": column_name}
            column_id_by_name[table_name][column_name] = column_id
        output_tables.append({"id": table_id, "name": table_name, "columns": tuple(canonical_columns)})
        table_by_id[table_id] = table_name
        table_id_by_name[table_name] = table_id
    return {
        "tables": tuple(output_tables),
        "table_by_id": table_by_id,
        "table_id_by_name": table_id_by_name,
        "column_by_id": column_by_id,
        "column_id_by_name": column_id_by_name,
    }


def conservative_reference(sql: str, dialect: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Derive a parser-only reference from frozen SQL and its physical catalog."""
    tables = schema.get("tables")
    if not isinstance(tables, (list, tuple)):
        raise ValueError("Canonical schema has no tables")
    physical = {
        table["name"]: [column["name"] for column in table["columns"]]
        for table in tables
        if isinstance(table, dict) and isinstance(table.get("name"), str)
        and isinstance(table.get("columns"), (list, tuple))
    }
    normalized_sql, parser_dialect = sql, "sqlite" if dialect.casefold() == "sqlite" else "postgres"
    if parser_dialect == "sqlite":
        physical = {table.lower(): [column.lower() for column in columns]
                    for table, columns in physical.items()}
        try:
            import sqlglot
            from sqlglot import exp
            tree = sqlglot.parse_one(sql, read="sqlite")
            for identifier in tree.find_all(exp.Identifier):
                identifier.set("this", identifier.name.lower())
            normalized_sql = tree.sql(dialect="sqlite")
        except Exception:
            pass
    from scripts.rc_evaluation.deepeye.evaluation import schema_coverage
    coverage = schema_coverage(normalized_sql, physical, dialect=parser_dialect)
    if coverage.get("status") != "available":
        return {"status": "unavailable", "reason": coverage.get("reason"), "tables": (), "columns": ()}
    referenced_tables = tuple(coverage.get("reference_tables", ()))
    referenced_columns = tuple(tuple(column) for column in coverage.get("reference_columns", ()))
    physical_columns = {(table, column) for table, columns in physical.items() for column in columns}
    if not set(referenced_tables) <= set(physical) or not set(referenced_columns) <= physical_columns:
        return {"status": "unavailable", "reason": "reference_elements_not_in_frozen_schema",
                "tables": (), "columns": ()}
    return {"status": "available", "reason": None,
            "tables": referenced_tables, "columns": referenced_columns}


def feature_tags(sql: str, dialect: str) -> tuple[str, ...]:
    """Return stable, intentionally conservative SQL-feature strata."""
    if not isinstance(sql, str) or not isinstance(dialect, str):
        raise ValueError("SQL and dialect must be strings")
    lowered = sql.casefold()
    tags = []
    if re.search(r"\bwith\b|\(\s*select\b", lowered):
        tags.append("cte_or_subquery")
    if re.search(r"\b(?:union(?:\s+all)?|intersect|except)\b", lowered):
        tags.append("set_operation")
    if re.search(r"\b(?:select|,)\s*(?:distinct\s+)?(?:[\w`\"]+\s*\.\s*)?\*", lowered):
        tags.append("wildcard")
    if "->" in sql or re.search(r"\b(?:jsonb?_(?:each|array_elements|extract)|json_extract)\b", lowered):
        tags.append("json")
    if re.search(r"\blateral\b|\b(?:jsonb?_each|unnest|generate_series)\s*\(", lowered):
        tags.append("lateral_or_table_function")
    if re.search(r"\b(?:using\s*\(|natural\s+(?:left\s+|right\s+|full\s+|inner\s+|cross\s+)?join)\b", lowered):
        tags.append("using_or_natural_join")
    if re.search(r"\b(?:from|join|update|into)\s+[\w`\"]+\s*\.\s*[\w`\"]+", lowered):
        tags.append("schema_qualified")
    if dialect.casefold() == "sqlite" and '"' in sql:
        tags.append("sqlite_double_quote_ambiguity")
    return tuple(tags or ["ordinary"])


def _root_with_groups(root: Path) -> Path:
    root = Path(root)
    if (root / "bird_dev" / "offline.json").is_file():
        return root
    nested = root / "outputs" / "analysis" / "deepeye"
    if (nested / "bird_dev" / "offline.json").is_file():
        return nested
    raise ValueError(f"Offline group root not found below {root}")


def _linked_schemas(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, tuple[str, ...]]]:
    linked = {}
    for record in records:
        if (record.get("item_key") == key and record.get("stage") == "schema_linking"
                and record.get("condition") in ("native", "rc")):
            condition = record["condition"]
            if condition in linked or record.get("status") != "succeeded" or not isinstance(record.get("linked"), dict):
                raise ValueError(f"Invalid schema-linking record for {key}/{condition}")
            if any(not isinstance(columns, list) for columns in record["linked"].values()):
                raise ValueError(f"Invalid linked columns for {key}/{condition}")
            linked[condition] = {
                str(table): tuple(sorted(map(str, columns)))
                for table, columns in sorted(record["linked"].items())
            }
    if set(linked) != {"native", "rc"}:
        raise ValueError(f"Missing native/RC schema-linking record for {key}")
    return linked


def load_offline_groups(root: str | Path) -> list[AnnotationTask]:
    """Load exactly the five verified offline snapshots without reopening source datasets."""
    root = _root_with_groups(Path(root))
    catalog_cache: dict[str, dict[str, Any]] = {}
    tasks = []
    for group in GROUPS:
        raw_bytes = (root / group / "offline.json").read_bytes()
        data = json.loads(raw_bytes)
        bindings = data.get("bindings")
        references = data.get("references")
        inputs = data.get("inputs")
        if not isinstance(bindings, list) or not isinstance(references, dict) or not isinstance(inputs, dict):
            raise ValueError(f"Malformed offline group: {group}")
        bindings_by_key = {row.get("task_key"): row for row in bindings if isinstance(row, dict)}
        if len(bindings_by_key) != len(bindings) or set(bindings_by_key) != set(references) or set(inputs) != set(references):
            raise ValueError(f"Offline bindings do not agree for {group}")
        if len(bindings) != GROUP_SIZES[group]:
            raise ValueError(f"Frozen group count mismatch for {group}")
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        for key in sorted(references):
            binding, reference, input_row = bindings_by_key[key], references[key], inputs[key]
            if (not isinstance(reference, dict) or reference.get("status") != "available"
                    or not isinstance(input_row, dict) or not isinstance(reference.get("sql"), str)):
                raise ValueError(f"Invalid reference/input for {key}")
            sql = reference["sql"]
            sql_hash = _digest(sql)
            if reference.get("sql_sha256") != sql_hash:
                raise ValueError(f"Reference SQL hash mismatch for {key}")
            schema = canonical_schema(input_row.get("database_schema", {}))
            schema_hash = _digest(schema["tables"])
            schema = catalog_cache.setdefault(schema_hash, schema)
            linked = _linked_schemas(data.get("records", []), key)
            dialect = binding.get("db_type")
            if not isinstance(dialect, str) or input_row.get("database_id") != binding.get("database_id"):
                raise ValueError(f"Invalid database binding for {key}")
            tasks.append(AnnotationTask(
                task_key=key, group=group, partition=str(binding.get("partition", "")),
                external_id=binding.get("external_id"), database_id=binding["database_id"], dialect=dialect,
                gold_sql=sql, sql_sha256=sql_hash, schema=schema, schema_sha256=schema_hash,
                source_schema_sha256=str(binding.get("schema_sha256", "")), source_hash=source_hash,
                reuse_key=(dialect, schema_hash, sql_hash), features=feature_tags(sql, dialect),
                native_linked_schema=linked["native"], rc_linked_schema=linked["rc"],
                conservative_reference=conservative_reference(sql, dialect, schema),
                source_reference=dict(reference),
            ))
    if len(tasks) != TOTAL_TASKS or len({task.task_key for task in tasks}) != TOTAL_TASKS:
        raise ValueError("Frozen total task count mismatch")
    return tasks


def select_pilot(tasks: list[AnnotationTask], size: int, seed: int) -> list[AnnotationTask]:
    """Choose a reproducible, group-balanced pilot while retaining every available stratum."""
    tasks = sorted(tasks, key=lambda task: task.task_key)
    if not isinstance(size, int) or size < 1 or size > len(tasks):
        raise ValueError("Pilot size must be between one and the task count")
    if not isinstance(seed, int):
        raise ValueError("Pilot seed must be an integer")
    rng = random.Random(seed)
    groups = sorted({task.group for task in tasks})
    pools = {group: [task for task in tasks if task.group == group] for group in groups}
    for pool in pools.values():
        rng.shuffle(pool)
    quotas = {group: size // len(groups) + (index < size % len(groups))
              for index, group in enumerate(groups)}
    selected: list[AnnotationTask] = []
    selected_keys: set[str] = set()

    def add(task: AnnotationTask) -> bool:
        if task.task_key in selected_keys or len(selected) >= size or sum(
                chosen.group == task.group for chosen in selected) >= quotas[task.group]:
            return False
        selected.append(task)
        selected_keys.add(task.task_key)
        return True

    for group in groups:
        if quotas[group]:
            add(pools[group][0])
    for feature in FEATURES:
        candidates = [task for task in tasks if feature in task.features]
        if candidates:
            rng.shuffle(candidates)
            next((task for task in candidates if add(task)), None)
    for group in groups:
        for task in pools[group]:
            if sum(chosen.group == group for chosen in selected) >= quotas[group]:
                break
            add(task)
    if len(selected) != size:
        raise ValueError("Could not satisfy pilot quotas")
    return selected
