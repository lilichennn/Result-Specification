"""Fail-closed validation for blind gold-SQL schema-linking annotations."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from .source import AnnotationTask


_FIELDS = frozenset({
    "task_key", "status", "required_table_ids", "required_column_ids",
    "evidence", "json_paths", "review_reasons",
})
_STATUSES = frozenset({"resolved", "needs_review", "invalid_sql"})
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(.*?)(?:\r?\n)?```\Z", re.IGNORECASE | re.DOTALL)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _require_string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    result = tuple(_require_string(item, f"{label} item") for item in value)
    duplicate = next((item for item in result if result.count(item) > 1), None)
    if duplicate is not None:
        raise ValueError(f"duplicate {label} item: {duplicate}")
    return result


def _catalog(task: AnnotationTask) -> tuple[dict[str, str], dict[str, dict[str, str]], tuple[str, ...], tuple[str, ...]]:
    """Return closed ID maps and their frozen ordering for one task."""
    schema = getattr(task, "canonical_schema", getattr(task, "schema", None))
    if not isinstance(schema, Mapping) or not isinstance(schema.get("tables"), (list, tuple)):
        raise ValueError("task has no canonical schema catalog")
    tables: dict[str, str] = {}
    columns: dict[str, dict[str, str]] = {}
    for table in schema["tables"]:
        if not isinstance(table, Mapping):
            raise ValueError("canonical schema table must be an object")
        table_id, table_name = table.get("id"), table.get("name")
        if not isinstance(table_id, str) or not isinstance(table_name, str) or table_id in tables:
            raise ValueError("canonical schema table IDs must be unique text")
        tables[table_id] = table_name
        raw_columns = table.get("columns")
        if not isinstance(raw_columns, (list, tuple)):
            raise ValueError("canonical schema table columns must be an array")
        for column in raw_columns:
            if not isinstance(column, Mapping):
                raise ValueError("canonical schema column must be an object")
            column_id, column_name = column.get("id"), column.get("name")
            if not isinstance(column_id, str) or not isinstance(column_name, str) or column_id in columns:
                raise ValueError("canonical schema column IDs must be unique text")
            columns[column_id] = {"table_id": table_id, "table": table_name, "column": column_name}
    return tables, columns, tuple(tables), tuple(columns)


def _ordered(values: Iterable[str], catalog_order: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(values)
    return tuple(value for value in catalog_order if value in selected)


def _validate_shape(raw: Any, task: AnnotationTask) -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        raise ValueError("annotation must be an object")
    present = set(raw)
    missing, unexpected = _FIELDS - present, present - _FIELDS
    if missing:
        raise ValueError(f"annotation missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"annotation has unexpected fields: {', '.join(sorted(unexpected))}")

    task_key = _require_string(raw["task_key"], "task_key")
    expected_key = _require_string(getattr(task, "task_key", None), "task task_key")
    if task_key != expected_key:
        raise ValueError(f"annotation task_key mismatch: {task_key!r} != {expected_key!r}")
    status = raw["status"]
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValueError("annotation status must be resolved, needs_review, or invalid_sql")

    table_ids = _require_string_list(raw["required_table_ids"], "required_table_ids")
    column_ids = _require_string_list(raw["required_column_ids"], "required_column_ids")
    evidence = _require_string_list(raw["evidence"], "evidence")
    review_reasons = _require_string_list(raw["review_reasons"], "review_reasons")
    if status == "resolved" and review_reasons:
        raise ValueError("resolved annotation cannot have review reasons")
    if status != "resolved" and not review_reasons:
        raise ValueError(f"{status} annotation requires a review reason")

    if not isinstance(raw["json_paths"], list):
        raise ValueError("json_paths must be an array")
    json_paths = []
    for path in raw["json_paths"]:
        if not isinstance(path, Mapping) or set(path) != {"column_id", "path"}:
            raise ValueError("json_paths entries must contain exactly column_id and path")
        json_paths.append((_require_string(path["column_id"], "json_paths column_id"),
                           _require_string(path["path"], "json_paths path")))
    if len(set(json_paths)) != len(json_paths):
        raise ValueError("duplicate json_paths entry")
    return task_key, status, table_ids, column_ids, evidence, tuple(json_paths), review_reasons


def normalize_annotation(raw: Mapping[str, Any], task: AnnotationTask) -> dict[str, Any]:
    """Validate one model entry and map its closed IDs back to physical objects."""
    task_key, status, table_ids, column_ids, evidence, json_paths, review_reasons = _validate_shape(raw, task)
    tables, columns, table_order, column_order = _catalog(task)
    for table_id in table_ids:
        if table_id not in tables:
            raise ValueError(f"unknown required_table_ids item: {table_id}")
    for column_id in column_ids:
        if column_id not in columns:
            raise ValueError(f"unknown required_column_ids item: {column_id}")
    for column_id, _ in json_paths:
        if column_id not in columns:
            raise ValueError(f"unknown json_paths column_id: {column_id}")
        if column_id not in column_ids:
            raise ValueError(f"json_paths column_id is not a required column: {column_id}")

    # A valid physical column necessarily establishes its physical parent table.
    # No parent is inferred from free text, aliases, or JSON-path fragments.
    selected_tables = set(table_ids) | {columns[column_id]["table_id"] for column_id in column_ids}
    ordered_tables = _ordered(selected_tables, table_order)
    ordered_columns = _ordered(column_ids, column_order)
    return {
        "task_key": task_key,
        "status": status,
        "required_table_ids": ordered_tables,
        "required_column_ids": ordered_columns,
        "required_tables": tuple(tables[table_id] for table_id in ordered_tables),
        "required_columns": tuple(
            (columns[column_id]["table"], columns[column_id]["column"])
            for column_id in ordered_columns
        ),
        "evidence": evidence,
        "json_paths": json_paths,
        "review_reasons": review_reasons,
    }


def _decode_json(text: str) -> list[Any]:
    if not isinstance(text, str):
        raise ValueError("model response must be text")
    payload = text.strip()
    fence = _FENCE.fullmatch(payload)
    if fence:
        payload = fence.group(1).strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("model response is not valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError("model response must be a JSON array")
    return parsed


def parse_response(text: str, tasks: list[AnnotationTask]) -> dict[str, dict[str, Any]]:
    """Parse one complete batch, rejecting any missing, extra, or duplicate task."""
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    task_by_key: dict[str, AnnotationTask] = {}
    for task in tasks:
        key = _require_string(getattr(task, "task_key", None), "task task_key")
        if key in task_by_key:
            raise ValueError(f"duplicate selected task: {key}")
        task_by_key[key] = task

    entries = _decode_json(text)
    raw_by_key: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("model response entries must be objects")
        key = _require_string(entry.get("task_key"), "response task_key")
        if key in raw_by_key:
            raise ValueError(f"duplicate response task_key: {key}")
        raw_by_key[key] = entry
    selected, returned = set(task_by_key), set(raw_by_key)
    extra, missing = returned - selected, selected - returned
    if extra:
        raise ValueError(f"extra response tasks: {', '.join(sorted(extra))}")
    if missing:
        raise ValueError(f"missing response tasks: {', '.join(sorted(missing))}")
    return {key: normalize_annotation(raw_by_key[key], task) for key, task in task_by_key.items()}
