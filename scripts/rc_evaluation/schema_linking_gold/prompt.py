"""Blind, JSON-only prompt construction for gold-SQL dependency annotation."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .source import AnnotationTask


def _catalog_for_prompt(task: AnnotationTask) -> dict[str, Any]:
    schema = getattr(task, "canonical_schema", getattr(task, "schema", None))
    if not isinstance(schema, Mapping) or not isinstance(schema.get("tables"), (list, tuple)):
        raise ValueError("task has no canonical schema catalog")
    tables = []
    for table in schema["tables"]:
        if not isinstance(table, Mapping) or not isinstance(table.get("columns"), (list, tuple)):
            raise ValueError("canonical schema catalog is malformed")
        tables.append({
            "id": table.get("id"), "name": table.get("name"),
            "columns": [
                {"id": column.get("id"), "name": column.get("name"), "type": column.get("type", "")}
                for column in table["columns"] if isinstance(column, Mapping)
            ],
        })
    return {"tables": tables}


def render_batch_prompt(tasks: list[AnnotationTask]) -> str:
    """Render only dialect, closed ID catalog, and saved gold SQL for each task."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    inputs = []
    seen = set()
    for task in tasks:
        task_key = getattr(task, "task_key", None)
        dialect = getattr(task, "dialect", None)
        gold_sql = getattr(task, "gold_sql", None)
        if not isinstance(task_key, str) or not task_key or task_key in seen:
            raise ValueError("tasks must have unique non-empty task keys")
        if not isinstance(dialect, str) or not dialect or not isinstance(gold_sql, str) or not gold_sql:
            raise ValueError(f"task {task_key!r} has incomplete blind inputs")
        seen.add(task_key)
        inputs.append({"task_key": task_key, "dialect": dialect,
                       "schema_catalog": _catalog_for_prompt(task), "gold_sql": gold_sql})

    instructions = """You annotate physical database dependencies of saved gold SQL. Return JSON only: one JSON array with exactly one object per input task. Do not use Markdown or prose.

The schema_catalog is closed: use only its T* table IDs and C* column IDs. Never invent IDs or report CTE names, subquery aliases, table aliases, output aliases, table-function output aliases, expressions, or JSON paths as physical objects.

For every SQL statement, trace CTEs, subqueries, set operations, correlated references, aliases, and derived expressions to physical base tables and columns. COUNT(*) creates a table dependency but no column dependency. Projection table.* expands to every physical column of that table; COUNT(*) and EXISTS(SELECT *) do not. Record a JSON/JSONB path only diagnostically and always include its physical carrier column. USING(column) includes both physical columns; NATURAL JOIN includes common physical columns only when the catalog determines them unambiguously. Function arguments, filters, joins, grouping, ordering, windows, and projections all contribute physical column dependencies. Apply SQLite double-quoted-token rules and PostgreSQL quoted/unquoted identifier rules for the supplied dialect.

If a reference cannot be resolved from the catalog or is schema-inconsistent, use needs_review or invalid_sql instead of inventing coverage. A resolved entry has no review reasons; needs_review and invalid_sql entries have at least one review reason.

Each response object has exactly these fields: task_key (string), status (resolved, needs_review, or invalid_sql), required_table_ids (unique array of T* IDs), required_column_ids (unique array of C* IDs), evidence (array of non-empty strings), json_paths (array of {\"column_id\": C* ID, \"path\": non-empty string}; each carrier must be listed in required_column_ids), and review_reasons (array of non-empty strings)."""
    return instructions + "\n\nINPUTS:\n" + json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
