"""Bounded native local-index builder for BIRD-Interact PostgreSQL values."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from app.vector_db.local_index import (
    get_local_index_path,
    write_local_index_column,
    write_local_index_manifest,
)
from app.vector_db.vector_db import (
    _is_number_column,
    _is_text_column_type,
    _is_uuid_column,
)


MAX_VALUE_LENGTH = 100
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _bounded_value_query(
    table_name: str, column_name: str, max_values_per_column: int
) -> str:
    table = _quote_identifier(table_name)
    column = _quote_identifier(column_name)
    return (
        f"SELECT DISTINCT {column} FROM {table} "
        f"WHERE {column} IS NOT NULL "
        f"AND LENGTH(CAST({column} AS TEXT)) <= {MAX_VALUE_LENGTH} "
        f"ORDER BY {column} ASC "
        f"LIMIT {max_values_per_column}"
    )


def _extract_values(rows: Any, max_values_per_column: int) -> list[str]:
    if not isinstance(rows, list):
        raise RuntimeError("PostgreSQL value query returned malformed rows")
    values: set[str] = set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 1:
            raise RuntimeError("PostgreSQL value query returned a malformed row")
        value = row[0]
        if value is None:
            continue
        text = str(value)
        if len(text) <= MAX_VALUE_LENGTH:
            values.add(text)
    return sorted(values)[:max_values_per_column]


def _embed_values(
    values: list[str], embedding_function: Any, embedding_batch_size: int
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(values), embedding_batch_size):
        batch = values[start : start + embedding_batch_size]
        batch_embeddings = embedding_function(batch)
        try:
            count = len(batch_embeddings)
        except TypeError as exc:
            raise RuntimeError("Embedding function must return one vector per value") from exc
        if count != len(batch):
            raise RuntimeError("Embedding function returned the wrong number of vectors")
        embeddings.extend(list(batch_embeddings))
    return embeddings


def build_postgres_value_index(
    data_item,
    store_root: Path,
    embedding_function,
    *,
    max_values_per_column: int = 50,
    embedding_batch_size: int = 20,
) -> dict[str, Any]:
    """Build a deterministic bounded native local index from the Meta allowlist."""

    if getattr(data_item, "db_type", None) != "postgresql":
        raise ValueError("PostgreSQL value indexing requires db_type='postgresql'")
    if not isinstance(max_values_per_column, int) or max_values_per_column <= 0:
        raise ValueError("max_values_per_column must be a positive integer")
    if not isinstance(embedding_batch_size, int) or embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be a positive integer")
    database_id = getattr(data_item, "database_id", None)
    if (
        not isinstance(database_id, str)
        or _SAFE_PATH_COMPONENT.fullmatch(database_id) is None
    ):
        raise ValueError(f"Database id must be one safe path component: {database_id!r}")
    schema = getattr(data_item, "database_schema", None)
    if not isinstance(schema, dict) or not isinstance(schema.get("tables"), dict):
        raise ValueError("PostgreSQL data item has a malformed database schema")

    from .postgres_execution import execute_postgres_sql

    eligible_columns: list[tuple[str, str]] = []
    for table_name, table_schema in schema["tables"].items():
        if not isinstance(table_name, str) or not isinstance(table_schema, dict):
            raise ValueError("PostgreSQL Meta contains a malformed table")
        columns = table_schema.get("columns")
        if not isinstance(columns, dict):
            raise ValueError(f"PostgreSQL Meta table {table_name!r} has no columns")
        for column_name, column_schema in columns.items():
            if not isinstance(column_name, str) or not isinstance(column_schema, dict):
                raise ValueError(f"PostgreSQL Meta table {table_name!r} has a malformed column")
            column_type = column_schema.get("column_type")
            if not isinstance(column_type, str):
                raise ValueError(
                    f"PostgreSQL Meta column {table_name}.{column_name} has no type"
                )
            if _is_text_column_type(column_type):
                eligible_columns.append((table_name, column_name))
    column_stats: list[dict[str, Any]] = []
    staged_entries: list[dict[str, Any]] = []
    vector_db_path = Path(store_root) / database_id
    vector_db_path.mkdir(parents=True, exist_ok=True)
    staged_path = Path(tempfile.mkdtemp(prefix=".local_index-", dir=vector_db_path))
    try:
        for table_name, column_name in eligible_columns:
            sql = _bounded_value_query(
                table_name, column_name, max_values_per_column
            )
            result = execute_postgres_sql(data_item, sql, timeout=None)
            if result.result_type in {"empty_result", "all_null_result"}:
                values: list[str] = []
            elif result.result_type == "success":
                values = _extract_values(result.result_rows, max_values_per_column)
            else:
                raise RuntimeError(
                    f"Failed to index PostgreSQL column {table_name}.{column_name} "
                    f"({result.result_type})"
                )

            status = "empty"
            if values and _is_uuid_column(values):
                status = "filtered_uuid"
                values = []
            elif values and _is_number_column(values):
                status = "filtered_numeric"
                values = []
            elif values:
                status = "indexed"
                embeddings = _embed_values(
                    values, embedding_function, embedding_batch_size
                )
                entry = write_local_index_column(
                    local_index_path=staged_path,
                    table_name=table_name.lower(),
                    column_name=column_name.lower(),
                    documents=values,
                    embeddings=embeddings,
                )
                if entry is None:
                    raise RuntimeError(
                        f"Native local-index writer rejected {table_name}.{column_name}"
                    )
                staged_entries.append(entry)
            column_stats.append(
                {
                    "table_name": table_name,
                    "column_name": column_name,
                    "status": status,
                    "value_count": len(values),
                }
            )

        statistics = {
            "database_id": database_id,
            "db_type": "postgresql",
            "max_values_per_column": max_values_per_column,
            "max_value_length": MAX_VALUE_LENGTH,
            "embedding_batch_size": embedding_batch_size,
            "text_column_count": len(eligible_columns),
            "indexed_column_count": len(staged_entries),
            "value_count": sum(entry["count"] for entry in staged_entries),
            "columns": column_stats,
        }
        write_local_index_manifest(staged_path, staged_entries)
        manifest_path = staged_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provenance"] = statistics
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        final_path = get_local_index_path(vector_db_path)
        if final_path.exists():
            shutil.rmtree(final_path)
        staged_path.replace(final_path)
        return statistics
    finally:
        if staged_path.exists():
            shutil.rmtree(staged_path)
