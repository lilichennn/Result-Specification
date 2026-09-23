"""BIRD-Interact adapter for the preprocessed PostgreSQL benchmark artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.dataset.dataset import BaseDataset, DataItem


_ADAPTER_WORKSPACE_ROOT = Path(__file__).resolve().parents[3] / "workspace"


_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_REQUIRED_META_FIELDS = {
    "original_column_name",
    "column_name",
    "column_description",
    "data_format",
    "value_description",
}


class BirdInteractDataItem(DataItem):
    """A BIRD-Interact query backed by read-only PostgreSQL execution."""

    instance_id: str = Field(..., description="Original BIRD-Interact instance ID")
    db_type: Literal["postgresql"] = Field(
        default="postgresql", description="Canonical database type"
    )
    gold_sql: str = Field(default="", description="Gold SQL is intentionally unavailable")
    difficulty: str = Field(default="", description="Difficulty is not supplied")


class BirdInteractDatasetConfig(BaseModel):
    """Dataset settings compatible with DeepEye without widening native config."""

    type: Literal["bird_interact"] = "bird_interact"
    split: Literal["lite", "full"]
    root_path: str
    save_path: Optional[str] = None
    max_samples: Optional[int] = Field(default=None, ge=0)
    max_samples_per_db: Optional[int] = Field(default=None, ge=0)
    snowflake_credential_path: Optional[str] = None
    sql_execution_timeout: int = Field(default=600, gt=0)
    max_value_example_length: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def set_default_save_path(self):
        if self.save_path is None:
            self.save_path = str(
                _ADAPTER_WORKSPACE_ROOT
                / "dataset"
                / self.type
                / f"{self.split}.snapshot"
            )
        else:
            self.save_path = str(self.save_path)
        self.root_path = str(self.root_path)
        return self


def _validate_safe_component(value: str, label: str) -> None:
    if not value or _SAFE_PATH_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{label} must be one safe path component: {value!r}")


def _require_string(
    row: dict[str, Any], field_name: str, *, context: str, allow_empty: bool = False
) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{context} field {field_name!r} must be {qualifier}")
    return value


def _parse_primary_key(value: str, *, context: str) -> bool:
    normalized = value.strip().casefold()
    if not normalized:
        return False
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{context} primary_key must be a boolean")


def _parse_foreign_keys(value: str, *, context: str) -> list[list[str]]:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} foreign_keys must be valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{context} foreign_keys must be a list")
    normalized: list[list[str]] = []
    for target in parsed:
        if (
            not isinstance(target, (list, tuple))
            or len(target) != 2
            or not all(isinstance(part, str) and part.strip() for part in target)
        ):
            raise ValueError(
                f"{context} foreign_keys entries must be [table_name, column_name]"
            )
        normalized.append([target[0], target[1]])
    return normalized


def _column_description(row: dict[str, str]) -> str:
    sections = []
    expanded_name = row.get("column_name", "").strip()
    if expanded_name:
        sections.append(f"Expanded Column Name: {expanded_name}")
    description = row.get("column_description", "").strip()
    if description:
        sections.append(f"Column Description: {description}")
    value_description = row.get("value_description", "").strip()
    if value_description:
        sections.append(f"Value Description: {value_description}")
    return " | ".join(sections)


def _load_meta_table(csv_path: Path, *, rows=None) -> dict[str, Any]:
    table_name = csv_path.stem
    if not table_name.strip():
        raise ValueError(f"Malformed Meta table name: {csv_path}")

    # A shared workload may already have decoded the CSV while binding its
    # hash. Reuse those rows through the exact same native Meta validation.
    from contextlib import nullcontext
    source_context = (csv_path.open("r", encoding="utf-8-sig", newline="")
                      if rows is None else nullcontext(None))
    with source_context as source:
        reader = csv.DictReader(source) if rows is None else rows
        headers = set(reader.fieldnames or []) if rows is None else set(rows[0] if rows else [])
        missing_headers = sorted(_REQUIRED_META_FIELDS - headers)
        if missing_headers:
            raise ValueError(
                f"Malformed Meta table {csv_path}: missing fields {missing_headers}"
            )

        columns: dict[str, dict[str, Any]] = {}
        seen_columns: set[str] = set()
        for row_number, raw_row in enumerate(reader, start=2):
            context = f"Meta table {table_name!r} row {row_number}"
            if None in raw_row:
                raise ValueError(f"Malformed {context}: unexpected CSV fields")
            row = {key: (value or "") for key, value in raw_row.items()}
            column_name = row["original_column_name"].strip()
            column_type = row["data_format"].strip()
            if not column_name:
                raise ValueError(f"Malformed {context}: column name is empty")
            if not column_type:
                raise ValueError(
                    f"Malformed {context}: column {column_name!r} has no data format"
                )
            folded_column_name = column_name.casefold()
            if folded_column_name in seen_columns:
                raise ValueError(
                    f"Duplicate column {table_name}.{column_name} in {csv_path}"
                )
            seen_columns.add(folded_column_name)

            primary_key = _parse_primary_key(
                row.get("primary_key", ""), context=context
            )
            foreign_keys = _parse_foreign_keys(
                row.get("foreign_keys", ""), context=context
            )
            columns[column_name] = {
                "column_name": column_name,
                "column_type": column_type,
                "is_unuseful": False,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "description": _column_description(row),
                "value_examples": None,
                "value_statistics": None,
            }

    if not columns:
        raise ValueError(f"Malformed Meta table {csv_path}: no columns")
    return {"table_name": table_name, "columns": columns}


class BirdInteractDataset(BaseDataset):
    """Load preprocessed queries and public Meta descriptions for model inputs."""

    _name = "bird_interact"

    def __init__(
        self,
        dataset_config: BirdInteractDatasetConfig,
        instance_ids: Optional[List[str]] = None,
    ):
        self._config = dataset_config
        self._database_schema_cache: Dict[str, Any] = {}
        self._requested_instance_ids = self._validate_requested_ids(instance_ids)
        self._data = self._load_data()

    @staticmethod
    def _validate_requested_ids(instance_ids: Optional[List[str]]) -> Optional[List[str]]:
        if instance_ids is None:
            return None
        if not isinstance(instance_ids, list):
            raise ValueError("instance_ids must be a list of strings")
        normalized: list[str] = []
        seen: set[str] = set()
        for instance_id in instance_ids:
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError("instance_ids must contain non-empty strings")
            if instance_id in seen:
                raise ValueError(f"Duplicate requested instance ID: {instance_id}")
            seen.add(instance_id)
            normalized.append(instance_id)
        return normalized

    def _get_database_path(self, database_id: str) -> str:
        return database_id

    def _load_database_schema(self, database_id: str) -> dict[str, Any]:
        cached = self._database_schema_cache.get(database_id)
        if cached is not None:
            return cached

        _validate_safe_component(database_id, "Database id")
        root_path = Path(self._config.root_path).resolve()
        meta_root = root_path / "meta"
        database_meta_dir = meta_root / database_id.casefold()
        try:
            database_meta_dir.resolve().relative_to(meta_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Database id must resolve below the Meta root: {database_id!r}"
            ) from exc
        if not database_meta_dir.is_dir():
            raise FileNotFoundError(
                f"Meta directory not found for database {database_id!r}: "
                f"{database_meta_dir}"
            )

        csv_paths = sorted(
            (path for path in database_meta_dir.iterdir() if path.suffix.casefold() == ".csv"),
            key=lambda path: (path.name.casefold(), path.name),
        )
        if not csv_paths:
            raise FileNotFoundError(
                f"No Meta CSV tables found for database {database_id!r}: "
                f"{database_meta_dir}"
            )

        tables: dict[str, dict[str, Any]] = {}
        seen_tables: set[str] = set()
        for csv_path in csv_paths:
            if not csv_path.is_file():
                raise ValueError(f"Malformed Meta table path: {csv_path}")
            try:
                csv_path.resolve().relative_to(database_meta_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe Meta table path: {csv_path}") from exc
            table = _load_meta_table(csv_path)
            table_name = table["table_name"]
            folded_table_name = table_name.casefold()
            if folded_table_name in seen_tables:
                raise ValueError(
                    f"Duplicate Meta table name (case-insensitive): {table_name}"
                )
            seen_tables.add(folded_table_name)
            tables[table_name] = table

        schema = {
            "db_id": database_id,
            "db_path": database_id,
            "db_type": "postgresql",
            "tables": tables,
        }
        self._database_schema_cache[database_id] = schema
        return schema

    def _load_source_rows(self) -> list[dict[str, Any]]:
        root_path = Path(self._config.root_path)
        data_path = root_path / f"bird_interact_{self._config.split}.json"
        if not data_path.is_file():
            raise FileNotFoundError(f"BIRD-Interact data file not found: {data_path}")
        try:
            raw_rows = json.loads(data_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed BIRD-Interact JSON: {data_path}") from exc
        if not isinstance(raw_rows, list):
            raise ValueError(f"BIRD-Interact data must be a JSON list: {data_path}")

        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for position, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                raise ValueError(
                    f"BIRD-Interact item at position {position} must be an object"
                )
            context = f"BIRD-Interact item at position {position}"
            instance_id = _require_string(raw_row, "index", context=context)
            if instance_id in seen_ids:
                raise ValueError(f"Duplicate BIRD-Interact instance ID: {instance_id}")
            seen_ids.add(instance_id)
            db_id = _require_string(raw_row, "db_id", context=context)
            _validate_safe_component(db_id, "Database id")
            _require_string(raw_row, "question", context=context)
            if "evidence" in raw_row:
                _require_string(
                    raw_row, "evidence", context=context, allow_empty=True
                )
            rows.append(raw_row)

        if self._requested_instance_ids is not None:
            rows_by_id = {row["index"]: row for row in rows}
            missing = [
                instance_id
                for instance_id in self._requested_instance_ids
                if instance_id not in rows_by_id
            ]
            if missing:
                raise ValueError(
                    "Requested BIRD-Interact instance IDs are missing: "
                    + ", ".join(missing)
                )
            rows = [rows_by_id[instance_id] for instance_id in self._requested_instance_ids]
        return rows

    def _load_data(self) -> list[BirdInteractDataItem]:
        rows = self._load_source_rows()
        data: list[BirdInteractDataItem] = []
        database_counts: dict[str, int] = {}
        for row in rows:
            if self._config.max_samples is not None and len(data) >= self._config.max_samples:
                break
            database_id = row["db_id"]
            if (
                self._config.max_samples_per_db is not None
                and database_counts.get(database_id, 0)
                >= self._config.max_samples_per_db
            ):
                continue

            database_schema = self._load_database_schema(database_id)
            data.append(
                BirdInteractDataItem(
                    question_id=len(data),
                    instance_id=row["index"],
                    question=row["question"],
                    evidence=row.get("evidence", ""),
                    gold_sql="",
                    difficulty="",
                    database_id=database_id,
                    database_path=database_id,
                    database_schema=database_schema,
                    db_type="postgresql",
                )
            )
            database_counts[database_id] = database_counts.get(database_id, 0) + 1
        return data
