from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MAX_DATABASE_COLUMNS = 3000
MAX_SAMPLE_VALUE_CHARS = 1000
META_FIELDNAMES = [
    "column_name",
    "column_type",
    "column_description",
    "sample_value",
    "ref_key",
]
REQUIRED_INSTANCE_FIELDS = {
    "instance_id",
    "instruction",
    "db_id",
    "external_knowledge",
}


def preprocess_spider2_snow(
    spider2_root: str | Path,
    output_dir: str | Path | None = None,
    max_database_columns: int = DEFAULT_MAX_DATABASE_COLUMNS,
) -> dict[str, Any]:
    """Preprocess Spider2-Snow while excluding oversized database catalogs."""
    if max_database_columns < 1:
        raise ValueError("max_database_columns must be positive")

    spider2_root = Path(spider2_root).resolve()
    output_dir = Path(
        output_dir if output_dir is not None
        else Path(__file__).resolve().parents[2] / "data" / "spider2_snow"
    ).resolve()
    snow_root = _resolve_snow_root(spider2_root)
    dataset_path = snow_root / "spider2-snow.jsonl"
    database_root = snow_root / "resource" / "databases"
    document_root = snow_root / "resource" / "documents"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Spider2-Snow dataset not found: {dataset_path}")
    if not database_root.is_dir():
        raise FileNotFoundError(
            f"Spider2-Snow database metadata directory not found: {database_root}"
        )
    if not document_root.is_dir():
        raise FileNotFoundError(
            f"Spider2-Snow document directory not found: {document_root}"
        )

    raw_instances = _read_jsonl(dataset_path)
    db_instance_counts: Counter[str] = Counter()
    seen_instance_ids: set[str] = set()
    for position, instance in enumerate(raw_instances):
        _validate_instance(instance, position)
        instance_id = instance["instance_id"]
        if instance_id in seen_instance_ids:
            raise ValueError(f"Duplicate Spider2-Snow instance_id: {instance_id}")
        seen_instance_ids.add(instance_id)
        db_instance_counts[instance["db_id"]] += 1

    database_stats = {
        db_id: _read_database_stats(database_root, db_id)
        for db_id in sorted(db_instance_counts)
    }
    excluded_db_ids = {
        db_id
        for db_id, stats in database_stats.items()
        if stats["column_count"] > max_database_columns
    }

    document_cache: dict[str, str] = {}
    instances: list[dict[str, Any]] = []
    excluded_instance_ids: list[str] = []
    for raw_instance in raw_instances:
        if raw_instance["db_id"] in excluded_db_ids:
            excluded_instance_ids.append(raw_instance["instance_id"])
            continue
        instances.append(
            {
                "index": raw_instance["instance_id"],
                "db_id": raw_instance["db_id"],
                "question": raw_instance["instruction"],
                "evidence": _read_external_knowledge(
                    raw_instance["external_knowledge"],
                    document_root,
                    document_cache,
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    instance_output_path = output_dir / "spider2_snow.json"
    _write_json_atomic(instance_output_path, instances)

    meta_output_root = output_dir / "meta"
    meta_file_count = 0
    column_count = 0
    for db_id in sorted(set(db_instance_counts) - excluded_db_ids):
        target_db_dir = meta_output_root / db_id
        target_db_dir.mkdir(parents=True, exist_ok=True)
        seen_table_names: set[str] = set()
        for source_path in _database_table_paths(database_root, db_id):
            table_name, rows = _read_table_metadata(source_path)
            normalized_name = table_name.casefold()
            if normalized_name in seen_table_names:
                raise ValueError(
                    f"Duplicate fully-qualified table name in {db_id}: {table_name}"
                )
            seen_table_names.add(normalized_name)
            _validate_table_filename(table_name, source_path)
            _write_meta_csv_atomic(target_db_dir / f"{table_name}.csv", rows)
            meta_file_count += 1
            column_count += len(rows)

    excluded_databases = [
        {
            "db_id": db_id,
            "instance_count": db_instance_counts[db_id],
            **database_stats[db_id],
        }
        for db_id in sorted(excluded_db_ids)
    ]
    manifest = {
        "dataset": "spider2",
        "split": "snow",
        "source": str(dataset_path),
        "filter": {
            "field": "database_column_count",
            "operator": ">",
            "threshold": max_database_columns,
            "action": "exclude_database_and_instances",
        },
        "source_instance_count": len(raw_instances),
        "retained_instance_count": len(instances),
        "excluded_instance_count": len(excluded_instance_ids),
        "source_database_count": len(db_instance_counts),
        "retained_database_count": len(db_instance_counts) - len(excluded_db_ids),
        "excluded_database_count": len(excluded_db_ids),
        "excluded_databases": excluded_databases,
        "excluded_instance_ids": excluded_instance_ids,
    }
    manifest_path = output_dir / "filter_manifest.json"
    _write_json_atomic(manifest_path, manifest)

    return {
        "split": "snow",
        "source_instance_count": len(raw_instances),
        "instance_count": len(instances),
        "excluded_instance_count": len(excluded_instance_ids),
        "source_database_count": len(db_instance_counts),
        "database_count": len(db_instance_counts) - len(excluded_db_ids),
        "excluded_database_count": len(excluded_db_ids),
        "max_database_columns": max_database_columns,
        "meta_file_count": meta_file_count,
        "column_count": column_count,
        "instance_output_path": str(instance_output_path),
        "meta_output_path": str(meta_output_root),
        "filter_manifest_path": str(manifest_path),
    }


def _resolve_snow_root(spider2_root: Path) -> Path:
    if (spider2_root / "spider2-snow.jsonl").is_file():
        return spider2_root
    return spider2_root / "spider2-snow"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
        records.append(value)
    return records


def _validate_instance(instance: dict[str, Any], position: int) -> None:
    missing = REQUIRED_INSTANCE_FIELDS - instance.keys()
    if missing:
        raise ValueError(
            f"Spider2-Snow instance at position {position} is missing fields: "
            f"{sorted(missing)}"
        )
    for field in ("instance_id", "instruction", "db_id"):
        value = instance[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Spider2-Snow {field} at position {position} must be non-empty text"
            )
    external_knowledge = instance["external_knowledge"]
    if external_knowledge is not None and not isinstance(external_knowledge, str):
        raise ValueError(
            "Spider2-Snow external_knowledge at position "
            f"{position} must be text or null"
        )


def _database_table_paths(database_root: Path, db_id: str) -> list[Path]:
    database_dir = database_root / db_id
    if not database_dir.is_dir():
        raise FileNotFoundError(
            f"Spider2-Snow metadata directory for {db_id!r} not found: {database_dir}"
        )
    paths = sorted(database_dir.glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No Spider2-Snow table metadata found for {db_id!r}: {database_dir}"
        )
    return paths


def _read_database_stats(database_root: Path, db_id: str) -> dict[str, int]:
    table_paths = _database_table_paths(database_root, db_id)
    column_count = 0
    seen_table_names: set[str] = set()
    for path in table_paths:
        value = _read_json_object(path)
        table_name = value.get("table_fullname")
        column_names = value.get("column_names")
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError(f"Invalid table_fullname in metadata: {path}")
        if not isinstance(column_names, list):
            raise ValueError(f"Invalid column_names in metadata: {path}")
        normalized_name = table_name.casefold()
        if normalized_name in seen_table_names:
            raise ValueError(
                f"Duplicate fully-qualified table name in {db_id}: {table_name}"
            )
        seen_table_names.add(normalized_name)
        column_count += len(column_names)
    return {"table_count": len(table_paths), "column_count": column_count}


def _read_table_metadata(path: Path) -> tuple[str, list[dict[str, str]]]:
    value = _read_json_object(path)
    required_fields = {
        "table_fullname",
        "column_names",
        "column_types",
        "description",
        "sample_rows",
    }
    missing = required_fields - value.keys()
    if missing:
        raise ValueError(f"Table metadata is missing {sorted(missing)}: {path}")

    table_name = value["table_fullname"]
    column_names = value["column_names"]
    column_types = value["column_types"]
    descriptions = value["description"]
    sample_rows = value["sample_rows"]
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError(f"Invalid table_fullname in metadata: {path}")
    if not all(isinstance(field, list) for field in (column_names, column_types, descriptions)):
        raise ValueError(f"Column metadata fields must be arrays: {path}")
    if not len(column_names) == len(column_types) == len(descriptions):
        raise ValueError(f"Column metadata arrays have different lengths: {path}")
    if not isinstance(sample_rows, list) or not all(
        isinstance(row, dict) for row in sample_rows
    ):
        raise ValueError(f"sample_rows must be an array of objects: {path}")

    rows: list[dict[str, str]] = []
    for position, (column_name, column_type, description) in enumerate(
        zip(column_names, column_types, descriptions, strict=True)
    ):
        if not isinstance(column_name, str) or not column_name:
            raise ValueError(f"Invalid column name at position {position}: {path}")
        if not isinstance(column_type, str):
            raise ValueError(f"Invalid column type at position {position}: {path}")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"Invalid column description at position {position}: {path}")
        rows.append(
            {
                "column_name": column_name,
                "column_type": column_type,
                "column_description": description or "",
                "sample_value": json.dumps(
                    _sample_column_values(sample_rows, column_name),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "ref_key": "",
            }
        )
    return table_name.strip(), rows


def _sample_column_values(
    sample_rows: list[dict[str, Any]],
    column_name: str,
) -> list[Any]:
    values: list[Any] = []
    serialized_values: set[str] = set()
    for row in sample_rows:
        found, value = _lookup_sample_value(row, column_name)
        if not found or value is None:
            continue
        value = _normalize_sample_value(value)
        if value is None:
            continue
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(serialized) > MAX_SAMPLE_VALUE_CHARS:
            continue
        if serialized in serialized_values:
            continue
        serialized_values.add(serialized)
        values.append(value)
        if len(values) == 3:
            break
    return values


def _normalize_sample_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_normalize_sample_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_sample_value(item) for key, item in value.items()}
    return value


def _lookup_sample_value(row: dict[str, Any], column_name: str) -> tuple[bool, Any]:
    if column_name in row:
        return True, row[column_name]
    matches = [key for key in row if key.casefold() == column_name.casefold()]
    if len(matches) == 1:
        return True, row[matches[0]]
    return False, None


def _read_external_knowledge(
    filename: str | None,
    document_root: Path,
    cache: dict[str, str],
) -> str:
    if filename is None or not filename.strip():
        return ""
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Invalid Spider2-Snow external knowledge path: {filename!r}")
    if filename not in cache:
        path = document_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"External knowledge document not found: {path}")
        cache[filename] = path.read_text(encoding="utf-8").strip()
    return cache[filename]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Table metadata must be a JSON object: {path}")
    return value


def _validate_table_filename(table_name: str, source_path: Path) -> None:
    if (
        not table_name
        or Path(table_name).name != table_name
        or "/" in table_name
        or "\\" in table_name
    ):
        raise ValueError(
            f"Table name cannot be used as a CSV filename in {source_path}: "
            f"{table_name!r}"
        )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_meta_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)
