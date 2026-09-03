from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_INSTANCE_FIELDS = {"db_id", "question"}
META_FIELDNAMES = ["column_name", "column_type", "sample_value", "ref_key"]


def preprocess_spider(
    spider_root: str | Path,
    split: str = "dev",
    output_dir: str | Path = "preprocessed_data",
) -> dict[str, Any]:
    """Preprocess a Spider 1.0 split and derive table metadata."""
    database_directory_by_split = {
        "dev": "database",
        "test": "test_database",
    }
    if split not in database_directory_by_split:
        raise ValueError(
            f"Unsupported Spider split: {split!r}; expected one of "
            f"{sorted(database_directory_by_split)}"
        )

    spider_root = Path(spider_root).resolve()
    output_dir = Path(output_dir).resolve()
    data_root = spider_root / "data"
    dataset_path = data_root / f"{split}.json"
    database_root = data_root / database_directory_by_split[split]

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Spider split file not found: {dataset_path}")
    if not database_root.is_dir():
        raise FileNotFoundError(f"Spider database directory not found: {database_root}")

    raw_instances = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_instances, list):
        raise ValueError(f"Spider split must be a JSON array: {dataset_path}")

    instances: list[dict[str, Any]] = []
    db_ids: set[str] = set()
    for index, raw_instance in enumerate(raw_instances):
        if not isinstance(raw_instance, dict):
            raise ValueError(f"Spider instance at position {index} must be an object")

        missing_fields = REQUIRED_INSTANCE_FIELDS - raw_instance.keys()
        if missing_fields:
            raise ValueError(
                f"Spider instance at position {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        db_id = raw_instance["db_id"]
        question = raw_instance["question"]
        if not isinstance(db_id, str) or not isinstance(question, str):
            raise ValueError(
                f"db_id and question must be strings at position {index}"
            )

        instances.append(
            {
                "index": index,
                "db_id": db_id,
                "question": question,
                "evidence": "",
            }
        )
        db_ids.add(db_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    instance_output_path = output_dir / f"spider_{split}.json"
    _write_json_atomic(instance_output_path, instances)

    meta_output_root = output_dir / "meta"
    meta_file_count = 0
    column_count = 0
    for db_id in sorted(db_ids):
        database_path = database_root / db_id / f"{db_id}.sqlite"
        if not database_path.is_file():
            raise FileNotFoundError(f"Spider SQLite database not found: {database_path}")

        database_meta_root = meta_output_root / db_id
        database_meta_root.mkdir(parents=True, exist_ok=True)
        table_metadata = _read_database_metadata(database_path)
        for table_name, rows in table_metadata.items():
            _validate_table_filename(table_name, database_path)
            _write_meta_csv_atomic(
                database_meta_root / f"{table_name}.csv",
                rows,
            )
            meta_file_count += 1
            column_count += len(rows)

    return {
        "split": split,
        "instance_count": len(instances),
        "database_count": len(db_ids),
        "meta_file_count": meta_file_count,
        "column_count": column_count,
        "instance_output_path": str(instance_output_path),
        "meta_output_path": str(meta_output_root),
    }


def _read_database_metadata(
    database_path: Path,
) -> dict[str, list[dict[str, str]]]:
    database_uri = f"{database_path.as_uri()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        table_names = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]

        metadata: dict[str, list[dict[str, str]]] = {}
        for table_name in table_names:
            columns = connection.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall()
            foreign_keys = _foreign_keys_by_column(connection, table_name)

            rows: list[dict[str, str]] = []
            for column in columns:
                column_name = column[1]
                column_type = column[2] or ""
                sample_values = _sample_column_values(
                    connection,
                    table_name,
                    column_name,
                )
                rows.append(
                    {
                        "column_name": column_name,
                        "column_type": column_type,
                        "sample_value": json.dumps(
                            sample_values,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        "ref_key": "; ".join(foreign_keys.get(column_name, [])),
                    }
                )
            metadata[table_name] = rows

    return metadata


def _foreign_keys_by_column(
    connection: sqlite3.Connection,
    table_name: str,
) -> dict[str, list[str]]:
    foreign_keys: defaultdict[str, list[str]] = defaultdict(list)
    rows = connection.execute(
        f"PRAGMA foreign_key_list({_quote_identifier(table_name)})"
    ).fetchall()
    for row in rows:
        referenced_table = row[2]
        source_column = row[3]
        referenced_column = row[4]
        if not source_column or not referenced_table:
            continue
        reference = str(referenced_table)
        if referenced_column:
            reference += f".{referenced_column}"
        foreign_keys[str(source_column)].append(reference)
    return dict(foreign_keys)


def _sample_column_values(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> list[str | int | float]:
    quoted_table = _quote_identifier(table_name)
    quoted_column = _quote_identifier(column_name)
    rows = connection.execute(
        f"""
        SELECT DISTINCT {quoted_column}
        FROM {quoted_table}
        WHERE {quoted_column} IS NOT NULL
        LIMIT 3
        """
    ).fetchall()
    return [_normalize_sqlite_value(row[0]) for row in rows]


def _normalize_sqlite_value(value: Any) -> str | int | float:
    if isinstance(value, bytes):
        return f"0x{value.hex()}"
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return str(value)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _validate_table_filename(table_name: str, database_path: Path) -> None:
    if not table_name or Path(table_name).name != table_name or "/" in table_name:
        raise ValueError(
            f"Table name cannot be used as a CSV filename in {database_path}: "
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
