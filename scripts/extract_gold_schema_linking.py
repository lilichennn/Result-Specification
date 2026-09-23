"""Extract the tables and columns actually referenced by each gold SQL query."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent


def dataset_paths(
    dataset: str, split: str, dataset_root: str | Path | None = None
) -> tuple[Path, Path, Path, Path, str]:
    dataset_split = f"{dataset}_{split}"
    group_root = CODE_ROOT / "data" / dataset_split
    instances_path = group_root / f"{dataset_split}.json"
    output_path = group_root / "gold_sql_schema_linking.json"

    if dataset == "spider":
        if split not in ("dev", "test"):
            raise ValueError("Available gold data: spider/dev, spider/test, and bird/dev")
        raw_root = Path(dataset_root) if dataset_root is not None else Path("Spider")
        gold_path = raw_root / "data" / f"{split}.json"
        database_root = raw_root / "data" / (
            "database" if split == "dev" else "test_database"
        )
        sql_field = "query"
    elif dataset == "bird" and split == "dev":
        raw_root = Path(dataset_root) if dataset_root is not None else Path("BIRD")
        gold_path = raw_root / "data" / "dev" / "dev.json"
        database_root = raw_root / "data" / "dev" / "dev_databases"
        sql_field = "SQL"
    else:
        raise ValueError("Available gold data: spider/dev, spider/test, and bird/dev")

    return instances_path, gold_path, database_root, output_path, sql_field


def referenced_schema(database_path: Path, sql: str) -> list[dict[str, Any]]:
    """Use SQLite query resolution to collect physical table and column names."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Gold SQL must be non-empty text")

    tables: dict[str, dict[str, Any]] = {}

    def authorizer(
        action: int,
        table_name: str | None,
        column_name: str | None,
        _database_name: str | None,
        _trigger_name: str | None,
    ) -> int:
        if action == sqlite3.SQLITE_READ and table_name:
            key = table_name.casefold()
            table = tables.setdefault(
                key,
                {"table": table_name, "columns": [], "_column_keys": set()},
            )
            if column_name:
                column_key = column_name.casefold()
                if column_key not in table["_column_keys"]:
                    table["columns"].append(column_name)
                    table["_column_keys"].add(column_key)
        return sqlite3.SQLITE_OK

    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.set_authorizer(authorizer)
        connection.execute("EXPLAIN " + sql).fetchall()
    finally:
        connection.set_authorizer(None)
        connection.close()

    result = []
    for table in tables.values():
        columns = table["columns"] or ["*"]
        result.append({"table": table["table"], "columns": columns})
    return result


def extract(
    dataset: str, split: str, dataset_root: str | Path | None = None
) -> list[dict[str, Any]]:
    instances_path, gold_path, database_root, output_path, sql_field = dataset_paths(
        dataset, split, dataset_root
    )
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(instances, list) or not isinstance(gold_rows, list):
        raise ValueError("Instance and gold files must both be JSON arrays")

    if dataset == "bird":
        gold_by_index = {str(row["question_id"]): row for row in gold_rows}
    else:
        gold_by_index = {str(index): row for index, row in enumerate(gold_rows)}

    output = []
    for instance in instances:
        index = str(instance["index"])
        if index not in gold_by_index:
            raise ValueError(f"No gold row for instance {index}")
        gold = gold_by_index[index]
        if gold["db_id"] != instance["db_id"] or gold["question"] != instance["question"]:
            raise ValueError(f"Gold row does not match preprocessed instance {index}")
        sql = gold[sql_field]
        db_id = instance["db_id"]
        output.append(
            {
                "index": instance["index"],
                "db_id": db_id,
                "question": instance["question"],
                "gold_sql": sql,
                "schemalinking": referenced_schema(
                    database_root / db_id / f"{db_id}.sqlite", sql
                ),
            }
        )

    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("spider", "bird"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument(
        "--dataset-root", type=Path,
        help="Raw dataset root; defaults to Spider or BIRD relative to the working directory.",
    )
    args = parser.parse_args()
    rows = extract(args.dataset, args.split, args.dataset_root)
    output_path = dataset_paths(args.dataset, args.split, args.dataset_root)[3]
    print(
        f"Wrote {len(rows)} instances to {output_path}"
    )


if __name__ == "__main__":
    main()
