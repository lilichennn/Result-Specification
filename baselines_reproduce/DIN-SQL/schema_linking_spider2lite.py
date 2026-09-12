"""Run DIN-SQL schema linking for Spider2-Lite SQLite and BigQuery instances."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

from schema_linking import (
    MODEL_ALIASES,
    Progress,
    model_config,
    official_examples,
    quote_identifier,
    run_results,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET_SPLIT = "spider2_lite"
DATA_PATH = ROOT / "code/scripts/spider2_lite/preprocessed_data/spider2_lite.json"
SOURCE_MAP_PATH = ROOT / "Spider2.0/spider2-lite/spider2-lite_datasource_map.csv"
DATABASE_ROOT = ROOT / "Spider2.0/spider2-lite/resource/databases/spider2-localdb"
BIGQUERY_RESOURCE_ROOT = (
    ROOT / "Spider2.0/spider2-lite/resource/databases/bigquery"
)
BIGQUERY_CREDENTIAL = (
    ROOT / "Spider2.0/spider2-lite/evaluation_suite/bigquery_credential.json"
)
BIGQUERY_VENDOR = ROOT / "Spider2.0/spider2-lite/evaluation_suite/_vendor"


def sqlite_schema_context(database: Path) -> str:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        columns = {
            table: connection.execute(
                f"PRAGMA table_info({quote_identifier(table)})"
            ).fetchall()
            for table in tables
        }
        parts = [
            f"Table {table}, columns = ["
            + ",".join(["*"] + [column[1] for column in columns[table]])
            + "]"
            for table in tables
        ]
        foreign_keys = []
        for table in tables:
            for key in connection.execute(
                f"PRAGMA foreign_key_list({quote_identifier(table)})"
            ):
                target_column = key[4]
                if target_column is None:
                    parent = next(
                        (name for name in tables if name.casefold() == key[2].casefold()),
                        None,
                    )
                    primary_keys = sorted(
                        (column for column in columns.get(parent, []) if column[5]),
                        key=lambda column: column[5],
                    )
                    if key[1] >= len(primary_keys):
                        raise ValueError(
                            f"Cannot resolve declared foreign key: "
                            f"{table}.{key[3]} -> {key[2]}"
                        )
                    target_column = primary_keys[key[1]][1]
                foreign_keys.append(
                    f"{table}.{key[3]} = {key[2]}.{target_column}"
                )
        parts.append("Foreign_keys = [" + ",".join(foreign_keys) + "]")
        return "\n".join(parts)
    finally:
        connection.close()


def create_bigquery_client():
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ModuleNotFoundError:
        sys.path.insert(0, str(BIGQUERY_VENDOR))
        from google.cloud import bigquery
        from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        str(BIGQUERY_CREDENTIAL)
    )
    credential_info = json.loads(BIGQUERY_CREDENTIAL.read_text(encoding="utf-8"))
    return bigquery.Client(
        credentials=credentials,
        project=credential_info["project_id"],
    )


def flatten_bigquery_columns(fields, prefix: str = "") -> list[str]:
    columns = []
    for field in fields:
        name = f"{prefix}.{field.name}" if prefix else field.name
        columns.append(name)
        if field.fields:
            columns.extend(flatten_bigquery_columns(field.fields, name))
    return columns


def bigquery_table_names(resource_dir: Path) -> list[str]:
    table_names = set()
    for path in sorted(resource_dir.rglob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        table_name = metadata.get("table_fullname")
        if not isinstance(table_name, str) or table_name.count(".") != 2:
            raise ValueError(f"Invalid BigQuery table metadata: {path}")
        table_names.add(".".join(part.strip() for part in table_name.split(".")))
    if not table_names:
        raise ValueError(f"No BigQuery tables found in {resource_dir}")
    return sorted(table_names)


def bigquery_schema_context(client, resource_dir: Path, table_cache: dict) -> str:
    parts = []
    foreign_keys = []
    for table_name in bigquery_table_names(resource_dir):
        table = table_cache.get(table_name)
        if table is None:
            table = client.get_table(table_name, timeout=60)
            table_cache[table_name] = table
        columns = flatten_bigquery_columns(table.schema)
        parts.append(
            f"Table {table_name}, columns = [" + ",".join(["*"] + columns) + "]"
        )
        constraints = getattr(table, "table_constraints", None)
        for foreign_key in (constraints.foreign_keys or []) if constraints else []:
            referenced = foreign_key.referenced_table
            referenced_table = (
                f"{referenced.project}.{referenced.dataset_id}.{referenced.table_id}"
            )
            for column in foreign_key.column_references:
                foreign_keys.append(
                    f"{table_name}.{column.referencing_column} = "
                    f"{referenced_table}.{column.referenced_column}"
                )
    parts.append("Foreign_keys = [" + ",".join(foreign_keys) + "]")
    return "\n".join(parts)


def build_prompt(examples: str, context: str, instance: dict) -> str:
    question = "Q: " + instance["question"]
    if instance.get("evidence"):
        question += "\nHint: " + instance["evidence"]
    return (
        "Find the schema_links for generating SQL queries for each question based "
        "on the database schema and Foreign keys.\n\n"
        + examples
        + "\n\nSchema of the current database:\n"
        + context
        + "\n\n"
        + question
        + "\nA: Let’s think step by step.\n"
    )


def datasource_map() -> dict[str, dict[str, str]]:
    with SOURCE_MAP_PATH.open(encoding="utf-8-sig", newline="") as source:
        return {row["instance_id"]: row for row in csv.DictReader(source)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="spider2", choices=("spider2",))
    parser.add_argument("--split", default="lite", choices=("lite",))
    parser.add_argument(
        "--llm", choices=MODEL_ALIASES, help="Omit to generate prompts only"
    )
    args = parser.parse_args()

    instances = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    sources = datasource_map()
    selected_instances = []
    source_by_index = {}
    for instance in instances:
        index = str(instance["index"])
        source = sources.get(index)
        if source is None:
            raise ValueError(f"Instance {index}: missing datasource mapping")
        backend = source["backend"].casefold()
        if backend == "sqlite":
            database = DATABASE_ROOT / f"{source['db']}.sqlite"
            if not database.is_file():
                raise FileNotFoundError(
                    f"Instance {index}: SQLite database not found: {database}"
                )
            source_by_index[index] = {"backend": backend, "path": database}
        elif backend == "bigquery":
            resource_dir = BIGQUERY_RESOURCE_ROOT / source["db"]
            if not resource_dir.is_dir():
                raise FileNotFoundError(
                    f"Instance {index}: BigQuery metadata not found: {resource_dir}"
                )
            source_by_index[index] = {"backend": backend, "path": resource_dir}
        else:
            continue
        selected_instances.append(instance)

    ids = [str(instance["index"]) for instance in selected_instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Instance index must be unique")

    output_dir = HERE / DATASET_SPLIT / "schema_linking"
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    completed_prompt_ids = {
        index
        for index in ids
        if (prompt_dir / f"{index}.txt").is_file()
        and (prompt_dir / f"{index}.txt").stat().st_size > 0
    }
    result_path = output_dir / f"{args.llm}_result.json" if args.llm else None
    results = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path and result_path.exists()
        else {}
    )
    progress = Progress(
        len(ids),
        sum(
            bool(results.get(index, {}).get("status", {}).get("success"))
            for index in ids
        ),
    )
    progress.prompts = len(completed_prompt_ids)
    progress.tick(force=True)
    bigquery_client = None
    try:
        examples = official_examples("spider")
        context_by_source = {}
        bigquery_table_cache = {}
        for instance in selected_instances:
            index = str(instance["index"])
            if index in completed_prompt_ids:
                progress.tick()
                continue
            source = source_by_index[index]
            source_key = (source["backend"], source["path"])
            try:
                context = context_by_source.get(source_key)
                if context is None:
                    if source["backend"] == "sqlite":
                        context = sqlite_schema_context(source["path"])
                    else:
                        if bigquery_client is None:
                            bigquery_client = create_bigquery_client()
                        context = bigquery_schema_context(
                            bigquery_client,
                            source["path"],
                            bigquery_table_cache,
                        )
                    context_by_source[source_key] = context
                prompt = build_prompt(examples, context, instance)
                (prompt_dir / f"{index}.txt").write_text(prompt, encoding="utf-8")
            except Exception as exc:
                progress.log(index, "prompt", False, str(exc))
                raise
            progress.prompts += 1
            progress.log(index, "prompt", True)
            progress.tick()
        progress.tick(force=True)
        if args.llm:
            run_results(
                ids,
                prompt_dir,
                result_path,
                results,
                model_config(args.llm),
                "spider",
                progress,
            )
    finally:
        if bigquery_client is not None:
            bigquery_client.close()
        progress.close()


if __name__ == "__main__":
    main()
