"""Enrich existing Spider2-Lite metadata CSVs with declared primary/foreign keys.

Only primary_key and ref_key are updated. primary_key is the 1-based position in
the table's primary key, or blank. ref_key retains the existing semicolon-separated
table.column format. Undeclared keys stay blank. Ambiguous mappings abort.
Confirmed missing databases/tables cause the whole logical database's Meta
and instances to be removed. Access errors abort before writing any changes.
Only our preprocessed_data is updated; original Spider2 resources are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LITE_ROOT = ROOT / "Spider2.0/spider2-lite"


def database_key(name: str) -> str:
    return name.casefold().replace("-", "_")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def source_map(lite_root: Path) -> dict[str, tuple[str, str, str]]:
    sources = defaultdict(set)
    with (lite_root / "spider2-lite_datasource_map.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["backend"] not in ("sqlite", "bigquery"):
                continue
            value = (row["backend"], row["db"], row["matched_resource_db"])
            for name in (row["db"], row["matched_resource_db"]):
                if name:
                    sources[database_key(name)].add(value)
    return {key: next(iter(values)) for key, values in sources.items() if len(values) == 1}


def match_table(meta_name: str, names: list[str], backend: str) -> str | None:
    """Prefer full names, then BigQuery dataset.table; allow only unique leaf names."""
    # Existing Meta uses Snowflake-compatible names (_311 for 311, for example).
    def key(name: str) -> str:
        return ".".join(re.sub(r"^_(?=\d)", "", part.strip().casefold())
                        for part in name.split("."))

    folded = key(meta_name)
    exact = [name for name in names if key(name) == folded]
    if len(exact) > 1:
        raise ValueError(f"Ambiguous table mapping for {meta_name}: {exact}")
    if len(exact) == 1:
        return exact[0]
    if backend == "bigquery":
        suffix = ".".join(folded.split(".")[-2:])
        matches = [name for name in names if ".".join(key(name).split(".")[-2:]) == suffix]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous table mapping for {meta_name}: {matches}")
        if matches:
            return matches[0] if len(matches) == 1 else None
    matches = [name for name in names if key(name).split(".")[-1] == folded.split(".")[-1]]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous table mapping for {meta_name}: {matches}")
    return matches[0] if len(matches) == 1 else None


def sqlite_keys(database: Path) -> dict[str, dict]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
        )]
        columns = {table: connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
                   for table in tables}
        keys = {table: {"primary": [col[1] for col in sorted(columns[table], key=lambda c: c[5]) if col[5]],
                        "foreign": []} for table in tables}
        canonical = {table.casefold(): table for table in tables}
        for table in tables:
            for key in connection.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})"):
                parent = canonical.get(key[2].casefold(), key[2])
                target = key[4]
                if target is None:
                    primary = keys.get(parent, {}).get("primary", [])
                    if key[1] >= len(primary):
                        print(f"{database.name}/{table}: unresolved FK {key[0]}, left blank", flush=True)
                        continue
                    target = primary[key[1]]
                keys[table]["foreign"].append((key[3], parent, target))
        return keys
    finally:
        connection.close()


def create_bigquery_client(credential: Path, vendor: Path):
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ModuleNotFoundError:
        sys.path.insert(0, str(vendor))
        from google.cloud import bigquery
        from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(str(credential))
    return bigquery.Client(credentials=credentials, project=credentials.project_id)


def bigquery_keys(client, table_name: str) -> dict:
    # tables.get reads schema metadata only; it does not execute a SQL query.
    table = client.get_table(table_name, timeout=60, retry=None)
    constraints = table.to_api_repr().get("tableConstraints") or {}
    primary = (constraints.get("primaryKey") or {}).get("columns") or []
    foreign = []
    for key in constraints.get("foreignKeys") or []:
        target = key["referencedTable"]
        parent = ".".join(target[field] for field in ("projectId", "datasetId", "tableId"))
        for column in key["columnReferences"]:
            foreign.append((column["referencingColumn"], parent, column["referencedColumn"]))
    return {"primary": primary, "foreign": foreign}


def bigquery_catalog(client, resource_names: list[str], meta_names: list[str]) -> tuple[list[str], str | None]:
    """Resolve old Meta names against the live catalog, including views."""
    resource_datasets = set(".".join(name.split(".")[:2]) for name in resource_names)
    datasets = set()
    for meta_name in meta_names:
        matched = match_table(meta_name, resource_names, "bigquery")
        if matched:
            datasets.add(".".join(matched.split(".")[:2]))
        else:
            parts = meta_name.split(".")
            candidates = [name for name in resource_datasets
                          if len(parts) >= 2 and name.split(".")[-1].casefold() == parts[-2].casefold()]
            if len(candidates) != 1:
                raise ValueError(f"Cannot determine physical BigQuery dataset for {meta_name}")
            datasets.add(candidates[0])
    live_names = []
    for dataset in sorted(datasets):
        try:
            client.get_dataset(dataset, timeout=60, retry=None)
            live_names.extend(f"{table.project}.{table.dataset_id}.{table.table_id}"
                              for table in client.list_tables(dataset, timeout=60, retry=None))
        except Exception as exc:
            if getattr(exc, "code", None) == 404:
                return [], f"BigQuery dataset not found: {dataset}"
            raise RuntimeError(f"Cannot verify BigQuery dataset {dataset}: {exc}") from exc
    return sorted(set(live_names)), None


def verify_bigquery(client, names: list[str]) -> tuple[dict, str | None]:
    """Only a confirmed 404 means missing; permissions/timeouts are not absence."""
    keys = {}
    for number, name in enumerate(names, 1):
        try:
            keys[name] = bigquery_keys(client, name)
        except Exception as exc:
            if getattr(exc, "code", None) == 404:
                return {}, f"BigQuery table not found: {name}"
            raise RuntimeError(f"Cannot verify BigQuery table {name}: {exc}") from exc
        print(f"BigQuery {number}/{len(names)}｜{name}", flush=True)
    return keys, None


def read_meta(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if "column_name" not in fields:
            raise ValueError(f"Missing column_name in {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed metadata CSV: {path}")
    return fields, rows


def write_keys(path: Path, fields: list[str], rows: list[dict], keys: dict,
               targets: dict[str, tuple[str, dict[str, str]]]) -> tuple[int, int]:
    primary = {name.casefold(): str(position) for position, name in enumerate(keys.get("primary", []), 1)}
    foreign = defaultdict(list)
    for column, parent, target_column in keys.get("foreign", []):
        target = targets.get(parent.casefold())
        table_name = target[0] if target else parent
        column_name = target[1].get(target_column.casefold(), target_column) if target else target_column
        ref = f"{table_name}.{column_name}"
        if ref not in foreign[column.casefold()]:
            foreign[column.casefold()].append(ref)
    output_fields = list(fields)
    for field in ("primary_key", "ref_key"):
        if field not in output_fields:
            output_fields.append(field)
    primary_count = foreign_count = 0
    for row in rows:
        name = row["column_name"].casefold()
        row["primary_key"] = primary.get(name, "")
        row["ref_key"] = "; ".join(foreign.get(name, []))
        primary_count += bool(row["primary_key"])
        foreign_count += bool(row["ref_key"])
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return primary_count, foreign_count


def preprocess(lite_root: Path, meta_root: Path, credential: Path,
               backend: str = "all", db_id: str | None = None) -> dict:
    instances_path = meta_root.parent / "spider2_lite.json"
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    sources = source_map(lite_root)
    directories = [meta_root / db_id] if db_id else sorted(p for p in meta_root.iterdir() if p.is_dir())
    client = None
    totals = {"databases": 0, "tables": 0, "primary_key_columns": 0,
              "foreign_key_columns": 0, "removed_databases": [], "removed_instances": 0}
    updates = []
    removed = {}
    try:
        for directory in directories:
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            source = sources.get(database_key(directory.name))
            if source is None:
                raise ValueError(f"No unique datasource mapping for {directory.name}")
            kind, db, resource_db = source
            if backend != "all" and backend != kind:
                continue
            files = sorted(directory.glob("*.csv"))
            metadata = {path: read_meta(path) for path in files}
            all_keys = {}
            if kind == "sqlite":
                database = lite_root / "resource/databases/spider2-localdb" / f"{db}.sqlite"
                if not database.exists():
                    removed[directory] = f"SQLite database not found: {database}"
                    continue
                all_keys = sqlite_keys(database)
                names = list(all_keys)
            else:
                resource = lite_root / "resource/databases/bigquery" / db
                if not resource.is_dir():
                    resource = lite_root / "resource/databases/bigquery" / resource_db
                names = sorted({".".join(part.strip() for part in
                                        json.loads(path.read_text(encoding="utf-8"))["table_fullname"].split("."))
                                for path in resource.rglob("*.json")})
                if not names:
                    raise ValueError(f"No source table catalog found for {directory.name}; cannot verify existence")
                if client is None:
                    client = create_bigquery_client(credential, lite_root / "evaluation_suite/_vendor")
                names, reason = bigquery_catalog(client, names, [path.stem for path in files])
                if reason:
                    removed[directory] = reason
                    print(f"{directory.name}｜will remove database, Meta and instances｜{reason}", flush=True)
                    continue
            matched = {path: match_table(path.stem, names, kind) for path in files}
            unmatched = [path.stem for path, name in matched.items() if name is None]
            if unmatched:
                removed[directory] = f"Tables absent from live {kind} catalog: {unmatched}"
                print(f"{directory.name}｜will remove database, Meta and instances｜{removed[directory]}", flush=True)
                continue
            # Translate physical FK targets into the names already used by our Meta.
            grouped = defaultdict(list)
            for path, name in matched.items():
                if name:
                    grouped[name.casefold()].append(path)
            targets = {
                name: (paths[0].stem, {row["column_name"].casefold(): row["column_name"]
                                      for row in metadata[paths[0]][1]})
                for name, paths in grouped.items() if len(paths) == 1
            }
            if kind == "bigquery":
                all_keys, reason = verify_bigquery(client, list(dict.fromkeys(matched.values())))
                if reason:
                    removed[directory] = reason
                    print(f"{directory.name}｜will remove database, Meta and instances｜{reason}", flush=True)
                    continue
            for path in files:
                name = matched[path]
                updates.append((path, *metadata[path], all_keys.get(name, {}), targets))
            totals["databases"] += 1
            print(f"{directory.name}｜{kind}｜verified {len(files)} tables", flush=True)
    finally:
        if client is not None:
            client.close()
    # Apply only after every selected database has been checked successfully.
    for update in updates:
        pk, fk = write_keys(*update)
        totals["tables"] += 1
        totals["primary_key_columns"] += pk
        totals["foreign_key_columns"] += fk
    if removed:
        removed_names = {database_key(directory.name) for directory in removed}
        retained = [record for record in instances if database_key(record["db_id"]) not in removed_names]
        totals["removed_instances"] = len(instances) - len(retained)
        temporary = instances_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(retained, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(instances_path)
        for directory, reason in removed.items():
            shutil.rmtree(directory)
            totals["removed_databases"].append(directory.name)
            print(f"Removed {directory.name}｜{reason}", flush=True)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("spider2",), default="spider2")
    parser.add_argument("--split", choices=("lite",), default="lite")
    parser.add_argument("--backend", choices=("all", "sqlite", "bigquery"), default="all")
    parser.add_argument("--db-id", help="Process one existing Meta database directory")
    parser.add_argument("--lite-root", type=Path, default=LITE_ROOT)
    parser.add_argument("--meta-root", type=Path, default=HERE / "spider2_lite/preprocessed_data/meta")
    parser.add_argument("--credential", type=Path, help="BigQuery service-account JSON")
    args = parser.parse_args()
    credential = args.credential or args.lite_root / "evaluation_suite/bigquery_credential.json"
    summary = preprocess(args.lite_root, args.meta_root, credential, args.backend, args.db_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
