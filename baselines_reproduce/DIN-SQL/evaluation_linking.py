"""Write baseline and filtered-metadata schema-linking coverage CSVs.

Coverage is recall over unique gold table.column references, averaged per question.
Both variants include every gold sample; missing/failed predictions score zero.
Tokens are the recorded linking call's total_tokens, excluding RC generation and
previous retries. Missing usage stays blank. No model calls or SQL evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIELDS = (
    "index", "q", "dbid", "gold_linking", "pred_linking", "missing_linking",
    "coverage", "total_tokens", "status",
)


def name_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def identifier_pattern(value: str) -> str:
    return r"\s+".join(re.escape(part) for part in value.split())


class Schema:
    def __init__(self, tables: dict[str, list[str]]):
        self.tables = {name_key(table): table for table in tables}
        self.columns = {
            name_key(table): {name_key(column): column for column in columns}
            for table, columns in tables.items()
        }
        self.patterns = {}
        for table, columns in tables.items():
            # Longest first prevents a column such as "name" matching "name full".
            alternatives = [identifier_pattern(column)
                            for column in sorted(columns, key=len, reverse=True)]
            alternatives.append(r"\*")
            self.patterns[name_key(table)] = re.compile(
                r"(?<![\w.])" + identifier_pattern(table) + r"\s*\.\s*("
                + "|".join(alternatives) + r")(?![\w])", re.IGNORECASE,
            )

    @classmethod
    def from_database(cls, database: Path) -> Schema:
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            return cls({
                table: [row[1] for row in connection.execute(
                    'PRAGMA table_info("' + table.replace('"', '""') + '")'
                )]
                for table in tables
            })
        finally:
            connection.close()

    def gold_links(self, linking: list[dict]) -> set[tuple[str, str]]:
        links = set()
        for item in linking:
            table = name_key(item["table"])
            for column_name in item["columns"]:
                column = name_key(column_name)
                if table not in self.tables or (column != "*" and column not in self.columns[table]):
                    raise ValueError(f"Unknown gold schema reference: {item['table']}.{column_name}")
                links.add((table, column))
        if not links:
            raise ValueError("Gold schema linking is empty; coverage is undefined")
        return links

    def predicted_links(self, linking: list[str]) -> tuple[set, set]:
        links, tables = set(), set()
        for item in linking:
            # Values are not schema links. Dequote identifier spellings only.
            text = re.sub(r"'(?:''|[^'])*'", " ", item)
            text = re.sub(r'`((?:``|[^`])*)`', lambda m: m[1].replace('``', '`'), text)
            text = re.sub(r'"((?:""|[^"])*)"', lambda m: m[1].replace('""', '"'), text)
            text = re.sub(r'\[([^\]]+)\]', r'\1', text)
            standalone = name_key(text.strip())
            if standalone in self.tables:
                tables.add(standalone)
            for table, pattern in self.patterns.items():
                for match in pattern.finditer(text):
                    column = name_key(match[1])
                    tables.add(table)
                    if column == "*":
                        links.update((table, name) for name in self.columns[table])
                    else:
                        links.add((table, column))
        return links, tables

    def display(self, links: set[tuple[str, str]]) -> list[str]:
        return [self.tables[table] + "." +
                ("*" if column == "*" else self.columns[table][column])
                for table, column in sorted(links)]


def evaluate_row(gold: dict, record: dict | None, schema: Schema) -> dict:
    gold_links = schema.gold_links(gold["schemalinking"])
    prediction = record.get("result") if isinstance(record, dict) else None
    if record is None:
        status = "missing"
    elif not isinstance(record, dict) or not isinstance(record.get("status"), dict):
        status = "invalid_result"
    elif record["status"].get("success") is not True:
        status = "failed"
    elif not isinstance(prediction, list) or not all(isinstance(item, str) for item in prediction):
        status = "invalid_result"
    else:
        status = "success"
    links, tables = schema.predicted_links(prediction) if status == "success" else (set(), set())
    missing = {link for link in gold_links
               if not (link in links or (link[1] == "*" and link[0] in tables))}
    # A gold '*' denotes a table-only dependency, e.g. COUNT(*), not all columns.
    resource = record.get("resource") if isinstance(record, dict) else None
    usage = resource[0] if isinstance(resource, list) and resource else None
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        tokens = ""
    return {
        "index": gold["index"], "q": gold["question"], "dbid": gold["db_id"],
        "gold_linking": json.dumps(schema.display(gold_links), ensure_ascii=False),
        "pred_linking": json.dumps(prediction, ensure_ascii=False),
        "missing_linking": json.dumps(schema.display(missing), ensure_ascii=False),
        "coverage": (len(gold_links) - len(missing)) / len(gold_links),
        "total_tokens": tokens, "status": status,
    }


def evaluate(dataset: str, split: str, llm: str = "qwen38") -> list[Path]:
    dataset_split = f"{dataset}_{split}"
    gold_path = ROOT / "code/scripts" / dataset_split / "gold_sql_schema_linking.json"
    stage = HERE / dataset_split / "schema_linking"
    if dataset == "bird" and split == "dev":
        databases = ROOT / "BIRD/data/dev/dev_databases"
    elif dataset == "spider" and split in ("dev", "test"):
        databases = ROOT / "Spider/data" / ("database" if split == "dev" else "test_database")
    else:
        raise ValueError("Supported splits: bird/dev, spider/dev, spider/test")
    gold_records = json.loads(gold_path.read_text(encoding="utf-8"))
    if len({str(record["index"]) for record in gold_records}) != len(gold_records):
        raise ValueError("Duplicate gold indices")
    variants = {
        suffix: json.loads((stage / f"{llm}_result{suffix}.json").read_text(encoding="utf-8"))
        for suffix in ("", "_filtered_meta")
    }
    if not all(isinstance(records, dict) for records in variants.values()):
        raise ValueError("Prediction files must contain objects keyed by sample index")
    schemas = {
        db_id: Schema.from_database(databases / db_id / f"{db_id}.sqlite")
        for db_id in dict.fromkeys(record["db_id"] for record in gold_records)
    }
    outputs = []
    for suffix, predictions in variants.items():
        rows = [evaluate_row(record, predictions.get(str(record["index"])), schemas[record["db_id"]])
                for record in gold_records]
        output_path = stage / f"evaluation{suffix}.csv"
        temporary = output_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
        coverage = f"{sum(row['coverage'] for row in rows) / len(rows):.2%}" if rows else "N/A"
        known_tokens = [row["total_tokens"] for row in rows if row["total_tokens"] != ""]
        print(f"{output_path.name}｜样本 {len(rows)}｜平均覆盖率 {coverage}｜"
              f"失败/缺失/无效 {sum(row['status'] != 'success' for row in rows)}｜"
              f"total_tokens 合计 {sum(known_tokens)}（有用量记录 {len(known_tokens)}/{len(rows)}）",
              flush=True)
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("bird", "spider"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--llm", default="qwen38")
    args = parser.parse_args()
    if args.dataset == "bird" and args.split != "dev":
        parser.error("Only bird/dev is supported for BIRD")
    for path in evaluate(args.dataset, args.split, args.llm):
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
