"""Build DIN-SQL schema-linking prompts; optionally run one configured LLM.

Examples:
    python schema_linking.py --dataset spider --split dev
    python schema_linking.py --dataset bird --split dev --llm qwen38
    python schema_linking.py --dataset spider --split dev --llm qwen38 --filteredmeta

No --llm means no API requests. Each invocation rebuilds all prompts and retries
only missing/failed results. Keep successful results only when resuming the same
experiment: changing prompts/configuration does not invalidate them automatically.
--filteredmeta uses only tables/columns from code/scripts/{dataset}_{split}/filtered_meta.json,
writing prompt_filtered_meta/ and {llm}_result_filtered_meta.json. No RC text is added.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import queue
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_ALIASES = ("qwen38", "kimik3", "gpt56", "opus48")
TIMEOUT = 900
INITIAL_CONCURRENCY = 50
MAX_CONCURRENCY = 1500
INSTRUCTION = (
    "Find the schema_links for generating SQL queries for each question based on "
    "the database schema and Foreign keys.\n"
    "Use column descriptions, value descriptions, sample rows, and Hint when provided."
)


def official_examples(dataset: str) -> str:
    """Read only the literal prompt assignment; never execute the author script."""
    filename = "DIN-SQL.py" if dataset == "spider" else "DIN-SQL_BIRD.py"
    name = "schema_linking_prompt" if dataset == "spider" else "SYSTEM_SCHEMA_LINKING_TEMPLATE"
    source = (ROOT / "code/baselines/DIN-SQL" / filename).read_text(encoding="utf-8")
    match = re.search(r"^" + name + r"\s*=\s*('''|\"\"\")(.*?)\1", source, re.M | re.S)
    if not match:
        raise ValueError(f"Official few-shot assignment not found: {filename}:{name}")
    literal = ast.literal_eval(match.group(1) + match.group(2) + match.group(1))
    if dataset == "bird":
        literal = literal.split("Few examples of this task are:\n###\n", 1)[1]
    return literal.strip()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def column_descriptions(meta_dir: Path) -> dict:
    descriptions = {}
    for path in sorted(meta_dir.glob("*.csv")):
        raw = path.read_bytes()
        try:
            contents = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            contents = raw.decode("latin-1")  # Official BIRD description encoding.
        columns = {}
        for row in csv.DictReader(io.StringIO(contents)):
            name = row.get("original_column_name") or row.get("column_name")
            if name:
                columns[name.casefold()] = (
                    row.get("column_description") or "",
                    row.get("value_description") or "",
                )
        descriptions[path.stem.casefold()] = columns
    return descriptions


def database_context(database: Path, meta_dir: Path, filtered: list | None = None) -> str:
    descriptions = column_descriptions(meta_dir)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        columns = {table: connection.execute(
            f"PRAGMA table_info({quote_identifier(table)})"
        ).fetchall() for table in tables}
        allowed = None
        if filtered is not None:
            allowed = {}
            for item in filtered:
                table = next((t for t in tables if t.casefold() == item["table_name"].casefold()), None)
                if table is None:
                    raise ValueError(f"Unknown filtered table: {item['table_name']}")
                names = [col[1] for col in columns[table]]
                keep = allowed.setdefault(table, set())
                for col in item["columns"]:
                    name = col.get("original_column_name", col.get("column_name"))
                    matches = [n for n in names if n.casefold() == name.casefold()]
                    if not matches:
                        # BIRD CSV names can contain padding absent from SQLite.
                        matches = [n for n in names if n.strip().casefold() == name.strip().casefold()]
                    if len(matches) != 1:
                        raise ValueError(f"Cannot resolve filtered column: {table}.{name}")
                    keep.add(matches[0])
        visible_tables = tables if allowed is None else [t for t in tables if t in allowed]
        parts = []
        foreign_keys = []
        for table in visible_tables:
            names = [column[1] for column in columns[table] if allowed is None or column[1] in allowed[table]]
            parts.append(f"Table {table}, columns = [" + ",".join(["*"] + names) + "]")
            for key in connection.execute(f"PRAGMA foreign_key_list({quote_identifier(table)})"):
                target = key[4]
                if target is None:
                    # REFERENCES parent with no column list refers to its primary key.
                    parent = next((name for name in tables if name.casefold() == key[2].casefold()), None)
                    primary = sorted((col for col in columns.get(parent, []) if col[5]), key=lambda col: col[5])
                    if key[1] >= len(primary):
                        raise ValueError(f"Cannot resolve declared foreign key: {table}.{key[3]} -> {key[2]}")
                    target = primary[key[1]][1]
                if allowed is not None:
                    parent = next((t for t in visible_tables if t.casefold() == key[2].casefold()), None)
                    if (key[3].casefold() not in {n.casefold() for n in names}
                            or parent is None
                            or target.casefold() not in {n.casefold() for n in allowed[parent]}):
                        continue
                foreign_keys.append(f"{table}.{key[3]} = {key[2]}.{target}")
        parts.append("Foreign_keys = [" + ",".join(foreign_keys) + "]")
        for table in visible_tables:
            names = [column[1] for column in columns[table] if allowed is None or column[1] in allowed[table]]
            # Same LIMIT 3 policy as the author's BIRD reader; no value search.
            projection = "*" if allowed is None else ", ".join(quote_identifier(name) for name in names)
            rows = connection.execute(f"SELECT {projection} FROM {quote_identifier(table)} LIMIT 3").fetchall() if names else []
            parts.append(f"Sample rows from {table} (columns: {json.dumps(names, ensure_ascii=False)}):")
            parts.append(json.dumps(rows, ensure_ascii=False, default=str))
            parts.append(f"Column descriptions and value descriptions for {table}:")
            for name in names:
                description, values = descriptions.get(table.casefold(), {}).get(name.casefold(), ("", ""))
                parts.append(
                    f"Column {name}: column description -> {description}, value description -> {values}"
                )
        return "\n".join(parts)
    finally:
        connection.close()


def build_prompt(examples: str, context: str, instance: dict) -> str:
    question = "Q: " + instance["question"]
    if instance.get("evidence"):
        question += "\nHint: " + instance["evidence"]
    return (
        INSTRUCTION + "\n\n" + examples + "\n\n"
        + "Schema of the current database with sample rows and column descriptions:\n"
        + context + "\n\n" + question + "\nA: Let’s think step by step.\n"
    )


def extract_schema_links(content: str) -> list[str]:
    match = re.search(r"Schema_links:\s*\[(.*?)\]", content, re.S)
    if not match or not match.group(1).strip():
        return []
    return [link.strip() for link in match.group(1).split(",")]


def model_config(alias: str) -> dict:
    config = {}
    for line in (ROOT / "code/config/.env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip("\"'")
    prefix = config[alias]
    return {
        "model": config[prefix],
        "api_key": config[prefix + "_API_KEY"],
        "url": config[prefix + "_BASE_URL"].rstrip("/") + "/chat/completions",
    }


def stream_events(response):
    """Decode SSE events, including comments and multi-line data fields."""
    data_lines = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
    if data_lines:
        yield "\n".join(data_lines)


def call_model(prompt: str, config: dict, dataset: str) -> dict:
    started = time.monotonic()
    usage = None
    try:
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 5000 if dataset == "spider" else 5000,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if dataset == "spider":
            payload.update(n=1, top_p=1.0, frequency_penalty=0.0,
                           presence_penalty=0.0, stop=["Q:"])
        request = Request(
            config["url"], data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + config["api_key"],
                     "Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with urlopen(request, timeout=TIMEOUT) as response:
            content_parts = []
            finished = False
            for data in stream_events(response):
                if data.strip() == "[DONE]":
                    finished = True
                    break
                chunk = json.loads(data)
                if not isinstance(chunk, dict):
                    raise ValueError("Model stream event must be a JSON object")
                if chunk.get("error") is not None:
                    raise RuntimeError(f"Model stream error: {chunk['error']}")
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    piece = (choice.get("delta") or {}).get("content")
                    if isinstance(piece, str):
                        content_parts.append(piece)
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        if finish_reason != "stop":
                            raise ValueError(f"Model stream finished with {finish_reason!r}")
                        finished = True
            if not finished:
                raise ValueError("Model stream ended before a completion marker")
        content = "".join(content_parts)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned empty content")
        result = extract_schema_links(content)
        status = {"success": True, "reason": ""}
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, HTTPError):
            try:
                reason += " | " + exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass  # Preserve the HTTP failure even if its response body is unreadable.
            finally:
                exc.close()
        reason = reason.replace(config["api_key"], "[redacted]") if config["api_key"] else reason
        result = None
        status = {"success": False, "reason": reason}
    return {"result": result, "status": status, "resource": [usage, round(time.monotonic() - started, 3)]}


class Progress:
    def __init__(self, total: int, results: int = 0):
        self.total, self.prompts, self.results = total, 0, results
        self.tty = sys.stdout.isatty()
        self.last_update = 0.0
        self.bar = ""
        self.tick(force=True)

    def tick(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last_update >= 10:
            self.last_update = now
            self.bar = f"In total {self.total} instance, {self.total - self.prompts} prompt left, {self.total -self.results} schema linking result left"
            if self.tty:
                print("\r\033[2K" + self.bar, end="", flush=True)
            else:
                print(self.bar, flush=True)

    def log(self, instance_id: str, stage: str, success: bool, reason: str = "") -> None:
        label = "success" if success else "failed"
        if self.tty:
            label = ("\033[32m" if success else "\033[31m") + label + "\033[0m"
        if stage != 'prompt':
            self.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜{instance_id}｜【{stage}】｜{label}")
        if reason:
            self.message("  " + " ".join(reason.splitlines()))

    def message(self, text: str) -> None:
        if self.tty:
            print("\r\033[2K" + text, flush=True)
            print(self.bar, end="", flush=True)
        else:
            print(text, flush=True)

    def close(self) -> None:
        self.tick(force=True)
        if self.tty:
            print()


class Concurrency:
    def __init__(self):
        self.limit = INITIAL_CONCURRENCY
        self.last_increase = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        while self.limit < MAX_CONCURRENCY and now - self.last_increase >= 1:
            self.limit = min(MAX_CONCURRENCY, self.limit + random.randint(40, 60))
            self.last_increase += 1


def save_results(path: Path, results: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_results(ids: list[str], prompt_dir: Path, result_path: Path, results: dict,
                config: dict, dataset: str, progress: Progress) -> None:
    pending = iter(index for index in ids if not results.get(index, {}).get("status", {}).get("success", False))
    completed = queue.Queue()
    active = {}
    exhausted = False
    concurrency = Concurrency()
    last_report = time.monotonic()
    progress.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜concurrency limit：{concurrency.limit}")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        while active or not exhausted:
            concurrency.tick()
            while not exhausted and len(active) < concurrency.limit:
                index = next(pending, None)
                if index is None:
                    exhausted = True
                    break
                prompt = (prompt_dir / f"{index}.txt").read_text(encoding="utf-8")
                future = executor.submit(call_model, prompt, config, dataset)
                active[future] = index
                future.add_done_callback(completed.put)
            if not active:
                break
            try:
                wait = (max(0.001, concurrency.last_increase + 1 - time.monotonic())
                        if concurrency.limit < MAX_CONCURRENCY else 1)
                future = completed.get(timeout=wait)
            except queue.Empty:
                pass
            else:
                index = active.pop(future)
                record = future.result()
                results[index] = record
                save_results(result_path, results)
                success = record["status"]["success"]
                progress.results += int(success)
                progress.log(index, "result", success, record["status"]["reason"])
            progress.tick()
            if time.monotonic() - last_report >= 30:
                last_report = time.monotonic()
                progress.message(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}｜concurrency.limit：{concurrency.limit}，active：{len(active)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("spider", "bird"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--llm", choices=MODEL_ALIASES, help="Omit to generate prompts only")
    parser.add_argument("--filteredmeta", action="store_true", help="Use filtered_meta.json tables/columns without RC text")
    args = parser.parse_args()
    if args.dataset == "bird" and args.split != "dev":
        parser.error("Only bird/dev, spider/dev and spider/test are currently supported")
    dataset_split = f"{args.dataset}_{args.split}"
    data_dir = ROOT / "code/scripts" / dataset_split / "preprocessed_data"
    instances = json.loads((data_dir / f"{dataset_split}.json").read_text(encoding="utf-8"))
    ids = [str(instance["index"]) for instance in instances]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[0-9]+", index) for index in ids):
        raise ValueError("Instance index must be a unique nonnegative integer")
    filtered_by_index = {}
    if args.filteredmeta:
        filtered_path = data_dir.parent / "filtered_meta.json"
        filtered_by_index = json.loads(filtered_path.read_text(encoding="utf-8"))
        for index in ids:
            record = filtered_by_index.get(index)
            if not isinstance(record, dict) or not isinstance(record.get("result"), list):
                raise ValueError(f"Instance {index}: missing or invalid filtered metadata in {filtered_path}")
    if args.dataset == "bird":
        databases = ROOT / "BIRD/data/dev/dev_databases"
    else:
        databases = ROOT / "Spider/data" / ("database" if args.split == "dev" else "test_database")
    output_dir = HERE / dataset_split / "schema_linking"
    prompt_dir = output_dir / ("prompt_filtered_meta" if args.filteredmeta else "prompts")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_filtered_meta" if args.filteredmeta else ""
    result_path = output_dir / f"{args.llm}_result{suffix}.json" if args.llm else None
    results = json.loads(result_path.read_text(encoding="utf-8")) if result_path and result_path.exists() else {}
    config = model_config(args.llm) if args.llm else None
    progress = Progress(len(ids), sum(bool(results.get(index, {}).get("status", {}).get("success")) for index in ids))
    try:
        examples = official_examples(args.dataset)
        # Full schema can be shared; filtered schema is instance-specific.
        by_database = {}
        for instance in instances:
            by_database.setdefault(instance["db_id"], []).append(instance)
        for db_id, group in by_database.items():
            try:
                context = None if args.filteredmeta else database_context(databases / db_id / f"{db_id}.sqlite", data_dir / "meta" / db_id)
            except Exception as exc:
                progress.log(str(group[0]["index"]), "prompt", False, str(exc))
                raise
            for instance in group:
                index = str(instance["index"])
                try:
                    if args.filteredmeta:
                        context = database_context(databases / db_id / f"{db_id}.sqlite", data_dir / "meta" / db_id,
                                                   filtered_by_index[index]["result"])
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
            run_results(ids, prompt_dir, result_path, results, config, args.dataset, progress)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
