"""Derive DIN-SQL difficulty from gold SQL and generate NESTED sub-questions."""

from __future__ import annotations

import argparse
import ast
import json
import queue
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from schema_linking import MODEL_ALIASES, database_context, model_config


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TIMEOUT = 500
DECOMPOSITION_INSTRUCTION = (
    "The difficulty label has already been determined from the gold SQL as NESTED.\n"
    "Do not classify the question again. Identify the sub-question or sub-questions "
    "needed to solve the nested query.\n"
    "Return exactly in this format:\n"
    'sub_questions: ["..."]'
)


def literal_assignment(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r"^" + re.escape(name) + r"\s*=\s*('''|\"\"\")(.*?)\1",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"String assignment {name!r} not found in {path}")
    value = ast.literal_eval(match.group(1) + match.group(2) + match.group(1))
    if not isinstance(value, str):
        raise ValueError(f"Assignment {name!r} is not text in {path}")
    return value


def unquoted_sql(sql: str) -> str:
    """Remove comments and quoted content before structural keyword matching."""
    output = []
    index = 0
    quote = None
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            if quote == "]" and char == "]":
                quote = None
            elif quote != "]" and char == quote:
                if following == quote:
                    index += 1
                else:
                    quote = None
            output.append(" ")
        elif char in ("'", '"', "`"):
            quote = char
            output.append(" ")
        elif char == "[":
            quote = "]"
            output.append(" ")
        elif char == "-" and following == "-":
            end = sql.find("\n", index + 2)
            index = len(sql) if end < 0 else end
            output.append(" ")
        elif char == "/" and following == "*":
            end = sql.find("*/", index + 2)
            index = len(sql) if end < 0 else end + 1
            output.append(" ")
        else:
            output.append(char)
        index += 1
    return "".join(output)


def gold_label(record: dict) -> str:
    sql = unquoted_sql(record["gold_sql"])
    if len(re.findall(r"\bSELECT\b", sql, flags=re.IGNORECASE)) > 1 or re.search(
        r"\b(?:INTERSECT|UNION|EXCEPT)\b", sql, flags=re.IGNORECASE
    ):
        return "NESTED"
    tables = record.get("schemalinking")
    if not isinstance(tables, list):
        raise ValueError("Gold schemalinking must be a list")
    if len(tables) > 1 or re.search(r"\bJOIN\b", sql, flags=re.IGNORECASE):
        return "NON-NESTED"
    return "EASY"


def spider_schema(db_id: str) -> str:
    schemas = []
    for filename in ("tables.json", "test_tables.json"):
        schemas.extend(
            json.loads((ROOT / "Spider/data" / filename).read_text(encoding="utf-8"))
        )
    schema = next((item for item in schemas if item["db_id"] == db_id), None)
    if schema is None:
        raise ValueError(f"Spider schema not found: {db_id}")
    tables = schema["table_names_original"]
    columns = {table: ["*"] for table in tables}
    for table_index, column in schema["column_names_original"]:
        if table_index >= 0:
            columns[tables[table_index]].append(column)
    lines = [
        f"Table {table}, columns = [" + ",".join(columns[table]) + "]"
        for table in sorted(tables)
    ]
    foreign_keys = []
    for first, second in schema["foreign_keys"]:
        first_table, first_column = schema["column_names_original"][first]
        second_table, second_column = schema["column_names_original"][second]
        foreign_keys.append(
            f"{tables[first_table]}.{first_column} = {tables[second_table]}.{second_column}"
        )
    lines.append("Foreign_keys = [" + ",".join(foreign_keys) + "]")
    return "\n".join(lines)


def build_prompt(
    dataset: str,
    instance: dict,
    schema_links: list[str],
    database_path: Path,
    meta_dir: Path,
) -> list[dict[str, str]]:
    official_root = ROOT / "code/baselines/DIN-SQL"
    links = "[" + ",".join(schema_links) + "]"
    if dataset == "spider":
        examples = literal_assignment(official_root / "DIN-SQL.py", "classification_prompt")
        prompt = (
            "# "
            + DECOMPOSITION_INSTRUCTION
            + "\n\n# DIN-SQL examples containing difficulty reasoning and nested-query decomposition:\n"
            + spider_schema("college_2")
            + "\n"
            + spider_schema(instance["db_id"])
            + "\n"
            + examples
            + f'Q: "{instance["question"]}\n'
            + "schema_links: "
            + links
            + "\nA: "
            + DECOMPOSITION_INSTRUCTION
        )
        return [{"role": "user", "content": prompt}]

    official_system = literal_assignment(
        official_root / "DIN-SQL_BIRD.py", "SYSTEM_CLASSIFICATION_TEMPLATE"
    )
    examples_marker = "###\nFew examples of this task are:\n###"
    if examples_marker not in official_system:
        raise ValueError("BIRD classification examples were not found")
    examples = official_system.split(examples_marker, 1)[1]
    system = DECOMPOSITION_INSTRUCTION + "\n" + examples_marker + examples
    context = database_context(database_path, meta_dir)
    human = (
        "Schema of the database with sample rows and column descriptions:\n#\n"
        + context
        + "\n#\nQ: "
        + instance["question"]
        + "\nHint: "
        + instance.get("evidence", "")
        + "\nSchema links: "
        + links
        + "\n"
        + DECOMPOSITION_INSTRUCTION
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": human}]


def parse_nested_output(raw: str) -> list[str]:
    match = re.search(
        r"(?:sub[-_ ]questions?|questions)\s*(?:=|:)\s*(\[.*?\])",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Model output does not contain sub-questions")
    text = match.group(1).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        content = text[1:-1].strip().strip("\"'")
        parsed = [content] if content else []
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(question, str) and question.strip() for question in parsed
    ):
        raise ValueError("Parsed sub-questions must be a non-empty list of text")
    return [question.strip() for question in parsed]


def call_model(messages: list[dict[str, str]], config: dict, dataset: str) -> dict:
    started = time.monotonic()
    usage = None
    raw = None
    try:
        payload = {
            "model": config["model"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": 600 if dataset == "spider" else 2000,
        }
        if dataset == "spider":
            payload.update(
                n=1,
                stream=False,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["Q:"],
            )
        request = Request(
            config["url"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + config["api_key"],
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=TIMEOUT) as response:
            body = json.load(response)
        usage = body.get("usage")
        raw = body["choices"][0]["message"]["content"]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Model returned empty content")
        sub_questions = parse_nested_output(raw)
        return {
            "result": {"label": "NESTED", "sub_questions": sub_questions},
            "raw_result": raw,
            "status": {"success": True, "reason": ""},
            "resource": [usage, round(time.monotonic() - started, 3)],
        }
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, HTTPError):
            try:
                reason += " | " + exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            finally:
                exc.close()
        if config.get("api_key"):
            reason = reason.replace(config["api_key"], "[redacted]")
        return {
            "result": None,
            "raw_result": raw,
            "status": {"success": False, "reason": reason},
            "resource": [usage, round(time.monotonic() - started, 3)],
        }


class Progress:
    def __init__(self, total: int, nested: int, prompts: int, results: int):
        self.total = total
        self.nested = nested
        self.prompts = prompts
        self.results = results
        self.tty = sys.stdout.isatty()
        self.last_update = 0.0
        self.bar = ""
        self.tick(force=True)

    def tick(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last_update >= 10:
            self.last_update = now
            self.bar = (
                f"共 {self.total} instance，其中 {self.nested} NESTED；"
                f"remain {self.total-self.results} result of subquestions"
            )
            if self.tty:
                print("\r\033[2K" + self.bar, end="", flush=True)
            else:
                print(self.bar, flush=True)

    def message(self, text: str) -> None:
        if self.tty:
            print("\r\033[2K" + text, flush=True)
            print(self.bar, end="", flush=True)
        else:
            print(text, flush=True)

    def log(self, index: str, stage: str, success: bool, reason: str = "") -> None:
        label = "success" if success else "failed"
        if self.tty:
            label = ("\033[32m" if success else "\033[31m") + label + "\033[0m"
        suffix = "｜" + " ".join(reason.splitlines()) if reason else ""
        if stage == 'result':
            self.message(
                f"{datetime.now():%Y-%m-%d %H:%M:%S}｜{index}｜【{stage}】｜{label}{suffix}"
            )

    def close(self) -> None:
        self.tick(force=True)
        if self.tty:
            print()


class Concurrency:
    def __init__(self):
        self.limit = 10
        self.successes = 0
        self.failures = 0

    def observe(self, success: bool) -> None:
        if success:
            self.failures = 0
            self.successes += 1
            if self.successes == 5:
                self.limit = min(100, self.limit + 5)
                self.successes = 0
        else:
            self.successes = 0
            self.failures += 1
            if self.failures == 3:
                self.limit = max(5, self.limit - 5)
                self.failures = 0


def save_results(path: Path, results: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_nested(
    pending_ids: list[str],
    messages: dict[str, list[dict[str, str]]],
    output_path: Path,
    results: dict,
    config: dict,
    dataset: str,
    progress: Progress,
) -> None:
    pending = iter(pending_ids)
    completed: queue.Queue = queue.Queue()
    active = {}
    exhausted = False
    concurrency = Concurrency()
    last_report = time.monotonic()
    progress.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发：10")
    with ThreadPoolExecutor(max_workers=100) as executor:
        while active or not exhausted:
            while not exhausted and len(active) < concurrency.limit:
                index = next(pending, None)
                if index is None:
                    exhausted = True
                    break
                future = executor.submit(call_model, messages[index], config, dataset)
                active[future] = index
                future.add_done_callback(completed.put)
            if not active:
                break
            try:
                future = completed.get(timeout=1)
            except queue.Empty:
                pass
            else:
                index = active.pop(future)
                record = future.result()
                results[index] = record
                save_results(output_path, results)
                success = record["status"]["success"]
                concurrency.observe(success)
                progress.results += int(success)
                progress.log(index, "result", success, record["status"]["reason"])
            progress.tick()
            if time.monotonic() - last_report >= 30:
                last_report = time.monotonic()
                progress.message(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发："
                    f"{concurrency.limit}，运行中：{len(active)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("spider", "bird"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--llm", required=True, choices=MODEL_ALIASES)
    args = parser.parse_args()
    if args.dataset == "bird" and args.split != "dev":
        parser.error("Only bird/dev, spider/dev and spider/test are supported")

    dataset_split = f"{args.dataset}_{args.split}"
    scripts_root = ROOT / "code/scripts" / dataset_split
    gold_records = json.loads(
        (scripts_root / "gold_schema_linking.json").read_text(encoding="utf-8")
    )
    instances = json.loads(
        (scripts_root / "preprocessed_data" / f"{dataset_split}.json").read_text(
            encoding="utf-8"
        )
    )
    instance_by_index = {str(item["index"]): item for item in instances}
    gold_by_index = {str(item["index"]): item for item in gold_records}
    if set(instance_by_index) != set(gold_by_index):
        raise ValueError("Gold schema-linking indices do not match dataset instances")

    linking_path = (
        HERE / dataset_split / "schema_linking" / "qwen38_result.json"
    )
    linking = json.loads(linking_path.read_text(encoding="utf-8"))
    stage_dir = HERE / dataset_split / "difficulty_decomposition"
    prompt_dir = stage_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    output_path = stage_dir / f"{args.llm}_result.json"
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    results = {index: value for index, value in results.items() if index in gold_by_index}

    labels = {index: gold_label(record) for index, record in gold_by_index.items()}
    nested_ids = [index for index in instance_by_index if labels[index] == "NESTED"]
    for index, label in labels.items():
        if label != "NESTED":
            results[index] = {
                "result": {"label": label, "sub_questions": []},
                "raw_result": None,
                "status": {"success": True, "reason": ""},
                "resource": [None, 0.0],
            }
    save_results(output_path, results)

    successful = lambda index: results.get(index, {}).get("status", {}).get("success") is True
    progress = Progress(
        len(instance_by_index),
        len(nested_ids),
        0,
        sum(successful(index) for index in instance_by_index),
    )
    messages = {}
    try:
        if args.dataset == "bird":
            databases = ROOT / "BIRD/data/dev/dev_databases"
        else:
            databases = ROOT / "Spider/data" / (
                "database" if args.split == "dev" else "test_database"
            )
        meta_root = scripts_root / "preprocessed_data/meta"
        for index in nested_ids:
            instance = instance_by_index[index]
            try:
                linking_record = linking[index]
                if not linking_record.get("status", {}).get("success"):
                    raise ValueError("qwen38 schema-linking result is unavailable")
                schema_links = linking_record.get("result")
                if not isinstance(schema_links, list) or not all(
                    isinstance(item, str) for item in schema_links
                ):
                    raise ValueError("qwen38 schema-linking result is invalid")
                db_id = instance["db_id"]
                messages[index] = build_prompt(
                    args.dataset,
                    instance,
                    schema_links,
                    databases / db_id / f"{db_id}.sqlite",
                    meta_root / db_id,
                )
                text = "\n\n".join(
                    f"[{message['role'].upper()}]\n{message['content']}"
                    for message in messages[index]
                )
                (prompt_dir / f"{index}.txt").write_text(text, encoding="utf-8")
                progress.prompts += 1
                progress.log(index, "prompt", True)
            except Exception as exc:
                progress.log(index, "prompt", False, str(exc))
                if not successful(index):
                    results[index] = {
                        "result": None,
                        "raw_result": None,
                        "status": {
                            "success": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        },
                        "resource": [None, 0.0],
                    }
                    save_results(output_path, results)
            progress.tick()

        pending_ids = [
            index for index in nested_ids if index in messages and not successful(index)
        ]
        if pending_ids:
            run_nested(
                pending_ids,
                messages,
                output_path,
                results,
                model_config(args.llm),
                args.dataset,
                progress,
            )
        save_results(output_path, results)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
