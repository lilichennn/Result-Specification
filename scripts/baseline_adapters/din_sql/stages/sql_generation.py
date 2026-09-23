"""Generate DIN-SQL initial SQL with optional Result Contract context."""

from __future__ import annotations

import argparse
import json
import queue
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from difficulty_decomposition import literal_assignment, spider_schema
from schema_linking import MODEL_ALIASES, database_context, model_config, stream_events


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TIMEOUT = 900
INITIAL_CONCURRENCY = 50
MAX_CONCURRENCY = 1500
LABELS = ("EASY", "NON-NESTED", "NESTED")
RC_INSTRUCTION = """
Use the Result Contract as a semantic specification of the table that the SQL must return. Translate each field into SQL as follows:

- population: Define exactly which entities or records qualify. Implement every stated eligibility condition through the necessary joins, predicates, subqueries, or HAVING conditions.
- row_grain: Ensure that each returned row represents the stated semantic unit. Use grouping, aggregation, DISTINCT, or other deduplication only when needed to enforce this grain.
- column_role: Return exactly the requested semantic outputs in the SELECT clause. Do not expose columns used only for joining, filtering, grouping, or ordering.
- derivation: Implement the stated calculation, aggregation, comparison, or set operation. "none" means that the requested outputs are projected directly.
- filter_policy: Apply the stated relative selection, ordering, or output limit after defining the eligible population. "none" means no additional ranking or cardinality restriction; it does not remove the eligibility conditions in population.
- meta_review: Use the accepted metadata clarification to resolve column meanings, entity ownership, physical representations, and relationship paths.

Ground every table, column, and relationship in the supplied schema and schema links. Before returning the final SQL, verify that the qualifying population, output columns, and result row grain all satisfy the Result Contract.
"""



def schema_links_text(schema_links: list[str]) -> str:
    return "[" + ",".join(schema_links) + "]"


def result_contract_text(record: dict) -> str:
    contract = record.get("rc_round3")
    if not isinstance(contract, dict):
        raise ValueError("rc_round3 must be an object")
    return (
        "\nResult Contract:\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
        + "\n"
        + RC_INSTRUCTION
        + "\n"
    )


def spider_tables(db_id: str, include_foreign_keys: bool) -> str:
    context = spider_schema(db_id)
    if include_foreign_keys:
        return context + "\n"
    return "\n".join(
        line for line in context.splitlines() if not line.startswith("Foreign_keys = ")
    ) + "\n"


def build_spider_messages(
    instance: dict,
    label: str,
    schema_links: list[str],
    sub_questions: list[str],
    rc_record: dict | None,
) -> list[dict[str, str]]:
    official = ROOT / "code/baselines/DIN-SQL/DIN-SQL.py"
    prompt_names = {
        "EASY": "easy_prompt",
        "NON-NESTED": "medium_prompt",
        "NESTED": "hard_prompt",
    }
    examples = literal_assignment(official, prompt_names[label])
    include_foreign_keys = label != "EASY"
    fields = spider_tables("college_2", include_foreign_keys)
    fields += spider_tables(instance["db_id"], include_foreign_keys)
    links = schema_links_text(schema_links)
    question = instance["question"]
    hint = f"\nHint: {instance['evidence']}" if instance.get("evidence") else ""
    rc = result_contract_text(rc_record) if rc_record is not None else ""

    if label == "EASY":
        prompt = (
            "# Use the the schema links to generate the SQL queries for each of "
            "the questions.\n"
            + fields
            + "\n"
            + examples
            + 'Q: "'
            + question
            + hint
            + "\nSchema_links: "
            + links
            + rc
            + "\nSQL:"
        )
    elif label == "NON-NESTED":
        prompt = (
            "# Use the the schema links and Intermediate_representation to generate "
            "the SQL queries for each of the questions.\n"
            + fields
            + "\n"
            + examples
            + 'Q: "'
            + question
            + hint
            + "\nSchema_links: "
            + links
            + rc
            + "\nA: Let’s think step by step."
        )
    else:
        if not sub_questions:
            raise ValueError("NESTED instance has no sub_questions")
        prompt = (
            "# Use the intermediate representation and the schema links to generate "
            "the SQL queries for each of the questions.\n"
            + fields
            + "\n"
            + examples
            + 'Q: "'
            + question
            + '"'
            + hint
            + "\nschema_links: "
            + links
            + rc
            + '\nA: Let\'s think step by step. "'
            + question
            + '" can be solved by knowing the answer to the following sub-question "'
            + " ; ".join(sub_questions)
            + '".\nThe SQL query for the sub-question"'
        )
    return [{"role": "user", "content": prompt}]


def insert_before_answer(human: str, block: str) -> str:
    positions = [position for marker in ("\nA:", "\nSQL:") if (position := human.rfind(marker)) >= 0]
    if not positions:
        raise ValueError("BIRD generation answer marker not found")
    position = max(positions)
    return human[:position] + block + human[position:]


def build_bird_messages(
    instance: dict,
    label: str,
    schema_links: list[str],
    sub_questions: list[str],
    context: str,
    rc_record: dict | None,
) -> list[dict[str, str]]:
    official = ROOT / "code/baselines/DIN-SQL/DIN-SQL_BIRD.py"
    names = {
        "EASY": ("SYSTEM_EASY_CLASS_TEMPLATE", "HUMAN_EASY_CLASS_TEMPLATE"),
        "NON-NESTED": (
            "SYSTEM_NON_NESTED_CLASS_TEMPLATE",
            "HUMAN_NON_NESTED_CLASS_TEMPLATE",
        ),
        "NESTED": ("SYSTEM_NESTED_CLASS_TEMPLATE", "HUMAN_NESTED_CLASS_TEMPLATE"),
    }
    system_name, human_name = names[label]
    system = literal_assignment(official, system_name)
    human_template = literal_assignment(official, human_name)
    values = {
        "schema": context,
        "columns_descriptions": "",
        "question": instance["question"],
        "hint": instance.get("evidence", ""),
        "schema_links": schema_links_text(schema_links),
        "sub_questions": json.dumps(sub_questions, ensure_ascii=False),
    }
    human = human_template.format(**values)
    if rc_record is not None:
        human = insert_before_answer(human, result_contract_text(rc_record))
    return [{"role": "system", "content": system}, {"role": "user", "content": human}]


def prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message['role'].upper()}]\n{message['content']}" for message in messages
    )


def extract_initial_sql(raw: str) -> str:
    text = raw.strip()
    sql_markers = list(re.finditer(r"\bSQL\s*:\s*", text, flags=re.IGNORECASE))
    if sql_markers:
        text = text[sql_markers[-1].end() :].strip()
    else:
        fences = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fences:
            text = fences[-1].strip()
        else:
            start = re.search(r"\b(?:SELECT|WITH)\b", text, flags=re.IGNORECASE)
            if not start:
                raise ValueError("Model output does not contain SQL")
            text = text[start.start() :].strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if not re.match(r"^(?:SELECT|WITH)\b", text, flags=re.IGNORECASE):
        raise ValueError("Parsed initial SQL does not start with SELECT or WITH")
    return text


def call_model(messages: list[dict[str, str]], config: dict, dataset: str) -> dict:
    started = time.monotonic()
    usage = None
    try:
        payload = {
            "model": config["model"],
            "messages": messages,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if dataset == "spider":
            payload.update(
                n=1,
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
                "Accept": "text/event-stream",
            },
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
        raw = "".join(content_parts)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Model returned empty content")
        result = extract_initial_sql(raw)
        status = {"success": True, "reason": ""}
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
        result = None
        status = {"success": False, "reason": reason}
    return {
        "result": result,
        "status": status,
        "resource": [usage, round(time.monotonic() - started, 3)],
    }


class Progress:
    def __init__(self, total: int, prompts: int, results: int):
        self.total = total
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
                f"共 {self.total} instance，已完成 {self.prompts} prompt，"
                f"{self.results} result"
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
        self.limit = INITIAL_CONCURRENCY
        self.last_increase = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        while self.limit < MAX_CONCURRENCY and now - self.last_increase >= 1:
            self.limit = min(MAX_CONCURRENCY, self.limit + random.randint(40, 60))
            self.last_increase += 1


def save_results(path: Path, results: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_results(
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
    progress.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发：{concurrency.limit}")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        while active or not exhausted:
            concurrency.tick()
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
                wait = (max(0.001, concurrency.last_increase + 1 - time.monotonic())
                        if concurrency.limit < MAX_CONCURRENCY else 1)
                future = completed.get(timeout=wait)
            except queue.Empty:
                pass
            else:
                index = active.pop(future)
                record = future.result()
                results[index] = record
                save_results(output_path, results)
                success = record["status"]["success"]
                progress.results += int(success)
                progress.log(index, "result", success, record["status"]["reason"])
            progress.tick()
            if time.monotonic() - last_report >= 30:
                last_report = time.monotonic()
                progress.message(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发："
                    f"{concurrency.limit}，运行中：{len(active)}"
                )


def load_rc(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["index"]): record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("spider", "bird"))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--llm", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--rc", action="store_true")
    args = parser.parse_args()
    if args.dataset == "bird" and args.split != "dev":
        parser.error("Only bird/dev, spider/dev and spider/test are supported")

    dataset_split = f"{args.dataset}_{args.split}"
    scripts_root = ROOT / "code/scripts" / dataset_split
    data_root = scripts_root / "preprocessed_data"
    instances = json.loads(
        (data_root / f"{dataset_split}.json").read_text(encoding="utf-8")
    )
    instance_by_index = {str(instance["index"]): instance for instance in instances}
    ids = [str(instance["index"]) for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Instance index must be unique")

    linking_path = HERE / dataset_split / "schema_linking/qwen38_result.json"
    difficulty_path = (
        HERE / dataset_split / "difficulty_decomposition/qwen38_result.json"
    )
    linking = json.loads(linking_path.read_text(encoding="utf-8"))
    difficulty = json.loads(difficulty_path.read_text(encoding="utf-8"))
    rc_by_index = load_rc(scripts_root / "rc.json") if args.rc else {}

    stage_dir = HERE / dataset_split / "sql_generation"
    prompt_dir = stage_dir / ("prompt_rc" if args.rc else "prompt")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rc" if args.rc else ""
    output_path = stage_dir / f"{args.llm}_result{suffix}.json"
    results = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )
    results = {index: record for index, record in results.items() if index in instance_by_index}
    successful = lambda index: (
        results.get(index, {}).get("status", {}).get("success") is True
    )
    progress = Progress(len(ids), 0, sum(successful(index) for index in ids))

    messages = {}
    bird_context = {}
    try:
        if args.dataset == "bird":
            databases = ROOT / "BIRD/data/dev/dev_databases"
        for index in ids:
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

                difficulty_record = difficulty[index]
                if not difficulty_record.get("status", {}).get("success"):
                    raise ValueError("qwen38 difficulty/decomposition result is unavailable")
                parsed_difficulty = difficulty_record.get("result")
                if not isinstance(parsed_difficulty, dict):
                    raise ValueError("qwen38 difficulty/decomposition result is invalid")
                label = parsed_difficulty.get("label")
                if label not in LABELS:
                    raise ValueError(f"Invalid difficulty label: {label!r}")
                sub_questions = parsed_difficulty.get("sub_questions", [])
                if not isinstance(sub_questions, list) or not all(
                    isinstance(item, str) and item.strip() for item in sub_questions
                ):
                    raise ValueError("Invalid sub_questions")

                rc_record = None
                if args.rc:
                    rc_record = rc_by_index.get(index)
                    if rc_record is None:
                        raise ValueError("Result Contract is unavailable")

                if args.dataset == "spider":
                    item_messages = build_spider_messages(
                        instance,
                        label,
                        schema_links,
                        sub_questions,
                        rc_record,
                    )
                else:
                    db_id = instance["db_id"]
                    context = bird_context.get(db_id)
                    if context is None:
                        context = database_context(
                            databases / db_id / f"{db_id}.sqlite",
                            data_root / "meta" / db_id,
                        )
                        bird_context[db_id] = context
                    item_messages = build_bird_messages(
                        instance,
                        label,
                        schema_links,
                        sub_questions,
                        context,
                        rc_record,
                    )
                messages[index] = item_messages
                (prompt_dir / f"{index}.txt").write_text(
                    prompt_text(item_messages), encoding="utf-8"
                )
                progress.prompts += 1
                progress.log(index, "prompt", True)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                progress.log(index, "prompt", False, reason)
                if not successful(index):
                    results[index] = {
                        "result": None,
                        "status": {"success": False, "reason": reason},
                        "resource": [None, 0.0],
                    }
                    save_results(output_path, results)
            progress.tick()

        pending_ids = [
            index for index in ids if index in messages and not successful(index)
        ]
        if pending_ids:
            run_results(
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
