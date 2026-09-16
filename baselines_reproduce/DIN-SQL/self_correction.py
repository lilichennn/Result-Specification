"""Run one DIN-SQL BIRD self-correction pass, optionally with Round-3 RC.

Both variants use sql_generation/qwen38_result.json as their initial SQL and the
same full database context. Outputs are written under self_correction/. Successful
outputs are resumed; move them aside before changing inputs or prompts.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from difficulty_decomposition import literal_assignment
from schema_linking import MODEL_ALIASES, database_context, model_config, stream_events
from sql_generation import (
    MAX_CONCURRENCY, TIMEOUT, Concurrency, Progress, extract_initial_sql,
    insert_before_answer, load_rc, prompt_text, result_contract_text, save_results,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def build_messages(instance: dict, initial_sql: str, context: str,
                   rc_record: dict | None) -> list[dict[str, str]]:
    official = ROOT / "code/baselines/DIN-SQL/DIN-SQL_BIRD.py"
    system = literal_assignment(official, "SYSTEM_SELF_CORRECTION_PROMPT")
    template = literal_assignment(official, "HUMAN_SELF_CORRECTION_PROMPT")
    human = template.format(
        schema=context, columns_descriptions="", question=instance["question"],
        hint=instance.get("evidence", ""), sql_query=initial_sql,
    )
    if rc_record is not None:
        block = result_contract_text(rc_record).replace(
            "Translate each field into SQL as follows:",
            "Check and correct the supplied SQL against each field as follows:",
        ).replace("supplied schema and schema links", "supplied schema")
        human = insert_before_answer(human, block)
    return [{"role": "system", "content": system}, {"role": "user", "content": human}]


def extract_corrected_sql(raw: str, initial_sql: str) -> tuple[str, str]:
    markers = list(re.finditer(r"\bRevised_SQL\s*:\s*", raw, flags=re.IGNORECASE))
    if markers:
        try:
            return extract_initial_sql(raw[markers[-1].end():]), ""
        except ValueError:
            pass
    # Official BIRD behavior: keep the initial query if Revised_SQL is unavailable.
    return initial_sql, "No valid Revised_SQL; kept initial SQL"


def call_model(messages: list[dict[str, str]], config: dict, initial_sql: str) -> dict:
    started = time.monotonic()
    usage = None
    try:
        payload = {
            "model": config["model"], "messages": messages, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
        }
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
        raw = "".join(content_parts)
        if not raw.strip():
            raise ValueError("Model returned empty content")
        result, reason = extract_corrected_sql(raw, initial_sql)
        status = {"success": True, "reason": reason}
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
    return {"result": result, "status": status,
            "resource": [usage, round(time.monotonic() - started, 3)]}


def run_results(ids: list[str], messages: dict, initial_sql: dict, output_path: Path,
                results: dict, config: dict, progress: Progress) -> None:
    pending = iter(ids)
    completed = queue.Queue()
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
                future = executor.submit(call_model, messages[index], config, initial_sql[index])
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
                progress.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发："
                                 f"{concurrency.limit}，运行中：{len(active)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("bird",))
    parser.add_argument("--split", required=True, choices=("dev",))
    parser.add_argument("--llm", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--rc", action="store_true")
    args = parser.parse_args()

    dataset_split = f"{args.dataset}_{args.split}"
    scripts_root = ROOT / "code/scripts" / dataset_split
    data_root = scripts_root / "preprocessed_data"
    instances = json.loads((data_root / f"{dataset_split}.json").read_text(encoding="utf-8"))
    ids = [str(instance["index"]) for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Instance index must be unique")
    # Fixed for both arms, regardless of --rc and the correction model alias.
    initial_records = json.loads(
        (HERE / dataset_split / "sql_generation/qwen38_result.json").read_text(encoding="utf-8")
    )
    rc_records = load_rc(scripts_root / "rc.json") if args.rc else {}
    stage_dir = HERE / dataset_split / "self_correction"
    prompt_dir = stage_dir / ("prompt_rc" if args.rc else "prompt")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rc" if args.rc else ""
    output_path = stage_dir / f"{args.llm}_result{suffix}.json"
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    valid_ids = set(ids)
    results = {index: record for index, record in results.items() if index in valid_ids}
    successful = lambda index: results.get(index, {}).get("status", {}).get("success") is True
    progress = Progress(len(ids), 0, sum(successful(index) for index in ids))
    messages, initial_sql, contexts = {}, {}, {}
    try:
        for instance in instances:
            index = str(instance["index"])
            try:
                initial = initial_records.get(index, {})
                sql = initial.get("result")
                if (initial.get("status", {}).get("success") is not True
                        or not isinstance(sql, str) or not sql.strip()):
                    raise ValueError("Baseline qwen38 initial SQL is unavailable")
                rc_record = None
                if args.rc:
                    rc_record = rc_records.get(index)
                    if rc_record is None:
                        raise ValueError("Round-3 Result Contract is unavailable")
                db_id = instance["db_id"]
                if db_id not in contexts:
                    contexts[db_id] = database_context(
                        ROOT / "BIRD/data/dev/dev_databases" / db_id / f"{db_id}.sqlite",
                        data_root / "meta" / db_id,
                    )
                item_messages = build_messages(instance, sql, contexts[db_id], rc_record)
                (prompt_dir / f"{index}.txt").write_text(prompt_text(item_messages), encoding="utf-8")
                messages[index] = item_messages
                initial_sql[index] = sql
                progress.prompts += 1
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                progress.message(f"{index}｜prompt failed｜{reason}")
                if not successful(index):
                    results[index] = {"result": None,
                                      "status": {"success": False, "reason": reason},
                                      "resource": [None, 0.0]}
                    save_results(output_path, results)
            progress.tick()
        pending_ids = [index for index in ids if index in messages and not successful(index)]
        if pending_ids:
            run_results(pending_ids, messages, initial_sql, output_path, results,
                        model_config(args.llm), progress)
        save_results(output_path, results)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
