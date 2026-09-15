"""Evaluate DIN-SQL SQL with a 180-second per-sample process deadline.

Completed comparisons with identical inputs are resumed; errors and timeouts are
retried. Use --restart after changing databases or the evaluation rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import sqlite3
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SAMPLE_TIMEOUT = 180
WORKERS = 5
REPORT_INTERVAL = 10
FIELDS = (
    "index", "q", "dbid", "goldsql", "predsql", "gold_executable", "correct",
    "pred_executable", "status", "error", "elapsed_seconds",
)


def database_root(dataset: str, split: str) -> Path:
    if dataset == "spider":
        return ROOT / "Spider/data" / (
            "database" if split == "dev" else "test_database"
        )
    if dataset == "bird" and split == "dev":
        return ROOT / "BIRD/data/dev/dev_databases"
    raise ValueError("Only bird/dev, spider/dev and spider/test are supported")


def execute_sql(database: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def row_set(row: tuple[Any, ...]) -> frozenset[Any]:
    return frozenset(bytes(value) if isinstance(value, memoryview) else value for value in row)


def result_contains_gold(
    gold_rows: list[tuple[Any, ...]], pred_rows: list[tuple[Any, ...]]
) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False

    gold_counts = Counter(row_set(row) for row in gold_rows)
    pred_counts = Counter(row_set(row) for row in pred_rows)

    # Matching equal row sets is always safe. Remove them before considering
    # containment so ordinary equal-result queries finish with hash lookups only.
    common = gold_counts & pred_counts
    gold_counts -= common
    pred_counts -= common
    if not gold_counts:
        return not pred_counts

    pred_types = list(pred_counts)
    postings: dict[Any, set[int]] = defaultdict(set)
    for pred_index, pred in enumerate(pred_types):
        for value in pred:
            postings[value].add(pred_index)

    candidates = []
    all_pred_indices = set(range(len(pred_types)))
    for gold in gold_counts:
        if not gold:
            matches = all_pred_indices.copy()
        else:
            value_postings = [postings.get(value, set()) for value in gold]
            if any(not indices for indices in value_postings):
                return False
            value_postings.sort(key=len)
            matches = value_postings[0].copy()
            for indices in value_postings[1:]:
                matches.intersection_update(indices)
                if not matches:
                    return False
        candidates.append(matches)

    # Maximum flow over unique row-set types. Capacities preserve duplicate rows
    # without expanding them into a large per-row bipartite graph.
    gold_types = list(gold_counts)
    gold_total = sum(gold_counts.values())
    source = 0
    gold_offset = 1
    pred_offset = gold_offset + len(gold_types)
    sink = pred_offset + len(pred_types)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int) -> None:
        forward = [end, len(graph[end]), capacity]
        backward = [start, len(graph[start]), 0]
        graph[start].append(forward)
        graph[end].append(backward)

    for gold_index, gold in enumerate(gold_types):
        count = gold_counts[gold]
        add_edge(source, gold_offset + gold_index, count)
        for pred_index in candidates[gold_index]:
            add_edge(gold_offset + gold_index, pred_offset + pred_index, count)
    for pred_index, pred in enumerate(pred_types):
        add_edge(pred_offset + pred_index, sink, pred_counts[pred])

    flow = 0
    while flow < gold_total:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for end, _reverse, capacity in graph[node]:
                if capacity > 0 and level[end] < 0:
                    level[end] = level[node] + 1
                    queue.append(end)
        if level[sink] < 0:
            return False

        next_edge = [0] * len(graph)

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while next_edge[node] < len(graph[node]):
                edge = graph[node][next_edge[node]]
                end, reverse, capacity = edge
                if capacity > 0 and level[end] == level[node] + 1:
                    sent = send(end, min(amount, capacity))
                    if sent:
                        edge[2] -= sent
                        graph[end][reverse][2] += sent
                        return sent
                next_edge[node] += 1
            return 0

        while pushed := send(source, gold_total - flow):
            flow += pushed

    return True


def evaluate_record(database: Path, row: dict, sender) -> None:
    """Keep SQL results and matching inside a disposable child process."""
    stage = "gold"
    try:
        gold_result = execute_sql(database, row["goldsql"])
        row["gold_executable"] = 1
        stage = "pred"
        sender.send((stage, row))
        pred_result = execute_sql(database, row["predsql"])
        row["pred_executable"] = 1
        stage = "match"
        sender.send((stage, row))
        row["correct"] = int(result_contains_gold(gold_result, pred_result))
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"{stage}_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            sender.send(("result", row))
        finally:
            sender.close()


def stop_worker(process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)
    if not process.is_alive():
        process.close()


def report_progress(completed: dict, total: int, active: dict, label: str) -> None:
    correct = sum(int(row["correct"]) for row in completed.values())
    statuses = Counter(row["status"] for row in completed.values())
    accuracy = f"{correct / len(completed):.2%}" if completed else "N/A"
    running = ", ".join(
        f"{index}:{job['stage']}({time.monotonic() - job['started']:.0f}s)"
        for index, job in active.items()
    ) or "none"
    print(
        f"{label}｜完成 {len(completed)}/{total}｜正确 {correct}｜"
        f"已完成样本正确率 {accuracy}（含超时/错误）｜"
        f"超时 {statuses['timeout']}｜错误 "
        f"{sum(count for status, count in statuses.items() if status.endswith('_error'))}｜"
        f"运行中 {running}",
        flush=True,
    )


def evaluate(dataset: str, split: str, use_rc: bool,
             timeout: float = SAMPLE_TIMEOUT, restart: bool = False,
             stage: str = "sql_generation") -> Path:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number of seconds")
    if stage not in ("sql_generation", "self_correction"):
        raise ValueError("Unsupported SQL evaluation stage")
    dataset_split = f"{dataset}_{split}"
    gold_path = ROOT / "code/scripts" / dataset_split / "gold_sql_schema_linking.json"
    stage_dir = HERE / dataset_split / stage
    suffix = "_rc" if use_rc else ""
    pred_path = stage_dir / f"qwen38_result{suffix}.json"
    baseline_path = stage_dir / "qwen38_result.json"
    rc_path = stage_dir / "qwen38_result_rc.json"
    output_path = stage_dir / f"evaluation{suffix}.csv"

    gold_records = json.loads(gold_path.read_text(encoding="utf-8"))
    pred_records = json.loads(pred_path.read_text(encoding="utf-8"))
    baseline_records = json.loads(baseline_path.read_text(encoding="utf-8"))
    rc_records = json.loads(rc_path.read_text(encoding="utf-8"))
    databases = database_root(dataset, split)

    def has_sql(records: dict, index: str) -> bool:
        record = records.get(index, {})
        return (
            record.get("status", {}).get("success") is True
            and isinstance(record.get("result"), str)
            and bool(record["result"].strip())
        )

    paired_ids = {
        str(record["index"])
        for record in gold_records
        if has_sql(baseline_records, str(record["index"]))
        and has_sql(rc_records, str(record["index"]))
    }
    print(f"In total {len(paired_ids)} overlap instance", flush=True)

    rows = {
        str(record["index"]): {
            "index": str(record["index"]), "q": record["question"],
            "dbid": record["db_id"], "goldsql": record["gold_sql"],
            "predsql": pred_records[str(record["index"])]["result"],
            "gold_executable": 0, "pred_executable": 0, "correct": 0,
            "status": "", "error": "", "elapsed_seconds": "",
        }
        for record in gold_records if str(record["index"]) in paired_ids
    }
    completed = {}
    if output_path.exists() and not restart:
        with output_path.open(encoding="utf-8", newline="") as previous:
            for saved in csv.DictReader(previous):
                index = saved.get("index")
                current = rows.get(index)
                # Only reuse fully evaluated, unchanged inputs. Retry errors/timeouts.
                if (current is not None and saved.get("status") == "ok"
                        and saved.get("correct") in ("0", "1")
                        and saved.get("gold_executable") == "1"
                        and saved.get("pred_executable") == "1"
                        and all(saved.get(key) == current[key]
                                for key in ("q", "dbid", "goldsql", "predsql"))
                        and all(saved.get(key) is not None for key in FIELDS)):
                    completed[index] = {key: saved[key] for key in FIELDS}

    print(f"Output: {output_path}\n单样本超时 {timeout:g}s｜并发 {WORKERS}｜"
          f"复用 {len(completed)}｜待评估 {len(rows) - len(completed)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(completed.values())
    temporary.replace(output_path)

    context = multiprocessing.get_context("spawn")
    pending = deque(index for index in rows if index not in completed)
    active = {}
    last_report = time.monotonic()
    report_progress(completed, len(rows), active, "开始")
    with output_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        try:
            while pending or active:
                while pending and len(active) < WORKERS:
                    index = pending.popleft()
                    row = rows[index].copy()
                    receiver, sender = context.Pipe(duplex=False)
                    database = databases / row["dbid"] / f"{row['dbid']}.sqlite"
                    process = context.Process(target=evaluate_record,
                                              args=(database, row, sender), daemon=True)
                    started = time.monotonic()
                    try:
                        process.start()
                    except BaseException:
                        receiver.close()
                        sender.close()
                        raise
                    sender.close()
                    active[index] = {"process": process, "receiver": receiver,
                                     "started": started, "stage": "gold", "row": row}

                for index, job in list(active.items()):
                    process, receiver = job["process"], job["receiver"]
                    finished = False
                    eof = False
                    try:
                        while receiver.poll():
                            stage, row = receiver.recv()
                            job["row"] = row
                            if stage == "result":
                                finished = True
                                break
                            job["stage"] = stage
                    except EOFError:
                        eof = True
                    elapsed = time.monotonic() - job["started"]
                    row = job["row"]
                    if not finished and elapsed >= timeout:
                        row.update(status="timeout", correct=0,
                                   error=f"Sample exceeded {timeout:g}s during {job['stage']}")
                        finished = True
                    elif not finished and not process.is_alive():
                        if not eof and receiver.poll():
                            continue  # A final message arrived between poll and exit.
                        row.update(status="worker_error", correct=0,
                                   error=f"Worker exited with code {process.exitcode} during {job['stage']}")
                        finished = True
                    if not finished:
                        continue
                    row["elapsed_seconds"] = round(elapsed, 3)
                    stop_worker(process)
                    receiver.close()
                    del active[index]
                    writer.writerow(row)
                    output.flush()
                    completed[index] = row
                    print(f"{index}｜{row['status']}｜correct={row['correct']}｜"
                          f"{elapsed:.1f}s｜{len(completed)}/{len(rows)}"
                          + ("｜" + " ".join(row["error"].splitlines()) if row["error"] else ""),
                          flush=True)

                if time.monotonic() - last_report >= REPORT_INTERVAL:
                    report_progress(completed, len(rows), active, "进度")
                    last_report = time.monotonic()
                if active:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("评估已中断，已完成结果已保存；重跑同一命令可继续。", flush=True)
            raise
        finally:
            for job in active.values():
                stop_worker(job["process"])
                job["receiver"].close()
            report_progress(completed, len(rows), {},
                            "最终" if len(completed) == len(rows) else "未完成")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("spider", "bird"))
    parser.add_argument("--split", "--spilt", dest="split", required=True, choices=("dev", "test"))
    parser.add_argument("--rc", action="store_true")
    parser.add_argument("--stage", choices=("sql_generation", "self_correction"),
                        default="sql_generation", help="SQL result directory to evaluate")
    parser.add_argument("--timeout", type=float, default=SAMPLE_TIMEOUT,
                        help="Seconds per sample including both SQL queries and matching (default: 180)")
    parser.add_argument("--restart", action="store_true",
                        help="Re-evaluate all paired samples, ignoring the existing CSV")
    args = parser.parse_args()
    if args.dataset == "bird" and args.split != "dev":
        parser.error("Only bird/dev, spider/dev and spider/test are supported")

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    output_path = evaluate(args.dataset, args.split, args.rc, args.timeout, args.restart, args.stage)
    print(f"Wrote evaluation to {output_path}")


if __name__ == "__main__":
    main()
