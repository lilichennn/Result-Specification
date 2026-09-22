from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
DEFAULT_CONCURRENCY = 50
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 2000
CONCURRENCY_GROWTH_MIN = 40
CONCURRENCY_GROWTH_MAX = 60
CONCURRENCY_GROWTH_INTERVAL_SECONDS = 1
MAX_ATTEMPTS = 3
CONCURRENCY_REPORT_INTERVAL_SECONDS = 10
COMPLETION_REPORT_INTERVAL_SECONDS = 5
LOGGER = logging.getLogger("generate_rc")
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
InstanceIndex = int | str

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from result_contract.rc import (
    MODEL_ALIASES,
    ModelCall,
    Round1RC,
    Round2RC,
    Round3RC,
    call_model,
    generate_round1,
    generate_round2,
    generate_round3,
)


class RampConcurrency:
    def __init__(self, initial: int) -> None:
        if not MIN_CONCURRENCY <= initial <= MAX_CONCURRENCY:
            raise ValueError(
                f"concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY}"
            )
        self._limit = initial
        self._active = 0
        self._condition = threading.Condition()

    @property
    def limit(self) -> int:
        with self._condition:
            return self._limit

    def acquire(self) -> None:
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1

    def release(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def grow(self) -> None:
        with self._condition:
            self._limit = min(
                MAX_CONCURRENCY,
                self._limit
                + random.randint(CONCURRENCY_GROWTH_MIN, CONCURRENCY_GROWTH_MAX),
            )
            self._condition.notify_all()


class ConsoleColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        text = text.replace(" | success", f" | {GREEN}success{RESET}")
        return text.replace(" | fail", f" | {RED}fail{RESET}")


class LiveStatusConsoleHandler(logging.StreamHandler):
    def __init__(self) -> None:
        super().__init__()
        self.terminal_rows = 0

    def update_status(self, text: str) -> None:
        self.acquire()
        try:
            rows = max(2, shutil.get_terminal_size(fallback=(80, 24)).lines)
            if rows != self.terminal_rows:
                self.stream.write(
                    f"\033[r\033[1;{rows - 1}r"
                    f"\033[{rows};1H\033[2K{text}"
                    f"\033[{rows - 1};1H"
                )
                self.terminal_rows = rows
            else:
                self.stream.write(
                    f"\0337\033[{rows};1H\033[2K{text}\0338"
                )
            self.flush()
        finally:
            self.release()

    def finish_status(self, text: str) -> None:
        self.acquire()
        try:
            rows = self.terminal_rows or max(
                2, shutil.get_terminal_size(fallback=(80, 24)).lines
            )
            self.stream.write(
                f"\033[r\033[{rows};1H\033[2K{text}{self.terminator}"
            )
            self.terminal_rows = 0
            self.flush()
        finally:
            self.release()


class CompletionReporter:
    def __init__(
        self,
        instances: Sequence[Mapping[str, Any]],
        records_by_index: Mapping[InstanceIndex, Mapping[str, Any]],
    ) -> None:
        self.instances = instances
        self.records_by_index = records_by_index
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.console_handler = next(
            (
                handler
                for handler in logging.getLogger().handlers
                if isinstance(handler, LiveStatusConsoleHandler)
                and handler.stream.isatty()
            ),
            None,
        )

    def start(self) -> None:
        if self.console_handler is not None:
            self.console_handler.update_status(self._line())
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(COMPLETION_REPORT_INTERVAL_SECONDS):
            if self.console_handler is not None:
                self.console_handler.update_status(self._line())
            else:
                _log_completion_counts(self.instances, self.records_by_index)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join()
        if self.console_handler is not None:
            self.console_handler.finish_status(self._line())

    def _line(self) -> str:
        return (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | INFO | "
            f"{_completion_counts_message(self.instances, self.records_by_index)}"
        )


def _start_concurrency_control(
    concurrency: RampConcurrency,
) -> tuple[threading.Event, tuple[threading.Thread, threading.Thread]]:
    stop = threading.Event()

    def grow() -> None:
        while not stop.wait(CONCURRENCY_GROWTH_INTERVAL_SECONDS):
            concurrency.grow()

    def report() -> None:
        while not stop.wait(CONCURRENCY_REPORT_INTERVAL_SECONDS):
            LOGGER.info("current concurrency=%d", concurrency.limit)

    growth_thread = threading.Thread(target=grow, daemon=True)
    report_thread = threading.Thread(target=report, daemon=True)
    growth_thread.start()
    report_thread.start()
    return stop, (growth_thread, report_thread)


def generate_rc_file(
    input_path: str | Path,
    meta_root: str | Path,
    output_path: str | Path,
    model_call: ModelCall | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Generate Round 1 and Round 2 with one shared concurrency limit."""
    if not MIN_CONCURRENCY <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY}"
        )

    input_path = Path(input_path).resolve()
    meta_root = Path(meta_root).resolve()
    output_path = Path(output_path).resolve()
    instances = _load_instances(input_path)

    db_ids = {instance["db_id"] for instance in instances}
    metadata_by_db = _load_metadata(meta_root, db_ids)

    existing_records = _load_existing_records(output_path)
    input_indices = {instance["index"] for instance in instances}
    records_by_index = {
        index: record
        for index, record in existing_records.items()
        if index in input_indices
    }

    recovery_instances: list[dict[str, Any]] = []
    pending_round2_instances: list[dict[str, Any]] = []
    new_instances: list[dict[str, Any]] = []

    for instance in instances:
        existing = records_by_index.get(instance["index"])
        if existing is None:
            new_instances.append(instance)
            continue
        if (
            not _has_reusable_round1(instance, existing)
            or existing.get("round2_status") == "failed"
        ):
            recovery_instances.append(instance)
            continue
        if _has_reusable_round2(instance, existing):
            continue
        else:
            pending_round2_instances.append(instance)

    _log_completion_counts(instances, records_by_index)
    completion_reporter = CompletionReporter(instances, records_by_index)
    completion_reporter.start()

    adaptive_concurrency = RampConcurrency(concurrency)
    stop_reporting, control_threads = _start_concurrency_control(
        adaptive_concurrency
    )
    try:
        unfinished_instances = (
            recovery_instances + pending_round2_instances + new_instances
        )
        _run_phase(
            phase_instances=unfinished_instances,
            all_instances=instances,
            records_by_index=records_by_index,
            metadata_by_db=metadata_by_db,
            output_path=output_path,
            model_call=model_call,
            adaptive_concurrency=adaptive_concurrency,
        )

        output = _ordered_records(instances, records_by_index)
        _write_json_atomic(output_path, output)
    finally:
        stop_reporting.set()
        for thread in control_threads:
            thread.join()
        completion_reporter.close()
    return output


def generate_round3_file(
    input_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path,
    model_call: ModelCall | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    *,
    allow_partial_gold: bool = False,
) -> list[dict[str, Any]]:
    """Correct selected contracts, preserving every existing output record.

    By default gold must cover the entire preprocessed input. Explicit partial
    mode selects the gold file's IDs; unknown IDs and mismatches still fail.
    """
    if not MIN_CONCURRENCY <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY}"
        )

    input_path = Path(input_path).resolve()
    gold_path = Path(gold_path).resolve()
    output_path = Path(output_path).resolve()
    instances = _load_instances(input_path)
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(gold_rows, list):
        raise ValueError(f"Gold SQL file must be a JSON array: {gold_path}")
    gold_by_index: dict[InstanceIndex, Mapping[str, Any]] = {}
    for row in gold_rows:
        if not isinstance(row, Mapping) or not _is_valid_instance_index(
            row.get("index")
        ):
            raise ValueError(f"Invalid record in Gold SQL file: {gold_path}")
        if row["index"] in gold_by_index:
            raise ValueError(f"Duplicate Gold SQL index: {row['index']}")
        gold_by_index[row["index"]] = row

    input_indices = {instance["index"] for instance in instances}
    if not set(gold_by_index).issubset(input_indices):
        raise ValueError("Gold SQL file contains IDs outside preprocessed data")
    if not allow_partial_gold and set(gold_by_index) != input_indices:
        raise ValueError("Preprocessed data and Gold SQL file have different ID sets")

    round3_instances: list[dict[str, Any]] = []
    for instance in instances:
        if instance["index"] not in gold_by_index:
            continue
        gold = gold_by_index[instance["index"]]
        if (
            gold.get("db_id") != instance["db_id"]
            or gold.get("question") != instance["question"]
        ):
            raise ValueError(
                f"Gold SQL record does not match instance {instance['index']!r}"
            )
        if not isinstance(gold.get("gold_sql"), str) or not gold["gold_sql"].strip():
            raise ValueError(f"Instance {instance['index']!r} has no non-empty gold_sql")
        round3_instances.append({**instance, "gold_sql": gold["gold_sql"]})
    instances = round3_instances

    # Never reconstruct output from a subset of inputs. Keep original order,
    # extra fields and even records outside this invocation's question list.
    records_by_index = _load_existing_records(output_path, infer_round1_status=False)
    output_order = list(records_by_index.values())
    validation_records = {
        index: _normalize_round1_status(record)
        for index, record in records_by_index.items()
        if index in gold_by_index
    }
    unavailable = [
        instance["index"]
        for instance in instances
        if not _has_reusable_round2(
            instance,
            validation_records.get(instance["index"], {}),
        )
    ]
    if unavailable:
        raise ValueError(
            "Round 3 requires reusable Round-2 contracts; unavailable indexes: "
            f"{unavailable}"
        )

    failed_instances = [
        instance
        for instance in instances
        if records_by_index[instance["index"]].get("round3_status") == "failed"
    ]
    new_instances = [
        instance
        for instance in instances
        if records_by_index[instance["index"]].get("round3_status") != "failed"
        and not _has_reusable_round3(instance, validation_records[instance["index"]])
    ]
    pending_instances = failed_instances + new_instances
    LOGGER.info(
        "Round3 selected=%d, pending=%d, reusable=%d, unselected_records=%d",
        len(instances), len(pending_instances), len(instances) - len(pending_instances),
        len(records_by_index) - len(instances),
    )
    if not pending_instances:
        return _ordered_records(output_order, records_by_index)
    _log_completion_counts(instances, records_by_index)
    completion_reporter = CompletionReporter(instances, records_by_index)
    completion_reporter.start()

    adaptive_concurrency = RampConcurrency(concurrency)
    stop_reporting, control_threads = _start_concurrency_control(
        adaptive_concurrency
    )
    try:
        _run_round3_phase(
            phase_instances=pending_instances,
            all_instances=output_order,
            records_by_index=records_by_index,
            output_path=output_path,
            model_call=model_call,
            adaptive_concurrency=adaptive_concurrency,
        )
        output = _ordered_records(output_order, records_by_index)
    finally:
        stop_reporting.set()
        for thread in control_threads:
            thread.join()
        completion_reporter.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or gold-correct Result Contracts for a dataset split."
    )
    parser.add_argument(
        "--dataset_split",
        required=True,
        help="Dataset split directory name, for example bird_dev or spider_test.",
    )
    parser.add_argument("--llm", required=True, choices=MODEL_ALIASES)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Initial API concurrency (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--round3",
        action="store_true",
        help="Correct existing Round-2 contracts using gold_sql_schema_linking.json.",
    )
    parser.add_argument(
        "--gold-file", type=Path,
        help="Round3 reference SQL JSON; defaults to the dataset's gold_sql_schema_linking.json.",
    )
    parser.add_argument(
        "--allow-partial-gold", action="store_true",
        help="Round3 only: generate for IDs present in the gold file and preserve all other RC records.",
    )
    args = parser.parse_args()
    if not args.round3 and (args.gold_file is not None or args.allow_partial_gold):
        parser.error("--gold-file and --allow-partial-gold require --round3")
    return args


def main() -> None:
    args = parse_args()
    input_path, meta_root, output_path, log_path = _dataset_split_paths(
        args.dataset_split
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Preprocessed dataset split not found: {input_path}")
    if not args.round3 and not meta_root.is_dir():
        raise FileNotFoundError(f"Preprocessed metadata directory not found: {meta_root}")

    configure_logging(log_path)
    model_call = lambda messages: call_model(messages, llm=args.llm)
    if args.round3:
        rows = generate_round3_file(
            input_path=input_path,
            gold_path=(args.gold_file if args.gold_file is not None
                       else output_path.parent / "gold_sql_schema_linking.json"),
            output_path=output_path,
            model_call=model_call,
            concurrency=args.concurrency,
            allow_partial_gold=args.allow_partial_gold,
        )
        records_by_index = {row["index"]: row for row in rows}
        counts = _round_status_counts(rows, records_by_index, round_number=3)
        print(
            f"Stored {len(rows)} records: total_round3_success={counts['success']}, "
            f"total_round3_failed={counts['fail']}, output={output_path.resolve()}"
        )
    else:
        rows = generate_rc_file(
            input_path=input_path,
            meta_root=meta_root,
            output_path=output_path,
            model_call=model_call,
            concurrency=args.concurrency,
        )
        counts = _status_counts(rows)
        print(
            f"Finished {len(rows)} instances: complete={counts['complete']}, "
            f"round1_failed={counts['round1_failed']}, "
            f"round2_failed={counts['round2_failed']}, "
            f"output={output_path.resolve()}"
        )


def _dataset_split_paths(dataset_split: str) -> tuple[Path, Path, Path, Path]:
    if (
        not isinstance(dataset_split, str)
        or not dataset_split
        or dataset_split in (".", "..")
        or Path(dataset_split).name != dataset_split
        or "/" in dataset_split
        or "\\" in dataset_split
    ):
        raise ValueError(
            "dataset_split must be a directory name such as bird_dev or spider_test"
        )

    dataset_split_root = CODE_ROOT / "data" / dataset_split
    input_path = dataset_split_root / f"{dataset_split}.json"
    meta_root = dataset_split_root / "meta"
    output_path = dataset_split_root / "rc.json"
    log_path = CODE_ROOT / "outputs" / "rc_generation" / dataset_split / "generate_rc.log"
    return input_path, meta_root, output_path, log_path


def configure_logging(log_path: str | Path) -> None:
    resolved_log_path = Path(log_path).resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = LiveStatusConsoleHandler()
    console_handler.setFormatter(
        ConsoleColorFormatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def _load_instances(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessed dataset split not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Preprocessed dataset split must be a JSON array")

    required_fields = {"index", "db_id", "question", "evidence"}
    seen_indices: set[InstanceIndex] = set()
    instances: list[dict[str, Any]] = []
    for position, instance in enumerate(value):
        if not isinstance(instance, dict):
            raise ValueError(f"Instance at position {position} must be an object")
        missing = required_fields - instance.keys()
        if missing:
            raise ValueError(
                f"Instance at position {position} is missing fields: {sorted(missing)}"
            )
        if not _is_valid_instance_index(instance["index"]):
            raise ValueError(
                f"Instance index at position {position} must be an integer or "
                "non-empty string"
            )
        if instance["index"] in seen_indices:
            raise ValueError(f"Duplicate instance index: {instance['index']}")
        if not isinstance(instance["db_id"], str) or not instance["db_id"].strip():
            raise ValueError(f"Instance db_id at position {position} must be non-empty")
        if not isinstance(instance["question"], str) or not instance["question"].strip():
            raise ValueError(f"Instance question at position {position} must be non-empty")
        if not isinstance(instance["evidence"], str):
            raise ValueError(f"Instance evidence at position {position} must be text")
        seen_indices.add(instance["index"])
        instances.append(instance)
    return instances


def _load_metadata(
    meta_root: Path,
    db_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not meta_root.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {meta_root}")

    metadata_by_db: dict[str, list[dict[str, Any]]] = {}
    for db_id in sorted(db_ids):
        db_meta_dir = meta_root / db_id
        if not db_meta_dir.is_dir():
            raise FileNotFoundError(
                f"Metadata directory for database {db_id!r} not found: {db_meta_dir}"
            )
        csv_paths = sorted(db_meta_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No metadata CSV files found for database {db_id!r}: {db_meta_dir}"
            )
        metadata_by_db[db_id] = [
            {
                "table_name": csv_path.stem,
                "columns": _read_metadata_csv(csv_path),
            }
            for csv_path in csv_paths
        ]
    return metadata_by_db


def _read_metadata_csv(path: Path) -> list[dict[str, Any]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"Metadata CSV has no header: {path}")
                return [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _normalize_round1_status(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "round1_status" not in normalized:
        normalized["round1_status"] = (
            "succeeded" if normalized.get("rc_round1") is not None else "failed"
        )
        normalized.setdefault("round1_error", None)
    return normalized


def _load_existing_records(
    path: Path, *, infer_round1_status: bool = True,
) -> dict[InstanceIndex, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Existing RC output must be a JSON array: {path}")

    records: dict[InstanceIndex, dict[str, Any]] = {}
    for position, record in enumerate(value):
        if not isinstance(record, dict) or not _is_valid_instance_index(
            record.get("index")
        ):
            raise ValueError(f"Invalid existing RC record at position {position}: {path}")
        index = record["index"]
        if index in records:
            raise ValueError(f"Duplicate index in existing RC output: {index}")
        records[index] = _normalize_round1_status(record) if infer_round1_status else dict(record)
    return records


def _has_valid_round1(record: Mapping[str, Any]) -> bool:
    if record.get("round1_status") != "succeeded":
        return False
    try:
        Round1RC.from_value(record.get("rc_round1"))
    except ValueError:
        return False
    return True


def _has_reusable_round1(
    instance: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    return (
        _has_valid_round1(record)
        and record.get("db_id") == instance.get("db_id")
        and record.get("question") == instance.get("question")
        and record.get("evidence") == instance.get("evidence")
    )


def _has_reusable_round2(
    instance: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    if record.get("round2_status") != "succeeded":
        return False
    try:
        Round1RC.from_value(record.get("rc_round1"))
        Round2RC.from_value(record.get("rc_round2"))
    except ValueError:
        return False
    return _has_reusable_round1(instance, record)


def _has_reusable_round3(
    instance: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    if record.get("round3_status") != "succeeded":
        return False
    try:
        Round3RC.from_value(record.get("rc_round3"))
    except ValueError:
        return False
    return _has_reusable_round2(instance, record)


def _round_status_counts(
    instances: Sequence[Mapping[str, Any]],
    records_by_index: Mapping[InstanceIndex, Mapping[str, Any]],
    round_number: int,
) -> dict[str, int]:
    counts = {"success": 0, "fail": 0, "empty": 0}
    status_field = f"round{round_number}_status"
    for instance in instances:
        record = records_by_index.get(instance["index"])
        status = record.get(status_field) if record is not None else None
        if status == "succeeded":
            counts["success"] += 1
        elif status == "failed":
            counts["fail"] += 1
        else:
            counts["empty"] += 1
    return counts


def _log_completion_counts(
    instances: Sequence[Mapping[str, Any]],
    records_by_index: Mapping[InstanceIndex, Mapping[str, Any]],
) -> None:
    LOGGER.info(_completion_counts_message(instances, records_by_index))


def _completion_counts_message(
    instances: Sequence[Mapping[str, Any]],
    records_by_index: Mapping[InstanceIndex, Mapping[str, Any]],
) -> str:
    completed = {
        round_number: _round_status_counts(
            instances,
            records_by_index,
            round_number=round_number,
        )["success"]
        for round_number in (1, 2, 3)
    }
    return (
        f"indexes={len(instances)} | round1 completed={completed[1]} | "
        f"round2 completed={completed[2]} | round3 completed={completed[3]}"
    )


def _run_phase(
    phase_instances: list[dict[str, Any]],
    all_instances: list[dict[str, Any]],
    records_by_index: dict[InstanceIndex, dict[str, Any]],
    metadata_by_db: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: Path,
    model_call: ModelCall | None,
    adaptive_concurrency: RampConcurrency,
) -> None:
    if not phase_instances:
        return
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENCY, len(phase_instances))
    ) as executor:
        work_items = iter(phase_instances)
        futures: dict[Any, dict[str, Any]] = {}

        def fill_available_slots() -> None:
            while len(futures) < adaptive_concurrency.limit:
                try:
                    instance = next(work_items)
                except StopIteration:
                    return
                future = executor.submit(
                    _generate_record,
                    instance,
                    records_by_index.get(instance["index"]),
                    metadata_by_db[instance["db_id"]],
                    model_call,
                    adaptive_concurrency,
                )
                futures[future] = instance

        fill_available_slots()
        while futures:
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                instance = futures.pop(future)
                record = future.result()
                records_by_index[instance["index"]] = record
                _write_json_atomic(
                    output_path,
                    _ordered_records(all_instances, records_by_index),
                )
            fill_available_slots()


def _run_round3_phase(
    phase_instances: list[dict[str, Any]],
    all_instances: list[dict[str, Any]],
    records_by_index: dict[InstanceIndex, dict[str, Any]],
    output_path: Path,
    model_call: ModelCall | None,
    adaptive_concurrency: RampConcurrency,
) -> None:
    if not phase_instances:
        return
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENCY, len(phase_instances))
    ) as executor:
        work_items = iter(phase_instances)
        futures: dict[Any, dict[str, Any]] = {}

        def fill_available_slots() -> None:
            while len(futures) < adaptive_concurrency.limit:
                try:
                    instance = next(work_items)
                except StopIteration:
                    return
                future = executor.submit(
                    _generate_round3_record,
                    instance,
                    records_by_index[instance["index"]],
                    model_call,
                    adaptive_concurrency,
                )
                futures[future] = instance

        fill_available_slots()
        while futures:
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                instance = futures.pop(future)
                records_by_index[instance["index"]] = future.result()
                _write_json_atomic(
                    output_path,
                    _ordered_records(all_instances, records_by_index),
                )
            fill_available_slots()


def _generate_record(
    instance: dict[str, Any],
    existing: Mapping[str, Any] | None,
    metadata: Sequence[Mapping[str, Any]],
    model_call: ModelCall | None,
    adaptive_concurrency: RampConcurrency,
) -> dict[str, Any]:
    prefix = {
        "index": instance["index"],
        "db_id": instance["db_id"],
        "question": instance["question"],
        "evidence": instance["evidence"],
    }

    if existing is not None and _has_reusable_round1(instance, existing):
        round1_rc = Round1RC.from_value(existing["rc_round1"])
    else:
        try:
            round1_rc = _run_round(
                index=instance["index"],
                round_number=1,
                adaptive_concurrency=adaptive_concurrency,
                operation=lambda: generate_round1(
                    question=instance["question"],
                    evidence=instance["evidence"],
                    model_call=model_call,
                    max_attempts=1,
                ),
            )
        except Exception as exc:
            root_error = _deepest_error(exc)
            return {
                **prefix,
                "round1_status": "failed",
                "round1_error": _error_value(root_error),
                "rc_round1": None,
                "round2_status": "blocked_round1",
                "round2_error": {
                    "type": "Round1Unavailable",
                    "message": "Round 2 requires a successful Round-1 contract.",
                },
                "rc_round2": None,
            }

    try:
        round2_rc = _run_round(
            index=instance["index"],
            round_number=2,
            adaptive_concurrency=adaptive_concurrency,
            operation=lambda: generate_round2(
                question=instance["question"],
                evidence=instance["evidence"],
                round1_rc=round1_rc,
                metadata=metadata,
                model_call=model_call,
                max_attempts=1,
                metadata_complete=True,
            ),
        )
    except Exception as exc:
        root_error = _deepest_error(exc)
        return {
            **prefix,
            "round1_status": "succeeded",
            "round1_error": None,
            "rc_round1": round1_rc.to_dict(),
            "round2_status": "failed",
            "round2_error": _error_value(root_error),
            "rc_round2": None,
        }

    return {
        **prefix,
        "round1_status": "succeeded",
        "round1_error": None,
        "rc_round1": round1_rc.to_dict(),
        "round2_status": "succeeded",
        "round2_error": None,
        "rc_round2": round2_rc.to_dict(),
    }


def _generate_round3_record(
    instance: dict[str, Any],
    existing: Mapping[str, Any],
    model_call: ModelCall | None,
    adaptive_concurrency: RampConcurrency,
) -> dict[str, Any]:
    round2_rc = Round2RC.from_value(existing["rc_round2"])
    try:
        round3_rc = _run_round(
            index=instance["index"],
            round_number=3,
            adaptive_concurrency=adaptive_concurrency,
            operation=lambda: generate_round3(
                question=instance["question"],
                evidence=instance["evidence"],
                round2_rc=round2_rc,
                gold_sql=instance["gold_sql"],
                model_call=model_call,
                max_attempts=1,
            ),
        )
    except Exception as exc:
        root_error = _deepest_error(exc)
        return {
            **existing,
            "round3_status": "failed",
            "round3_error": _error_value(root_error),
            "rc_round3": None,
        }

    return {
        **existing,
        "round3_status": "succeeded",
        "round3_error": None,
        "rc_round3": round3_rc.to_dict(),
    }


def _run_round(
    index: InstanceIndex,
    round_number: int,
    adaptive_concurrency: RampConcurrency,
    operation: Callable[[], Any],
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        adaptive_concurrency.acquire()
        try:
            result = operation()
        except Exception as exc:
            adaptive_concurrency.release()
            last_error = _deepest_error(exc)
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt)
        else:
            adaptive_concurrency.release()
            LOGGER.info("%5s | round%d | success", index, round_number)
            return result

    assert last_error is not None
    LOGGER.info("%5s | round%d | fail", index, round_number)
    raise RuntimeError(
        f"Round-{round_number} RC generation failed after {MAX_ATTEMPTS} attempts"
    ) from last_error


def _error_value(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _deepest_error(error: Exception) -> Exception:
    while isinstance(error.__cause__, Exception):
        error = error.__cause__
    return error


def _ordered_records(
    instances: list[dict[str, Any]],
    records_by_index: Mapping[InstanceIndex, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        records_by_index[instance["index"]]
        for instance in instances
        if instance["index"] in records_by_index
    ]


def _is_valid_instance_index(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _status_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "complete": sum(
            record.get("round1_status") == "succeeded"
            and record.get("round2_status") == "succeeded"
            for record in records
        ),
        "round1_failed": sum(
            record.get("round1_status") != "succeeded" for record in records
        ),
        "round2_failed": sum(
            record.get("round1_status") == "succeeded"
            and record.get("round2_status") != "succeeded"
            for record in records
        ),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
