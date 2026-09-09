from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
DEFAULT_CONCURRENCY = 50
MIN_CONCURRENCY = 10
MAX_CONCURRENCY = 500
CONCURRENCY_STEP = 5
STREAK_THRESHOLD = 10
MAX_ATTEMPTS = 3
CONCURRENCY_REPORT_INTERVAL_SECONDS = 15
LOGGER = logging.getLogger("generate_rc")
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
InstanceIndex = int | str

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from result_contract.rc import (
    ModelCall,
    Round1RC,
    Round2RC,
    generate_round1,
    generate_round2,
)


class AdaptiveConcurrency:
    def __init__(self, initial: int) -> None:
        if not MIN_CONCURRENCY <= initial <= MAX_CONCURRENCY:
            raise ValueError(
                f"concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY}"
            )
        self._limit = initial
        self._active = 0
        self._success_streak = 0
        self._failure_streak = 0
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

    def release(self, succeeded: bool) -> None:
        with self._condition:
            self._active -= 1
            if succeeded:
                self._success_streak += 1
                self._failure_streak = 0
                if self._success_streak == STREAK_THRESHOLD:
                    self._limit = min(
                        MAX_CONCURRENCY,
                        self._limit + CONCURRENCY_STEP,
                    )
                    self._success_streak = 0
            else:
                self._failure_streak += 1
                self._success_streak = 0
                if self._failure_streak == STREAK_THRESHOLD:
                    self._limit = max(
                        MIN_CONCURRENCY,
                        self._limit - CONCURRENCY_STEP,
                    )
                    self._failure_streak = 0
            self._condition.notify_all()


class ConsoleColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        text = text.replace(
            " | success in attempt ",
            f" | {GREEN}success{RESET} in attempt ",
        )
        return text.replace(
            " | fail in attempt ",
            f" | {RED}fail{RESET} in attempt ",
        )


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

    round1_counts = _round_status_counts(instances, records_by_index, round_number=1)
    round2_counts = _round_status_counts(instances, records_by_index, round_number=2)
    LOGGER.info(
        "instances=%d | round1 success=%d fail=%d empty=%d | "
        "round2 success=%d fail=%d empty=%d",
        len(instances),
        round1_counts["success"],
        round1_counts["fail"],
        round1_counts["empty"],
        round2_counts["success"],
        round2_counts["fail"],
        round2_counts["empty"],
    )

    adaptive_concurrency = AdaptiveConcurrency(concurrency)
    stop_reporting = threading.Event()

    def report_concurrency() -> None:
        while not stop_reporting.wait(CONCURRENCY_REPORT_INTERVAL_SECONDS):
            LOGGER.info("current concurrency=%d", adaptive_concurrency.limit)

    reporter = threading.Thread(target=report_concurrency, daemon=True)
    reporter.start()
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
        reporter.join()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two-round Result Contracts for a dataset split."
    )
    parser.add_argument(
        "--dataset_split",
        required=True,
        help="Dataset split directory name, for example bird_dev or spider_test.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Initial adaptive API concurrency (default: {DEFAULT_CONCURRENCY}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path, meta_root, output_path, log_path = _dataset_split_paths(
        args.dataset_split
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Preprocessed dataset split not found: {input_path}")
    if not meta_root.is_dir():
        raise FileNotFoundError(f"Preprocessed metadata directory not found: {meta_root}")

    configure_logging(log_path)
    rows = generate_rc_file(
        input_path=input_path,
        meta_root=meta_root,
        output_path=output_path,
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
        not dataset_split
        or Path(dataset_split).name != dataset_split
        or "/" in dataset_split
        or "\\" in dataset_split
    ):
        raise ValueError(
            "dataset_split must be a directory name such as bird_dev or spider_test"
        )

    dataset_split_root = SCRIPT_DIR / dataset_split
    preprocessed_root = dataset_split_root / "preprocessed_data"
    input_path = preprocessed_root / f"{dataset_split}.json"
    meta_root = preprocessed_root / "meta"
    output_path = dataset_split_root / "rc.json"
    log_path = dataset_split_root / "generate_rc.log"
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
    console_handler = logging.StreamHandler()
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


def _load_existing_records(path: Path) -> dict[InstanceIndex, dict[str, Any]]:
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
        normalized = dict(record)
        if "round1_status" not in normalized:
            normalized["round1_status"] = (
                "succeeded" if normalized.get("rc_round1") is not None else "failed"
            )
            normalized.setdefault("round1_error", None)
        records[index] = normalized
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


def _run_phase(
    phase_instances: list[dict[str, Any]],
    all_instances: list[dict[str, Any]],
    records_by_index: dict[InstanceIndex, dict[str, Any]],
    metadata_by_db: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: Path,
    model_call: ModelCall | None,
    adaptive_concurrency: AdaptiveConcurrency,
) -> None:
    if not phase_instances:
        return
    with ThreadPoolExecutor(max_workers=len(phase_instances)) as executor:
        work_items = iter(enumerate(phase_instances, start=1))
        futures: dict[Any, dict[str, Any]] = {}

        def fill_available_slots() -> None:
            while len(futures) < adaptive_concurrency.limit:
                try:
                    _, instance = next(work_items)
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
            future = next(as_completed(futures))
            instance = futures.pop(future)
            record = future.result()
            records_by_index[instance["index"]] = record
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
    adaptive_concurrency: AdaptiveConcurrency,
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
        round1_mode = "recover" if existing is not None else "begin"
        try:
            round1_rc = _run_round(
                index=instance["index"],
                round_number=1,
                mode=round1_mode,
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

    round2_mode = (
        "recover"
        if existing is not None and existing.get("round2_status") == "failed"
        else "begin"
    )
    try:
        round2_rc = _run_round(
            index=instance["index"],
            round_number=2,
            mode=round2_mode,
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


def _run_round(
    index: InstanceIndex,
    round_number: int,
    mode: str,
    adaptive_concurrency: AdaptiveConcurrency,
    operation: Callable[[], Any],
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        adaptive_concurrency.acquire()
        LOGGER.info(
            "%5s | round%d | %s | attempt %d/%d",
            index,
            round_number,
            mode,
            attempt,
            MAX_ATTEMPTS,
        )
        try:
            result = operation()
        except Exception as exc:
            adaptive_concurrency.release(succeeded=False)
            last_error = _deepest_error(exc)
            LOGGER.info(
                "%5s | round%d | fail in attempt %d/%d",
                index,
                round_number,
                attempt,
                MAX_ATTEMPTS,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt)
        else:
            adaptive_concurrency.release(succeeded=True)
            LOGGER.info(
                "%5s | round%d | success in attempt %d/%d",
                index,
                round_number,
                attempt,
                MAX_ATTEMPTS,
            )
            return result

    assert last_error is not None
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
