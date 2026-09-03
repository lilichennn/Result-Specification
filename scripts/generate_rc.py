from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
DEFAULT_CONCURRENCY = 5
LOGGER = logging.getLogger("generate_rc")

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from result_contract.rc import (
    ModelCall,
    Round1RC,
    Round2RC,
    generate_round1,
    generate_round2,
)


def generate_rc_file(
    input_path: str | Path,
    meta_root: str | Path,
    output_path: str | Path,
    model_call: ModelCall | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Generate Round 1 and Round 2 with one shared concurrency limit."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    input_path = Path(input_path).resolve()
    meta_root = Path(meta_root).resolve()
    output_path = Path(output_path).resolve()
    instances = _load_instances(input_path)
    LOGGER.info("Loaded %d instances from %s", len(instances), input_path)

    db_ids = {instance["db_id"] for instance in instances}
    metadata_by_db = _load_metadata(meta_root, db_ids)
    LOGGER.info(
        "Loaded metadata for %d databases from %s",
        len(metadata_by_db),
        meta_root,
    )

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
    reusable_count = 0

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
            reusable_count += 1
        else:
            pending_round2_instances.append(instance)

    LOGGER.info(
        "Resume state: complete=%d recovery=%d pending_round2=%d new=%d "
        "concurrency=%d",
        reusable_count,
        len(recovery_instances),
        len(pending_round2_instances),
        len(new_instances),
        concurrency,
    )

    for phase, phase_instances in (
        ("recovery", recovery_instances),
        ("round2", pending_round2_instances),
        ("new", new_instances),
    ):
        _run_phase(
            phase=phase,
            phase_instances=phase_instances,
            all_instances=instances,
            records_by_index=records_by_index,
            metadata_by_db=metadata_by_db,
            output_path=output_path,
            model_call=model_call,
            concurrency=concurrency,
        )

    output = _ordered_records(instances, records_by_index)
    _write_json_atomic(output_path, output)
    counts = _status_counts(output)
    LOGGER.info(
        "Wrote %d records to %s: complete=%d round1_failed=%d "
        "round2_failed=%d",
        len(output),
        output_path,
        counts["complete"],
        counts["round1_failed"],
        counts["round2_failed"],
    )
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
        help=f"Maximum concurrent instances/API requests (default: {DEFAULT_CONCURRENCY}).",
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
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

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
    seen_indices: set[int] = set()
    instances: list[dict[str, Any]] = []
    for position, instance in enumerate(value):
        if not isinstance(instance, dict):
            raise ValueError(f"Instance at position {position} must be an object")
        missing = required_fields - instance.keys()
        if missing:
            raise ValueError(
                f"Instance at position {position} is missing fields: {sorted(missing)}"
            )
        if not isinstance(instance["index"], int):
            raise ValueError(f"Instance index at position {position} must be an integer")
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


def _load_existing_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Existing RC output must be a JSON array: {path}")

    records: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(value):
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
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


def _run_phase(
    phase: str,
    phase_instances: list[dict[str, Any]],
    all_instances: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    metadata_by_db: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: Path,
    model_call: ModelCall | None,
    concurrency: int,
) -> None:
    if not phase_instances:
        return
    LOGGER.info(
        "Starting %s phase: instances=%d concurrency=%d",
        phase,
        len(phase_instances),
        concurrency,
    )
    worker_count = min(concurrency, len(phase_instances))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        work_items = iter(enumerate(phase_instances, start=1))
        futures: dict[Any, dict[str, Any]] = {}

        def submit_next() -> bool:
            try:
                position, instance = next(work_items)
            except StopIteration:
                return False
            future = executor.submit(
                _generate_record,
                instance,
                records_by_index.get(instance["index"]),
                metadata_by_db[instance["db_id"]],
                model_call,
                phase,
                position,
                len(phase_instances),
            )
            futures[future] = instance
            return True

        for _ in range(worker_count):
            submit_next()

        completed = 0
        while futures:
            future = next(as_completed(futures))
            instance = futures.pop(future)
            record = future.result()
            records_by_index[instance["index"]] = record
            _write_json_atomic(
                output_path,
                _ordered_records(all_instances, records_by_index),
            )
            completed += 1
            LOGGER.info(
                "%s phase checkpoint: completed=%d/%d index=%s "
                "round1=%s round2=%s",
                phase,
                completed,
                len(phase_instances),
                instance["index"],
                record["round1_status"],
                record["round2_status"],
            )
            submit_next()
    LOGGER.info("Finished %s phase", phase)


def _generate_record(
    instance: dict[str, Any],
    existing: Mapping[str, Any] | None,
    metadata: Sequence[Mapping[str, Any]],
    model_call: ModelCall | None,
    phase: str,
    position: int,
    total: int,
) -> dict[str, Any]:
    prefix = {
        "index": instance["index"],
        "db_id": instance["db_id"],
        "question": instance["question"],
        "evidence": instance["evidence"],
    }

    if existing is not None and _has_reusable_round1(instance, existing):
        round1_rc = Round1RC.from_value(existing["rc_round1"])
        LOGGER.info(
            "[%s %d/%d] Reusing Round-1 RC for index=%s db_id=%s",
            phase,
            position,
            total,
            instance["index"],
            instance["db_id"],
        )
    else:
        LOGGER.info(
            "[%s %d/%d] Generating Round-1 RC for index=%s db_id=%s",
            phase,
            position,
            total,
            instance["index"],
            instance["db_id"],
        )
        try:
            round1_rc = generate_round1(
                question=instance["question"],
                evidence=instance["evidence"],
                model_call=model_call,
            )
        except Exception as exc:
            LOGGER.error(
                "[%s %d/%d] Round-1 failed for index=%s after retry exhaustion: "
                "%s: %s",
                phase,
                position,
                total,
                instance["index"],
                type(exc).__name__,
                exc,
            )
            root_error = exc.__cause__ or exc
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

    LOGGER.info(
        "[%s %d/%d] Generating Round-2 RC for index=%s db_id=%s",
        phase,
        position,
        total,
        instance["index"],
        instance["db_id"],
    )
    try:
        round2_rc = generate_round2(
            question=instance["question"],
            evidence=instance["evidence"],
            round1_rc=round1_rc,
            metadata=metadata,
            model_call=model_call,
            metadata_complete=True,
        )
    except Exception as exc:
        LOGGER.error(
            "[%s %d/%d] Round-2 failed for index=%s after retry exhaustion: %s: %s",
            phase,
            position,
            total,
            instance["index"],
            type(exc).__name__,
            exc,
        )
        root_error = exc.__cause__ or exc
        return {
            **prefix,
            "round1_status": "succeeded",
            "round1_error": None,
            "rc_round1": round1_rc.to_dict(),
            "round2_status": "failed",
            "round2_error": _error_value(root_error),
            "rc_round2": None,
        }

    LOGGER.info(
        "[%s %d/%d] Round-2 RC completed for index=%s",
        phase,
        position,
        total,
        instance["index"],
    )
    return {
        **prefix,
        "round1_status": "succeeded",
        "round1_error": None,
        "rc_round1": round1_rc.to_dict(),
        "round2_status": "succeeded",
        "round2_error": None,
        "rc_round2": round2_rc.to_dict(),
    }


def _error_value(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _ordered_records(
    instances: list[dict[str, Any]],
    records_by_index: Mapping[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        records_by_index[instance["index"]]
        for instance in instances
        if instance["index"] in records_by_index
    ]


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
