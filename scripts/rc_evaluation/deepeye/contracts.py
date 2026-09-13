"""Strict, offline loading of pre-generated Result Contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from result_contract.rc import Round1RC, Round2RC

from .injection import render_rc_block


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_sha256(record: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(canonical)


def _load_source(path: Path) -> tuple[dict[int | str, dict[str, Any]], str]:
    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"RC source cannot be read: {resolved}") from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"RC source is not a valid JSON array: {resolved}") from error
    if not isinstance(value, list):
        raise ValueError(f"RC source must be a JSON array: {resolved}")

    records: dict[int | str, dict[str, Any]] = {}
    for position, record in enumerate(value):
        if not isinstance(record, dict):
            raise ValueError(f"RC record at position {position} must be an object: {resolved}")
        index = record.get("index")
        if not isinstance(index, (int, str)) or isinstance(index, bool) or index == "":
            raise ValueError(f"RC record at position {position} has an invalid index: {resolved}")
        if index in records:
            raise ValueError(f"RC source contains duplicate index {index!r}: {resolved}")
        records[index] = record
    return records, _sha256(raw)


def _task_identity(variant: Any, item: Any) -> tuple[str, int | str, str]:
    if not isinstance(variant, str) or not variant:
        raise ValueError("task variant must be non-empty text")
    instance_id = getattr(item, "instance_id", None)
    if (
        not isinstance(instance_id, (int, str))
        or isinstance(instance_id, bool)
        or instance_id == ""
    ):
        raise ValueError(f"task in variant {variant!r} has an invalid instance_id")
    return variant, instance_id, f"{variant}/{instance_id}"


def _validated_contract(
    *,
    task_key: str,
    item: Any,
    record: Mapping[str, Any],
    source_file: Path,
    source_file_sha256: str,
) -> dict[str, Any]:
    expected = {
        "database": (record.get("db_id"), getattr(item, "database_id", None)),
        "question": (record.get("question"), getattr(item, "question", None)),
        "evidence": (record.get("evidence"), getattr(item, "evidence", None)),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(
                f"RC {label} mismatch for {task_key}: source={actual!r}, task={wanted!r}"
            )

    if record.get("round1_status") != "succeeded":
        raise ValueError(
            f"Round 1 RC failed for {task_key}: {record.get('round1_error')!r}"
        )
    if record.get("round2_status") != "succeeded":
        raise ValueError(
            f"Round 2 RC failed for {task_key}: {record.get('round2_error')!r}"
        )
    try:
        round1 = Round1RC.from_value(record.get("rc_round1")).to_dict()
        round2 = Round2RC.from_value(record.get("rc_round2")).to_dict()
    except ValueError as error:
        raise ValueError(f"Invalid RC for {task_key}: {error}") from error

    return {
        "task_key": task_key,
        "db_id": record["db_id"],
        "question": record["question"],
        "evidence": record["evidence"],
        "round1": round1,
        "round2": round2,
        "source_file": str(source_file.resolve()),
        "source_file_sha256": source_file_sha256,
        "record_sha256": _record_sha256(record),
    }


def load_contracts(
    paths: dict[str, Path], tasks: list[tuple]
) -> dict[str, dict]:
    """Load and validate the pre-generated RC for every selected task.

    Records are joined strictly by ``(variant, instance_id)``.  Loading never
    invokes either RC generation function.
    """

    if not isinstance(paths, Mapping):
        raise TypeError("paths must map variants to RC source files")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise TypeError("tasks must be a sequence of (variant, DataItem) pairs")

    parsed: dict[str, tuple[Path, dict[int | str, dict[str, Any]], str]] = {}
    contracts: dict[str, dict] = {}
    for task in tasks:
        if not isinstance(task, tuple) or len(task) != 2:
            raise ValueError("each task must be a (variant, DataItem) pair")
        variant, item = task
        variant, instance_id, task_key = _task_identity(variant, item)
        if task_key in contracts:
            raise ValueError(f"duplicate task selection: {task_key}")
        if variant not in paths:
            raise ValueError(f"RC source is missing for variant {variant!r}")
        if variant not in parsed:
            source_file = Path(paths[variant])
            records, file_hash = _load_source(source_file)
            parsed[variant] = (source_file, records, file_hash)
        source_file, records, file_hash = parsed[variant]
        record = records.get(instance_id)
        if record is None:
            raise ValueError(f"RC record is missing for {task_key}")
        contracts[task_key] = _validated_contract(
            task_key=task_key,
            item=item,
            record=record,
            source_file=source_file,
            source_file_sha256=file_hash,
        )
    return contracts
