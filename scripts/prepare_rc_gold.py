"""Prepare reference SQL for approved result-contract correction datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


DATASET_SPLITS = (
    "bird_interact_full",
    "bird_interact_lite",
    "spider2_lite",
)
REPORT_VERSION = 1
_SAFE_SPIDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\r\n]*(?:\r?\n|\Z)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference source not found: {path}") from None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _resolve_from_cwd(path: str | Path, cwd: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve()


def _portable_path(path: Path, cwd: Path) -> str:
    return Path(os.path.relpath(path, cwd)).as_posix()


def _relative_source(path: Path, declared_root: Path) -> str:
    try:
        return path.relative_to(declared_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Reference source {path} is outside declared dataset root {declared_root}"
        ) from exc


def _require_nonempty_text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _require_text(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")
    return value


def _load_preprocessed(path: Path) -> list[dict[str, str]]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError("Preprocessed input must be a JSON array")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for position, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Preprocessed row at position {position} must be an object")
        index = _require_nonempty_text(
            raw.get("index"), f"Preprocessed index at position {position}"
        )
        if index in seen:
            raise ValueError(f"Duplicate preprocessed index: {index}")
        seen.add(index)
        result.append(
            {
                "index": index,
                "db_id": _require_nonempty_text(
                    raw.get("db_id"), f"Preprocessed db_id for {index!r}"
                ),
                "question": _require_nonempty_text(
                    raw.get("question"), f"Preprocessed question for {index!r}"
                ),
                "evidence": _require_text(
                    raw.get("evidence"), f"Preprocessed evidence for {index!r}"
                ),
            }
        )
    return result


def _parse_exclusions(
    exclusions: Iterable[str] | Mapping[str, str] | None,
    known_ids: set[str],
) -> dict[str, str]:
    if exclusions is None:
        return {}
    entries = (
        [f"{index}={reason}" for index, reason in exclusions.items()]
        if isinstance(exclusions, Mapping)
        else list(exclusions)
    )
    parsed: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, str) or "=" not in entry:
            raise ValueError("Each exclusion must use INDEX=REASON")
        index, reason = entry.split("=", 1)
        if not index.strip():
            raise ValueError("Exclusion must have a non-empty index")
        if not reason.strip():
            raise ValueError(f"Exclusion for {index!r} must have a non-empty reason")
        if index in parsed:
            raise ValueError(f"Duplicate exclusion id: {index}")
        parsed[index] = reason.strip()
    unknown = sorted(set(parsed) - known_ids)
    if unknown:
        raise ValueError(f"Unknown exclusion id(s): {unknown}")
    return parsed


def _has_required_sql(value: Any, field: str, index: str) -> bool:
    if value is None or value == "" or value == []:
        return False
    if isinstance(value, str):
        required = bool(value.strip())
    elif isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} for {index!r} must contain only strings")
        required = any(item.strip() for item in value)
    else:
        raise ValueError(f"{field} for {index!r} must be text, a list, or null")
    return required


def _validate_query_sql(sql: str, description: str) -> str:
    if not sql.strip():
        raise ValueError(f"{description} must contain non-empty SQL")
    without_comments = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", sql))
    if not without_comments.strip():
        raise ValueError(f"{description} must contain non-comment SQL")
    return sql


def _bird_sol_sql(row: dict[str, Any], index: str) -> str | None:
    for field in ("preprocess_sql", "clean_up_sqls"):
        if _has_required_sql(row.get(field), field, index):
            raise ValueError(f"{field} for {index!r} requires setup or cleanup SQL")
    value = row.get("sol_sql")
    if value is None or value == []:
        return None
    if isinstance(value, str):
        sql = value
    elif isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"ambiguous sol_sql for {index!r}: expected one SQL answer")
        if not isinstance(value[0], str):
            raise ValueError(f"sol_sql for {index!r} must contain a string")
        sql = value[0]
    else:
        raise ValueError(f"sol_sql for {index!r} must be a string or one-element list")
    return _validate_query_sql(sql, f"sol_sql for {index!r}")


def _index_rows(
    rows: list[dict[str, Any]], id_field: str, source_name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        index = _require_nonempty_text(
            row.get(id_field), f"{source_name} {id_field} at position {position}"
        )
        if index in result:
            raise ValueError(f"Duplicate {source_name} {id_field}: {index}")
        result[index] = row
    return result


def _resolve_bird_root(root: Path, variant: str) -> Path:
    expected = f"bird-interact-{variant}"
    return root if root.name == expected else root / expected


def _resolve_spider_root(root: Path) -> Path:
    return root if root.name == "spider2-lite" else root / "spider2-lite"


def _source_record(role: str, path: Path, root_name: str, root: Path) -> dict[str, str]:
    return {
        "role": role,
        "root": root_name,
        "path": _relative_source(path, root),
        "sha256": _sha256(path),
    }


def _record(
    instance: dict[str, str],
    status: str,
    reason: str,
    reference: dict[str, str],
) -> dict[str, Any]:
    return {
        "index": instance["index"],
        "db_id": instance["db_id"],
        "status": status,
        "reason": reason,
        "reference": reference,
    }


def _prepare_bird(
    dataset_split: str,
    instances: list[dict[str, str]],
    dataset_root: Path,
    livesqlbench_root: Path | None,
    exclusions: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]]]:
    variant = "full" if dataset_split.endswith("_full") else "lite"
    variant_root = _resolve_bird_root(dataset_root, variant)
    bird_path = variant_root / "bird_interact_data.jsonl"
    bird_query_rows = [
        row for row in _read_jsonl(bird_path) if row.get("category") == "Query"
    ]
    bird_by_id = _index_rows(
        bird_query_rows, "instance_id", "BIRD-Interact Query"
    )
    sources = [_source_record("bird_interact", bird_path, "dataset_root", dataset_root)]

    livesqlbench_by_id: dict[str, dict[str, Any]] = {}
    if variant == "full":
        if livesqlbench_root is None:
            raise ValueError("livesqlbench_root is required for bird_interact_full")
        live_path = livesqlbench_root / "livesqlbench_data.jsonl"
        livesqlbench_by_id = _index_rows(
            _read_jsonl(live_path), "instance_id", "LiveSQLBench"
        )
        sources.append(
            _source_record(
                "livesqlbench", live_path, "livesqlbench_root", livesqlbench_root
            )
        )

    gold: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    reference = {
        "root": "dataset_root",
        "path": _relative_source(bird_path, dataset_root),
        "sha256": _sha256(bird_path),
    }
    for instance in instances:
        index = instance["index"]
        source = bird_by_id.get(index)
        if source is None:
            raise ValueError(f"No BIRD-Interact Query source row for {index!r}")
        source_db = _require_nonempty_text(
            source.get("selected_database"), f"selected_database for {index!r}"
        )
        if source_db != instance["db_id"]:
            raise ValueError(
                f"Database mismatch for {index!r}: source={source_db!r}, "
                f"preprocessed={instance['db_id']!r}"
            )
        if variant == "full":
            live = livesqlbench_by_id.get(index)
            if live is None:
                raise ValueError(f"Missing LiveSQLBench query for {index!r}")
            if live.get("category") != "Query":
                raise ValueError(f"LiveSQLBench row for {index!r} is not category Query")
            live_db = _require_nonempty_text(
                live.get("selected_database"),
                f"LiveSQLBench selected_database for {index!r}",
            )
            if live_db != source_db:
                raise ValueError(
                    f"Database mismatch for {index!r}: BIRD-Interact={source_db!r}, "
                    f"LiveSQLBench={live_db!r}"
                )
            source_question = live.get("query")
        else:
            source_question = source.get("query")
        question = _require_nonempty_text(
            source_question, f"Reference question for {index!r}"
        )
        if question != instance["question"]:
            raise ValueError(f"Question mismatch for {index!r}")

        sql = _bird_sol_sql(source, index)
        if index in exclusions:
            records.append(
                _record(instance, "excluded", exclusions[index], reference.copy())
            )
        elif sql is None:
            records.append(
                _record(
                    instance,
                    "missing_gold_sql",
                    "BIRD-Interact sol_sql is unavailable",
                    reference.copy(),
                )
            )
        else:
            gold.append(
                {
                    "index": index,
                    "db_id": instance["db_id"],
                    "question": instance["question"],
                    "gold_sql": sql,
                }
            )
            records.append(
                _record(instance, "ready", "Matched reference SQL", reference.copy())
            )
    return gold, records, sources


def _canonical_spider_db(db_id: str) -> str:
    return db_id.replace("-", "_").upper()


def _prepare_spider(
    instances: list[dict[str, str]],
    dataset_root: Path,
    exclusions: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]]]:
    spider_root = _resolve_spider_root(dataset_root)
    index_path = spider_root / "spider2-lite.jsonl"
    source_by_id = _index_rows(
        _read_jsonl(index_path), "instance_id", "Spider2-Lite"
    )
    sources = [_source_record("spider2_index", index_path, "dataset_root", dataset_root)]
    sql_root = spider_root / "evaluation_suite" / "gold" / "sql"
    gold: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for instance in instances:
        index = instance["index"]
        if (
            _SAFE_SPIDER_ID.fullmatch(index) is None
            or index in {".", ".."}
            or Path(index).name != index
        ):
            raise ValueError(f"Spider2 instance_id must be a safe filename: {index!r}")
        source = source_by_id.get(index)
        if source is None:
            raise ValueError(f"No Spider2-Lite source row for {index!r}")
        source_db = _require_nonempty_text(source.get("db"), f"Spider2 db for {index!r}")
        if _canonical_spider_db(source_db) != instance["db_id"]:
            raise ValueError(
                f"Database mismatch for {index!r}: source={source_db!r}, "
                f"preprocessed={instance['db_id']!r}"
            )
        question = _require_nonempty_text(
            source.get("question"), f"Spider2 question for {index!r}"
        )
        if question != instance["question"]:
            raise ValueError(f"Question mismatch for {index!r}")

        sql_path = sql_root / f"{index}.sql"
        reference = {
            "root": "dataset_root",
            "path": _relative_source(sql_path, dataset_root),
        }
        if sql_path.is_file():
            sql = _validate_query_sql(
                sql_path.read_text(encoding="utf-8"),
                f"Spider2 gold SQL for {index!r}",
            )
            sql_hash = _sha256(sql_path)
            reference["sha256"] = sql_hash
            sources.append(
                {
                    "role": "gold_sql",
                    "root": "dataset_root",
                    "path": reference["path"],
                    "sha256": sql_hash,
                }
            )
        else:
            sql = None

        if index in exclusions:
            records.append(
                _record(instance, "excluded", exclusions[index], reference)
            )
        elif sql is None:
            records.append(
                _record(
                    instance,
                    "missing_gold_sql",
                    "Spider2 gold SQL file is unavailable",
                    reference,
                )
            )
        else:
            gold.append(
                {
                    "index": index,
                    "db_id": instance["db_id"],
                    "question": instance["question"],
                    "gold_sql": sql,
                }
            )
            records.append(_record(instance, "ready", "Matched reference SQL", reference))
    return gold, records, sources


def _write_outputs_atomic(output_dir: Path, gold: Any, report: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_payload = (
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    report["gold_sql"] = {
        "path": "gold_sql.json",
        "sha256": hashlib.sha256(gold_payload).hexdigest(),
    }
    payloads = {
        "gold_sql.json": gold_payload,
        "gold_sql_preparation.json": (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8"),
    }
    temporary_paths: list[Path] = []
    try:
        for name, payload in payloads.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".tmp", dir=output_dir
            )
            temporary_path = Path(temporary_name)
            temporary_paths.append(temporary_path)
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
        for temporary_path, name in zip(temporary_paths, payloads):
            os.replace(temporary_path, output_dir / name)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def prepare_rc_gold(
    dataset_split: str,
    *,
    dataset_root: str | Path | None = None,
    livesqlbench_root: str | Path | None = None,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exclusions: Iterable[str] | Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Validate reference joins and publish SQL plus an auditable preparation report."""
    if dataset_split not in DATASET_SPLITS:
        raise ValueError(f"Unsupported dataset_split: {dataset_split!r}")
    working_dir = Path.cwd().resolve() if cwd is None else Path(cwd).resolve()
    if dataset_root is None:
        dataset_root = (
            "Spider2.0"
            if dataset_split == "spider2_lite"
            else "BIRD-Interact/BIRD-Interact-ADK"
        )
    declared_dataset_root = _resolve_from_cwd(dataset_root, working_dir)
    declared_livesqlbench_root = (
        _resolve_from_cwd(livesqlbench_root, working_dir)
        if livesqlbench_root is not None
        else (
            _resolve_from_cwd("livesqlbench-base-full-v1", working_dir)
            if dataset_split == "bird_interact_full"
            else None
        )
    )
    data_dir = Path(__file__).resolve().parent.parent / "data" / dataset_split
    resolved_input = (
        _resolve_from_cwd(input_path, working_dir)
        if input_path is not None
        else data_dir / f"{dataset_split}.json"
    ).resolve()
    resolved_output = (
        _resolve_from_cwd(output_dir, working_dir)
        if output_dir is not None
        else data_dir
    ).resolve()
    output_paths = {
        (resolved_output / "gold_sql.json").resolve(),
        (resolved_output / "gold_sql_preparation.json").resolve(),
    }
    if resolved_input in output_paths:
        raise ValueError(f"Input path collides with an output path: {resolved_input}")

    instances = _load_preprocessed(resolved_input)
    parsed_exclusions = _parse_exclusions(
        exclusions, {instance["index"] for instance in instances}
    )
    if dataset_split.startswith("bird_interact_"):
        gold, records, sources = _prepare_bird(
            dataset_split,
            instances,
            declared_dataset_root,
            declared_livesqlbench_root,
            parsed_exclusions,
        )
    else:
        gold, records, sources = _prepare_spider(
            instances, declared_dataset_root, parsed_exclusions
        )

    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("ready", "missing_gold_sql", "excluded")
    }
    report = {
        "version": REPORT_VERSION,
        "dataset_split": dataset_split,
        "input": {
            "path": _portable_path(resolved_input, working_dir),
            "sha256": _sha256(resolved_input),
        },
        "source_roots": {
            "dataset_root": _portable_path(declared_dataset_root, working_dir),
            **(
                {
                    "livesqlbench_root": _portable_path(
                        declared_livesqlbench_root, working_dir
                    )
                }
                if declared_livesqlbench_root is not None
                else {}
            ),
        },
        "sources": sources,
        "counts": {"total": len(instances), **status_counts},
        "records": records,
    }
    _write_outputs_atomic(resolved_output, gold, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_split", required=True, choices=DATASET_SPLITS)
    parser.add_argument("--dataset-root")
    parser.add_argument("--livesqlbench-root")
    parser.add_argument("--input-path")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="INDEX=REASON",
        help="Exclude one validated instance with an explicit reason; repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = prepare_rc_gold(
        args.dataset_split,
        dataset_root=args.dataset_root,
        livesqlbench_root=args.livesqlbench_root,
        input_path=args.input_path,
        output_dir=args.output_dir,
        exclusions=args.exclude,
    )
    counts = report["counts"]
    print(
        f"Prepared {counts['ready']} ready, {counts['missing_gold_sql']} missing, "
        f"and {counts['excluded']} excluded rows out of {counts['total']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
