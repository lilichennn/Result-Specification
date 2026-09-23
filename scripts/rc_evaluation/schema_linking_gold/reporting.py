"""Deterministic exports and quality metrics for the gold-annotation pilot."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .source import AnnotationTask, load_offline_groups
from .store import AnnotationStore


_STATUSES = ("resolved", "needs_review", "invalid_sql", "pending")


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float]) -> tuple[float | None, int]:
    selected = list(values)
    return (_ratio(sum(selected), len(selected)), len(selected))


def _set_f1(tp: int, fp: int, fn: int) -> float | None:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _set_metrics(pairs: Sequence[tuple[set[Any], set[Any]]]) -> dict[str, Any]:
    """Aggregate set metrics while leaving zero-denominator question metrics undefined."""
    totals = Counter(tp=0, fp=0, fn=0)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    exact = full = full_eligible = 0
    for gold, prediction in pairs:
        tp = len(gold & prediction)
        fp = len(prediction - gold)
        fn = len(gold - prediction)
        totals.update(tp=tp, fp=fp, fn=fn)
        precision = None if not gold and not prediction else _ratio(tp, tp + fp) or 0.0
        recall = _ratio(tp, tp + fn)
        question_f1 = _set_f1(tp, fp, fn)
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if question_f1 is not None:
            f1s.append(question_f1)
        exact += gold == prediction
        if gold:
            full_eligible += 1
            full += not (gold - prediction)

    micro_precision = (
        None if totals["tp"] + totals["fp"] + totals["fn"] == 0
        else _ratio(totals["tp"], totals["tp"] + totals["fp"]) or 0.0
    )
    micro_recall = _ratio(totals["tp"], totals["tp"] + totals["fn"])
    macro_precision, macro_precision_count = _mean(precisions)
    macro_recall, macro_recall_count = _mean(recalls)
    macro_f1, macro_f1_count = _mean(f1s)
    questions = len(pairs)
    return {
        "questions": questions,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": _set_f1(totals["tp"], totals["fp"], totals["fn"]),
        },
        "macro": {
            "precision": macro_precision,
            "precision_questions": macro_precision_count,
            "recall": macro_recall,
            "recall_questions": macro_recall_count,
            "f1": macro_f1,
            "f1_questions": macro_f1_count,
        },
        "exact_set": {"count": exact, "ratio": _ratio(exact, questions)},
        "full_recall": {"count": full, "eligible": full_eligible, "ratio": _ratio(full, full_eligible)},
    }


def _prediction_sets(task: AnnotationTask, field: str) -> tuple[set[str], set[tuple[str, str]]]:
    linked = getattr(task, field)
    if not isinstance(linked, Mapping):
        raise ValueError(f"{field} must be a linked-schema object for {task.task_key}")
    tables: set[str] = set()
    columns: set[tuple[str, str]] = set()
    for table, raw_columns in linked.items():
        if not isinstance(table, str) or not isinstance(raw_columns, (list, tuple)):
            raise ValueError(f"invalid linked schema for {task.task_key}")
        tables.add(table)
        for column in raw_columns:
            if not isinstance(column, str):
                raise ValueError(f"invalid linked column for {task.task_key}")
            columns.add((table, column))
    return tables, columns


def _annotation_sets(annotation: Mapping[str, Any]) -> tuple[set[str], set[tuple[str, str]]]:
    tables = annotation.get("required_tables")
    columns = annotation.get("required_columns")
    if not isinstance(tables, list) or not all(isinstance(table, str) for table in tables):
        raise ValueError("accepted annotation has invalid required_tables")
    if not isinstance(columns, list) or any(
        not isinstance(column, list) or len(column) != 2 or not all(isinstance(item, str) for item in column)
        for column in columns
    ):
        raise ValueError("accepted annotation has invalid required_columns")
    return set(tables), {tuple(column) for column in columns}


def _paired_directions(
    records: Sequence[tuple[set[Any], set[Any], set[Any]]],
) -> dict[str, Any]:
    counts = Counter()
    for gold, native, rc3 in records:
        native_errors = len(gold ^ native)
        rc3_errors = len(gold ^ rc3)
        counts["improvements" if rc3_errors < native_errors else
               "regressions" if rc3_errors > native_errors else "unchanged"] += 1
    return {
        "basis": "symmetric_difference_errors",
        "questions": len(records),
        "improvements": counts["improvements"],
        "regressions": counts["regressions"],
        "unchanged": counts["unchanged"],
    }


def _empty_status_counts() -> dict[str, int]:
    return {status: 0 for status in _STATUSES}


def _status_counts(
    tasks: Sequence[AnnotationTask], annotations: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    overall = _empty_status_counts()
    by_group: dict[str, dict[str, int]] = {}
    by_feature: dict[str, dict[str, int]] = {}
    for task in tasks:
        annotation = annotations[task.task_key]
        status = "pending" if annotation is None else annotation.get("status")
        if status not in _STATUSES:
            raise ValueError(f"accepted annotation has invalid status for {task.task_key}")
        overall[status] += 1
        by_group.setdefault(task.group, _empty_status_counts())[status] += 1
        for feature in task.features:
            by_feature.setdefault(feature, _empty_status_counts())[status] += 1
    return {
        "overall": overall,
        "by_group": {key: by_group[key] for key in sorted(by_group)},
        "by_feature": {key: by_feature[key] for key in sorted(by_feature)},
    }


def _parser_agreement(
    tasks: Sequence[AnnotationTask], annotations: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    table_exact = column_exact = both_exact = eligible = 0
    excluded = Counter()
    disagreements = []
    for task in tasks:
        annotation = annotations[task.task_key]
        if annotation is None or annotation.get("status") != "resolved":
            excluded["annotation_unresolved"] += 1
            continue
        reference = task.conservative_reference
        if not isinstance(reference, Mapping) or reference.get("status") != "available":
            excluded["parser_unavailable"] += 1
            continue
        gold_tables, gold_columns = _annotation_sets(annotation)
        parser_tables = set(reference.get("tables", ()))
        parser_columns = {tuple(column) for column in reference.get("columns", ())}
        tables_match = gold_tables == parser_tables
        columns_match = gold_columns == parser_columns
        eligible += 1
        table_exact += tables_match
        column_exact += columns_match
        both_exact += tables_match and columns_match
        if not (tables_match and columns_match):
            disagreements.append({
                "task_key": task.task_key,
                "table_exact": tables_match,
                "column_exact": columns_match,
                "annotation_tables": sorted(gold_tables),
                "parser_tables": sorted(parser_tables),
                "annotation_columns": [list(value) for value in sorted(gold_columns)],
                "parser_columns": [list(value) for value in sorted(parser_columns)],
            })
    return {
        "eligible": eligible,
        "table_exact": {"count": table_exact, "ratio": _ratio(table_exact, eligible)},
        "column_exact": {"count": column_exact, "ratio": _ratio(column_exact, eligible)},
        "both_exact": {"count": both_exact, "ratio": _ratio(both_exact, eligible)},
        "excluded": {
            "annotation_unresolved": excluded["annotation_unresolved"],
            "parser_unavailable": excluded["parser_unavailable"],
        },
        "disagreements": disagreements,
    }


def _numeric_token(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value


def _request_accounting(
    database: Path, task_keys: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        all_rows = connection.execute(
            "SELECT a.attempt_id, a.batch_json, a.attempt_no, a.started_at, "
            "o.status, o.raw_response, o.usage_json, o.latency_seconds, o.error_json "
            "FROM attempts a LEFT JOIN outcomes o ON o.attempt_id = a.attempt_id "
            "ORDER BY a.started_at, a.attempt_id"
        ).fetchall()
    finally:
        connection.close()
    rows = [row for row in all_rows if set(json.loads(row["batch_json"])) <= task_keys]
    finished = [row for row in rows if row["status"] is not None]
    latencies = [float(row["latency_seconds"]) for row in finished]
    prompt = completion = total = known_total = 0
    rejected = []
    for row in finished:
        usage = json.loads(row["usage_json"])
        prompt += _numeric_token(usage.get("prompt_tokens")) or 0
        completion += _numeric_token(usage.get("completion_tokens")) or 0
        total_value = _numeric_token(usage.get("total_tokens"))
        if total_value is not None:
            total += total_value
            known_total += 1
        if row["status"] == "failed" and row["raw_response"] is not None:
            rejected.append({
                "attempt_id": row["attempt_id"],
                "attempt_no": row["attempt_no"],
                "task_keys": json.loads(row["batch_json"]),
                "raw_response": row["raw_response"],
                "error": json.loads(row["error_json"]) if row["error_json"] is not None else None,
                "latency_seconds": row["latency_seconds"],
            })
    accounting = {
        "attempts": len(rows),
        "finished": len(finished),
        "unfinished": len(rows) - len(finished),
        "succeeded": sum(row["status"] == "succeeded" for row in finished),
        "failed": sum(row["status"] == "failed" for row in finished),
        "retries": sum(row["attempt_no"] > 1 for row in rows),
        "latency_seconds": {
            "total": sum(latencies),
            "mean": _ratio(sum(latencies), len(latencies)),
            "maximum": max(latencies) if latencies else None,
        },
        "tokens": {
            "prompt": prompt,
            "completion": completion,
            "total": total,
            "known_total_attempts": known_total,
            "unknown_total_attempts": len(finished) - known_total,
        },
    }
    return accounting, rejected


def _selected_tasks(manifest: Mapping[str, Any], tasks: Sequence[AnnotationTask]) -> list[AnnotationTask]:
    provided = {task.task_key: task for task in tasks}
    if len(provided) != len(tasks):
        raise ValueError("source tasks must have unique task keys")
    frozen_tasks = manifest.get("tasks")
    if not isinstance(frozen_tasks, list):
        raise ValueError("frozen manifest has no tasks")
    selected = []
    for frozen in sorted(frozen_tasks, key=lambda value: value.get("task_key", "")):
        if not isinstance(frozen, Mapping) or not isinstance(frozen.get("task_key"), str):
            raise ValueError("frozen manifest has a malformed task")
        task = provided.get(frozen["task_key"])
        if task is None:
            raise ValueError(f"frozen task is unavailable from source: {frozen['task_key']}")
        for field in ("group", "dialect", "schema_sha256", "sql_sha256"):
            if frozen.get(field) != getattr(task, field):
                raise ValueError(f"frozen task provenance mismatch for {task.task_key}: {field}")
        if "source_hash" in frozen and frozen["source_hash"] != task.source_hash:
            raise ValueError(f"frozen task provenance mismatch for {task.task_key}: source_hash")
        if "source_schema_sha256" in frozen and frozen["source_schema_sha256"] != task.source_schema_sha256:
            raise ValueError(f"frozen task provenance mismatch for {task.task_key}: source_schema_sha256")
        if "features" in frozen and tuple(frozen["features"]) != task.features:
            raise ValueError(f"frozen task provenance mismatch for {task.task_key}: features")
        selected.append(task)
    return selected


def _scoped_tasks(
    manifest: Mapping[str, Any], tasks: Sequence[AnnotationTask], scope: str,
) -> list[AnnotationTask]:
    if scope == "full":
        return list(tasks)
    if scope != "pilot":
        raise ValueError("scope must be pilot or full")
    pilot_keys = manifest.get("pilot_task_keys")
    pilot_size = manifest.get("pilot_size")
    if (not isinstance(pilot_keys, list) or not all(isinstance(key, str) for key in pilot_keys)
            or len(pilot_keys) != len(set(pilot_keys)) or pilot_size != len(pilot_keys)):
        raise ValueError("frozen manifest has malformed pilot membership")
    by_key = {task.task_key: task for task in tasks}
    if not set(pilot_keys) <= set(by_key):
        raise ValueError("frozen pilot contains a task outside the manifest")
    return [by_key[key] for key in sorted(pilot_keys)]


def _build_report(
    manifest: Mapping[str, Any],
    status: Mapping[str, int],
    tasks: Sequence[AnnotationTask],
    annotations: Mapping[str, Mapping[str, Any] | None],
    request_accounting: Mapping[str, Any],
    rejected_outputs: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    resolved = [task for task in tasks if annotations[task.task_key] is not None
                and annotations[task.task_key].get("status") == "resolved"]
    metrics: dict[str, dict[str, Any]] = {}
    table_triples: list[tuple[set[str], set[str], set[str]]] = []
    column_triples: list[tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]] = []
    for task in resolved:
        annotation = annotations[task.task_key]
        assert annotation is not None
        gold_tables, gold_columns = _annotation_sets(annotation)
        native_tables, native_columns = _prediction_sets(task, "native_linked_schema")
        rc3_tables, rc3_columns = _prediction_sets(task, "rc_linked_schema")
        table_triples.append((gold_tables, native_tables, rc3_tables))
        column_triples.append((gold_columns, native_columns, rc3_columns))
    for name, index in (("native", 1), ("rc3", 2)):
        metrics[name] = {
            "tables": _set_metrics([(triple[0], triple[index]) for triple in table_triples]),
            "columns": _set_metrics([(triple[0], triple[index]) for triple in column_triples]),
        }

    counts = _status_counts(tasks, annotations)
    overall = counts["overall"]
    total_tasks = len(tasks)
    accepted_tasks = sum(annotation is not None for annotation in annotations.values())
    unique_inputs = len({task.reuse_key for task in tasks})
    accepted_unique_inputs = len({task.reuse_key for task in tasks if annotations[task.task_key] is not None})
    excluded_annotations = [
        {
            "task_key": task.task_key,
            "status": annotations[task.task_key]["status"],
            "review_reasons": annotations[task.task_key].get("review_reasons", []),
        }
        for task in tasks
        if annotations[task.task_key] is not None and annotations[task.task_key].get("status") != "resolved"
    ]
    return {
        "format": f"schema-linking-gold-{scope}-report-v1",
        "scope": scope,
        "manifest": dict(manifest),
        "coverage": {
            "total_tasks": total_tasks,
            "accepted_tasks": accepted_tasks,
            "resolved_tasks": overall["resolved"],
            "needs_review_tasks": overall["needs_review"],
            "invalid_sql_tasks": overall["invalid_sql"],
            "pending_tasks": overall["pending"],
            "accepted_ratio": _ratio(accepted_tasks, total_tasks),
            "estimated_full_run_coverage": _ratio(overall["resolved"], total_tasks),
            "cached_unique_input_count": unique_inputs,
            "accepted_unique_input_count": accepted_unique_inputs,
            "manifest_total_tasks": status["total_tasks"],
            "store_accepted_tasks": status["accepted_tasks"],
        },
        "status_counts": counts,
        "metrics": metrics,
        "paired_native_to_rc3": {
            "tables": _paired_directions(table_triples),
            "columns": _paired_directions(column_triples),
        },
        "parser_agreement": _parser_agreement(tasks, annotations),
        "excluded_annotations": excluded_annotations,
        "request_accounting": dict(request_accounting),
        "rejected_outputs": rejected_outputs,
    }


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        f"# Gold-SQL Schema-Linking {report['scope'].title()} Report",
        "",
        f"Resolved coverage: {coverage['resolved_tasks']}/{coverage['total_tasks']} "
        f"({_number(coverage['estimated_full_run_coverage'])}).",
        "",
        "## Annotation status",
        "",
        "| Resolved | Needs review | Invalid SQL | Pending | Unique inputs |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {coverage['resolved_tasks']} | {coverage['needs_review_tasks']} | "
        f"{coverage['invalid_sql_tasks']} | {coverage['pending_tasks']} | "
        f"{coverage['cached_unique_input_count']} |",
        "",
        "## Original and RS metrics",
        "",
        "| Prediction | Level | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Exact sets | Full recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prediction in ("native", "rc3"):
        prediction_label = "Original" if prediction == "native" else "RS"
        for level in ("tables", "columns"):
            metric = report["metrics"][prediction][level]
            lines.append(
                f"| {prediction_label} | {level} | {_number(metric['micro']['precision'])} | "
                f"{_number(metric['micro']['recall'])} | {_number(metric['micro']['f1'])} | "
                f"{_number(metric['macro']['precision'])} | {_number(metric['macro']['recall'])} | "
                f"{_number(metric['macro']['f1'])} | {metric['exact_set']['count']}/{metric['questions']} | "
                f"{metric['full_recall']['count']}/{metric['full_recall']['eligible']} |"
            )
    lines += ["", "## Paired Original-to-RS directions", "",
              "Lower symmetric-difference error is an improvement.", "",
              "| Level | Improvements | Regressions | Unchanged |", "| --- | ---: | ---: | ---: |"]
    for level in ("tables", "columns"):
        direction = report["paired_native_to_rc3"][level]
        lines.append(f"| {level} | {direction['improvements']} | {direction['regressions']} | {direction['unchanged']} |")

    parser = report["parser_agreement"]
    lines += ["", "## Conservative-parser agreement", "",
              f"Eligible resolved tasks: {parser['eligible']}. Table exact: {parser['table_exact']['count']}; "
              f"column exact: {parser['column_exact']['count']}; both exact: {parser['both_exact']['count']}.",
              "", "### Disagreements", ""]
    if not parser["disagreements"]:
        lines.append("None.")
    for row in parser["disagreements"]:
        lines.append(f"- `{row['task_key']}`: table exact={str(row['table_exact']).lower()}, "
                     f"column exact={str(row['column_exact']).lower()}")

    lines += ["", "## Unresolved annotations", ""]
    if not report["excluded_annotations"]:
        lines.append("None.")
    for row in report["excluded_annotations"]:
        reasons = "; ".join(row["review_reasons"]) or "no reason"
        lines.append(f"- `{row['task_key']}` — {row['status']}: {reasons}")

    requests = report["request_accounting"]
    lines += ["", "## Request accounting", "",
              f"Attempts: {requests['attempts']}; succeeded: {requests['succeeded']}; failed: "
              f"{requests['failed']}; retries: {requests['retries']}; latency: "
              f"{requests['latency_seconds']['total']:.4f}s; known total tokens: "
              f"{requests['tokens']['total']}; unknown-token attempts: "
              f"{requests['tokens']['unknown_total_attempts']}.",
              "", "### Rejected model outputs", ""]
    if not report["rejected_outputs"]:
        lines.append("None.")
    for row in report["rejected_outputs"]:
        lines.append(f"#### Attempt `{row['attempt_id']}` (tasks: {', '.join(row['task_keys'])})")
        lines.append("")
        lines.extend(f"    {line}" for line in row["raw_response"].splitlines() or [""])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def export_annotations(
    store_path: str | Path,
    output: str | Path,
    *,
    tasks: Sequence[AnnotationTask] | None = None,
    source_root: str | Path | None = None,
    scope: str = "pilot",
) -> dict[str, Any]:
    """Write one immutable pilot or full annotation/report bundle."""
    with AnnotationStore.open(store_path) as store:
        verification = store.verify()
        if not verification["ok"]:
            raise ValueError("annotation store verification failed")
        manifest = store.manifest
        if tasks is None and source_root is None:
            raise ValueError("source_root is required when annotation tasks are not supplied")
        frozen = _selected_tasks(manifest, list(tasks) if tasks is not None else load_offline_groups(source_root))
        selected = _scoped_tasks(manifest, frozen, scope)
        annotations = {task.task_key: store.annotation_for_task(task.task_key) for task in selected}
        database = store.path
        status = verification["status"]

    accounting, rejected = _request_accounting(database, {task.task_key for task in selected})
    report = _build_report(manifest, status, selected, annotations, accounting, rejected, scope)
    annotation_rows = [annotations[task.task_key] for task in selected if annotations[task.task_key] is not None]
    jsonl = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in annotation_rows
    )
    report_json = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    report_markdown = _markdown(report)

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = ({
        "annotations.jsonl": jsonl,
        "pilot_report.json": report_json,
        "pilot_report.md": report_markdown,
    } if scope == "pilot" else {
        "annotations.full.jsonl": jsonl,
        "full_report.json": report_json,
        "full_report.md": report_markdown,
    })
    existing = next((destination / name for name in artifacts if (destination / name).exists()), None)
    if existing is not None:
        raise FileExistsError(existing)
    for name, text in artifacts.items():
        _write_exclusive(destination / name, text)
    return {
        "status": "success",
        "output": str(destination),
        "files": list(artifacts),
        "scope": scope,
        "coverage": report["coverage"],
    }
