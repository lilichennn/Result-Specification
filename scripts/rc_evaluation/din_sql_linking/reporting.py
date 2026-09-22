"""Merge the sealed DIN handoff with its focused RS Linking extension.

The exporter treats the old handoff as an immutable, independently sealed
base.  It copies only files authenticated by the old index, preserves the old
compact node/request bytes as prefixes, and adds the two extension nodes to the
same logical record files.  Provider bodies and prompts are deliberately never
selected from the extension RunStores.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from scripts.baseline_adapters.din_sql.inputs import NODES as BASE_NODES, TaskKey
from scripts.baseline_adapters.din_sql.records import read_json
from scripts.baseline_adapters.din_sql_linking.records import (
    LinkingRecords,
    NODES as EXTENSION_NODES,
)
from scripts.rc_evaluation.din_sql.compact_reporting import (
    REQUEST_KINDS,
    _group_snapshot,
    _snapshot_sqlite,
)


RECORD_FORMAT = "din-current-records-v2"
EVALUATION_FORMAT = "din-compact-evaluation-v2"
HANDOFF_FORMAT = "din-handoff-v2"
_SAFE_REQUEST_KINDS = frozenset(REQUEST_KINDS)


def _json_payload(value: Any) -> bytes:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_line(value: Any) -> bytes:
    return _json_payload(value) + b"\n"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _json_line(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for row in rows:
                stream.write(_json_line(dict(row)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{number}")
        values.append(value)
    return values


def _copy_regular(source: Path, destination: Path) -> None:
    """Copy one authenticated regular file without following a symlink."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source indexed path is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / ("." + destination.name + ".copy-" +
                                      next(tempfile._get_candidate_names()))
    try:
        cloned = False
        if os.uname().sysname == "Darwin":
            try:
                cloned = subprocess.run(
                    ["/bin/cp", "-c", str(source), str(temporary)], check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ).returncode == 0
            except OSError:
                cloned = False
        if not cloned:
            shutil.copy2(source, temporary, follow_symlinks=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_copied_source(stage: Path, source: Mapping[str, Any]) -> None:
    """Prove that every authenticated base payload survived copying unchanged."""
    for relative, expected in source["files"].items():
        path = stage / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"copied source file is missing or unsafe: {relative}")
        if expected.get("bytes") != path.stat().st_size or \
                expected.get("sha256") != _sha_file(path):
            raise ValueError(f"copied source file hash mismatch: {relative}")


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("source index contains an invalid path")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"source index contains an unsafe path: {value!r}")
    return relative


def _file_info(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": _sha_bytes(raw)}


def _validate_source(source: Path) -> dict[str, Any]:
    """Authenticate the old handoff, its index, and compact-record manifest."""
    source = source.resolve(strict=True)
    complete_path, index_path = source / "COMPLETE.json", source / "index.json"
    if not complete_path.is_file() or not index_path.is_file():
        raise ValueError("source handoff is not sealed")
    complete, index = read_json(complete_path), read_json(index_path)
    if complete.get("complete") is not True or index.get("complete") is not True:
        raise ValueError("source handoff is not complete")
    if complete.get("index_sha256") != _sha_file(index_path):
        raise ValueError("source index hash does not match source COMPLETE marker")
    if index.get("format") != "din-handoff-v1":
        raise ValueError("source handoff format is not din-handoff-v1")
    listed: dict[Path, dict[str, Any]] = {}
    rows = index.get("files")
    if not isinstance(rows, list):
        raise ValueError("source index has no file list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source index has a malformed file row")
        relative = _safe_relative(row.get("path"))
        if relative in listed:
            raise ValueError(f"source index repeats {relative}")
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source indexed file is missing or unsafe: {relative}")
        raw = path.read_bytes()
        if row.get("bytes") != len(raw) or row.get("sha256") != _sha_bytes(raw):
            raise ValueError(f"source file hash mismatch: {relative}")
        listed[relative] = row
    required = {
        Path("records/questions.jsonl"), Path("records/versions.jsonl"),
        Path("records/nodes.jsonl"), Path("records/requests.jsonl"),
        Path("records/failed_questions.jsonl"), Path("records/summary.json"),
        Path("records/verification.json"), Path("evaluation/summary.json"),
        Path("evaluation/tables.md"), Path("evaluation/versions.json"),
        Path("raw_records/manifest.json"), Path("raw_records/prepared/inputs.json"),
    }
    missing = required - set(listed)
    if missing:
        raise ValueError(f"source index omits required files: {sorted(map(str, missing))}")
    verification = read_json(source / "records/verification.json")
    if verification.get("ok") is not True or verification.get("format") != "din-current-records-v1":
        raise ValueError("source compact-record verification is not valid")
    for name, expected in verification.get("files", {}).items():
        relative = Path("records") / _safe_relative(name)
        path = source / relative
        if not path.is_file() or expected.get("bytes") != path.stat().st_size or \
                expected.get("sha256") != _sha_file(path):
            raise ValueError(f"source record hash mismatch: {relative}")
    questions = _read_jsonl(source / "records/questions.jsonl")
    versions = _read_jsonl(source / "records/versions.jsonl")
    nodes = _read_jsonl(source / "records/nodes.jsonl")
    keys = [(str(row.get("group")), str(row.get("question_id"))) for row in questions]
    if len(keys) != len(set(keys)):
        raise ValueError("source questions contain duplicate identities")
    version_map = {(str(row.get("group")), str(row.get("question_id"))): row for row in versions}
    if len(version_map) != len(versions) or set(version_map) != set(keys):
        raise ValueError("source versions and questions differ")
    node_map: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in keys}
    for row in nodes:
        key = (str(row.get("group")), str(row.get("question_id")))
        if key not in node_map:
            raise ValueError(f"source node belongs to an unknown question: {key}")
        node_map[key].append(row)
    for key in keys:
        rows_for_key = node_map[key]
        if len(rows_for_key) != len(BASE_NODES) or \
                {row.get("node") for row in rows_for_key} != set(BASE_NODES):
            raise ValueError(f"source question does not have exactly six nodes: {key}")
        version_id = version_map[key].get("version_id")
        if version_id is None or any(row.get("version_id") != version_id for row in rows_for_key):
            raise ValueError(f"source node version mismatch: {key}")
    return {
        "root": source, "index": index, "complete": complete, "files": listed,
        "questions": questions, "versions": versions, "nodes": nodes,
        "keys": keys, "version_map": version_map, "node_map": node_map,
        "index_sha256": _sha_file(index_path),
        "complete_sha256": _sha_file(complete_path),
    }


def _load_extension(extension: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    extension = extension.resolve(strict=True)
    manifest = read_json(extension / "manifest.json")
    if manifest.get("format") != "din-sql-linking-v1":
        raise ValueError("extension manifest format is not din-sql-linking-v1")
    groups = manifest.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("extension manifest has no groups")
    expected = {(group, question_id) for group, question_id in source["keys"]}
    provided = {(str(group), str(question_id)) for group, specification in groups.items()
                for question_id in specification.get("ids", [])}
    if provided != expected:
        raise ValueError("extension and source question sets differ")
    source_lineage = manifest.get("source")
    expected_lineage = {
        "batch_id": source["index"].get("batch_id"),
        "manifest_sha256": _sha_file(source["root"] / "raw_records/manifest.json"),
        "prepared_sha256": _sha_file(source["root"] / "raw_records/prepared/inputs.json"),
    }
    if not isinstance(source_lineage, dict) or any(
            source_lineage.get(name) != value for name, value in expected_lineage.items()):
        raise ValueError("extension source lineage does not identify the sealed base")
    prepared_path = extension / "prepared/inputs.json"
    if not prepared_path.is_file() or prepared_path.is_symlink():
        raise ValueError("extension prepared inputs are missing or unsafe")
    prepared = read_json(prepared_path)
    task_rows = prepared.get("tasks") if isinstance(prepared, dict) else None
    metadata = prepared.get("metadata") if isinstance(prepared, dict) else None
    if not isinstance(task_rows, list) or not isinstance(metadata, dict):
        raise ValueError("extension prepared inputs have no structured tasks/metadata")
    prepared_tasks = {}
    for row in task_rows:
        key_value = row.get("key") if isinstance(row, dict) else None
        if not isinstance(key_value, dict):
            raise ValueError("extension prepared inputs contain a malformed task")
        key = (str(key_value.get("group")), str(key_value.get("question_id")))
        if key in prepared_tasks or not isinstance(row.get("schema_ref"), str):
            raise ValueError("extension prepared inputs contain duplicate or malformed tasks")
        prepared_tasks[key] = row
    if set(prepared_tasks) != expected:
        raise ValueError("extension prepared tasks and source question sets differ")
    source_schema_refs = {(str(row["group"]), str(row["question_id"])): row.get("schema_ref")
                          for row in source["questions"]}
    if any(prepared_tasks[key]["schema_ref"] != source_schema_refs[key]
           or prepared_tasks[key]["schema_ref"] not in metadata for key in expected):
        raise ValueError("extension prepared schema bindings differ from the sealed base")
    if prepared.get("source") != source_lineage:
        raise ValueError("extension prepared source lineage differs from its manifest")
    verification = {}
    attempt_identity = {}
    with LinkingRecords(extension, manifest, read_only=True) as records:
        actual_attempt_keys = {records.key(version) for version in records.rows}
        expected_task_keys = {TaskKey(group, question_id) for group, question_id in expected}
        if not actual_attempt_keys <= expected_task_keys:
            raise ValueError("extension contains attempts for unknown questions")
        for row in records.rows.values():
            if row.get("stage") != "din_linking_question":
                raise ValueError("extension contains a non-Linking attempt")
            attempt_identity[row["attempt_id"]] = (str(row["group"]), str(row["item_key"]))
        for group, store in records.stores.items():
            report = dict(store.verify())
            if report.get("ok") is not True:
                raise ValueError(f"extension RunStore verification failed: {group}")
            verification[group] = report
    snapshots = {
        group: _group_snapshot(extension, group, [str(value) for value in specification["ids"]], requests=True)
        for group, specification in groups.items()
    }
    current: dict[tuple[str, str], dict[str, Any]] = {}
    for group, specification in groups.items():
        snapshot = snapshots[group]
        versions = {str(row["question_id"]): row for row in snapshot["versions"]}
        nodes: dict[str, list[dict[str, Any]]] = {str(value): [] for value in specification["ids"]}
        requests: dict[str, list[dict[str, Any]]] = {str(value): [] for value in specification["ids"]}
        for row in snapshot["nodes"]:
            nodes[str(row["question_id"])].append(row)
        for row in snapshot["requests"]:
            if row.get("kind") not in _SAFE_REQUEST_KINDS:
                raise ValueError("unsafe request payload selected from extension")
            requests[str(row["question_id"])].append(row)
        for question_id in map(str, specification["ids"]):
            row = versions[question_id]
            if row.get("version_id") is None or row.get("state") not in ("succeeded", "failed"):
                raise ValueError(f"extension question is not terminal: {group}/{question_id}")
            question_nodes = nodes[question_id]
            if len(question_nodes) != len(EXTENSION_NODES) or \
                    {value.get("node") for value in question_nodes} != set(EXTENSION_NODES):
                raise ValueError(f"extension question does not have exactly two nodes: {group}/{question_id}")
            parent = row.get("parent_version_id")
            if parent is not None and attempt_identity.get(parent) != (group, question_id):
                raise ValueError(f"extension parent attempt mismatch: {group}/{question_id}")
            current[(group, question_id)] = {"version": row, "nodes": question_nodes,
                                             "requests": requests[question_id]}
    return {"root": extension, "manifest": manifest, "prepared": prepared,
            "prepared_tasks": prepared_tasks, "snapshots": snapshots,
            "current": current, "verification": verification,
            "manifest_sha256": _sha_file(extension / "manifest.json"),
            "prepared_sha256": _sha_file(prepared_path)}


def _task_key(group: str, question_id: str) -> str:
    prefixes = {
        "bird_dev": "bird/dev/i:", "spider_dev": "spider/dev/i:",
        "spider_test": "spider/test/i:",
        "bird_interact_full": "bird_interact/full/s:",
        "bird_interact_lite": "bird_interact/lite/s:",
    }
    try:
        return prefixes[group] + question_id
    except KeyError as exc:
        raise ValueError(f"no annotation identity mapping for group {group!r}") from exc


def _load_annotations(path: Path, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl(path)
    by_task = {}
    for row in rows:
        key = row.get("task_key")
        if not isinstance(key, str) or key in by_task:
            raise ValueError("gold annotations have missing or duplicate task keys")
        by_task[key] = row
    expected = {_task_key(group, question_id): (group, question_id) for group, question_id in keys}
    if set(by_task) != set(expected):
        raise ValueError("gold annotation and source question sets differ")
    result = {}
    for task_key, key in expected.items():
        row = by_task[task_key]
        if row.get("status") == "resolved":
            tables, columns = row.get("required_tables"), row.get("required_columns")
            if not isinstance(tables, list) or not all(isinstance(value, str) for value in tables):
                raise ValueError(f"gold annotation has invalid tables: {task_key}")
            if not isinstance(columns, list) or any(
                    not isinstance(value, list) or len(value) != 2 or
                    not all(isinstance(item, str) for item in value) for value in columns):
                raise ValueError(f"gold annotation has invalid columns: {task_key}")
        result[key] = row
    return result


def _catalog_from_metadata(metadata: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(metadata, list):
        raise ValueError("extension metadata entry is not a table list")
    catalog = {}
    folded_tables = set()
    for table in metadata:
        raw_name = table.get("table_name") if isinstance(table, dict) else None
        name = raw_name.strip() if isinstance(raw_name, str) else None
        columns = table.get("columns") if isinstance(table, dict) else None
        if not isinstance(name, str) or not name or not isinstance(columns, list) or \
                name.casefold() in folded_tables:
            raise ValueError("extension metadata contains a malformed or duplicate table")
        values, folded_columns = [], set()
        for column in columns:
            raw_value = (column.get("original_column_name", column.get("column_name"))
                         if isinstance(column, dict) else None)
            value = raw_value.strip() if isinstance(raw_value, str) else None
            if not isinstance(value, str) or not value or value.casefold() in folded_columns:
                raise ValueError(f"extension metadata contains a malformed column in {name!r}")
            values.append(value)
            folded_columns.add(value.casefold())
        catalog[name] = tuple(values)
        folded_tables.add(name.casefold())
    if not catalog:
        raise ValueError("extension metadata has an empty table catalog")
    return catalog


def _catalogs(source: Mapping[str, Any],
              extension: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, tuple[str, ...]]]:
    metadata = extension["prepared"].get("metadata")
    cache: dict[str, dict[str, tuple[str, ...]]] = {}
    result = {}
    for question in source["questions"]:
        key = (str(question["group"]), str(question["question_id"]))
        ref = question.get("schema_ref")
        if ref not in cache:
            if ref not in metadata:
                raise ValueError(f"question references unknown extension metadata: {ref}")
            cache[ref] = _catalog_from_metadata(metadata[ref])
        result[key] = cache[ref]
    return result


def _filtered_catalog(result: Any, full: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(result, dict) or not isinstance(result.get("filtered_metadata"), list):
        raise ValueError("schema_filter_rc3 returned malformed filtered metadata")
    table_lookup = {name.casefold(): name for name in full}
    output = {}
    for table in result["filtered_metadata"]:
        if not isinstance(table, dict) or not isinstance(table.get("table_name"), str) or \
                not isinstance(table.get("columns"), list):
            raise ValueError("schema_filter_rc3 returned malformed filtered metadata")
        table_name = table["table_name"].strip()
        canonical_table = table_lookup.get(table_name.casefold()) if table_name else None
        if canonical_table is None or canonical_table in output:
            raise ValueError("schema_filter_rc3 returned a duplicate or unknown table")
        column_lookup = {name.casefold(): name for name in full[canonical_table]}
        columns = []
        for row in table["columns"]:
            if not isinstance(row, dict):
                raise ValueError("schema_filter_rc3 returned a malformed column")
            raw_name = row.get("original_column_name", row.get("column_name"))
            name = raw_name.strip() if isinstance(raw_name, str) else None
            canonical = column_lookup.get(name.casefold()) if name else None
            if canonical is None or canonical in columns:
                raise ValueError("schema_filter_rc3 returned a duplicate or unknown column")
            columns.append(canonical)
        output[canonical_table] = tuple(columns)
    return output


def _identifier_pattern(name: str) -> str:
    variants = (
        name,
        '"' + name.replace('"', '""') + '"',
        "`" + name.replace("`", "``") + "`",
        "[" + name.replace("]", "]]") + "]",
    )
    return "(?:" + "|".join(re.escape(value) for value in dict.fromkeys(variants)) + ")"


def _prediction(result: Any, catalog: Mapping[str, Sequence[str]]) -> tuple[set[str], set[tuple[str, str]]]:
    """Resolve DIN schema-link strings against the supplied expansion domain."""
    if not isinstance(result, list):
        return set(), set()
    texts = [value for value in result if isinstance(value, str)]
    tables: set[str] = set()
    columns: set[tuple[str, str]] = set()
    for table, names in catalog.items():
        table_pattern = _identifier_pattern(table)
        exact_table = re.compile(rf"^\s*{table_pattern}\s*$", re.IGNORECASE)
        wildcard = re.compile(rf"(?<![\w]){table_pattern}\s*\.\s*\*(?![\w])", re.IGNORECASE)
        if any(exact_table.search(text) for text in texts):
            tables.add(table)
        if any(wildcard.search(text) for text in texts):
            tables.add(table)
            columns.update((table, name) for name in names)
        for name in names:
            reference = re.compile(
                rf"(?<![\w]){table_pattern}\s*\.\s*{_identifier_pattern(name)}(?![\w])",
                re.IGNORECASE,
            )
            if any(reference.search(text) for text in texts):
                tables.add(table)
                columns.add((table, name))
    return tables, columns


def _canonical_gold(annotation: Mapping[str, Any], catalog: Mapping[str, Sequence[str]]) -> \
        tuple[set[str], set[tuple[str, str]]]:
    if annotation.get("status") != "resolved":
        return set(), set()
    table_lookup = {table.casefold(): table for table in catalog}
    tables = set()
    columns = set()
    for raw in annotation["required_tables"]:
        table = table_lookup.get(raw.casefold())
        if table is None:
            raise ValueError(f"gold annotation names unknown table {raw!r}")
        tables.add(table)
    for raw_table, raw_column in annotation["required_columns"]:
        table = table_lookup.get(raw_table.casefold())
        column_lookup = ({name.casefold(): name for name in catalog[table]}
                         if table is not None else {})
        column = column_lookup.get(raw_column.casefold())
        if table is None or column is None:
            raise ValueError(f"gold annotation names unknown column {raw_table}.{raw_column}")
        tables.add(table)
        columns.add((table, column))
    return tables, columns


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _set_metrics(pairs: Sequence[tuple[set[Any], set[Any]]]) -> dict[str, Any]:
    totals = Counter(tp=0, fp=0, fn=0)
    precision_values, recall_values, f1_values = [], [], []
    exact = full = full_eligible = 0
    for gold, prediction in pairs:
        tp, fp, fn = len(gold & prediction), len(prediction - gold), len(gold - prediction)
        totals.update(tp=tp, fp=fp, fn=fn)
        precision = None if not gold and not prediction else (_ratio(tp, tp + fp) or 0.0)
        recall = _ratio(tp, tp + fn)
        question_f1 = _f1(tp, fp, fn)
        if precision is not None:
            precision_values.append(precision)
        if recall is not None:
            recall_values.append(recall)
        if question_f1 is not None:
            f1_values.append(question_f1)
        exact += gold == prediction
        if gold:
            full_eligible += 1
            full += not (gold - prediction)
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    mean = lambda values: (_ratio(sum(values), len(values)), len(values))
    macro_p, macro_p_n = mean(precision_values)
    macro_r, macro_r_n = mean(recall_values)
    macro_f, macro_f_n = mean(f1_values)
    return {
        "questions": len(pairs), "tp": tp, "fp": fp, "fn": fn,
        "micro": {"precision": (None if tp + fp + fn == 0 else _ratio(tp, tp + fp) or 0.0),
                  "recall": _ratio(tp, tp + fn), "f1": _f1(tp, fp, fn)},
        "macro": {"precision": macro_p, "precision_questions": macro_p_n,
                  "recall": macro_r, "recall_questions": macro_r_n,
                  "f1": macro_f, "f1_questions": macro_f_n},
        "exact_set": {"count": exact, "ratio": _ratio(exact, len(pairs))},
        "full_recall": {"count": full, "eligible": full_eligible,
                        "ratio": _ratio(full, full_eligible)},
    }


def _usage(value: Any) -> dict[str, int | float | None]:
    value = value if isinstance(value, dict) else {}
    result = {}
    aliases = {"prompt_tokens": ("prompt_tokens", "input_tokens"),
               "completion_tokens": ("completion_tokens", "output_tokens"),
               "total_tokens": ("total_tokens",)}
    for target, names in aliases.items():
        selected = next((value.get(name) for name in names if name in value), None)
        result[target] = (selected if isinstance(selected, (int, float)) and not isinstance(selected, bool)
                          and math.isfinite(selected) and selected >= 0 else None)
    if result["total_tokens"] is None and result["prompt_tokens"] is not None and \
            result["completion_tokens"] is not None:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def _combine_usage(*values: Mapping[str, int | float | None]) -> dict[str, int | float | None]:
    return {field: (sum(value[field] for value in values) if values and
                    all(value.get(field) is not None for value in values) else None)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _token_summary(details: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, Any]:
    values = []
    for detail in details:
        value: Any = detail
        for part in path:
            value = value.get(part) if isinstance(value, Mapping) else None
        values.append(value if isinstance(value, Mapping) else {})
    result = {"questions": len(values)}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        known = [value[field] for value in values if value.get(field) is not None]
        result[field] = {"sum": sum(known), "known_questions": len(known),
                         "unknown_questions": len(values) - len(known)}
    return result


def _paired(details: Sequence[Mapping[str, Any]], level: str) -> dict[str, Any]:
    counter = Counter()
    for row in details:
        base, rc3 = row["base"][level + "_recall"], row["rc3"][level + "_recall"]
        if base is None or rc3 is None:
            continue
        counter["improvements" if rc3 > base else "regressions" if rc3 < base else "unchanged"] += 1
    eligible = sum(counter.values())
    return {"basis": "per_question_recall", "eligible": eligible,
            "improvements": counter["improvements"], "regressions": counter["regressions"],
            "unchanged": counter["unchanged"]}


def _summarize(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in details if row["annotation_status"] == "resolved"]
    result = {"questions": len(details), "resolved_annotations": len(resolved),
              "primary_metric": "column_macro_recall"}
    for condition in ("base", "rc3"):
        result[condition] = {
            "tables": _set_metrics([(set(row["gold"]["tables"]),
                                      set(row[condition]["tables"]))
                                     for row in resolved]),
            "columns": _set_metrics([(set(map(tuple, row["gold"]["columns"])),
                                       set(map(tuple, row[condition]["columns"])))
                                      for row in resolved]),
        }
    result["paired_tables"] = _paired(resolved, "table")
    result["paired_columns"] = _paired(resolved, "column")
    filter_pairs_tables = [(set(row["gold"]["tables"]),
                            set(row["filter"]["tables"]))
                           for row in resolved]
    filter_pairs_columns = [(set(map(tuple, row["gold"]["columns"])),
                             set(map(tuple, row["filter"]["columns"])))
                            for row in resolved]
    filter_tables, filter_columns = _set_metrics(filter_pairs_tables), _set_metrics(filter_pairs_columns)
    result["filter"] = {
        "tables": filter_tables, "columns": filter_columns,
        "table_macro_recall": filter_tables["macro"]["recall"],
        "table_macro_recall_questions": filter_tables["macro"]["recall_questions"],
        "column_macro_recall": filter_columns["macro"]["recall"],
        "column_macro_recall_questions": filter_columns["macro"]["recall_questions"],
    }
    table_reductions = [row["filter"]["table_reduction"] for row in details
                        if row["filter"]["table_reduction"] is not None]
    column_reductions = [row["filter"]["column_reduction"] for row in details
                         if row["filter"]["column_reduction"] is not None]
    result["schema_reduction"] = {
        "macro_table_reduction": _ratio(sum(table_reductions), len(table_reductions)),
        "table_questions": len(table_reductions),
        "macro_column_reduction": _ratio(sum(column_reductions), len(column_reductions)),
        "column_questions": len(column_reductions),
    }
    result["tokens"] = {
        "base_linking": _token_summary(details, ("tokens", "base_linking")),
        "schema_filter_rc3": _token_summary(details, ("tokens", "schema_filter_rc3")),
        "linking_rc3": _token_summary(details, ("tokens", "linking_rc3")),
        "rc3_combined": _token_summary(details, ("tokens", "rc3_combined")),
    }
    return result


def _details(source: Mapping[str, Any], extension: Mapping[str, Any],
             annotations: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    catalogs = _catalogs(source, extension)
    values = []
    for group, question_id in source["keys"]:
        key = (group, question_id)
        full = catalogs[key]
        annotation = annotations[key]
        gold_tables, gold_columns = _canonical_gold(annotation, full)
        base_node = next(row for row in source["node_map"][key] if row["node"] == "linking")
        extension_nodes = {row["node"]: row for row in extension["current"][key]["nodes"]}
        filter_node, rc_node = extension_nodes["schema_filter_rc3"], extension_nodes["linking_rc3"]
        filtered = (_filtered_catalog(filter_node.get("result"), full)
                    if filter_node.get("status") == "succeeded" else {})
        base_tables, base_columns = _prediction(
            base_node.get("result") if base_node.get("status") == "succeeded" else None, full)
        rc_tables, rc_columns = _prediction(
            rc_node.get("result") if rc_node.get("status") == "succeeded" else None, filtered)
        filtered_tables = set(filtered)
        filtered_columns = {(table, column) for table, columns in filtered.items() for column in columns}
        recall = lambda gold, predicted: _ratio(len(gold & predicted), len(gold))
        base_usage, filter_usage, rc_usage = (_usage(base_node.get("usage")),
                                               _usage(filter_node.get("usage")),
                                               _usage(rc_node.get("usage")))
        full_columns = sum(len(columns) for columns in full.values())
        filtered_column_count = len(filtered_columns)
        values.append({
            "group": group, "question_id": question_id,
            "version_id": source["version_map"][key]["version_id"],
            "extension_attempt_id": extension["current"][key]["version"]["version_id"],
            "task_key": annotation["task_key"], "annotation_status": annotation.get("status"),
            "gold": {"tables": sorted(gold_tables),
                     "columns": [list(value) for value in sorted(gold_columns)]},
            "base": {"status": base_node.get("status"), "tables": sorted(base_tables),
                     "columns": [list(value) for value in sorted(base_columns)],
                     "table_recall": recall(gold_tables, base_tables),
                     "column_recall": recall(gold_columns, base_columns)},
            "rc3": {"status": rc_node.get("status"), "tables": sorted(rc_tables),
                    "columns": [list(value) for value in sorted(rc_columns)],
                    "table_recall": recall(gold_tables, rc_tables),
                    "column_recall": recall(gold_columns, rc_columns)},
            "filter": {"status": filter_node.get("status"), "tables": sorted(filtered_tables),
                       "columns": [list(value) for value in sorted(filtered_columns)],
                       "table_recall": recall(gold_tables, filtered_tables),
                       "column_recall": recall(gold_columns, filtered_columns),
                       "full_tables": len(full), "filtered_tables": len(filtered),
                       "full_columns": full_columns, "filtered_columns": filtered_column_count,
                       "table_reduction": (_ratio(len(full) - len(filtered), len(full))
                                           if filter_node.get("status") == "succeeded" else None),
                       "column_reduction": (_ratio(full_columns - filtered_column_count, full_columns)
                                            if filter_node.get("status") == "succeeded" else None)},
            "tokens": {"base_linking": base_usage, "schema_filter_rc3": filter_usage,
                       "linking_rc3": rc_usage,
                       "rc3_combined": _combine_usage(filter_usage, rc_usage)},
        })
    return values


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _linking_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["", "## RS schema filtering and DIN Linking", "",
             "Primary metric: per-question column recall averaged over questions with at least one gold column.", "",
             "| Group | Questions | Eligible gold-column questions | Base column macro recall | RS column macro recall | Delta | Base table macro recall | RS table macro recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = [(group, value) for group, value in summary["groups"].items()] + [("Overall", summary["overall"])]
    for group, value in rows:
        base_column = value["base"]["columns"]["macro"]
        rc_column = value["rc3"]["columns"]["macro"]
        delta = (None if base_column["recall"] is None or rc_column["recall"] is None
                 else rc_column["recall"] - base_column["recall"])
        lines.append(
            f"| {group} | {value['questions']} | {base_column['recall_questions']} | "
            f"{_number(base_column['recall'])} | {_number(rc_column['recall'])} | {_number(delta)} | "
            f"{_number(value['base']['tables']['macro']['recall'])} | "
            f"{_number(value['rc3']['tables']['macro']['recall'])} |"
        )
    lines += ["", "`table.*` is expanded against the full catalog for Base and against the filtered catalog for RS. Empty-gold column questions are excluded from column macro recall; failed or empty predictions score zero when gold is non-empty.", ""]
    return "\n".join(lines)


def _extension_node(row: Mapping[str, Any], base_version: str) -> dict[str, Any]:
    event = {**row["refs"]["event"], "store": "linking_extension"}
    refs = {**row["refs"], "event": event,
            "lineage": {"store": "linking_extension", "attempt_id": event["attempt_id"]}}
    return {**row, "version_id": base_version, "origin": row.get("origin") or "linking_extension",
            "refs": refs, "extension_attempt_id": event["attempt_id"]}


def _extension_request(row: Mapping[str, Any], base_version: str) -> dict[str, Any]:
    ref = {**row["ref"], "store": "linking_extension"}
    return {**row, "version_id": base_version, "ref": ref,
            "extension_attempt_id": ref["attempt_id"]}


def _update_records(stage: Path, source: Mapping[str, Any], extension: Mapping[str, Any]) -> None:
    records = stage / "records"
    old_nodes = (source["root"] / "records/nodes.jsonl").read_bytes()
    additions = []
    extension_requests = []
    extension_failed = []
    version_rows = []
    for old in source["versions"]:
        key = (str(old["group"]), str(old["question_id"]))
        current = extension["current"][key]
        extension_nodes = [_extension_node(row, old["version_id"]) for row in current["nodes"]]
        additions.extend(extension_nodes)
        extension_requests.extend(_extension_request(row, old["version_id"])
                                  for row in current["requests"])
        extension_version = current["version"]
        merged_refs = dict(old.get("node_refs", {}))
        merged_refs.update({row["node"]: row["refs"]["event"] for row in extension_nodes})
        base_seal = {
            "version_id": old.get("version_id"), "state": old.get("state"),
            "attempt_checksum": old.get("attempt_checksum"),
            "finish_checksum": old.get("finish_checksum"),
            "node_refs": old.get("node_refs", {}), "request_events": old.get("request_events", 0),
            "source_record_checksum": _sha_bytes(_json_payload(old)),
        }
        extension_lineage = {
            "attempt_id": extension_version["version_id"], "state": extension_version["state"],
            "attempt_no": extension_version.get("attempt_no"),
            "parent_extension_attempt_id": extension_version.get("parent_version_id"),
            "attempt_checksum": extension_version.get("attempt_checksum"),
            "finish_checksum": extension_version.get("finish_checksum"),
            "node_refs": {row["node"]: row["refs"]["event"] for row in extension_nodes},
            "request_events": len(current["requests"]),
            "snapshot": f"raw_records/group-{key[0]}/linking_extension.sqlite3",
        }
        updated = {**old, "record_format": RECORD_FORMAT, "node_refs": merged_refs,
                   "request_events": old.get("request_events", 0) + len(current["requests"]),
                   "source_seals": {"base": base_seal},
                   "extension_lineage": extension_lineage}
        updated["unified_record_checksum"] = _sha_bytes(_json_payload({
            "version_id": old["version_id"], "source_seals": {"base": base_seal},
            "extension_lineage": extension_lineage, "node_refs": merged_refs,
        }))
        version_rows.append(updated)
        if extension_version["state"] == "failed":
            extension_failed.append({"group": key[0], "question_id": key[1],
                                     "version_id": old["version_id"], "state": "failed",
                                     "scope": "linking_extension",
                                     "extension_attempt_id": extension_version["version_id"]})
    _atomic_write(records / "nodes.jsonl", old_nodes + b"".join(_json_line(row) for row in additions))
    old_requests = (source["root"] / "records/requests.jsonl").read_bytes()
    _atomic_write(records / "requests.jsonl", old_requests +
                  b"".join(_json_line(row) for row in extension_requests))
    _write_jsonl(records / "versions.jsonl", version_rows)
    old_failed = _read_jsonl(source["root"] / "records/failed_questions.jsonl")
    _write_jsonl(records / "failed_questions.jsonl", [*old_failed, *extension_failed])
    summary = read_json(source["root"] / "records/summary.json")
    extension_states = Counter(value["version"]["state"] for value in extension["current"].values())
    summary.update({"format": RECORD_FORMAT,
                    "nodes": len(source["nodes"]) + len(additions),
                    "request_events": summary.get("request_events", 0) + len(extension_requests),
                    "extension_nodes": len(additions),
                    "extension_request_events": len(extension_requests),
                    "extension_states": dict(sorted(extension_states.items()))})
    for group, value in summary.get("groups", {}).items():
        states = Counter(extension["current"][(group, str(question_id))]["version"]["state"]
                         for question_id in extension["manifest"]["groups"][group]["ids"])
        value["extension_states"] = dict(sorted(states.items()))
    _atomic_json(records / "summary.json", summary)
    names = ("questions.jsonl", "versions.jsonl", "nodes.jsonl", "requests.jsonl",
             "failed_questions.jsonl", "summary.json")
    node_rows = {(group, question): [] for group, question in source["keys"]}
    for row in _read_jsonl(records / "nodes.jsonl"):
        node_rows[(str(row["group"]), str(row["question_id"]))].append(row)
    eight_ok = all(len(rows) == len(BASE_NODES) + len(EXTENSION_NODES) and
                   {row["node"] for row in rows} == set(BASE_NODES) | set(EXTENSION_NODES)
                   for rows in node_rows.values())
    versions_ok = all(all(row.get("version_id") == source["version_map"][key]["version_id"]
                          for row in rows) for key, rows in node_rows.items())
    prefix_ok = (records / "nodes.jsonl").read_bytes().startswith(old_nodes)
    verification = {
        "format": RECORD_FORMAT, "ok": eight_ok and versions_ok and prefix_ok,
        "manifest_questions": len(source["keys"]), "exported_questions": len(node_rows),
        "current_versions": len(version_rows), "eight_node_refs_ok": eight_ok,
        "shared_base_version_ids_ok": versions_ok,
        "old_nodes": {"rows": len(source["nodes"]),
                      "sha256": _sha_file(source["root"] / "records/nodes.jsonl"),
                      "byte_prefix_preserved": prefix_ok},
        "source_record_verification_sha256": _sha_file(source["root"] / "records/verification.json"),
        "extension_source_event_checksums": {
            group: {"count": len(snapshot["event_checksums"]),
                    "sha256": _sha_bytes(_json_payload(snapshot["event_checksums"]))}
            for group, snapshot in extension["snapshots"].items()},
        "files": {name: {"bytes": (records / name).stat().st_size,
                          "sha256": _sha_file(records / name)} for name in names},
    }
    _atomic_json(records / "verification.json", verification)


def _update_evaluation(stage: Path, source: Mapping[str, Any], extension: Mapping[str, Any],
                       annotations: Mapping[tuple[str, str], Mapping[str, Any]],
                       annotation_path: Path) -> None:
    evaluation = stage / "evaluation"
    details = _details(source, extension, annotations)
    _write_jsonl(evaluation / "linking_details.jsonl", details)
    groups = {}
    for group in extension["manifest"]["groups"]:
        groups[group] = _summarize([row for row in details if row["group"] == group])
    linking = {"primary_metric": "column_macro_recall", "overall": _summarize(details),
               "groups": groups,
               "semantics": {"empty_gold_columns": "excluded_from_column_macro_recall",
                             "failed_or_empty_prediction": "zero_recall_when_gold_nonempty",
                             "base_wildcard_domain": "full_catalog",
                             "rc3_wildcard_domain": "filtered_catalog"}}
    summary = read_json(source["root"] / "evaluation/summary.json")
    base_evaluation_format = summary.get("format", "din-compact-evaluation-v1")
    summary["format"] = EVALUATION_FORMAT
    summary["base_evaluation_format"] = base_evaluation_format
    summary["linking"] = linking
    for group, value in groups.items():
        summary.setdefault("groups", {}).setdefault(group, {})["linking"] = value
        summary.setdefault(group, {})["linking"] = value
    _atomic_json(evaluation / "summary.json", summary)
    old_tables = (source["root"] / "evaluation/tables.md").read_text(encoding="utf-8").rstrip()
    _atomic_write(evaluation / "tables.md", (old_tables + "\n" + _linking_markdown(linking)).encode("utf-8"))
    versions = read_json(source["root"] / "evaluation/versions.json")
    versions["format"] = EVALUATION_FORMAT
    versions["base_evaluation_format"] = "din-compact-evaluation-v1"
    versions["linking_extension"] = {
        "manifest_sha256": extension["manifest_sha256"],
        "annotations_sha256": _sha_file(annotation_path),
        "details_sha256": _sha_file(evaluation / "linking_details.jsonl"),
        "nodes": list(EXTENSION_NODES),
    }
    current = extension["current"]
    for row in versions.get("versions", []):
        item = current[(str(row["group"]), str(row["question_id"]))]
        row["linking_extension_attempt_id"] = item["version"]["version_id"]
        row["linking_extension_state"] = item["version"]["state"]
    _atomic_json(evaluation / "versions.json", versions)


def _update_guide(stage: Path, source: Mapping[str, Any], guide: Path | None) -> str:
    if guide is not None:
        guide = guide.resolve(strict=True)
        relative = Path(guide.name)
        custom_preamble = guide.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        relative = _safe_relative(source["index"].get("guide") or "guide.md")
        # The source guide describes the sealed six-node v1 package.  Reusing
        # it verbatim and appending an amendment leaves mutually contradictory
        # instructions (v1/six nodes versus v2/eight nodes).  The unified
        # export therefore writes one self-contained v2 guide at the same
        # indexed path.  An explicitly supplied guide is treated only as a
        # caller-owned preamble.
        custom_preamble = ""
    if relative in {Path("index.json"), Path("COMPLETE.json")}:
        raise ValueError("guide path collides with a root handoff seal")
    text = custom_preamble + """# DIN-SQL × Qwen3.8 2.4T：五组 RS 统一记录解析指南

本目录是一个已经封印的统一交付包。导出格式为 `din-handoff-v2`，记录
格式为 `din-current-records-v2`，评估格式为
`din-compact-evaluation-v2`。它把原 DIN 流程与后来补跑的 RS schema
filtering + DIN Linking 对照合并成每题一个版本、八个节点；分析时不需要
把它当成两次实验，也不需要另行连接第二套题目记录。

本文使用论文术语 Result Specification（RS）。实际代码、文件名与记录
标识符保留历史 RC 命名，例如 `generation_rc3`、`schema_filter_rc3`
和 `rc3`；解析记录时应使用这些原始名称。

## 1. 建议读取顺序

1. 读取根目录 `index.json`，不要根据目录名猜测文件位置。
2. 校验 `COMPLETE.json.index_sha256` 与 `index.json` 文件的 SHA-256。
3. 校验 `index.json.files` 中每个文件的 `bytes` 与 `sha256`。
4. 确认 `records/verification.json` 中 `ok=true`、
   `eight_node_refs_ok=true`、`shared_base_version_ids_ok=true`，且
   `old_nodes.byte_prefix_preserved=true`。
5. 日常分析优先读取 `records/*.jsonl`、`evaluation/summary.json`、
   `evaluation/tables.md` 与 `evaluation/linking_details.jsonl`。
6. 只有追查完整 prompt、provider response 或请求重试时，才读取
   `raw_records/group-*/run.sqlite3` 与
   `raw_records/group-*/linking_extension.sqlite3`。

题目主键始终是 `(group, question_id)`，其中 `question_id` 按字符串处理。

## 2. Unified RS Linking extension

### Eight-node unified view

`records/nodes.jsonl` 是唯一的逻辑节点流。每道题恰好有以下八个节点，
并共享同一个 base `version_id`：

- `linking`：原生 DIN Schema Linking；
- `decomposition`：原生难度/分解节点；
- `generation_base`、`generation_rc3`：Generation 原生与 RS 对照；
- `revision_base`、`revision_rc3`：Revision 原生与 RS 对照；
- `schema_filter_rc3`：根据 question、evidence、RS 和完整公开 metadata
  生成过滤后的 schema；
- `linking_rc3`：把过滤后的 schema 交给原 DIN Linking prompt 得到的
  对照结果。RS 文本不会直接进入该 Linking prompt。

`records/versions.jsonl` 每题一行。`source_seals.base` 保存原六节点版本的
封印，`extension_lineage` 保存两个新增节点的独立来源，
`unified_record_checksum` 标识合并后的统一视图。不要把旧
`finish_checksum` 解释为它曾封印新增节点。

`records/failed_questions.jsonl` 可能同时包含原 DIN 复合流程失败与 Linking
扩展失败。`scope=linking_extension` 的行只表示新增两节点没有完整成功；
它不是 SQL 答错清单。

## 3. 紧凑记录

- `records/questions.jsonl`：每题身份、问题、evidence、数据库绑定、RS、
  schema 引用和当前版本。
- `records/versions.jsonl`：当前统一版本、基础封印与扩展 lineage。
- `records/nodes.jsonl`：八节点解析结果、状态、usage 与事件引用。
- `records/requests.jsonl`：安全的请求事件元数据，不含完整 prompt 和
  provider body；同一次模型调用会对应排队、发送、结果等多类事件，
  不可直接用文件行数作为调用次数。
- `records/summary.json`：全局及分组题数、状态和记录规模。
- `records/verification.json`：八节点结构、版本关系、旧节点字节前缀及
  文件哈希校验。

节点状态可能是 `succeeded`、`failed` 或 `dependency_failed`。失败记录
必须保留为失败，不能在统计时自动补成空结果或错误答案。

## 4. 评估结果

原 Generation/Revision 的 SQL 执行评估仍在
`evaluation/evaluation.sqlite3`、`summary.json` 与 `tables.md`。比较口径
要求列数和列位置一致，忽略行顺序并保留重复行数量。

Linking 的逐题统一事实位于 `evaluation/linking_details.jsonl`：

- `gold`：gold SQL 辅助标注出的必需表和列；
- `base`：原生 DIN Linking 解析结果及逐题 recall；
- `filter`：RS schema filtering 保留的表列、recall 与 schema 缩减比例；
- `rc3`：基于过滤 schema 的 DIN Linking 结果及逐题 recall；
- `tokens`：原生 Linking、filter、RS Linking 和 RS 两节点合计用量。

汇总结果位于 `evaluation/summary.json` 的 `linking` 字段以及
`evaluation/tables.md`。主指标是 column-level macro
recall：先计算每题列召回率，再对 gold 至少含一列的题目取平均。
gold 列为空的题不进入该指标；gold 非空而预测失败或为空时，该题召回率
记为 0。原生结果中的 `table.*` 在完整 catalog 上展开，RS 结果中的
`table.*` 只在该题过滤后的 catalog 上展开。

## 5. Raw audit material

`raw_records/` 同时保留两类独立、只读的 SQLite 快照：

- `group-*/run.sqlite3`：原 DIN 六节点原始事件；
- `group-*/linking_extension.sqlite3`：RS filtering 与 Linking 的原始
  追加事件。

扩展批次的冻结输入和审计信息平铺为
`linking_extension_manifest.json`、
`linking_extension_prepared_inputs.json`、
`linking_extension_monitoring_*.json`。这些文件是追溯材料，不是第二套
需要人工拼接的分析结果。

## 6. Integrity verification

只有当根目录 `COMPLETE.json` 有效、`index.json.complete=true`、
`index.json.acceptance` 全部为真、所有索引文件哈希匹配且
`records/verification.json.ok=true` 时，才应把目录视为完整交付包。
各组的两类 SQLite 快照应以只读方式打开，并可运行
`PRAGMA quick_check`。

所有正式路径、文件大小、SHA-256、题目数和失败状态均以本目录自身的
`index.json`、`records/summary.json`、`records/verification.json` 和
`evaluation/summary.json` 为准。
"""
    _atomic_write(stage / relative, text.encode("utf-8"))
    return str(relative)


def _copy_extension_audit_files(stage: Path, extension: Mapping[str, Any]) -> dict[str, str]:
    """Place extension audit inputs flat in the existing raw_records directory."""
    raw = stage / "raw_records"
    copied = {
        "manifest": "raw_records/linking_extension_manifest.json",
        "prepared_inputs": "raw_records/linking_extension_prepared_inputs.json",
    }
    _copy_regular(extension["root"] / "manifest.json", stage / copied["manifest"])
    _copy_regular(extension["root"] / "prepared/inputs.json", stage / copied["prepared_inputs"])
    for name in ("implementation.json", "endpoint.json", "live.json"):
        source = extension["root"] / "monitoring" / name
        if source.is_file() and not source.is_symlink():
            key = "monitoring_" + name.removesuffix(".json")
            relative = f"raw_records/linking_extension_monitoring_{name}"
            _copy_regular(source, stage / relative)
            copied[key] = relative
    return copied


def _build_index(stage: Path, source: Mapping[str, Any], extension: Mapping[str, Any],
                 annotations: Path, guide_name: str,
                 extension_audit_files: Mapping[str, str]) -> dict[str, Any]:
    record_verification = read_json(stage / "records/verification.json")
    details = _read_jsonl(stage / "evaluation/linking_details.jsonl")
    acceptance = {
        "source_complete_and_authenticated": True,
        "extension_runstores_ok": all(value.get("ok") is True
                                       for value in extension["verification"].values()),
        "question_sets_equal": len(details) == len(source["keys"]),
        "eight_nodes_per_question": record_verification.get("eight_node_refs_ok") is True,
        "old_node_prefix_preserved": record_verification.get("old_nodes", {}).get("byte_prefix_preserved") is True,
        "record_verification_ok": record_verification.get("ok") is True,
    }
    files = []
    for path in sorted(stage.rglob("*")):
        relative = path.relative_to(stage)
        if path.is_file() and relative not in {Path("index.json"), Path("COMPLETE.json")}:
            files.append(_file_info(path, stage))
    return {
        "format": HANDOFF_FORMAT, "complete": all(acceptance.values()),
        "batch_id": source["index"].get("batch_id"),
        "manifest_questions": len(source["keys"]), "current_questions": len(source["keys"]),
        "files": files, "raw_records_directory": "raw_records",
        "record_directory": "records", "evaluation_directory": "evaluation",
        "guide": guide_name, "record_summary": "records/summary.json",
        "record_verification": "records/verification.json",
        "evaluation_summary": "evaluation/summary.json",
        "evaluation_versions": "evaluation/versions.json",
        "evaluation_progress": source["index"].get("evaluation_progress"),
        "evaluation_tables": "evaluation/tables.md",
        "evaluation_database": source["index"].get("evaluation_database"),
        "linking_details": "evaluation/linking_details.jsonl",
        "acceptance": acceptance,
        "source_seal": {"format": source["index"]["format"],
                        "index_sha256": source["index_sha256"],
                        "complete_sha256": source["complete_sha256"]},
        "extension_lineage": {"format": extension["manifest"]["format"],
                              "batch_id": extension["manifest"].get("batch_id"),
                              "manifest_sha256": extension["manifest_sha256"],
                              "prepared_sha256": extension["prepared_sha256"],
                              "annotations_sha256": _sha_file(annotations),
                              "audit_files": dict(extension_audit_files),
                              "snapshots": {group: f"raw_records/group-{group}/linking_extension.sqlite3"
                                            for group in extension["manifest"]["groups"]}},
        "source_runstore_verification": source["index"].get("source_runstore_verification"),
        "extension_runstore_verification": extension["verification"],
        "source_boundary": "authenticated base handoff plus independently sealed per-group Linking snapshots",
    }


def _replace_output(stage: Path, output: Path) -> None:
    prefix = "." + output.name + ".old-"
    for stale in output.parent.glob(prefix + "*"):
        if stale.is_dir() and not stale.is_symlink():
            try:
                shutil.rmtree(stale)
            except OSError:
                pass
    backup = None
    try:
        if output.exists():
            backup = output.parent / (prefix + next(tempfile._get_candidate_names()))
            os.replace(output, backup)
        os.replace(stage, output)
    except BaseException:
        if not output.exists() and backup is not None and backup.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError:
            # Publication is already complete.  Retired-output cleanup is
            # housekeeping and must not turn a successful publication into a
            # reported failure; a later publication retries stale backups.
            pass


def _verify_output_boundary(stage: Path, index: Mapping[str, Any]) -> None:
    """Require every non-seal output payload to be authenticated by the index."""
    listed = {Path(row["path"]): row for row in index.get("files", [])}
    if len(listed) != len(index.get("files", [])):
        raise ValueError("new handoff index repeats a payload path")
    actual = {path.relative_to(stage) for path in stage.rglob("*") if path.is_file()
              and path.relative_to(stage) not in {Path("index.json"), Path("COMPLETE.json")}}
    if set(listed) != actual:
        raise ValueError("new handoff index does not cover the output payload boundary")
    for relative, expected in listed.items():
        path = stage / relative
        if path.is_symlink() or not path.is_file() or \
                path.stat().st_size != expected.get("bytes") or \
                _sha_file(path) != expected.get("sha256"):
            raise ValueError(f"new handoff indexed payload mismatch: {relative}")


def export_unified_handoff(source_handoff: str | Path, extension_batch: str | Path,
                           annotations_jsonl: str | Path, output: str | Path,
                           guide: str | Path | None = None) -> Path:
    """Create one authenticated v2 handoff containing all eight DIN nodes."""
    source_path = Path(source_handoff).resolve(strict=True)
    extension_path = Path(extension_batch).resolve(strict=True)
    annotation_path = Path(annotations_jsonl).resolve(strict=True)
    output = Path(output).resolve()
    if output == source_path or source_path in output.parents or output in source_path.parents:
        raise ValueError("output must be independent of the immutable source handoff")
    if output == extension_path or extension_path in output.parents or output in extension_path.parents:
        raise ValueError("output must be independent of the extension batch")
    source = _validate_source(source_path)
    extension = _load_extension(extension_path, source)
    annotations = _load_annotations(annotation_path, source["keys"])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".build-", dir=output.parent))
    try:
        for relative in source["files"]:
            _copy_regular(source_path / relative, stage / relative)
        _verify_copied_source(stage, source)
        extension_audit_files = _copy_extension_audit_files(stage, extension)
        for group in extension["manifest"]["groups"]:
            destination = stage / f"raw_records/group-{group}/linking_extension.sqlite3"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_sqlite(extension_path / f"group-{group}/run.sqlite3", destination)
        _update_records(stage, source, extension)
        _update_evaluation(stage, source, extension, annotations, annotation_path)
        guide_name = _update_guide(stage, source, Path(guide) if guide is not None else None)
        index = _build_index(stage, source, extension, annotation_path, guide_name,
                             extension_audit_files)
        _atomic_json(stage / "index.json", index)
        if not index["complete"]:
            raise ValueError("unified handoff acceptance checks failed")
        _atomic_json(stage / "COMPLETE.json", {
            "complete": True, "index_sha256": _sha_file(stage / "index.json")})
        _verify_output_boundary(stage, index)
        _replace_output(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output
