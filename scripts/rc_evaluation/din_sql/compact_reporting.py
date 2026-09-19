"""Read-only DIN current-record export, compact SQL evaluation, and handoff.

The source batch is never mutated.  Record export selects the greatest sealed
attempt number per manifest question (a later unfinished attempt cannot hide
it), reads the five group databases independently, and deliberately never
selects ``node_input`` or ``request_result`` payloads.  SQL observations are
stored once as typed JSON compressed with zlib in a resumable SQLite database.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from typing import Any
import zlib

from scripts.baseline_adapters.dail_sql.execution import execute_sql
from scripts.baseline_adapters.deepeye.run_store import RunStore, restore_jsonable, to_jsonable
from scripts.baseline_adapters.din_sql.inputs import DinSettings, NODES, OUTPUT_NODES, TaskKey, digest
from scripts.baseline_adapters.din_sql.records import read_json
from .evaluation import evaluate_pair, normalize_usage, summarize_stage, summary_markdown


RECORD_FORMAT = "din-current-records-v1"
EVALUATION_FORMAT = "din-compact-evaluation-v1"
REQUEST_KINDS = ("attempt_queued", "request_attempt", "request_dispatch", "request_outcome")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_info(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": _sha(raw)}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_line(stream, value: Any) -> None:
    stream.write(_json_bytes(value) + b"\n")


def _record_checksum(record: dict[str, Any]) -> str:
    return _sha(_json_bytes(record))


def _checked_payload(row: sqlite3.Row, label: str) -> Any:
    text = row["payload_json"]
    if _sha(text.encode("utf-8")) != row["payload_checksum"]:
        raise ValueError(f"payload checksum mismatch for {label}")
    return restore_jsonable(json.loads(text))


def _check_attempt(row: sqlite3.Row) -> None:
    record = {name: row[name] for name in
              ("attempt_id", "item_key", "stage", "input_fingerprint", "attempt_no", "started_at")}
    if _record_checksum(record) != row["attempt_checksum"]:
        raise ValueError(f"attempt checksum mismatch for {row['attempt_id']}")
    if row["status"] is not None:
        finish = {"attempt_id": row["attempt_id"], "status": row["status"],
                  "payload_json": row["finish_payload_json"],
                  "payload_checksum": row["finish_payload_checksum"],
                  "finished_at": row["finished_at"]}
        if _record_checksum(finish) != row["finish_checksum"]:
            raise ValueError(f"finish checksum mismatch for {row['attempt_id']}")
        if _sha(row["finish_payload_json"].encode("utf-8")) != row["finish_payload_checksum"]:
            raise ValueError(f"finish payload checksum mismatch for {row['attempt_id']}")


def _check_event(row: sqlite3.Row) -> Any:
    record = {name: row[name] for name in
              ("attempt_id", "event_no", "kind", "payload_json", "payload_checksum", "created_at")}
    if _record_checksum(record) != row["record_checksum"]:
        raise ValueError(f"event checksum mismatch for {row['attempt_id']}:{row['event_no']}")
    return _checked_payload(row, f"event {row['attempt_id']}:{row['event_no']}")


def _connect_group(batch: Path, group: str) -> sqlite3.Connection:
    database = batch / f"group-{group}" / "run.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True,
                                 isolation_level=None, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        connection.close()
        raise ValueError(f"source RunStore quick_check failed: {group}")
    return connection


_ATTEMPT_SQL = """
SELECT a.attempt_id,a.item_key,a.stage,a.input_fingerprint,a.attempt_no,a.started_at,
       a.record_checksum AS attempt_checksum,
       f.status,f.payload_json AS finish_payload_json,
       f.payload_checksum AS finish_payload_checksum,f.finished_at,
       f.record_checksum AS finish_checksum
FROM attempts a LEFT JOIN finishes f ON f.attempt_id=a.attempt_id
ORDER BY a.item_key,a.attempt_no
"""


def _group_snapshot(batch: Path, group: str, question_ids: list[str], *, requests: bool) -> dict[str, Any]:
    """Read one group on its own connection; excluded event kinds are never selected."""
    connection = _connect_group(batch, group)
    try:
        connection.execute("BEGIN")
        finished: dict[str, sqlite3.Row] = {}
        pending: dict[str, sqlite3.Row] = {}
        for row in connection.execute(_ATTEMPT_SQL):
            _check_attempt(row)
            target = finished if row["status"] in ("succeeded", "failed") else pending
            previous = target.get(row["item_key"])
            if previous is None or previous["attempt_no"] < row["attempt_no"]:
                target[row["item_key"]] = row
        current_ids = {row["attempt_id"] for row in finished.values()}
        events: dict[str, list[tuple[sqlite3.Row, Any]]] = {version: [] for version in current_ids}
        if current_ids:
            kinds = ("question_start", "node_result") + (REQUEST_KINDS if requests else ())
            placeholders = ",".join("?" for _ in current_ids)
            kind_places = ",".join("?" for _ in kinds)
            sql = (f"SELECT event_id,attempt_id,event_no,kind,payload_json,payload_checksum,created_at,record_checksum "
                   f"FROM events WHERE attempt_id IN ({placeholders}) AND kind IN ({kind_places}) "
                   "ORDER BY event_id")
            cursor = connection.execute(sql, (*sorted(current_ids), *kinds))
            while True:
                rows = cursor.fetchmany(256)
                if not rows:
                    break
                for row in rows:
                    events[row["attempt_id"]].append((row, _check_event(row)))
        result = {"versions": [], "nodes": [], "requests": [], "failed": [],
                  "event_checksums": [], "current": {}}
        for question_id in question_ids:
            row = finished.get(question_id)
            waiting = pending.get(question_id)
            if row is None:
                state = "pending" if waiting is not None else "missing"
                result["versions"].append({"group": group, "question_id": question_id,
                    "version_id": None, "state": state,
                    "pending_version_id": waiting["attempt_id"] if waiting is not None else None,
                    "pending_attempt_no": waiting["attempt_no"] if waiting is not None else None})
                result["current"][question_id] = {"version_id": None, "state": state, "nodes": {}}
                continue
            version_id = row["attempt_id"]
            node_values: dict[str, dict[str, Any]] = {}
            parent_version = None
            request_count = 0
            for event, payload in events[version_id]:
                ref = {"group": group, "attempt_id": version_id, "event_no": event["event_no"]}
                result["event_checksums"].append(event["record_checksum"])
                if event["kind"] == "question_start":
                    parent_version = payload.get("parent_version")
                elif event["kind"] == "node_result":
                    node = payload["node"]
                    refs = {"event": ref, "parent_refs": payload.get("parent_refs", {}),
                            "response_ref": payload.get("response_ref"),
                            "source_refs": payload.get("source_refs", {})}
                    compact = {"group": group, "question_id": question_id, "version_id": version_id,
                               "node": node, "result": payload.get("result"),
                               "status": payload.get("status"), "origin": payload.get("origin"),
                               "usage": payload.get("usage"), "reason": payload.get("reason"),
                               "fallback_used": payload.get("fallback_used", False), "refs": refs,
                               "input_fingerprint": payload.get("input_fingerprint"),
                               "event_checksum": event["record_checksum"]}
                    result["nodes"].append(compact)
                    node_values[node] = compact
                elif event["kind"] in REQUEST_KINDS:
                    common = {name: payload.get(name) for name in
                              ("node", "batch_id", "group", "question_id", "round_execution_id",
                               "sample_position", "request_id", "attempt_no") if name in payload}
                    if event["kind"] == "request_dispatch":
                        common["telemetry"] = payload.get("telemetry")
                    elif event["kind"] == "request_outcome":
                        common.update({name: payload.get(name) for name in
                                      ("status", "response_ref", "usage", "response_model", "error")})
                    result["requests"].append({"group": group, "question_id": question_id,
                        "version_id": version_id, "kind": event["kind"], "metadata": common,
                        "ref": ref, "event_checksum": event["record_checksum"]})
                    request_count += 1
            version = {"group": group, "question_id": question_id, "version_id": version_id,
                       "state": row["status"], "attempt_no": row["attempt_no"],
                       "started_at": row["started_at"], "finished_at": row["finished_at"],
                       "parent_version_id": parent_version,
                       "attempt_checksum": row["attempt_checksum"],
                       "finish_checksum": row["finish_checksum"],
                       "node_refs": {node: value["refs"]["event"] for node, value in node_values.items()},
                       "request_events": request_count}
            result["versions"].append(version)
            result["current"][question_id] = {"version_id": version_id, "state": row["status"],
                                               "nodes": node_values}
            if row["status"] == "failed":
                result["failed"].append({"group": group, "question_id": question_id,
                                          "version_id": version_id, "state": "failed"})
        connection.commit()
        return result
    finally:
        connection.close()


def _manifest_keys(manifest: dict[str, Any]) -> list[TaskKey]:
    keys = [TaskKey(group, str(question_id)) for group, spec in manifest["groups"].items()
            for question_id in spec["ids"]]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate questions")
    return keys


def _prepared_tasks(batch: Path) -> dict[TaskKey, dict[str, Any]]:
    data = read_json(batch / "prepared" / "inputs.json")
    result = {}
    for row in data["tasks"]:
        key = TaskKey(**row["key"])
        result[key] = {"group": key.group, "question_id": key.question_id,
                       **{name: row.get(name) for name in
                          ("question", "evidence", "database", "schema_ref", "rc3", "label", "source_refs")}}
    return result


def _snapshots(batch: Path, manifest: dict[str, Any], *, requests: bool) -> dict[str, dict[str, Any]]:
    groups = list(manifest["groups"])
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        futures = {group: pool.submit(_group_snapshot, batch, group,
                    [str(value) for value in manifest["groups"][group]["ids"]], requests=requests)
                   for group in groups}
        return {group: futures[group].result() for group in groups}


def export_records(batch: Path, output: Path) -> Path:
    """Export compact current records without reading prompts/provider bodies."""
    batch, output = Path(batch), Path(output)
    manifest = read_json(batch / "manifest.json")
    keys = _manifest_keys(manifest)
    tasks = _prepared_tasks(batch)
    if set(tasks) != set(keys):
        raise ValueError("prepared and manifest question sets differ")
    snapshots = _snapshots(batch, manifest, requests=True)
    output.mkdir(parents=True, exist_ok=True)
    values = {
        "questions.jsonl": [tasks[key] | {"version_id": snapshots[key.group]["current"][key.question_id]["version_id"]}
                            for key in keys],
        "versions.jsonl": [row for group in manifest["groups"] for row in snapshots[group]["versions"]],
        "nodes.jsonl": [row for group in manifest["groups"] for row in snapshots[group]["nodes"]],
        "requests.jsonl": [row for group in manifest["groups"] for row in snapshots[group]["requests"]],
        "failed_questions.jsonl": [row for group in manifest["groups"] for row in snapshots[group]["failed"]],
    }
    for name, rows in values.items():
        temporary = output / ("." + name + ".tmp")
        with temporary.open("wb") as stream:
            for row in rows:
                _write_line(stream, row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / name)
    states: dict[str, int] = {}
    for row in values["versions.jsonl"]:
        states[row["state"]] = states.get(row["state"], 0) + 1
    summary = {"format": RECORD_FORMAT, "batch_id": manifest["batch_id"],
               "manifest_questions": len(keys), "groups": {
                   group: {"questions": len(manifest["groups"][group]["ids"]),
                           "states": {state: sum(row["group"] == group and row["state"] == state
                                                for row in values["versions.jsonl"])
                                      for state in ("succeeded", "failed", "pending", "missing")}}
                   for group in manifest["groups"]},
               "states": states, "nodes": len(values["nodes.jsonl"]),
               "request_events": len(values["requests.jsonl"]),
               "complete_questions": states.get("succeeded", 0) + states.get("failed", 0)}
    _atomic_json(output / "summary.json", summary)
    identity = lambda collection: _sha(_json_bytes(sorted([asdict(key) for key in collection],
                                                           key=lambda row: (row["group"], row["question_id"]))))
    exported_keys = {TaskKey(row["group"], row["question_id"]) for row in values["versions.jsonl"]}
    node_sets = {(group, question): {row["node"] for row in values["nodes.jsonl"]
                                     if row["group"] == group and row["question_id"] == question}
                 for group, question in ((key.group, key.question_id) for key in keys)}
    current = [row for row in values["versions.jsonl"] if row["version_id"] is not None]
    six_nodes_ok = all(node_sets[(row["group"], row["question_id"])] == set(NODES) for row in current)
    files = {}
    for name in (*values, "summary.json"):
        info = _file_info(output / name, output)
        files[name] = {"bytes": info["bytes"], "sha256": info["sha256"]}
    verification = {"format": RECORD_FORMAT, "ok": set(keys) == exported_keys and six_nodes_ok,
                    "manifest_questions": len(keys), "exported_questions": len(exported_keys),
                    "manifest_question_set_sha256": identity(keys),
                    "exported_question_set_sha256": identity(exported_keys),
                    "current_versions": len(current), "six_node_refs_ok": six_nodes_ok,
                    "source_event_checksums": {
                        group: {"count": len(snapshots[group]["event_checksums"]),
                                "sha256": _sha(_json_bytes(snapshots[group]["event_checksums"]))}
                        for group in manifest["groups"]}, "files": files}
    _atomic_json(output / "verification.json", verification)
    return output


def _query_identity(key: TaskKey, database: dict[str, Any], sql: str) -> tuple[str, str]:
    if database["dialect"] == "sqlite":
        scope = {"dialect": "sqlite", "path": str(Path(database["path"]).resolve(strict=True)),
                 "database_id": database.get("database_id")}
    elif database["dialect"] == "postgresql":
        scope = {"dialect": "postgresql", "group": key.group, "question_id": key.question_id,
                 "database_id": database.get("database_id")}
    else:
        raise ValueError("unsupported evaluation dialect")
    scope_json = _json_bytes(scope).decode("utf-8")
    return _sha(_json_bytes([scope_json, sql])), scope_json


class EvaluationStore:
    def __init__(self, path: Path, metadata: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=120, isolation_level=None,
                                          check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS queries(
                cache_key TEXT PRIMARY KEY,scope_json TEXT NOT NULL,sql TEXT NOT NULL,status TEXT NOT NULL,
                result BLOB NOT NULL,raw_sha256 TEXT NOT NULL,raw_bytes INTEGER NOT NULL,
                compressed_bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS items(
                grp TEXT NOT NULL,question TEXT NOT NULL,version_id TEXT,state TEXT NOT NULL,
                payload_json TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(grp,question));
        """)
        frozen = _json_bytes(metadata).decode("utf-8")
        row = self.connection.execute("SELECT value FROM metadata WHERE key='frozen'").fetchone()
        if row is None:
            self.connection.execute("INSERT INTO metadata VALUES('frozen',?)", (frozen,))
        elif row[0] != frozen:
            raise ValueError("existing evaluation has different batch metadata or DIN settings")

    def close(self) -> None:
        with self.lock:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.connection.close()

    def get_query(self, cache_key: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT result,raw_sha256 FROM queries WHERE cache_key=?", (cache_key,)).fetchone()
        if row is None:
            return None
        raw = zlib.decompress(row[0])
        if _sha(raw) != row[1]:
            raise ValueError("compact query checksum mismatch")
        return restore_jsonable(json.loads(raw))

    def put_query(self, cache_key: str, scope_json: str, sql: str, result: dict[str, Any]) -> None:
        raw = _json_bytes(result)
        compressed = zlib.compress(raw, 6)
        with self.lock:
            existing = self.connection.execute(
                "SELECT raw_sha256 FROM queries WHERE cache_key=?", (cache_key,)).fetchone()
            if existing is not None:
                if existing[0] != _sha(raw):
                    raise ValueError("conflicting SQL observation for cache identity")
                return
            self.connection.execute("INSERT INTO queries VALUES(?,?,?,?,?,?,?,?)",
                (cache_key, scope_json, sql, result.get("status", "unknown"), compressed,
                 _sha(raw), len(raw), len(compressed)))

    def item(self, key: TaskKey) -> tuple[str | None, str] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT version_id,state,payload_json,sha256 FROM items WHERE grp=? AND question=?",
                (key.group, key.question_id)).fetchone()
        if row is None:
            return None
        if _sha(row[2].encode("utf-8")) != row[3]:
            raise ValueError(f"compact item checksum mismatch: {key.group}/{key.question_id}")
        return row[0], row[1]

    def put_item(self, key: TaskKey, version_id: str | None, state: str, payload: dict[str, Any]) -> None:
        raw = _json_bytes(payload)
        with self.lock:
            self.connection.execute("INSERT OR REPLACE INTO items VALUES(?,?,?,?,?,?)",
                (key.group, key.question_id, version_id, state, raw.decode("utf-8"), _sha(raw)))

    def payloads(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = list(self.connection.execute("SELECT grp,question,payload_json,sha256 "
                                                "FROM items ORDER BY grp,question"))
        values = []
        for group, question, text, checksum in rows:
            if _sha(text.encode("utf-8")) != checksum:
                raise ValueError(f"compact item checksum mismatch: {group}/{question}")
            values.append(restore_jsonable(json.loads(text)))
        return values

    def counts(self) -> dict[str, Any]:
        with self.lock:
            return {"items": self.connection.execute("SELECT count(*) FROM items").fetchone()[0],
                    "queries": self.connection.execute("SELECT count(*) FROM queries").fetchone()[0],
                    "query_statuses": dict(self.connection.execute(
                        "SELECT status,count(*) FROM queries GROUP BY status")),
                    "raw_bytes": self.connection.execute(
                        "SELECT coalesce(sum(raw_bytes),0) FROM queries").fetchone()[0],
                    "compressed_bytes": self.connection.execute(
                        "SELECT coalesce(sum(compressed_bytes),0) FROM queries").fetchone()[0]}

    def quick_check(self) -> None:
        with self.lock:
            result = self.connection.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise ValueError("evaluation SQLite quick_check failed: " + str(result))


def _evaluation_item(key: TaskKey, current: dict[str, Any], references: dict[str, str],
                     results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    version_id, nodes = current["version_id"], current["nodes"]
    stages = {}
    gold = results[references["gold"]]
    for stage in ("generation", "revision"):
        names = (stage + "_base", stage + "_rc3")
        executions = [results[references[name]] if name in references else {"status": "no_sql"}
                      for name in names]
        value = evaluate_pair(executions[0], executions[1], gold)
        value["execution_refs"] = {name: references.get(name) for name in names}
        value["gold_ref"] = references["gold"]
        for prefix, name in zip(("base", "rc"), names):
            usage = normalize_usage(nodes.get(name, {}).get("usage")
                                    if nodes.get(name, {}).get("status") == "succeeded" else None)
            value.update({prefix + "_" + field: amount for field, amount in usage.items()})
        stages[stage] = value
    compact_nodes = {node: {name: nodes.get(node, {}).get(name) for name in
                            ("status", "result", "usage", "origin", "reason", "fallback_used", "refs")}
                     for node in NODES}
    return {"group": key.group, "question_id": key.question_id, "version_id": version_id,
            "state": "evaluated", "stages": stages, "nodes": compact_nodes}


def _execute_queries_bounded(store: EvaluationStore,
                             query_specs: dict[str, tuple[TaskKey, dict[str, Any], str, str]],
                             settings: DinSettings) -> None:
    """Persist each observation promptly while retaining at most N futures."""
    pending = iter(query_specs.items())
    exhausted = False
    with ThreadPoolExecutor(max_workers=settings.sql_workers) as pool:
        inflight = {}

        def fill() -> None:
            nonlocal exhausted
            while not exhausted and len(inflight) < settings.sql_workers:
                try:
                    cache_key, (_key, database, sql, scope) = next(pending)
                except StopIteration:
                    exhausted = True
                    return
                if store.get_query(cache_key) is not None:
                    continue
                future = pool.submit(execute_sql, database, sql,
                                     timeout_seconds=settings.sql_timeout_seconds)
                inflight[future] = (cache_key, scope, sql)

        fill()
        while inflight:
            done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
            for future in done:
                cache_key, scope, sql = inflight.pop(future)
                store.put_query(cache_key, scope, sql, future.result())
            fill()


def evaluate_compact(batch: Path, output: Path) -> Path:
    """Evaluate one frozen current-version map using the batch's DIN SQL limits."""
    batch, output = Path(batch), Path(output)
    manifest = read_json(batch / "manifest.json")
    keys = _manifest_keys(manifest)
    prepared = read_json(batch / "prepared" / "inputs.json")
    settings = DinSettings(**manifest.get("settings", {}))
    output.mkdir(parents=True, exist_ok=True)
    snapshots = _snapshots(batch, manifest, requests=False)
    current = {key: snapshots[key.group]["current"][key.question_id] for key in keys}
    evaluation = prepared["evaluation"]
    metadata = {"format": EVALUATION_FORMAT,
                "manifest_sha256": _sha((batch / "manifest.json").read_bytes()),
                "settings": {"sql_workers": settings.sql_workers,
                             "sql_timeout_seconds": settings.sql_timeout_seconds},
                "comparison": "DIN evaluate_pair/compare_results"}
    store = EvaluationStore(output / "evaluation.sqlite3", metadata)
    progress = {"phase": "sql_evaluation", "groups": {
        group: {"total": len(spec["ids"]), "completed": 0} for group, spec in manifest["groups"].items()}}
    try:
        query_specs: dict[str, tuple[TaskKey, dict[str, Any], str, str]] = {}
        references: dict[TaskKey, dict[str, str]] = {}
        for key in keys:
            value = current[key]
            prior = store.item(key)
            if prior is not None and prior[0] == value["version_id"]:
                progress["groups"][key.group]["completed"] += 1
                continue
            if value["version_id"] is None:
                store.put_item(key, None, value["state"], {"group": key.group,
                    "question_id": key.question_id, "version_id": None, "state": value["state"],
                    "stages": {}, "nodes": {}})
                progress["groups"][key.group]["completed"] += 1
                continue
            binding = evaluation[f"{key.group}/{key.question_id}"]
            refs = {}
            sqls = {"gold": binding["gold_sql"]}
            sqls.update({node: data["result"] for node, data in value["nodes"].items()
                         if node in OUTPUT_NODES and data.get("status") == "succeeded"
                         and isinstance(data.get("result"), str)})
            for name, sql in sqls.items():
                cache_key, scope = _query_identity(key, binding["database"], sql)
                refs[name] = cache_key
                query_specs.setdefault(cache_key, (key, binding["database"], sql, scope))
            references[key] = refs
        _atomic_json(output / "progress.json", progress)

        _execute_queries_bounded(store, query_specs, settings)
        for key in keys:
            value = current[key]
            prior = store.item(key)
            if value["version_id"] is None or prior is not None and prior[0] == value["version_id"]:
                continue
            refs = references[key]
            # Keep decompressed query data question-local: gold plus at most
            # four successful output-node observations, released after insert.
            results = {cache_key: store.get_query(cache_key) for cache_key in set(refs.values())}
            if any(result is None for result in results.values()):
                raise ValueError(f"missing cached SQL observation for {key.group}/{key.question_id}")
            payload = _evaluation_item(key, value, refs, results)
            store.put_item(key, value["version_id"], "evaluated", payload)
            progress["groups"][key.group]["completed"] += 1
            _atomic_json(output / "progress.json", progress)
        payloads = store.payloads()
        summary = {}
        for group, specification in manifest["groups"].items():
            summary[group] = {}
            for stage in ("generation", "revision"):
                rows = [{"group": item["group"], "question_id": item["question_id"],
                         **item["stages"][stage]}
                        for item in payloads if item["group"] == group and item["state"] == "evaluated"]
                summary[group][stage] = {"total_questions": len(specification["ids"]),
                                         **summarize_stage(rows)}
        _atomic_json(output / "summary.json", {"format": EVALUATION_FORMAT,
            **summary, "groups": summary, "storage": store.counts(),
            "timeouts_are_retried": False,
            "comparison": "exact columns/positions, duplicate-preserving row bag"})
        (output / "tables.md").write_text(summary_markdown(summary), encoding="utf-8")
        versions = [{"group": key.group, "question_id": key.question_id,
                     "version_id": current[key]["version_id"], "state": current[key]["state"]}
                    for key in keys]
        _atomic_json(output / "versions.json", {**metadata, "versions": versions,
            "settings": {"sql_workers": settings.sql_workers,
                         "sql_timeout_seconds": settings.sql_timeout_seconds}})
        progress["phase"] = "complete"
        _atomic_json(output / "progress.json", progress)
        store.quick_check()
    finally:
        store.close()
    with sqlite3.connect((output / "evaluation.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as check:
        if check.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("final evaluation SQLite quick_check failed")
    return output


def _copy_file(source: Path, destination: Path) -> None:
    """Atomically copy a regular file, preferring macOS clonefile semantics."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"handoff source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / ("." + destination.name + ".copy-" + next(tempfile._get_candidate_names()))
    try:
        cloned = False
        if os.uname().sysname == "Darwin":
            try:
                completed = subprocess.run(["/bin/cp", "-c", str(source), str(temporary)],
                                           check=False, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                cloned = completed.returncode == 0
            except OSError:
                cloned = False
        if not cloned:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    """Create a standalone database containing any committed WAL pages."""
    source_uri = source.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as reader:
        if reader.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"source SQLite quick_check failed: {source}")
        temporary_dir = Path(tempfile.mkdtemp(prefix=".sqlite-snapshot-", dir=destination.parent))
        snapshot = temporary_dir / "run.sqlite3"
        try:
            with sqlite3.connect(snapshot) as writer:
                reader.backup(writer)
            with sqlite3.connect(snapshot) as check:
                if check.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError(f"snapshot SQLite quick_check failed: {source}")
            _copy_file(snapshot, destination)
        finally:
            shutil.rmtree(temporary_dir)


def export_handoff(batch: Path, output: Path, *, guide: Path | None = None) -> Path:
    """Build or resume an allowlisted handoff tree from a read-only DIN batch."""
    batch, output = Path(batch), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "COMPLETE.json"
    if marker.exists():
        marker.unlink()
    manifest = read_json(batch / "manifest.json")
    raw = output / "raw_records"
    raw.mkdir(exist_ok=True)
    allowed = ["manifest.json", "prepared/inputs.json", "prepared/import_report.json",
               "monitoring/implementation.json", "monitoring/endpoint.json", "monitoring/live.json"]
    expected_raw: set[Path] = set()
    source_verification = {}
    for group in manifest["groups"]:
        identity = {"format": "din-sql-v1", "group": group,
                    "batch_fingerprint": digest(manifest)}
        with RunStore.open(batch / f"group-{group}", expected_manifest=identity,
                           read_only=True) as store:
            verification = dict(store.verify())
        if not verification["ok"]:
            raise ValueError(f"source RunStore verification failed: {group}")
        source_verification[group] = verification
    for relative in allowed:
        source = batch / relative
        if source.is_file():
            destination = raw / relative
            _copy_file(source, destination)
            expected_raw.add(destination.relative_to(raw))
    for group in manifest["groups"]:
        destination = raw / f"group-{group}" / "run.sqlite3"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _snapshot_sqlite(batch / f"group-{group}" / "run.sqlite3", destination)
        expected_raw.add(destination.relative_to(raw))
    # A resumed handoff remains an allowlisted package even if an earlier or
    # manually altered output directory contains logs, locks, WALs, or reports.
    for path in sorted(raw.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        relative = path.relative_to(raw)
        if path.is_symlink() or path.is_file():
            if relative not in expected_raw:
                path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    export_records(batch, output / "records")
    evaluate_compact(batch, output / "evaluation")
    guide_relative = None
    if guide is not None:
        guide = Path(guide)
        guide_relative = guide.name
        _copy_file(guide, output / guide_relative)
    summary = read_json(output / "records" / "summary.json")
    record_verification = read_json(output / "records" / "verification.json")
    evaluation_versions = read_json(output / "evaluation" / "versions.json")
    evaluation_progress = read_json(output / "evaluation" / "progress.json")
    expected_questions = {(key.group, key.question_id) for key in _manifest_keys(manifest)}
    version_rows = evaluation_versions.get("versions", [])
    version_questions = {(row["group"], row["question_id"]) for row in version_rows}
    evaluation_database = output / "evaluation" / "evaluation.sqlite3"
    with sqlite3.connect(evaluation_database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        evaluation_quick_check_ok = connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        item_questions = set(connection.execute("SELECT grp,question FROM items"))
    progress_groups = evaluation_progress.get("groups", {})
    progress_complete = (evaluation_progress.get("phase") == "complete" and
                         set(progress_groups) == set(manifest["groups"]) and
                         all(value.get("completed") == value.get("total")
                             for value in progress_groups.values()))
    acceptance = {
        "source_runstores_ok": all(value["ok"] for value in source_verification.values()),
        "current_questions_complete": summary["complete_questions"] == summary["manifest_questions"],
        "record_verification_ok": record_verification.get("ok") is True,
        "evaluation_items_complete": item_questions == expected_questions,
        "evaluation_versions_complete": (len(version_rows) == len(expected_questions) and
                                         version_questions == expected_questions),
        "evaluation_progress_complete": progress_complete,
        "evaluation_quick_check_ok": evaluation_quick_check_ok,
    }
    complete = all(acceptance.values())
    files = [_file_info(path, output) for path in sorted(output.rglob("*"))
             if path.is_file() and path.name not in ("index.json", "COMPLETE.json")
             and "-wal" not in path.name and "-shm" not in path.name]
    index = {"format": "din-handoff-v1", "complete": complete,
             "batch_id": manifest["batch_id"], "manifest_questions": summary["manifest_questions"],
             "current_questions": summary["complete_questions"], "files": files,
             "raw_records_directory": "raw_records", "record_directory": "records",
             "evaluation_directory": "evaluation", "guide": guide_relative,
             "raw_manifest": "raw_records/manifest.json",
             "raw_prepared_inputs": "raw_records/prepared/inputs.json",
             "raw_import_report": "raw_records/prepared/import_report.json",
             "record_summary": "records/summary.json",
             "record_verification": "records/verification.json",
             "evaluation_summary": "evaluation/summary.json",
             "evaluation_versions": "evaluation/versions.json",
             "evaluation_progress": "evaluation/progress.json",
             "evaluation_tables": "evaluation/tables.md",
             "evaluation_database": "evaluation/evaluation.sqlite3",
             "acceptance": acceptance,
             "source_runstore_verification": source_verification,
             "source_boundary": "per-group independent SQLite snapshots; no cross-group transaction"}
    _atomic_json(output / "index.json", index)
    if complete:
        _atomic_json(marker, {"complete": True, "index_sha256": _sha((output / "index.json").read_bytes())})
    return output
