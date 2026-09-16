"""Append-only SQLite storage for frozen gold-SQL annotations.

The store deliberately records remote work as an attempt first, then its
terminal outcome, and only then an accepted normalized annotation.  SQLite
constraints and immutable-row triggers make that order auditable after a
crash or resume.
"""
from __future__ import annotations

from copy import deepcopy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping
import uuid


_SCHEMA_VERSION = 1
_TASK_FIELDS = frozenset({"task_key", "group", "dialect", "schema_sha256", "sql_sha256"})


def _canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically, rejecting non-portable values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _json_copy(value: Any) -> Any:
    """Return a detached JSON-compatible copy while checking serializability."""
    return json.loads(_canonical_json(value))


def _decode(text: str, checksum: str, label: str) -> Any:
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != checksum:
        raise ValueError(f"checksum mismatch for {label}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for {label}") from exc


_SCHEMA = """
CREATE TABLE schema_info (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
);
CREATE TABLE manifest (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    payload_json TEXT NOT NULL,
    payload_checksum TEXT NOT NULL
);
CREATE TABLE tasks (
    task_key TEXT PRIMARY KEY,
    cache_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_checksum TEXT NOT NULL,
    record_checksum TEXT NOT NULL
);
CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    batch_key TEXT NOT NULL,
    batch_json TEXT NOT NULL,
    batch_checksum TEXT NOT NULL,
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    started_at TEXT NOT NULL,
    record_checksum TEXT NOT NULL,
    UNIQUE (batch_key, attempt_no)
);
CREATE TABLE outcomes (
    attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    raw_response TEXT,
    usage_json TEXT,
    usage_checksum TEXT,
    latency_seconds REAL,
    endpoint_hash TEXT,
    error_json TEXT,
    error_checksum TEXT,
    finished_at TEXT NOT NULL,
    record_checksum TEXT NOT NULL
);
CREATE TABLE annotations (
    annotation_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    source_task_key TEXT NOT NULL REFERENCES tasks(task_key),
    cache_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_checksum TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    record_checksum TEXT NOT NULL,
    UNIQUE (cache_key)
);
CREATE INDEX tasks_cache_key ON tasks(cache_key, task_key);
CREATE INDEX attempts_batch_key ON attempts(batch_key, attempt_no DESC);
CREATE TRIGGER schema_info_no_update BEFORE UPDATE ON schema_info BEGIN SELECT RAISE(ABORT, 'schema_info is immutable'); END;
CREATE TRIGGER schema_info_no_delete BEFORE DELETE ON schema_info BEGIN SELECT RAISE(ABORT, 'schema_info is immutable'); END;
CREATE TRIGGER manifest_no_update BEFORE UPDATE ON manifest BEGIN SELECT RAISE(ABORT, 'manifest is immutable'); END;
CREATE TRIGGER manifest_no_delete BEFORE DELETE ON manifest BEGIN SELECT RAISE(ABORT, 'manifest is immutable'); END;
CREATE TRIGGER tasks_no_update BEFORE UPDATE ON tasks BEGIN SELECT RAISE(ABORT, 'tasks are immutable'); END;
CREATE TRIGGER tasks_no_delete BEFORE DELETE ON tasks BEGIN SELECT RAISE(ABORT, 'tasks are immutable'); END;
CREATE TRIGGER attempts_no_update BEFORE UPDATE ON attempts BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
CREATE TRIGGER attempts_no_delete BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
CREATE TRIGGER outcomes_no_update BEFORE UPDATE ON outcomes BEGIN SELECT RAISE(ABORT, 'outcomes are immutable'); END;
CREATE TRIGGER outcomes_no_delete BEFORE DELETE ON outcomes BEGIN SELECT RAISE(ABORT, 'outcomes are immutable'); END;
CREATE TRIGGER annotations_no_update BEFORE UPDATE ON annotations BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
CREATE TRIGGER annotations_no_delete BEFORE DELETE ON annotations BEGIN SELECT RAISE(ABORT, 'annotations are immutable'); END;
"""


class AnnotationStore:
    """A durable, append-only store for one frozen annotation manifest."""

    def __init__(self, path: Path, connection: sqlite3.Connection, manifest: dict[str, Any]) -> None:
        self.path = path
        self._connection = connection
        self._manifest = _json_copy(manifest)
        self._closed = False

    @classmethod
    def create(cls, path: str | Path, manifest: Mapping[str, Any]) -> "AnnotationStore":
        """Create a store, or safely reopen it when the manifest is identical."""
        database = cls._database_path(path)
        frozen_manifest, task_rows = cls._validated_manifest(manifest)
        if database.exists():
            return cls.open(database, frozen_manifest)
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = cls._connect(database)
        try:
            connection.executescript(_SCHEMA)
            manifest_json = _canonical_json(frozen_manifest)
            with connection:
                connection.execute("INSERT INTO schema_info(singleton, version) VALUES (1, ?)", (_SCHEMA_VERSION,))
                connection.execute(
                    "INSERT INTO manifest(singleton, payload_json, payload_checksum) VALUES (1, ?, ?)",
                    (manifest_json, _digest(frozen_manifest)),
                )
                for task in task_rows:
                    task_json = _canonical_json(task["payload"])
                    record = {
                        "task_key": task["task_key"], "cache_key": task["cache_key"],
                        "payload_json": task_json, "payload_checksum": _digest(task["payload"]),
                    }
                    connection.execute(
                        "INSERT INTO tasks(task_key, cache_key, payload_json, payload_checksum, record_checksum) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (*record.values(), _digest(record)),
                    )
            return cls(database, connection, frozen_manifest)
        except BaseException:
            connection.close()
            raise

    @classmethod
    def open(cls, path: str | Path, manifest: Mapping[str, Any] | None = None) -> "AnnotationStore":
        """Resume an existing store after verifying its immutable manifest."""
        database = cls._database_path(path)
        if not database.is_file():
            raise FileNotFoundError(database)
        connection = cls._connect(database)
        try:
            row = connection.execute("SELECT version FROM schema_info WHERE singleton = 1").fetchone()
            if row is None or row["version"] != _SCHEMA_VERSION:
                raise ValueError("unsupported AnnotationStore schema")
            row = connection.execute("SELECT payload_json, payload_checksum FROM manifest WHERE singleton = 1").fetchone()
            if row is None:
                raise ValueError("manifest is missing")
            stored_manifest = _decode(row["payload_json"], row["payload_checksum"], "manifest")
            if not isinstance(stored_manifest, dict):
                raise ValueError("manifest must be an object")
            cls._validated_manifest(stored_manifest)
            if manifest is not None:
                expected, _ = cls._validated_manifest(manifest)
                if _canonical_json(expected) != row["payload_json"]:
                    raise ValueError("manifest does not match the existing store")
            return cls(database, connection, stored_manifest)
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _database_path(path: str | Path) -> Path:
        resolved = Path(path)
        return resolved / "annotations.sqlite3" if resolved.is_dir() else resolved

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(journal_mode).lower() != "wal":
            connection.close()
            raise RuntimeError("SQLite WAL mode is required")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @classmethod
    def _validated_manifest(cls, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be an object")
        frozen = _json_copy(dict(manifest))
        model = frozen.get("model", frozen.get("model_name"))
        prompt_version = frozen.get("prompt_version")
        if not isinstance(model, str) or not model or not isinstance(prompt_version, str) or not prompt_version:
            raise ValueError("manifest requires non-empty model and prompt_version")
        tasks = frozen.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("manifest requires a non-empty tasks list")
        rows, seen = [], set()
        for raw_task in tasks:
            if not isinstance(raw_task, dict) or not _TASK_FIELDS <= set(raw_task):
                raise ValueError("each manifest task requires task_key, group, dialect, schema_sha256, and sql_sha256")
            task = {field: raw_task[field] for field in _TASK_FIELDS}
            if not all(isinstance(task[field], str) and task[field] for field in _TASK_FIELDS):
                raise ValueError("manifest task fields must be non-empty text")
            if task["task_key"] in seen:
                raise ValueError(f"duplicate manifest task: {task['task_key']}")
            if any(len(task[field]) != 64 or any(char not in "0123456789abcdef" for char in task[field].lower())
                   for field in ("schema_sha256", "sql_sha256")):
                raise ValueError("manifest task hashes must be SHA-256 hex")
            seen.add(task["task_key"])
            cache_key = _digest({
                "model": model, "prompt_version": prompt_version, "dialect": task["dialect"],
                "schema_sha256": task["schema_sha256"], "sql_sha256": task["sql_sha256"],
            })
            rows.append({"task_key": task["task_key"], "cache_key": cache_key, "payload": task})
        return frozen, rows

    @property
    def manifest(self) -> dict[str, Any]:
        self._assert_open()
        return deepcopy(self._manifest)

    def __enter__(self) -> "AnnotationStore":
        self._assert_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("AnnotationStore is closed")

    def _transaction(self) -> sqlite3.Connection:
        self._assert_open()
        self._connection.execute("BEGIN IMMEDIATE")
        return self._connection

    @staticmethod
    def _task_keys(batch: Iterable[Any]) -> tuple[str, ...]:
        if isinstance(batch, (str, bytes)):
            raise TypeError("batch must be an iterable of task keys or task objects")
        keys = []
        for item in batch:
            key = item if isinstance(item, str) else (
                item.get("task_key") if isinstance(item, Mapping) else getattr(item, "task_key", None)
            )
            if not isinstance(key, str) or not key:
                raise ValueError("batch task keys must be non-empty text")
            keys.append(key)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("batch must contain one or more distinct task keys")
        return tuple(sorted(keys))

    def start_attempt(self, batch: Iterable[Any]) -> dict[str, Any]:
        """Durably start one remote batch before it is sent to the model."""
        task_keys = self._task_keys(batch)
        batch_key = _digest(task_keys)
        batch_json = _canonical_json(list(task_keys))
        connection = self._transaction()
        try:
            placeholders = ",".join("?" for _ in task_keys)
            rows = connection.execute(
                f"SELECT task_key, cache_key FROM tasks WHERE task_key IN ({placeholders})", task_keys
            ).fetchall()
            if len(rows) != len(task_keys):
                found = {row["task_key"] for row in rows}
                missing = sorted(set(task_keys) - found)
                raise ValueError(f"batch contains task outside the frozen manifest: {', '.join(missing)}")
            cache_keys = {row["cache_key"] for row in rows}
            if len(cache_keys) != len(rows):
                raise ValueError("batch may contain only one representative per cache key")
            accepted = connection.execute(
                f"SELECT cache_key FROM annotations WHERE cache_key IN ({','.join('?' for _ in cache_keys)})", tuple(cache_keys)
            ).fetchone()
            if accepted is not None:
                raise ValueError("batch contains an already accepted annotation")
            row = connection.execute("SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_no FROM attempts WHERE batch_key = ?", (batch_key,)).fetchone()
            attempt_no = int(row["next_no"])
            attempt_id, started_at = str(uuid.uuid4()), _now()
            record = {
                "attempt_id": attempt_id, "batch_key": batch_key, "batch_json": batch_json,
                "batch_checksum": _digest(list(task_keys)), "attempt_no": attempt_no, "started_at": started_at,
            }
            connection.execute(
                "INSERT INTO attempts(attempt_id, batch_key, batch_json, batch_checksum, attempt_no, started_at, record_checksum) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (*record.values(), _digest(record)),
            )
            connection.commit()
            return {"attempt_id": attempt_id, "attempt_no": attempt_no, "task_keys": task_keys}
        except BaseException:
            connection.rollback()
            raise

    def finish_attempt(
        self,
        attempt_id: str,
        status: str,
        *,
        raw_response: str | None = None,
        usage: Any | None = None,
        latency_seconds: float | None = None,
        endpoint_hash: str | None = None,
        error: Any | None = None,
    ) -> None:
        """Append a terminal model outcome; it cannot be revised later."""
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt_id must be non-empty text")
        if status not in {"succeeded", "failed"}:
            raise ValueError("attempt status must be succeeded or failed")
        if raw_response is not None and not isinstance(raw_response, str):
            raise TypeError("raw_response must be text")
        if status == "succeeded" and raw_response is None:
            raise ValueError("a successful attempt requires its raw_response")
        if latency_seconds is not None and (not isinstance(latency_seconds, (int, float)) or latency_seconds < 0):
            raise ValueError("latency_seconds must be non-negative")
        if endpoint_hash is not None and (not isinstance(endpoint_hash, str) or not endpoint_hash):
            raise ValueError("endpoint_hash must be non-empty text")
        usage_json = _canonical_json(usage) if usage is not None else None
        error_json = _canonical_json(error) if error is not None else None
        connection = self._transaction()
        try:
            if connection.execute("SELECT 1 FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone() is None:
                raise ValueError(f"unknown attempt: {attempt_id}")
            if connection.execute("SELECT 1 FROM outcomes WHERE attempt_id = ?", (attempt_id,)).fetchone() is not None:
                raise ValueError("attempt already has a terminal outcome")
            finished_at = _now()
            record = {
                "attempt_id": attempt_id, "status": status, "raw_response": raw_response,
                "usage_json": usage_json, "usage_checksum": _digest(usage) if usage is not None else None,
                "latency_seconds": latency_seconds, "endpoint_hash": endpoint_hash,
                "error_json": error_json, "error_checksum": _digest(error) if error is not None else None,
                "finished_at": finished_at,
            }
            connection.execute(
                "INSERT INTO outcomes(attempt_id, status, raw_response, usage_json, usage_checksum, latency_seconds, "
                "endpoint_hash, error_json, error_checksum, finished_at, record_checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*record.values(), _digest(record)),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def accept_annotations(self, attempt_id: str, annotations: Mapping[str, Mapping[str, Any]]) -> None:
        """Append validated labels for every task in one successful batch exactly once."""
        if not isinstance(annotations, Mapping):
            raise TypeError("annotations must map task keys to annotation objects")
        connection = self._transaction()
        try:
            attempt = connection.execute(
                "SELECT batch_json, batch_checksum FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            outcome = connection.execute("SELECT status FROM outcomes WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if attempt is None or outcome is None or outcome["status"] != "succeeded":
                raise ValueError("annotations require a finished successful attempt")
            task_keys = tuple(_decode(attempt["batch_json"], attempt["batch_checksum"], "attempt batch"))
            if set(annotations) != set(task_keys):
                raise ValueError("annotations must have complete batch membership")
            task_rows = {
                row["task_key"]: row for row in connection.execute(
                    f"SELECT task_key, cache_key FROM tasks WHERE task_key IN ({','.join('?' for _ in task_keys)})", task_keys
                )
            }
            if len({task_rows[key]["cache_key"] for key in task_keys}) != len(task_keys):
                raise ValueError("a batch may contain only one representative per cache key")
            for task_key in task_keys:
                annotation = annotations[task_key]
                if not isinstance(annotation, Mapping) or annotation.get("task_key") != task_key:
                    raise ValueError(f"annotation task identity mismatch for {task_key}")
                payload = _json_copy(dict(annotation))
                payload_json = _canonical_json(payload)
                cache_key, accepted_at = task_rows[task_key]["cache_key"], _now()
                if connection.execute("SELECT 1 FROM annotations WHERE cache_key = ?", (cache_key,)).fetchone() is not None:
                    raise ValueError("accepted annotation cannot be replaced")
                record = {
                    "annotation_id": str(uuid.uuid4()), "attempt_id": attempt_id, "source_task_key": task_key,
                    "cache_key": cache_key, "payload_json": payload_json, "payload_checksum": _digest(payload),
                    "accepted_at": accepted_at,
                }
                connection.execute(
                    "INSERT INTO annotations(annotation_id, attempt_id, source_task_key, cache_key, payload_json, "
                    "payload_checksum, accepted_at, record_checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (*record.values(), _digest(record)),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def pending_tasks(self) -> list[dict[str, Any]]:
        """Return one deterministic representative for each annotation cache miss."""
        self._assert_open()
        rows = self._connection.execute(
            "SELECT t.task_key, t.payload_json, t.payload_checksum FROM tasks t "
            "WHERE NOT EXISTS (SELECT 1 FROM annotations a WHERE a.cache_key = t.cache_key) "
            "AND t.task_key = (SELECT MIN(peer.task_key) FROM tasks peer WHERE peer.cache_key = t.cache_key) "
            "ORDER BY t.task_key"
        ).fetchall()
        return [_decode(row["payload_json"], row["payload_checksum"], f"task {row['task_key']}") for row in rows]

    def annotation_for_task(self, task_key: str) -> dict[str, Any] | None:
        """Return the accepted cached label, rebinding only its task identity."""
        self._assert_open()
        row = self._connection.execute(
            "SELECT t.task_key, a.payload_json, a.payload_checksum FROM tasks t "
            "LEFT JOIN annotations a ON a.cache_key = t.cache_key WHERE t.task_key = ?", (task_key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown manifest task: {task_key}")
        if row["payload_json"] is None:
            return None
        annotation = _decode(row["payload_json"], row["payload_checksum"], f"annotation for {task_key}")
        annotation["task_key"] = task_key
        return annotation

    def status(self) -> dict[str, int]:
        """Return stable progress counts without making any network request."""
        self._assert_open()
        counts = self._connection.execute(
            "SELECT (SELECT COUNT(*) FROM tasks) AS total_tasks, "
            "(SELECT COUNT(DISTINCT cache_key) FROM tasks) AS unique_inputs, "
            "(SELECT COUNT(*) FROM annotations) AS accepted_inputs, "
            "(SELECT COUNT(*) FROM tasks t WHERE EXISTS (SELECT 1 FROM annotations a WHERE a.cache_key = t.cache_key)) AS accepted_tasks, "
            "(SELECT COUNT(*) FROM attempts) AS attempts, "
            "(SELECT COUNT(*) FROM outcomes) AS finished_attempts, "
            "(SELECT COUNT(*) FROM attempts a WHERE NOT EXISTS (SELECT 1 FROM outcomes o WHERE o.attempt_id = a.attempt_id)) AS unfinished_attempts"
        ).fetchone()
        result = dict(counts)
        result["pending_tasks"] = result["unique_inputs"] - result["accepted_inputs"]
        return {key: int(value) for key, value in result.items()}

    def verify(self) -> dict[str, Any]:
        """Recompute payload and record checksums for every immutable row."""
        self._assert_open()
        errors: list[str] = []

        def checked(label: str, operation: Any) -> None:
            try:
                operation()
            except (TypeError, ValueError, sqlite3.DatabaseError) as exc:
                errors.append(f"{label}: {exc}")

        def verify_manifest() -> None:
            row = self._connection.execute("SELECT payload_json, payload_checksum FROM manifest WHERE singleton = 1").fetchone()
            if row is None:
                raise ValueError("manifest is missing")
            manifest = _decode(row["payload_json"], row["payload_checksum"], "manifest")
            _, expected_tasks = self._validated_manifest(manifest)
            stored_tasks = self._connection.execute(
                "SELECT task_key, cache_key, payload_json, payload_checksum FROM tasks ORDER BY task_key"
            ).fetchall()
            expected_by_key = {task["task_key"]: task for task in expected_tasks}
            if set(expected_by_key) != {row["task_key"] for row in stored_tasks}:
                raise ValueError("stored task membership differs from manifest")
            for stored in stored_tasks:
                expected = expected_by_key[stored["task_key"]]
                if stored["cache_key"] != expected["cache_key"]:
                    raise ValueError(f"cache key mismatch for task {stored['task_key']}")
                if stored["payload_json"] != _canonical_json(expected["payload"]):
                    raise ValueError(f"stored task payload differs from manifest for {stored['task_key']}")

        checked("manifest", verify_manifest)
        for row in self._connection.execute("SELECT * FROM tasks ORDER BY task_key"):
            def verify_task(row: sqlite3.Row = row) -> None:
                payload = _decode(row["payload_json"], row["payload_checksum"], f"task {row['task_key']}")
                record = {"task_key": row["task_key"], "cache_key": row["cache_key"],
                          "payload_json": row["payload_json"], "payload_checksum": row["payload_checksum"]}
                if _digest(record) != row["record_checksum"]:
                    raise ValueError(f"checksum mismatch for task {row['task_key']}")
                _, tasks = self._validated_manifest({**self._manifest, "tasks": [payload]})
                if tasks[0]["cache_key"] != row["cache_key"]:
                    raise ValueError(f"cache key mismatch for task {row['task_key']}")
            checked(f"task {row['task_key']}", verify_task)
        for row in self._connection.execute("SELECT * FROM attempts ORDER BY rowid"):
            def verify_attempt(row: sqlite3.Row = row) -> None:
                task_keys = _decode(row["batch_json"], row["batch_checksum"], f"attempt {row['attempt_id']} batch")
                if not isinstance(task_keys, list) or not task_keys or task_keys != sorted(set(task_keys)):
                    raise ValueError(f"invalid batch membership for attempt {row['attempt_id']}")
                if _digest(tuple(task_keys)) != row["batch_key"]:
                    raise ValueError(f"batch key mismatch for attempt {row['attempt_id']}")
                existing = self._connection.execute(
                    f"SELECT COUNT(*) AS count FROM tasks WHERE task_key IN ({','.join('?' for _ in task_keys)})", task_keys
                ).fetchone()["count"]
                if existing != len(task_keys):
                    raise ValueError(f"attempt {row['attempt_id']} contains an unknown task")
                record = {key: row[key] for key in ("attempt_id", "batch_key", "batch_json", "batch_checksum", "attempt_no", "started_at")}
                if _digest(record) != row["record_checksum"]:
                    raise ValueError(f"checksum mismatch for attempt {row['attempt_id']}")
            checked(f"attempt {row['attempt_id']}", verify_attempt)
        for row in self._connection.execute("SELECT * FROM outcomes ORDER BY rowid"):
            def verify_outcome(row: sqlite3.Row = row) -> None:
                if row["usage_json"] is not None:
                    _decode(row["usage_json"], row["usage_checksum"], f"outcome usage {row['attempt_id']}")
                if row["error_json"] is not None:
                    _decode(row["error_json"], row["error_checksum"], f"outcome error {row['attempt_id']}")
                record = {key: row[key] for key in (
                    "attempt_id", "status", "raw_response", "usage_json", "usage_checksum", "latency_seconds",
                    "endpoint_hash", "error_json", "error_checksum", "finished_at",
                )}
                if _digest(record) != row["record_checksum"]:
                    raise ValueError(f"checksum mismatch for outcome {row['attempt_id']}")
            checked(f"outcome {row['attempt_id']}", verify_outcome)
        for row in self._connection.execute(
            "SELECT a.*, o.status AS outcome_status, t.cache_key AS task_cache_key, at.batch_json, at.batch_checksum "
            "FROM annotations a JOIN outcomes o ON o.attempt_id = a.attempt_id "
            "JOIN tasks t ON t.task_key = a.source_task_key JOIN attempts at ON at.attempt_id = a.attempt_id ORDER BY a.rowid"
        ):
            def verify_annotation(row: sqlite3.Row = row) -> None:
                _decode(row["payload_json"], row["payload_checksum"], f"annotation {row['annotation_id']}")
                if row["outcome_status"] != "succeeded":
                    raise ValueError(f"annotation {row['annotation_id']} has no successful outcome")
                if row["cache_key"] != row["task_cache_key"]:
                    raise ValueError(f"annotation {row['annotation_id']} has the wrong cache key")
                task_keys = _decode(row["batch_json"], row["batch_checksum"], f"annotation batch {row['annotation_id']}")
                if row["source_task_key"] not in task_keys:
                    raise ValueError(f"annotation {row['annotation_id']} is outside its attempt batch")
                record = {key: row[key] for key in (
                    "annotation_id", "attempt_id", "source_task_key", "cache_key", "payload_json", "payload_checksum", "accepted_at",
                )}
                if _digest(record) != row["record_checksum"]:
                    raise ValueError(f"checksum mismatch for annotation {row['annotation_id']}")
            checked(f"annotation {row['annotation_id']}", verify_annotation)
        return {"ok": not errors, "errors": tuple(errors), "status": self.status()}
