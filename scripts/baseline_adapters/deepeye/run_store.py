"""Durable, append-only recording for DeepEye experiment runs.

The database is intentionally outside the baseline implementation.  It stores
engine-supplied artifacts and metrics without interpreting their semantics.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from contextlib import contextmanager
import copy
import dataclasses
import datetime as dt
from decimal import Decimal
import enum
import errno
import fcntl
import hashlib
import importlib
import ipaddress
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator
import uuid


_DATABASE_NAME = "run.sqlite3"
_LOCK_NAME = ".writer.lock"
_SCHEMA_VERSION = 1
_TYPE_TAG = "__run_store_type__"
_IP_TYPES = {cls.__name__: cls for cls in (
    ipaddress.IPv4Address, ipaddress.IPv6Address,
    ipaddress.IPv4Interface, ipaddress.IPv6Interface,
    ipaddress.IPv4Network, ipaddress.IPv6Network,
)}
_CORE_EXPORT_NAMES = frozenset({
    "manifest.json",
    "summary.json",
    "verification.json",
    "attempts.jsonl",
    "events.jsonl",
    "COMPLETE.json",
})


class _VerificationReport(dict[str, Any]):
    """Public report fields plus private provenance for safe in-process reuse."""

    __slots__ = ("_store_token", "_revision", "_contents_checksum")

    def __init__(
        self,
        values: dict[str, Any],
        *,
        store_token: object,
        revision: tuple[int, int],
    ) -> None:
        super().__init__(values)
        self._store_token = store_token
        self._revision = revision
        self._contents_checksum = self._current_contents_checksum()

    def _current_contents_checksum(self) -> bytes:
        contents = _canonical_dumps(dict(self)).encode("utf-8")
        return hashlib.sha256(contents).digest()

    def _contents_unchanged(self) -> bool:
        return self._contents_checksum == self._current_contents_checksum()


def _type_reference(value: object) -> dict[str, str]:
    cls = type(value)
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def to_jsonable(value: Any) -> Any:
    """Convert supported Python values to deterministic, lossless JSON data.

    Non-finite binary floats are rejected because JSON has no portable encoding
    for them.  ``Decimal`` values, including Decimal NaNs and infinities, use a
    tagged string representation and therefore remain valid JSON.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            name = "nan"
        else:
            name = "infinity" if value > 0 else "-infinity"
        return {_TYPE_TAG: "float", "value": name}
    if isinstance(value, enum.Enum):
        return {_TYPE_TAG: "enum", **_type_reference(value), "name": value.name}
    if isinstance(value, Decimal):
        return {_TYPE_TAG: "decimal", "value": str(value)}
    if isinstance(value, dt.datetime):
        return {_TYPE_TAG: "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {_TYPE_TAG: "date", "value": value.isoformat()}
    if isinstance(value, dt.time):
        return {_TYPE_TAG: "time", "value": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {_TYPE_TAG: "timedelta", "days": value.days, "seconds": value.seconds,
                "microseconds": value.microseconds}
    if isinstance(value, Path):
        return {_TYPE_TAG: "path", "value": str(value)}
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return {_TYPE_TAG: "bytes", "value": encoded}
    if isinstance(value, uuid.UUID):
        return {_TYPE_TAG: "uuid", "value": str(value)}
    if type(value) in _IP_TYPES.values():
        return {_TYPE_TAG: "ipaddress", "kind": type(value).__name__, "value": str(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
        return {_TYPE_TAG: "dataclass", **_type_reference(value), "fields": fields}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        fields = value.model_dump(mode="python")
        return {_TYPE_TAG: "pydantic", **_type_reference(value), "fields": to_jsonable(fields)}
    if hasattr(value, "dict") and callable(value.dict):
        # Pydantic v1 compatibility.  Restrict this fallback to Pydantic types.
        if any(cls.__module__.startswith("pydantic") for cls in type(value).__mro__):
            return {
                _TYPE_TAG: "pydantic",
                **_type_reference(value),
                "fields": to_jsonable(value.dict()),
            }
    if isinstance(value, tuple):
        return {_TYPE_TAG: "tuple", "items": [to_jsonable(item) for item in value]}
    if isinstance(value, set):
        items = [to_jsonable(item) for item in value]
        items.sort(key=_canonical_dumps)
        return {_TYPE_TAG: "set", "items": items}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and _TYPE_TAG not in value:
            return {key: to_jsonable(item) for key, item in value.items()}
        items = [[to_jsonable(key), to_jsonable(item)] for key, item in value.items()]
        items.sort(key=lambda pair: _canonical_dumps(pair[0]))
        return {_TYPE_TAG: "dict", "items": items}
    raise TypeError(f"unsupported JSON value: {type(value).__module__}.{type(value).__qualname__}")


def _resolve_type(module_name: str, qualname: str) -> type[Any]:
    if "<locals>" in qualname:
        raise ValueError(f"cannot restore local type {module_name}.{qualname}")
    # Resolve historical records without modifying their stored JSON/checksums.
    legacy_prefix = "result_contract.baseline_adapters."
    if module_name.startswith(legacy_prefix):
        module_name = "scripts.baseline_adapters." + module_name[len(legacy_prefix):]
    resolved: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        resolved = getattr(resolved, part)
    if not isinstance(resolved, type):
        raise TypeError(f"stored type reference is not a type: {module_name}.{qualname}")
    return resolved


def restore_jsonable(value: Any) -> Any:
    """Restore data produced by :func:`to_jsonable`."""

    if isinstance(value, list):
        return [restore_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TYPE_TAG)
    if tag is None:
        return {key: restore_jsonable(item) for key, item in value.items()}
    if tag == "decimal":
        return Decimal(value["value"])
    if tag == "float":
        names = {"nan": math.nan, "infinity": math.inf, "-infinity": -math.inf}
        try:
            return names[value["value"]]
        except KeyError as exc:
            raise ValueError(f"invalid tagged float: {value.get('value')!r}") from exc
    if tag == "datetime":
        return dt.datetime.fromisoformat(value["value"])
    if tag == "date":
        return dt.date.fromisoformat(value["value"])
    if tag == "time":
        return dt.time.fromisoformat(value["value"])
    if tag == "timedelta":
        return dt.timedelta(days=value["days"], seconds=value["seconds"], microseconds=value["microseconds"])
    if tag == "path":
        return Path(value["value"])
    if tag == "bytes":
        return base64.b64decode(value["value"], validate=True)
    if tag == "uuid":
        return uuid.UUID(value["value"])
    if tag == "ipaddress":
        if value.get("kind") not in _IP_TYPES:
            raise ValueError("unsupported stored IP address type")
        return _IP_TYPES[value["kind"]](value["value"])
    if tag == "tuple":
        return tuple(restore_jsonable(item) for item in value["items"])
    if tag == "set":
        return {restore_jsonable(item) for item in value["items"]}
    if tag == "dict":
        return {restore_jsonable(key): restore_jsonable(item) for key, item in value["items"]}
    if tag == "enum":
        cls = _resolve_type(value["module"], value["qualname"])
        return cls[value["name"]]
    if tag in {"dataclass", "pydantic"}:
        cls = _resolve_type(value["module"], value["qualname"])
        fields = restore_jsonable(value["fields"])
        return cls(**fields)
    raise ValueError(f"unknown stored JSON type tag: {tag!r}")


def _canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encode_json(value: Any) -> tuple[str, str]:
    text = _canonical_dumps(to_jsonable(value))
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_json(text: str, expected_checksum: str, label: str) -> Any:
    actual_checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual_checksum != expected_checksum:
        raise ValueError(f"checksum mismatch for {label}")
    try:
        return restore_jsonable(json.loads(text))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON for {label}") from exc


def _record_checksum(record: dict[str, Any]) -> str:
    text = _canonical_dumps(to_jsonable(record))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


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
CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    item_key TEXT NOT NULL,
    stage TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
    started_at TEXT NOT NULL,
    record_checksum TEXT NOT NULL,
    UNIQUE (item_key, stage, attempt_no)
);
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    event_no INTEGER NOT NULL CHECK (event_no > 0),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    record_checksum TEXT NOT NULL,
    UNIQUE (attempt_id, event_no)
);
CREATE TABLE finishes (
    attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    payload_json TEXT NOT NULL,
    payload_checksum TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    record_checksum TEXT NOT NULL
);
CREATE INDEX attempts_lookup ON attempts(item_key, stage, input_fingerprint, attempt_no DESC);
CREATE TRIGGER schema_info_no_update BEFORE UPDATE ON schema_info BEGIN SELECT RAISE(ABORT, 'schema_info is immutable'); END;
CREATE TRIGGER schema_info_no_delete BEFORE DELETE ON schema_info BEGIN SELECT RAISE(ABORT, 'schema_info is immutable'); END;
CREATE TRIGGER manifest_no_update BEFORE UPDATE ON manifest BEGIN SELECT RAISE(ABORT, 'manifest is immutable'); END;
CREATE TRIGGER manifest_no_delete BEFORE DELETE ON manifest BEGIN SELECT RAISE(ABORT, 'manifest is immutable'); END;
CREATE TRIGGER attempts_no_update BEFORE UPDATE ON attempts BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
CREATE TRIGGER attempts_no_delete BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
CREATE TRIGGER finishes_no_update BEFORE UPDATE ON finishes BEGIN SELECT RAISE(ABORT, 'finishes are immutable'); END;
CREATE TRIGGER finishes_no_delete BEFORE DELETE ON finishes BEGIN SELECT RAISE(ABORT, 'finishes are immutable'); END;
"""


class RunStore:
    """A single-run, append-only SQLite event and outcome store."""

    def __init__(
        self,
        run_dir: Path,
        connection: sqlite3.Connection,
        manifest: dict[str, Any],
        *,
        read_only: bool,
        lock_file: Any | None,
    ) -> None:
        self._run_dir = run_dir
        self._connection = connection
        self._manifest = manifest
        self._manifest_fingerprint = hashlib.sha256(
            _canonical_dumps(to_jsonable(manifest)).encode("utf-8")
        ).hexdigest()
        self._verification_token = object()
        self._read_only = read_only
        self._lock_file = lock_file
        self._mutex = threading.RLock()
        self._snapshot_depth = 0
        self._closed = False

    @classmethod
    def create(cls, run_dir: Path, manifest: dict[str, Any]) -> "RunStore":
        run_dir = Path(run_dir)
        if not isinstance(manifest, dict):
            raise TypeError("run manifest must be a dictionary")
        manifest_json, manifest_checksum = _encode_json(manifest)
        verified_manifest = _decode_json(manifest_json, manifest_checksum, "manifest")
        if not isinstance(verified_manifest, dict):
            raise ValueError("run manifest must be a dictionary")
        run_dir.mkdir()
        lock_file = None
        connection = None
        try:
            lock_file = cls._acquire_writer_lock(run_dir, create=True)
            connection = cls._connect_writer(run_dir / _DATABASE_NAME)
            connection.executescript(_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("INSERT INTO schema_info(singleton, version) VALUES (1, ?)", (_SCHEMA_VERSION,))
                connection.execute(
                    "INSERT INTO manifest(singleton, payload_json, payload_checksum) VALUES (1, ?, ?)",
                    (manifest_json, manifest_checksum),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            return cls(
                run_dir,
                connection,
                verified_manifest,
                read_only=False,
                lock_file=lock_file,
            )
        except BaseException:
            if connection is not None:
                connection.close()
            cls._release_writer_lock(lock_file)
            raise

    @classmethod
    def open(
        cls,
        run_dir: Path,
        expected_manifest: dict[str, Any] | None = None,
        read_only: bool = False,
    ) -> "RunStore":
        run_dir = Path(run_dir)
        database = run_dir / _DATABASE_NAME
        manifest_json, manifest_checksum, manifest = cls._read_manifest(database)
        if expected_manifest is not None:
            expected_json, _ = _encode_json(expected_manifest)
            if expected_json != manifest_json:
                raise ValueError("run manifest does not match expected manifest")

        lock_file = None
        connection = None
        try:
            if read_only:
                connection = cls._connect_reader(database)
            else:
                lock_file = cls._acquire_writer_lock(run_dir, create=False)
                connection = cls._connect_writer(database)
            cls._check_schema_version(connection)
            return cls(
                run_dir,
                connection,
                manifest,
                read_only=read_only,
                lock_file=lock_file,
            )
        except BaseException:
            if connection is not None:
                connection.close()
            cls._release_writer_lock(lock_file)
            raise

    @staticmethod
    def _connect_writer(database: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database, timeout=0.25, isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            connection.close()
            raise RuntimeError("SQLite WAL mode is required")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=250")
        return connection

    @staticmethod
    def _connect_reader(database: Path) -> sqlite3.Connection:
        if not database.is_file():
            raise FileNotFoundError(database)
        uri = f"{database.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @classmethod
    def _read_manifest(cls, database: Path) -> tuple[str, str, dict[str, Any]]:
        connection = cls._connect_reader(database)
        try:
            cls._check_schema_version(connection)
            row = connection.execute(
                "SELECT payload_json, payload_checksum FROM manifest WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise ValueError("run manifest is missing")
            manifest = _decode_json(row["payload_json"], row["payload_checksum"], "manifest")
            if not isinstance(manifest, dict):
                raise ValueError("run manifest must be a dictionary")
            return row["payload_json"], row["payload_checksum"], manifest
        finally:
            connection.close()

    @staticmethod
    def _check_schema_version(connection: sqlite3.Connection) -> None:
        try:
            row = connection.execute("SELECT version FROM schema_info WHERE singleton = 1").fetchone()
        except sqlite3.DatabaseError as exc:
            raise ValueError("not a RunStore database") from exc
        if row is None or row["version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported RunStore schema version")

    @staticmethod
    def _acquire_writer_lock(run_dir: Path, *, create: bool) -> Any:
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        try:
            descriptor = os.open(run_dir / _LOCK_NAME, flags, 0o600)
        except FileNotFoundError as exc:
            raise ValueError("run writer lock is missing") from exc
        lock_file = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(f"run already has a writer: {run_dir}") from exc
            raise
        return lock_file

    @staticmethod
    def _release_writer_lock(lock_file: Any | None) -> None:
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def manifest_fingerprint(self) -> str:
        return self._manifest_fingerprint

    def __enter__(self) -> "RunStore":
        self._assert_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            try:
                self._connection.close()
            finally:
                self._release_writer_lock(self._lock_file)
                self._lock_file = None

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("RunStore is closed")

    def _assert_writable(self) -> None:
        self._assert_open()
        if self._read_only:
            raise PermissionError("RunStore was opened read-only")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._assert_writable()
        with self._mutex:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    @contextmanager
    def _read_snapshot(self) -> Iterator[sqlite3.Connection]:
        """Hold one explicit SQLite snapshot, nesting safely within store reads."""

        self._assert_open()
        with self._mutex:
            outermost = self._snapshot_depth == 0
            if outermost:
                if self._connection.in_transaction:
                    raise RuntimeError("cannot start a read snapshot inside an external transaction")
                self._connection.execute("BEGIN")
            self._snapshot_depth += 1
            try:
                yield self._connection
            except BaseException:
                self._snapshot_depth -= 1
                if outermost:
                    self._connection.rollback()
                raise
            else:
                self._snapshot_depth -= 1
                if outermost:
                    self._connection.commit()

    def begin_attempt(self, item_key: str, stage: str, input_fingerprint: str) -> str:
        if not all(isinstance(value, str) and value for value in (item_key, stage, input_fingerprint)):
            raise ValueError("item_key, stage, and input_fingerprint must be non-empty strings")
        attempt_id = str(uuid.uuid4())
        started_at = _now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM attempts WHERE item_key = ? AND stage = ?",
                (item_key, stage),
            ).fetchone()
            attempt_no = int(row[0])
            checksum = _record_checksum(
                {
                    "attempt_id": attempt_id,
                    "item_key": item_key,
                    "stage": stage,
                    "input_fingerprint": input_fingerprint,
                    "attempt_no": attempt_no,
                    "started_at": started_at,
                }
            )
            connection.execute(
                "INSERT INTO attempts(attempt_id, item_key, stage, input_fingerprint, attempt_no, started_at, record_checksum) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (attempt_id, item_key, stage, input_fingerprint, attempt_no, started_at, checksum),
            )
        return attempt_id

    def append_event(self, attempt_id: str, kind: str, payload: dict[str, Any]) -> int:
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("event payload must be a dictionary")
        payload_json, payload_checksum = _encode_json(payload)
        created_at = _now()
        with self._transaction() as connection:
            attempt = connection.execute(
                "SELECT attempt_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise ValueError(f"unknown attempt: {attempt_id}")
            if connection.execute("SELECT 1 FROM finishes WHERE attempt_id = ?", (attempt_id,)).fetchone():
                raise ValueError("cannot append an event after an attempt is finished")
            row = connection.execute(
                "SELECT COALESCE(MAX(event_no), 0) + 1 FROM events WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            event_no = int(row[0])
            checksum = _record_checksum(
                {
                    "attempt_id": attempt_id,
                    "event_no": event_no,
                    "kind": kind,
                    "payload_json": payload_json,
                    "payload_checksum": payload_checksum,
                    "created_at": created_at,
                }
            )
            connection.execute(
                "INSERT INTO events(attempt_id, event_no, kind, payload_json, payload_checksum, created_at, record_checksum) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (attempt_id, event_no, kind, payload_json, payload_checksum, created_at, checksum),
            )
        return event_no

    def finish_attempt(self, attempt_id: str, status: str, payload: dict[str, Any]) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("attempt status must be 'succeeded' or 'failed'")
        if not isinstance(payload, dict):
            raise TypeError("finish payload must be a dictionary")
        payload_json, payload_checksum = _encode_json(payload)
        finished_at = _now()
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown attempt: {attempt_id}")
            if connection.execute("SELECT 1 FROM finishes WHERE attempt_id = ?", (attempt_id,)).fetchone():
                raise ValueError("attempt already has a terminal outcome")
            checksum = _record_checksum(
                {
                    "attempt_id": attempt_id,
                    "status": status,
                    "payload_json": payload_json,
                    "payload_checksum": payload_checksum,
                    "finished_at": finished_at,
                }
            )
            connection.execute(
                "INSERT INTO finishes(attempt_id, status, payload_json, payload_checksum, finished_at, record_checksum) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (attempt_id, status, payload_json, payload_checksum, finished_at, checksum),
            )

    @staticmethod
    def _verify_attempt_row(row: sqlite3.Row) -> None:
        record = {
            "attempt_id": row["attempt_id"],
            "item_key": row["item_key"],
            "stage": row["stage"],
            "input_fingerprint": row["input_fingerprint"],
            "attempt_no": row["attempt_no"],
            "started_at": row["started_at"],
        }
        if _record_checksum(record) != row["attempt_checksum"]:
            raise ValueError(f"checksum mismatch for attempt {row['attempt_id']}")

    @staticmethod
    def _verify_finish_row(row: sqlite3.Row) -> Any | None:
        if row["status"] is None:
            return None
        record = {
            "attempt_id": row["attempt_id"],
            "status": row["status"],
            "payload_json": row["finish_payload_json"],
            "payload_checksum": row["finish_payload_checksum"],
            "finished_at": row["finished_at"],
        }
        if _record_checksum(record) != row["finish_checksum"]:
            raise ValueError(f"checksum mismatch for finish {row['attempt_id']}")
        return _decode_json(
            row["finish_payload_json"],
            row["finish_payload_checksum"],
            f"finish {row['attempt_id']}",
        )

    @staticmethod
    def _attempt_dict(row: sqlite3.Row) -> dict[str, Any]:
        RunStore._verify_attempt_row(row)
        payload = RunStore._verify_finish_row(row)
        status = row["status"] if row["status"] is not None else "interrupted"
        return {
            "attempt_id": row["attempt_id"],
            "item_key": row["item_key"],
            "stage": row["stage"],
            "input_fingerprint": row["input_fingerprint"],
            "attempt_no": row["attempt_no"],
            "started_at": row["started_at"],
            "status": status,
            "finished_at": row["finished_at"],
            "payload": payload,
        }

    @staticmethod
    def _attempt_query(where: str = "") -> str:
        return (
            "SELECT a.attempt_id, a.item_key, a.stage, a.input_fingerprint, a.attempt_no, a.started_at, "
            "a.record_checksum AS attempt_checksum, f.status, f.payload_json AS finish_payload_json, "
            "f.payload_checksum AS finish_payload_checksum, f.finished_at, f.record_checksum AS finish_checksum "
            "FROM attempts a LEFT JOIN finishes f ON f.attempt_id = a.attempt_id " + where
        )

    def completed(self, item_key: str, stage: str, input_fingerprint: str) -> dict[str, Any] | None:
        self._assert_open()
        with self._mutex:
            rows = self._connection.execute(
                self._attempt_query(
                    "WHERE a.item_key = ? AND a.stage = ? "
                    "AND f.status = 'succeeded' ORDER BY a.attempt_no DESC"
                ),
                (item_key, stage),
            ).fetchall()
        if not rows:
            return None
        attempts = [self._attempt_dict(row) for row in rows]
        conflicting = next(
            (attempt for attempt in attempts if attempt["input_fingerprint"] != input_fingerprint),
            None,
        )
        if conflicting is not None:
            raise ValueError(
                f"successful outcome for {item_key!r}/{stage!r} belongs to a different input fingerprint"
            )
        attempt = attempts[0]
        return {
            "attempt_id": attempt["attempt_id"],
            "attempt_no": attempt["attempt_no"],
            "payload": attempt["payload"],
        }

    def attempt(self, attempt_id: str) -> dict[str, Any]:
        """Read one verified attempt without rescanning the run's history."""
        self._assert_open()
        with self._mutex:
            row = self._connection.execute(self._attempt_query('WHERE a.attempt_id = ?'), (attempt_id,)).fetchone()
        if row is None:
            raise ValueError('unknown attempt')
        return self._attempt_dict(row)

    def attempts(self) -> list[dict[str, Any]]:
        self._assert_open()
        with self._mutex:
            rows = self._connection.execute(
                self._attempt_query("ORDER BY a.rowid")
            ).fetchall()
        return [self._attempt_dict(row) for row in rows]

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
        record = {
            "attempt_id": row["attempt_id"],
            "event_no": row["event_no"],
            "kind": row["kind"],
            "payload_json": row["payload_json"],
            "payload_checksum": row["payload_checksum"],
            "created_at": row["created_at"],
        }
        if _record_checksum(record) != row["record_checksum"]:
            raise ValueError(f"checksum mismatch for event {row['event_id']}")
        payload = _decode_json(row["payload_json"], row["payload_checksum"], f"event {row['event_id']}")
        return {
            "event_id": row["event_id"],
            "attempt_id": row["attempt_id"],
            "event_no": row["event_no"],
            "kind": row["kind"],
            "payload": payload,
            "created_at": row["created_at"],
        }

    def event(self, attempt_id: str, event_no: int) -> dict[str, Any]:
        """Point-read one verified event using its owning attempt and sequence."""
        self._assert_open()
        if not isinstance(attempt_id, str) or not attempt_id or type(event_no) is not int or event_no < 1:
            raise ValueError("valid attempt_id and positive event_no are required")
        with self._mutex:
            row = self._connection.execute(
                "SELECT * FROM events WHERE attempt_id = ? AND event_no = ?",
                (attempt_id, event_no),
            ).fetchone()
        if row is None:
            raise ValueError("unknown event for attempt")
        return self._event_dict(row)

    def iter_events(
        self,
        attempt_id: str | None = None,
        kinds: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield a bounded snapshot of verified events without loading the log.

        Rows are fetched in modest batches and the connection lock is released
        between batches. Events committed after iteration begins are excluded so
        a live writer cannot make a diagnostic traversal unbounded.
        """

        self._assert_open()
        if kinds is None:
            selected_kinds: tuple[str, ...] | None = None
        elif isinstance(kinds, str):
            selected_kinds = (kinds,)
        else:
            selected_kinds = tuple(dict.fromkeys(kinds))
        if selected_kinds is not None:
            if not all(isinstance(kind, str) and kind for kind in selected_kinds):
                raise ValueError("event kinds must be non-empty strings")
            if not selected_kinds:
                return

        clauses = ["event_id > ?", "event_id <= ?"]
        if attempt_id is not None:
            clauses.append("attempt_id = ?")
        if selected_kinds is not None:
            placeholders = ",".join("?" for _ in selected_kinds)
            clauses.append(f"kind IN ({placeholders})")

        with self._mutex:
            upper_bound = int(self._connection.execute(
                "SELECT COALESCE(MAX(event_id), 0) FROM events"
            ).fetchone()[0])
        last_event_id = 0
        while last_event_id < upper_bound:
            parameters: list[Any] = [last_event_id, upper_bound]
            if attempt_id is not None:
                parameters.append(attempt_id)
            if selected_kinds is not None:
                parameters.extend(selected_kinds)
            sql = (
                "SELECT * FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY event_id LIMIT 256"
            )
            with self._mutex:
                self._assert_open()
                rows = self._connection.execute(sql, parameters).fetchall()
            if not rows:
                return
            last_event_id = int(rows[-1]["event_id"])
            for row in rows:
                yield self._event_dict(row)

    def events(self, attempt_id: str | None = None) -> list[dict[str, Any]]:
        return list(self.iter_events(attempt_id))

    def _database_revision(self) -> tuple[int, int]:
        data_version = int(self._connection.execute("PRAGMA data_version").fetchone()[0])
        return data_version, self._connection.total_changes

    def verify(self) -> dict[str, Any]:
        self._assert_open()
        checksum_errors = 0
        records_checked = 0
        with self._read_snapshot():
            integrity_rows = self._connection.execute("PRAGMA integrity_check").fetchall()
            integrity = [str(row[0]) for row in integrity_rows]
            try:
                row = self._connection.execute(
                    "SELECT payload_json, payload_checksum FROM manifest WHERE singleton = 1"
                ).fetchone()
                if row is None:
                    raise ValueError("manifest missing")
                records_checked += 1
                _decode_json(row["payload_json"], row["payload_checksum"], "manifest")
            except (sqlite3.DatabaseError, ValueError):
                checksum_errors += 1
            try:
                attempt_rows = self._connection.execute(self._attempt_query("ORDER BY a.rowid"))
                for row in attempt_rows:
                    records_checked += 1
                    try:
                        self._verify_attempt_row(row)
                    except ValueError:
                        checksum_errors += 1
                    if row["status"] is not None:
                        records_checked += 1
                        try:
                            self._verify_finish_row(row)
                        except ValueError:
                            checksum_errors += 1
            except sqlite3.DatabaseError:
                checksum_errors += 1
            try:
                event_rows = self._connection.execute("SELECT * FROM events ORDER BY event_id")
                for row in event_rows:
                    records_checked += 1
                    try:
                        self._event_dict(row)
                    except ValueError:
                        checksum_errors += 1
            except sqlite3.DatabaseError:
                checksum_errors += 1
            revision = self._database_revision()
        sqlite_ok = integrity == ["ok"]
        return _VerificationReport(
            {
                "ok": sqlite_ok and checksum_errors == 0,
                "sqlite_integrity": integrity,
                "checksum_errors": checksum_errors,
                "records_checked": records_checked,
            },
            store_token=self._verification_token,
            revision=revision,
        )

    def _summary_counts(self) -> dict[str, int]:
        row = self._connection.execute(
            "SELECT COUNT(*) AS attempts, "
            "SUM(CASE WHEN f.status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded, "
            "SUM(CASE WHEN f.status = 'failed' THEN 1 ELSE 0 END) AS failed, "
            "SUM(CASE WHEN f.status IS NULL THEN 1 ELSE 0 END) AS interrupted "
            "FROM attempts a LEFT JOIN finishes f ON f.attempt_id = a.attempt_id"
        ).fetchone()
        event_count = self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "attempts": int(row["attempts"]),
            "succeeded": int(row["succeeded"] or 0),
            "failed": int(row["failed"] or 0),
            "interrupted": int(row["interrupted"] or 0),
            "events": int(event_count),
        }

    def _validate_verification(self, verification: Mapping[str, Any]) -> None:
        if not isinstance(verification, _VerificationReport):
            raise TypeError("verification must be produced by this RunStore")
        if verification._store_token is not self._verification_token:
            raise ValueError("verification belongs to a different RunStore")
        if verification._revision != self._database_revision():
            raise ValueError("verification is stale for the current RunStore snapshot")
        if not verification._contents_unchanged():
            raise ValueError("verification report contents were modified")
        required = {"ok", "sqlite_integrity", "checksum_errors", "records_checked"}
        if not required.issubset(verification):
            raise ValueError("verification is missing required fields")
        if type(verification["ok"]) is not bool:
            raise TypeError("verification ok field must be a boolean")
        for name in ("checksum_errors", "records_checked"):
            if type(verification[name]) is not int or verification[name] < 0:
                raise TypeError(f"verification {name} field must be a non-negative integer")
        integrity = verification["sqlite_integrity"]
        if not isinstance(integrity, list) or not all(isinstance(item, str) for item in integrity):
            raise TypeError("verification sqlite_integrity field must be a list of strings")
        if verification["ok"] is not True or integrity != ["ok"] or verification["checksum_errors"] != 0:
            raise ValueError("cannot summarize a RunStore that fails verification")

    def summary(self, *, verification: Mapping[str, Any] | None = None) -> dict[str, int]:
        with self._read_snapshot():
            if verification is None:
                verification = self.verify()
            self._validate_verification(verification)
            return self._summary_counts()

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _validate_extra_reports(
        extra_reports: Mapping[str, Callable[["RunStore"], Any]] | None,
    ) -> dict[str, Callable[["RunStore"], Any]]:
        if extra_reports is None:
            return {}
        if not isinstance(extra_reports, Mapping):
            raise TypeError("extra_reports must be a mapping of JSON filenames to callables")
        validated: dict[str, Callable[["RunStore"], Any]] = {}
        reserved = {name.casefold() for name in _CORE_EXPORT_NAMES}
        seen: set[str] = set()
        for name, report in extra_reports.items():
            if not isinstance(name, str):
                raise TypeError("extra report filenames must be strings")
            folded = name.casefold()
            if (
                not name
                or "\x00" in name
                or "/" in name
                or "\\" in name
                or Path(name).is_absolute()
                or not folded.endswith(".json")
            ):
                raise ValueError(f"unsafe extra report filename: {name!r}")
            if folded in reserved or folded in seen:
                raise ValueError(f"extra report filename collides with export output: {name!r}")
            if not callable(report):
                raise TypeError(f"extra report producer must be callable: {name!r}")
            seen.add(folded)
            validated[name] = report
        return validated

    def export(
        self,
        export_dir: Path,
        *,
        extra_reports: Mapping[str, Callable[["RunStore"], Any]] | None = None,
    ) -> Path:
        self._assert_open()
        export_dir = Path(export_dir)
        reports = self._validate_extra_reports(extra_reports)
        export_dir.mkdir()
        with self._read_snapshot():
            verification = self.verify()
            if not verification["ok"]:
                raise ValueError("cannot export a RunStore that fails verification")
            manifest_row = self._connection.execute(
                "SELECT payload_json, payload_checksum FROM manifest WHERE singleton = 1"
            ).fetchone()
            if manifest_row is None:
                raise ValueError("run manifest is missing")
            manifest = _decode_json(
                manifest_row["payload_json"], manifest_row["payload_checksum"], "manifest"
            )
            report_values = {
                name: producer(self)
                for name, producer in reports.items()
            }
            self._write_json(export_dir / "manifest.json", manifest)
            self._write_json(export_dir / "summary.json", self._summary_counts())
            self._write_json(export_dir / "verification.json", verification)
            with (export_dir / "attempts.jsonl").open("w", encoding="utf-8") as output:
                for row in self._connection.execute(self._attempt_query("ORDER BY a.rowid")):
                    output.write(_canonical_dumps(to_jsonable(self._attempt_dict(row))) + "\n")
            with (export_dir / "events.jsonl").open("w", encoding="utf-8") as output:
                for row in self._connection.execute("SELECT * FROM events ORDER BY event_id"):
                    output.write(_canonical_dumps(to_jsonable(self._event_dict(row))) + "\n")
            for name, value in report_values.items():
                self._write_json(export_dir / name, value)
        self._write_json(export_dir / "COMPLETE.json", {
            "complete": True,
            "extra_reports": sorted(report_values),
            "verification": verification,
        })
        return export_dir
