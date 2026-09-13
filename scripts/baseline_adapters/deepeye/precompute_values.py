"""Freeze bounded, question-independent PostgreSQL values from the Meta allowlist."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import psycopg
from psycopg import sql

from app.vector_db.vector_db import (
    _is_number_column,
    _is_text_column_type,
    _is_uuid_column,
)


MIN_VALUE_LENGTH = 1
MAX_VALUE_LENGTH = 100
_ALGORITHM_VERSION = "deepeye-postgres-values-v1"
_SUCCESS_STATUSES = {"collected", "empty", "filtered_uuid", "filtered_numeric"}
_ENTRY_FIELDS = (
    "table_name",
    "column_name",
    "eligible_distinct_count",
    "documents",
    "status",
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("PostgreSQL Meta must be JSON serializable") from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal_content(value: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed.pop("content_hash", None)
    sealed["content_hash"] = _fingerprint(sealed)
    return sealed


def _verify_content_hash(
    value: Any, *, artifact: str, allow_unsealed: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid {artifact} integrity")
    recorded = value.get("content_hash")
    if recorded is None and allow_unsealed:
        return value
    if not isinstance(recorded, str):
        if recorded is None:
            raise RuntimeError(f"{artifact} is unsealed; integrity seal required")
        raise RuntimeError(f"Invalid {artifact} integrity")
    payload = dict(value)
    payload.pop("content_hash")
    if not hmac.compare_digest(recorded, _fingerprint(payload)):
        raise RuntimeError(f"Invalid {artifact} integrity")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid database-value artifact: {path.name}") from exc


def _validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"PostgreSQL Meta {label} must be a non-empty identifier")
    return value


def _eligible_columns(schema: dict[str, Any]) -> list[tuple[str, str]]:
    tables = schema.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("PostgreSQL data item has a malformed database schema")
    columns: list[tuple[str, str]] = []
    for raw_table_name, table_schema in tables.items():
        table_name = _validate_identifier(raw_table_name, "table name")
        if not isinstance(table_schema, dict) or not isinstance(
            table_schema.get("columns"), dict
        ):
            raise ValueError(f"PostgreSQL Meta table {table_name!r} has no columns")
        for raw_column_name, column_schema in table_schema["columns"].items():
            column_name = _validate_identifier(raw_column_name, "column name")
            if not isinstance(column_schema, dict):
                raise ValueError(
                    f"PostgreSQL Meta column {table_name}.{column_name} is malformed"
                )
            column_type = column_schema.get("column_type")
            if not isinstance(column_type, str) or not column_type.strip():
                raise ValueError(
                    f"PostgreSQL Meta column {table_name}.{column_name} has no type"
                )
            if _is_text_column_type(column_type):
                columns.append((table_name, column_name))
    return sorted(columns, key=lambda pair: (pair[0].casefold(), pair[0], pair[1].casefold(), pair[1]))


def _value_query(table_name: str, column_name: str) -> sql.Composed:
    """Build the sole server round-trip for one trusted Meta text column."""

    column = sql.Identifier(column_name)
    return sql.SQL(
        "WITH eligible_values AS ("
        "SELECT DISTINCT CAST({column} AS TEXT) AS value "
        "FROM {table} "
        "WHERE {column} IS NOT NULL "
        "AND LENGTH(CAST({column} AS TEXT)) BETWEEN 1 AND 100"
        "), counted_values AS ("
        "SELECT value, COUNT(*) OVER () AS eligible_distinct_count "
        "FROM eligible_values"
        ") "
        "SELECT value, eligible_distinct_count FROM counted_values "
        "ORDER BY md5(value), value LIMIT %s"
    ).format(column=column, table=sql.Identifier("public", table_name))


def _parse_rows(rows: Any, cap: int) -> tuple[list[str], int]:
    if not isinstance(rows, (list, tuple)):
        raise RuntimeError("PostgreSQL value query returned malformed rows")
    if not rows:
        return [], 0
    documents: list[str] = []
    seen: set[str] = set()
    exact_count: int | None = None
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise RuntimeError("PostgreSQL value query returned a malformed row")
        value, count = row
        if value is None:
            raise RuntimeError("PostgreSQL value query returned an ineligible null")
        text = str(value)
        if len(text) < MIN_VALUE_LENGTH:
            raise RuntimeError("PostgreSQL value query returned an empty value")
        if len(text) > MAX_VALUE_LENGTH:
            raise RuntimeError("PostgreSQL value query returned an over-length value")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError("PostgreSQL value query returned an invalid exact count")
        if exact_count is None:
            exact_count = count
        elif exact_count != count:
            raise RuntimeError("PostgreSQL value query returned inconsistent exact counts")
        if text in seen:
            raise RuntimeError("PostgreSQL value query returned duplicate values")
        seen.add(text)
        documents.append(text)
    if len(documents) > cap or exact_count is None or exact_count < len(documents):
        raise RuntimeError("PostgreSQL value query returned inconsistent bounded results")
    return documents, exact_count


def _classify_column(
    table_name: str, column_name: str, rows: Any, cap: int
) -> dict[str, Any]:
    documents, eligible_distinct_count = _parse_rows(rows, cap)
    status = "empty"
    if documents and _is_uuid_column(documents):
        status = "filtered_uuid"
        documents = []
    elif documents and _is_number_column(documents):
        status = "filtered_numeric"
        documents = []
    elif documents:
        status = "collected"
    return {
        "table_name": table_name,
        "column_name": column_name,
        "eligible_distinct_count": eligible_distinct_count,
        "documents": documents,
        "status": status,
    }


def _checkpoint_entry(checkpoint: dict[str, Any]) -> dict[str, Any]:
    try:
        return {key: checkpoint[key] for key in _ENTRY_FIELDS}
    except KeyError as exc:
        raise RuntimeError("Invalid database-value checkpoint integrity") from exc


def _validate_success_entry(
    entry: Any,
    *,
    max_values_per_column: int,
    table_name: str | None = None,
    column_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(entry, dict) or any(key not in entry for key in _ENTRY_FIELDS):
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if entry["status"] not in _SUCCESS_STATUSES:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if table_name is not None and entry["table_name"] != table_name:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if column_name is not None and entry["column_name"] != column_name:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    count = entry["eligible_distinct_count"]
    documents = entry["documents"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if not isinstance(documents, list) or len(documents) > max_values_per_column:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if (
        any(
            not isinstance(value, str)
            or not MIN_VALUE_LENGTH <= len(value) <= MAX_VALUE_LENGTH
            for value in documents
        )
        or len(documents) != len(set(documents))
        or count < len(documents)
    ):
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if entry["status"] == "collected" and not documents:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if entry["status"] != "collected" and documents:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    if entry["status"] == "empty" and count != 0:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    return {key: entry[key] for key in _ENTRY_FIELDS}


def _checkpoint_path(root: Path, table_name: str, column_name: str) -> Path:
    digest = hashlib.sha256(
        _canonical_bytes([table_name, column_name])
    ).hexdigest()
    return root / "column_checkpoints" / f"{digest}.json"


def _stats(columns: list[dict[str, Any]]) -> dict[str, int]:
    successful = [entry for entry in columns if entry["status"] in _SUCCESS_STATUSES]
    return {
        "text_column_count": len(columns),
        "successful_column_count": len(successful),
        "eligible_distinct_count": sum(
            entry["eligible_distinct_count"] for entry in successful
        ),
        "document_count": sum(len(entry["documents"]) for entry in successful),
        "empty_column_count": sum(entry["status"] == "empty" for entry in columns),
        "filtered_numeric_column_count": sum(
            entry["status"] == "filtered_numeric" for entry in columns
        ),
        "filtered_uuid_column_count": sum(
            entry["status"] == "filtered_uuid" for entry in columns
        ),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["stats"] = _stats(manifest["columns"])
    sealed = _seal_content(manifest)
    manifest.clear()
    manifest.update(sealed)
    _atomic_json(path, manifest)


def _source_identity(
    database_id: str, connection_kwargs: dict[str, Any]
) -> dict[str, Any]:
    host = connection_kwargs.get("host")
    port = connection_kwargs.get("port")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("connection_kwargs must declare a non-empty host")
    try:
        normalized_port = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("connection_kwargs must declare a valid port") from exc
    if not 1 <= normalized_port <= 65535:
        raise ValueError("connection_kwargs must declare a valid port")
    for key in ("dbname", "database"):
        supplied_database = connection_kwargs.get(key)
        if supplied_database is not None and str(supplied_database) != database_id:
            raise ValueError(
                "connection_kwargs database must match the data item database_id"
            )
    return {"host": host, "port": normalized_port, "database": database_id}


def _connection_options(
    database_id: str, connection_kwargs: dict[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    kwargs = dict(connection_kwargs)
    kwargs.pop("database", None)
    kwargs["dbname"] = database_id
    kwargs["connect_timeout"] = min(timeout_seconds, 10)
    kwargs["autocommit"] = False
    milliseconds = timeout_seconds * 1000
    kwargs["options"] = (
        "-c default_transaction_read_only=on "
        "-c search_path=pg_catalog "
        f"-c statement_timeout={milliseconds} "
        f"-c lock_timeout={min(milliseconds, 5000)} "
        f"-c idle_in_transaction_session_timeout={milliseconds}"
    )
    return kwargs


def _expected_policy(max_values_per_column: int) -> dict[str, Any]:
    return {
        "algorithm_version": _ALGORITHM_VERSION,
        "max_values_per_column": max_values_per_column,
        "min_value_length": MIN_VALUE_LENGTH,
        "max_value_length": MAX_VALUE_LENGTH,
        "ordering": "md5(value) ascending, value ascending",
        "skipped_value_kinds": ["uuid", "numeric"],
    }


def _validate_complete_manifest_contract(
    manifest: Any, schema: Any, *, require_checkpoint_hashes: bool
) -> tuple[int, list[tuple[str, str]]]:
    if not isinstance(manifest, dict) or not isinstance(schema, dict):
        raise RuntimeError("Invalid database-value collection integrity")
    database_id = manifest.get("database_id")
    schema_hash = manifest.get("schema_hash")
    policy = manifest.get("policy")
    provenance = manifest.get("provenance")
    columns = manifest.get("columns")
    if (
        not isinstance(database_id, str)
        or not isinstance(schema_hash, str)
        or schema_hash != _fingerprint(schema)
        or schema.get("db_id", database_id) != database_id
        or manifest.get("status") != "complete"
        or not isinstance(policy, dict)
        or not isinstance(provenance, dict)
        or not isinstance(columns, list)
    ):
        raise RuntimeError("Invalid database-value collection integrity")
    cap = policy.get("max_values_per_column")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise RuntimeError("Invalid database-value collection integrity")
    if policy != _expected_policy(cap):
        raise RuntimeError("Invalid database-value collection integrity")
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Invalid database-value collection integrity")
    try:
        normalized_source = _source_identity(database_id, source)
    except ValueError as exc:
        raise RuntimeError("Invalid database-value collection integrity") from exc
    if source != normalized_source or provenance != {
        "collector": _ALGORITHM_VERSION,
        "source": source,
        "schema_file": "schema.json",
        "checkpoint_directory": "column_checkpoints",
    }:
        raise RuntimeError("Invalid database-value collection integrity")
    identity = {"schema_hash": schema_hash, "policy": policy, "source": source}
    if manifest.get("collection_fingerprint") != _fingerprint(identity):
        raise RuntimeError("Invalid database-value collection integrity")

    expected_columns = _eligible_columns(schema)
    actual_columns: list[tuple[str, str]] = []
    for entry in columns:
        validated = _validate_success_entry(
            entry, max_values_per_column=cap
        )
        actual_columns.append(
            (validated["table_name"], validated["column_name"])
        )
        checkpoint_hash = entry.get("checkpoint_hash")
        if require_checkpoint_hashes and not isinstance(checkpoint_hash, str):
            raise RuntimeError("Invalid database-value manifest integrity")
        if not require_checkpoint_hashes and checkpoint_hash is not None:
            raise RuntimeError("Legacy database-value collection is inconsistent")
    if actual_columns != expected_columns or len(set(actual_columns)) != len(actual_columns):
        raise RuntimeError("Invalid database-value collection integrity")
    if manifest.get("stats") != _stats(columns):
        raise RuntimeError("Invalid database-value manifest integrity")
    return cap, expected_columns


def verify_collection(output_dir: Path) -> dict:
    """Strictly verify a sealed schema, manifest, and every linked checkpoint."""

    root = Path(output_dir)
    schema = _read_json(root / "schema.json")
    manifest = _verify_content_hash(
        _read_json(root / "manifest.json"), artifact="database-value manifest"
    )
    cap, expected_columns = _validate_complete_manifest_contract(
        manifest, schema, require_checkpoint_hashes=True
    )
    manifest_by_column = {
        (entry["table_name"], entry["column_name"]): entry
        for entry in manifest["columns"]
    }
    expected_paths = {
        _checkpoint_path(root, table_name, column_name)
        for table_name, column_name in expected_columns
    }
    actual_paths = set((root / "column_checkpoints").glob("*.json"))
    if actual_paths != expected_paths:
        raise RuntimeError("Invalid database-value checkpoint integrity")
    for table_name, column_name in expected_columns:
        checkpoint = _verify_content_hash(
            _read_json(_checkpoint_path(root, table_name, column_name)),
            artifact="database-value checkpoint",
        )
        if checkpoint.get("collection_fingerprint") != manifest.get(
            "collection_fingerprint"
        ):
            raise RuntimeError("Invalid database-value checkpoint integrity")
        checkpoint_entry = _validate_success_entry(
            _checkpoint_entry(checkpoint),
            max_values_per_column=cap,
            table_name=table_name,
            column_name=column_name,
        )
        manifest_entry = manifest_by_column[(table_name, column_name)]
        if (
            any(manifest_entry[key] != checkpoint_entry[key] for key in _ENTRY_FIELDS)
            or manifest_entry["checkpoint_hash"] != _fingerprint(checkpoint)
        ):
            raise RuntimeError("Invalid database-value checkpoint integrity")
    return manifest


def seal_legacy_collection(output_dir: Path) -> dict:
    """Seal one trusted, existing v1 collection created during this run.

    This explicit migration is only for a one-time upgrade of known-origin legacy
    artifacts. Normal collection and verification never accept unsealed caches.
    """

    root = Path(output_dir)
    schema = _read_json(root / "schema.json")
    manifest = _read_json(root / "manifest.json")
    if isinstance(manifest, dict) and manifest.get("content_hash") is not None:
        return verify_collection(root)
    cap, expected_columns = _validate_complete_manifest_contract(
        manifest, schema, require_checkpoint_hashes=False
    )
    manifest_by_column = {
        (entry["table_name"], entry["column_name"]): entry
        for entry in manifest["columns"]
    }
    expected_paths = {
        _checkpoint_path(root, table_name, column_name)
        for table_name, column_name in expected_columns
    }
    actual_paths = set((root / "column_checkpoints").glob("*.json"))
    if actual_paths != expected_paths:
        raise RuntimeError("Legacy database-value collection is inconsistent")

    validated_checkpoints: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for table_name, column_name in expected_columns:
        path = _checkpoint_path(root, table_name, column_name)
        checkpoint = _verify_content_hash(
            _read_json(path),
            artifact="database-value checkpoint",
            allow_unsealed=True,
        )
        if checkpoint.get("collection_fingerprint") != manifest.get(
            "collection_fingerprint"
        ):
            raise RuntimeError("Legacy database-value collection is inconsistent")
        checkpoint_entry = _validate_success_entry(
            _checkpoint_entry(checkpoint),
            max_values_per_column=cap,
            table_name=table_name,
            column_name=column_name,
        )
        manifest_entry = manifest_by_column[(table_name, column_name)]
        if any(manifest_entry[key] != checkpoint_entry[key] for key in _ENTRY_FIELDS):
            raise RuntimeError("Legacy database-value collection is inconsistent")
        validated_checkpoints.append((path, checkpoint, checkpoint_entry))

    sealed_entries: list[dict[str, Any]] = []
    for path, checkpoint, checkpoint_entry in validated_checkpoints:
        sealed_checkpoint = _seal_content(checkpoint)
        _atomic_json(path, sealed_checkpoint)
        sealed_entries.append(
            {**checkpoint_entry, "checkpoint_hash": _fingerprint(sealed_checkpoint)}
        )
    manifest["columns"] = sealed_entries
    _write_manifest(root / "manifest.json", manifest)
    return verify_collection(root)


def collect_database_values(
    data_item,
    output_dir: Path,
    *,
    connection_kwargs: dict,
    max_values_per_column: int,
    timeout_seconds: int = 60,
) -> dict:
    """Collect deterministic samples and exact eligible counts for one database.

    Only public columns declared as text by the item's Meta schema are read. Values
    longer than 100 characters are ineligible, and native DeepEye numeric/UUID
    column filters are applied after deterministic bounded sampling.
    """

    if getattr(data_item, "db_type", None) != "postgresql":
        raise ValueError("Database-value collection requires db_type='postgresql'")
    database_id = getattr(data_item, "database_id", None)
    if not isinstance(database_id, str) or not database_id:
        raise ValueError("PostgreSQL data item must have a database_id")
    if isinstance(max_values_per_column, bool) or not isinstance(
        max_values_per_column, int
    ) or max_values_per_column <= 0:
        raise ValueError("max_values_per_column must be a positive integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    if not isinstance(connection_kwargs, dict):
        raise ValueError("connection_kwargs must be a dictionary")

    schema = getattr(data_item, "database_schema", None)
    if not isinstance(schema, dict):
        raise ValueError("PostgreSQL data item has a malformed database schema")
    eligible_columns = _eligible_columns(schema)
    schema_hash = _fingerprint(schema)
    source = _source_identity(database_id, connection_kwargs)
    policy = _expected_policy(max_values_per_column)
    identity = {
        "schema_hash": schema_hash,
        "policy": policy,
        "source": source,
    }
    collection_fingerprint = _fingerprint(identity)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    schema_path = root / "schema.json"
    manifest_path = root / "manifest.json"
    if schema_path.exists():
        if _fingerprint(_read_json(schema_path)) != schema_hash:
            raise ValueError("Existing database-value cache configuration does not match")
    else:
        _atomic_json(schema_path, schema)

    existing_manifest: dict[str, Any] | None = None
    existing_columns: dict[tuple[str, str], dict[str, Any]] = {}
    if manifest_path.exists():
        existing_manifest = _verify_content_hash(
            _read_json(manifest_path), artifact="database-value manifest"
        )
        if existing_manifest.get("collection_fingerprint") != collection_fingerprint:
            raise ValueError("Existing database-value cache configuration does not match")
        if existing_manifest.get("status") == "complete":
            return verify_collection(root)
        raw_columns = existing_manifest.get("columns")
        if not isinstance(raw_columns, list):
            raise RuntimeError("Invalid database-value manifest integrity")
        for entry in raw_columns:
            if not isinstance(entry, dict):
                raise RuntimeError("Invalid database-value manifest integrity")
            key = (entry.get("table_name"), entry.get("column_name"))
            if key in existing_columns:
                raise RuntimeError("Invalid database-value manifest integrity")
            existing_columns[key] = entry

    entries: list[dict[str, Any]] = []
    pending: list[tuple[str, str, Path]] = []
    for table_name, column_name in eligible_columns:
        checkpoint_path = _checkpoint_path(root, table_name, column_name)
        if checkpoint_path.exists():
            checkpoint = _verify_content_hash(
                _read_json(checkpoint_path), artifact="database-value checkpoint"
            )
            if (
                checkpoint.get("collection_fingerprint") != collection_fingerprint
            ):
                raise RuntimeError("Invalid database-value checkpoint integrity")
            entry = _validate_success_entry(
                _checkpoint_entry(checkpoint),
                max_values_per_column=max_values_per_column,
                table_name=table_name,
                column_name=column_name,
            )
            checkpoint_hash = _fingerprint(checkpoint)
            prior_entry = existing_columns.get((table_name, column_name))
            if prior_entry is not None and prior_entry.get("status") in _SUCCESS_STATUSES:
                if (
                    any(prior_entry.get(key) != entry[key] for key in _ENTRY_FIELDS)
                    or prior_entry.get("checkpoint_hash") != checkpoint_hash
                ):
                    raise RuntimeError("Invalid database-value checkpoint integrity")
            entry["checkpoint_hash"] = checkpoint_hash
            entries.append(entry)
        else:
            prior_entry = existing_columns.get((table_name, column_name))
            if prior_entry is not None and prior_entry.get("status") in _SUCCESS_STATUSES:
                raise RuntimeError("Invalid database-value checkpoint integrity")
            entries.append({
                "table_name": table_name,
                "column_name": column_name,
                "eligible_distinct_count": 0,
                "documents": [],
                "status": "pending",
            })
            pending.append((table_name, column_name, checkpoint_path))

    manifest = {
        "database_id": database_id,
        "schema_hash": schema_hash,
        "collection_fingerprint": collection_fingerprint,
        "policy": policy,
        "status": "complete" if not pending else "in_progress",
        "columns": entries,
        "stats": {},
        "provenance": {
            "collector": _ALGORITHM_VERSION,
            "source": source,
            "schema_file": "schema.json",
            "checkpoint_directory": "column_checkpoints",
        },
    }
    _write_manifest(manifest_path, manifest)
    if not pending:
        return verify_collection(root)

    connection = cursor = None
    try:
        try:
            connection = psycopg.connect(
                **_connection_options(database_id, connection_kwargs, timeout_seconds)
            )
            cursor = connection.cursor()
        except Exception:
            for entry in entries:
                if entry["status"] == "pending":
                    entry["status"] = "failed"
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"Failed to connect for database-value collection: {database_id}"
            ) from None
        positions = {
            (entry["table_name"], entry["column_name"]): position
            for position, entry in enumerate(entries)
        }
        for table_name, column_name, checkpoint_path in pending:
            position = positions[(table_name, column_name)]
            try:
                cursor.execute(
                    _value_query(table_name, column_name),
                    (max_values_per_column,),
                )
                entry = _classify_column(
                    table_name, column_name, cursor.fetchall(), max_values_per_column
                )
            except Exception:
                entries[position] = {
                    "table_name": table_name,
                    "column_name": column_name,
                    "eligible_distinct_count": 0,
                    "documents": [],
                    "status": "failed",
                }
                manifest["status"] = "failed"
                _write_manifest(manifest_path, manifest)
                raise RuntimeError(
                    f"Failed to collect database values for {table_name}.{column_name}"
                ) from None
            checkpoint = _seal_content(
                {**entry, "collection_fingerprint": collection_fingerprint}
            )
            _atomic_json(checkpoint_path, checkpoint)
            entries[position] = {
                **entry,
                "checkpoint_hash": _fingerprint(checkpoint),
            }
            _write_manifest(manifest_path, manifest)
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
        return verify_collection(root)
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
