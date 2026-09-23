"""Single-statement, database-enforced read-only SQL execution.

Call on the campaign's SQL executor, never the HTTP event loop. PostgreSQL
follows the existing evaluator's transaction/protocol/cleanup strategy without
importing DeepEye result or model classes. A restricted PG role is required;
read-only transactions are not a sandbox for privileged server functions.
"""
import math
import os
from pathlib import Path
import sqlite3
import time

import psycopg


def _type_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def execute_sql(database: dict, sql: str, *, timeout_seconds: float) -> dict:
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("SQL timeout must be positive and finite")
    dialect = database["dialect"]
    if dialect not in ("sqlite", "postgresql"):
        raise ValueError("Unsupported SQL dialect")
    started = time.monotonic()
    result = {"status": "error", "dialect": dialect, "database_id": database.get("database_id"),
              "sql": sql, "timeout_seconds": timeout_seconds, "rows": [], "columns": [],
              "column_count": 0, "column_types": [], "value_types": [], "elapsed_seconds": 0.0, "error": None}
    connection = cursor = None
    deadline = started + timeout_seconds
    try:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Empty SQL")
        if dialect == "sqlite":
            path = Path(database["path"]).resolve(strict=True)
            connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds)
            connection.execute("PRAGMA query_only=ON")
            # Database operation authorization, not a table/column whitelist.
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            connection.set_authorizer(lambda operation, *args: sqlite3.SQLITE_OK if operation in allowed else sqlite3.SQLITE_DENY)
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = connection.cursor()
            cursor.execute(sql)  # sqlite3 rejects multiple statements.
        else:
            # Environment is installed once by the caller. Never infer an admin
            # account or place connection credentials in task/event payloads.
            user = os.environ.get("PG_USER")
            if not user:
                raise ValueError("PG_USER is required")
            millis = max(1, math.ceil(timeout_seconds * 1000))
            kwargs = {"dbname": database["database_id"], "host": os.environ.get("PG_HOST", "localhost"),
                      "port": int(os.environ.get("PG_PORT", "5432")), "user": user,
                      "password": os.environ.get("PG_PASSWORD", ""),
                      "connect_timeout": max(1, min(10, math.ceil(timeout_seconds))), "autocommit": False,
                      "options": "-c default_transaction_read_only=on -c search_path=pg_catalog,public "
                                 "-c standard_conforming_strings=on "
                                 f"-c statement_timeout={millis} -c lock_timeout={min(millis, 5000)} "
                                 f"-c idle_in_transaction_session_timeout={millis}"}
            if os.environ.get("PG_SSLMODE"):
                kwargs["sslmode"] = os.environ["PG_SSLMODE"]
            connection = psycopg.connect(**kwargs)
            connection.read_only = True
            cursor = connection.cursor()
            cursor.execute(sql, prepare=True)  # Extended protocol: one statement.
        if cursor.description is None:
            raise ValueError("Statement did not return a result table")
        result["columns"] = [description[0] for description in cursor.description]
        result["column_types"] = [description[1] for description in cursor.description]
        result["column_count"] = len(cursor.description)
        result["rows"] = [tuple(row) for row in cursor.fetchall()]
        result["value_types"] = [[_type_name(value) for value in row] for row in result["rows"]]
        result["status"] = "success"
    except psycopg.errors.QueryCanceled:
        result.update(status="timeout", error={"type": "QueryCanceled", "sqlstate": "57014", "message": "PostgreSQL statement timeout or query cancellation"})
    except psycopg.Error as exc:
        code = exc.sqlstate
        message = "PostgreSQL connection failed; connection details withheld"
        if connection is not None and code and code[:2] not in ("08", "28"):
            message = exc.diag.message_primary or f"PostgreSQL error {code}"
        result["error"] = {"type": type(exc).__name__, "sqlstate": code, "message": message}
    except sqlite3.Error as exc:
        timed_out = getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT and time.monotonic() >= deadline
        result.update(status="timeout" if timed_out else "error", error={"type": type(exc).__name__, "message": str(exc), "sqlite_errorcode": getattr(exc, "sqlite_errorcode", None)})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc) if dialect == "sqlite" else "PostgreSQL adapter failed; check database and PG environment configuration"}
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
        result["elapsed_seconds"] = time.monotonic() - started
    return result
