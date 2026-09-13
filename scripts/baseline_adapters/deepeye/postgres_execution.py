"""Execute original PostgreSQL SQL with database-level read-only protection.

Meta describes model inputs, not the SQL execution scope. PostgreSQL performs
parsing, name resolution and permission checks; this adapter does not rewrite
SQL or maintain a list of supported functions or syntax.
"""
import os
import time

import psycopg

from app.db_utils.defaults import DEFAULT_SQL_EXECUTION_TIMEOUT
from app.db_utils.execution import SQLExecutionResult, _build_sql_execution_result


MAX_TIMEOUT_SECONDS = 600


def _postgres_error_message(error: psycopg.Error, *, connection_failed: bool) -> str:
    """Keep query diagnostics useful without forwarding connection strings.

    Primary messages and hints identify missing columns, type errors, permissions,
    etc. Full exception strings on connection errors can contain host/user details;
    server contexts and tracebacks are not part of the model's execution feedback.
    """
    code = error.sqlstate
    if connection_failed or (code and code[:2] in {"08", "28"}):
        return "PostgreSQL connection failed; check connection settings and server availability"
    if not code:
        return f"PostgreSQL execution failed ({type(error).__name__}); no server SQLSTATE available"
    primary = error.diag.message_primary or str(error)
    message = f"PostgreSQL [{code}]: {primary}"
    if error.diag.message_hint:
        message += f"\nHINT: {error.diag.message_hint}"
    return message


def execute_postgres_sql(data_item, sql: str, timeout: int | None = None) -> SQLExecutionResult:
    """Submit one original statement and return its full result in native form.

    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD and optional PG_SSLMODE are read
    only at execution time. database_id selects the database, not PG_DATABASE.
    Each call owns and rolls back/closes its connection, including on failure.
    No SQL or exception text is printed here; the run recorder owns artifacts.
    The database role must have appropriate privileges: read-only transactions
    are not a sandbox for arbitrary privileged functions.
    """
    started = time.perf_counter()
    database_id = str(getattr(data_item, "database_id", ""))
    connection = cursor = None
    seconds = MAX_TIMEOUT_SECONDS
    try:
        if not database_id:
            raise ValueError("Missing database identifier")
        seconds = min(MAX_TIMEOUT_SECONDS, max(1, int(timeout if timeout is not None else DEFAULT_SQL_EXECUTION_TIMEOUT)))
        # Builtins resolve before public objects, while normal unqualified
        # benchmark table names work without adding public. to the input SQL.
        kwargs = {
            "dbname": database_id,
            "host": os.environ.get("PG_HOST", "localhost"),
            "port": int(os.environ.get("PG_PORT", "5432")),
            "user": os.environ.get("PG_USER", "postgres"),
            "password": os.environ.get("PG_PASSWORD", ""),
            "connect_timeout": min(seconds, 10),
            "options": ("-c default_transaction_read_only=on -c search_path=pg_catalog,public "
                        "-c standard_conforming_strings=on "
                        f"-c statement_timeout={seconds * 1000} "
                        f"-c lock_timeout={min(seconds * 1000, 5000)} "
                        f"-c idle_in_transaction_session_timeout={seconds * 1000}"),
            "autocommit": False,
        }
        if os.environ.get("PG_SSLMODE"):
            kwargs["sslmode"] = os.environ["PG_SSLMODE"]
        connection = psycopg.connect(**kwargs)
        # Psycopg begins the next transaction with READ ONLY before the query.
        connection.read_only = True
        cursor = connection.cursor()
        # Extended protocol rejects multiple statements on the server, including
        # a COMMIT followed by another query. No Python SQL splitting/parsing.
        cursor.execute(sql, prepare=True)
        if cursor.description is None:
            return _build_sql_execution_result(db_path=database_id, sql=sql,
                execution_time=time.perf_counter() - started,
                error_message="PostgreSQL statement did not return a result table")
        # Match the native executor's full-result semantics. Do not silently
        # truncate rows or classify a valid large result as a SQL error.
        rows = cursor.fetchall()
        return _build_sql_execution_result(db_path=database_id, sql=sql,
            execution_time=time.perf_counter() - started,
            result_cols=[description[0] for description in cursor.description],
            result_rows=[tuple(row) for row in rows])
    except psycopg.errors.QueryCanceled:
        return SQLExecutionResult(result_type="timeout", db_path=database_id, sql=sql,
            execution_time=time.perf_counter() - started,
            error_message=(f"SQL execution timed out after {seconds} seconds "
                           "(PostgreSQL 57014: statement timeout or query cancellation)"))
    except psycopg.Error as exc:
        error = _postgres_error_message(exc, connection_failed=connection is None)
    except Exception as exc:
        error = f"PostgreSQL adapter failed ({type(exc).__name__}); connection details are withheld"
    finally:
        # Cleanup must also run after timeout, failed execute/fetch and return.
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
    return _build_sql_execution_result(db_path=database_id, sql=sql,
        execution_time=time.perf_counter() - started, error_message=error)
