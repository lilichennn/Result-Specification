import datetime as dt
from decimal import Decimal
import importlib
import importlib.util
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

import psycopg
from scripts.baseline_adapters.deepeye.run_store import to_jsonable, restore_jsonable


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        name = "scripts.baseline_adapters.dail_sql.execution"
        self.assertIsNotNone(importlib.util.find_spec(name), "Task5 SQL executor missing")
        self.api = importlib.import_module(name)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "bound.sqlite"
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE t(id INTEGER)")
            conn.executemany("INSERT INTO t VALUES (?)", [(1,), (1,), (None,)])
        self.db = {"dialect": "sqlite", "path": str(self.path), "database_id": "bound"}

    def test_sqlite_complete_results_empty_duplicate_names_and_null(self):
        result = self.api.execute_sql(self.db, "SELECT id AS same, id AS same FROM t", timeout_seconds=1)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["columns"], ["same", "same"])
        self.assertEqual(result["rows"], [(1, 1), (1, 1), (None, None)])
        self.assertEqual(result["column_count"], 2)
        empty = self.api.execute_sql(self.db, "SELECT id FROM t WHERE 0", timeout_seconds=1)
        self.assertEqual((empty["rows"], empty["column_count"]), ([], 1))
        null = self.api.execute_sql(self.db, "SELECT NULL, NULL", timeout_seconds=1)
        self.assertEqual(null["rows"], [(None, None)])
        self.assertEqual(null["value_types"], [["builtins.NoneType", "builtins.NoneType"]])
        large = self.api.execute_sql(self.db, "WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<10050) SELECT n FROM x", timeout_seconds=2)
        self.assertEqual(len(large["rows"]), 10050)

    def test_readonly_database_mechanism_and_single_statement(self):
        for sql in ("INSERT INTO t VALUES (9)", "DELETE FROM t RETURNING id", "DROP TABLE t",
                    "PRAGMA query_only=OFF", "ATTACH DATABASE ':memory:' AS other", "SELECT 1; SELECT 2", "SELEC 1"):
            with self.subTest(sql=sql):
                result = self.api.execute_sql(self.db, sql, timeout_seconds=1)
                self.assertEqual(result["status"], "error")
                self.assertTrue(result["error"])
        self.assertEqual(self.api.execute_sql(self.db, "SELECT count(*) FROM t", timeout_seconds=1)["rows"], [(3,)])
        self.assertEqual(self.api.execute_sql(self.db, "SELECT name FROM sqlite_master WHERE type='table'", timeout_seconds=1)["rows"], [("t",)])

    def test_recursive_sql_is_really_interrupted_and_connection_released(self):
        started = time.monotonic()
        result = self.api.execute_sql(self.db, "WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x", timeout_seconds=.025)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(time.monotonic() - started, 1)
        with sqlite3.connect(self.path, timeout=.1) as conn:
            conn.execute("BEGIN EXCLUSIVE")
            conn.execute("INSERT INTO t VALUES(2)")

    def test_pg_extended_protocol_readonly_and_typed_full_results(self):
        row = (Decimal("1.200"), dt.date(2020, 1, 2), dt.time(1, 2, 3), dt.datetime(2020, 1, 2),
               uuid.UUID(int=3), b"\x00\xff", {"a": [1, None]}, [1, 2], dt.timedelta(days=-3, seconds=7, microseconds=13))
        conn = FakeConnection([row] * 10050)
        with patch.dict(os.environ, {"PG_HOST": "host", "PG_PORT": "5432", "PG_USER": "evaluator", "PG_PASSWORD": "SECRET"}), patch.object(psycopg, "connect", return_value=conn) as connect:
            result = self.api.execute_sql({"dialect": "postgresql", "database_id": "ordinary", "path": None}, "SELECT types", timeout_seconds=.125)
        kwargs = connect.call_args.kwargs
        self.assertEqual(kwargs["dbname"], "ordinary")
        self.assertEqual(kwargs["user"], "evaluator")
        self.assertGreater(kwargs["connect_timeout"], 0)
        self.assertIn("statement_timeout=125", kwargs["options"])
        self.assertIn("default_transaction_read_only=on", kwargs["options"])
        self.assertTrue(conn.read_only)
        self.assertEqual(conn.cur.executed, [("SELECT types", True)])
        self.assertTrue(conn.closed and conn.rolled_back and conn.cur.closed)
        self.assertEqual(result["rows"], [row] * 10050)
        self.assertEqual(restore_jsonable(to_jsonable(result)), result)

    def test_pg_timeout_and_connection_errors_cleanup_without_secrets(self):
        conn = FakeConnection([], psycopg.errors.QueryCanceled("SECRET"))
        with patch.dict(os.environ, {"PG_USER": "evaluator"}), patch.object(psycopg, "connect", return_value=conn):
            result = self.api.execute_sql({"dialect": "postgresql", "database_id": "db"}, "SELECT slow", timeout_seconds=.01)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(conn.closed and conn.rolled_back and conn.cur.closed)
        with patch.dict(os.environ, {"PG_USER": "evaluator"}), patch.object(psycopg, "connect", side_effect=psycopg.OperationalError("password=SECRET")):
            result = self.api.execute_sql({"dialect": "postgresql", "database_id": "db"}, "SELECT 1", timeout_seconds=1)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("SECRET", str(result))


class FakeCursor:
    def __init__(self, rows, error):
        self.rows, self.error = rows, error
        self.description = [(f"c{i}", 1700, None, None, None, None, None) for i in range(len(rows[0]) if rows else 1)]
        self.executed = []
        self.closed = False

    def execute(self, sql, *, prepare):
        self.executed.append((sql, prepare))
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, rows, error=None):
        self.cur = FakeCursor(rows, error)
        self.closed = self.rolled_back = self.read_only = False

    def cursor(self):
        return self.cur

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True
