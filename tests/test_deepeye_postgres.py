"""Offline PG adapter tests: only the external database connection is replaced."""
import importlib
import importlib.util
import os
import shlex
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psycopg
from psycopg.pq import DiagnosticField

BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "DeepEye-SQL"
sys.path.insert(0, str(BASELINE))
SCHEMA = {"db_type": "postgresql", "tables": {
    "funds": {"columns": {"id": {"column_type": "INTEGER"}, "fundclass": {"column_type": "JSONB"}}},
    "Prices": {"columns": {"Fund ID": {"column_type": "INTEGER"}, "Value": {"column_type": "NUMERIC"}}},
}}


class Cursor:
    def __init__(self, connection, rows, error=None, fetch_error=None):
        self.connection = connection
        self.fetch_error = fetch_error
        self.read_only_at_query = None
        self.rows, self.error, self.commands, self.closed = rows, error, [], False
        self.description = [("id",)]
    def execute(self, sql, params=None, *, prepare=None):
        self.commands.append((sql, params, prepare))
        if sql != "BEGIN READ ONLY":
            self.read_only_at_query = self.connection.read_only
            if self.error:
                raise self.error
    def fetchmany(self, size):
        return self.rows[:size]
    def fetchall(self):
        if self.fetch_error:
            raise self.fetch_error
        return list(self.rows)
    def close(self):
        self.closed = True


class Connection:
    def __init__(self, rows=(), error=None, fetch_error=None):
        self.read_only = None
        self.handle = Cursor(self, rows, error, fetch_error)
        self.rolled_back = self.closed = False
    def cursor(self):
        return self.handle
    def rollback(self):
        self.rolled_back = True
    def close(self):
        self.closed = True


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        name = "scripts.baseline_adapters.deepeye.postgres_execution"
        self.assertIsNotNone(importlib.util.find_spec(name), "PostgreSQL execution is not implemented")
        self.backend = importlib.import_module(name)
        self.item = SimpleNamespace(db_type="postgresql", database_id="smoke", database_schema=SCHEMA)

    def test_original_sql_and_physical_columns_reach_database_without_meta(self):
        """A new Meta check, AST rewrite or outer LIMIT must fail this test."""
        self.item = SimpleNamespace(db_type="postgresql", database_id="smoke")
        sql = '  SELECT f.* FROM funds AS f; -- original text\n'
        connection = Connection([(1, "physical-only")])
        connection.handle.description = [("id",), ("geozone",)]
        with patch("psycopg.connect", return_value=connection):
            result = self.backend.execute_postgres_sql(self.item, sql)
        self.assertEqual(result.result_type, "success")
        self.assertEqual(result.sql, sql)
        self.assertEqual(result.result_cols, ["id", "geozone"])
        self.assertEqual(result.result_rows, [(1, "physical-only")])
        self.assertEqual(connection.handle.commands[-1], (sql, None, True))

    def test_postgres_syntax_is_submitted_unchanged_not_approved_by_python(self):
        """Transport test only: actual SQL semantics belong to the live suite."""
        queries = [
            "SELECT ARRAY_REMOVE(ARRAY[1,NULL,2], NULL), CARDINALITY(ARRAY[1,2])",
            "SELECT REGEXP_REPLACE('a1b2', '[0-9]', '', 'g')",
            "SELECT ROW_TO_JSON(t) FROM (VALUES (1, 'x')) AS t(id,label)",
            "SELECT * FROM UNNEST(ARRAY[1,2]) AS u(x)",
            "SELECT * FROM (VALUES (1)) AS t(x) CROSS JOIN LATERAL (SELECT t.x+1) AS u",
            "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<3) SELECT * FROM x",
            "SELECT COALESCE(fundclass -> 'solar' ->> 'watts', '') FROM funds",
            "SELECT id IS DISTINCT FROM NULL FROM funds",
            "SELECT fundclass AS id FROM funds ORDER BY (((id)))",
            "SELECT id FROM funds ORDER BY CASE id WHEN 1 THEN 0 ELSE 1 END",
            "SELECT unlisted_column FROM other_schema.unlisted_table",
        ]
        for sql in queries:
            connection = Connection([(1,)])
            with self.subTest(sql=sql), patch("psycopg.connect", return_value=connection):
                result = self.backend.execute_postgres_sql(self.item, sql)
                self.assertEqual(result.result_type, "success")
                self.assertEqual(connection.handle.commands[-1], (sql, None, True))

    def test_readonly_timeouts_environment_and_cleanup(self):
        connection = Connection([(1,)])
        with patch.dict(os.environ, {"PG_HOST": "localhost", "PG_PORT": "5433", "PG_USER": "reader", "PG_PASSWORD": "secret"}), patch("psycopg.connect", return_value=connection) as connect:
            result = self.backend.execute_postgres_sql(self.item, "SELECT id FROM funds", timeout=7)
        self.assertEqual(result.result_type, "success")
        self.assertEqual(result.result_rows, [(1,)])
        kwargs = connect.call_args.kwargs
        self.assertEqual((kwargs["dbname"], kwargs["host"], kwargs["port"], kwargs["user"]), ("smoke", "localhost", 5433, "reader"))
        self.assertGreater(kwargs["connect_timeout"], 0)
        self.assertIn("default_transaction_read_only=on", kwargs["options"])
        self.assertIn("statement_timeout=7000", kwargs["options"])
        self.assertIn("lock_timeout=", kwargs["options"])
        self.assertIn("search_path=pg_catalog,public", shlex.split(kwargs["options"]))
        self.assertFalse(kwargs["autocommit"])
        self.assertTrue(connection.handle.read_only_at_query)
        self.assertIn("standard_conforming_strings=on", kwargs["options"])
        self.assertTrue(connection.rolled_back and connection.closed and connection.handle.closed)

    def test_classification_and_full_results_past_previous_row_limit(self):
        for rows, expected in [([], "empty_result"), ([(None,)], "all_null_result"),
                               ([(2,), (1,), (1,), (None,)] * 1501, "success")]:
            with self.subTest(expected=expected, count=len(rows)), patch("psycopg.connect", return_value=Connection(rows)):
                result = self.backend.execute_postgres_sql(self.item, "SELECT id FROM funds")
                self.assertEqual(result.result_type, expected)
                self.assertEqual(result.result_rows, rows)

    def test_postgres_error_code_primary_and_hint_reach_native_feedback(self):
        primary = 'column cr.coregistry does not exist'
        hint = 'Perhaps you meant to reference the column "cr.coreregistry".'
        error = psycopg.errors.UndefinedColumn("unused full exception text", info={
            DiagnosticField.MESSAGE_PRIMARY: primary.encode(),
            DiagnosticField.MESSAGE_HINT: hint.encode(),
            DiagnosticField.CONTEXT: b"server internal context marker",
        })
        connection = Connection(error=error)
        with patch("psycopg.connect", return_value=connection):
            result = self.backend.execute_postgres_sql(self.item, 'SELECT cr.coregistry FROM core_record cr')
        self.assertEqual(result.result_type, "execution_error")
        for expected in ("42703", primary, hint):
            self.assertIn(expected, result.error_message)
            self.assertIn(expected, result.result_table_str)
        self.assertNotIn("internal context", result.error_message)
        self.assertIsNone(result.result_rows)
        self.assertTrue(connection.rolled_back and connection.closed and connection.handle.closed)

    def test_database_readonly_and_multi_statement_errors_are_preserved(self):
        for sql, error, code, message in [
            ("DELETE FROM funds RETURNING id",
             psycopg.errors.ReadOnlySqlTransaction("cannot execute DELETE in a read-only transaction"),
             "25006", "read-only transaction"),
            ("SELECT 1; SELECT 2",
             psycopg.errors.SyntaxError("cannot insert multiple commands into a prepared statement"),
             "42601", "multiple commands"),
        ]:
            connection = Connection(error=error)
            with self.subTest(code=code), patch("psycopg.connect", return_value=connection):
                result = self.backend.execute_postgres_sql(self.item, sql)
                self.assertTrue(connection.handle.commands, "SQL did not reach the database boundary")
                self.assertEqual(connection.handle.commands[-1], (sql, None, True))
                self.assertEqual(result.result_type, "execution_error")
                self.assertIn(code, result.error_message)
                self.assertIn(message, result.error_message)
                self.assertTrue(connection.rolled_back and connection.closed)

    def test_non_row_statement_does_not_become_success(self):
        connection = Connection()
        connection.handle.description = None
        with patch("psycopg.connect", return_value=connection):
            result = self.backend.execute_postgres_sql(self.item, "SET LOCAL application_name = 'probe'")
        self.assertEqual(result.result_type, "execution_error")
        self.assertIn("did not return", result.error_message.lower())
        self.assertIsNone(result.result_rows)
        self.assertTrue(connection.rolled_back and connection.closed and connection.handle.closed)

    def test_connection_errors_hide_credentials_and_identify_connection_failure(self):
        for error in [psycopg.OperationalError("connect failed password=secret"),
                      psycopg.errors.InvalidPassword('password authentication failed for user "private"')]:
            with self.subTest(error=type(error).__name__), patch("psycopg.connect", side_effect=error):
                result = self.backend.execute_postgres_sql(self.item, "SELECT 1")
                self.assertEqual(result.result_type, "execution_error")
                self.assertIn("connection", result.error_message.lower())
                self.assertNotIn("secret", result.error_message)
                self.assertNotIn("private", result.error_message)

    def test_fetch_failure_never_returns_a_partial_success(self):
        connection = Connection([(1,)], fetch_error=RuntimeError("password secret"))
        with patch("psycopg.connect", return_value=connection):
            result = self.backend.execute_postgres_sql(self.item, "SELECT id FROM funds")
        self.assertEqual(result.result_type, "execution_error")
        self.assertIsNone(result.result_rows)
        self.assertNotIn("secret", result.error_message)
        self.assertTrue(connection.rolled_back and connection.closed and connection.handle.closed)

    def test_timeout_bounds_and_ssl_mode_remain_effective(self):
        for requested, expected in [(0, 1), (7, 7), (1200, 600)]:
            with self.subTest(timeout=requested), patch.dict(os.environ, {"PG_SSLMODE": "require"}), patch("psycopg.connect", return_value=Connection([(1,)])) as connect:
                self.backend.execute_postgres_sql(self.item, "SELECT 1", timeout=requested)
                self.assertIn(f"statement_timeout={expected * 1000}", shlex.split(connect.call_args.kwargs["options"]))
                self.assertEqual(connect.call_args.kwargs["sslmode"], "require")
                self.assertEqual(connect.call_args.kwargs["connect_timeout"], min(expected, 10))

    def test_errors_do_not_expose_connection_or_physical_schema_details(self):
        connection = Connection(error=RuntimeError("password secret; physical column geozone"))
        with patch("psycopg.connect", return_value=connection):
            result = self.backend.execute_postgres_sql(self.item, "SELECT id FROM funds")
        self.assertEqual(result.result_type, "execution_error")
        self.assertNotIn("secret", result.error_message)
        self.assertNotIn("geozone", result.error_message)
        self.assertTrue(connection.rolled_back and connection.closed)

    def test_statement_timeout_classification_and_cleanup(self):
        import psycopg
        connection = Connection(error=psycopg.errors.QueryCanceled("private server diagnostics"))
        with patch("psycopg.connect", return_value=connection):
            result = self.backend.execute_postgres_sql(self.item, "SELECT id FROM funds", timeout=2)
        self.assertEqual(result.result_type, "timeout")
        self.assertIsNone(result.result_rows)
        self.assertNotIn("private", result.error_message)
        self.assertTrue(connection.rolled_back and connection.closed and connection.handle.closed)

    def test_execution_and_timing_route_postgres_without_sqlite(self):
        from app.db_utils import execution
        from scripts.baseline_adapters.deepeye import backend_hooks as hooks
        execute_sql_for_data_item = hooks.make_execute_sql_for_data_item(execution.execute_sql_for_data_item)
        measure_execution_time_for_data_item = hooks.make_measure_execution_time_for_data_item(execution.measure_execution_time_for_data_item)
        with patch("psycopg.connect", side_effect=lambda **kw: Connection([(1,)])), patch("sqlite3.connect", side_effect=AssertionError("no SQLite")):
            self.assertEqual(execute_sql_for_data_item(self.item, "SELECT id FROM funds").result_type, "success")
            measured = measure_execution_time_for_data_item(self.item, "SELECT id FROM funds", repeat=2)
            self.assertLess(measured, float("inf"))

    def test_service_executes_and_times_queries_even_after_meta_scope_narrows(self):
        from app.services import execution_service
        from scripts.baseline_adapters.deepeye import backend_hooks as hooks
        ExecutionService = execution_service.ExecutionService
        self.item.database_path = "smoke"
        service = ExecutionService()
        with patch.multiple(execution_service,
                execute_sql_for_data_item=hooks.make_execute_sql_for_data_item(execution_service.execute_sql_for_data_item),
                measure_execution_time_for_data_item=hooks.make_measure_execution_time_for_data_item(execution_service.measure_execution_time_for_data_item)), \
             patch.object(ExecutionService, "_build_result_key", staticmethod(hooks.make_result_cache_key(ExecutionService._build_result_key))), \
             patch.object(ExecutionService, "_build_time_key", staticmethod(hooks.make_time_cache_key(ExecutionService._build_time_key))), \
             patch("psycopg.connect", side_effect=lambda **kw: Connection([(1,)])):
            self.assertEqual(service.execute(self.item, "SELECT fundclass FROM funds").result_type, "success")
            self.assertLess(service.measure_time(self.item, "SELECT fundclass FROM funds", repeat=1), float("inf"))
            self.item.database_schema = {"tables": {"funds": {"columns": {"id": {}}}}}
            self.assertEqual(service.execute(self.item, "SELECT fundclass FROM funds").result_type, "success")
            self.assertLess(service.measure_time(self.item, "SELECT fundclass FROM funds", repeat=1), float("inf"))

    def test_postgres_result_hash_accepts_native_decimal_date_and_json(self):
        from app.pipeline import utils
        from scripts.baseline_adapters.deepeye.backend_hooks import make_execution_result_hash
        get_execution_result_hash = make_execution_result_hash(utils.get_execution_result_hash)
        from datetime import date
        from decimal import Decimal
        rows = [(Decimal("1.23"), date(2026, 9, 11), {"x": [1, None]})]
        actual = get_execution_result_hash(self.item, rows)
        self.assertEqual(actual, frozenset({(Decimal("1.23"), date(2026, 9, 11), (("x", (1, None)),))}))

    def test_postgres_with_instance_id_preserves_column_order_and_nulls(self):
        from app.pipeline import utils
        from scripts.baseline_adapters.deepeye.backend_hooks import make_execution_result_hash
        get_execution_result_hash = make_execution_result_hash(utils.get_execution_result_hash)
        self.item.instance_id = "bird_interact_example"
        self.assertNotEqual(get_execution_result_hash(self.item, [(1, 2)]), get_execution_result_hash(self.item, [(2, 1)]))
        self.assertNotEqual(get_execution_result_hash(self.item, [(None,)]), get_execution_result_hash(self.item, [(0,)]))

    def test_external_hooks_preserve_non_postgres_dispatch_arguments(self):
        from scripts.baseline_adapters.deepeye import backend_hooks as hooks
        item = SimpleNamespace(db_type="sqlite")
        def original(*args, **kwargs):
            return args, kwargs
        args, kwargs = hooks.make_execute_sql_for_data_item(original)(item, "SELECT 1", timeout=7, bigquery_credential_path="credential-marker")
        self.assertEqual(args, (item, "SELECT 1"))
        self.assertEqual(kwargs, {"timeout": 7, "bigquery_credential_path": "credential-marker"})
        args, kwargs = hooks.make_measure_execution_time_for_data_item(original)(item, "SELECT 1", 7, 3, 0.2)
        self.assertEqual(args, (item, "SELECT 1"))
        self.assertEqual(kwargs, {"timeout": 7, "repeat": 3, "initial_execution_time": 0.2})
        self.assertEqual(hooks.make_result_cache_key(original)(item, "SELECT 1", 7), ((item, "SELECT 1", 7), {}))
        self.assertEqual(hooks.make_time_cache_key(original)(item, "SELECT 1", 7, 3), ((item, "SELECT 1", 7, 3), {}))
        self.assertEqual(hooks.make_execution_result_hash(original)(item, [(1,)]), ((item, [(1,)]), {}))


if __name__ == "__main__":
    unittest.main()
