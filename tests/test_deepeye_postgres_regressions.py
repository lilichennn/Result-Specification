"""Opt-in PostgreSQL execution regressions; no model calls or database fixtures.

Default collection skips every test without importing the database adapter.
To run against an existing PostgreSQL database, explicitly set
DEEPEYE_TEST_PG=1 and DEEPEYE_TEST_PG_DATABASE to its database name. The adapter
reads the existing PG_HOST, PG_PORT, PG_USER, PG_PASSWORD and PG_SSLMODE
environment variables. This module never loads .env or prints credentials.

From the repository root, with connection variables already configured:

    DEEPEYE_TEST_PG=1 DEEPEYE_TEST_PG_DATABASE=your_existing_database \
        code/.venv/bin/python -m unittest discover -s code/tests \
        -p test_deepeye_postgres_regressions.py -v

All queries use constants, VALUES, CTEs, builtins or read-only catalogs. They
create no objects and require no benchmark tables. The timeout regression
requests pg_sleep(2) with a one-second execution timeout.
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


@unittest.skipUnless(
    os.environ.get("DEEPEYE_TEST_PG") == "1",
    "Real PostgreSQL tests require DEEPEYE_TEST_PG=1 and DEEPEYE_TEST_PG_DATABASE",
)
class PostgresExecutionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database = os.environ.get("DEEPEYE_TEST_PG_DATABASE", "").strip()
        if not database:
            raise RuntimeError(
                "DEEPEYE_TEST_PG_DATABASE must name an existing database when "
                "DEEPEYE_TEST_PG=1"
            )

        code_root = Path(__file__).resolve().parents[1]
        for path in (code_root, code_root / "baselines" / "DeepEye-SQL"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from scripts.baseline_adapters.deepeye.postgres_execution import (
            execute_postgres_sql,
        )

        cls.execute = staticmethod(execute_postgres_sql)
        cls.database = database

    def setUp(self):
        # Native execution must work independently of a populated Meta schema.
        self.item = SimpleNamespace(
            db_type="postgresql", database_id=self.database, database_schema={}
        )

    def assert_rows(self, sql, expected, columns=None):
        result = self.execute(self.item, sql, timeout=2)
        self.assertEqual(result.result_type, "success", result.error_message)
        self.assertEqual(result.result_rows, expected)
        if columns is not None:
            self.assertEqual(result.result_cols, columns)
        self.assertEqual(result.sql, sql)
        return result

    def test_constant_query_does_not_require_meta_schema(self):
        """Reintroducing a Meta prerequisite must reject this otherwise valid query."""
        self.assert_rows("SELECT 42 AS answer", [(42,)], ["answer"])

    def test_arrays_preserve_null_elements_and_postgres_subscripts(self):
        """A cast whitelist or cross-dialect subscript rewrite changes these values."""
        self.assert_rows(
            "SELECT ARRAY[1, NULL, 3]::integer[] AS numbers, "
            "(ARRAY[7, 11])[2] AS second",
            [([1, None, 3], 11)],
            ["numbers", "second"],
        )

    def test_whole_row_json_preserves_all_cte_fields(self):
        """Whole-row rejection or column projection must not hide source fields."""
        self.assert_rows(
            "WITH t(id, label) AS (VALUES (3, 'three')) "
            "SELECT to_jsonb(t) AS object, row_to_json(t.*) AS row_object FROM t",
            [({"id": 3, "label": "three"}, {"id": 3, "label": "three"})],
        )

    def test_unnest_with_ordinality_returns_native_array_rows(self):
        """Rejecting table functions loses valid PostgreSQL row sources."""
        self.assert_rows(
            "SELECT value, position FROM unnest(ARRAY[4, 8, NULL]) "
            "WITH ORDINALITY AS u(value, position) ORDER BY position",
            [(4, 1), (8, 2), (None, 3)],
            ["value", "position"],
        )

    def test_lateral_json_function_can_reference_values_source(self):
        """Rejecting LATERAL or losing correlation changes expanded JSON rows."""
        self.assert_rows(
            "SELECT t.id, kv.key, kv.value "
            "FROM (VALUES (7, '{\"b\": 2, \"a\": 1}'::jsonb)) AS t(id, payload) "
            "CROSS JOIN LATERAL jsonb_each_text(t.payload) AS kv(key, value) "
            "ORDER BY kv.key",
            [(7, "a", "1"), (7, "b", "2")],
        )

    def test_recursive_cte_returns_every_recursive_step(self):
        """A recursive-CTE fence prevents a finite native query from executing."""
        self.assert_rows(
            "WITH RECURSIVE seq(n) AS ("
            "SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 5"
            ") SELECT n FROM seq ORDER BY n",
            [(1,), (2,), (3,), (4,), (5,)],
        )

    def test_json_arrows_inside_functions_preserve_json_and_text(self):
        """Treating PostgreSQL arrows as lambdas breaks nested function arguments."""
        self.assert_rows(
            "WITH t(payload) AS (VALUES "
            "('{\"solar\": {\"watts\": \"15\"}, \"wind\": {\"watts\": \"\"}}'::jsonb)) "
            "SELECT NULLIF(payload -> 'solar' ->> 'watts', ''), "
            "COALESCE(payload -> 'missing', payload -> 'solar'), "
            "NULLIF(payload -> 'wind' ->> 'watts', '') FROM t",
            [("15", {"watts": "15"}, None)],
        )

    def test_json_arrows_preserve_dynamic_keys_and_negative_indexes(self):
        """JSONPath rewriting can confuse literal keys with paths or array indexes."""
        self.assert_rows(
            "WITH t(payload, key, index_value) AS (VALUES "
            "('{\"0\": \"zero\", \"a.b[0]\": \"literal\", \"a''b\": \"quote\"}'::jsonb, "
            "'a.b[0]', 1)) "
            "SELECT payload ->> '0', payload ->> key, payload ->> 'a''b', "
            "'[10, 20, 30]'::jsonb -> -1, "
            "'[10, 20, 30]'::jsonb -> (index_value + 0) FROM t",
            [("zero", "literal", "quote", 30, 20)],
        )

    def test_null_safe_comparisons_keep_null_rows(self):
        """Replacing IS DISTINCT FROM with ordinary equality changes NULL results."""
        self.assert_rows(
            "SELECT a IS DISTINCT FROM b, a IS NOT DISTINCT FROM b "
            "FROM (VALUES (1, 1, 1), (2, 1, NULL), (3, NULL, NULL)) "
            "AS t(position, a, b) ORDER BY position",
            [(False, True), (True, False), (False, True)],
        )

    def test_parenthesized_order_name_uses_output_alias(self):
        """Qualifying a bare output name as an input column reverses this order."""
        self.assert_rows(
            "SELECT label AS id FROM (VALUES (1, 'b'), (2, 'a')) AS t(id, label) "
            "ORDER BY ((id))",
            [("a",), ("b",)],
        )

    def test_order_expression_uses_input_column_despite_output_alias(self):
        """Expanding output aliases inside expressions changes PostgreSQL binding."""
        self.assert_rows(
            "SELECT -id AS id FROM (VALUES (1), (2)) AS t(id) ORDER BY id + 0",
            [(-1,), (-2,)],
        )

    def test_distinct_on_uses_output_alias_for_duplicate_elimination(self):
        """Binding DISTINCT ON to the input would incorrectly retain a third row."""
        self.assert_rows(
            "SELECT DISTINCT ON ((id)) id % 2 AS id "
            "FROM (VALUES (1), (2), (3)) AS t(id) ORDER BY id",
            [(0,), (1,)],
        )

    def test_result_larger_than_five_thousand_rows_is_complete(self):
        """A result cap, outer LIMIT or bounded fetch must not discard later rows."""
        result = self.execute(
            self.item,
            "SELECT n FROM generate_series(1, 6001) AS seq(n) ORDER BY n",
            timeout=2,
        )
        self.assertEqual(result.result_type, "success", result.error_message)
        self.assertEqual(result.result_rows, [(n,) for n in range(1, 6002)])

    def test_query_runs_inside_read_only_transaction(self):
        """Dropping read-only startup or transaction settings removes this boundary."""
        self.assert_rows(
            "SELECT current_setting('transaction_read_only'), "
            "current_setting('default_transaction_read_only')",
            [("on", "on")],
        )

    def test_search_path_resolves_catalog_then_public(self):
        """Omitting public breaks normal unqualified PostgreSQL name resolution."""
        self.assert_rows(
            "SELECT current_schemas(false) AS schemas",
            [(["pg_catalog", "public"],)],
        )

    def test_catalog_query_is_not_restricted_by_meta(self):
        """A schema whitelist must not block server-authorized catalog reads."""
        self.assert_rows(
            "SELECT nspname::text FROM pg_catalog.pg_namespace "
            "WHERE nspname = 'pg_catalog'",
            [("pg_catalog",)],
        )

    def test_missing_column_returns_postgres_sqlstate_and_primary_message(self):
        """Local validation or generic sanitization would erase server diagnostics."""
        result = self.execute(
            self.item,
            "SELECT missing_deepeye_column FROM (VALUES (1)) AS t(present_column)",
            timeout=2,
        )
        self.assertEqual(result.result_type, "execution_error")
        self.assertIsNone(result.result_rows)
        self.assertIn("42703", result.error_message)
        self.assertIn("missing_deepeye_column", result.error_message)

    def test_prepared_execution_rejects_multiple_select_statements(self):
        """Simple-query execution could silently run or return only one statement."""
        result = self.execute(self.item, "SELECT 1; SELECT 2", timeout=2)
        self.assertEqual(result.result_type, "execution_error")
        self.assertIsNone(result.result_rows)
        self.assertIn("42601", result.error_message)
        self.assertIn("multiple commands", result.error_message)
        self.assertIn("prepared statement", result.error_message)

    def test_statement_timeout_returns_server_cancellation(self):
        """A missing timeout would let pg_sleep finish instead of returning 57014."""
        result = self.execute(self.item, "SELECT pg_sleep(2)", timeout=1)
        self.assertEqual(result.result_type, "timeout", result.error_message)
        self.assertIsNone(result.result_rows)
        self.assertIn("57014", result.error_message)
        self.assertIn("statement timeout", result.error_message)

    def test_single_statement_accepts_semicolons_in_literals_and_comments(self):
        """Naive splitting or wrapping the raw statement breaks valid SQL syntax."""
        self.assert_rows(
            "-- leading comment with ;\n"
            "SELECT 'a; b' AS value; -- trailing comment with ;",
            [("a; b",)],
            ["value"],
        )


if __name__ == "__main__":
    unittest.main()
