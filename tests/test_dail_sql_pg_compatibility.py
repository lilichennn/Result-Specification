"""Regressions for executable PostgreSQL rejected by the DAIL boundary."""
import ipaddress
import json
from pathlib import Path
import tempfile
import unittest

from scripts.baseline_adapters.dail_sql.native import sql_skeleton, SkeletonError
from scripts.baseline_adapters.dail_sql.prompts import extract_sql
from scripts.baseline_adapters.dail_sql.selection import prepare_vote_sql
from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable, restore_jsonable


SCHEMA = {"table_names_original": ["t"], "column_names_original": [[-1, "*"], [0, "id"]]}


class PostgreSQLCompatibilityTests(unittest.TestCase):
    def skeleton(self, sql):
        try:
            return sql_skeleton(sql, SCHEMA, "postgresql")
        except SkeletonError as exc:
            self.fail(f"Valid query rejected: {exc.__cause__}")

    def test_trailing_comments_are_not_extra_statements(self):
        for sql in ("SELECT 1; -- tail", "SELECT 1; /* tail */", ";SELECT 1;; -- tail"):
            with self.subTest(sql=sql):
                self.assertEqual(self.skeleton(sql), "select _")
        for sql in ("SELECT 1; SELECT 2; -- tail", "-- only a comment", "SELECT 1; DELETE FROM t"):
            with self.subTest(sql=sql), self.assertRaises(SkeletonError):
                sql_skeleton(sql, SCHEMA, "postgresql")

    def test_full_parenthesized_queries_are_not_select_continuations(self):
        for sql in ("(SELECT 1 AS x, 2 AS y)", "(SELECT x FROM (VALUES (1),(2)) t(x))",
                    "/* intro */ (SELECT 1) UNION ALL (SELECT 2)", "VALUES (1),(2)"):
            with self.subTest(sql=sql):
                result = extract_sql(sql, "postgresql")
                self.assertEqual(result["candidate_sql"], sql)
                self.assertEqual(result["transformations"], [])
                self.assertIsNone(result["extraction_error"])
                self.assertTrue(self.skeleton(sql))

    def test_expression_continuations_still_get_select(self):
        for sql in ("(1+2)", "id FROM t", "1::integer; -- tail"):
            with self.subTest(sql=sql):
                result = extract_sql(sql, "postgresql")
                self.assertEqual(result["candidate_sql"], "SELECT " + sql)
        self.assertTrue(extract_sql("(SELECT 1); DELETE FROM t", "postgresql")["extraction_error"])

    def test_json_and_escaped_strings_cannot_introduce_structure_tokens(self):
        for sql in ('''SELECT '{"a":")"}'::jsonb''', "SELECT '{}'::jsonb ->> 'a\")'", "SELECT $$x') AS (z$$",
                    r"SELECT E'a\'b)'", '''SELECT 'He said "x)"' '''):
            with self.subTest(sql=sql):
                skeleton = self.skeleton(sql)
                self.assertEqual(skeleton.count("("), skeleton.count(")"))
                self.assertNotIn("said", skeleton)
                self.assertEqual(extract_sql(sql, "postgresql")["candidate_sql"], sql.strip())

    def test_quoted_aliases_are_atomic_in_skeleton(self):
        for name in ("rank)", "score (max", "two words", "order"):
            sql = f'WITH c AS (SELECT 1 AS "{name}") SELECT "{name}" FROM c'
            with self.subTest(name=name):
                skeleton = self.skeleton(sql)
                self.assertEqual(skeleton.count("("), skeleton.count(")"))
                self.assertNotIn(name, skeleton)
                self.assertEqual(extract_sql(sql, "postgresql")["candidate_sql"], sql)

    def test_fence_characters_inside_sql_are_not_markdown(self):
        for raw in ("SELECT '```' AS x", "```sql\nSELECT '```' AS x\n```"):
            self.assertEqual(extract_sql(raw, "postgresql")["candidate_sql"], "SELECT '```' AS x")
        self.assertTrue(extract_sql("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```", "postgresql")["extraction_error"])

    def test_distinct_predicates_survive_ordinary_distinct_removal(self):
        cases = [
            ("SELECT 1 IS DISTINCT FROM 2", "SELECT 1 IS DISTINCT FROM 2"),
            ("SELECT 1 IS NOT DISTINCT FROM NULL", "SELECT 1 IS NOT DISTINCT FROM NULL"),
            ("SELECT DISTINCT id FROM t WHERE id IS DISTINCT FROM NULL",
             "SELECT  id FROM t WHERE id IS DISTINCT FROM NULL"),
            ("SELECT COUNT(DISTINCT id) FROM t WHERE id IS /* note */ NOT DISTINCT FROM 1",
             "SELECT COUNT( id) FROM t WHERE id IS /* note */ NOT DISTINCT FROM 1"),
            ("SELECT DISTINCT 'IS DISTINCT FROM' FROM t", "SELECT  'IS DISTINCT FROM' FROM t"),
        ]
        for sql, expected in cases:
            with self.subTest(sql=sql):
                result = prepare_vote_sql(sql, "postgresql")
                self.assertEqual(result["vote_sql"], expected)
                self.assertIsNone(result["error"])
        distinct_on = "SELECT DISTINCT ON (id) id FROM t ORDER BY id"
        self.assertEqual(prepare_vote_sql(distinct_on, "postgresql")["vote_sql"], distinct_on)

    def test_ip_values_roundtrip_through_durable_records(self):
        values = [ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("2001:db8::1"),
                  ipaddress.ip_interface("192.0.2.1/24"), ipaddress.ip_interface("2001:db8::1/64"),
                  ipaddress.ip_network("192.0.2.0/24"), ipaddress.ip_network("2001:db8::/64")]
        payload = {"rows": [(value,) for value in values],
                   "literal_tag": {"__run_store_type__": "ipaddress", "value": "user string"}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            with RunStore.create(root, {}) as store:
                attempt = store.begin_attempt("q", "vote", "f")
                event = store.append_event(attempt, "vote_execution", payload)
            with RunStore.open(root, read_only=True) as store:
                restored = store.event(attempt, event)["payload"]
        self.assertEqual(restored, payload)
        self.assertEqual([type(row[0]) for row in restored["rows"]], [type(v) for v in values])
        self.assertEqual(restore_jsonable(json.loads(json.dumps(to_jsonable(payload)))), payload)
        with self.assertRaises(TypeError):
            to_jsonable(object())


if __name__ == "__main__":
    unittest.main()
