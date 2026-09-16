import importlib
import importlib.util
import unittest


SCHEMA = {"table_names_original": ["t"], "column_names_original": [[-1, "*"], [0, "id"]],
          "column_types": ["text", "number"], "column_types_original": ["text", "INTEGER"],
          "primary_keys": [1], "foreign_keys": [], "unresolved_foreign_keys": [{"reference": "secret.bad"}]}


class PromptTests(unittest.TestCase):
    def setUp(self):
        name = "scripts.baseline_adapters.dail_sql.prompts"
        self.assertIsNotNone(importlib.util.find_spec(name), "Task5 prompt renderer missing")
        self.api = importlib.import_module(name)

    def test_native_layout_nine_qa_order_evidence_and_no_target_history(self):
        task = {"question": "Count?", "evidence": " Use id.", "schema": SCHEMA,
                "database": {"dialect": "sqlite"}, "gold_sql": "SECRET", "previous_prediction": "HISTORY"}
        examples = [{"question": f"Q{i}", "evidence": f"E{i}", "sql": f"SELECT {i}"} for i in range(9)]
        expected = "/* Some SQL examples are provided based on similar problems: */\n" + "\n\n".join(
            [f"/* Answer the following: Q{i} E{i} */\nSELECT {i}" for i in range(9)] + [
            '/* Given the following database schema: */\nCREATE TABLE "t" (\n  "id" INTEGER,\n  PRIMARY KEY ("id")\n);\n\n/* Answer the following: Count?  Use id. */\nSELECT '])
        self.assertEqual(self.api.build_prompt(task, examples), [{"role": "user", "content": expected}])
        task.update(round_no=2, error="HISTORY2")
        self.assertEqual(self.api.build_prompt(task, examples)[0]["content"], expected)
        with self.assertRaises(ValueError):
            self.api.build_prompt(task, examples[:8])

    def test_pg_instruction_same_across_modes_training_sql_not_translated(self):
        examples = [{"question": "Q", "evidence": "", "sql": "SELECT strftime('%Y', d) FROM t"}] * 9
        task = {"question": "Q", "schema": SCHEMA, "database": {"dialect": "postgresql"}}
        outputs = [self.api.build_prompt({**task, "mode": mode, "round_no": r}, examples)
                   for mode in ("native", "rc_first", "rc_second", "rc_both") for r in (1, 2)]
        self.assertTrue(all(value == outputs[0] for value in outputs))
        content = outputs[0][0]["content"]
        self.assertIn("PostgreSQL", content)
        self.assertEqual(content.count("SELECT strftime('%Y', d) FROM t"), 9)
        self.assertNotIn("secret.bad", content)

    def test_extract_continuation_full_fence_cte_and_pg_literals(self):
        cases = [(" id FROM t ", "sqlite", "SELECT id FROM t"),
                 ("SELECT id FROM t", "sqlite", "SELECT id FROM t"),
                 ("```sql\nSELECT 1\n```", "sqlite", "SELECT 1"),
                 ("WITH x AS (SELECT 1) SELECT * FROM x", "sqlite", "WITH x AS (SELECT 1) SELECT * FROM x"),
                 ("SELECT '{\"a\": \"/* raw */\"}'::jsonb->>'a'", "postgresql", "SELECT '{\"a\": \"/* raw */\"}'::jsonb->>'a'"),
                 ("1::integer", "postgresql", "SELECT 1::integer"),
                 ("id FROM t /* Answer next */ SELECT 2", "sqlite", "SELECT id FROM t ")]
        for raw, dialect, expected in cases:
            with self.subTest(raw=raw):
                result = self.api.extract_sql(raw, dialect)
                self.assertEqual(result["raw_text"], raw)
                self.assertEqual(result["candidate_sql"], expected)
                self.assertIsNone(result["extraction_error"])
                value = raw
                for step in result["transformations"]:
                    self.assertEqual(step["before"], value)
                    value = step["after"]
                self.assertEqual(value, expected)

    def test_unextractable_stays_empty_and_does_not_repair(self):
        for raw in ("", "I cannot answer this question.", "```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```"):
            result = self.api.extract_sql(raw, "sqlite")
            self.assertIsNone(result["candidate_sql"])
            self.assertTrue(result["extraction_error"])

    def test_sqlite_native_whitespace_cleanup_before_duplication(self):
        result = self.api.extract_sql(" id,\n  name\tFROM t  /* repeated QA */", "sqlite")
        self.assertEqual(result["candidate_sql"], "SELECT id, name FROM t ")

    def test_pg_real_sql_comments_remain_in_query(self):
        raw = "WITH x AS (SELECT /* column */ 1) SELECT * FROM x"
        self.assertEqual(self.api.extract_sql(raw, "postgresql")["candidate_sql"], raw)

    def test_sqlite_spaced_operator_continuations_reach_native_vote_repair(self):
        from scripts.baseline_adapters.dail_sql.selection import prepare_vote_sql

        for spaced, repaired in (("> =", ">="), ("< =", "<="), ("! =", "!=")):
            with self.subTest(operator=spaced):
                raw = f"id FROM t WHERE id {spaced} 1"
                candidate = "SELECT " + raw
                result = self.api.extract_sql(raw, "sqlite")
                self.assertEqual(result["candidate_sql"], candidate)
                self.assertEqual(result["raw_text"], raw)
                self.assertIsNone(result["extraction_error"])
                self.assertEqual(result["transformations"], [{"name": "native.select_continuation_prefix",
                    "before": raw, "after": candidate}])
                vote = prepare_vote_sql(result["candidate_sql"], "sqlite")
                executed = f"SELECT id FROM t WHERE id {repaired} 1"
                self.assertEqual(vote["vote_sql"], executed)
                self.assertEqual(vote["transformations"], [{"name": "native.postprocess",
                    "before": candidate, "after": executed}])

    def test_pg_leading_comments_recognize_complete_select_and_with_unchanged(self):
        queries = ("SELECT 1", "WITH x AS (SELECT 1) SELECT * FROM x")
        prefixes = ("/* explanation */ ", "-- explanation\n", "/* first */\n-- second\n")
        for prefix in prefixes:
            for query in queries:
                with self.subTest(prefix=prefix, query=query):
                    raw = prefix + query
                    result = self.api.extract_sql(raw, "postgresql")
                    self.assertEqual(result["candidate_sql"], raw)
                    self.assertEqual(result["raw_text"], raw)
                    self.assertEqual(result["transformations"], [])
                    self.assertIsNone(result["extraction_error"])
