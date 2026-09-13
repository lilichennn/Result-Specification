from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/DeepEye-SQL"))

from app.prompt.factory import PromptFactory
from app.prompt.prompt_template import DC_SQL_GENERATION_PROMPT
from app.db_utils import schema as schema_module
from app.pipeline.sql_revision.checkers.time_checker import TimeChecker
from scripts.baseline_adapters.deepeye.hooks import install_postgres_support


class PostgresPromptTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(install_postgres_support())

    def test_pg_prompts_use_pg_dialect_without_replacing_input_text(self):
        # Missing dialect routing would request SQLite/STRFTIME from PG.
        calls = [
            lambda: PromptFactory.format_direct_linking_prompt("S", "Q", "E", "postgresql"),
            lambda: PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E", "postgresql"),
            lambda: PromptFactory.format_skeleton_sql_generation_prompt("S", "Q", "E", "postgresql"),
            lambda: PromptFactory.format_icl_sql_generation_prompt([{"question": "Example", "sql": "SELECT 1"}], "S", "Q", "E", "postgresql"),
            lambda: PromptFactory.format_execution_checker_prompt("S", "Q", "E", "SELECT 1", "R", "postgresql"),
            lambda: PromptFactory.format_common_checker_prompt("S", "Q", "E", "SELECT 1", "Advice", "postgresql"),
            lambda: PromptFactory.format_br_pair_selection_prompt("S", "Q", "E", "SELECT 1", "R", "SELECT 2", "R2", "postgresql"),
        ]
        for call in calls:
            prompt = call()
            self.assertIn("PostgreSQL", prompt)
            self.assertNotIn("SQLite", prompt)
            self.assertNotIn("STRFTIME", prompt.upper())
        text = "What does SQLite mean in this document?"
        self.assertIn(text, PromptFactory.format_dc_sql_generation_prompt("S", text, "E", "postgresql"))

    def test_existing_default_prompt_stays_unchanged(self):
        self.assertEqual(
            PromptFactory.format_dc_sql_generation_prompt("S", "Q", "E"),
            DC_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA="S", QUESTION="Q", HINT="E"),
        )

    def test_schema_profile_identifies_postgresql(self):
        profile = schema_module.get_database_schema_profile({"db_id": "fixture", "db_type": "postgresql", "tables": {}})
        self.assertIn("POSTGRESQL", profile)

    def test_sqlite_time_rewrite_does_not_mutate_postgresql(self):
        # PostgreSQL syntax errors belong to SyntaxChecker, not a SQLite regex.
        sql = "SELECT strftime('%Y', recorded_at) >= 2000 FROM events"
        checker = TimeChecker()
        pg_sql, _ = checker.check_and_revise(sql, SimpleNamespace(db_type="postgresql"), None)
        self.assertEqual(pg_sql, sql)
        sqlite_sql, _ = checker.check_and_revise(sql, SimpleNamespace(), None)
        self.assertIn(">= '2000'", sqlite_sql)


class HookLifecycleTest(unittest.TestCase):
    def test_old_undo_does_not_uninstall_or_unlock_a_new_installation(self):
        old_undo = install_postgres_support()
        old_undo()
        new_undo = install_postgres_support()
        try:
            old_undo()
            with self.assertRaises(RuntimeError):
                install_postgres_support()
        finally:
            new_undo()

    def test_restores_native_functions_and_class_descriptors(self):
        from app.services.execution_service import ExecutionService
        from app.db_utils import execution
        original_execute = execution.execute_sql_for_data_item
        original_key = vars(ExecutionService)["_build_result_key"]
        original_prompt = vars(PromptFactory)["format_dc_sql_generation_prompt"]
        undo = install_postgres_support()
        try:
            self.assertIsNot(execution.execute_sql_for_data_item, original_execute)
        finally:
            undo()
        self.assertIs(execution.execute_sql_for_data_item, original_execute)
        self.assertIs(vars(ExecutionService)["_build_result_key"], original_key)
        self.assertIs(vars(PromptFactory)["format_dc_sql_generation_prompt"], original_prompt)


if __name__ == "__main__":
    unittest.main()
