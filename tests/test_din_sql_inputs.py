import tempfile
import unittest
from pathlib import Path
from scripts.baseline_adapters.din_sql.inputs import (
    TaskKey, DinSettings, classify_gold, physical_tables, public_pg_context,
    validate_settings, load_templates, sub_questions_bound,
)


class InputTests(unittest.TestCase):
    def test_legacy_parent_json_escaping_is_not_a_mismatch(self):
        self.assertTrue(sub_questions_bound(['What is "Mirrodin"?'], 'sub-questions = ["What is \\"Mirrodin\\"?"]'))
        self.assertFalse(sub_questions_bound(['different question'], 'sub-questions = ["What is \\"Mirrodin\\"?"]'))

    def test_ids_are_namespaced_strings(self):
        self.assertEqual(TaskKey('a', 1), TaskKey('a', '1'))
        self.assertNotEqual(TaskKey('a', 1), TaskKey('b', 1))

    def test_gold_classification_excludes_quoted_select(self):
        self.assertEqual(classify_gold("SELECT 'SELECT' FROM scores", ['scores']), 'EASY')
        self.assertEqual(classify_gold('SELECT * FROM a JOIN b USING(id)', ['a', 'b']), 'NON-NESTED')
        self.assertEqual(classify_gold('SELECT * FROM a UNION SELECT * FROM a', ['a']), 'NESTED')
        self.assertEqual(physical_tables('WITH c AS (SELECT * FROM real_table) SELECT * FROM c', 'postgres'), ['real_table'])

    def test_rejects_unsupported_request_settings(self):
        with self.assertRaises(ValueError):
            validate_settings(DinSettings(n=5))

    def test_pg_context_projects_only_public_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'t.csv').write_text('original_column_name,data_type,column_description\nDoc,jsonb,contents\n')
            calls = []
            def execute(database, sql, **kwargs):
                calls.append(sql)
                return {'status':'success', 'rows':[({'x':1},)], 'columns':['Doc']}
            context = public_pg_context({'database_id':'test','dialect':'postgresql'}, p, execute=execute)
            self.assertIn('jsonb', context)
            self.assertEqual(calls, ['SELECT "Doc" FROM "t" LIMIT 3'])
            self.assertNotIn('hidden', context)

    def test_templates_loaded_without_executing_official_script(self):
        root = Path(__file__).resolve().parents[1]
        templates = load_templates(root)
        self.assertIn('easy_prompt', templates['spider'])
        self.assertIn('RC_INSTRUCTION', templates['colleague'])


if __name__ == '__main__':
    unittest.main()
