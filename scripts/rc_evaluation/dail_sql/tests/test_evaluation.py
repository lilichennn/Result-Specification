import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.baseline_adapters.dail_sql.execution import execute_sql


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.rc_evaluation.dail_sql.evaluation'),
                             'Task7 evaluation missing')
        from scripts.rc_evaluation.dail_sql.evaluation import evaluate_candidate
        self.evaluate = evaluate_candidate
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / 'test.sqlite'
        with sqlite3.connect(path) as db:
            db.executescript('CREATE TABLE t(a,b); INSERT INTO t VALUES(1,2),(1,2),(3,4);')
        self.binding = {'database': {'dialect': 'sqlite', 'database_id': 'test', 'path': str(path)},
                        'reference_sql': 'SELECT a,b FROM t'}
        self.calls = []
        def execute(database, sql):
            self.calls.append(sql)
            return execute_sql(database, sql, timeout_seconds=1)
        self.execute = execute

    def check(self, sql, **kwargs):
        return self.evaluate({'candidate_id': 'c', 'candidate_sql': sql, **kwargs},
                             self.binding, execute=self.execute, cache={})

    def test_rows_and_aliases_ignored_duplicates_and_positions_preserved(self):
        self.assertTrue(self.check('SELECT a AS x,b AS y FROM t ORDER BY a DESC')['bag_equal'])
        self.assertFalse(self.check('SELECT DISTINCT a,b FROM t')['bag_equal'])
        self.assertFalse(self.check('SELECT b,a FROM t')['bag_equal'])
        self.binding['reference_sql'] = 'SELECT a,b FROM t WHERE 0'
        self.assertFalse(self.check('SELECT a FROM t WHERE 0')['bag_equal'])

    def test_error_categories_and_unknown_types_remain_unknown(self):
        self.assertEqual(self.check('SELECT missing FROM t')['status'], 'prediction_error')
        self.binding['reference_sql'] = 'SELECT missing FROM t'
        result = self.check('SELECT a,b FROM t')
        self.assertEqual(result['status'], 'reference_error')
        self.assertIsNone(result['bag_equal'])
        result = self.evaluate({'candidate_id': 'c', 'candidate_sql': 'SELECT 1'}, self.binding,
            execute=lambda db, sql: {'status': 'success', 'columns': ['x'], 'rows': [(object(),)]}, cache={})
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['bag_equal'])

    def test_vote_requires_exact_sql_identity_and_explicit_database_version(self):
        vote = {'database': self.binding['database'], 'database_version': 'v1',
                'sql': 'SELECT a,b FROM t', 'status': 'success', 'columns': ['a','b'], 'rows': []}
        self.assertTrue(self.check('SELECT a,b FROM t', vote_execution=vote)['bag_equal'])
        self.assertIn('SELECT a,b FROM t', self.calls)
        self.binding['database_version'] = 'v1'
        self.calls.clear()
        self.assertFalse(self.check('SELECT a,b FROM t', vote_execution=vote)['bag_equal'])
        self.assertEqual(len(self.calls), 1)  # Reference is still independently executed.
        self.assertFalse(self.check('SELECT DISTINCT a,b FROM t', vote_execution=vote)['bag_equal'])

    def test_cache_is_bound_to_database_and_sql_and_never_reuses_unknown_version_vote(self):
        cache = {}
        candidate = {'candidate_id': 'c', 'candidate_sql': 'SELECT a,b FROM t'}
        for _ in range(2):
            self.evaluate(candidate, self.binding, execute=self.execute, cache=cache)
        self.assertEqual(len(self.calls), 1)
        self.binding['database_version'] = 'new'
        self.evaluate(candidate, self.binding, execute=self.execute, cache=cache)
        self.assertEqual(len(self.calls), 2)


if __name__ == '__main__':
    unittest.main()
