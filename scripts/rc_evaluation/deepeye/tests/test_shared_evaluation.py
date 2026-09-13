"""Read-only backend and reference boundary; no model or remote service calls."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'baselines/DeepEye-SQL'))
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.rc_evaluation.deepeye.evaluation import evaluate_run


class SharedEvaluationTests(unittest.TestCase):
    def fixture(self, root, benchmark='bird', split='dev', db_type='sqlite', *, reference=True):
        special = benchmark in ('bird_interact', 'spider2')
        identity = 'bq001' if db_type == 'bigquery' else ('a' if special else 7)
        partition = f'{benchmark}/{split}'
        key = f'{partition}/' + (f's:{identity}' if special else f'i:{identity}')
        path = root / 'reference.json'
        row = {'index': identity, 'db_id': 'db'}
        if reference:
            row[{'bird': 'SQL', 'spider': 'query', 'spider2': 'sql', 'bird_interact': 'sol_sql'}[benchmark]] = 'SELECT 1 AS x'
        path.write_text(json.dumps([row]))
        database = root / 'db.sqlite'
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE guard(x)')
        binding = {'task_key': key, 'partition': partition, 'benchmark': benchmark, 'split': split,
                   'external_id': identity, 'database_id': 'db', 'database_path': str(database) if db_type == 'sqlite' else 'db',
                   'db_type': db_type, 'reference': {'path': str(path), 'external_id': identity, 'source_row': 0}}
        manifest = {'format': 'deepeye-run-v2', 'items': [binding], 'effective_config': {
            'dataset': {'sql_execution_timeout_seconds': 1},
            'native': {'dataset_config': {'bigquery_credential_path': '/fixture/bq.json'}}}}
        run = root / 'run'
        with RunStore.create(run, manifest) as store:
            aid = store.begin_attempt(key, 'sql_selection', 'fixture')
            store.finish_attempt(aid, 'succeeded', {'artifact': {'final_selected_sql': 'SELECT 1 AS x'}})
        return run, key, database

    def test_six_reference_formats_use_frozen_locator_without_postgres_or_model(self):
        for benchmark, split in [('bird', 'dev'), ('spider', 'dev'), ('spider', 'test'),
                                 ('spider2', 'lite'), ('bird_interact', 'lite'), ('bird_interact', 'full')]:
            with self.subTest(benchmark=benchmark, split=split), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run, key, _ = self.fixture(root, benchmark, split)
                calls = []
                def execute(*args):
                    calls.append(args)
                    return {'result_type': 'success', 'result_cols': ['x'], 'result_rows': [[1]]}
                result = evaluate_run(run, root/'eval', {}, env_file=root/'no-env',
                    database_version='fixture', execute_fn=execute)
                self.assertTrue(result['items'][key]['bag_equal'])
                self.assertEqual(result['items'][key]['evaluation_status'], 'evaluated')
                self.assertEqual(len(calls), 1)  # identical reference/candidate shares only this item's execution

    def test_missing_gold_is_not_evaluated_not_incorrect_and_does_not_open_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, key, _ = self.fixture(root, 'spider2', 'lite', 'bigquery', reference=False)
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                result = evaluate_run(run, root/'eval', {}, env_file=root/'no-env', database_version='fixture')
            self.assertIsNone(result['items'][key]['bag_equal'])
            self.assertEqual(result['items'][key]['evaluation_status'], 'not_evaluated')
            self.assertEqual(result['summary']['incorrect'], 0)

    def test_sqlite_uses_native_read_only_execution_without_pg_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, key, database = self.fixture(root)
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                result = evaluate_run(run, root/'eval', {}, env_file=root/'no-env', database_version='fixture')
            self.assertTrue(result['items'][key]['bag_equal'])
            # A later candidate mutation must not be sent to any backend.
            with RunStore.open(run) as store:
                aid = store.begin_attempt(key, 'sql_selection', 'second')
                store.finish_attempt(aid, 'succeeded', {'artifact': {'final_selected_sql': 'DROP TABLE guard'}})
            result = evaluate_run(run, root/'eval2', {}, env_file=root/'no-env', database_version='fixture')
            self.assertIsNone(result['items'][key]['bag_equal'])
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM guard').fetchone(), (0,))

    def test_missing_pg_reference_does_not_require_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, key, _ = self.fixture(root, 'bird_interact', 'lite', 'postgresql', reference=False)
            result = evaluate_run(run, root/'eval', {}, env_file=root/'no-env', database_version='fixture')
            self.assertEqual(result['items'][key]['evaluation_status'], 'not_evaluated')

    def test_bigquery_routes_to_native_backend_and_blocks_sql_mutation(self):
        from app.db_utils.execution import SQLExecutionResult
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, key, _ = self.fixture(root, 'spider2', 'lite', 'bigquery')
            with patch('app.db_utils.cloud_execution.execute_bigquery_sql', return_value=SQLExecutionResult(
                result_type='success', db_path='db', sql='SELECT 1 AS x', result_cols=['x'], result_rows=[(1,)])) as execute:
                result = evaluate_run(run, root/'eval', {}, env_file=root/'no-env', database_version='fixture')
            self.assertTrue(result['items'][key]['bag_equal'])
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[2], '/fixture/bq.json')
            with RunStore.open(run) as store:
                aid = store.begin_attempt(key, 'sql_selection', 'second')
                store.finish_attempt(aid, 'succeeded', {'artifact': {'final_selected_sql': 'DELETE FROM guard WHERE TRUE'}})
            with patch('app.db_utils.cloud_execution.execute_bigquery_sql', return_value=SQLExecutionResult(
                result_type='success', db_path='db', sql='SELECT 1 AS x', result_cols=['x'], result_rows=[(1,)])) as execute:
                result = evaluate_run(run, root/'eval2', {}, env_file=root/'no-env', database_version='fixture')
            self.assertIsNone(result['items'][key]['bag_equal'])
            self.assertEqual(execute.call_count, 1)  # reference only
