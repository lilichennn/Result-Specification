"""Portable profiles and supplied RC3 inventory, without API or database calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/DeepEye-SQL'))
from scripts.baseline_adapters.deepeye.workloads import load_workload, question_rows, _database_path, PATH_FIELDS
from scripts.rc_evaluation.deepeye.contracts import load_contracts

COUNTS = {'bird_dev': 1534, 'spider_dev': 1034, 'spider_test': 2147,
          'bird_interact_lite': 195, 'bird_interact_full': 410, 'spider2_lite': 121}


class RC3ProfileTests(unittest.TestCase):
    def profile(self, name):
        path = ROOT / 'config/deepeye' / (name + '_rc3.json')
        self.assertTrue(path.is_file(), f'Missing portable RC3 profile: {path.name}')
        return load_workload(path)

    def test_profiles_resolve_relative_paths_and_explicit_policies(self):
        for name in COUNTS:
            with self.subTest(name=name):
                workload = self.profile(name)
                raw = json.loads(Path(workload['_path']).read_text())
                self.assertEqual(workload['rc_version'], 3)
                self.assertNotIn('bigquery_credential_path', raw)
                for key in PATH_FIELDS:
                    if raw.get(key):
                        self.assertFalse(Path(raw[key]).is_absolute(), key)
                for path in raw.get('database_paths', {}).values():
                    self.assertFalse(Path(path).is_absolute())
                self.assertEqual(workload['prepared_dataset'], str(ROOT / 'baselines_reproduce' /
                    'deepeye_shared_preparation/prepared' / workload['benchmark'] / (workload['split'] + '.snapshot')))
                self.assertEqual(workload['preparation_root'], str(ROOT / 'baselines_reproduce/deepeye_shared_preparation'))
                self.assertEqual(workload['few_shot_strategy'],
                                 'none' if name == 'spider2_lite' else 'native_dynamic')

    def test_native_configuration_preserves_dynamic_budget_and_shares_training_indexes(self):
        from scripts import deepeye_run as entry
        environment = {'DASH_MODELS': 'qwen3.8', 'DASH_BASE_URL': 'https://offline.invalid/v1',
            'DASH_API_KEY': 'offline', 'EMBEDDING_MODEL': 'qwen3.7-text-embedding-flash',
            'EMBEDDING_BASE_URL': 'https://embedding.invalid/v1', 'EMBEDDING_API_KEY': 'offline'}
        indexes = {}
        with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
            for name in COUNTS:
                workload = self.profile(name)
                args = entry._build_parser().parse_args(['prepare', '--workload', workload['_path'],
                                                        '--run-dir', '/unused-profile-test'])
                config = entry.build_runtime_config(environment, args, Path('/unused-profile-test'))
                self.assertEqual(config.dataset_config.type, workload['benchmark'])
                self.assertEqual(config.dataset_config.split, workload['split'])
                if name == 'spider2_lite':
                    self.assertNotIn('few_shot_source', workload)
                    continue
                few = config.few_shot_index_config
                self.assertEqual((few.num_examples, few.question_weight, few.sql_weight), (7, .6, .4))
                self.assertEqual((few.preliminary_sql.enabled, few.preliminary_sql.dc_sampling_budget,
                                  few.preliminary_sql.skeleton_sampling_budget), (True, 4, 4))
                self.assertEqual(few.llm.model, 'qwen3.8')
                self.assertEqual(few.embedding.embedding_model_name_or_path, 'qwen3.7-text-embedding-flash')
                indexes[name] = few.save_path
        self.assertEqual(indexes['bird_dev'], indexes['bird_interact_lite'])
        self.assertEqual(indexes['bird_dev'], indexes['bird_interact_full'])
        self.assertEqual(indexes['spider_dev'], indexes['spider_test'])
        self.assertNotEqual(indexes['bird_dev'], indexes['spider_dev'])

    def test_supplied_inventory_has_exact_counts_and_strict_rc3_matches(self):
        for name, count in COUNTS.items():
            with self.subTest(name=name):
                workload = self.profile(name)
                if not Path(workload['questions']).is_file() or not Path(workload['rc']).is_file():
                    self.skipTest('Supplied benchmark resources are not installed in this checkout')
                rows = question_rows(workload)
                self.assertEqual(len(rows), count)
                tasks = [(workload['partition'], SimpleNamespace(instance_id=row['external_id'],
                    database_id=row['database_id'], question=row['question'], evidence=row['evidence'])) for row in rows]
                with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                    contracts = load_contracts({workload['partition']: Path(workload['rc'])}, tasks, rc_version=3)
                self.assertEqual(len(contracts), count)
                self.assertTrue(all(value['rc_version'] == 3 and value['gold_corrected'] for value in contracts.values()))

    def test_spider2_selection_preserves_source_and_resolves_only_eligible_local_databases(self):
        workload = self.profile('spider2_lite')
        if not Path(workload['questions']).is_file():
            self.skipTest('Supplied Spider2 resources are not installed')
        original = json.loads(Path(workload['questions']).read_text())
        self.assertEqual(len(original), 280)
        rows = question_rows(workload)
        self.assertEqual(len(workload['question_ids']), 121)
        local = [row for row in rows if row['external_id'].startswith('local')]
        self.assertEqual(len(local), 24)
        self.assertEqual(len(rows) - len(local), 97)
        self.assertEqual(set(workload['database_paths']), {row['database_id'] for row in local})
        if not Path(workload['resource_root']).is_dir():
            self.skipTest('Local Spider2 database resources are not installed')
        for row in local:
            self.assertTrue(Path(_database_path(workload, row['database_id'], 'sqlite')).is_file())


if __name__ == '__main__':
    unittest.main()
