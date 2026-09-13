"""Source lineage tests: real stores, no model or PostgreSQL calls."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from . import OfflineTestCase

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE / 'baselines/DeepEye-SQL'))
from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, _checkpoint
from tests.test_deepeye_run_inheritance import make_item, manifest, complete_stage, cost

try:
    from scripts.rc_evaluation.deepeye.source import snapshot_source, restore_seed, validate_manifest
except ImportError:
    snapshot_source = restore_seed = validate_manifest = None


def make_source(path, *, calls=1, stage='sql_revision', partial=False, bad_hash=False, source_config=None, code_hashes=None, terminal='succeeded'):
    tasks = [('lite', make_item('a'))]
    data = manifest(tasks, upgrade=True)
    if source_config is not None:
        data['effective_config'] = source_config
    if code_hashes is not None:
        data['sources']['code'] = code_hashes
    with RunStore.create(path, data) as store:
        item = copy.deepcopy(tasks[0][1])
        identity = fingerprint({'manifest': data, 'input': to_jsonable(item.model_dump(exclude={'gold_sql'}))})
        for current in STAGES:
            input_hash = fingerprint({'input': identity, 'stage': current})
            complete_stage(item, current)
            count = calls if current == stage else 1
            setattr(item, current + '_llm_cost', cost(count))
            payload = _checkpoint(item, current)
            payload['attempt_wall_seconds'] = 1.0
            attempt = store.begin_attempt('lite/a', current, 'wrong' if bad_hash and current == stage else input_hash)
            for number in range(count):
                call = f'{attempt}-{number}'
                store.append_event(attempt, 'api_request', {'call_id': call})
                if not partial or current != stage:
                    store.append_event(attempt, 'api_response', {'call_id': call, 'response': {'usage': cost(1)}})
            if current != stage or terminal != 'interrupted':
                store.finish_attempt(attempt, terminal if current == stage else 'succeeded', payload)
            identity = fingerprint({'input': input_hash, 'output': payload})
    return tasks


class SourceTests(OfflineTestCase):
    def test_inherited_target_uses_original_call_trace_not_empty_import_trace(self):
        from scripts.baseline_adapters.deepeye.run_inheritance import inherit_checkpoints
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path, calls=1)
            inherited = Path(temporary) / 'inherited'
            with RunStore.open(path, read_only=True) as original:
                with RunStore.create(inherited, original.manifest) as destination:
                    inherit_checkpoints(original, destination, tasks)
                    self.assertEqual(destination.events(), [])
            snapshot = snapshot_source(inherited, tasks, 'sql_revision')
            self.assertEqual(snapshot['source_checkpoints']['lite/a']['stages']['sql_revision']['api_trace']['requests'], 1)

    def test_zero_request_failed_or_unfinished_target_cannot_be_replayed(self):
        for terminal in ('failed', 'interrupted'):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'source'
                tasks = make_source(path, calls=0, terminal=terminal)
                with self.assertRaisesRegex(ValueError, 'not successfully completed'):
                    snapshot_source(path, tasks, 'sql_revision')

    def test_modified_embedded_checkpoint_is_rejected_before_restore_execution(self):
        from .test_runner import experiment_manifest
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path)
            result = experiment_manifest(snapshot_source(path, tasks, 'sql_revision'))
            result['source_checkpoints']['lite/a']['stages']['sql_revision']['payload']['artifact']['sql_candidates_after_revision'] = ['SELECT 999']
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                validate_manifest(result)

    def test_manifest_condition_cannot_activate_missing_contracts(self):
        from .test_runner import experiment_manifest
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path)
            result = experiment_manifest(snapshot_source(path, tasks, 'sql_revision'), condition='rc')
            with self.assertRaisesRegex(ValueError, 'contract'):
                validate_manifest(result)

    def test_validated_prefix_restores_no_target_or_downstream(self):
        self.assertIsNotNone(snapshot_source, 'source snapshot implementation is required')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path)
            result = snapshot_source(path, tasks, 'sql_revision')
            item = restore_seed(result['source_checkpoints']['lite/a'], 'sql_revision')
            self.assertEqual(item.sql_candidates, ['SELECT x FROM t ORDER BY x'])
            self.assertIsNone(item.sql_candidates_after_revision)
            self.assertIsNone(item.final_selected_sql)
            self.assertEqual(item.gold_sql, '')
            self.assertIsNone(tasks[0][1].sql_candidates)
            self.assertEqual(result['source_checkpoints']['lite/a']['stages']['sql_revision']['api_trace']['requests'], 1)

    def test_incomplete_call_trace_is_not_zero_call_proof(self):
        self.assertIsNotNone(snapshot_source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path, partial=True)
            with self.assertRaisesRegex(ValueError, 'trace|unanswered'):
                snapshot_source(path, tasks, 'sql_revision')

    def test_wrong_target_hash_rejects_even_with_success_artifact(self):
        self.assertIsNotNone(snapshot_source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path, bad_hash=True)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                snapshot_source(path, tasks, 'sql_revision')

    def test_full_input_change_rejects_source(self):
        self.assertIsNotNone(snapshot_source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path)
            tasks[0][1].retrieved_values = {'unbound': {}}
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                snapshot_source(path, tasks, 'sql_revision')

    def test_successful_zero_calls_are_explicit_and_target_is_available(self):
        self.assertIsNotNone(snapshot_source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path, calls=0)
            result = snapshot_source(path, tasks, 'sql_revision', continue_downstream=True)
            entry = result['source_checkpoints']['lite/a']['stages']['sql_revision']
            self.assertEqual(entry['api_trace']['requests'], 0)
            self.assertTrue(entry['api_trace']['complete'])
            self.assertIn('sql_selection', result['source_checkpoints']['lite/a']['stages'])


if __name__ == '__main__':
    unittest.main()
