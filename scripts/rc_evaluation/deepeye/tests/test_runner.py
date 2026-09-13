"""The independent stage controller runs against real temporary RunStores."""
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from . import OfflineTestCase

from .test_source import make_source, complete_stage
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_pipeline import STAGE_METHODS
from scripts.rc_evaluation.deepeye.source import snapshot_source

try:
    from scripts.rc_evaluation.deepeye.runner import run_experiment
except ImportError:
    run_experiment = None


class OfflineTrace:
    secrets = ()
    @contextmanager
    def context(self, attempt):
        yield
    def raise_if_failed(self):
        pass
    def instrument_runner(self, runner, stage):
        return lambda: None


def experiment_manifest(source, *, condition='none', downstream=False):
    return {'format': 'deepeye-rc-evaluation-run-v1', 'target_stage': 'sql_revision',
            'condition': condition, 'repeat_id': '1', 'continue_downstream': downstream,
            'effective_config': source['source_manifest']['effective_config'],
            'sources': source['source_manifest']['sources'], 'contracts': {}, **source}


class Factory:
    def __init__(self, failure=None):
        self.seen, self.closed, self.failure = [], [], failure
    def __call__(self, stage, items):
        owner = self
        class Runner:
            def _clean_up(self):
                owner.closed.append(stage)
        runner = Runner()
        def execute(item):
            owner.seen.append((stage, copy.deepcopy(item)))
            if stage == owner.failure:
                item.sql_candidates_after_revision = ['MUTATED FAILED STATE']
                raise RuntimeError('synthetic stage failure')
            complete_stage(item, stage)
            if stage == 'sql_revision':
                item.sql_candidates_after_revision = ['SELECT 2']
        setattr(runner, STAGE_METHODS[stage], execute)
        return runner


class RunnerTests(OfflineTestCase):
    def test_called_rc_target_without_injection_is_failed_with_request_evidence(self):
        from tests.test_deepeye_sampling import llm_fixture, response
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path, calls=1)
            data = experiment_manifest(snapshot_source(path, tasks, 'sql_revision'), condition='rc')
            data['contracts'] = {'lite/a': {'task_key': 'lite/a', 'db_id': 'db',
                'question': 'Return x', 'evidence': 'Ascending order',
                'round2': {key: 'Meaning' for key in ('population', 'row_grain', 'column_role',
                                                    'derivation', 'filter_policy', 'meta_review')}}}
            llm, calls = llm_fixture([response()])
            def factory(stage, items):
                def revise(item):
                    llm.ask([{'role': 'user', 'content': 'deliberately bypassed formatter'}])
                    complete_stage(item, stage)
                return SimpleNamespace(_llm=llm, _checkers=[], _revise_sql=revise, _clean_up=lambda: None)
            with RunStore.create(Path(temporary) / 'run', data) as store:
                result = run_experiment(store, factory, TraceRecorder(store))
                self.assertEqual((result['failed'], len(calls)), (1, 1))
                payload = store.attempts()[0]['payload']
                self.assertEqual(payload['error_type'], 'MissingRCParticipation')
                self.assertEqual(payload['rc_participation']['native_request_count'], 1)
                self.assertEqual(payload['rc_participation']['actual_request_count'], 0)

    def test_partial_sampling_can_succeed_via_native_fallback(self):
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from app.llm_extractor import LLMExtractor
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from types import SimpleNamespace
        llm, calls = llm_fixture([response()] * 3 + [response('bad')] * 4 + [response()])
        def factory(stage, items):
            def revise(item):
                LLMExtractor().extract_with_retry(llm, [], parse, n=5)
                complete_stage(item, stage)
            return SimpleNamespace(_llm=llm, _checkers=[], _revise_sql=revise, _clean_up=lambda: None)
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            with patch('app.llm_extractor.extractor.logger.warning') as warning:
                result = run_experiment(store, factory, TraceRecorder(store), workers=1)
            warning.assert_called_once()
            self.assertEqual((result['succeeded'], result['failed'], len(calls)), (1, 0, 8))
            row = store.attempts()[0]
            self.assertEqual(row['status'], 'succeeded')
            self.assertFalse(row['payload']['sampling']['complete'])
            self.assertNotIn('error_type', row['payload'])
            before = store.attempts()
            run_experiment(store, lambda *args: self.fail('Completed RC stage must not be retried'),
                           TraceRecorder(store))
            self.assertEqual(store.attempts(), before)
            self.assertEqual(len(calls), 8)

    def test_successful_downstream_after_missing_target_rejected_before_constructor(self):
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, downstream=True) as store:
            attempt = store.begin_attempt('lite/a', 'sql_selection', 'unrelated-source')
            store.finish_attempt(attempt, 'succeeded', {'artifact': {'final_selected_sql': 'SELECT 999'}, 'metrics': {}})
            with self.assertRaisesRegex(ValueError, 'prefix|downstream|lineage'):
                run_experiment(store, lambda *args: self.fail('stale downstream reached constructor'), OfflineTrace(), workers=1)
            self.assertEqual(len(store.attempts()), 1)

    def test_failed_integrity_check_stops_before_constructor_or_attempt(self):
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            with patch.object(store, 'verify', return_value={'ok': False}):
                with self.assertRaisesRegex(ValueError, 'verification'):
                    run_experiment(store, lambda *args: self.fail('corrupt store constructed runner'), OfflineTrace(), workers=1)
            self.assertEqual(store.attempts(), [])

    def prepared(self, temporary, *, calls=1, downstream=False):
        source_path = Path(temporary) / 'source'
        tasks = make_source(source_path, calls=calls)
        snapshot = snapshot_source(source_path, tasks, 'sql_revision', continue_downstream=downstream)
        return RunStore.create(Path(temporary) / 'run', experiment_manifest(snapshot, downstream=downstream))

    def test_zero_call_target_reused_without_factory(self):
        self.assertIsNotNone(run_experiment, 'independent controller is required')
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, calls=0) as store:
            def forbidden(*args):
                raise AssertionError('zero-call reuse constructed a native/model runner')
            result = run_experiment(store, forbidden, OfflineTrace(), workers=1)
            self.assertEqual(result['succeeded'], 1)
            row = store.attempts()[0]
            self.assertEqual(row['stage'], 'sql_revision')
            self.assertEqual(row['payload']['execution_origin'], 'reused_no_native_llm_call')
            self.assertEqual(row['payload']['rc_participation']['reason'], 'no_native_llm_call')
            self.assertEqual(row['payload']['rc_participation'].get('status'), 'rc_not_participating')
            self.assertEqual(store.events(), [])

    def test_rc_condition_also_reuses_zero_call_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'
            tasks = make_source(path, calls=0)
            data = experiment_manifest(snapshot_source(path, tasks, 'sql_revision'), condition='rc')
            data['contracts'] = {'lite/a': {'task_key': 'lite/a', 'db_id': 'db',
                'question': 'Return x', 'evidence': 'Ascending order',
                'round2': {key: 'Meaning' for key in ('population', 'row_grain', 'column_role',
                                                    'derivation', 'filter_policy', 'meta_review')}}}
            with RunStore.create(Path(temporary) / 'run', data) as store:
                run_experiment(store, lambda *args: self.fail('RC forced a source no-call path'), OfflineTrace(), workers=1)
                self.assertEqual(store.attempts()[0]['payload']['rc_participation']['actual_request_count'], 0)

    def test_stage_failure_drains_native_children_before_cleanup(self):
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        order = []
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            def factory(stage, items):
                class Runner:
                    def __init__(self):
                        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=1)
                    def _revise_sql(self, item):
                        def child():
                            entered.set()
                            release.wait(5)
                            order.append('child drained')
                        self._inner_thread_pool_executor.submit(child)
                        raise RuntimeError('parent failed')
                    def _clean_up(self):
                        order.append('cleanup')
                return Runner()
            with ThreadPoolExecutor(max_workers=1) as outer:
                future = outer.submit(run_experiment, store, factory, OfflineTrace(), workers=1)
                try:
                    self.assertTrue(entered.wait(3))
                    self.assertFalse(future.done())
                finally:
                    release.set()
                result = future.result(timeout=5)
            self.assertEqual(result['failed'], 1)
            self.assertEqual(order, ['child drained', 'cleanup'])

    def test_call_enabled_target_receives_only_original_prefix(self):
        self.assertIsNotNone(run_experiment)
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            factory = Factory()
            result = run_experiment(store, factory, OfflineTrace(), workers=1)
            self.assertEqual(result['succeeded'], 1)
            self.assertEqual([stage for stage, _ in factory.seen], ['sql_revision'])
            seed = factory.seen[0][1]
            self.assertEqual(seed.sql_candidates, ['SELECT x FROM t ORDER BY x'])
            self.assertIsNone(seed.sql_candidates_after_revision)
            self.assertIsNone(seed.final_selected_sql)
            self.assertEqual(seed.gold_sql, '')
            self.assertEqual(factory.closed, ['sql_revision'])

    def test_failure_is_durable_retry_does_not_restore_failed_mutations(self):
        self.assertIsNotNone(run_experiment)
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            first = run_experiment(store, Factory('sql_revision'), OfflineTrace(), workers=1)
            self.assertEqual(first['failed'], 1)
            factory = Factory()
            run_experiment(store, factory, OfflineTrace(), workers=1)
            self.assertEqual([row['status'] for row in store.attempts()], ['failed', 'succeeded'])
            self.assertIsNone(factory.seen[0][1].sql_candidates_after_revision)
            count = len(store.attempts())
            run_experiment(store, lambda *args: self.fail('resume constructed runner'), OfflineTrace(), workers=1)
            self.assertEqual(len(store.attempts()), count)

    def test_downstream_is_opt_in_and_gets_changed_target_output(self):
        self.assertIsNotNone(run_experiment)
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, downstream=True) as store:
            factory = Factory()
            run_experiment(store, factory, OfflineTrace(), workers=1)
            self.assertEqual([stage for stage, _ in factory.seen], ['sql_revision', 'sql_selection'])
            self.assertEqual(factory.seen[1][1].sql_candidates_after_revision, ['SELECT 2'])
            self.assertIsNone(factory.seen[1][1].final_selected_sql)

    def test_replayed_target_can_replay_unchanged_downstream_without_clients(self):
        self.assertIsNotNone(run_experiment)
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, calls=0, downstream=True) as store:
            run_experiment(store, lambda *args: self.fail('unchanged replay constructed runner'), OfflineTrace(), workers=1)
            self.assertEqual([row['stage'] for row in store.attempts()], ['sql_revision', 'sql_selection'])
            self.assertEqual(store.attempts()[1]['payload']['execution_origin'], 'reused_unchanged_upstream')


if __name__ == '__main__':
    unittest.main()
