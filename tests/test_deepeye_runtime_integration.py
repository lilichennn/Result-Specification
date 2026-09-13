"""Offline acceptance of the run-owned runtime in the native entrypoint."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tests.test_deepeye_run_pipeline import Factory, FakeTrace, item
from scripts import deepeye_bird_interact_run as entry
from scripts.baseline_adapters.deepeye.run_pipeline import run_pipeline, STAGES, STAGE_METHODS
from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder


class RuntimeIntegrationTests(unittest.TestCase):
    def test_stop_between_workflow_admission_and_coordinator_submission_is_a_pause(self):
        runtime = SamplingRuntime(coordinator_workers=1, request_workers=1)
        self.addCleanup(runtime.close)
        recorder = FakeTrace()
        recorder.stop_event = runtime.stop_event
        original = runtime.submit_coordinator
        def stop_at_submission(*args, **kwargs):
            runtime.stop()
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, RunStore.create(Path(temporary) / 'run', {}) as store, \
                patch.object(runtime, 'submit_coordinator', stop_at_submission):
            result = run_pipeline(store, [('lite', item(k)) for k in 'ab'], Factory(), recorder, runtime=runtime)
            self.assertEqual((result['succeeded'], result['failed'], result['paused']), (0, 0, 2))
            self.assertEqual(store.attempts(), [])

    def test_explicit_offline_renewal_targets_only_exhausted_samples_and_launches_nothing(self):
        from app.llm_extractor import LLMExtractor
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from scripts.rc_evaluation.deepeye import cli
        for command in (entry.main, cli.main):
            with self.subTest(command=command.__module__), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'run'
                with RunStore.create(path, {}) as store:
                    llm, calls = llm_fixture([response()] + [response('bad')] * 4)
                    recorder = TraceRecorder(store)
                    aid = store.begin_attempt('lite/a', 'sql_generation', 'input')
                    with recorder.context(aid):
                        LLMExtractor().extract_with_retry(llm, [], parse, n=2)
                    store.finish_attempt(aid, 'failed', {'error_type': 'IncompleteSamplingGroup'})
                    before = store.events()
                    group_id = next(e['payload']['group_id'] for e in before if e['kind'] == 'sampling_group_bound')
                output = io.StringIO()
                with redirect_stdout(output):
                    result = command(['renew-samples', '--run-dir', str(path), '--group-id', group_id,
                                      '--decision', 'Explicit offline fixture renewal'])
                self.assertEqual(result, 0)
                with RunStore.open(path, read_only=True) as store:
                    self.assertEqual(store.events()[:len(before)], before)
                    renewed = [e for e in store.events() if e['kind'] == 'sampling_retry_authorized']
                    self.assertEqual(len(renewed), 1)
                    self.assertEqual(renewed[0]['payload']['allowances'], [
                        {'sample_index': 1, 'prior_limit': 4, 'attempt_limit': 8}])
                    self.assertEqual(store.attempts()[-1]['status'], 'interrupted')
                self.assertEqual(len(calls), 5)

    def test_actual_native_entrypoint_overlaps_model_and_pg_under_independent_caps(self):
        import asyncio
        import time
        from app.llm import LLM
        from app.db_utils.execution import SQLExecutionResult
        from app.services.schema_service import SchemaService
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID
        from scripts.rc_evaluation.deepeye.tests.test_dynamic_concurrency import AsyncClientFixture, completion
        from scripts.rc_evaluation.deepeye.tests.test_cli import ENV
        from types import SimpleNamespace
        clients, pg_active, pg_peak, model_active, model_peak = [], 0, 0, 0, 0
        pg_seen = threading.Event()
        overlap = threading.Event()
        lock = threading.Lock()
        def pg(target, sql, timeout=None):
            nonlocal pg_active, pg_peak
            with lock:
                pg_active += 1
                pg_peak = max(pg_peak, pg_active)
                if model_active:
                    overlap.set()
            pg_seen.set()
            time.sleep(.01)
            with lock:
                pg_active -= 1
            return SQLExecutionResult(result_type='success', db_path='db', sql=sql,
                execution_time=.01, result_rows=[(1,)], result_cols=['x'])
        async def create(**kwargs):
            nonlocal model_active, model_peak
            row = store.attempt(_ATTEMPT_ID.get())
            with lock:
                model_active += 1
                model_peak = max(model_peak, model_active)
                if pg_active:
                    overlap.set()
            if row['item_key'] == 'lite/b':
                deadline = time.monotonic() + 4
                while not pg_seen.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(.005)
                self.assertTrue(pg_seen.is_set(), 'A slow question must not block another question reaching PG')
            await asyncio.sleep(.005)
            with lock:
                model_active -= 1
            self.assertEqual((kwargs['model'], kwargs['n'], kwargs['timeout']), ('offline', 1, 660))
            prompt = kwargs['messages'][0]['content']
            self.assertNotIn('Refined Contract', prompt)
            return completion('<result><table table_name="t"><column column_name="x" /></table></result>'
                if 'pinpoint the specific tables and columns' in prompt else '<result>SELECT x FROM t</result>')
        def client(**kwargs):
            fixture = AsyncClientFixture(create)
            clients.append(fixture)
            return fixture
        args = entry._build_parser().parse_args(['run', '--run-dir', 'unused', '--precompute-dir', 'unused',
            '--request-limit', '5', '--request-workers', '8', '--coordinator-workers', '2',
            '--http-connections', '8', '--request-start-rate', '10000', '--pg-concurrency', '1'])
        with tempfile.TemporaryDirectory() as temporary, RunStore.create(Path(temporary) / 'native', {}) as store, \
             patch('openai.AsyncOpenAI', side_effect=client), \
             patch.object(LLM, '_create_client', side_effect=AssertionError('unused sync client opened')), \
             patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)), \
             patch('scripts.baseline_adapters.deepeye.backend_hooks.execute_postgres_sql', side_effect=pg), \
             patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), \
             redirect_stdout(io.StringIO()):
            tasks = [('lite', item(key)) for key in 'ab']
            for _, target in tasks:
                target.few_shot_examples = [{'question': 'Other question', 'evidence': '', 'sql': 'SELECT 2'}]
            result = entry._execute_pipeline(store, tasks, ENV, args)
            self.assertEqual((result['succeeded'], result['failed']), (2, 0), store.attempts())
            self.assertTrue(overlap.is_set())
            self.assertEqual(pg_peak, 1)
            self.assertEqual(model_peak, 5)
            self.assertEqual(result['runtime']['requests']['peak_in_flight'], 5)
            self.assertEqual(result['runtime']['limits']['start_rate'], 10000)
            self.assertEqual(result['admission']['postgres']['current_limit'], 1)
            self.assertTrue(all(c.closed for c in clients))
            self.assertFalse(any(t.name.startswith(('deepeye-http', 'deepeye-coordinator', 'deepeye-sample'))
                                 for t in threading.enumerate()))
            self.assertTrue(store.verify()['ok'])

    def test_shared_executor_views_drain_locally_and_nested_work_progresses_at_cap_one(self):
        runtime = SamplingRuntime(coordinator_workers=1, request_workers=1)
        self.addCleanup(runtime.close)
        self.assertTrue(hasattr(runtime, 'executor_view'), 'Native pools need run-owned executor views')
        first, second = runtime.executor_view(), runtime.executor_view()
        result = first.submit(lambda: second.submit(lambda: 42).result(2)).result(3)
        self.assertEqual(result, 42)
        first.shutdown(wait=True)
        self.assertEqual(second.submit(lambda: 43).result(2), 43)
        self.assertEqual(runtime.snapshot()['coordinators']['peak'], 1)

    def test_all_605_questions_are_admitted_with_one_shared_coordinator(self):
        runtime = SamplingRuntime(coordinator_workers=1, request_workers=1)
        self.addCleanup(runtime.close)
        entered, release = threading.Event(), threading.Event()
        base, order = Factory(), {}
        original_pools = []
        def factory(stage, items):
            runner = base(stage, items)
            pool = ThreadPoolExecutor(1)
            original_pools.append(pool)
            runner._inner_thread_pool_executor = pool
            process = getattr(runner, STAGE_METHODS[stage])
            def execute(target):
                if target.instance_id == '0' and stage == STAGES[0]:
                    entered.set()
                    self.assertTrue(release.wait(10))
                order.setdefault(target.instance_id, []).append(stage)
                runner._inner_thread_pool_executor.submit(process, target).result(2)
            setattr(runner, STAGE_METHODS[stage], execute)
            return runner
        with tempfile.TemporaryDirectory() as temporary, RunStore.create(Path(temporary) / 'run', {}) as store:
            recorder = FakeTrace()
            recorder.stop_event = runtime.stop_event
            with ThreadPoolExecutor(1) as caller, redirect_stdout(io.StringIO()):
                future = caller.submit(run_pipeline, store, [('lite', item(str(i))) for i in range(605)],
                                       factory, recorder, workers=1, runtime=runtime)
                try:
                    did_enter = entered.wait(5)
                    if future.done():
                        future.result()
                    self.assertTrue(did_enter)
                    # Every question is queued onto the shared executor while
                    # the only running coordinator is still on question zero.
                    import time
                    deadline = time.monotonic() + 5
                    while runtime.coordinators._work_queue.qsize() < 604 and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertEqual(runtime.coordinators._work_queue.qsize(), 604)
                finally:
                    release.set()
                result = future.result(30)
            self.assertEqual((result['succeeded'], result['failed']), (605, 0))
            self.assertEqual(len(order), 605)
            self.assertTrue(all(stages == list(STAGES) for stages in order.values()))
            self.assertTrue(all(pool._shutdown for pool in original_pools))
            self.assertEqual(runtime.snapshot()['coordinators']['peak'], 1)
            self.assertTrue(store.verify()['ok'])

    def test_cli_defaults_are_effective_runtime_caps_and_formal_budgets(self):
        from tests.test_deepeye_run_entry import DeepEyeRunEntryTests
        parser = entry._build_parser()
        args = parser.parse_args(['run', '--run-dir', 'unused', '--precompute-dir', 'unused'])
        env = {'DASH_MODELS': 'qwen3.8-2.4t-a95b', 'DASH_BASE_URL': 'https://invalid.test',
               'PG_HOST': 'invalid.test', 'PG_PORT': '5432', 'PG_USER': 'reader'}
        effective = entry.build_effective_config(env, args)
        self.assertEqual(effective.get('runtime'), {
            'version': 'shared-sampling-runtime-v1', 'request_limit': 8000,
            'request_workers': 8000, 'coordinator_workers': 6000, 'http_connections': 8000,
            'start_rate': 50.0, 'request_timeout': 660.0, 'retry_delay': 0.0})
        self.assertEqual(effective['chat']['max_tokens'], 16384)
        self.assertEqual(effective['stages']['sql_generation'], {
            'dc_sampling_budget': 4, 'skeleton_sampling_budget': 4, 'icl_sampling_budget': 4})
        self.assertEqual(effective['stages']['sql_revision']['checker_sampling_budget'], 5)
        self.assertEqual(effective['stages']['sql_selection']['evaluator_sampling_budget'], 5)
        for flags in (['--adaptive-concurrency'], ['--workers', '1'], ['--inner-workers', '2'],
                      ['--concurrency-max', '100'], ['--thinking-budget', '5'], ['--inherit-from', 'old']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                entry._validate_run_args(parser, parser.parse_args([
                    'run', '--run-dir', 'unused', '--precompute-dir', 'unused', *flags]))
