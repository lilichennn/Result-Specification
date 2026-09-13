"""Offline acceptance of the run-owned runtime in the native entrypoint."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
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
    def test_actual_native_workflow_overlaps_generation_branches_and_revision_candidates(self):
        from types import SimpleNamespace
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from tests.deepeye_native_overlap import (NativeOverlapTransport, GENERATION_SQL, REVISED_SQL,
                                                 execute_sql, assert_overlap, assert_sequential_checkers)
        from scripts.rc_evaluation.deepeye.tests.test_dynamic_concurrency import AsyncClientFixture
        from scripts.rc_evaluation.deepeye.tests.test_cli import ENV
        args = entry._build_parser().parse_args(['run', '--run-dir', 'unused', '--precompute-dir', 'unused',
            '--request-limit', '32', '--request-workers', '32', '--coordinator-workers', '32',
            '--http-connections', '32', '--request-start-rate', '10000'])
        with tempfile.TemporaryDirectory() as temporary, RunStore.create(Path(temporary) / 'run', {}) as store:
            transport = NativeOverlapTransport(store)
            client = AsyncClientFixture(transport.create)
            target = item('a')
            target.few_shot_examples = [{'question': 'Other question', 'evidence': '', 'sql': 'SELECT 2'}]
            with patch('openai.AsyncOpenAI', return_value=client), \
                 patch.object(LLM, '_create_client', side_effect=AssertionError('sync client forbidden')), \
                 patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)), \
                 patch('scripts.baseline_adapters.deepeye.backend_hooks.execute_postgres_sql', side_effect=execute_sql), \
                 patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), \
                 redirect_stdout(io.StringIO()):
                result = entry._execute_pipeline(store, [('lite', target)], ENV, args)
            self.assertEqual((result['succeeded'], result['failed']), (1, 0))
            rows = {a['stage']: a for a in store.attempts()}
            self.assertEqual(rows['sql_generation']['payload']['artifact']['sql_candidates'], GENERATION_SQL)
            self.assertEqual(rows['sql_revision']['payload']['artifact']['sql_candidates_after_revision'], REVISED_SQL * 6)
            assert_sequential_checkers(self, store, rows['sql_revision'])
            for stage, expected in [('sql_generation', 12), ('sql_revision', 10)]:
                with self.subTest(stage=stage):
                    assert_overlap(self, transport, 'lite/a', stage, expected)
            self.assertTrue(client.closed)
            self.assertTrue(store.verify()['ok'])

    def test_stop_between_workflow_admission_and_submission_is_a_pause(self):
        runtime = SamplingRuntime(coordinator_workers=1, request_workers=1)
        self.addCleanup(runtime.close)
        recorder = FakeTrace()
        recorder.stop_event = runtime.stop_event
        original = runtime.submit_workflow
        def stop_at_submission(*args, **kwargs):
            runtime.stop()
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, RunStore.create(Path(temporary) / 'run', {}) as store, \
                patch.object(runtime, 'submit_workflow', stop_at_submission):
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
                    with recorder.context(aid), patch('app.llm_extractor.extractor.logger.warning') as warning:
                        LLMExtractor().extract_with_retry(llm, [], parse, n=2)
                    warning.assert_called_once_with('Incomplete sampling group: 1/2')
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
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID, _BRANCH_PATH
        from app.llm.sampling import sampling_identity
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
            # Hold one remote response, leaving capacity for the other question.
            # Holding all of b's parallel branches would exhaust the fake cap
            # while waiting for an unrelated future PG operation.
            if (row['item_key'] == 'lite/b' and 'schema_linking.direct' in _BRANCH_PATH.get()
                    and sampling_identity()['sample_index'] == 0):
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
            self.assertFalse(any(t.name.startswith(('deepeye-http', 'deepeye-coordinator', 'deepeye-sample', 'deepeye-workflow'))
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
        population_lock, population = threading.Lock(), 0
        base, order = Factory(), {}
        original_pools = []
        def factory(stage, items):
            runner = base(stage, items)
            pool = ThreadPoolExecutor(1)
            original_pools.append(pool)
            runner._inner_thread_pool_executor = pool
            process = getattr(runner, STAGE_METHODS[stage])
            def execute(target):
                nonlocal population
                if stage == STAGES[0]:
                    with population_lock:
                        population += 1
                        if population == 605:
                            entered.set()
                    self.assertTrue(release.wait(15))
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
                    self.assertEqual(runtime.snapshot()['workflows'], {'cap': 605, 'active': 605, 'peak': 605})
                    self.assertEqual(runtime.snapshot()['coordinators']['active'], 0)
                finally:
                    release.set()
                result = future.result(30)
            self.assertEqual((result['succeeded'], result['failed']), (605, 0))
            self.assertEqual(len(order), 605)
            self.assertTrue(all(stages == list(STAGES) for stages in order.values()))
            self.assertTrue(all(pool._shutdown for pool in original_pools))
            self.assertEqual(runtime.snapshot()['coordinators']['peak'], 1)
            self.assertTrue(store.verify()['ok'])

    def test_runtime_close_drains_workflows_before_their_native_children(self):
        runtime = SamplingRuntime(coordinator_workers=1, request_workers=1)
        self.addCleanup(runtime.close)
        workflow = runtime.workflow_executor(1)
        entered, closing = threading.Event(), threading.Event()
        original_shutdown = runtime.workflows.shutdown
        def shutdown(*args, **kwargs):
            closing.set()
            return original_shutdown(*args, **kwargs)
        def parent():
            entered.set()
            self.assertTrue(closing.wait(3))
            return runtime.executor_view().submit(lambda: 42).result(3)
        future = workflow.submit(parent)
        self.assertTrue(entered.wait(3))
        with patch.object(runtime.workflows, 'shutdown', shutdown):
            runtime.close()
        self.assertEqual(future.result(), 42)
        self.assertEqual(runtime.snapshot()['workflows'], {'cap': 1, 'active': 0, 'peak': 1})
        self.assertEqual(runtime.snapshot()['coordinators']['active'], 0)

    def test_failed_workflow_thread_start_cannot_run_unowned_work(self):
        runtime = SamplingRuntime(coordinator_workers=1, request_workers=1)
        self.addCleanup(runtime.close)
        workflow = runtime.workflow_executor(2)
        unowned = []
        original_start = threading.Thread.start
        rejected = False
        def start(thread):
            nonlocal rejected
            if thread.name.startswith('deepeye-workflow') and not rejected:
                rejected = True
                raise RuntimeError('injected workflow thread creation failure')
            return original_start(thread)
        with patch.object(threading.Thread, 'start', start):
            with self.assertRaisesRegex(RuntimeError, 'injected workflow thread'):
                workflow.submit(lambda: unowned.append('ran'))
            self.assertEqual(workflow.submit(lambda: 'owned').result(3), 'owned')
        runtime.close()
        self.assertEqual(unowned, [])
        self.assertEqual(runtime.snapshot()['workflows']['active'], 0)

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
            diagnostics = io.StringIO()
            with self.subTest(flags=flags):
                with redirect_stderr(diagnostics), self.assertRaises(SystemExit) as failure:
                    entry._validate_run_args(parser, parser.parse_args([
                        'run', '--run-dir', 'unused', '--precompute-dir', 'unused', *flags]))
                self.assertEqual(failure.exception.code, 2)
                self.assertIn('error:', diagnostics.getvalue())
