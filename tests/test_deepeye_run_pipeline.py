from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'baselines/DeepEye-SQL'))


def item(identity='one'):
    from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataItem
    schema = {'db_id': 'db', 'db_path': 'db', 'db_type': 'postgresql',
        'tables': {'t': {'table_name': 't', 'columns': {'x': {
            'column_type': 'INTEGER', 'primary_key': False, 'foreign_keys': []}}}}}
    return BirdInteractDataItem(question_id=0, instance_id=identity, question='Return x',
        database_id='db', database_path='db', database_schema=schema,
        database_schema_after_value_retrieval=schema, question_keywords=[], retrieved_values={},
        value_retrieval_time=0.0, value_retrieval_llm_cost=cost(), total_time=0.0,
        total_llm_cost=cost(), few_shot_examples=[])


def cost(tokens=0):
    return {'prompt_tokens': tokens, 'completion_tokens': 0, 'total_tokens': tokens}


class FakeTrace:
    error = None
    def context(self, attempt_id):
        return nullcontext()
    def instrument_runner(self, runner, stage):
        return lambda: None
    def raise_if_failed(self):
        if self.error:
            raise self.error


class Factory:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.closed = []

    def __call__(self, stage, items):
        def process(target):
            self.calls.append((stage, target.instance_id))
            if self.failure == (stage, target.instance_id):
                target.total_time += 999
                raise ValueError('controlled stage failure')
            if stage == 'schema_linking':
                for prefix in ('direct', 'reversed', 'value', 'final'):
                    setattr(target, prefix + '_linked_tables_and_columns', {'t': ['x']})
                target.database_schema_after_schema_linking = target.database_schema
            elif stage == 'sql_generation':
                target.sql_candidates = ['SELECT x FROM t', 'SELECT x FROM t']
            elif stage == 'sql_revision':
                target.sql_candidates_after_revision = ['SELECT x FROM t']
            elif stage == 'sql_selection':
                target.final_selected_sql = 'SELECT x FROM t'
            setattr(target, stage + '_time', 1.0)
            setattr(target, stage + '_llm_cost', cost(1))
            target.total_time += 1
            target.total_llm_cost = cost(target.total_llm_cost['total_tokens'] + 1)
        module = importlib.import_module('scripts.baseline_adapters.deepeye.run_pipeline')
        return SimpleNamespace(**{module.STAGE_METHODS[stage]: process,
            '_clean_up': lambda: self.closed.append(stage)})


class PipelineTest(unittest.TestCase):
    def test_swallowed_partial_sampling_group_fails_stage_with_fallback_output(self):
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from app.llm_extractor import LLMExtractor
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        pipeline, Store = self.modules()
        base = Factory()
        llm, calls = llm_fixture([response()] * 3 + [response('bad')] * 4 + [response()])
        def factory(stage, items):
            runner = base(stage, items)
            if stage == 'schema_linking':
                original = runner._link_tables_and_columns
                def process(target):
                    LLMExtractor().extract_with_retry(llm, [], parse, n=5)
                    original(target)  # Native fallback populates every required field.
                runner._link_tables_and_columns = process
                runner._llm = llm
            return runner
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            recorder = TraceRecorder(store)
            # This fixture only has stage entry points, no native component tree.
            with patch.object(recorder, 'instrument_runner', return_value=lambda: None), patch('app.llm_extractor.extractor.logger.warning') as warning:
                result = pipeline.run_pipeline(store, [('lite', item())], factory, recorder, workers=1)
            warning.assert_called_once()
            self.assertEqual((result['failed'], len(calls)), (1, 8))
            row = next(a for a in store.attempts() if a['stage'] == 'schema_linking')
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['payload']['error_type'], 'IncompleteSamplingGroup')
            self.assertEqual(base.calls, [('schema_linking', 'one')])

    def modules(self):
        pipeline = importlib.import_module('scripts.baseline_adapters.deepeye.run_pipeline')
        store = importlib.import_module('scripts.baseline_adapters.deepeye.run_store')
        return pipeline, store.RunStore

    def test_questions_refill_slots_without_waiting_for_a_blocked_first_stage(self):
        # A batch stage barrier would leave B without Selection and C unadmitted.
        pipeline, Store = self.modules()
        release_a, started_c = threading.Event(), threading.Event()
        factory = Factory()
        constructed, errors, results = [], [], []
        def make_runner(stage, items):
            constructed.append(stage)
            runner = factory(stage, items)
            original = getattr(runner, pipeline.STAGE_METHODS[stage])
            def process(target):
                self.assertEqual(constructed, list(pipeline.STAGES))
                if stage == 'schema_linking' and target.instance_id == 'A':
                    if not release_a.wait(5):
                        raise AssertionError('test did not release A')
                if stage == 'schema_linking' and target.instance_id == 'C':
                    started_c.set()
                original(target)
            setattr(runner, pipeline.STAGE_METHODS[stage], process)
            return runner
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            def run():
                try:
                    results.append(pipeline.run_pipeline(store, [('lite', item(k)) for k in 'ABC'],
                        make_runner, FakeTrace(), workers=2))
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(started_c.wait(3), 'C must start while A remains blocked')
                attempts = store.attempts()
                b_selection = next(a for a in attempts if a['item_key'] == 'lite/B' and a['stage'] == 'sql_selection')
                self.assertEqual(b_selection['status'], 'succeeded')
                self.assertEqual(b_selection['payload']['artifact']['final_selected_sql'], 'SELECT x FROM t')
                b_master = next(a for a in attempts if a['item_key'] == 'lite/B' and a['stage'] == 'pipeline')
                self.assertEqual(b_master['status'], 'succeeded')
                self.assertEqual(len(b_master['payload']['stage_attempts']), 4)
                self.assertEqual(b_master['payload']['ticket']['item_key'], 'lite/B')
                self.assertTrue(any(e['kind'] == 'pipeline_release_intent' and e['attempt_id'] == b_master['attempt_id']
                    for e in store.events()))
            finally:
                release_a.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results[0]['succeeded'], 3)
            store.verify()

    def test_failure_durably_releases_slot_before_next_question(self):
        pipeline, Store = self.modules()
        factory = Factory(('sql_generation', 'A'))
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            def make_runner(stage, items):
                runner = factory(stage, items)
                original = getattr(runner, pipeline.STAGE_METHODS[stage])
                def process(target):
                    if stage == 'schema_linking' and target.instance_id == 'B':
                        master = next(a for a in store.attempts() if a['stage'] == 'pipeline' and a['item_key'] == 'lite/A')
                        self.assertEqual(master['status'], 'failed')
                        self.assertEqual(master['payload']['failed_stage'], 'sql_generation')
                    original(target)
                setattr(runner, pipeline.STAGE_METHODS[stage], process)
                return runner
            result = pipeline.run_pipeline(store, [('lite', item(k)) for k in 'AB'], make_runner, FakeTrace(), workers=1)
            self.assertEqual((result['failed'], result['succeeded']), (1, 1))
            masters = [a for a in store.attempts() if a['stage'] == 'pipeline']
            self.assertEqual([a['status'] for a in masters], ['failed', 'succeeded'])

    def test_model_health_growth_can_start_a_new_pipeline_while_old_pipeline_is_blocked(self):
        # Executor width tied to initial workers would make the extra slot inert.
        pipeline, Store = self.modules()
        from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
        from scripts.baseline_adapters.deepeye.run_admission import AdaptivePolicy
        now, started_b = [0.0], threading.Event()
        slots = PipelineSlots(policy=AdaptivePolicy(initial_limit=1, step=1, min_limit=1,
            max_limit=2, stable_window_s=1, min_successes=1), clock=lambda: now[0])
        factory = Factory()
        def make_runner(stage, items):
            runner = factory(stage, items)
            original = getattr(runner, pipeline.STAGE_METHODS[stage])
            def process(target):
                if stage == 'schema_linking' and target.instance_id == 'A':
                    now[0] = 1.0
                    slots(lambda: 'healthy response', (), {})
                    self.assertTrue(started_b.wait(2), 'Growth must refill without a completed pipeline')
                elif stage == 'schema_linking' and target.instance_id == 'B':
                    started_b.set()
                original(target)
            setattr(runner, pipeline.STAGE_METHODS[stage], process)
            return runner
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            result = pipeline.run_pipeline(store, [('lite', item(k)) for k in 'AB'], make_runner, FakeTrace(),
                workers=1, slot_controller=slots)
            self.assertEqual(result['succeeded'], 2)
            self.assertEqual(slots.snapshot()['peak_active'], 2)
            self.assertEqual(len([a for a in store.attempts() if a['stage'] == 'sql_selection'
                and a['status'] == 'succeeded']), 2)

    def test_all_complete_resume_creates_no_attempt_or_runner(self):
        pipeline, Store = self.modules()
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            pipeline.run_pipeline(store, [('lite', item())], Factory(), FakeTrace())
            before = store.attempts()
            result = pipeline.run_pipeline(store, [('lite', item())],
                lambda *args: self.fail('Completed resume constructed a runner'), FakeTrace())
            self.assertEqual(result['succeeded'], 1)
            self.assertEqual(store.attempts(), before)

    def test_invalid_later_prefix_is_rejected_before_any_constructor(self):
        pipeline, Store = self.modules()
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            pipeline.run_pipeline(store, [('lite', item('B'))], Factory(), FakeTrace())
            actual_completed = store.completed
            def completed(key, stage, digest):
                prior = actual_completed(key, stage, digest)
                if prior and stage == 'sql_generation':
                    prior['payload']['artifact']['sql_candidates'] = []
                return prior
            with patch.object(store, 'completed', side_effect=completed):
                with self.assertRaises(ValueError):
                    pipeline.run_pipeline(store, [('lite', item(k)) for k in 'AB'],
                        lambda *args: self.fail('Preflight must precede constructors'), FakeTrace())

    def test_fatal_finish_error_halts_admission_and_leaves_master_unfinished(self):
        pipeline, Store = self.modules()
        from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
        slots = PipelineSlots(fixed_limit=1)
        factory = Factory()
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            with patch.object(store, 'finish_attempt', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    pipeline.run_pipeline(store, [('lite', item(k)) for k in 'ABC'], factory, FakeTrace(),
                        workers=1, slot_controller=slots)
            self.assertEqual(factory.calls, [('schema_linking', 'A')])
            self.assertTrue(any(a['stage'] == 'pipeline' for a in store.attempts()))
            self.assertFalse(any(a['status'] == 'succeeded' for a in store.attempts()))
            self.assertEqual(slots.snapshot()['active'], 0, 'Fatal tickets are reclaimed after all work drains')

    def test_stage_failure_waits_for_its_native_inner_work_before_releasing_slot(self):
        pipeline, Store = self.modules()
        release, entered, next_started = threading.Event(), threading.Event(), threading.Event()
        factory = Factory()
        errors, results = [], []
        def make_runner(stage, items):
            runner = factory(stage, items)
            if stage == 'schema_linking':
                runner._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=2)
                original = runner._link_tables_and_columns
                def process(target):
                    if target.instance_id == 'A':
                        def child():
                            runner._inner_thread_pool_executor.submit(lambda: (entered.set(), release.wait(5)))
                        runner._inner_thread_pool_executor.submit(child)
                        raise ValueError('native parent failed before joining branch')
                    next_started.set()
                    original(target)
                runner._link_tables_and_columns = process
            return runner
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            def run():
                try:
                    results.append(pipeline.run_pipeline(store, [('lite', item(k)) for k in 'AB'],
                        make_runner, FakeTrace(), workers=1))
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertFalse(next_started.wait(0.1), 'Slot released while its native branch remained alive')
                self.assertFalse(any(a['status'] == 'failed' for a in store.attempts()))
            finally:
                release.set()
                thread.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(results[0]['succeeded'], 1)

    def test_storage_failure_during_refill_prevents_additional_worker_submission(self):
        pipeline, Store = self.modules()
        from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
        failed = threading.Event()
        class PausingSlots(PipelineSlots):
            def try_acquire(self, key):
                ticket = super().try_acquire(key)
                if key == 'lite/B':
                    if not failed.wait(2):
                        raise AssertionError('Expected the first worker to reach the failing store')
                return ticket
        slots = PausingSlots(fixed_limit=3)
        calls = []
        def begin(key, stage, digest):
            calls.append((key, stage))
            failed.set()
            raise OSError('store failed during admission burst')
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            with patch.object(store, 'begin_attempt', side_effect=begin):
                with self.assertRaises(OSError):
                    pipeline.run_pipeline(store, [('lite', item(k)) for k in 'ABC'], Factory(), FakeTrace(),
                        slot_controller=slots)
            self.assertEqual(calls, [('lite/A', 'pipeline')])
            self.assertEqual(slots.snapshot()['active'], 0)

    def test_outer_shutdown_interrupt_keeps_guard_trace_and_slots_until_question_worker_drains(self):
        pipeline, Store = self.modules()
        from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
        from app.services.schema_service import get_schema_service
        entered, release, interrupted, cleaned = (threading.Event() for _ in range(4))
        interrupt = KeyboardInterrupt('outer shutdown interrupted')
        class InterruptedPool(ThreadPoolExecutor):
            def shutdown(self, *args, **kwargs):
                if not interrupted.is_set():
                    interrupted.set()
                    raise interrupt
                return super().shutdown(*args, **kwargs)
        slots, factory, errors, undone = PipelineSlots(fixed_limit=1), Factory(), [], []
        trace = FakeTrace()
        trace.instrument_runner = lambda runner, stage: lambda: undone.append(stage)
        actual_check = trace.raise_if_failed
        def check():
            if entered.is_set() and threading.current_thread() is run_thread:
                trace.error = OSError('controlled fatal recorder failure')
            actual_check()
        trace.raise_if_failed = check
        def make_runner(stage, items):
            runner = factory(stage, items)
            runner._clean_up = lambda: cleaned.set()
            if stage == 'schema_linking':
                original = runner._link_tables_and_columns
                def process(target):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError('Test failed to release native worker')
                    original(target)
                runner._link_tables_and_columns = process
            return runner
        service = get_schema_service()
        original_cache = service._schema_profile_cache
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            def run():
                try:
                    with patch.object(pipeline, 'ThreadPoolExecutor', InterruptedPool):
                        pipeline.run_pipeline(store, [('lite', item())], make_runner, trace, slot_controller=slots)
                except BaseException as exc:
                    errors.append(exc)
            run_thread = threading.Thread(target=run)
            run_thread.start()
            try:
                self.assertTrue(interrupted.wait(2))
                self.assertFalse(cleaned.wait(0.1), 'Native cleanup preceded root question drain')
                self.assertEqual(undone, [])
                self.assertEqual(slots.snapshot()['active'], 1)
                self.assertIsNot(service._schema_profile_cache, original_cache)
                self.assertTrue(pipeline._PIPELINE_LOCK.locked())
            finally:
                release.set()
                run_thread.join(5)
            self.assertFalse(run_thread.is_alive())
            self.assertEqual(errors, [interrupt])
            self.assertTrue(cleaned.is_set())
            self.assertEqual(len(undone), 4)
            self.assertEqual(slots.snapshot()['active'], 0)
            self.assertIs(service._schema_profile_cache, original_cache)
            self.assertFalse(pipeline._PIPELINE_LOCK.locked())
            self.assertFalse(any(a['status'] == 'succeeded' for a in store.attempts()))

    def test_native_partial_constructor_failure_drains_and_closes_allocated_resources(self):
        pipeline, _ = self.modules()
        module = importlib.import_module('app.pipeline.sql_generation.sql_generation')
        from scripts.deepeye_bird_interact_smoke import build_runtime_config
        env = {'DASH_MODELS': 'fake', 'DASH_BASE_URL': 'https://invalid.test/v1', 'DASH_API_KEY': 'fake',
            'EMBEDDING_MODEL': 'fake', 'EMBEDDING_BASE_URL': 'https://invalid.test/v1', 'EMBEDDING_API_KEY': 'fake'}
        captured, pools, artifacts = [], [], []
        original_init = module.SQLGenerationRunner.__init__
        def capture(runner, *args, **kwargs):
            captured.append(runner)
            try:
                original_init(runner, *args, **kwargs)
            finally:
                pools.extend([runner._thread_pool_executor, runner._inner_thread_pool_executor])
                artifacts.append(runner._artifact_store)
        with tempfile.TemporaryDirectory() as temp:
            config = build_runtime_config(env, 'lite', Path(temp), Path(temp) / 'native')
            original_loader = module.load_stage_dataset
            try:
                with patch.object(module.SQLGenerationRunner, '__init__', new=capture), \
                     patch.object(module, 'ICLGenerator', side_effect=ValueError('constructor failed')):
                    with self.assertRaisesRegex(ValueError, 'constructor failed'):
                        pipeline.native_runner_factory(config)('sql_generation', [item()])
                self.assertTrue(all(pool._shutdown for pool in pools))
                self.assertTrue(all(artifact._closed for artifact in artifacts))
                self.assertIs(module.load_stage_dataset, original_loader)
            finally:
                for runner in captured:
                    runner._clean_up()

    def test_failed_item_does_not_block_other_items_and_resume_only_retries_missing_stages(self):
        pipeline, Store = self.modules()
        tasks = [('lite', item('one')), ('full', item('two'))]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            with Store.create(run, {'model': 'fake'}) as store:
                first = Factory(('sql_generation', 'one'))
                report = pipeline.run_pipeline(store, tasks, first, FakeTrace(), workers=2)
                self.assertEqual(report['succeeded'], 1)
                self.assertEqual(report['failed'], 1)
                old_ids = [a['attempt_id'] for a in store.attempts()]
                failed_attempt = next(a for a in store.attempts() if a['status'] == 'failed' and a['stage'] == 'sql_generation')
                self.assertGreaterEqual(failed_attempt['payload']['attempt_wall_seconds'], 0)
                self.assertEqual(failed_attempt['payload']['error_message'], 'controlled stage failure')
                self.assertEqual(tasks[0][1].total_time, 0.0, 'Caller inputs must remain pristine')
            with Store.open(run) as store:
                second = Factory()
                report = pipeline.run_pipeline(store, tasks, second, FakeTrace(), workers=2)
                self.assertEqual(second.calls, [('sql_generation', 'one'), ('sql_revision', 'one'), ('sql_selection', 'one')])
                self.assertEqual(report['succeeded'], 2)
                self.assertTrue(set(old_ids).issubset(a['attempt_id'] for a in store.attempts()))
                completed = [a for a in store.attempts() if a['item_key'] == 'lite/one' and a['stage'] == 'sql_selection'][0]
                self.assertEqual(completed['status'], 'succeeded')
                self.assertEqual(completed['payload']['artifact']['final_selected_sql'], 'SELECT x FROM t')
                history = [a for a in store.attempts() if a['item_key'] == 'lite/one' and a['stage'] == 'sql_generation']
                self.assertEqual([a['attempt_no'] for a in history], [1, 2])
                final = report['items']['lite/one']
                self.assertEqual(final['total_time'], 4.0)

    def test_changed_question_refuses_resume_before_any_new_calls(self):
        pipeline, Store = self.modules()
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            pipeline.run_pipeline(store, [('lite', item())], Factory(), FakeTrace())
            changed = item()
            changed.question = 'Different question'
            factory = Factory()
            with self.assertRaises(ValueError):
                pipeline.run_pipeline(store, [('lite', changed)], factory, FakeTrace())
            self.assertEqual(factory.calls, [])

    def test_gold_and_duplicate_identifiers_are_rejected_without_writes(self):
        pipeline, Store = self.modules()
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            target = item()
            target.gold_sql = 'SELECT hidden_answer'
            with self.assertRaises(ValueError):
                pipeline.run_pipeline(store, [('lite', target)], Factory(), FakeTrace())
            with self.assertRaises(ValueError):
                pipeline.run_pipeline(store, [('lite', item()), ('lite', item())], Factory(), FakeTrace())
            self.assertEqual(store.attempts(), [])

    def test_swallowed_storage_failure_cannot_be_marked_success(self):
        pipeline, Store = self.modules()
        trace = FakeTrace()
        factory = Factory()
        def failing_factory(stage, items):
            runner = factory(stage, items)
            original = getattr(runner, pipeline.STAGE_METHODS[stage])
            def process(target):
                original(target)
                trace.error = OSError('disk full')
            setattr(runner, pipeline.STAGE_METHODS[stage], process)
            return runner
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            with self.assertRaises(OSError):
                pipeline.run_pipeline(store, [('lite', item())], failing_factory, trace)
            self.assertFalse(any(a['status'] == 'succeeded' for a in store.attempts()))
            self.assertEqual(factory.closed, list(pipeline.STAGES))

    def test_native_factory_only_injects_dataset_and_restores_loader(self):
        pipeline, _ = self.modules()
        module = importlib.import_module('app.pipeline.sql_generation.sql_generation')
        from scripts.deepeye_bird_interact_smoke import build_runtime_config
        env = {'DASH_MODELS': 'fake', 'DASH_BASE_URL': 'https://invalid.test/v1', 'DASH_API_KEY': 'fake',
            'EMBEDDING_MODEL': 'fake', 'EMBEDDING_BASE_URL': 'https://invalid.test/v1', 'EMBEDDING_API_KEY': 'fake'}
        with tempfile.TemporaryDirectory() as temp:
            config = build_runtime_config(env, 'lite', Path(temp), Path(temp) / 'native')
            original = module.load_stage_dataset
            runner = pipeline.native_runner_factory(config)('sql_generation', [item()])
            try:
                self.assertEqual(runner._dataset[0].instance_id, 'one')
                self.assertIs(module.load_stage_dataset, original)
                self.assertEqual(runner._dc_generator.__class__.__name__, 'DCGenerator')
            finally:
                runner._clean_up()
            self.assertFalse(list(Path(temp).rglob('*.snapshot')))

    def test_client_is_closed_when_instrumentation_fails(self):
        pipeline, Store = self.modules()
        closed = []
        client = SimpleNamespace(close=lambda: closed.append('client'))
        runner = SimpleNamespace(_llm=SimpleNamespace(_get_client=lambda: client),
            _clean_up=lambda: closed.append('runner'))
        trace = FakeTrace()
        trace.instrument_runner = lambda *args: (_ for _ in ()).throw(ValueError('partial instrumentation'))
        with tempfile.TemporaryDirectory() as temp, Store.create(Path(temp) / 'run', {}) as store:
            with self.assertRaises(ValueError):
                pipeline.run_pipeline(store, [('lite', item())], lambda *args: runner, trace)
            self.assertEqual(closed, ['runner', 'client'])

    def test_all_four_native_stages_with_fake_services_record_and_resume_offline(self):
        self._native_offline_fixture(cross_stage=False)

    def test_native_questions_cross_stages_and_keep_services_alive_until_all_pools_drain(self):
        self._native_offline_fixture(cross_stage=True)

    def _native_offline_fixture(self, cross_stage):
        pipeline, Store = self.modules()
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from scripts.baseline_adapters.deepeye.run_usage import observed_usage
        from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
        from app.llm import LLM
        from app.services.schema_service import SchemaService, get_schema_service
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID
        from app.db_utils.execution import SQLExecutionResult
        from openai.types.chat import ChatCompletion
        from scripts.deepeye_bird_interact_smoke import build_runtime_config
        from scripts.deepeye_bird_interact_run import admission_context, _build_parser
        args = _build_parser().parse_args(['run', '--run-dir', 'unused',
            '--precompute-dir', 'unused'])
        requests, native_pools = [], []
        release_a, crossed = threading.Event(), threading.Event()
        original_reset = SchemaService.reset
        def reset(service):
            self.assertTrue(all(pool._shutdown for pool in native_pools), 'Reset preceded native pool drain')
            original_reset(service)
        def create(**kwargs):
            requests.append(kwargs)
            if cross_stage:
                attempt = next(a for a in store.attempts() if a['attempt_id'] == _ATTEMPT_ID.get())
                self.assertEqual(len(get_schema_service()._schema_profile_cache), 0)
                if attempt['item_key'] == 'lite/A' and attempt['stage'] == 'schema_linking':
                    if not release_a.wait(3):
                        raise AssertionError('Native A blocked the independently completing B/C pipelines')
                if attempt['item_key'] == 'lite/C':
                    b_selection = next(a for a in store.attempts() if a['item_key'] == 'lite/B' and a['stage'] == 'sql_selection')
                    self.assertEqual(b_selection['status'], 'succeeded')
                    self.assertEqual(b_selection['payload']['artifact']['final_selected_sql'], 'SELECT x FROM t')
                    crossed.set()
                    release_a.set()
            prompt = kwargs['messages'][0]['content']
            content = ('<result><table table_name="t"><column column_name="x" /></table></result>'
                if 'pinpoint the specific tables and columns' in prompt else '<result>SELECT x FROM t</result>')
            return ChatCompletion(id='offline', created=0, model='fixture', object='chat.completion',
                choices=[{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}],
                usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5})
        def execute(target, sql, timeout=None):
            return SQLExecutionResult(result_type='success', db_path='db', sql=sql,
                execution_time=0.01, result_rows=[(1,)], result_cols=['x'])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
        env = {'DASH_MODELS': 'fixture', 'DASH_BASE_URL': 'https://invalid.test/v1', 'DASH_API_KEY': 'fake',
            'EMBEDDING_MODEL': 'fake', 'EMBEDDING_BASE_URL': 'https://invalid.test/v1', 'EMBEDDING_API_KEY': 'fake'}
        target = item()
        target.few_shot_examples = [{'question': 'A separate training question', 'evidence': '', 'sql': 'SELECT 2'}]
        tasks = [('lite', target)]
        if cross_stage:
            tasks = [('lite', item(k)) for k in 'ABC']
            for _, fixture in tasks:
                fixture.few_shot_examples = target.few_shot_examples
        expected_items, expected_calls = (3, 15) if cross_stage else (1, 5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run'
            cfg = build_runtime_config(env, 'lite', Path(temp), path / 'native')
            cfg.run_config.parallelism = 8
            original_factory = pipeline.native_runner_factory(cfg)
            def native_factory(stage, items):
                runner = original_factory(stage, items)
                native_pools.extend([runner._thread_pool_executor, runner._inner_thread_pool_executor])
                return runner
            undo = install_postgres_support()
            try:
                with patch.object(LLM, '_create_client', return_value=client), \
                     patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=lambda text: list(text))), \
                     patch.object(SchemaService, 'reset', new=reset), \
                     patch('scripts.baseline_adapters.deepeye.backend_hooks.execute_postgres_sql', side_effect=execute), \
                     patch('socket.socket.connect', side_effect=AssertionError('No network allowed')):
                    with Store.create(path, {'test': 'all_native_stages'}) as store:
                        trace = TraceRecorder(store)
                        with trace.install(), admission_context(trace, args, population=2) as gates:
                            result = pipeline.run_pipeline(store, tasks, native_factory, trace,
                                slot_controller=gates['pipeline'])
                        self.assertEqual(result['succeeded'], expected_items, store.attempts())
                        for outcome in result['items'].values():
                            self.assertEqual(outcome['final_selected_sql'], 'SELECT x FROM t')
                        self.assertEqual(len(requests), expected_calls)
                        self.assertEqual(len([a for a in store.attempts() if a['stage'] in pipeline.STAGES]), expected_items * 4)
                        self.assertEqual(len([a for a in store.attempts() if a['stage'] == 'pipeline']), expected_items)
                        api_requests = [e for e in store.events() if e['kind'] == 'api_request']
                        self.assertEqual(len(api_requests), expected_calls)
                        stage_ids = {a['attempt_id'] for a in store.attempts() if a['stage'] in pipeline.STAGES}
                        self.assertTrue(all(e['attempt_id'] in stage_ids and e['payload']['branch_path'] for e in api_requests))
                        if cross_stage:
                            self.assertTrue(crossed.is_set())
                            self.assertEqual(gates['pipeline'].snapshot()['peak_active'], 2)
                        admitted = [e['payload']['call_id'] for e in store.events() if e['kind'] == 'api_request']
                        responded = [e['payload']['call_id'] for e in store.events() if e['kind'] == 'api_response']
                        self.assertEqual(sorted(admitted), sorted(responded))
                        self.assertNotIn('model', gates['pipeline'].snapshot())
                        self.assertGreater(gates['postgres'].snapshot()['completed'], 0)
                        self.assertEqual(observed_usage(store)['reported_tokens']['total_tokens'], expected_calls * 5)
                        self.assertTrue(any(e['kind'] == 'sql_execute_result' for e in store.events()))
                        store.verify()
                    with Store.open(path) as store:
                        trace = TraceRecorder(store)
                        with trace.install(), admission_context(trace, args) as gates:
                            result = pipeline.run_pipeline(store, tasks,
                                lambda *args: self.fail('Completed stages must not construct runners'), trace,
                                slot_controller=gates['pipeline'])
                        self.assertEqual(result['succeeded'], expected_items)
                        self.assertEqual(len(requests), expected_calls)
                        self.assertNotIn('model', gates['pipeline'].snapshot())
                        self.assertEqual(gates['postgres'].snapshot()['completed'], 0)
                        store.export(Path(temp) / 'export')
            finally:
                undo()


if __name__ == '__main__':
    unittest.main()
