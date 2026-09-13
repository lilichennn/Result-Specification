"""Offline RC execution through the real pipeline admission boundary."""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import OfflineTestCase
from .test_cli import ENV, native_args
from .test_runner import experiment_manifest
from tests.test_deepeye_run_inheritance import make_item, manifest, complete_stage, cost
from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, STAGE_METHODS, _checkpoint
from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from scripts.deepeye_bird_interact_run import build_effective_config, code_source_hashes
from scripts.rc_evaluation.deepeye import cli
from scripts.rc_evaluation.deepeye.source import snapshot_source


class Clock:
    """Only controller time advances; thread synchronization uses real events."""
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class DynamicConcurrencyTests(OfflineTestCase):
    def prepared(self, temporary, *, calls=0, keys='a', stage='sql_revision'):
        root = Path(temporary)
        args = SimpleNamespace(**{
            **vars(native_args()), 'adaptive_concurrency': True,
            'inner_workers': 8,
            'concurrency_initial': 2, 'concurrency_step': 1,
            'concurrency_min': 1, 'concurrency_max': 3,
            'concurrency_window': 1.0,
        })
        config = build_effective_config(ENV, args)
        tasks = [('lite', make_item(key)) for key in keys]
        source = manifest(tasks, upgrade=True)
        source['effective_config'] = config
        source['sources']['code'] = code_source_hashes()
        with RunStore.create(root / 'source', source) as store:
            for _, original in tasks:
                item = copy.deepcopy(original)
                identity = fingerprint({'manifest': source,
                                        'input': to_jsonable(item.model_dump(exclude={'gold_sql'}))})
                for current in STAGES:
                    input_hash = fingerprint({'input': identity, 'stage': current})
                    complete_stage(item, current)
                    count = calls if current == stage else 1
                    setattr(item, current + '_llm_cost', cost(count))
                    payload = _checkpoint(item, current)
                    payload['attempt_wall_seconds'] = 1.0
                    attempt = store.begin_attempt(f'lite/{item.instance_id}', current, input_hash)
                    for number in range(count):
                        call = f'{attempt}-{number}'
                        store.append_event(attempt, 'api_request', {'call_id': call})
                        store.append_event(attempt, 'api_response', {'call_id': call,
                            'response': {'usage': cost(1)}})
                    store.finish_attempt(attempt, 'succeeded', payload)
                    identity = fingerprint({'input': input_hash, 'output': payload})
        data = experiment_manifest(snapshot_source(root / 'source', tasks, stage))
        data['target_stage'] = stage
        data['sources'] = {**data['sources'], 'rc_evaluation_code_sha256': cli.production_hash()}
        return RunStore.create(root / 'run', data)

    @contextmanager
    def controlled_slots(self, clock):
        captured = {}

        def create(*args, **kwargs):
            slots = PipelineSlots(*args, **kwargs, clock=clock)
            captured['pipeline'] = slots
            return slots

        with patch('scripts.baseline_adapters.deepeye.run_slots.PipelineSlots', side_effect=create):
            yield captured

    @contextmanager
    def fake_revision(self, execute, *, create=None, cleanup=None):
        """Keep the bounded factory and trace wrappers; replace native work only."""
        from app.llm import LLM
        from openai.types.chat import ChatCompletion

        closed = threading.Event()

        def response(**kwargs):
            if create is not None:
                create(**kwargs)
            return ChatCompletion(id='offline', created=0, model='offline', object='chat.completion',
                choices=[{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': '<result>SELECT 1</result>'}}],
                usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5})

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=response)),
                                 close=closed.set)

        def factory(stage, items):
            self.assertEqual(stage, 'sql_revision')
            runner = SimpleNamespace(_llm=SimpleNamespace(_get_client=lambda: client),
                                     _checkers=[], _clean_up=cleanup or (lambda: None))

            def revise(item):
                execute(item, client.chat.completions.create)
                complete_stage(item, stage)

            setattr(runner, STAGE_METHODS[stage], revise)
            return runner

        with patch('scripts.baseline_adapters.deepeye.run_pipeline.native_runner_factory',
                   return_value=factory), \
             patch.object(LLM, '_create_client', side_effect=AssertionError('real model client forbidden')):
            yield closed

    def test_dynamic_replay_reconstructs_slots_without_native_clients(self):
        # Break caught: frozen dynamic settings are downgraded, or replay skips
        # the admission boundary / constructs a native or database client.
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            with patch.object(cli, 'build_runtime_config', side_effect=AssertionError('replay built native config')), \
                 patch.object(cli, 'bounded_runner_factory', side_effect=AssertionError('replay built model clients')), \
                 patch('scripts.baseline_adapters.deepeye.hooks.install_postgres_support',
                       side_effect=AssertionError('replay installed database resources')):
                result = cli.execute_run(store, ENV)
            self.assertEqual(result['succeeded'], 1)
            slots = result['admission']['pipeline']
            self.assertEqual((slots['current_limit'], slots['max_limit']), (2, 3))
            self.assertEqual((slots['completed'], slots['active'], slots['pending']), (1, 0, 0))
            self.assertEqual(slots['model']['requested'], 0)
            self.assertEqual(store.attempts()[0]['payload']['execution_origin'], 'reused_no_native_llm_call')
            self.assertTrue(store.verify()['ok'])

    def test_native_workers_one_admits_two_questions_and_refills_pending(self):
        # Break caught: the RC executor sizes its pool from workers=1, or waits
        # for every admitted question before refilling a freed pipeline slot.
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from openai.types.chat import ChatCompletion
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID

        entered = {key: threading.Event() for key in 'abc'}
        release = {key: threading.Event() for key in 'abc'}
        closed = threading.Event()
        with tempfile.TemporaryDirectory() as temporary, \
             self.prepared(temporary, calls=1, keys='abc', stage='sql_generation') as store:
            def create(**kwargs):
                attempt = next(row for row in store.attempts() if row['attempt_id'] == _ATTEMPT_ID.get())
                key = attempt['item_key'].split('/')[1]
                entered[key].set()
                if not release[key].wait(5):
                    raise AssertionError(f'native question {key} did not receive release')
                return ChatCompletion(id='offline', created=0, model='offline', object='chat.completion',
                    choices=[{'index': 0, 'finish_reason': 'stop', 'message': {
                        'role': 'assistant', 'content': '<result>SELECT x FROM t</result>'}}],
                    usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5})

            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                     close=closed.set)
            with patch.object(LLM, '_create_client', return_value=client), \
                 patch.object(SchemaService, '_get_encoding',
                              return_value=SimpleNamespace(encode=lambda text: list(text))), \
                 ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(cli.execute_run, store, ENV)
                try:
                    self.assertTrue(entered['a'].wait(5), 'first native question never started')
                    self.assertTrue(entered['b'].wait(5), 'workers=1 capped dynamic initial=2')
                    self.assertFalse(entered['c'].is_set(), 'a third question bypassed initial=2')
                    release['b'].set()
                    self.assertTrue(entered['c'].wait(5), 'pending question did not fill b\'s freed slot')
                    self.assertFalse(release['a'].is_set())
                finally:
                    for event in release.values():
                        event.set()
                result = future.result(timeout=5)
            self.assertEqual((result['succeeded'], result['failed']), (3, 0))
            slots = result['admission']['pipeline']
            self.assertEqual((slots['peak_active'], slots['current_limit'], slots['max_limit']), (2, 2, 3))
            self.assertEqual((slots['active'], slots['pending'], slots['model']['transport_in_flight']), (0, 0, 0))
            self.assertEqual(slots['model']['completed'], 9)
            self.assertEqual(len([event for event in store.events() if event['kind'] == 'api_request']), 9)
            self.assertTrue(all(row['status'] == 'succeeded' for row in store.attempts()))
            self.assertTrue(closed.is_set())
            self.assertTrue(store.verify()['ok'])

    def test_stability_and_errors_resize_active_questions_and_persist_adjustments(self):
        # Break caught: transport observations fail to reach the actual slots,
        # growth cannot start a third question, or reduction interrupts active
        # questions / admits a pending one before occupancy falls below its cap.
        clock = Clock()
        entered = {key: threading.Event() for key in 'abcd'}
        release = {key: threading.Event() for key in 'abc'}
        shrunk, c_released = threading.Event(), threading.Event()

        def create(*, fail=False):
            if fail:
                raise TimeoutError('offline transient transport failure')

        def execute(item, request):
            key = item.instance_id
            entered[key].set()
            if key == 'a':
                if not entered['b'].wait(5):
                    raise AssertionError('initial=2 never admitted b')
                for number in range(50):
                    if number == 49:
                        clock.now = 1.0
                    request()
            elif key == 'c':
                clock.now = 61.0
                for _ in range(5):
                    try:
                        request(fail=True)
                    except TimeoutError:
                        pass
                shrunk.set()
            if key in release and not release[key].wait(5):
                raise AssertionError(f'question {key} did not receive release')
            if key in 'bd':
                request()

        with tempfile.TemporaryDirectory() as temporary, \
             self.prepared(temporary, calls=1, keys='abcd') as store, \
             self.controlled_slots(clock) as captured, \
             self.fake_revision(execute, create=create) as closed, \
             ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(cli.execute_run, store, ENV)
            try:
                self.assertTrue(shrunk.wait(5), 'stable demand never admitted c, or failures did not complete')
                slots = captured['pipeline']
                state = slots.snapshot()
                self.assertEqual((state['current_limit'], state['active'], state['pending']), (2, 3, 1))
                self.assertFalse(entered['d'].is_set())
                original_release = slots.release

                def completed(ticket, *, status):
                    original_release(ticket, status=status)
                    if ticket['item_key'] == 'lite/c':
                        c_released.set()

                slots.release = completed
                release['c'].set()
                self.assertTrue(c_released.wait(5), 'c did not release its pipeline ticket')
                self.assertEqual(slots.snapshot()['active'], 2)
                self.assertFalse(entered['d'].is_set(), 'downshift admitted d while two questions remain active')
                release['b'].set()
                self.assertTrue(entered['d'].wait(5), 'pending d was not admitted once active fell below two')
                self.assertFalse(release['a'].is_set())
            finally:
                for event in release.values():
                    event.set()
            result = future.result(timeout=5)
            self.assertEqual((result['succeeded'], result['failed']), (4, 0))
            state = result['admission']['pipeline']
            self.assertEqual((state['peak_active'], state['completed'], state['active'], state['pending']), (3, 4, 0, 0))
            self.assertEqual((state['model']['completed'], state['model']['transient_errors'],
                              state['model']['transport_in_flight']), (57, 5, 0))
            changes = [event for event in store.events() if event['kind'] == 'pipeline_concurrency_adjustment']
            self.assertEqual([(event['payload']['old_limit'], event['payload']['new_limit'],
                               event['payload']['reason']) for event in changes],
                             [(2, 3, 'stable_demand'), (3, 2, 'transient_failures')])
            attempt_ids = {row['attempt_id'] for row in store.attempts()}
            request_ids = {event['payload']['call_id'] for event in store.events() if event['kind'] == 'api_request'}
            self.assertTrue(all(event['attempt_id'] in attempt_ids and
                                event['payload']['call_id'] in request_ids for event in changes))
            self.assertTrue(all(row['status'] == 'succeeded' for row in store.attempts()))
            self.assertTrue(closed.is_set())
            self.assertTrue(store.verify()['ok'])

    def test_fatal_question_drains_inflight_work_and_releases_every_ticket(self):
        # Break caught: fatal unwinding leaks admitted tickets or resets/closes
        # native resources while a different admitted question still uses them.
        class FatalQuestion(BaseException):
            pass

        from scripts.rc_evaluation.deepeye import runner

        entered = {key: threading.Event() for key in 'abc'}
        release_a, release_b = threading.Event(), threading.Event()
        draining, b_finished, cleaned = threading.Event(), threading.Event(), threading.Event()
        original_close = runner.close_runners

        def drain(*args, **kwargs):
            draining.set()
            return original_close(*args, **kwargs)

        def execute(item, request):
            key = item.instance_id
            entered[key].set()
            if key == 'a':
                if not release_a.wait(5):
                    raise AssertionError('fatal question was not released')
                raise FatalQuestion('offline fatal question')
            if key == 'b':
                if not release_b.wait(5):
                    raise AssertionError('inflight question was not released')
                request()
                b_finished.set()

        def cleanup():
            self.assertTrue(b_finished.is_set(), 'native cleanup ran before b drained')
            cleaned.set()

        with tempfile.TemporaryDirectory() as temporary, \
             self.prepared(temporary, calls=1, keys='abc') as store, \
             self.controlled_slots(Clock()) as captured, \
             self.fake_revision(execute, cleanup=cleanup) as closed, \
             patch.object(runner, 'close_runners', new=drain), \
             ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(cli.execute_run, store, ENV)
            try:
                self.assertTrue(entered['a'].wait(5))
                self.assertTrue(entered['b'].wait(5))
                release_a.set()
                self.assertTrue(draining.wait(5), 'fatal question never reached scheduler unwinding')
                self.assertFalse(cleaned.is_set())
                self.assertFalse(closed.is_set())
                self.assertFalse(future.done(), 'fatal run returned while b still owns native work')
            finally:
                release_a.set()
                release_b.set()
            with self.assertRaisesRegex(FatalQuestion, 'offline fatal question'):
                future.result(timeout=5)
            self.assertFalse(entered['c'].is_set())
            slots = captured['pipeline'].snapshot()
            self.assertEqual((slots['active'], slots['pending'], slots['completed']), (0, 0, 2))
            self.assertEqual(slots['model']['transport_in_flight'], 0)
            self.assertTrue(cleaned.is_set())
            self.assertTrue(closed.is_set())
            self.assertEqual({row['item_key'] for row in store.attempts()}, {'lite/a', 'lite/b'})


if __name__ == '__main__':
    unittest.main()
