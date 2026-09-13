"""Durable sampling acceptance; fake transport, real retry engine and SQLite."""
from contextlib import nullcontext
from functools import partial
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tests.test_deepeye_sampling import llm_fixture, response, parse
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
from app.llm_extractor import LLMExtractor


class PowerLoss(BaseException):
    pass


class SuffixRule:
    __slots__ = ('suffix',)

    def __init__(self, suffix):
        self.suffix = suffix

    def __call__(self, content):
        return content + self.suffix


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RunStore.create(Path(self.temp.name) / 'run', {'condition': 'rc', 'contract': 'one'})
        self.addCleanup(self.store.close)

    def attempt(self, llm, *, n=5, messages=None, stop=None, recorder=None):
        recorder = recorder or TraceRecorder(self.store)
        if stop is not None:
            recorder.stop_event = stop
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        with recorder.context(attempt):
            result = LLMExtractor().extract_with_retry(llm, messages or [], parse, n=n)
        return result, attempt

    def test_three_committed_samples_survive_loss_and_only_two_are_requested(self):
        llm, calls = llm_fixture([response().model_copy(update={'id': f'r{i}'}) for i in range(3)])
        original = llm._client.chat.completions.create
        def request(**kwargs):
            if len(calls) == 3:
                raise PowerLoss()
            return original(**kwargs)
        llm._client.chat.completions.create = request
        with self.assertRaises(PowerLoss):
            self.attempt(llm)
        before = list(self.store.iter_events(kinds='sample_checkpoint'))
        self.assertEqual(len(before), 3)
        path = self.store.run_dir
        self.store.close()
        self.store = RunStore.open(path)
        self.addCleanup(self.store.close)
        llm, calls = llm_fixture([response(), response()])
        (values, usage), attempt = self.attempt(llm)
        self.assertEqual((len(values), len(calls), usage['total_tokens']), (5, 2, 150))
        restored = list(self.store.iter_events(attempt, kinds='sample_result'))[:3]
        self.assertEqual([x['payload']['response_id'] for x in restored], ['r0', 'r1', 'r2'])
        self.assertEqual([x['payload']['restored_from_event'] for x in restored], [x['event_id'] for x in before])
        self.assertTrue(self.store.verify()['ok'])

    def test_commit_before_group_finish_restores_without_new_requests(self):
        llm, _ = llm_fixture([response(tokens=None)])
        recorder = TraceRecorder(self.store)
        original = recorder.record_sampling
        def observe(kind, payload):
            if kind == 'sampling_group_result':
                raise PowerLoss()
            original(kind, payload)
        with patch.object(recorder, 'record_sampling', side_effect=observe), self.assertRaises(PowerLoss):
            self.attempt(llm, n=1, recorder=recorder)
        llm, calls = llm_fixture([])
        (values, usage), attempt = self.attempt(llm, n=1)
        self.assertEqual((len(values), len(calls)), (1, 0))
        from scripts.baseline_adapters.deepeye.run_usage import _effective_sampling
        effective = _effective_sampling(self.store.iter_events(attempt))
        self.assertFalse(effective['usage_complete'])

    def test_uncertain_attempts_consume_budget_and_never_reset_on_resume(self):
        for _ in range(4):
            llm, _ = llm_fixture([])
            llm._client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(PowerLoss())
            with self.assertRaises(PowerLoss):
                self.attempt(llm, n=1)
        llm, calls = llm_fixture([response()])
        with patch('app.llm_extractor.extractor.logger.warning') as warning:
            (values, _), _ = self.attempt(llm, n=1)
        self.assertEqual((values, calls), ([], []))
        warning.assert_called_once()
        self.assertEqual(len(list(self.store.iter_events(kinds='sample_attempt_started'))), 4)
        self.assertGreaterEqual(len(list(self.store.iter_events(kinds='sample_attempt_uncertain'))), 4)

    def test_same_prompt_independent_invocations_are_not_collapsed(self):
        llm, calls = llm_fixture([response(), response()])
        recorder = TraceRecorder(self.store)
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        with recorder.context(attempt):
            for _ in range(2):
                LLMExtractor().extract_with_retry(llm, [], parse, n=1)
        groups = list(self.store.iter_events(attempt, kinds='sampling_group_start'))
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(groups[0]['payload']['group_id'], groups[1]['payload']['group_id'])

    def test_restored_usage_counts_once_across_attempts_and_no_new_api_request(self):
        from scripts.baseline_adapters.deepeye.run_usage import observed_usage
        llm, _ = llm_fixture([response()])
        self.attempt(llm, n=1)
        llm, calls = llm_fixture([])
        _, attempt = self.attempt(llm, n=1)
        usage = observed_usage(self.store)
        self.assertEqual(usage['effective_sampling']['known_tokens']['total_tokens'], 30)
        self.assertEqual(calls, [])
        self.assertEqual(list(self.store.iter_events(attempt, kinds='api_request')), [])

    def test_duplicate_valid_restore_within_one_attempt_is_rejected(self):
        llm, _ = llm_fixture([response()])
        self.attempt(llm, n=1)
        llm, _ = llm_fixture([])
        _, attempt = self.attempt(llm, n=1)
        restored = next(self.store.iter_events(attempt, kinds='sample_result'))
        recorder = TraceRecorder(self.store)
        before = len(self.store.events(attempt))
        with recorder.context(attempt), self.assertRaisesRegex(ValueError, 'duplicate sample result'):
            recorder.record_sampling('sample_result', restored['payload'])
        self.assertEqual(len(self.store.events(attempt)), before)
        self.store.append_event(attempt, 'sample_result', restored['payload'])
        with self.assertRaisesRegex(ValueError, 'duplicate sample result'):
            TraceRecorder(self.store)

    def test_partial_configuration_changes_cannot_restore_old_parsed_result(self):
        def rule(content, *, suffix):
            return content + suffix
        for suffix, expected_calls in ((' first', 1), (' changed', 1), (' changed', 0)):
            llm, calls = llm_fixture([response()])
            recorder = TraceRecorder(self.store)
            attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
            with recorder.context(attempt):
                values, _ = LLMExtractor().extract_with_retry(llm, [], partial(rule, suffix=suffix), n=1)
            self.assertEqual(values, ['SELECT wrong_but_parseable' + suffix])
            self.assertEqual(len(calls), expected_calls)

    def test_callable_with_mutable_class_attribute_is_rejected_before_transport(self):
        from app.llm.sampling import SamplingIdentityError
        class Rule:
            suffix = ' first'
            def __call__(self, content):
                return content + self.suffix
        rule = Rule()
        for suffix in (' first', ' changed'):
            with self.subTest(suffix=suffix):
                Rule.suffix = suffix
                llm, calls = llm_fixture([response()])
                recorder = TraceRecorder(self.store)
                attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
                with recorder.context(attempt), self.assertRaisesRegex(SamplingIdentityError, 'unsupported parser callable'):
                    LLMExtractor().extract_with_retry(llm, [], rule, n=1)
                self.assertEqual(calls, [])

    def test_native_fallback_cannot_turn_invalid_parser_identity_into_success(self):
        from app.llm.sampling import SamplingIdentityError
        from tests.test_deepeye_run_pipeline import Factory, item
        from scripts.baseline_adapters.deepeye.run_pipeline import run_pipeline
        base = Factory()
        llm, calls = llm_fixture([response()])
        def factory(stage, items):
            runner = base(stage, items)
            if stage == 'schema_linking':
                original = runner._link_tables_and_columns
                def process(target):
                    try:
                        LLMExtractor().extract_with_retry(llm, [], SuffixRule(' unsupported'), n=1)
                    except Exception:
                        original(target)  # Native optional-failure fallback.
                runner._link_tables_and_columns = process
            return runner
        recorder = TraceRecorder(self.store)
        with patch.object(recorder, 'instrument_runner', return_value=lambda: None):
            with self.assertRaises(SamplingIdentityError):
                run_pipeline(self.store, [('lite', item())], factory, recorder, workers=1)
        self.assertEqual(calls, [])
        self.assertFalse(any(row['status'] == 'succeeded' for row in self.store.attempts()))

    def test_duplicate_success_and_bad_restore_reference_fail_closed(self):
        llm, _ = llm_fixture([response()])
        _, attempt = self.attempt(llm, n=1)
        checkpoint = next(self.store.iter_events(kinds='sample_checkpoint'))
        self.store.append_event(attempt, 'sample_checkpoint', checkpoint['payload'])
        with self.assertRaises(ValueError):
            TraceRecorder(self.store)

    def test_forged_restore_reference_is_rejected(self):
        llm, _ = llm_fixture([response()])
        _, attempt = self.attempt(llm, n=1)
        event = next(self.store.iter_events(attempt, kinds='sample_result'))
        self.store.append_event(attempt, 'sample_result', {**event['payload'], 'restored_from_event': 999})
        with self.assertRaises(ValueError):
            TraceRecorder(self.store)

    def test_changed_source_rejects_completed_native_stage_before_constructors(self):
        from tests.test_deepeye_run_pipeline import Factory, item
        from scripts.baseline_adapters.deepeye.run_pipeline import run_pipeline
        recorder = TraceRecorder(self.store)
        with patch.object(recorder, 'instrument_runner', return_value=lambda: None):
            run_pipeline(self.store, [('lite', item())], Factory(), recorder, workers=1)
        recorder = TraceRecorder(self.store)
        recorder.sampling_checkpoints.source_version = 'different implementation'
        with self.assertRaises(ValueError):
            run_pipeline(self.store, [('lite', item())], lambda *a: self.fail('constructed'), recorder, workers=1)

    def test_rc_actual_runner_pauses_then_reuses_success_without_losing_participation(self):
        from scripts.rc_evaluation.deepeye.tests.test_runner import Factory, experiment_manifest
        from scripts.rc_evaluation.deepeye.tests.test_source import make_source
        from scripts.rc_evaluation.deepeye.source import snapshot_source
        from scripts.rc_evaluation.deepeye.contracts import render_rc_block
        from scripts.rc_evaluation.deepeye.runner import run_experiment
        stop = threading.Event()
        llm, calls = llm_fixture([response()] * 3)
        original = llm._client.chat.completions.create
        def request(**kw):
            result = original(**kw)
            if len(calls) == 3:
                stop.set()
            return result
        llm._client.chat.completions.create = request
        base = Factory()
        def factory(stage, items):
            runner = base(stage, items)
            original_stage = runner._revise_sql
            def revise(target):
                LLMExtractor().extract_with_retry(llm, [{'role': 'user', 'content': block}], parse, n=3)
                original_stage(target)
            runner._revise_sql = revise
            return runner
        source_path = Path(self.temp.name) / 'source'
        tasks = make_source(source_path, calls=1)
        data = experiment_manifest(snapshot_source(source_path, tasks, 'sql_revision'), condition='rc')
        contract = {'task_key': 'lite/a', 'db_id': 'db', 'question': 'Return x', 'evidence': 'Ascending order',
                    'round2': {key: 'Meaning' for key in ('population', 'row_grain', 'column_role',
                                                        'derivation', 'filter_policy', 'meta_review')}}
        data['contracts'] = {'lite/a': contract}
        block = render_rc_block(contract)
        with RunStore.create(Path(self.temp.name) / 'rc', data) as store:
            recorder = TraceRecorder(store, stop_event=stop)
            with patch.object(recorder, 'instrument_runner', return_value=lambda: None):
                outcome = run_experiment(store, factory, recorder, workers=1)
            self.assertEqual(outcome.get('paused'), 1)
            self.assertFalse(any(a['status'] == 'succeeded' for a in store.attempts()))
            self.assertEqual(base.seen, [])
            stop.clear()
            llm, calls = llm_fixture([])
            recorder = TraceRecorder(store, stop_event=stop)
            with patch.object(recorder, 'instrument_runner', return_value=lambda: None):
                outcome = run_experiment(store, factory, recorder, workers=1)
            self.assertEqual((outcome['succeeded'], len(calls)), (1, 0))
            row = store.attempts()[-1]
            self.assertEqual(row['payload']['rc_participation']['restored_sample_count'], 3)
            self.assertEqual(row['payload']['rc_participation']['status'], 'participating')
            self.assertEqual(row['payload']['rc_participation']['actual_request_count'], 0)

    def test_same_code_parser_with_changed_closed_over_rule_is_not_reused(self):
        def rule(prefix):
            return lambda content: content if content.startswith(prefix) else None
        for prefix in ('SELECT', 'S'):
            llm, calls = llm_fixture([response()])
            recorder = TraceRecorder(self.store)
            attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
            with recorder.context(attempt):
                LLMExtractor().extract_with_retry(llm, [], rule(prefix), n=1)
            self.assertEqual(len(calls), 1)

    def test_response_before_durable_commit_remains_uncertain_and_spends_attempt(self):
        from scripts.baseline_adapters.deepeye.run_sampling import SamplingSession
        llm, calls = llm_fixture([response()])
        with patch.object(SamplingSession, 'finish_sample_attempt', side_effect=PowerLoss), self.assertRaises(PowerLoss):
            self.attempt(llm, n=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.store.iter_events(kinds='sample_checkpoint')), [])
        llm, calls = llm_fixture([response()])
        _, attempt = self.attempt(llm, n=1)
        result = next(self.store.iter_events(attempt, kinds='sample_result'))['payload']
        self.assertEqual((len(calls), result['attempt_count']), (1, 2))
        self.assertEqual(len(list(self.store.iter_events(kinds='sample_attempt_uncertain'))), 1)

    def test_interrupt_after_start_before_transport_conservatively_consumes_attempt(self):
        from scripts.baseline_adapters.deepeye.run_sampling import SamplingSession
        original = SamplingSession.start_attempt
        def started(session, identity):
            original(session, identity)
            raise PowerLoss()
        llm, calls = llm_fixture([response()])
        with patch.object(SamplingSession, 'start_attempt', new=started), self.assertRaises(PowerLoss):
            self.attempt(llm, n=1)
        self.assertEqual(calls, [])
        llm, _ = llm_fixture([response()])
        _, attempt = self.attempt(llm, n=1)
        self.assertEqual(next(self.store.iter_events(attempt, kinds='sample_result'))['payload']['attempt_count'], 2)

    def test_parallel_native_nodes_drain_on_stop_and_recover_same_submissions(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from app.llm.sampling import SamplingPaused
        from scripts.baseline_adapters.deepeye.run_resources import instrument_native_pools, native_stage_work
        stop, release, entered = threading.Event(), threading.Event(), threading.Event()
        llm, calls = llm_fixture([response(), response()])
        recorder = TraceRecorder(self.store, stop_event=stop)
        runner = SimpleNamespace(_inner_thread_pool_executor=ThreadPoolExecutor(2))
        self.addCleanup(runner._inner_thread_pool_executor.shutdown)
        undo = instrument_native_pools(runner)
        self.addCleanup(undo)
        original = llm._client.chat.completions.create
        transport_lock = threading.Lock()
        started = []
        def request(**kw):
            with transport_lock:
                started.append(1)
                second = len(started) == 2
            if second:
                stop.set()
                release.set()
            else:
                entered.set()
                if not release.wait(3):
                    raise AssertionError('second request never started')
            with transport_lock:
                return original(**kw)
        llm._client.chat.completions.create = request
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        with recorder.context(attempt), native_stage_work():
            a = runner._inner_thread_pool_executor.submit(LLMExtractor().extract_with_retry, llm, [], parse, n=1)
            self.assertTrue(entered.wait(3))
            b = runner._inner_thread_pool_executor.submit(LLMExtractor().extract_with_retry, llm, [], parse, n=1)
            for job in (a, b):
                with self.assertRaises(SamplingPaused):
                    job.result()
            with self.assertRaises(SamplingPaused):
                runner._inner_thread_pool_executor.submit(lambda: self.fail('new node'))
        self.assertEqual(len(list(self.store.iter_events(kinds='sample_checkpoint'))), 2)
        stop.clear()
        llm, calls = llm_fixture([])
        recorder = TraceRecorder(self.store, stop_event=stop)
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        with recorder.context(attempt), native_stage_work():
            jobs = [runner._inner_thread_pool_executor.submit(LLMExtractor().extract_with_retry, llm, [], parse, n=1) for _ in range(2)]
            self.assertEqual([len(job.result()[0]) for job in jobs], [1, 1])
        self.assertEqual(calls, [])
        self.assertTrue(self.store.verify()['ok'])

    def test_restored_direct_ask_retains_message_type(self):
        llm, _ = llm_fixture([response()])
        recorder = TraceRecorder(self.store)
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        with recorder.context(attempt):
            original, _ = llm.ask([], n=1)
        llm, calls = llm_fixture([])
        recorder = TraceRecorder(self.store)
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        with recorder.context(attempt):
            restored, _ = llm.ask([], n=1)
        self.assertEqual(type(restored[0]), type(original[0]))
        self.assertEqual(restored[0].content, 'SELECT wrong_but_parseable')
        self.assertEqual(calls, [])

    def test_recovered_source_is_model_participating_for_replay_even_with_zero_new_calls(self):
        from scripts.rc_evaluation.deepeye.source import _source_trace
        from scripts.rc_evaluation.deepeye.runner import _replay
        llm, _ = llm_fixture([response()])
        self.attempt(llm, n=1)
        llm, _ = llm_fixture([])
        (_, usage), attempt = self.attempt(llm, n=1)
        self.store.finish_attempt(attempt, 'succeeded', {'artifact': {'schema_linking_llm_cost': usage}})
        trace = _source_trace(self.store, self.store.attempt(attempt), 'schema_linking')
        self.assertEqual((trace['requests'], trace['restored_samples']), (0, 1))
        self.assertFalse(_replay('schema_linking', 'schema_linking', True, {'api_trace': trace}))

    def test_explicit_renewal_only_retries_exhausted_sample_and_retains_history(self):
        llm, _ = llm_fixture([response()] + [response('bad')] * 4)
        with patch('app.llm_extractor.extractor.logger.warning') as warning:
            (_, _), first = self.attempt(llm, n=2)
        warning.assert_called_once()
        self.store.finish_attempt(first, 'failed', {'reason': 'exhausted'})
        prior = self.store.events(first)
        llm, calls = llm_fixture([])
        with patch('app.llm_extractor.extractor.logger.warning') as warning:
            (values, _), _ = self.attempt(llm, n=2)
        warning.assert_called_once()
        self.assertEqual((len(values), len(calls)), (1, 0))
        recorder = TraceRecorder(self.store)
        group_id = next(self.store.iter_events(first, kinds='sampling_group_start'))['payload']['group_id']
        decision_attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        recorder.sampling_checkpoints.renew_exhausted(group_id, attempt_id=decision_attempt,
                                                     decision='Explicit offline retry round authorized')
        llm, calls = llm_fixture([response()])
        with recorder.context(decision_attempt):
            values, usage = LLMExtractor().extract_with_retry(llm, [], parse, n=2)
        self.assertEqual((len(values), len(calls), usage['total_tokens']), (2, 1, 60))
        self.assertEqual(self.store.events(first), prior)
        renewal = next(self.store.iter_events(kinds='sampling_retry_authorized'))['payload']
        self.assertEqual(renewal['allowances'], [{'sample_index': 1, 'prior_limit': 4, 'attempt_limit': 8}])
        self.assertTrue(self.store.verify()['ok'])

    def test_fatal_renewal_grants_four_new_attempts_not_unused_old_allowance(self):
        import httpx
        from openai import AuthenticationError, APIConnectionError
        request = httpx.Request('POST', 'https://invalid.test')
        fatal = AuthenticationError('fixture', response=httpx.Response(401, request=request), body=None)
        transient = APIConnectionError(request=request)
        llm, _ = llm_fixture([fatal])
        with patch('app.llm_extractor.extractor.logger.warning'):
            _, first = self.attempt(llm, n=1)
        recorder = TraceRecorder(self.store)
        group = next(self.store.iter_events(first, kinds='sampling_group_start'))['payload']['group_id']
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        recorder.sampling_checkpoints.renew_exhausted(group, attempt_id=attempt, decision='Explicit new round')
        llm, calls = llm_fixture([transient] * 7)
        with recorder.context(attempt), patch('app.llm_extractor.extractor.logger.warning') as warning:
            values, _ = LLMExtractor().extract_with_retry(llm, [], parse, n=1)
        warning.assert_called_once()
        self.assertEqual((values, len(calls)), ([], 4))
        renewal = next(self.store.iter_events(kinds='sampling_retry_authorized'))['payload']
        self.assertEqual(renewal['allowances'], [{'sample_index': 0, 'prior_limit': 4, 'attempt_limit': 5}])

    def test_closed_session_late_response_cannot_overwrite_recovered_success(self):
        from app.llm.sampling import SampleOutcome, SampleAttempt
        recorder = TraceRecorder(self.store)
        attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        session = recorder.sampling_checkpoints.session(attempt, recorder)
        llm, _ = llm_fixture([])
        identity = llm.sampling_request_identity([], parser={'kind': 'message'})
        group = session.group(identity, 1, 4)
        started = {'group_id': group, 'sample_index': 0, 'sample_attempt': 1}
        session.start_attempt(started)
        session.close()
        row_count = len(self.store.events())
        late = SampleOutcome(group, 0, result='late', succeeded=True,
                             attempts=[SampleAttempt(1, 'succeeded', None)])
        with self.assertRaisesRegex(ValueError, 'late'):
            session.finish_sample_attempt(started, late)
        self.assertEqual(len(self.store.events()), row_count)
        next_attempt = self.store.begin_attempt('lite/one', 'schema_linking', 'input')
        next_session = recorder.sampling_checkpoints.session(next_attempt, recorder)
        self.addCleanup(next_session.close)
        self.assertEqual(next_session.group(identity, 1, 4), group)
        restored = next_session.restore(group, 0)
        self.assertEqual(restored.attempts[0].status, 'uncertain')
        self.assertFalse(restored.succeeded)

    def test_changed_actual_prompt_model_or_source_never_reuses(self):
        llm, _ = llm_fixture([response()])
        self.attempt(llm, n=1)
        for change in ('prompt', 'model', 'source'):
            llm, calls = llm_fixture([response()])
            messages = [{'role': 'user', 'content': 'changed'}] if change == 'prompt' else []
            if change == 'model':
                llm._config.temperature = 0.9
            recorder = TraceRecorder(self.store)
            if change == 'source':
                recorder.sampling_checkpoints.source_version = 'changed'
            self.attempt(llm, n=1, messages=messages, recorder=recorder)
            self.assertEqual(len(calls), 1, change)

    def test_actual_pipeline_pauses_mid_group_and_resumes(self):
        from tests.test_deepeye_run_pipeline import Factory, item
        from scripts.baseline_adapters.deepeye.run_pipeline import run_pipeline
        stop = threading.Event()
        llm, calls = llm_fixture([response()] * 5)
        original = llm._client.chat.completions.create
        def request(**kw):
            result = original(**kw)
            if len(calls) == 3:
                stop.set()
            return result
        llm._client.chat.completions.create = request
        base = Factory()
        def factory(stage, items):
            runner = base(stage, items)
            if stage == 'schema_linking':
                original_stage = runner._link_tables_and_columns
                def process(target):
                    LLMExtractor().extract_with_retry(llm, [], parse, n=5)
                    original_stage(target)
                runner._link_tables_and_columns = process
            return runner
        recorder = TraceRecorder(self.store)
        recorder.stop_event = stop
        with patch.object(recorder, 'instrument_runner', return_value=lambda: None):
            outcome = run_pipeline(self.store, [('lite', item())], factory, recorder, workers=1)
        self.assertEqual((outcome.get('paused'), outcome['succeeded'], len(calls)), (1, 0, 3))
        self.assertFalse(any(a['status'] == 'succeeded' for a in self.store.attempts()))
        self.assertEqual(base.calls, [])
        stop.clear()
        llm, calls = llm_fixture([response()] * 2)
        recorder = TraceRecorder(self.store, stop_event=stop)
        with patch.object(recorder, 'instrument_runner', return_value=lambda: None):
            outcome = run_pipeline(self.store, [('lite', item())], factory, recorder, workers=1)
        self.assertEqual((outcome['succeeded'], len(calls)), (1, 2))
