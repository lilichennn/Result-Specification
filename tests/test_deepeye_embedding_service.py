"""Offline shared embedding budget, cache, retry and lifecycle regressions."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import threading
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))

from scripts.baseline_adapters.deepeye.precompute_cache import VectorCache

try:
    service_module = importlib.import_module('scripts.baseline_adapters.deepeye.embedding_service')
except ModuleNotFoundError as error:
    if error.name != 'scripts.baseline_adapters.deepeye.embedding_service':
        raise
    service_module = None


ENV = {'EMBEDDING_MODEL': 'offline', 'EMBEDDING_BASE_URL': 'https://invalid.test/v1',
       'EMBEDDING_API_KEY': 'offline-secret'}


def response(texts, *, usage=True):
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(ord(text[0])), 1.])
                                 for i, text in reversed(list(enumerate(texts)))],
                           usage=SimpleNamespace(prompt_tokens=len(texts), total_tokens=len(texts)) if usage else None)


class FakeClient:
    def __init__(self, create):
        self.embeddings = SimpleNamespace(create=create)
        self.max_retries = 99
        self.closed = False

    def close(self):
        self.closed = True


class EndpointError(Exception):
    def __init__(self, status, message, retry_after=None):
        super().__init__(message)
        self.status_code = status
        self.body = {'message': message}
        self.response = SimpleNamespace(headers={} if retry_after is None else {'retry-after': str(retry_after)})


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(service_module, 'Shared embedding service is missing')
        self.addCleanup(lambda: self.assertFalse(
            any(thread.name.startswith('deepeye-embedding-') for thread in threading.enumerate()),
            'Embedding service left worker/admission threads alive'))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / 'embedding.jsonl'
        self.namespace = service_module.embedding_namespace(ENV, dimension=2)
        self.cache = VectorCache(self.root / 'vectors.sqlite', self.namespace)
        self.addCleanup(self.cache.close)
        self.limits = service_module.EmbeddingLimits(dimension=2, request_limit=2, request_workers=2,
            http_connections=2, start_rate=10000, input_tokens_per_second=100000,
            input_tokens_per_minute=6000000, retry_delay=0)

    def service(self, create, **limits):
        client = FakeClient(create)
        service = service_module.EmbeddingService(ENV, self.cache, self.log,
            limits=replace(self.limits, **limits), client=client)
        self.addCleanup(service.close)
        return service, client

    def rows(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def request_rows(self):
        return [row for row in self.rows() if row['kind'] == 'embedding_request']

    def test_overlapping_calls_deduplicate_pending_text_and_preserve_order(self):
        release, two_batches = threading.Event(), threading.Event()
        lock, calls = threading.Lock(), []
        def create(**kwargs):
            with lock:
                calls.append(kwargs['input'])
                if len(calls) == 2:
                    two_batches.set()
            self.assertTrue(release.wait(3))
            return response(kwargs['input'])
        service, client = self.service(create)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(service.embed, ['a', 'b', 'a'], purpose='values')
            second = callers.submit(service.embed, ['b', 'c'], purpose='query')
            try:
                self.assertTrue(two_batches.wait(3))
            finally:
                release.set()
            np.testing.assert_equal(first.result(3), [[97., 1.], [98., 1.], [97., 1.]])
            np.testing.assert_equal(second.result(3), [[98., 1.], [99., 1.]])
        self.assertEqual(Counter(t for batch in calls for t in batch), Counter('abc'))
        self.assertEqual(client.max_retries, 0)
        self.assertEqual(service(['b']).dtype, np.dtype('float32'))
        self.assertEqual(len(calls), 2)

    def test_reopening_cache_emits_no_requests_with_different_batch_and_concurrency(self):
        service, _ = self.service(lambda **kwargs: response(kwargs['input']))
        service.embed(['a', 'b', 'a'])
        service.close()
        with VectorCache(self.root / 'vectors.sqlite', self.namespace) as reopened:
            client = FakeClient(lambda **_: self.fail('Reopened vector was embedded again'))
            with service_module.EmbeddingService(ENV, reopened, self.root / 'reopened.jsonl',
                    limits=replace(self.limits, batch_size=1, request_limit=1), client=client) as second:
                np.testing.assert_equal(second(['b', 'a']), [[98., 1.], [97., 1.]])
        self.assertTrue(client.closed)
        rows = [json.loads(line) for line in (self.root / 'reopened.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        timestamp = rows[0].pop('timestamp')
        self.assertIsInstance(timestamp, str)
        self.assertEqual(rows, [{
            'kind': 'embedding_cache', 'purpose': 'embedding',
            'requested_texts': 2, 'unique_texts': 2,
            'cache_hits': 2, 'pending_hits': 0, 'cache_misses': 0,
        }])

    def test_failed_batch_retries_without_reembedding_success_and_uses_four_attempts(self):
        calls = Counter()
        def create(**kwargs):
            text, = kwargs['input']
            calls[text] += 1
            if text == 'b':
                raise TimeoutError('offline timeout')
            return response(kwargs['input'])
        service, _ = self.service(create, batch_size=1)
        with self.assertRaises(TimeoutError):
            service(['a', 'b'])
        self.assertEqual(calls, {'a': 1, 'b': 4})
        self.assertIsNotNone(self.cache.get('a'))
        self.assertIsNone(self.cache.get('b'))
        service.close()
        retried, _ = self.service(lambda **kwargs: response(kwargs['input']), batch_size=1)
        np.testing.assert_equal(retried(['a', 'b']), [[97., 1.], [98., 1.]])
        self.assertEqual(self.cache.count(), 2)
        self.assertEqual(len(self.request_rows()), 6)

    def test_retry_backoff_releases_capacity_for_other_batches(self):
        calls, seen_retry = [], threading.Event()
        def create(**kwargs):
            text, = kwargs['input']
            calls.append(text)
            if text == 'a' and calls.count('a') == 1:
                seen_retry.set()
                raise EndpointError(429, 'requests per second exceeded', .15)
            return response(kwargs['input'])
        service, _ = self.service(create, request_limit=1, batch_size=1)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(service.embed, ['a'])
            self.assertTrue(seen_retry.wait(2))
            second = callers.submit(service.embed, ['b'])
            second.result(3)
            first.result(3)
        self.assertEqual(calls, ['a', 'b', 'a'])
        self.assertEqual([r['attempt'] for r in self.request_rows() if r['purpose'] == 'embedding'], [1, 1, 2])

    def test_map_is_ordered_and_bounds_workers_without_nested_pool_deadlock(self):
        entered, release = threading.Event(), threading.Event()
        calls, active, peak = [], 0, 0
        lock = threading.Lock()
        def create(**kwargs):
            nonlocal active, peak
            with lock:
                calls.append(kwargs['input'])
                active += 1
                peak = max(peak, active)
                if active == 2:
                    entered.set()
            try:
                self.assertTrue(release.wait(3))
                return response(kwargs['input'])
            finally:
                with lock:
                    active -= 1
        service, _ = self.service(create, batch_size=1)
        with ThreadPoolExecutor(1) as caller:
            future = caller.submit(lambda: list(service.map(service, [[s] for s in 'abcdef'])))
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(len(calls), 2)
            finally:
                release.set()
            result = future.result(5)
        np.testing.assert_equal(np.concatenate(result), [[float(ord(s)), 1.] for s in 'abcdef'])
        self.assertEqual(peak, 2)

    def test_close_cancels_queued_batches_drains_active_then_closes_client(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def create(**kwargs):
            calls.append(kwargs['input'])
            entered.set()
            self.assertTrue(release.wait(3))
            return response(kwargs['input'])
        service, client = self.service(create, batch_size=1, request_limit=1)
        with ThreadPoolExecutor(2) as callers:
            future = callers.submit(service.embed, ['a', 'b', 'c'])
            self.assertTrue(entered.wait(2))
            close = callers.submit(service.close)
            self.assertFalse(client.closed)
            release.set()
            close.result(3)
            with self.assertRaises(service_module.EmbeddingClosedError):
                future.result(3)
        self.assertEqual(calls, [['a']])
        self.assertIsNotNone(self.cache.get('a'))
        self.assertTrue(client.closed)
        with self.assertRaises(service_module.EmbeddingClosedError):
            service(['z'])

    def test_invalid_responses_never_enter_cache(self):
        bad = [
            SimpleNamespace(data=[], usage=None),
            SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1., 2.])], usage=None),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0., 0.])], usage=None),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[float('nan'), 2.])], usage=None),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1., 2., 3.])], usage=None),
        ]
        for index, value in enumerate(bad):
            with self.subTest(index=index):
                service, _ = self.service(lambda **_: value)
                with self.assertRaises(ValueError):
                    service([str(index)])
                self.assertIsNone(self.cache.get(str(index)))
                self.assertEqual(self.rows()[-1]['http_status'], 200)
                service.close()

    def test_batch_limit_long_input_and_unknown_usage_are_explicit(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return response(kwargs['input'], usage=False)
        service, _ = self.service(create)
        service([f'a{i}' for i in range(41)])
        self.assertEqual(sorted(len(call['input']) for call in calls), [1, 20, 20])
        self.assertTrue(all((call['dimensions'], call['encoding_format'], call['timeout']) == (2, 'float', 60)
                            for call in calls))
        self.assertTrue(all(row['usage']['prompt_tokens'] is None for row in self.request_rows()))
        self.assertNotIn(ENV['EMBEDDING_API_KEY'], self.log.read_text())
        slow, _ = self.service(create, input_tokens_per_second=2, input_tokens_per_minute=2)
        np.testing.assert_equal(slow(['long-input']), [[108., 1.]])
        capped, _ = self.service(create, max_input_tokens=2)
        with self.assertRaisesRegex(ValueError, 'limit'):
            capped(['too long'])

    def test_namespace_has_no_scheduling_or_rc_invalidation(self):
        extended = {**ENV, 'RC_VERSION': 'rc3', 'batch_size': 1, 'coordinator_workers': 8000}
        self.assertEqual(service_module.embedding_namespace(extended, dimension=2), self.namespace)
        self.assertNotEqual(service_module.embedding_namespace({**ENV, 'EMBEDDING_MODEL': 'other'}, dimension=2), self.namespace)
        self.assertNotEqual(service_module.embedding_namespace(ENV, dimension=3), self.namespace)

    def test_invalid_later_input_leaves_no_orphaned_pending_text(self):
        service, _ = self.service(lambda **kwargs: response(kwargs['input']), max_input_tokens=2)
        with self.assertRaisesRegex(ValueError, 'limit'):
            service(['a', 'too long'])
        finished, results = threading.Event(), []
        def retry():
            try:
                results.append(service(['a']))
            finally:
                finished.set()
        thread = threading.Thread(target=retry, daemon=True)
        thread.start()
        self.assertTrue(finished.wait(.5), 'A rejected text list left earlier texts pending forever')
        thread.join(1)
        np.testing.assert_equal(results[0], [[97., 1.]])

    def test_scheduler_applies_token_pacing_and_every_retry_reenters_request_budget(self):
        for policy in ('tokens', 'requests'):
            with self.subTest(policy=policy):
                times, calls = [], Counter()
                inputs = ['r', 's'] if policy == 'tokens' else ['t', 'u']
                def create(**kwargs):
                    times.append(time.monotonic())
                    text, = kwargs['input']
                    calls[text] += 1
                    if text == inputs[0] and calls[text] == 1:
                        raise EndpointError(429, policy + ' limit exceeded')
                    return response(kwargs['input'])
                service, _ = self.service(create, batch_size=1,
                    start_rate=20 if policy == 'requests' else 10000,
                    input_tokens_per_second=20 if policy == 'tokens' else 100000)
                service(inputs)
                self.assertEqual(sum(calls.values()), 3)
                self.assertTrue(all(b - a >= .045 for a, b in zip(times, times[1:])), times)
                feedback = [row for row in self.request_rows() if row['rate_feedback'] == policy]
                self.assertTrue(feedback)
                lowered = 'input_tokens_per_second' if policy == 'tokens' else 'start_rate'
                self.assertEqual(feedback[-1]['rates'][lowered], 16)
                service.close()

    def test_verified_legacy_namespace_reuses_vectors_without_migration(self):
        legacy = {'model': 'offline', 'endpoint': 'https://invalid.test/v1',
                  'encoding_format': 'float', 'storage': 'float32', 'local_metric': 'cosine'}
        with VectorCache(self.root / 'legacy.sqlite', legacy) as cache:
            cache.embed(['a'], lambda _: [[97., 1.]])
            client = FakeClient(lambda **_: self.fail('Legacy vector unexpectedly re-embedded'))
            with service_module.EmbeddingService(ENV, cache, self.root / 'legacy.jsonl',
                                                 limits=self.limits, client=client) as service:
                np.testing.assert_equal(service(['a']), [[97., 1.]])
                self.assertEqual(cache.namespace_config, legacy)

    def test_cache_without_endpoint_identity_is_rejected_before_requests(self):
        with VectorCache(self.root / 'unknown.sqlite', {'model': 'offline'}) as cache:
            client = FakeClient(lambda **_: self.fail('Unknown namespace reached endpoint'))
            with self.assertRaisesRegex(ValueError, 'namespace'):
                service = service_module.EmbeddingService(ENV, cache, self.root / 'unknown.jsonl',
                                                          limits=self.limits, client=client)
                self.addCleanup(service.close)

    def test_authentication_error_is_not_retried_and_unknown_usage_stays_unknown(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            raise EndpointError(401, 'offline authentication failure')
        service, _ = self.service(create)
        with self.assertRaises(EndpointError):
            service(['a'])
        self.assertEqual(len(calls), 1)
        row, = self.request_rows()
        self.assertEqual(row['usage'], {'prompt_tokens': None, 'total_tokens': None})
        self.assertFalse(row['retry_scheduled'])

    def test_telemetry_failure_stops_new_calls_without_discarding_successful_vectors(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs['input'])
            return response(kwargs['input'])
        service, client = self.service(create, batch_size=1, request_limit=1)
        write = service._log.write
        def fail_request_audit(value):
            if '"kind": "embedding_request"' in value:
                raise OSError('offline audit disk full')
            return write(value)
        with patch.object(service._log, 'write', side_effect=fail_request_audit):
            with self.assertRaisesRegex(OSError, 'audit disk full'):
                service(['a', 'b'])
        service.close()
        self.assertEqual(calls, [['a']])
        self.assertIsNotNone(self.cache.get('a'))
        self.assertIsNone(self.cache.get('b'))
        self.assertTrue(client.closed)

    def test_failed_request_thread_start_never_executes_unowned_endpoint_call(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs['input'])
            return response(kwargs['input'])
        service, _ = self.service(create, batch_size=1, request_limit=1)
        original_start, rejected = threading.Thread.start, False
        def start(thread):
            nonlocal rejected
            if thread.name.startswith('deepeye-embedding-request') and not rejected:
                rejected = True
                raise RuntimeError('offline request worker creation failure')
            return original_start(thread)
        with patch.object(threading.Thread, 'start', start):
            with self.assertRaisesRegex(RuntimeError, 'worker creation failure'):
                service(['a', 'b'])
        service.close()
        self.assertEqual(calls, [['b']])
        self.assertIsNone(self.cache.get('a'))
        self.assertIsNotNone(self.cache.get('b'))


class EmbeddingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(service_module, 'Shared embedding service is missing')
        self.limits = service_module.EmbeddingLimits(start_rate=2, input_tokens_per_second=10,
                                                     input_tokens_per_minute=20)
        self.policy = service_module._AdmissionPolicy(self.limits)

    def test_estimated_usage_spaces_requests_and_actual_usage_calibrates(self):
        self.assertEqual(self.policy.delay(0, 10), 0)
        first = self.policy.reserve(0, 10)
        self.assertEqual(self.policy.delay(.1, 10), .9)
        self.policy.calibrate(first, 5, .1)
        self.assertAlmostEqual(self.policy.delay(.1, 10), .4)
        self.policy.reserve(.5, 10)
        self.assertEqual(self.policy.delay(2, 10), 58)

    def test_single_text_over_minute_budget_can_start_once_then_obeys_debt(self):
        self.assertEqual(self.policy.delay(0, 100), 0)
        self.policy.reserve(0, 100)
        self.assertEqual(self.policy.delay(1, 1), 59)
        self.assertEqual(self.policy.delay(60, 1), 0)

    def test_429_feedback_targets_correct_budget_once_per_window_and_recovers(self):
        self.policy.throttle('tokens', 0)
        self.policy.throttle('tokens', 1)
        self.assertEqual(self.policy.snapshot()['input_tokens_per_second'], 8)
        self.assertEqual(self.policy.snapshot()['start_rate'], 2)
        self.policy.throttle('requests', 2)
        self.assertEqual(self.policy.snapshot()['start_rate'], 1.6)
        self.policy.recover(61)
        self.assertEqual(self.policy.snapshot()['input_tokens_per_second'], 8.4)
        self.assertEqual(self.policy.snapshot()['start_rate'], 1.6)
        self.policy.recover(62)
        self.assertAlmostEqual(self.policy.snapshot()['start_rate'], 1.68)

    def test_default_limits_are_independent_from_chat_and_invalid_limits_fail(self):
        default = service_module.EmbeddingLimits()
        self.assertEqual((default.batch_size, default.dimension, default.request_limit, default.request_workers,
                          default.http_connections, default.start_rate, default.input_tokens_per_second,
                          default.input_tokens_per_minute, default.request_timeout, default.max_attempts),
                         (20, 1024, 64, 64, 64, 200, 12000, 720000, 60, 4))
        for values in ({'batch_size': 21}, {'request_limit': 0}, {'max_attempts': 5},
                       {'input_tokens_per_second': float('nan')}, {'dimension': True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                service_module.EmbeddingLimits(**values)

    def test_non_rate_failures_restart_the_stable_recovery_window(self):
        self.policy.throttle('tokens', 0)
        self.policy.note_failure(30)
        self.policy.recover(60)
        self.assertEqual(self.policy.snapshot()['input_tokens_per_second'], 8)
        self.policy.recover(90)
        self.assertEqual(self.policy.snapshot()['input_tokens_per_second'], 8.4)

    def test_feedback_parameters_are_configurable_without_changing_namespace(self):
        limits = replace(self.limits, feedback_window_seconds=10, decrease_factor=.9,
                         recovery_window_seconds=20, recovery_factor=1.1)
        policy = service_module._AdmissionPolicy(limits)
        policy.throttle('tokens', 0)
        policy.throttle('tokens', 9)
        self.assertEqual(policy.snapshot()['input_tokens_per_second'], 9)
        policy.throttle('tokens', 10)
        self.assertAlmostEqual(policy.snapshot()['input_tokens_per_second'], 8.1)
        policy.recover(29)
        self.assertAlmostEqual(policy.snapshot()['input_tokens_per_second'], 8.1)
        policy.recover(30)
        self.assertAlmostEqual(policy.snapshot()['input_tokens_per_second'], 8.91)

    def test_due_oversized_batch_blocks_only_younger_work_until_its_bounded_turn(self):
        limits = service_module.EmbeddingLimits(start_rate=1000,
            input_tokens_per_second=1000, input_tokens_per_minute=60)
        policy = service_module._AdmissionPolicy(limits)
        for second in range(60):
            policy.reserve(float(second), 1)
        batches = [SimpleNamespace(due=0., estimated_tokens=61),
                   SimpleNamespace(due=0., estimated_tokens=1)]
        self.assertEqual(policy.select(60., batches), (0, 59.))
        self.assertEqual(policy.select(119., batches), (0, 0.))

        # Explicit retry backoff still releases the request slot to other work.
        batches[0].due = 180.
        self.assertEqual(policy.select(120., batches), (1, 0.))

    def test_large_queue_selection_reads_rolling_window_independent_of_queue_depth(self):
        class CountingDeque(type(self.policy.window)):
            def __init__(self, values):
                super().__init__(values)
                self.yielded = 0
            def __iter__(self):
                for value in super().__iter__():
                    self.yielded += 1
                    yield value

        for index in range(12000):
            self.policy.reserve(index / 200, 1)
        window = CountingDeque(self.policy.window)
        self.policy.window = window
        batches = [SimpleNamespace(due=0., estimated_tokens=1) for _ in range(1000)]
        self.assertIsNotNone(self.policy.select(59.999, batches))
        self.assertLessEqual(window.yielded, 24000,
            'One admission choice must not rescan the rolling window per queued batch')
