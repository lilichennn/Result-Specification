"""Offline concurrency acceptance through the C1 retry and C3 runtime path."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import importlib.util
import threading
import time
import unittest
from unittest.mock import patch

from tests.test_deepeye_sampling import response
from app.llm.sampling import execute_group, sampling_identity


def parse_message(message):
    return message.content if message.content.startswith('SELECT') else None


class ParallelSamplingTests(unittest.TestCase):
    def runtime(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('scripts.baseline_adapters.deepeye.sampling'),
                             'C3 shared sample scheduler is missing')
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        runtime = SamplingRuntime(**kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def test_capacities_one_two_five_allow_progress_and_bound_overlap(self):
        for cap in (1, 2, 5):
            with self.subTest(cap=cap):
                runtime = self.runtime(request_workers=cap, request_limit=cap,
                                       coordinator_workers=2, start_rate=10000)
                barrier = threading.Barrier(cap)
                active = peak = 0
                lock = threading.Lock()
                def request():
                    nonlocal active, peak
                    with lock:
                        active += 1
                        peak = max(peak, active)
                    if sampling_identity()['sample_index'] < cap:
                        barrier.wait(2)
                    time.sleep(.005)
                    with lock:
                        active -= 1
                    return response()
                with runtime.context():
                    group = execute_group(request, parse_message, n=10)
                self.assertTrue(group.complete)
                self.assertEqual(peak, cap)
                self.assertEqual([s.sample_index for s in group.samples], list(range(10)))

    def test_completion_order_is_not_candidate_order_and_context_survives(self):
        runtime = self.runtime(request_workers=5, coordinator_workers=1, start_rate=10000)
        events = [threading.Event() for _ in range(5)]
        started = threading.Barrier(6)
        finished = []
        marker = ContextVar('test_rc_context', default=None)
        marker.set('contract-bound')
        def request():
            index = sampling_identity()['sample_index']
            self.assertEqual(marker.get(), 'contract-bound')
            started.wait(2)
            self.assertTrue(events[index].wait(2))
            finished.append(index)
            return response(f'SELECT {index}')
        with runtime.context(), ThreadPoolExecutor(1) as caller:
            import contextvars
            future = caller.submit(contextvars.copy_context().run,
                                   execute_group, request, parse_message, n=5)
            started.wait(2)
            for index in (4, 2, 0, 3, 1):
                events[index].set()
                deadline = time.monotonic() + 2
                while index not in finished and time.monotonic() < deadline:
                    time.sleep(.001)
            group = future.result(3)
        self.assertEqual(finished, [4, 2, 0, 3, 1])
        self.assertEqual(group.results, [f'SELECT {i}' for i in range(5)])

    def test_failed_sample_thread_start_cannot_execute_unowned_sample(self):
        runtime = self.runtime(request_workers=2, start_rate=10000)
        release = threading.Event()
        entered = []
        original_start = threading.Thread.start
        injected = False
        def start(thread):
            nonlocal injected
            if thread.name.startswith('deepeye-sample') and not injected:
                injected = True
                raise RuntimeError('injected sample thread creation failure')
            return original_start(thread)
        def request():
            index = sampling_identity()['sample_index']
            entered.append(index)
            if index == 0:
                release.wait(2)
            return response()
        try:
            with patch.object(threading.Thread, 'start', start), runtime.context():
                with self.assertRaisesRegex(RuntimeError, 'injected sample thread'):
                    execute_group(request, parse_message, n=3)
            self.assertTrue(injected)
            self.assertCountEqual(entered, [1, 2])
        finally:
            release.set()
            runtime.close()
        self.assertCountEqual(entered, [1, 2])
        self.assertEqual(runtime.samples.snapshot()['active'], 0)

    def test_failed_coordinator_thread_start_cannot_execute_unowned_parent(self):
        runtime = self.runtime(request_workers=1, coordinator_workers=2, start_rate=10000)
        release = threading.Event()
        entered = threading.Event()
        original_start = threading.Thread.start
        injected = False
        def start(thread):
            nonlocal injected
            if thread.name.startswith('deepeye-coordinator') and not injected:
                injected = True
                raise RuntimeError('injected coordinator thread creation failure')
            return original_start(thread)
        def unowned():
            entered.set()
            release.wait(2)
        try:
            with patch.object(threading.Thread, 'start', start):
                with self.assertRaisesRegex(RuntimeError, 'injected coordinator thread'):
                    runtime.submit_coordinator(unowned)
                second = runtime.submit_coordinator(lambda: 'owned second')
                third = runtime.submit_coordinator(lambda: 'owned third')
                self.assertEqual(second.result(1), 'owned second')
                self.assertEqual(third.result(1), 'owned third')
            self.assertFalse(entered.is_set(), 'A rejected coordinator executed without an owned Future')
        finally:
            release.set()
            runtime.close()
        self.assertFalse(entered.is_set())

    def test_capacity_partial_submission_stops_and_drains_before_close(self):
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        from tests.deepeye_runtime_capacity import run_capacity
        original_submit = SamplingRuntime.submit_coordinator
        original_close = SamplingRuntime.close
        submissions = 0
        observed = []
        def submit(runtime, *args, **kwargs):
            nonlocal submissions
            submissions += 1
            if submissions == 2:
                raise RuntimeError('injected partial capacity submission')
            future = original_submit(runtime, *args, **kwargs)
            deadline = time.monotonic() + 2
            while runtime.dispatch.snapshot()['in_flight'] == 0 and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 1)
            return future
        def close(runtime):
            was_stopped = runtime.stop_event.is_set()
            if not was_stopped:
                # Bound RED teardown without waiting for a30s barrier deadline.
                runtime.stop(cancel_active=True)
            original_close(runtime)
            observed.append(dict(stopped_before_close=was_stopped,
                inflight=runtime.dispatch.snapshot()['in_flight'],
                http_stopped=not runtime.dispatch._thread.is_alive(),
                samples_stopped=all(not t.is_alive() for t in runtime.samples._pool._threads),
                coordinators_stopped=all(not t.is_alive() for t in runtime.coordinators._threads)))
        started = time.monotonic()
        with patch.object(SamplingRuntime, 'submit_coordinator', submit), patch.object(SamplingRuntime, 'close', close):
            with self.assertRaisesRegex(RuntimeError, 'injected partial capacity submission'):
                run_capacity(request_cap=2, coordinator_cap=2, transport='fake', deadline=30)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(observed, [dict(stopped_before_close=True, inflight=0,
                                       http_stopped=True, samples_stopped=True, coordinators_stopped=True)])

    def test_one_parse_retry_does_not_repeat_successes(self):
        runtime = self.runtime(request_workers=5, start_rate=10000)
        calls = Counter()
        lock = threading.Lock()
        def request():
            identity = sampling_identity()
            with lock:
                calls[identity['sample_index']] += 1
            return response('invalid' if identity['sample_index'] == 2 and
                            identity['sample_attempt'] == 1 else 'SELECT 1')
        with runtime.context():
            group = execute_group(request, parse_message, n=5)
        self.assertTrue(group.complete)
        self.assertEqual(calls, {0: 1, 1: 1, 2: 2, 3: 1, 4: 1})

    def test_coordinators_do_not_consume_sample_workers_and_groups_are_fair(self):
        runtime = self.runtime(request_workers=2, coordinator_workers=2, start_rate=10000)
        first_started = threading.Event()
        release = threading.Event()
        order = []
        def request():
            name = label.get()
            order.append(name)
            first_started.set()
            self.assertTrue(release.wait(2))
            time.sleep(.002)
            return response()
        label = ContextVar('group_label')
        with runtime.context():
            label.set('large')
            large = runtime.submit_coordinator(execute_group, request, parse_message, n=30)
            self.assertTrue(first_started.wait(2))
            label.set('small')
            small = runtime.submit_coordinator(execute_group, request, parse_message, n=4)
            time.sleep(.02)
            release.set()
            self.assertTrue(small.result(3).complete)
            self.assertTrue(large.result(3).complete)
        self.assertLess(order.index('small'), 6)
        self.assertLess(order.index('small') + 3, len(order))

    def test_base_exception_drains_siblings_before_propagating(self):
        from app.llm.sampling import SamplingIdentityError
        runtime = self.runtime(request_workers=5, start_rate=10000)
        started = threading.Barrier(5)
        finished = []
        def request():
            index = sampling_identity()['sample_index']
            started.wait(2)
            if index == 0:
                raise SamplingIdentityError('integrity failure')
            time.sleep(.03)
            finished.append(index)
            return response()
        with runtime.context(), self.assertRaises(SamplingIdentityError):
            execute_group(request, parse_message, n=5)
        self.assertCountEqual(finished, [1, 2, 3, 4])

    def test_nested_coordinator_submission_at_cap_one_makes_progress(self):
        runtime = self.runtime(request_workers=2, coordinator_workers=1, start_rate=10000)
        def parent():
            child = runtime.submit_coordinator(execute_group, response, parse_message, n=5)
            return child.result(.3)
        with runtime.context():
            self.assertTrue(runtime.submit_coordinator(parent).result(2).complete)

    def test_new_runtime_offline_capacity_uses_real_checkpoints_and_trace(self):
        self.assertIsNotNone(importlib.util.find_spec('tests.deepeye_runtime_capacity'),
                             'Repeatable C3 runtime capacity helper is missing')
        from tests.deepeye_runtime_capacity import run_capacity
        result = run_capacity(request_cap=32, coordinator_cap=16, transport='fake')
        self.assertEqual(result['actual_request_peak'], 32)
        self.assertEqual(result['sample_worker_peak'], 32)
        self.assertLessEqual(result['coordinator_peak'], 16)
        self.assertEqual(result['checkpoint_successes'], result['total_samples'])
        self.assertEqual(result['request_trace_events'], result['total_samples'])
        self.assertEqual(result['final_inflight'], 0)
        self.assertTrue(result['store_verified'])
        self.assertEqual(result['barrier_threads']['joint_request_inflight'], 32)
        self.assertEqual(result['barrier_threads']['joint_coordinator_active'], 16)
        self.assertTrue(result['cleanup']['http_loop_stopped'])
        self.assertTrue(result['cleanup']['sample_threads_stopped'])
        self.assertTrue(result['cleanup']['coordinator_threads_stopped'])
