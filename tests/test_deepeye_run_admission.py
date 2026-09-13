"""Deterministic tests for run-boundary concurrency admission."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
import threading
import unittest

BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "DeepEye-SQL"
sys.path.insert(0, str(BASELINE))

from scripts.baseline_adapters.deepeye.run_admission import (
    AdaptiveAdmission,
    AdaptivePolicy,
    FixedAdmission,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value += seconds


class AdaptivePolicyTests(unittest.TestCase):
    def test_defaults_describe_the_bounded_paid_batch(self):
        """Changing the approved 50/10/60/10..100 policy must fail."""
        policy = AdaptivePolicy()

        self.assertEqual(policy.initial_limit, 50)
        self.assertEqual(policy.step, 10)
        self.assertEqual(policy.stable_window_s, 60.0)
        self.assertEqual(policy.min_successes, 50)
        self.assertEqual(policy.failure_threshold, 5)
        self.assertEqual(policy.failure_rate, 0.10)
        self.assertEqual(policy.min_limit, 10)
        self.assertEqual(policy.max_limit, 100)

    def test_invalid_bounds_and_thresholds_are_rejected(self):
        """Nonpositive, inverted, or out-of-range controls must not deadlock a run."""
        invalid = (
            {"initial_limit": 0},
            {"step": 0},
            {"min_limit": 20, "initial_limit": 10},
            {"initial_limit": 101},
            {"stable_window_s": 0},
            {"stable_window_s": math.nan},
            {"stable_window_s": math.inf},
            {"min_successes": 0},
            {"failure_threshold": 0},
            {"failure_rate": 0},
            {"failure_rate": 1.1},
            {"demand_utilization": 0},
            {"demand_utilization": 1.1},
            {"adjustment_cooldown_s": -1},
        )

        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                AdaptivePolicy(**values)


class AdaptiveAdmissionTests(unittest.TestCase):
    def test_success_is_called_once_unchanged_and_records_timing(self):
        """A wrapper that retries, mutates inputs, or omits boundary timing must fail."""
        clock = FakeClock()
        events: list[tuple[str, dict]] = []
        admission = AdaptiveAdmission(
            AdaptivePolicy(initial_limit=2, min_limit=1, max_limit=3),
            emit=lambda kind, payload: events.append((kind, payload)),
            clock=clock,
        )
        marker = object()
        args = (marker,)
        kwargs = {"option": marker}
        calls = []
        response = object()

        def original(*actual_args, **actual_kwargs):
            calls.append((actual_args, actual_kwargs))
            clock.advance(3.0)
            return response

        actual = admission(original, args, kwargs)

        self.assertIs(actual, response)
        self.assertEqual(calls, [(args, kwargs)])
        self.assertEqual(args, (marker,))
        self.assertEqual(kwargs, {"option": marker})
        self.assertEqual([kind for kind, _ in events], ["api_admission", "api_completion"])
        admitted = events[0][1]
        completed = events[1][1]
        self.assertEqual(admitted["admission_id"], completed["admission_id"])
        self.assertEqual(admitted["time_monotonic"], 0.0)
        self.assertEqual(admitted["queue_wait_seconds"], 0.0)
        self.assertEqual(admitted["in_flight"], 1)
        self.assertEqual(admitted["current_limit"], 2)
        self.assertEqual(completed["time_monotonic"], 3.0)
        self.assertEqual(completed["service_seconds"], 3.0)
        self.assertEqual(completed["in_flight"], 0)
        self.assertTrue(completed["success"])
        self.assertIsNone(completed["error_type"])
        self.assertIsNone(completed["status_code"])

        snapshot = admission.snapshot()
        self.assertEqual(snapshot["requested"], 1)
        self.assertEqual(snapshot["completed"], 1)
        self.assertEqual(snapshot["errors"], 0)
        self.assertEqual(snapshot["transient_errors"], 0)
        self.assertEqual(snapshot["peak_inflight"], 1)
        self.assertEqual(snapshot["in_flight"], 0)
        self.assertEqual(snapshot["current_limit"], 2)
        self.assertEqual(snapshot["queue_wait_seconds"]["samples"], [0.0])
        self.assertEqual(snapshot["service_seconds"]["samples"], [3.0])

    def test_slow_admission_logging_is_not_counted_as_transport_service(self):
        """Starting transport counters or service timing before durable admission must fail."""
        clock = FakeClock()
        events = []
        observed_during_emit = []
        admission = None

        def emit(kind, payload):
            events.append((kind, payload))
            if kind == "api_admission":
                observed_during_emit.append(admission.snapshot())
                clock.advance(5)

        admission = AdaptiveAdmission(
            AdaptivePolicy(initial_limit=1, min_limit=1, max_limit=1),
            emit=emit,
            clock=clock,
        )

        def original():
            clock.advance(2)
            return "ok"

        self.assertEqual(admission(original, (), {}), "ok")

        self.assertEqual(observed_during_emit[0]["in_flight"], 1)
        self.assertEqual(observed_during_emit[0]["transport_in_flight"], 0)
        completion = events[-1][1]
        self.assertEqual(completion["transport_started_at"], 5.0)
        self.assertEqual(completion["finished_at"], 7.0)
        self.assertEqual(completion["service_seconds"], 2.0)
        self.assertEqual(completion["peak_transport_inflight"], 1)
        snapshot = admission.snapshot()
        self.assertEqual(snapshot["transport_in_flight"], 0)
        self.assertEqual(snapshot["peak_transport_inflight"], 1)

    def test_only_transport_transients_are_classified_and_never_retried(self):
        """Misclassifying 4xx or retrying any original transport must fail."""
        class HTTPError(RuntimeError):
            def __init__(self, status_code):
                super().__init__("secret provider response")
                self.status_code = status_code

        cases = (
            (TimeoutError("secret timeout"), True, None),
            (ConnectionError("secret connection"), True, None),
            (HTTPError(429), True, 429),
            (HTTPError(503), True, 503),
            (HTTPError(400), False, 400),
            (ValueError("bad request with secret"), False, None),
        )

        for error, expected_transient, expected_status in cases:
            with self.subTest(error=type(error).__name__, status=expected_status):
                events = []
                admission = AdaptiveAdmission(emit=lambda k, p: events.append((k, p)))
                calls = 0

                def original():
                    nonlocal calls
                    calls += 1
                    raise error

                try:
                    admission(original, (), {})
                except BaseException as actual:
                    self.assertIs(actual, error)
                else:  # pragma: no cover - assertion guard
                    self.fail("original exception was swallowed")

                self.assertEqual(calls, 1)
                completion = events[-1][1]
                self.assertEqual(completion["transient_error"], expected_transient)
                self.assertEqual(completion["status_code"], expected_status)
                self.assertEqual(
                    completion["error_type"],
                    f"{type(error).__module__}.{type(error).__qualname__}",
                )
                self.assertNotIn("secret", repr(completion))
                self.assertEqual(admission.snapshot()["transient_errors"],
                                 int(expected_transient))

    def test_stable_saturated_window_increases_by_one_bounded_step(self):
        """Increasing without elapsed stability, successes, or observed demand must fail."""
        clock = FakeClock()
        events = []
        admission = AdaptiveAdmission(
            AdaptivePolicy(
                initial_limit=2,
                step=1,
                min_limit=1,
                max_limit=3,
                stable_window_s=10,
                min_successes=2,
                failure_threshold=2,
                adjustment_cooldown_s=10,
            ),
            emit=lambda kind, payload: events.append((kind, payload)),
            clock=clock,
        )
        release = threading.Event()
        both_running = threading.Event()
        entered = 0
        lock = threading.Lock()

        def original(number):
            nonlocal entered
            with lock:
                entered += 1
                if entered == 2:
                    both_running.set()
            self.assertTrue(release.wait(2))
            return number

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(admission, original, (number,), {})
                       for number in range(2)]
            self.assertTrue(both_running.wait(2))
            self.assertEqual(admission.snapshot()["current_limit"], 2)
            clock.advance(10)
            release.set()
            self.assertEqual([future.result(timeout=2) for future in futures], [0, 1])

        snapshot = admission.snapshot()
        self.assertEqual(snapshot["current_limit"], 3)
        self.assertEqual(snapshot["peak_inflight"], 2)
        self.assertEqual(len(snapshot["adjustments"]), 1)
        adjustment = snapshot["adjustments"][0]
        self.assertEqual((adjustment["old_limit"], adjustment["new_limit"]), (2, 3))
        self.assertEqual(adjustment["reason"], "stable_demand")
        self.assertEqual(
            [kind for kind, _ in events].count("api_concurrency_adjustment"), 1
        )

    def test_repeated_transients_decrease_once_then_require_a_fresh_threshold(self):
        """Reusing the same five failures for several downward steps must fail."""
        clock = FakeClock()
        events = []
        admission = AdaptiveAdmission(
            AdaptivePolicy(
                initial_limit=10,
                step=2,
                min_limit=2,
                max_limit=10,
                stable_window_s=60,
                min_successes=50,
                failure_threshold=5,
                failure_rate=0.10,
                adjustment_cooldown_s=60,
            ),
            emit=lambda kind, payload: events.append((kind, payload)),
            clock=clock,
        )

        for _ in range(5):
            admission(lambda: "ok", (), {})

        def fail():
            raise TimeoutError("provider detail must not be logged")

        for _ in range(5):
            with self.assertRaises(TimeoutError):
                admission(fail, (), {})
        self.assertEqual(admission.snapshot()["current_limit"], 8)

        for _ in range(4):
            with self.assertRaises(TimeoutError):
                admission(fail, (), {})
        self.assertEqual(admission.snapshot()["current_limit"], 8)

        clock.advance(60)
        with self.assertRaises(TimeoutError):
            admission(fail, (), {})

        snapshot = admission.snapshot()
        self.assertEqual(snapshot["current_limit"], 6)
        self.assertEqual(
            [(item["old_limit"], item["new_limit"]) for item in snapshot["adjustments"]],
            [(10, 8), (8, 6)],
        )
        self.assertTrue(all(item["reason"] == "transient_failures"
                            for item in snapshot["adjustments"]))
        self.assertEqual(
            [kind for kind, _ in events].count("api_concurrency_adjustment"), 2
        )

    def test_expired_transient_does_not_block_a_later_stable_increase(self):
        """Keeping an old 429 forever must not poison a later clean demand window."""
        clock = FakeClock()
        admission = AdaptiveAdmission(
            AdaptivePolicy(
                initial_limit=2,
                step=1,
                min_limit=1,
                max_limit=3,
                stable_window_s=60,
                min_successes=2,
                failure_threshold=5,
                adjustment_cooldown_s=60,
            ),
            clock=clock,
        )

        class RateLimited(RuntimeError):
            status_code = 429

        with self.assertRaises(RateLimited):
            admission(lambda: (_ for _ in ()).throw(RateLimited()), (), {})
        clock.advance(61)

        release = threading.Event()
        both_running = threading.Event()
        entered = 0
        lock = threading.Lock()

        def succeed():
            nonlocal entered
            with lock:
                entered += 1
                if entered == 2:
                    both_running.set()
            self.assertTrue(release.wait(2))
            return "ok"

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(admission, succeed, (), {}) for _ in range(2)]
            self.assertTrue(both_running.wait(2))
            release.set()
            self.assertEqual([future.result(timeout=2) for future in futures],
                             ["ok", "ok"])

        self.assertEqual(admission.snapshot()["current_limit"], 3)

    def test_sparse_transients_do_not_accumulate_across_the_stable_window(self):
        """Five failures spanning more than 60 seconds must not cause a decrease."""
        clock = FakeClock()
        admission = AdaptiveAdmission(
            AdaptivePolicy(
                initial_limit=10,
                step=2,
                min_limit=2,
                max_limit=10,
                stable_window_s=60,
                min_successes=50,
                failure_threshold=5,
                failure_rate=0.10,
                adjustment_cooldown_s=60,
            ),
            clock=clock,
        )

        def fail():
            raise TimeoutError()

        for number in range(5):
            if number:
                clock.advance(16)
            with self.assertRaises(TimeoutError):
                admission(fail, (), {})

        self.assertEqual(admission.snapshot()["current_limit"], 10)
        self.assertEqual(admission.snapshot()["adjustments"], [])


class FixedAdmissionTests(unittest.TestCase):
    def test_fixed_gate_caps_actual_calls_and_emits_prefixed_metrics(self):
        """Letting more originals than the fixed limit overlap must fail."""
        events = []
        admission = FixedAdmission(
            2, emit=lambda kind, payload: events.append((kind, payload))
        )
        lock = threading.Lock()
        active = 0
        observed_peak = 0
        release = threading.Event()
        first_pair = threading.Event()

        def original(number):
            nonlocal active, observed_peak
            with lock:
                active += 1
                observed_peak = max(observed_peak, active)
                if active == 2:
                    first_pair.set()
            if number < 2:
                self.assertTrue(release.wait(2))
            with lock:
                active -= 1
            return number

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(admission, original, (number,), {})
                       for number in range(6)]
            self.assertTrue(first_pair.wait(2))
            self.assertEqual(admission.snapshot()["transport_in_flight"], 2)
            release.set()
            self.assertEqual([future.result(timeout=2) for future in futures],
                             list(range(6)))

        self.assertEqual(observed_peak, 2)
        self.assertEqual(admission.snapshot()["peak_transport_inflight"], 2)
        self.assertEqual([kind for kind, _ in events].count("postgres_admission"), 6)
        self.assertEqual([kind for kind, _ in events].count("postgres_completion"), 6)

    def test_emitter_failure_is_visible_and_never_leaks_a_permit(self):
        """Swallowing storage failure or leaking its acquired permit must fail."""
        emissions = 0

        def emit(kind, payload):
            nonlocal emissions
            emissions += 1
            if emissions == 1:
                raise RuntimeError("store unavailable")

        admission = FixedAdmission(1, emit=emit)
        calls = 0

        def original():
            nonlocal calls
            calls += 1
            return "ok"

        with self.assertRaisesRegex(RuntimeError, "store unavailable"):
            admission(original, (), {})
        self.assertEqual(calls, 0)
        self.assertEqual(admission.snapshot()["in_flight"], 0)
        self.assertEqual(admission.snapshot()["transport_in_flight"], 0)
        self.assertEqual(admission(original, (), {}), "ok")
        self.assertEqual(calls, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
