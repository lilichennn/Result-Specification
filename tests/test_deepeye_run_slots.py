"""Offline behavioral checks for pipeline capacity and model observations."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import unittest

BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "DeepEye-SQL"
sys.path.insert(0, str(BASELINE))

from scripts.baseline_adapters.deepeye.run_admission import AdaptivePolicy
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder

try:
    from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
except ModuleNotFoundError as error:
    if error.name != "scripts.baseline_adapters.deepeye.run_slots":
        raise
    PipelineSlots = None


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class PipelineSlotsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(PipelineSlots, "PipelineSlots controller is not implemented")

    def adaptive(self, **policy_kwargs):
        clock = FakeClock()
        events = []
        slots = PipelineSlots(
            policy=AdaptivePolicy(**policy_kwargs), clock=clock,
            emit=lambda kind, payload: events.append((kind, payload)),
        )
        return slots, clock, events

    def observe(self, slots, count, error=None):
        def original():
            if error is not None:
                raise error
            return "answer"
        for _ in range(count):
            if error is None:
                self.assertEqual(slots(original, (), {}), "answer")
            else:
                with self.assertRaises(type(error)) as caught:
                    slots(original, (), {})
                self.assertIs(caught.exception, error)

    def occupy(self, slots, count):
        tickets = [slots.try_acquire(str(i)) for i in range(count)]
        self.assertTrue(all(ticket is not None for ticket in tickets))
        return tickets

    def test_capacity_reuses_freed_slot_only_after_pipeline_release(self):
        """Missing occupancy accounting must admit a forbidden third pipeline."""
        events = []
        slots = PipelineSlots(2, emit=lambda *event: events.append(event))
        slots.set_pending(3)
        first, second = self.occupy(slots, 2)
        self.assertIsNone(slots.try_acquire("third"))
        self.assertEqual(slots.snapshot()["pending"], 1)
        slots.release(first, status="succeeded")
        third = slots.try_acquire("third")
        self.assertEqual(first["slot_id"], third["slot_id"])
        self.assertNotEqual(first["admission_id"], third["admission_id"])
        self.assertEqual(third["item_key"], "third")
        self.assertEqual(third["current_limit"], 2)
        self.assertIsInstance(third["admitted_at"], (int, float))
        slots.release(second, status="failed")
        slots.release(third, status="succeeded")
        snapshot = slots.snapshot()
        self.assertEqual(
            {key: snapshot[key] for key in (
                "scope", "current_limit", "active", "peak_active", "pending", "completed", "failed"
            )},
            {"scope": "pipeline", "current_limit": 2, "active": 0,
             "peak_active": 2, "pending": 0, "completed": 3, "failed": 1},
        )
        self.assertEqual(events, [], "scheduler owns pipeline lifecycle persistence")

    def test_duplicate_and_foreign_releases_cannot_create_capacity(self):
        """Unknown or already released permits must not decrement live occupancy."""
        slots = PipelineSlots(1)
        ticket = slots.try_acquire("owned")
        foreign = PipelineSlots(1).try_acquire("foreign")
        with self.assertRaises(ValueError):
            slots.release(foreign, status="succeeded")
        self.assertIsNone(slots.try_acquire("blocked"))
        slots.release(ticket, status="succeeded")
        with self.assertRaises(ValueError):
            slots.release(ticket, status="succeeded")
        self.assertEqual(slots.snapshot()["completed"], 1)

    def test_invalid_controls_and_release_status_preserve_capacity(self):
        """Invalid capacity or lifecycle input must not silently strand slots."""
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                PipelineSlots(limit)
        slots = PipelineSlots(1)
        for count in (-1, True, 0.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                slots.set_pending(count)
        ticket = slots.try_acquire("a")
        with self.assertRaises(ValueError):
            slots.release(ticket, status="unknown")
        self.assertEqual(slots.snapshot()["active"], 1)
        slots.release(ticket, status="failed")

    def test_single_pipeline_does_not_gate_parallel_model_calls(self):
        """A second API semaphore would prevent both transports from starting."""
        slots = PipelineSlots(1)
        ticket = slots.try_acquire("a")
        both_started = threading.Barrier(3)
        finish = threading.Event()

        def original(value):
            both_started.wait(timeout=3)
            if not finish.wait(timeout=3):
                raise TimeoutError("test transport was not released")
            return value

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(slots, original, (i,), {}) for i in (1, 2)]
            try:
                both_started.wait(timeout=3)
                snapshot = slots.snapshot()
                self.assertEqual(snapshot["active"], 1)
                self.assertEqual(snapshot["model"]["transport_in_flight"], 2)
                self.assertEqual(snapshot["model"]["peak_transport_inflight"], 2)
            finally:
                finish.set()
            self.assertEqual([future.result(timeout=3) for future in futures], [1, 2])
        self.assertEqual(slots.snapshot()["active"], 1)
        slots.release(ticket, status="succeeded")

    def test_observation_preserves_arguments_results_and_transport_timing(self):
        """Mutating calls, counting logging as service, or confusing scopes must fail."""
        clock = FakeClock()
        events = []

        def emit(kind, payload):
            events.append((kind, payload))
            if kind == "api_admission":
                clock.advance(10)

        slots = PipelineSlots(1, clock=clock, emit=emit)
        slots.try_acquire("a")
        marker = object()
        seen = []

        def original(value, *, option):
            seen.append((value, option))
            clock.advance(3)
            return marker

        self.assertIs(slots(original, (marker,), {"option": marker}), marker)
        self.assertEqual(seen, [(marker, marker)])
        self.assertEqual([kind for kind, _ in events], ["api_admission", "api_completion"])
        admitted, completed = [payload for _, payload in events]
        self.assertEqual(admitted["admission_id"], completed["admission_id"])
        self.assertEqual(completed["service_seconds"], 3)
        self.assertEqual(completed["queue_wait_seconds"], 0)
        self.assertEqual(completed["transport_started_at"], 10)
        self.assertEqual(completed["finished_at"], 13)
        self.assertEqual(completed["concurrency_scope"], "pipeline")
        self.assertEqual(completed["pipeline_active"], 1)
        self.assertEqual(completed["current_limit"], 1)
        self.assertEqual(completed["transport_in_flight"], 0)
        model = slots.snapshot()["model"]
        self.assertEqual(model["requested"], 1)
        self.assertEqual(model["completed"], 1)
        self.assertEqual(model["service_seconds"]["samples"], [3])

    def test_transient_error_is_not_retried_and_does_not_release_pipeline(self):
        """Observation must re-raise the original error after exactly one transport."""
        slots = PipelineSlots(1)
        slots.try_acquire("a")
        original_error = ConnectionError("offline transport fixture")
        attempts = []

        def original():
            attempts.append(1)
            raise original_error

        with self.assertRaises(ConnectionError) as caught:
            slots(original, (), {})
        self.assertIs(caught.exception, original_error)
        self.assertEqual(len(attempts), 1)
        snapshot = slots.snapshot()
        self.assertEqual(snapshot["active"], 1)
        self.assertEqual(snapshot["model"]["errors"], 1)
        self.assertEqual(snapshot["model"]["transient_errors"], 1)
        self.assertEqual(snapshot["model"]["transport_in_flight"], 0)

    def test_growth_requires_recent_success_threshold_and_pending_demand(self):
        """Growing before 50 successes or without waiting work must fail."""
        slots, clock, events = self.adaptive()
        self.occupy(slots, 40)
        slots.set_pending(10)
        clock.advance(60)
        self.observe(slots, 49)
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        slots.set_pending(0)
        self.observe(slots, 1)
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        slots.set_pending(10)
        self.observe(slots, 1)
        self.assertEqual(slots.snapshot()["current_limit"], 60)
        changes = [payload for kind, payload in events if kind == "pipeline_concurrency_adjustment"]
        self.assertEqual(len(changes), 1)
        self.assertEqual((changes[0]["old_limit"], changes[0]["new_limit"]), (50, 60))
        self.assertEqual(changes[0]["concurrency_scope"], "pipeline")

    def test_growth_requires_pipeline_demand_not_model_peak(self):
        """Busy model observations from one pipeline must not mimic 80% slot demand."""
        slots, clock, _ = self.adaptive()
        self.occupy(slots, 1)
        slots.set_pending(50)
        clock.advance(60)
        self.observe(slots, 60)
        self.assertEqual(slots.snapshot()["current_limit"], 50)

    def test_growth_waits_for_stable_window_and_expires_old_successes(self):
        """Old successes or an incomplete stability interval must not justify growth."""
        slots, clock, _ = self.adaptive()
        self.occupy(slots, 40)
        slots.set_pending(20)
        self.observe(slots, 50)
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        clock.advance(61)
        self.observe(slots, 1)
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        self.observe(slots, 49)
        self.assertEqual(slots.snapshot()["current_limit"], 60)

    def test_frequent_failures_shrink_without_killing_active_pipelines(self):
        """Shrinking must retain tickets and block refilling above the new limit."""
        slots, clock, _ = self.adaptive()
        tickets = self.occupy(slots, 50)
        slots.set_pending(20)
        self.observe(slots, 4, TimeoutError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        self.observe(slots, 1, TimeoutError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 40)
        self.assertEqual(slots.snapshot()["active"], 50)
        for ticket in tickets[:10]:
            slots.release(ticket, status="failed")
        self.assertIsNone(slots.try_acquire("blocked"))
        slots.release(tickets[10], status="succeeded")
        self.assertIsNotNone(slots.try_acquire("refill"))
        self.observe(slots, 5, ConnectionError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 40, "cooldown prevents repeated shrink")
        clock.advance(60)
        self.observe(slots, 1, ConnectionError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 30)

    def test_low_failure_fraction_and_nontransient_errors_do_not_shrink(self):
        """Failure count alone, or permanent failures, must not trigger backoff."""
        slots, _, _ = self.adaptive()
        self.observe(slots, 50)
        self.observe(slots, 5, TimeoutError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        self.observe(slots, 10, ValueError("invalid fixture"))
        self.assertEqual(slots.snapshot()["current_limit"], 50)

    def test_stale_failures_expire_and_do_not_block_later_growth(self):
        """Failures older than the rolling window must neither shrink nor veto growth."""
        slots, clock, _ = self.adaptive()
        self.occupy(slots, 40)
        slots.set_pending(20)
        self.observe(slots, 4, TimeoutError("old"))
        clock.advance(61)
        self.observe(slots, 1, TimeoutError("new"))
        self.assertEqual(slots.snapshot()["current_limit"], 50)
        clock.advance(61)
        self.observe(slots, 50)
        self.assertEqual(slots.snapshot()["current_limit"], 60)

    def test_capacity_adjustments_respect_minimum_and_maximum(self):
        """Step-sized changes must clamp at policy bounds instead of overshooting."""
        slots, clock, _ = self.adaptive(initial_limit=15, min_limit=10, max_limit=22)
        self.occupy(slots, 15)
        slots.set_pending(30)
        clock.advance(60)
        self.observe(slots, 50)
        self.assertEqual(slots.snapshot()["current_limit"], 22)
        self.assertEqual(slots.max_limit, 22)
        clock.advance(61)
        self.observe(slots, 5, TimeoutError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 12)
        clock.advance(61)
        self.observe(slots, 5, TimeoutError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 10)
        clock.advance(61)
        self.observe(slots, 5, TimeoutError("offline"))
        self.assertEqual(slots.snapshot()["current_limit"], 10)

    def test_callbacks_and_clock_execute_outside_state_lock(self):
        """A callback or injected clock waiting on another reader must not deadlock."""
        holder = {}

        def inspect_from_another_thread():
            if "slots" not in holder:
                return
            done = threading.Event()
            thread = threading.Thread(target=lambda: (holder["slots"].snapshot(), done.set()), daemon=True)
            thread.start()
            self.assertTrue(done.wait(2), "controller lock held while calling external code")
            thread.join(timeout=2)

        def clock():
            inspect_from_another_thread()
            return 0.0

        def emit(kind, payload):
            inspect_from_another_thread()

        slots = PipelineSlots(1, clock=clock, emit=emit)
        holder["slots"] = slots
        ticket = slots.try_acquire("a")
        self.observe(slots, 1)
        slots.release(ticket, status="succeeded")

    def test_logging_errors_stop_unstarted_transport_and_preserve_original_error(self):
        """Broken admission logging must stop the call; completion logging cannot mask its error."""
        logging_error = OSError("offline recorder failure")

        def emit(kind, payload):
            raise logging_error

        slots = PipelineSlots(1, emit=emit)

        def forbidden():
            self.fail("transport started after admission logging failed")

        with self.assertRaises(OSError) as caught:
            slots(forbidden, (), {})
        self.assertIs(caught.exception, logging_error)
        self.assertEqual(slots.snapshot()["model"]["transport_in_flight"], 0)

        def completion_emit(kind, payload):
            if kind == "api_completion":
                raise logging_error

        slots = PipelineSlots(1, emit=completion_emit)
        self.observe(slots, 1, TimeoutError("original"))
        self.assertEqual(slots.snapshot()["model"]["transport_in_flight"], 0)

    def test_latency_samples_are_bounded_but_totals_are_full_history(self):
        """Discarding old metric samples must not lose total request or latency statistics."""
        clock = FakeClock()
        slots = PipelineSlots(1, clock=clock)

        def original():
            clock.advance(2)

        for _ in range(1100):
            slots(original, (), {})
        model = slots.snapshot()["model"]
        self.assertEqual(model["completed"], 1100)
        self.assertLessEqual(len(model["service_seconds"]["samples"]), 1000)
        self.assertEqual(model["service_seconds"]["count"], 1100)
        self.assertEqual(model["service_seconds"]["total"], 2200)
        self.assertEqual(model["service_seconds"]["max"], 2)

    def test_adjustment_is_persisted_in_originating_api_call_context(self):
        """Moving adjustment logging out of the API context loses durable call attribution."""
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {"fixture": "slots"}) as store:
                attempt = store.begin_attempt("q", "generation", "fixture-input")
                recorder = TraceRecorder(store)
                slots = PipelineSlots(policy=AdaptivePolicy(), emit=recorder.record_admission)
                recorder.api_call = slots
                ticket = slots.try_acquire("q")

                def original():
                    raise TimeoutError("offline fixture")

                traced = recorder._api_wrapper(original)
                with recorder.context(attempt):
                    for _ in range(5):
                        with self.assertRaises(TimeoutError):
                            traced()
                slots.release(ticket, status="failed")
                store.finish_attempt(attempt, "failed", {"fixture": "timeout"})
                events = store.events(attempt)
        changes = [event for event in events if event["kind"] == "pipeline_concurrency_adjustment"]
        completions = [event for event in events if event["kind"] == "api_completion"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(len(completions), 5)
        self.assertEqual(changes[0]["attempt_id"], attempt)
        self.assertIsNotNone(changes[0]["payload"]["call_id"])
        self.assertEqual(changes[0]["payload"]["call_id"], completions[-1]["payload"]["call_id"])
        self.assertEqual(changes[0]["payload"]["admission_id"], completions[-1]["payload"]["admission_id"])
        self.assertEqual(changes[0]["payload"]["current_limit"], 40)


if __name__ == "__main__":
    unittest.main()
