"""Question-pipeline capacity and ungated observations of model transports.

The scheduler owns durable pipeline lifecycle records. Only API observation
emits events here, while its caller still owns the originating trace context.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any, Callable
import uuid

from .run_admission import AdaptiveAdmission, AdaptivePolicy


class PipelineSlots:
    def __init__(
        self,
        fixed_limit: int = 4,
        policy: AdaptivePolicy | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if policy is not None and not isinstance(policy, AdaptivePolicy):
            raise TypeError("policy must be AdaptivePolicy or None")
        if policy is None and (type(fixed_limit) is not int or fixed_limit <= 0):
            raise ValueError("fixed_limit must be a positive integer")
        if emit is not None and not callable(emit):
            raise TypeError("emit must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.policy = policy
        self._emit = emit
        self._clock = clock
        self._lock = threading.Lock()
        self._current_limit = policy.initial_limit if policy else fixed_limit
        self._max_limit = policy.max_limit if policy else fixed_limit
        self._tickets: dict[str, tuple[dict[str, Any], int]] = {}
        self._occupied: set[int] = set()
        self._pending = 0
        self._peak_active = 0
        self._completed = 0
        self._failed = 0
        self._requested = 0
        self._model_completed = 0
        self._errors = 0
        self._transient_errors = 0
        self._transport_in_flight = 0
        self._peak_transport_inflight = 0
        self._service_samples: deque[float] = deque(maxlen=1000)
        self._service_total = 0.0
        self._service_max = 0.0
        # Exact health observations expire after the policy window. Only
        # latency samples and the adjustment preview retain a bounded history;
        # durable API events remain the source for full-history distributions.
        self._recent_completions: deque[tuple[float, bool, bool]] = deque()
        self._adjustments: deque[dict[str, Any]] = deque(maxlen=1000)
        self._adjustment_count = 0
        self._last_change_at = clock()
        self._last_adjustment_at: float | None = None

    @property
    def max_limit(self) -> int:
        return self._max_limit

    def set_pending(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("pending must be a nonnegative integer")
        with self._lock:
            self._pending = count

    def try_acquire(self, item_key: Any) -> dict[str, Any] | None:
        admitted_at = self._clock()
        admission_id = uuid.uuid4().hex
        with self._lock:
            if len(self._tickets) >= self._current_limit:
                return None
            slot_id = next(slot for slot in range(self._max_limit) if slot not in self._occupied)
            ticket = {
                "slot_id": slot_id,
                "admission_id": admission_id,
                "item_key": item_key,
                "admitted_at": admitted_at,
                "current_limit": self._current_limit,
            }
            self._tickets[admission_id] = (ticket, slot_id)
            self._occupied.add(slot_id)
            self._pending = max(0, self._pending - 1)
            self._peak_active = max(self._peak_active, len(self._tickets))
            return ticket

    def release(self, ticket: dict[str, Any], *, status: str) -> None:
        if status not in ("succeeded", "failed"):
            raise ValueError("status must be succeeded or failed")
        if type(ticket) is not dict or type(ticket.get("admission_id")) is not str:
            raise ValueError("unknown pipeline ticket")
        admission_id = ticket["admission_id"]
        with self._lock:
            owned = self._tickets.get(admission_id)
            if owned is None or owned[0] is not ticket:
                raise ValueError("unknown or already released pipeline ticket")
            del self._tickets[admission_id]
            self._occupied.remove(owned[1])
            self._completed += 1
            self._failed += status == "failed"

    def _capacity_payload_locked(self) -> dict[str, Any]:
        return {
            "concurrency_scope": "pipeline",
            "current_limit": self._current_limit,
            "pipeline_active": len(self._tickets),
            "pipeline_peak_active": self._peak_active,
            "pipeline_pending": self._pending,
            "transport_in_flight": self._transport_in_flight,
            "peak_transport_inflight": self._peak_transport_inflight,
        }

    def _maybe_adjust_locked(
        self, now: float, admission_id: str, completed_transient: bool,
    ) -> dict[str, Any] | None:
        policy = self.policy
        if policy is None:
            return None
        cutoff = now - policy.stable_window_s
        # Clock readings occur outside the lock; concurrent completions can
        # enter it out of timestamp order, so do not assume deque order here.
        self._recent_completions = deque(
            item for item in self._recent_completions if item[0] >= cutoff
        )
        completed = len(self._recent_completions)
        successes = sum(success for _, success, _ in self._recent_completions)
        failures = sum(transient for _, _, transient in self._recent_completions)
        failure_rate = failures / completed if completed else 0.0
        cooldown_ready = (
            self._last_adjustment_at is None
            or now - self._last_adjustment_at >= policy.adjustment_cooldown_s
        )
        new_limit = self._current_limit
        reason = None
        if (
            completed_transient and cooldown_ready
            and failures >= policy.failure_threshold
            and failure_rate >= policy.failure_rate
        ):
            new_limit = max(policy.min_limit, self._current_limit - policy.step)
            reason = "transient_failures"
        elif (
            cooldown_ready
            and now - self._last_change_at >= policy.stable_window_s
            and successes >= policy.min_successes
            and failures == 0
            and self._pending > 0
            and len(self._tickets) >= policy.demand_utilization * self._current_limit
        ):
            new_limit = min(policy.max_limit, self._current_limit + policy.step)
            reason = "stable_demand"
        if new_limit == self._current_limit:
            return None
        old_limit = self._current_limit
        self._current_limit = new_limit
        adjustment = {
            **self._capacity_payload_locked(),
            "admission_id": admission_id,
            "time_monotonic": now,
            "old_limit": old_limit,
            "new_limit": new_limit,
            "direction": "increase" if new_limit > old_limit else "decrease",
            "reason": reason,
            "window_seconds": min(policy.stable_window_s, max(0.0, now - self._last_change_at)),
            "window_completed": completed,
            "window_successes": successes,
            "window_transient_errors": failures,
            "window_transient_error_rate": failure_rate,
        }
        self._adjustments.append(dict(adjustment))
        self._adjustment_count += 1
        self._last_change_at = now
        self._last_adjustment_at = now
        self._recent_completions.clear()
        return adjustment

    def _finish(
        self, admission_id: str, started_at: float, error: BaseException | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        finished_at = self._clock()
        transient = AdaptiveAdmission._is_transient(error)
        status_code = AdaptiveAdmission._status_code(error)
        error_type = f"{type(error).__module__}.{type(error).__qualname__}" if error else None
        service_seconds = max(0.0, finished_at - started_at)
        with self._lock:
            self._transport_in_flight -= 1
            self._model_completed += 1
            self._errors += error is not None
            self._transient_errors += transient
            self._service_samples.append(service_seconds)
            self._service_total += service_seconds
            self._service_max = max(self._service_max, service_seconds)
            if self.policy is not None:
                self._recent_completions.append((finished_at, error is None, transient))
            adjustment = self._maybe_adjust_locked(finished_at, admission_id, transient)
            payload = {
                **self._capacity_payload_locked(),
                "admission_id": admission_id,
                "time_monotonic": finished_at,
                "transport_started_at": started_at,
                "finished_at": finished_at,
                "queue_wait_seconds": 0.0,
                "service_seconds": service_seconds,
                "success": error is None,
                "transient_error": transient,
                "error_type": error_type,
                "status_code": status_code,
            }
        return payload, adjustment

    def _emit_event(self, kind: str, payload: dict[str, Any]) -> None:
        if self._emit is not None:
            self._emit(kind, payload)

    def __call__(
        self, original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> Any:
        admission_id = uuid.uuid4().hex
        requested_at = self._clock()
        with self._lock:
            self._requested += 1
            admitted_payload = {
                **self._capacity_payload_locked(),
                "admission_id": admission_id,
                "time_monotonic": requested_at,
                "queue_wait_seconds": 0.0,
            }
        self._emit_event("api_admission", admitted_payload)
        started_at = self._clock()
        with self._lock:
            self._transport_in_flight += 1
            self._peak_transport_inflight = max(self._peak_transport_inflight, self._transport_in_flight)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            payload, adjustment = self._finish(admission_id, started_at, error)
            try:
                self._emit_event("api_completion", payload)
                if adjustment is not None:
                    self._emit_event("pipeline_concurrency_adjustment", adjustment)
            except BaseException as emit_error:
                error.add_note(
                    "pipeline API completion logging failed: "
                    f"{type(emit_error).__module__}.{type(emit_error).__qualname__}"
                )
            raise
        payload, adjustment = self._finish(admission_id, started_at, None)
        self._emit_event("api_completion", payload)
        if adjustment is not None:
            self._emit_event("pipeline_concurrency_adjustment", adjustment)
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scope": "pipeline",
                "current_limit": self._current_limit,
                "max_limit": self._max_limit,
                "active": len(self._tickets),
                "peak_active": self._peak_active,
                "pending": self._pending,
                "completed": self._completed,
                "failed": self._failed,
                "adjustment_count": self._adjustment_count,
                "adjustments": [dict(event) for event in self._adjustments],
                "model": {
                    "requested": self._requested,
                    "completed": self._model_completed,
                    "errors": self._errors,
                    "transient_errors": self._transient_errors,
                    "transport_in_flight": self._transport_in_flight,
                    "peak_transport_inflight": self._peak_transport_inflight,
                    "service_seconds": {
                        "count": self._model_completed,
                        "total": self._service_total,
                        "max": self._service_max,
                        "samples": list(self._service_samples),
                        "sample_count": len(self._service_samples),
                    },
                },
            }
