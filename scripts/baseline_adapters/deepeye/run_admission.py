"""Bound actual DeepEye model transports and PostgreSQL executions.

The PostgreSQL gate observes Python-call success or failure only.  A SQL error
returned as an ordinary value is therefore a successful transport from this
module's perspective.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable
import uuid


@dataclass(frozen=True)
class AdaptivePolicy:
    initial_limit: int = 50
    step: int = 10
    min_limit: int = 10
    max_limit: int = 100
    stable_window_s: float = 60.0
    min_successes: int = 50
    failure_threshold: int = 5
    failure_rate: float = 0.10
    demand_utilization: float = 0.80
    adjustment_cooldown_s: float = 60.0

    def __post_init__(self) -> None:
        integer_fields = (
            "initial_limit", "step", "min_limit", "max_limit",
            "min_successes", "failure_threshold",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in integer_fields
        ):
            raise ValueError("integer policy controls must be positive integers")
        if not self.min_limit <= self.initial_limit <= self.max_limit:
            raise ValueError("initial_limit must be within min_limit and max_limit")
        if (
            isinstance(self.stable_window_s, bool)
            or not isinstance(self.stable_window_s, (int, float))
            or not math.isfinite(self.stable_window_s)
            or self.stable_window_s <= 0
        ):
            raise ValueError("stable_window_s must be positive")
        if (
            isinstance(self.adjustment_cooldown_s, bool)
            or not isinstance(self.adjustment_cooldown_s, (int, float))
            or not math.isfinite(self.adjustment_cooldown_s)
            or self.adjustment_cooldown_s < 0
        ):
            raise ValueError("adjustment_cooldown_s must not be negative")
        if not 0 < self.failure_rate <= 1:
            raise ValueError("failure_rate must be in (0, 1]")
        if not 0 < self.demand_utilization <= 1:
            raise ValueError("demand_utilization must be in (0, 1]")


class AdaptiveAdmission:
    def __init__(
        self,
        policy: AdaptivePolicy | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or AdaptivePolicy()
        if emit is not None and not callable(emit):
            raise TypeError("emit must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._emit = emit
        self._clock = clock
        self._condition = threading.Condition()
        self._current_limit = self.policy.initial_limit
        self._requested = 0
        self._completed = 0
        self._errors = 0
        self._transient_errors = 0
        self._in_flight = 0
        self._waiting = 0
        self._peak_inflight = 0
        self._transport_in_flight = 0
        self._peak_transport_inflight = 0
        self._queue_samples: deque[float] = deque(maxlen=1000)
        self._service_samples: deque[float] = deque(maxlen=1000)
        self._adjustments: list[dict[str, Any]] = []
        self._last_change_at = self._clock()
        self._last_adjustment_at: float | None = None
        self._recent_completions: deque[tuple[float, bool, bool]] = deque()
        self._recent_transport: deque[tuple[float, int]] = deque()
        self._recent_queue: deque[float] = deque()

    def _emit_event(self, kind: str, payload: dict[str, Any]) -> None:
        if self._emit is not None:
            self._emit(kind, payload)

    def _admit(self) -> tuple[str, float, dict[str, Any]]:
        requested_at = self._clock()
        admission_id = uuid.uuid4().hex
        with self._condition:
            self._requested += 1
            self._waiting += 1
            if self._in_flight >= self._current_limit:
                self._recent_queue.append(requested_at)
            try:
                self._condition.wait_for(
                    lambda: self._in_flight < self._current_limit
                )
            finally:
                self._waiting -= 1
            admitted_at = self._clock()
            queue_wait = max(0.0, admitted_at - requested_at)
            self._in_flight += 1
            self._peak_inflight = max(self._peak_inflight, self._in_flight)
            self._queue_samples.append(queue_wait)
            payload = {
                "admission_id": admission_id,
                "time_monotonic": admitted_at,
                "queue_wait_seconds": queue_wait,
                "in_flight": self._in_flight,
                "waiting": self._waiting,
                "current_limit": self._current_limit,
            }
        return admission_id, queue_wait, payload

    def _release_unstarted(self) -> None:
        with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    def _start_transport(self) -> float:
        with self._condition:
            started_at = self._clock()
            self._transport_in_flight += 1
            self._peak_transport_inflight = max(
                self._peak_transport_inflight, self._transport_in_flight
            )
            self._recent_transport.append(
                (started_at, self._transport_in_flight)
            )
            return started_at

    @staticmethod
    def _status_code(error: BaseException | None) -> int | None:
        if error is None:
            return None
        status = getattr(error, "status_code", None)
        if type(status) is not int:
            status = getattr(getattr(error, "response", None), "status_code", None)
        return status if type(status) is int else None

    @classmethod
    def _is_transient(cls, error: BaseException | None) -> bool:
        if error is None:
            return False
        status = cls._status_code(error)
        if status == 429 or (status is not None and 500 <= status <= 599):
            return True
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        names = {base.__name__.casefold() for base in type(error).__mro__}
        return any(
            "timeout" in name
            or name in {"connecterror", "connectionerror", "apiconnectionerror"}
            for name in names
        )

    def _finish(
        self,
        admission_id: str,
        transport_started_at: float,
        queue_wait: float,
        error: BaseException | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        transient = self._is_transient(error)
        with self._condition:
            finished_at = self._clock()
            service_seconds = max(0.0, finished_at - transport_started_at)
            self._transport_in_flight -= 1
            self._in_flight -= 1
            self._completed += 1
            if error is not None:
                self._errors += 1
                if transient:
                    self._transient_errors += 1
            self._recent_completions.append(
                (finished_at, error is None, transient)
            )
            self._service_samples.append(service_seconds)
            adjustment = self._maybe_adjust_locked(
                finished_at, admission_id, transient
            )
            payload = {
                "admission_id": admission_id,
                "time_monotonic": finished_at,
                "transport_started_at": transport_started_at,
                "finished_at": finished_at,
                "queue_wait_seconds": queue_wait,
                "service_seconds": service_seconds,
                "success": error is None,
                "transient_error": transient,
                "error_type": (
                    f"{type(error).__module__}.{type(error).__qualname__}"
                    if error is not None else None
                ),
                "status_code": self._status_code(error),
                "in_flight": self._in_flight,
                "transport_in_flight": self._transport_in_flight,
                "peak_transport_inflight": self._peak_transport_inflight,
                "waiting": self._waiting,
                "current_limit": self._current_limit,
            }
            self._condition.notify_all()
        return payload, adjustment

    def _maybe_adjust_locked(
        self, now: float, admission_id: str, completed_transient: bool
    ) -> dict[str, Any] | None:
        cutoff = now - self.policy.stable_window_s
        while self._recent_completions and self._recent_completions[0][0] < cutoff:
            self._recent_completions.popleft()
        while self._recent_transport and self._recent_transport[0][0] < cutoff:
            self._recent_transport.popleft()
        while self._recent_queue and self._recent_queue[0] < cutoff:
            self._recent_queue.popleft()
        recent_completed = len(self._recent_completions)
        recent_successes = sum(success for _, success, _ in self._recent_completions)
        recent_transient_errors = sum(
            transient for _, _, transient in self._recent_completions
        )
        recent_peak_transport = max(
            (peak for _, peak in self._recent_transport), default=0
        )
        cooldown_ready = (
            self._last_adjustment_at is None
            or now - self._last_adjustment_at >= self.policy.adjustment_cooldown_s
        )
        failure_fraction = (
            recent_transient_errors / recent_completed
            if recent_completed else 0.0
        )
        new_limit: int | None = None
        reason: str | None = None
        if (
            completed_transient
            and cooldown_ready
            and self._current_limit > self.policy.min_limit
            and recent_transient_errors >= self.policy.failure_threshold
            and failure_fraction >= self.policy.failure_rate
        ):
            new_limit = max(
                self.policy.min_limit, self._current_limit - self.policy.step
            )
            reason = "transient_failures"
        elif (
            cooldown_ready
            and self._current_limit < self.policy.max_limit
            and now - self._last_change_at >= self.policy.stable_window_s
            and recent_successes >= self.policy.min_successes
            and recent_transient_errors == 0
            and recent_peak_transport
            >= self.policy.demand_utilization * self._current_limit
        ):
            new_limit = min(
                self.policy.max_limit, self._current_limit + self.policy.step
            )
            reason = "stable_demand"
        if new_limit is None or new_limit == self._current_limit:
            return None

        old_limit = self._current_limit
        adjustment = {
            "admission_id": admission_id,
            "time_monotonic": now,
            "old_limit": old_limit,
            "new_limit": new_limit,
            "current_limit": new_limit,
            "direction": "increase" if new_limit > old_limit else "decrease",
            "reason": reason,
            "window_seconds": min(
                self.policy.stable_window_s,
                max(0.0, now - self._last_change_at),
            ),
            "window_completed": recent_completed,
            "window_successes": recent_successes,
            "window_transient_errors": recent_transient_errors,
            "window_transient_error_rate": failure_fraction,
            "window_peak_transport_inflight": recent_peak_transport,
            "window_had_queue": bool(self._recent_queue),
            "in_flight": self._in_flight,
            "waiting": self._waiting,
        }
        self._current_limit = new_limit
        self._last_adjustment_at = now
        self._adjustments.append(adjustment)
        self._last_change_at = now
        self._recent_completions.clear()
        self._recent_transport.clear()
        self._recent_queue.clear()
        if self._transport_in_flight:
            self._recent_transport.append((now, self._transport_in_flight))
        return dict(adjustment)

    def __call__(
        self,
        original: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        admission_id, queue_wait, admitted_payload = self._admit()
        try:
            self._emit_event("api_admission", admitted_payload)
        except BaseException:
            self._release_unstarted()
            raise

        transport_started_at = self._start_transport()
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            completed_payload, adjustment = self._finish(
                admission_id, transport_started_at, queue_wait, error
            )
            try:
                self._emit_event("api_completion", completed_payload)
                if adjustment is not None:
                    self._emit_event("api_concurrency_adjustment", adjustment)
            except BaseException as emit_error:
                error.add_note(
                    "admission completion logging failed: "
                    f"{type(emit_error).__module__}.{type(emit_error).__qualname__}"
                )
            raise

        completed_payload, adjustment = self._finish(
            admission_id, transport_started_at, queue_wait, None
        )
        self._emit_event("api_completion", completed_payload)
        if adjustment is not None:
            self._emit_event("api_concurrency_adjustment", adjustment)
        return result

    @staticmethod
    def _metric_snapshot(samples: deque[float]) -> dict[str, Any]:
        values = list(samples)
        return {
            "count": len(values),
            "total": sum(values),
            "max": max(values, default=0.0),
            "samples": values,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "requested": self._requested,
                "completed": self._completed,
                "errors": self._errors,
                "transient_errors": self._transient_errors,
                "peak_inflight": self._peak_inflight,
                "in_flight": self._in_flight,
                "in_flight_semantics": "admitted_permits_including_admission_logging",
                "transport_in_flight": self._transport_in_flight,
                "peak_transport_inflight": self._peak_transport_inflight,
                "waiting": self._waiting,
                "current_limit": self._current_limit,
                "adjustments": [dict(item) for item in self._adjustments],
                "queue_wait_seconds": self._metric_snapshot(self._queue_samples),
                "service_seconds": self._metric_snapshot(self._service_samples),
            }


class FixedAdmission(AdaptiveAdmission):
    """A non-adaptive gate for the actual PostgreSQL execution boundary."""

    def __init__(
        self,
        limit: int,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        prefix: str = "postgres",
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        self.prefix = prefix
        super().__init__(
            AdaptivePolicy(
                initial_limit=limit,
                step=1,
                min_limit=limit,
                max_limit=limit,
            ),
            emit=emit,
            clock=clock,
        )

    def _emit_event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind.startswith("api_"):
            kind = f"{self.prefix}_{kind.removeprefix('api_')}"
        super()._emit_event(kind, payload)

    def snapshot(self) -> dict[str, Any]:
        return {"prefix": self.prefix, **super().snapshot()}
