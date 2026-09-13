"""Fine-grained, reversible tracing for the unmodified DeepEye runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import contextvars
import base64
import dataclasses
import datetime as dt
import enum
import functools
import inspect
import json
import math
import threading
import time
from typing import Any, Callable, Iterator
import uuid

try:
    import numpy as np
except ImportError:  # pragma: no cover - DeepEye's runtime installs NumPy.
    np = None

try:
    from psycopg.types.multirange import Multirange as PostgresMultirange
    from psycopg.types.range import Range as PostgresRange
except ImportError:  # pragma: no cover - the adapter's PostgreSQL extra installs it.
    PostgresMultirange = None
    PostgresRange = None

from .run_store import to_jsonable


_ATTEMPT_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "deepeye_trace_attempt_id", default=None
)
_BRANCH_PATH: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "deepeye_trace_branch_path", default=()
)
_COMPONENT_CALL_IDS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "deepeye_trace_component_call_ids", default=()
)
_API_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "deepeye_trace_api_call_id", default=None
)
_INSTALL_LOCK = threading.Lock()
_ACTIVE_RECORDER: "TraceRecorder | None" = None
_CREDENTIAL_FIELDS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "passwd",
    "access_token",
    "refresh_token",
}


def _is_data_item(value: Any) -> bool:
    return (
        hasattr(value, "question")
        and hasattr(value, "database_id")
        and (
            hasattr(value, "gold_sql")
            or any(cls.__name__ == "DataItem" for cls in type(value).__mro__)
        )
    )


def _data_item_summary(value: Any) -> dict[str, Any]:
    fields = ("question_id", "instance_id", "database_id", "question", "evidence")
    return {
        key: getattr(value, key)
        for key in fields
        if hasattr(value, key)
    }


def _prepare_value(value: Any) -> Any:
    """Normalize runtime objects to portable trace data before store tagging."""

    if _is_data_item(value):
        return _prepare_value(_data_item_summary(value))
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "NaN"
        elif value > 0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return {"nonfinite_float": label}
    if isinstance(value, dt.timedelta):
        return {
            "trace_value_type": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, memoryview):
        return {
            "trace_value_type": "memoryview",
            "base64": base64.b64encode(value.tobytes()).decode("ascii"),
            "format": value.format,
            "itemsize": value.itemsize,
            "ndim": value.ndim,
            "shape": list(value.shape) if value.shape is not None else None,
            "strides": list(value.strides) if value.strides is not None else None,
            "readonly": value.readonly,
        }
    if PostgresRange is not None and isinstance(value, PostgresRange):
        return {
            "trace_value_type": "postgres_range",
            "empty": value.isempty,
            "bounds": value.bounds,
            "lower": _prepare_value(value.lower),
            "upper": _prepare_value(value.upper),
        }
    if PostgresMultirange is not None and isinstance(value, PostgresMultirange):
        return {
            "trace_value_type": "postgres_multirange",
            "ranges": [_prepare_value(item) for item in value],
        }
    if isinstance(value, enum.Enum):
        return _prepare_value(value.value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return _prepare_value(value.tolist())
        if isinstance(value, np.generic):
            return _prepare_value(value.item())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _prepare_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            fields = model_dump(mode="python")
        except TypeError:
            fields = model_dump()
        return _prepare_value(fields)
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict) and any(
        cls.__module__.startswith("pydantic") for cls in type(value).__mro__
    ):
        return _prepare_value(legacy_dict())
    if isinstance(value, dict):
        return {key: _prepare_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_prepare_value(item) for item in value]
    if isinstance(value, tuple):
        return [_prepare_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_prepare_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(
            to_jsonable(item), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ))
        return items
    return value


def _callable_reference(value: Callable[..., Any]) -> dict[str, str]:
    return {
        "module": getattr(value, "__module__", type(value).__module__),
        "qualname": getattr(value, "__qualname__", type(value).__qualname__),
    }


def _exception_payload(error: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": {
            "module": type(error).__module__,
            "qualname": type(error).__qualname__,
        },
        "args": error.args,
    }
    for name in ("status_code", "body", "request_id"):
        if hasattr(error, name):
            payload[name] = getattr(error, name)
    return payload


class TraceRecorder:
    """Persist logical SDK, SQL, and native branch calls under attempt context.

    An ``api_request`` represents one invocation of the SDK ``create`` method.
    Transport retries performed internally by that SDK are not separately visible.
    """

    def __init__(self, store, secrets=(), api_call=None, stop_event=None):
        self.store = store
        self.secrets = tuple(
            str(secret) for secret in secrets if secret is not None and str(secret)
        )
        if api_call is not None and not callable(api_call):
            raise TypeError("api_call must be callable")
        self.api_call = api_call
        self.error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._instrument_lock = threading.RLock()
        self._instrumented: dict[tuple[int, str], dict[str, Any]] = {}
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        from .run_sampling import SamplingCheckpoints
        self.sampling_checkpoints = SamplingCheckpoints(store, self._remember_error) if hasattr(store, 'attempt') else None

    @contextmanager
    def context(self, attempt_id: str) -> Iterator[None]:
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt_id must be a non-empty string")
        attempt_token = _ATTEMPT_ID.set(attempt_id)
        branch_token = _BRANCH_PATH.set(())
        component_token = _COMPONENT_CALL_IDS.set(())
        session = None
        try:
            from app.llm.sampling import observe_sampling, sampling_checkpoints
            from .run_sampling import stable_sampling_node
            session = self.sampling_checkpoints.session(attempt_id, self) if self.sampling_checkpoints else None
            checkpoint_context = sampling_checkpoints(session) if session else nullcontext()
            with observe_sampling(self.record_sampling), checkpoint_context, stable_sampling_node('stage'):
                yield
        finally:
            if session is not None:
                session.close()
            _COMPONENT_CALL_IDS.reset(component_token)
            _BRANCH_PATH.reset(branch_token)
            _ATTEMPT_ID.reset(attempt_token)

    @contextmanager
    def _branch(self, branch: str) -> Iterator[None]:
        token = _BRANCH_PATH.set((*_BRANCH_PATH.get(), branch))
        try:
            yield
        finally:
            _BRANCH_PATH.reset(token)

    @contextmanager
    def _component_call(self, branch: str, call_id: str) -> Iterator[None]:
        branch_token = _BRANCH_PATH.set((*_BRANCH_PATH.get(), branch))
        component_token = _COMPONENT_CALL_IDS.set(
            (*_COMPONENT_CALL_IDS.get(), call_id)
        )
        try:
            from .run_sampling import stable_sampling_node
            with stable_sampling_node(branch):
                yield
        finally:
            _COMPONENT_CALL_IDS.reset(component_token)
            _BRANCH_PATH.reset(branch_token)

    def _remember_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self.error is None:
                self.error = error

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self.error
        if error is not None:
            raise RuntimeError("trace storage failed") from error

    def _redact(self, value: Any, *, field_name: str | None = None) -> Any:
        if field_name is not None and field_name.casefold() in _CREDENTIAL_FIELDS:
            return "[REDACTED]"
        if isinstance(value, str):
            redacted = value
            for secret in self.secrets:
                redacted = redacted.replace(secret, "[REDACTED]")
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._redact(item, field_name=key if isinstance(key, str) else None)
                for key, item in value.items()
            }
        return value

    def _serialize(self, value: Any) -> Any:
        return self._redact(to_jsonable(_prepare_value(value)))

    def _append(self, kind: str, payload: dict[str, Any]) -> None:
        attempt_id = _ATTEMPT_ID.get()
        if attempt_id is None:
            return
        with self._error_lock:
            prior_error = self.error
        if prior_error is not None:
            raise prior_error
        try:
            serialized = self._serialize(payload)
            self.store.append_event(attempt_id, kind, serialized)
        except BaseException as error:
            self._remember_error(error)
            raise

    @staticmethod
    def _base_payload() -> dict[str, Any]:
        from app.llm.sampling import sampling_identity
        component_ids = _COMPONENT_CALL_IDS.get()
        return {
            **sampling_identity(),
            "branch_path": list(_BRANCH_PATH.get()),
            "component_call_id": component_ids[-1] if component_ids else None,
        }

    def record_sampling(self, kind: str, payload: dict[str, Any]) -> None:
        """Persist sample outcomes, including failures swallowed by native code."""
        self._append(kind, {**self._base_payload(), **payload})

    def record_admission(self, kind: str, payload: dict[str, Any]) -> None:
        """Record gate telemetry with recorder-owned execution linkage."""

        if not isinstance(kind, str) or not kind:
            raise ValueError("admission event kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("admission event payload must be a dictionary")
        call_id = _API_CALL_ID.get()
        if kind.startswith("api_") and call_id is None:
            raise RuntimeError("api admission event requires an active API call")
        self._append(kind, {
            **payload,
            **self._base_payload(),
            "call_id": call_id,
        })

    def _api_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def traced(*args, **kwargs):
            if _ATTEMPT_ID.get() is None:
                return original(*args, **kwargs)
            from app.llm.sampling import check_sampling_stop
            check_sampling_stop()
            call_id = uuid.uuid4().hex
            self._append("api_request", {
                **self._base_payload(),
                "call_id": call_id,
                "logical_sdk_call": True,
                "args": args,
                "kwargs": kwargs,
            })
            try:
                call_token = _API_CALL_ID.set(call_id)
                try:
                    if self.api_call is None:
                        response = original(*args, **kwargs)
                    else:
                        response = self.api_call(original, args, kwargs)
                finally:
                    _API_CALL_ID.reset(call_token)
            except BaseException as error:
                try:
                    from app.llm.sampling import error_usage
                    self._append("api_error", {
                        **self._base_payload(),
                        "call_id": call_id,
                        "error": _exception_payload(error),
                        "response": {"usage": error_usage(error)},
                    })
                except BaseException:
                    pass
                raise
            self._append("api_response", {
                **self._base_payload(),
                "call_id": call_id,
                "response": response,
            })
            return response

        return traced

    def _component_inputs(
        self, original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            arguments = dict(inspect.signature(original).bind_partial(*args, **kwargs).arguments)
        except (TypeError, ValueError):
            arguments = {"args": args, "kwargs": kwargs}
        arguments.pop("llm", None)
        parser = arguments.get("rule_parser")
        if callable(parser):
            arguments["rule_parser"] = _callable_reference(parser)
        return arguments

    def _component_wrapper(
        self, original: Callable[..., Any], component: str
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def traced(*args, **kwargs):
            if _ATTEMPT_ID.get() is None:
                return original(*args, **kwargs)
            from app.llm.sampling import check_sampling_stop
            check_sampling_stop()
            component_call_id = uuid.uuid4().hex
            component_ids = _COMPONENT_CALL_IDS.get()
            parent_component_call_id = component_ids[-1] if component_ids else None
            started_at = time.perf_counter()
            with self._component_call(component, component_call_id):
                inputs = self._component_inputs(original, args, kwargs)
                start_payload = {
                    **self._base_payload(),
                    "component": component,
                    "parent_component_call_id": parent_component_call_id,
                    "inputs": inputs,
                }
                if "sql" in inputs:
                    start_payload["input_sql"] = inputs["sql"]
                self._append("component_start", start_payload)
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    try:
                        self._append("component_error", {
                            **self._base_payload(),
                            "component": component,
                            "parent_component_call_id": parent_component_call_id,
                            "elapsed_seconds": time.perf_counter() - started_at,
                            "error": _exception_payload(error),
                        })
                    except BaseException:
                        pass
                    raise
                result_payload = {
                    **self._base_payload(),
                    "component": component,
                    "parent_component_call_id": parent_component_call_id,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "result": result,
                }
                if "sql" in inputs:
                    result_payload["input_sql"] = inputs["sql"]
                    if isinstance(result, tuple) and result:
                        result_payload["output_sql"] = result[0]
                if component == "selection.pairwise_comparison":
                    result_payload["votes"] = result[0] if isinstance(result, tuple) else result
                self._append("component_result", result_payload)
                return result

        return traced

    def _instrument_method(
        self, owner: Any, name: str, component: str, *, api: bool = False
    ) -> tuple[int, str]:
        key = (id(owner), name)
        with self._instrument_lock:
            existing = self._instrumented.get(key)
            if existing is not None:
                if existing["owner"] is not owner:
                    raise RuntimeError("Trace instrumentation identity collision")
                existing["references"] += 1
                return key
            original = getattr(owner, name)
            instance_values = getattr(owner, "__dict__", {})
            had_instance_value = name in instance_values
            prior_instance_value = instance_values.get(name)
            wrapper = self._api_wrapper(original) if api else self._component_wrapper(
                original, component
            )
            setattr(owner, name, wrapper)
            self._instrumented[key] = {
                "owner": owner,
                "name": name,
                "had_instance_value": had_instance_value,
                "prior_instance_value": prior_instance_value,
                "references": 1,
            }
        return key

    def _release_method(self, key: tuple[int, str]) -> None:
        with self._instrument_lock:
            state = self._instrumented.get(key)
            if state is None:
                return
            state["references"] -= 1
            if state["references"]:
                return
            owner = state["owner"]
            name = state["name"]
            if state["had_instance_value"]:
                setattr(owner, name, state["prior_instance_value"])
            else:
                delattr(owner, name)
            del self._instrumented[key]

    def _instrument_component(
        self, owner: Any, method: str, component: str, keys: list[tuple[int, str]]
    ) -> None:
        keys.append(self._instrument_method(owner, method, component))
        extractor = getattr(owner, "_extractor", None)
        if extractor is not None and hasattr(extractor, "extract_with_retry"):
            keys.append(self._instrument_method(
                extractor, "extract_with_retry", f"{component}.extraction"
            ))

    def instrument_runner(self, runner, stage) -> Callable[[], None]:
        """Instrument one runner and return an idempotent restoration callback."""

        stage_components = {
            "value_retrieval": (),
            "schema_linking": (
                ("_direct_linker", "link", "schema_linking.direct"),
                ("_reversed_linker", "link", "schema_linking.reversed"),
                ("_value_linker", "link", "schema_linking.value"),
            ),
            "sql_generation": (
                ("_dc_generator", "generate", "generation.dc"),
                ("_skeleton_generator", "generate", "generation.skeleton"),
                ("_icl_generator", "generate", "generation.icl"),
            ),
            "sql_revision": tuple(
                (None, "check_and_revise", f"revision.{type(checker).__name__}")
                for checker in getattr(runner, "_checkers", ())
            ),
            "sql_selection": ((None, "_compare_sqls", "selection.pairwise_comparison"),),
        }
        if stage not in stage_components:
            raise ValueError(f"Unknown DeepEye stage: {stage!r}")
        keys: list[tuple[int, str]] = []
        cleaned = False
        try:
            client = runner._llm._get_client()
            keys.append(self._instrument_method(
                client.chat.completions, "create", "api.chat", api=True
            ))
            if stage == "value_retrieval":
                extractor = runner._keyword_extractor
                keys.append(self._instrument_method(
                    extractor, "extract_with_retry", "value_retrieval.keyword_extraction"
                ))
            elif stage == "sql_revision":
                if hasattr(runner, "_revise_one_candidate"):
                    keys.append(self._instrument_method(
                        runner, "_revise_one_candidate", "revision.candidate"
                    ))
                for checker, (_, method, component) in zip(
                    runner._checkers, stage_components[stage]
                ):
                    self._instrument_component(checker, method, component, keys)
            else:
                for owner_name, method, component in stage_components[stage]:
                    owner = runner if owner_name is None else getattr(runner, owner_name)
                    self._instrument_component(owner, method, component, keys)
                if stage == "sql_selection":
                    for method, component in (
                        ("_get_top_k_sql_candidates", "selection.shortlist"),
                        ("_get_pair_sqls_to_eval", "selection.pairs"),
                        ("_compute_robust_win_matrix", "selection.win_matrix"),
                    ):
                        if hasattr(runner, method):
                            keys.append(self._instrument_method(
                                runner, method, component
                            ))
        except BaseException:
            for key in reversed(keys):
                self._release_method(key)
            raise

        def cleanup() -> None:
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            for key in reversed(keys):
                self._release_method(key)

        return cleanup

    def _execution_wrapper(
        self, original: Callable[..., Any], operation: str, branch: str
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def traced(service, data_item, sql, *args, **kwargs):
            if _ATTEMPT_ID.get() is None:
                return original(service, data_item, sql, *args, **kwargs)
            with self._branch(branch):
                sql_call_id = uuid.uuid4().hex
                started_at = time.perf_counter()
                common_payload = {
                    **self._base_payload(),
                    "sql_call_id": sql_call_id,
                    "data_item": data_item,
                    "sql": sql,
                    "args": args,
                    "kwargs": kwargs,
                }
                self._append(f"{operation}_start", common_payload)
                try:
                    result = original(service, data_item, sql, *args, **kwargs)
                except BaseException as error:
                    try:
                        self._append(f"{operation}_error", {
                            **common_payload,
                            "elapsed_seconds": time.perf_counter() - started_at,
                            "error": _exception_payload(error),
                        })
                    except BaseException:
                        pass
                    raise
                self._append(f"{operation}_result", {
                    **common_payload,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "result": result,
                })
                return result

        return traced

    @contextmanager
    def install(self) -> Iterator["TraceRecorder"]:
        """Exclusively install reversible global executor and SQL tracing patches."""

        global _ACTIVE_RECORDER
        with _INSTALL_LOCK:
            if _ACTIVE_RECORDER is not None:
                raise RuntimeError("TraceRecorder is already installed")
            _ACTIVE_RECORDER = self

        from app.services.execution_service import ExecutionService

        original_submit = ThreadPoolExecutor.submit
        original_execute = ExecutionService.execute
        original_measure_time = ExecutionService.measure_time

        @functools.wraps(original_submit)
        def submit_with_context(executor, function, /, *args, **kwargs):
            context = contextvars.copy_context()
            return original_submit(executor, context.run, function, *args, **kwargs)

        try:
            ThreadPoolExecutor.submit = submit_with_context
            ExecutionService.execute = self._execution_wrapper(
                original_execute, "sql_execute", "execution.execute"
            )
            ExecutionService.measure_time = self._execution_wrapper(
                original_measure_time, "sql_measure_time", "execution.measure_time"
            )
            yield self
        finally:
            ThreadPoolExecutor.submit = original_submit
            ExecutionService.execute = original_execute
            ExecutionService.measure_time = original_measure_time
            with _INSTALL_LOCK:
                _ACTIVE_RECORDER = None
