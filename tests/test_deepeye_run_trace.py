"""Offline integration tests for fine-grained DeepEye tracing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import datetime as dt
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import BaseModel

BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "DeepEye-SQL"
sys.path.insert(0, str(BASELINE))

from scripts.baseline_adapters.deepeye.run_store import RunStore


class EventStore:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.rows = []
        self.lock = threading.Lock()

    def append_event(self, attempt_id, kind, payload):
        if self.error is not None:
            raise self.error
        with self.lock:
            self.rows.append({
                "attempt_id": attempt_id,
                "kind": kind,
                "payload": payload,
            })
            return len(self.rows)

    def events(self, attempt_id=None):
        with self.lock:
            return [
                row for row in self.rows
                if attempt_id is None or row["attempt_id"] == attempt_id
            ]


class FakeCompletions:
    def __init__(self):
        self.requests = []
        self.responses = []

    def create(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        response = {
            "id": f"response-{len(self.requests)}",
            "choices": [{"message": {"content": "<result>SELECT 1</result>"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "reasoning": {"content": "complete reasoning payload"},
        }
        self.responses.append(response)
        return response


class FakeLLM:
    def __init__(self, completions):
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

    def _get_client(self):
        return self.client


class FakeGenerator:
    def generate(self, data_item, llm, sampling_budget=1):
        response = llm._get_client().chat.completions.create(
            model="fixture-model",
            messages=[{"role": "user", "content": "same prompt"}],
            api_key="credential-value",
            extra_body={"private": data_item.question},
        )
        return [response["choices"][0]["message"]["content"]], {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }


@dataclass
class PortableUsage:
    prompt_tokens: int
    completion_tokens: int


class PortableResponse(BaseModel):
    id: str
    usage: PortableUsage
    details: dict[str, object]


def data_item(instance_id="q1", question="normal question"):
    return SimpleNamespace(
        question_id=7,
        instance_id=instance_id,
        database_id="db",
        database_path="db",
        db_type="postgresql",
        question=question,
        evidence="evidence",
        gold_sql="SELECT forbidden_gold",
    )


class TraceRecorderTests(unittest.TestCase):
    def setUp(self):
        module_name = "scripts.baseline_adapters.deepeye.run_trace"
        self.assertIsNotNone(
            importlib.util.find_spec(module_name),
            "fine-grained DeepEye trace recorder is not implemented",
        )
        self.module = importlib.import_module(module_name)

    @staticmethod
    def runner(completions):
        return SimpleNamespace(
            _llm=FakeLLM(completions),
            _dc_generator=FakeGenerator(),
            _skeleton_generator=FakeGenerator(),
            _icl_generator=FakeGenerator(),
        )

    def test_real_thread_pool_propagates_attempt_and_nested_branch_context(self):
        """Losing contextvars in native executors or mixing equal prompts must fail."""
        store = EventStore()
        completions = FakeCompletions()
        runner = self.runner(completions)
        recorder = self.module.TraceRecorder(store)
        original_create = completions.create
        cleanup = recorder.instrument_runner(runner, "sql_generation")
        try:
            with recorder.install(), ThreadPoolExecutor(max_workers=2) as pool:
                with recorder.context("attempt-a"):
                    first = pool.submit(
                        runner._dc_generator.generate,
                        data_item("a"), runner._llm, 1,
                    )
                with recorder.context("attempt-b"):
                    second = pool.submit(
                        runner._skeleton_generator.generate,
                        data_item("b"), runner._llm, 1,
                    )
                first_result = first.result()
                second_result = second.result()
                outside = completions.create(
                    model="fixture-model", messages=[{"content": "same prompt"}]
                )
        finally:
            cleanup()

        self.assertEqual(first_result[0], ["<result>SELECT 1</result>"])
        self.assertEqual(second_result[0], ["<result>SELECT 1</result>"])
        self.assertEqual(outside["usage"]["total_tokens"], 5)
        self.assertEqual({row["attempt_id"] for row in store.rows}, {"attempt-a", "attempt-b"})
        for attempt, branch in (("attempt-a", "generation.dc"),
                                ("attempt-b", "generation.skeleton")):
            events = store.events(attempt)
            self.assertEqual(
                [event["kind"] for event in events],
                ["component_start", "api_request", "api_response", "component_result"],
            )
            self.assertEqual(events[0]["payload"]["branch_path"], [branch])
            self.assertEqual(events[1]["payload"]["branch_path"], [branch])
            component_call_id = events[0]["payload"]["component_call_id"]
            self.assertEqual(
                events[-1]["payload"]["component_call_id"], component_call_id
            )
            self.assertIsNone(events[0]["payload"]["parent_component_call_id"])
            self.assertEqual(events[1]["payload"]["component_call_id"], component_call_id)
            self.assertGreaterEqual(events[-1]["payload"]["elapsed_seconds"], 0)
            self.assertEqual(
                events[1]["payload"]["call_id"],
                events[2]["payload"]["call_id"],
            )
        self.assertNotEqual(
            store.events("attempt-a")[0]["payload"]["component_call_id"],
            store.events("attempt-b")[0]["payload"]["component_call_id"],
        )
        self.assertEqual(completions.create, original_create)

    def test_nested_extractor_has_parent_id_and_api_uses_leaf_component_id(self):
        """Repeated nested calls need IDs, not branch names, for provenance."""
        class Extractor:
            def extract_with_retry(self, text, llm):
                return llm._get_client().chat.completions.create(
                    model="fixture-model", messages=[{"content": text}]
                )["id"]

        class Generator:
            def __init__(self):
                self._extractor = Extractor()

            def generate(self, data_item, llm, sampling_budget=1):
                return self._extractor.extract_with_retry(data_item.question, llm)

        completions = FakeCompletions()
        generator = Generator()
        runner = SimpleNamespace(
            _llm=FakeLLM(completions),
            _dc_generator=generator,
            _skeleton_generator=FakeGenerator(),
            _icl_generator=FakeGenerator(),
        )
        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(runner, "sql_generation")
        try:
            with recorder.context("attempt"):
                actual = generator.generate(data_item(), runner._llm, 1)
        finally:
            cleanup()

        self.assertEqual(actual, "response-1")
        generation_start = store.rows[0]["payload"]
        extraction_start = store.rows[1]["payload"]
        api_request = store.rows[2]["payload"]
        extraction_result = store.rows[4]["payload"]
        generation_result = store.rows[5]["payload"]
        generation_id = generation_start["component_call_id"]
        extraction_id = extraction_start["component_call_id"]
        self.assertNotEqual(generation_id, extraction_id)
        self.assertIsNone(generation_start["parent_component_call_id"])
        self.assertEqual(extraction_start["parent_component_call_id"], generation_id)
        self.assertEqual(api_request["component_call_id"], extraction_id)
        self.assertEqual(extraction_result["component_call_id"], extraction_id)
        self.assertEqual(generation_result["component_call_id"], generation_id)

    def test_api_records_before_transport_redacts_copy_and_preserves_response(self):
        """Mutating requests, omitting full responses, or bypassing api_call must fail."""
        store = EventStore()
        completions = FakeCompletions()
        runner = self.runner(completions)
        calls = []

        def api_call(original, args, kwargs):
            calls.append((original, args, kwargs))
            self.assertEqual(store.rows[-1]["kind"], "api_request")
            return original(*args, **kwargs)

        recorder = self.module.TraceRecorder(
            store, secrets=("normal-secret",), api_call=api_call
        )
        cleanup = recorder.instrument_runner(runner, "sql_generation")
        try:
            with recorder.context("attempt"):
                response = completions.create(
                    model="fixture-model",
                    messages=[{"role": "user", "content": "keep normal-secret safe"}],
                    credential="credential-value",
                )
        finally:
            cleanup()

        self.assertIs(response, completions.responses[0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            completions.requests[0][1]["messages"][0]["content"],
            "keep normal-secret safe",
        )
        request_payload = store.rows[0]["payload"]
        serialized = json.dumps(request_payload, ensure_ascii=False)
        self.assertNotIn("__run_store_type__", serialized)
        self.assertIn("[REDACTED]", serialized)
        self.assertNotIn("normal-secret", serialized)
        self.assertNotIn("credential-value", serialized)
        self.assertEqual(store.rows[1]["payload"]["response"], response)

    def test_concurrent_admission_events_use_their_api_call_linkage(self):
        """A shared gate must not cross-link calls or replace recorder-owned context."""
        class ConcurrentCompletions:
            def __init__(self):
                self.barrier = threading.Barrier(2)

            def create(self, label, /, *, suffix="done"):
                self.barrier.wait()
                return f"{label}-{suffix}"

        completions = ConcurrentCompletions()
        runner = self.runner(completions)
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {"fixture": "admission"}) as store:
                attempt_id = store.begin_attempt("q", "generation", "fp")
                recorder = self.module.TraceRecorder(store, secrets=("gate-secret",))

                def gated_call(original, args, kwargs):
                    label = args[0]
                    admission_id = f"admission-{label}"
                    common = {
                        "admission_id": admission_id,
                        "label": label,
                        "call_id": "untrusted-call-id",
                        "branch_path": ["untrusted-branch"],
                        "component_call_id": "untrusted-component",
                        "credential": "gate-secret",
                    }
                    recorder.record_admission("api_admission", common)
                    try:
                        return original(*args, **kwargs)
                    finally:
                        recorder.record_admission("api_completion", common)

                recorder.api_call = gated_call
                cleanup = recorder.instrument_runner(runner, "sql_generation")
                try:
                    with recorder.install(), recorder.context(attempt_id):
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            futures = [
                                pool.submit(completions.create, label, suffix="ok")
                                for label in ("one", "two")
                            ]
                            self.assertEqual(
                                {future.result() for future in futures},
                                {"one-ok", "two-ok"},
                            )
                        recorder.record_admission(
                            "postgres_admission",
                            {"admission_id": "postgres-one", "call_id": "untrusted"},
                        )
                finally:
                    cleanup()

                events = store.events(attempt_id)

        api_requests = {
            event["payload"]["args"][0]: event["payload"]["call_id"]
            for event in events
            if event["kind"] == "api_request"
        }
        self.assertEqual(set(api_requests), {"one", "two"})
        self.assertNotEqual(api_requests["one"], api_requests["two"])
        admission_events = [
            event for event in events
            if event["kind"] in {"api_admission", "api_completion"}
        ]
        self.assertEqual(len(admission_events), 4)
        for event in admission_events:
            payload = event["payload"]
            self.assertEqual(payload["call_id"], api_requests[payload["label"]])
            self.assertEqual(payload["branch_path"], [])
            self.assertIsNone(payload["component_call_id"])
            self.assertEqual(payload["credential"], "[REDACTED]")
        postgres = next(event for event in events if event["kind"] == "postgres_admission")
        self.assertEqual(postgres["payload"]["admission_id"], "postgres-one")
        self.assertIsNone(postgres["payload"]["call_id"])

    def test_api_hook_preserves_signature_results_errors_and_restores_call_context(self):
        """Leaked API context or a changed callable contract must fail this test."""
        class SignatureCompletions:
            def create(self, value, /, *, fail=False):
                if fail:
                    raise LookupError(f"failed-{value}")
                return {"value": value}

        completions = SignatureCompletions()
        native_signature = inspect.signature(completions.create)
        runner = self.runner(completions)
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                attempt_id = store.begin_attempt("q", "generation", "fp")
                recorder = self.module.TraceRecorder(store)

                def gated_call(original, args, kwargs):
                    admission_id = f"gate-{args[0]}"
                    recorder.record_admission(
                        "api_admission", {"admission_id": admission_id}
                    )
                    try:
                        return original(*args, **kwargs)
                    finally:
                        recorder.record_admission(
                            "api_completion", {"admission_id": admission_id}
                        )

                recorder.api_call = gated_call
                cleanup = recorder.instrument_runner(runner, "sql_generation")
                try:
                    self.assertEqual(inspect.signature(completions.create), native_signature)
                    with recorder.context(attempt_id):
                        self.assertEqual(completions.create("ok"), {"value": "ok"})
                        with self.assertRaisesRegex(LookupError, "failed-bad"):
                            completions.create("bad", fail=True)
                        with self.assertRaisesRegex(RuntimeError, "active API call"):
                            recorder.record_admission(
                                "api_admission", {"admission_id": "leak-check"}
                            )
                finally:
                    cleanup()

                events = store.events(attempt_id)

        for admission_id in ("gate-ok", "gate-bad"):
            linked = [
                event["payload"]["call_id"]
                for event in events
                if event["payload"].get("admission_id") == admission_id
            ]
            self.assertEqual(len(linked), 2)
            self.assertEqual(len(set(linked)), 1)
        self.assertEqual(
            [event["kind"] for event in events],
            [
                "api_request", "api_admission", "api_completion", "api_response",
                "api_request", "api_admission", "api_completion", "api_error",
            ],
        )

    def test_admission_storage_failure_is_sticky_and_skips_transport(self):
        """A gate callback write failure must survive wrapper cleanup and fail closed."""
        class CountingCompletions:
            def __init__(self):
                self.calls = 0

            def create(self):
                self.calls += 1
                return "unreachable"

        completions = CountingCompletions()
        runner = self.runner(completions)
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                attempt_id = store.begin_attempt("q", "generation", "fp")
                recorder = self.module.TraceRecorder(store)

                def fail_during_admission(original, args, kwargs):
                    store.finish_attempt(attempt_id, "failed", {"fixture": True})
                    recorder.record_admission(
                        "api_admission", {"admission_id": "cannot-append"}
                    )
                    return original(*args, **kwargs)

                recorder.api_call = fail_during_admission
                cleanup = recorder.instrument_runner(runner, "sql_generation")
                try:
                    with recorder.context(attempt_id):
                        with self.assertRaisesRegex(ValueError, "after an attempt is finished"):
                            completions.create()
                    self.assertIsInstance(recorder.error, ValueError)
                    with self.assertRaisesRegex(RuntimeError, "trace storage failed"):
                        recorder.raise_if_failed()
                finally:
                    cleanup()

        self.assertEqual(completions.calls, 0)

    def test_model_error_is_recorded_but_not_sticky_after_recovery(self):
        """A recovered model-service error must not become a storage failure."""
        store = EventStore()

        class RecoveringCompletions(FakeCompletions):
            def create(self, *args, **kwargs):
                if not self.requests:
                    self.requests.append((args, kwargs))
                    raise RuntimeError("temporary model outage")
                return super().create(*args, **kwargs)

        completions = RecoveringCompletions()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(self.runner(completions), "sql_generation")
        try:
            with recorder.context("attempt"):
                with self.assertRaisesRegex(RuntimeError, "temporary model outage"):
                    completions.create(model="m", messages=[])
                recovered = completions.create(model="m", messages=[])
            recorder.raise_if_failed()
        finally:
            cleanup()
        self.assertEqual(recovered["usage"]["total_tokens"], 5)
        self.assertEqual(
            [row["kind"] for row in store.rows],
            ["api_request", "api_error", "api_request", "api_response"],
        )

    def test_sdk_pydantic_and_nested_dataclass_are_saved_as_portable_plain_data(self):
        """Trace readers must not need response-model or baseline imports."""
        store = EventStore()
        response = PortableResponse(
            id="portable-response",
            usage=PortableUsage(prompt_tokens=4, completion_tokens=2),
            details={"reasoning": {"summary": "kept in full"}},
        )

        class ModelCompletions:
            def create(self, *args, **kwargs):
                return response

        completions = ModelCompletions()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(self.runner(completions), "sql_generation")
        try:
            with recorder.context("attempt"):
                actual = completions.create(model="fixture-model", messages=[])
        finally:
            cleanup()

        self.assertIs(actual, response)
        saved = store.rows[-1]["payload"]["response"]
        self.assertEqual(saved, {
            "id": "portable-response",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            "details": {"reasoning": {"summary": "kept in full"}},
        })
        serialized = json.dumps(saved, allow_nan=False)
        self.assertNotIn("__run_store_type__", serialized)
        self.assertNotIn("qualname", serialized)
        self.assertNotIn("module", serialized)

    def test_storage_failure_is_sticky_even_when_native_swallows_it(self):
        """Swallowed append failures must remain observable at the engine boundary."""
        failure = OSError("disk full")
        store = EventStore(error=failure)
        completions = FakeCompletions()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(self.runner(completions), "sql_generation")
        try:
            with recorder.context("attempt"):
                try:
                    completions.create(model="m", messages=[])
                except OSError:
                    pass
            self.assertIs(recorder.error, failure)
            with self.assertRaisesRegex(RuntimeError, "trace storage"):
                recorder.raise_if_failed()
            self.assertEqual(completions.requests, [])
        finally:
            cleanup()

    def test_install_records_cached_execution_and_restores_global_methods(self):
        """Cache hits, measurements, overlapping installs, or leaked patches must fail."""
        from app.db_utils.execution import SQLExecutionResult
        from app.services.execution_service import ExecutionService
        from app.services import execution_service as execution_module

        store = EventStore()
        recorder = self.module.TraceRecorder(store, secrets=("forbidden_gold",))
        competing = self.module.TraceRecorder(EventStore())
        original_submit = ThreadPoolExecutor.submit
        original_execute = ExecutionService.execute
        original_measure = ExecutionService.measure_time
        external_calls = []

        def execute_external(item, sql, **kwargs):
            external_calls.append((item, sql, kwargs))
            return SQLExecutionResult(
                result_type="success", db_path="db", sql=sql,
                result_cols=["value"], result_rows=[("alpha",)],
                execution_time=0.25,
            )

        measured = []
        def measure_external(*args, **kwargs):
            measured.append((args, kwargs))
            return 0.5

        service = ExecutionService()
        with patch.object(execution_module, "execute_sql_for_data_item", side_effect=execute_external), \
             patch.object(execution_module, "measure_execution_time_for_data_item", side_effect=measure_external):
            with recorder.install():
                with self.assertRaisesRegex(RuntimeError, "already installed"):
                    with competing.install():
                        pass
                with recorder.context("attempt"):
                    first = service.execute(data_item(), "SELECT value FROM public.items")
                    second = service.execute(data_item(), "SELECT value FROM public.items")
                    first_time = service.measure_time(data_item(), "SELECT value FROM public.items")
                    second_time = service.measure_time(data_item(), "SELECT value FROM public.items")

        self.assertIs(first, second)
        self.assertEqual((first_time, second_time), (0.5, 0.5))
        self.assertEqual(len(external_calls), 1)
        self.assertEqual(len(measured), 1)
        self.assertEqual(
            [row["kind"] for row in store.rows],
            [
                "sql_execute_start", "sql_execute_result",
                "sql_execute_start", "sql_execute_result",
                "sql_measure_time_start", "sql_measure_time_result",
                "sql_measure_time_start", "sql_measure_time_result",
            ],
        )
        for start, result in zip(store.rows[::2], store.rows[1::2]):
            self.assertEqual(
                start["payload"]["sql_call_id"], result["payload"]["sql_call_id"]
            )
        serialized = json.dumps(store.rows, ensure_ascii=False)
        self.assertNotIn("SELECT forbidden_gold", serialized)
        self.assertIn('"instance_id": "q1"', serialized)
        self.assertIs(ThreadPoolExecutor.submit, original_submit)
        self.assertIs(ExecutionService.execute, original_execute)
        self.assertIs(ExecutionService.measure_time, original_measure)

    def test_sql_start_is_durable_before_failure_and_links_current_component(self):
        """A crashing SQL call must leave an attributable in-flight marker."""
        from app.services.execution_service import ExecutionService
        from app.services import execution_service as execution_module

        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        service = ExecutionService()

        class SQLGenerator:
            def generate(self, data_item, llm, sampling_budget=1):
                return service.execute(data_item, "SELECT failing_fixture")

        generator = SQLGenerator()
        runner = SimpleNamespace(
            _llm=FakeLLM(FakeCompletions()),
            _dc_generator=generator,
            _skeleton_generator=FakeGenerator(),
            _icl_generator=FakeGenerator(),
        )
        cleanup = recorder.instrument_runner(runner, "sql_generation")

        def fail_after_start(*args, **kwargs):
            self.assertEqual(store.rows[-1]["kind"], "sql_execute_start")
            raise RuntimeError("postgres unavailable")

        try:
            with patch.object(
                execution_module, "execute_sql_for_data_item", side_effect=fail_after_start
            ):
                with recorder.install(), recorder.context("attempt"):
                    with self.assertRaisesRegex(RuntimeError, "postgres unavailable"):
                        generator.generate(data_item(), runner._llm, 1)
        finally:
            cleanup()

        component_start, sql_start, sql_error, component_error = store.rows
        component_id = component_start["payload"]["component_call_id"]
        self.assertEqual(sql_start["kind"], "sql_execute_start")
        self.assertEqual(sql_error["kind"], "sql_execute_error")
        self.assertEqual(
            sql_start["payload"]["sql_call_id"], sql_error["payload"]["sql_call_id"]
        )
        self.assertEqual(sql_start["payload"]["component_call_id"], component_id)
        self.assertEqual(sql_error["payload"]["component_call_id"], component_id)
        self.assertEqual(component_error["payload"]["component_call_id"], component_id)
        self.assertGreaterEqual(component_error["payload"]["elapsed_seconds"], 0)

    def test_sql_timing_infinity_is_tagged_without_changing_native_return(self):
        """Timing failures legitimately return infinity and must remain transparent."""
        from app.services.execution_service import ExecutionService
        from app.services import execution_service as execution_module

        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        service = ExecutionService()
        with patch.object(
            execution_module,
            "measure_execution_time_for_data_item",
            return_value=float("inf"),
        ):
            with recorder.install(), recorder.context("attempt"):
                actual = service.measure_time(data_item(), "SELECT infinity_fixture")

        self.assertTrue(math.isinf(actual) and actual > 0)
        self.assertEqual(
            store.rows[-1]["payload"]["result"],
            {"nonfinite_float": "Infinity"},
        )
        recorder.raise_if_failed()

    def test_sql_result_portably_tags_interval_bytea_and_postgres_range(self):
        """Valid PostgreSQL result values must never make tracing fail."""
        from app.db_utils.execution import SQLExecutionResult
        from app.services.execution_service import ExecutionService
        from app.services import execution_service as execution_module
        from psycopg.types.range import Range

        interval = dt.timedelta(days=-2, seconds=3, microseconds=4)
        bytea = memoryview(b"\x00\xfftrace")
        pg_range = Range(1, 9, bounds="[)")
        native_result = SQLExecutionResult(
            result_type="success",
            db_path="db",
            sql="SELECT typed_fixture",
            result_cols=["interval", "bytea", "window"],
            result_rows=[(interval, bytea, pg_range)],
            execution_time=0.1,
        )
        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        service = ExecutionService()
        with patch.object(
            execution_module, "execute_sql_for_data_item", return_value=native_result
        ):
            with recorder.install(), recorder.context("attempt"):
                actual = service.execute(data_item(), "SELECT typed_fixture")

        self.assertIs(actual, native_result)
        values = store.rows[-1]["payload"]["result"]["result_rows"][0]
        self.assertEqual(values[0], {
            "trace_value_type": "timedelta",
            "days": -2,
            "seconds": 3,
            "microseconds": 4,
        })
        self.assertEqual(values[1]["trace_value_type"], "memoryview")
        self.assertEqual(values[1]["base64"], "AP90cmFjZQ==")
        self.assertEqual(values[1]["format"], "B")
        self.assertEqual(values[2], {
            "trace_value_type": "postgres_range",
            "empty": False,
            "bounds": "[)",
            "lower": 1,
            "upper": 9,
        })
        json.dumps(store.rows, allow_nan=False)
        recorder.raise_if_failed()

    def test_selection_numpy_votes_and_scalars_are_portable_and_transparent(self):
        """Native ranking arrays must be recorded without changing their result."""
        class SelectionRunner:
            def __init__(self):
                self._llm = FakeLLM(FakeCompletions())
                self.result = (
                    np.array([1, 0], dtype=np.int64),
                    {
                        "score": np.float64(0.75),
                        "edge": np.float64("-inf"),
                        "unknown": np.float64("nan"),
                    },
                )

            def _compare_sqls(self, sql1, sql2, data_item):
                return self.result

        runner = SelectionRunner()
        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(runner, "sql_selection")
        try:
            with recorder.context("attempt"):
                actual = runner._compare_sqls("SELECT 1", "SELECT 2", data_item())
        finally:
            cleanup()

        self.assertIs(actual, runner.result)
        payload = store.rows[-1]["payload"]
        self.assertEqual(payload["votes"], [1, 0])
        self.assertEqual(
            payload["result"][1],
            {
                "score": 0.75,
                "edge": {"nonfinite_float": "-Infinity"},
                "unknown": {"nonfinite_float": "NaN"},
            },
        )
        self.assertNotIn("numpy", json.dumps(payload, allow_nan=False))
        recorder.raise_if_failed()

    def test_native_checker_records_input_output_and_cleanup_restores_instance(self):
        """Revision wrappers must preserve the native result and both SQL versions."""
        from app.pipeline.sql_revision.checkers.time_checker import TimeChecker

        checker = TimeChecker()
        original_bound = checker.check_and_revise
        runner = SimpleNamespace(
            _llm=FakeLLM(FakeCompletions()),
            _checkers=[checker],
        )
        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(runner, "sql_revision")
        sql = "SELECT strftime('%Y', recorded_at) >= 2020 FROM events"
        try:
            with recorder.context("attempt"):
                result = checker.check_and_revise(sql, data_item(), None)
        finally:
            cleanup()

        expected = "SELECT strftime('%Y', recorded_at) >= '2020' FROM events"
        self.assertEqual(result[0], expected)
        event = next(row for row in store.rows if row["kind"] == "component_result")
        self.assertEqual(event["payload"]["input_sql"], sql)
        self.assertEqual(event["payload"]["output_sql"], expected)
        self.assertEqual(event["payload"]["branch_path"], ["revision.TimeChecker"])
        self.assertEqual(checker.check_and_revise, original_bound)

    def test_revision_candidate_is_unique_parent_of_each_checker_chain(self):
        """Converging intermediate SQL must retain its initial-candidate lineage."""
        class Checker:
            def check_and_revise(self, sql, data_item, llm, sampling_budget=1):
                return sql.replace("raw", "revised"), {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                }

        class RevisionRunner:
            def __init__(self):
                self._llm = FakeLLM(FakeCompletions())
                self._checkers = [Checker()]

            def _revise_one_candidate(self, sql, data_item):
                return self._checkers[0].check_and_revise(
                    sql, data_item, self._llm, 1
                )

        runner = RevisionRunner()
        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(runner, "sql_revision")
        try:
            with recorder.context("attempt"):
                first = runner._revise_one_candidate("SELECT raw AS a", data_item())
                second = runner._revise_one_candidate("SELECT raw AS b", data_item())
        finally:
            cleanup()

        self.assertEqual(first[0], "SELECT revised AS a")
        self.assertEqual(second[0], "SELECT revised AS b")
        starts = [row["payload"] for row in store.rows if row["kind"] == "component_start"]
        candidates = [row for row in starts if row["component"] == "revision.candidate"]
        checkers = [row for row in starts if row["component"] == "revision.Checker"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(checkers), 2)
        self.assertNotEqual(candidates[0]["component_call_id"], candidates[1]["component_call_id"])
        self.assertEqual(
            [row["parent_component_call_id"] for row in checkers],
            [row["component_call_id"] for row in candidates],
        )

    def test_selection_analysis_helpers_record_actual_shortlist_pairs_and_matrix(self):
        """Selection traces must explain both shortcuts and pairwise scoring."""
        class SelectionAnalysisRunner:
            def __init__(self):
                self._llm = FakeLLM(FakeCompletions())

            def _get_top_k_sql_candidates(self, data_item):
                return [("SELECT 1", "1", 0.75, 0.01), ("SELECT 2", "2", 0.25, 0.02)]

            def _get_pair_sqls_to_eval(self, candidates):
                return [(candidates[0], candidates[1])]

            def _compute_robust_win_matrix(self, matrix):
                return np.mean(matrix, axis=2)

            def _compare_sqls(self, *args, **kwargs):
                return ["A"], {"total_tokens": 1}

        runner = SelectionAnalysisRunner()
        store = EventStore()
        recorder = self.module.TraceRecorder(store)
        cleanup = recorder.instrument_runner(runner, "sql_selection")
        try:
            with recorder.context("attempt"):
                shortlist = runner._get_top_k_sql_candidates(data_item())
                pairs = runner._get_pair_sqls_to_eval(shortlist)
                matrix = runner._compute_robust_win_matrix(
                    np.array([[[1.0], [0.75]], [[0.25], [1.0]]])
                )
        finally:
            cleanup()

        self.assertEqual(pairs, [(shortlist[0], shortlist[1])])
        np.testing.assert_allclose(matrix, [[1.0, 0.75], [0.25, 1.0]])
        results = {
            row["payload"]["component"]: row["payload"]["result"]
            for row in store.rows if row["kind"] == "component_result"
        }
        self.assertEqual(set(results), {
            "selection.shortlist", "selection.pairs", "selection.win_matrix",
        })
        self.assertEqual(results["selection.shortlist"][0][0], "SELECT 1")
        self.assertEqual(results["selection.pairs"][0][0][0], "SELECT 1")
        self.assertEqual(results["selection.win_matrix"], [[1.0, 0.75], [0.25, 1.0]])


if __name__ == "__main__":
    unittest.main()
