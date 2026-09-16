"""Offline request budgets/provenance plus the real cancellable HTTP path."""
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import openai  # Load SDK outside the short-deadline async test bodies.

from scripts.baseline_adapters.dail_sql.config import DailSettings, TaskKey
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.rc_evaluation.dail_sql.campaign import _interrupt_unanswered


def response(text="invalid SQL is still an API success", usage=None, count=1):
    return {"id": "offline", "choices": [{"index": 7, "message": {"content": text},
            "finish_reason": "stop"} for _ in range(count)], "usage": usage}


def inspection_error(message, *, nested=False, code="data_inspection_failed", status=400):
    import httpx2
    request = httpx2.Request("POST", "http://offline.invalid")
    body = {"code": code, "type": "data_inspection_failed", "message": message, "param": None}
    if nested:
        body = {"error": body}
    return openai.APIStatusError("fixture rejection", response=httpx2.Response(status, request=request), body=body)


class InspectionClassificationTests(unittest.TestCase):
    def test_only_exact_known_400_inspection_bodies_are_sample_retryable(self):
        from scripts.baseline_adapters.dail_sql.transport import classify_error
        for message in ("Output data may contain inappropriate content.",
                        "Input text data may contain inappropriate content."):
            for nested in (False, True):
                with self.subTest(message=message, nested=nested):
                    result = classify_error(inspection_error(message, nested=nested))
                    self.assertEqual((result["category"], result["retryable"], result["pause"]),
                                     ("data_inspection", True, False))

    def test_unknown_inspection_and_unsupported400_or_auth_still_pause(self):
        from scripts.baseline_adapters.dail_sql.transport import classify_error
        known = "Output data may contain inappropriate content."
        cases = [dict(message="Input data may contain inappropriate content."),
                 dict(message=known + " More details."),
                 dict(message=known, code="invalid_parameter"),
                 dict(message=known, status=401), dict(message=known, status=403),
                 dict(message=known, status=422)]
        for values in cases:
            for nested in (False, True):
                with self.subTest(values=values, nested=nested):
                    result = classify_error(inspection_error(**values, nested=nested))
                    self.assertEqual((result["category"], result["retryable"], result["pause"]),
                                     ("configuration", False, True))


class FakeDispatcher:
    """Only replaces remote I/O; requester, records and recovery are real."""
    def __init__(self, sequences=None):
        self.sequences = sequences or {}
        self.calls = []
        self.stop_event = threading.Event()
        self.fatal_error = None

    def stop(self, **kwargs):
        self.stop_event.set()

    async def call_chat(self, client, *, identity, sdk_kwargs, on_started=None, on_finished=None):
        self.calls.append((dict(identity), sdk_kwargs))
        on_started()
        pos, number = identity["sample_position"], identity["attempt_no"]
        await asyncio.sleep((4 - pos) * .001)
        sequence = self.sequences.get(pos, [])
        result = sequence[number - 1] if number <= len(sequence) else response(str(pos), {"total_tokens": 2})
        if isinstance(result, Exception):
            raise result
        return result


class RequesterTests(unittest.IsolatedAsyncioTestCase):
    async def test_exhausted_interrupted_samples_are_terminal_without_sixth_send(self):
        for persisted_cancel in (False, True):
            with self.subTest(persisted_cancel=persisted_cancel):
                self.version = self.records.begin_version(TaskKey('batch', 'spider_dev', '1'))
                for position in range(5):
                    for number in range(1, 6):
                        attempt = self.records.append(self.version, 'request_attempt', {
                            'round_execution_id': 'r1', 'sample_position': position, 'attempt_no': number})
                        if number < 5 or persisted_cancel:
                            self.records.append(self.version, 'request_result', {
                                'request_attempt_id': attempt, 'status': 'failed', 'usage': None,
                                'error': {'category': 'interrupted' if number == 5 else 'timeout',
                                          'retryable': number != 5, 'pause': False}})
                _interrupt_unanswered(self.records, self.version)
                before = self.records.events(self.version)
                dispatch = FakeDispatcher()
                result = await self.generate(dispatch)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual([s['error']['category'] for s in result['samples']], ['attempts_exhausted'] * 5)
                self.assertFalse(any(s['error'].get('pause') for s in result['samples']))
                self.assertEqual(dispatch.calls, [])
                self.assertEqual(self.records.events(self.version), before)
                self.assertEqual(len(result['request_attempt_ids']), 25)
                self.assertIsNone(result['success_usage'])

    async def test_explicit_repaired_auth_resume_uses_remaining_budget_and_preserves_successes(self):
        import inspect
        import httpx2
        from openai import AuthenticationError
        from scripts.baseline_adapters.dail_sql.transport import GroupRequester
        self.assertIn('resume_configuration_errors', inspect.signature(GroupRequester).parameters,
                      'explicit operator resume authorization missing')
        error = AuthenticationError('bad key', response=httpx2.Response(401,
            request=httpx2.Request('POST', 'https://offline.invalid')), body={})
        first = await self.generate(FakeDispatcher({0: [error]}))
        self.assertEqual([len(s['request_attempt_ids']) for s in first['samples']], [1] * 5)
        self.assertEqual(len(first['successful_request_ids']), 4)
        ordinary = FakeDispatcher()
        unchanged = await self.generate(ordinary)
        self.assertEqual(ordinary.calls, [])
        self.assertTrue(unchanged['samples'][0]['error']['pause'])
        still_broken = FakeDispatcher({0: [error, error]})
        failed_repair = await GroupRequester(still_broken, object(), DailSettings(), self.records,
            resume_configuration_errors=True).generate(version_id=self.version, round_execution_id='r1',
                model='fixture', messages=[{'role': 'user', 'content': 'SELECT 1'}])
        self.assertEqual([(i['sample_position'], i['attempt_no']) for i, _ in still_broken.calls], [(0, 2)])
        self.assertTrue(failed_repair['samples'][0]['error']['pause'])
        self.assertTrue(still_broken.stop_event.is_set())
        repaired = FakeDispatcher()
        requester = GroupRequester(repaired, object(), DailSettings(), self.records,
                                   resume_configuration_errors=True)
        recovered = await requester.generate(version_id=self.version, round_execution_id='r1',
            model='fixture', messages=[{'role': 'user', 'content': 'SELECT 1'}])
        self.assertEqual(recovered['status'], 'success')
        self.assertEqual([(i['sample_position'], i['attempt_no']) for i, _ in repaired.calls], [(0, 3)])
        self.assertEqual(recovered['successful_request_ids'][1:], first['successful_request_ids'])
        after = FakeDispatcher()
        self.assertEqual((await self.generate(after))['status'], 'success')
        self.assertEqual(after.calls, [])

    async def test_explicit_auth_resume_does_not_reset_exhausted_budget(self):
        import inspect
        from scripts.baseline_adapters.dail_sql.transport import GroupRequester
        self.assertIn('resume_configuration_errors', inspect.signature(GroupRequester).parameters)
        for position in range(5):
            for number in range(1, 6):
                attempt = self.records.append(self.version, 'request_attempt', {
                    'round_execution_id': 'r1', 'sample_position': position, 'attempt_no': number})
                self.records.append(self.version, 'request_result', {'request_attempt_id': attempt,
                    'status': 'failed', 'usage': None,
                    'error': {'category': 'configuration', 'retryable': False, 'pause': True}})
        dispatch = FakeDispatcher()
        requester = GroupRequester(dispatch, object(), DailSettings(), self.records,
                                   resume_configuration_errors=True)
        result = await requester.generate(version_id=self.version, round_execution_id='r1',
            model='fixture', messages=[{'role': 'user', 'content': 'SELECT 1'}])
        self.assertEqual([s['error']['category'] for s in result['samples']], ['attempts_exhausted'] * 5)
        self.assertEqual(dispatch.calls, [])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest = {"batch_id": "batch", "groups": {"spider_dev": {"ids": ["1"]}}}
        self.root = Path(self.tmp.name) / "run"
        self.records = DailRecords(self.root, self.manifest)
        self.addCleanup(lambda: self.records.close())
        self.version = self.records.begin_version(TaskKey("batch", "spider_dev", "1"))

    def requester(self, dispatch):
        self.assertIsNotNone(importlib.util.find_spec("scripts.baseline_adapters.dail_sql.transport"),
                             "DAIL five-slot transport is missing")
        from scripts.baseline_adapters.dail_sql.transport import GroupRequester
        return GroupRequester(dispatch, object(), DailSettings(), self.records)

    async def generate(self, dispatch):
        return await self.requester(dispatch).generate(version_id=self.version, round_execution_id="r1",
            model="fixture", messages=[{"role": "user", "content": "SELECT 1"}])

    async def test_mixed_retry_budget_order_and_exact_parameters(self):
        import httpx2
        from openai import APITimeoutError, RateLimitError, InternalServerError
        request = httpx2.Request("POST", "http://offline.invalid")
        timeout = APITimeoutError(request=request)
        rate = RateLimitError("rate", response=httpx2.Response(429, request=request), body={})
        server = InternalServerError("server", response=httpx2.Response(500, request=request), body={})
        dispatch = FakeDispatcher({2: [timeout, rate, server, timeout, response("bad SQL", {"total_tokens": 3})]})
        result = await self.generate(dispatch)
        self.assertEqual(result["status"], "success")
        self.assertEqual([s["sample_position"] for s in result["samples"]], list(range(5)))
        self.assertEqual([len(s["request_attempt_ids"]) for s in result["samples"]], [1, 1, 5, 1, 1])
        self.assertEqual([c["message"]["content"] for c in result["choices"]], ["0", "1", "bad SQL", "3", "4"])
        self.assertEqual(result["success_usage"], {"total_tokens": 11})
        for identity, kwargs in dispatch.calls:
            self.assertEqual(kwargs, {"model": "fixture", "messages": [{"role": "user", "content": "SELECT 1"}],
                "n": 1, "temperature": .6, "extra_body": {"enable_thinking": True}})
            self.assertEqual((identity["batch_id"], identity["group"], identity["question_id"]),
                             ("batch", "spider_dev", "1"))
        for sample in result["samples"]:
            source = self.records.get_event(self.version, sample["successful_request_id"])
            self.assertEqual(source["choice"]["index"], 7)
            self.assertEqual(source["choice"], sample["choice"])
            self.assertEqual(source["request_attempt_id"], sample["request_attempt_ids"][-1])

    async def test_known_inspection_retries_same_slot_with_unchanged_request(self):
        for message in ("Output data may contain inappropriate content.",
                        "Input text data may contain inappropriate content."):
            for nested in (False, True):
                with self.subTest(message=message, nested=nested):
                    self.version = self.records.begin_version(TaskKey("batch", "spider_dev", "1"))
                    dispatch = FakeDispatcher({2: [inspection_error(message, nested=nested)]})
                    result = await self.generate(dispatch)
                    self.assertEqual(result["status"], "success")
                    self.assertFalse(dispatch.stop_event.is_set())
                    self.assertEqual([len(s["request_attempt_ids"]) for s in result["samples"]], [1, 1, 2, 1, 1])
                    self.assertEqual(len(dispatch.calls), 6)
                    expected = {"model": "fixture", "messages": [{"role": "user", "content": "SELECT 1"}],
                                "n": 1, "temperature": .6, "extra_body": {"enable_thinking": True}}
                    self.assertTrue(all(kwargs == expected for _, kwargs in dispatch.calls))
                    self.assertEqual([identity["attempt_no"] for identity, _ in dispatch.calls
                                      if identity["sample_position"] == 2], [1, 2])

    async def test_repeated_known_inspection_exhausts_only_slot_without_sixth_attempt(self):
        for message in ("Output data may contain inappropriate content.",
                        "Input text data may contain inappropriate content."):
            for nested in (False, True):
                with self.subTest(message=message, nested=nested):
                    self.version = self.records.begin_version(TaskKey("batch", "spider_dev", "1"))
                    dispatch = FakeDispatcher({2: [inspection_error(message, nested=nested)] * 5})
                    result = await self.generate(dispatch)
                    self.assertEqual(result["status"], "failed")
                    self.assertFalse(dispatch.stop_event.is_set())
                    self.assertEqual([len(s["request_attempt_ids"]) for s in result["samples"]], [1, 1, 5, 1, 1])
                    self.assertEqual(len(result["choices"]), 4)
                    self.assertIsNone(result["success_usage"])
                    self.assertEqual(len(dispatch.calls), 9)
                    restored = await self.generate(dispatch)
                    self.assertEqual(restored["successful_request_ids"], result["successful_request_ids"])
                    self.assertEqual(len(dispatch.calls), 9)
                    dispatch.sequences.clear()
                    independent = await self.requester(dispatch).generate(version_id=self.version,
                        round_execution_id="independent", model="fixture",
                        messages=[{"role": "user", "content": "SELECT 1"}])
                    self.assertEqual(independent["status"], "success")

    async def test_all_exhausted_stop_at_25_and_resume_sends_nothing(self):
        dispatch = FakeDispatcher({p: [TimeoutError()] * 5 for p in range(5)})
        result = await self.generate(dispatch)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(dispatch.calls), 25)
        self.assertEqual(len(result["request_attempt_ids"]), 25)
        self.assertIsNone(result["success_usage"])
        again = FakeDispatcher()
        restored = await self.generate(again)
        self.assertEqual(again.calls, [])
        self.assertEqual(restored["request_attempt_ids"], result["request_attempt_ids"])
        self.assertFalse(dispatch.stop_event.is_set())

    async def test_protocol_failure_keeps_other_successes_and_known_failed_usage(self):
        dispatch = FakeDispatcher({2: [response(usage={"total_tokens": 9}, count=2)] * 5})
        result = await self.generate(dispatch)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["choices"]), 4)
        self.assertEqual(len(dispatch.calls), 9)
        self.assertIsNone(result["success_usage"])
        failed = [e["payload"] for e in self.records.events(self.version, "request_result")
                  if e["payload"]["status"] == "failed"]
        self.assertEqual(len(failed), 5)
        self.assertTrue(all(e["usage"] == {"total_tokens": 9} for e in failed))

    async def test_unknown_usage_and_bad_sql_never_retry(self):
        dispatch = FakeDispatcher({p: [response()] for p in range(5)})
        result = await self.generate(dispatch)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(dispatch.calls), 5)
        self.assertIsNone(result["success_usage"])

    async def test_existing_round_rejects_changed_model_or_prompt_before_reuse(self):
        await self.generate(FakeDispatcher())
        dispatch = FakeDispatcher()
        for model, text in (("changed", "SELECT 1"), ("fixture", "SELECT 2")):
            with self.assertRaises(ValueError):
                await self.requester(dispatch).generate(version_id=self.version, round_execution_id="r1",
                    model=model, messages=[{"role": "user", "content": text}])
        self.assertEqual(dispatch.calls, [])

    async def test_result_storage_failure_pauses_and_is_not_an_api_retry(self):
        from scripts.baseline_adapters.shared.transport import RecordingError
        append = self.records.append
        def fail_result(version, kind, payload):
            if kind == "request_result":
                raise OSError("storage unavailable")
            return append(version, kind, payload)
        dispatch = FakeDispatcher()
        with patch.object(self.records, "append", side_effect=fail_result):
            with self.assertRaises(RecordingError):
                await self.generate(dispatch)
        self.assertTrue(dispatch.stop_event.is_set())
        self.assertTrue(all(identity["attempt_no"] == 1 for identity, _ in dispatch.calls))

    async def test_reopen_restores_each_budget_started_unknown_and_success(self):
        for pos in range(5):
            for number in range(1, pos + 2):
                source = self.records.append(self.version, "request_attempt", {
                    "round_execution_id": "r1", "sample_position": pos, "attempt_no": number})
                if pos == 0:
                    self.records.append(self.version, "request_result", {"request_attempt_id": source,
                        "status": "success", "choice": response("retained")["choices"][0], "usage": None})
        self.records.append(self.version, "attempt_queued", {
            "round_execution_id": "r1", "sample_position": 1, "attempt_no": 3, "request_id": "unsent"})
        self.records.close()
        self.records = DailRecords(self.root, self.manifest)
        dispatch = FakeDispatcher({p: [TimeoutError()] * 5 for p in range(5)})
        result = await self.generate(dispatch)
        self.assertEqual([(i["sample_position"], i["attempt_no"]) for i, _ in dispatch.calls],
                         [(1, 3), (2, 4), (3, 5), (2, 5), (1, 4), (1, 5)])
        self.assertEqual(result["samples"][0]["choice"]["message"]["content"], "retained")
        self.assertEqual([len(s["request_attempt_ids"]) for s in result["samples"]], [1, 5, 5, 5, 5])
        self.assertEqual(result["status"], "failed")

    async def test_auth_failure_pauses_dispatch_and_never_retries(self):
        import httpx2
        from openai import AuthenticationError
        request = httpx2.Request("POST", "http://offline.invalid")
        error = AuthenticationError("auth", response=httpx2.Response(401, request=request),
                                    body={"message": "Incorrect key: test-only-secret"})
        dispatch = FakeDispatcher({p: [error] for p in range(5)})
        result = await self.generate(dispatch)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(dispatch.stop_event.is_set())
        self.assertEqual(len(dispatch.calls), 5)
        self.assertNotIn("test-only-secret", json.dumps(self.records.events(self.version)))


class SharedHTTPTests(unittest.IsolatedAsyncioTestCase):
    def dispatch(self, **overrides):
        self.assertIsNotNone(importlib.util.find_spec("scripts.baseline_adapters.shared"),
                             "Shared asynchronous transport package is missing")
        self.assertIsNotNone(importlib.util.find_spec("scripts.baseline_adapters.shared.transport"),
                             "Shared asynchronous transport is missing")
        from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits
        limits = RequestLimits(**{**dict(start_rate=10000, request_limit=2, http_connections=2,
                                        request_timeout=.15), **overrides})
        dispatch = RequestDispatcher(limits)
        self.addCleanup(dispatch.close)
        return dispatch

    def identity(self, number=1):
        return dict(batch_id="b", group="g", question_id="q", round_execution_id="r",
                    sample_position=0, request_id=str(number), attempt_no=number)

    async def test_single_http_send_persistence_precedes_send_and_no_sdk_retry(self):
        from tests.test_deepeye_request_dispatch import LoopbackServer
        from openai import InternalServerError
        server = LoopbackServer(["error", "ok"])
        self.addCleanup(server.close)
        dispatch = self.dispatch()
        client = dispatch.make_client(api_key="offline", base_url=server.url)
        starts = []
        def on_started():
            starts.append(len(server.requests))
        with self.assertRaises(InternalServerError):
            await dispatch.call_chat(client, identity=self.identity(),
                sdk_kwargs=DailSettings().request_kwargs("fixture", [{"role": "user", "content": "SELECT 1"}]),
                on_started=on_started)
        self.assertEqual(starts, [0])
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(server.requests[0]["enable_thinking"], True)
        self.assertNotIn("max_tokens", server.requests[0])

    async def test_start_record_failure_sends_nothing(self):
        from scripts.baseline_adapters.shared.transport import RecordingError
        from tests.test_deepeye_request_dispatch import LoopbackServer
        server = LoopbackServer([])
        self.addCleanup(server.close)
        dispatch = self.dispatch()
        client = dispatch.make_client(api_key="offline", base_url=server.url)
        def cannot_record():
            raise OSError("disk failed")
        with self.assertRaises(RecordingError):
            await dispatch.call_chat(client, identity=self.identity(), sdk_kwargs={"model": "fixture", "messages": []},
                                     on_started=cannot_record)
        self.assertEqual(server.requests, [])
        self.assertEqual(dispatch.snapshot()["in_flight"], 0)
        self.assertTrue(dispatch.stop_event.is_set())

    async def test_finished_callback_failure_is_once_after_release_and_pauses(self):
        from scripts.baseline_adapters.shared.transport import RecordingError
        from tests.test_deepeye_request_dispatch import LoopbackServer
        server = LoopbackServer([])
        self.addCleanup(server.close)
        dispatch = self.dispatch()
        client = dispatch.make_client(api_key="offline", base_url=server.url)
        observations = []
        def finished(identity, telemetry):
            observations.append((identity, telemetry, dispatch.snapshot()["in_flight"]))
            raise OSError("disk failed")
        with self.assertRaises(RecordingError):
            await dispatch.call_chat(client, identity=self.identity(),
                sdk_kwargs={"model": "fixture", "messages": []}, on_finished=finished)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0][0]["request_id"], "1")
        self.assertEqual(observations[0][2], 0)
        self.assertTrue(observations[0][1]["released"])
        self.assertTrue(dispatch.stop_event.is_set())
        self.assertEqual(len(server.requests), 1)

    async def test_deadline_drains_actual_http_and_cancellation_does_not_use_waiter_threads(self):
        from tests.test_deepeye_request_dispatch import LoopbackServer
        from openai import APITimeoutError
        server = LoopbackServer(["trickle", "no_headers", "ok"])
        self.addCleanup(server.close)
        dispatch = self.dispatch()
        client = dispatch.make_client(api_key="offline", base_url=server.url)
        params = {"model": "fixture", "messages": []}
        with self.assertRaises(APITimeoutError):
            await dispatch.call_chat(client, identity=self.identity(), sdk_kwargs=params)
        # SDK caches platform metadata in its own initial worker. Count after
        # that one-off initialization; call_chat itself must not add a waiter.
        thread_count = threading.active_count()
        task = asyncio.create_task(dispatch.call_chat(client, identity=self.identity(2), sdk_kwargs=params))
        for _ in range(100):
            if len(server.requests) == 2:
                break
            await asyncio.sleep(.005)
        self.assertLessEqual(threading.active_count(), thread_count)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(dispatch.snapshot()["in_flight"], 0)
        for _ in range(100):
            if {0, 1} <= set(server.disconnected):
                break
            await asyncio.sleep(.005)
        self.assertTrue({0, 1} <= set(server.disconnected))
        result = await dispatch.call_chat(client, identity=self.identity(3), sdk_kwargs=params)
        self.assertEqual(len(result.choices), 1)
        self.assertEqual(len(server.requests), 3)

    async def test_cancelled_group_queued_slots_cost_nothing_and_resume_keeps_budget(self):
        from tests.test_deepeye_request_dispatch import LoopbackServer
        from scripts.baseline_adapters.dail_sql.transport import GroupRequester
        server = LoopbackServer(["no_headers"])
        self.addCleanup(server.close)
        dispatch = self.dispatch(request_timeout=910, request_limit=1)
        client = dispatch.make_client(api_key="offline", base_url=server.url)
        with tempfile.TemporaryDirectory() as temp:
            manifest = {"batch_id": "batch", "groups": {"g": {"ids": ["q"]}}}
            with DailRecords(Path(temp), manifest) as records:
                version = records.begin_version(TaskKey("batch", "g", "q"))
                requester = GroupRequester(dispatch, client, DailSettings(), records)
                params = dict(version_id=version, round_execution_id="r", model="fixture",
                              messages=[{"role": "user", "content": "SELECT 1"}])
                task = asyncio.create_task(requester.generate(**params))
                for _ in range(100):
                    if server.requests:
                        break
                    await asyncio.sleep(.005)
                self.assertEqual(len(server.requests), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(len(records.events(version, "request_attempt")), 1)
                self.assertEqual(dispatch.snapshot()["in_flight"], 0)
            with DailRecords(Path(temp), manifest) as records:
                result = await GroupRequester(dispatch, client, DailSettings(), records).generate(**params)
                self.assertEqual(result["status"], "success")
                self.assertEqual([len(s["request_attempt_ids"]) for s in result["samples"]], [2, 1, 1, 1, 1])
                self.assertEqual(len(server.requests), 6)


class PreflightCLITests(unittest.TestCase):
    def test_single_then_group_and_group_resume_are_isolated_and_keep_secrets_out(self):
        from tests.test_deepeye_request_dispatch import LoopbackServer
        server = LoopbackServer([])
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, "DASH_BASE_URL": server.url, "DASH_API_KEY": "test-only-secret",
                   "DASH_MODELS": "fixture"}
            command = [sys.executable, "-E", "-B", "-m", "scripts.rc_evaluation.dail_sql.cli",
                       "preflight", "--output", temp]
            results = []
            for mode, previous in (("single", None), ("group", None), ("group", "resume")):
                args = [*command, "--mode", mode]
                if previous:
                    args.extend(["--version-id", results[-1]["version_id"]])
                run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode, 0, run.stderr)
                result = json.loads(run.stdout)
                self.assertEqual(result["status"], "success")
                results.append(result)
                self.assertNotIn("test-only-secret", run.stdout + run.stderr)
            self.assertEqual(len(server.requests), 6)
            self.assertNotEqual(results[0]["records_dir"], results[1]["records_dir"])
            changed = subprocess.run([*command, "--mode", "group", "--version-id", results[-1]["version_id"]],
                env={**env, "DASH_MODELS": "changed-model"}, capture_output=True, text=True, timeout=15)
            self.assertEqual(changed.returncode, 1)
            self.assertEqual(json.loads(changed.stdout)["status"], "failed")
            self.assertEqual(len(server.requests), 6)
            for path in Path(temp).rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"test-only-secret", path.read_bytes())

    def test_failed_single_preflight_exhausts_only_one_slot_and_reports_failure(self):
        from tests.test_deepeye_request_dispatch import LoopbackServer
        server = LoopbackServer(["error"] * 5)
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run([sys.executable, "-E", "-B", "-m", "scripts.rc_evaluation.dail_sql.cli",
                "preflight", "--mode", "single", "--output", temp],
                env={**os.environ, "DASH_BASE_URL": server.url, "DASH_API_KEY": "offline",
                     "DASH_MODELS": "fixture"}, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertTrue(run.stdout.strip(), run.stderr)
            self.assertEqual(json.loads(run.stdout)["status"], "failed")
            self.assertEqual(len(server.requests), 5)
