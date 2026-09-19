import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.baseline_adapters.din_sql.inputs import DinSettings, TaskKey, digest


def transport_api():
    try:
        from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
        from scripts.baseline_adapters.din_sql_linking.transport import LinkingRequester
    except ModuleNotFoundError as exc:
        raise AssertionError("DIN Linking requester is missing") from exc
    return LinkingRecords, LinkingRequester


def manifest():
    return {
        "format": "din-sql-linking-v1",
        "batch_id": "fixture",
        "groups": {"bird_dev": {"ids": ["0"]}},
    }


def response(content):
    return {
        "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}],
        "model": "qwen3.8-2.4t-a95b",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


class SequenceDispatcher:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.stopped = False

    async def call_chat(self, client, *, identity, sdk_kwargs, on_started, on_finished):
        action = self.actions[self.calls]
        self.calls += 1
        if isinstance(action, BaseException) and getattr(action, "before_send", False):
            raise action
        on_started()
        on_finished(identity, {"request_wall_seconds": 0.01})
        if isinstance(action, BaseException):
            raise action
        return action

    def stop(self, **kwargs):
        self.stopped = True


class EchoDispatcher(SequenceDispatcher):
    """Return the table named in the final user message after both calls overlap."""

    def __init__(self):
        super().__init__([])
        self.release = asyncio.Event()
        self.entered = 0

    async def call_chat(self, client, *, identity, sdk_kwargs, on_started, on_finished):
        self.calls += 1
        self.entered += 1
        on_started()
        if self.entered == 2:
            self.release.set()
        try:
            await asyncio.wait_for(self.release.wait(), 0.2)
        except TimeoutError:
            self.release.set()
        on_finished(identity, {"request_wall_seconds": 0.01})
        name = sdk_kwargs["messages"][-1]["content"]
        return response(json.dumps({"tables": [name]}))


class BrokenResponse:
    def model_dump(self, **_kwargs):
        raise OSError("local serialization failure")


def parse_json(content):
    value = json.loads(content)
    if "tables" not in value:
        raise ValueError("tables missing")
    return value


def parse_model_json(content):
    """Test-side explicit declaration that a parse ValueError came from output."""

    from scripts.baseline_adapters.din_sql_linking.transport import OutputValidationError

    try:
        return parse_json(content)
    except ValueError as exc:
        raise OutputValidationError("invalid fixture model output") from exc


class LinkingTransportTests(unittest.IsolatedAsyncioTestCase):
    async def request_fixture(self, actions, validator=parse_model_json):
        Records, Requester = transport_api()
        tmp = tempfile.TemporaryDirectory()
        records = Records(Path(tmp.name) / "batch", manifest())
        version = records.begin(TaskKey("bird_dev", "0"))
        dispatcher = SequenceDispatcher(actions)
        requester = Requester(dispatcher, None, DinSettings(), records)
        kwargs = {"model": "fixture", "messages": [{"role": "user", "content": "filter"}]}
        return tmp, records, version, dispatcher, requester, kwargs, validator

    async def test_transport_and_validator_failures_share_one_five_send_budget(self):
        fixture = await self.request_fixture(
            [response("not json"), TimeoutError("network"), response("{}"), TimeoutError("network"), response("[]")]
        )
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with patch("scripts.baseline_adapters.din_sql_linking.transport.retry_delay", return_value=0):
                result = await requester.request(version, "schema_filter_rc3", kwargs, validator)
                again = await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["category"], "retry_budget_exhausted")
            self.assertEqual(again["status"], "failed")
            self.assertEqual(dispatcher.calls, 5)
            history = records.request_history(version)
            self.assertEqual(len(history["attempts"]["schema_filter_rc3"]), 5)
            self.assertEqual(len(history["outcomes"]["schema_filter_rc3"]), 5)
        finally:
            records.close()
            tmp.cleanup()

    async def test_success_records_raw_usage_model_parsed_value_and_is_reused(self):
        fixture = await self.request_fixture([response("not json"), response('{"tables":[]}')])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with patch("scripts.baseline_adapters.din_sql_linking.transport.retry_delay", return_value=0):
                result = await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["parsed"], {"tables": []})
            self.assertEqual(result["usage"]["total_tokens"], 5)
            self.assertEqual(result["response_model"], "qwen3.8-2.4t-a95b")
            raw = records.read_ref(result["response_ref"])
            self.assertEqual(raw["body"]["choices"][0]["message"]["content"], '{"tables":[]}')
            self.assertEqual(
                records.request_history(version)["inputs"]["schema_filter_rc3"]["input_fingerprint"],
                digest(kwargs),
            )

            reused = await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertEqual(reused["parsed"], {"tables": []})
            self.assertEqual(dispatcher.calls, 2)
        finally:
            records.close()
            tmp.cleanup()

    async def test_resume_after_reopen_reuses_success_without_dispatch(self):
        Records, Requester = transport_api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            records = Records(root, manifest())
            version = records.begin(TaskKey("bird_dev", "0"))
            kwargs = {"model": "fixture", "messages": [{"role": "user", "content": "link"}]}
            first_dispatcher = SequenceDispatcher([response('{"tables":["scores"]}')])
            first = Requester(first_dispatcher, None, DinSettings(), records)
            result = await first.request(version, "linking_rc3", kwargs, parse_json)
            records.close()

            records = Records(root, manifest())
            no_calls = SequenceDispatcher([])
            resumed = Requester(no_calls, None, DinSettings(), records)
            replay = await resumed.request(version, "linking_rc3", kwargs, parse_json)
            self.assertEqual(replay["parsed"], result["parsed"])
            self.assertEqual(no_calls.calls, 0)
            records.close()

    async def test_changed_request_input_pauses_without_an_http_send(self):
        fixture = await self.request_fixture([response('{"tables":[]}')])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            await requester.request(version, "schema_filter_rc3", kwargs, validator)
            changed = {**kwargs, "temperature": 1}
            with self.assertRaises(RuntimeError):
                await requester.request(version, "schema_filter_rc3", changed, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 1)
        finally:
            records.close()
            tmp.cleanup()

    async def test_failure_before_send_pauses_and_does_not_spend_attempt(self):
        error = TypeError("local configuration")
        error.before_send = True
        fixture = await self.request_fixture([error])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with self.assertRaises(RuntimeError):
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(records.request_history(version)["attempts"].get("schema_filter_rc3", []), [])
        finally:
            records.close()
            tmp.cleanup()

    async def test_record_read_failure_pauses_before_http(self):
        fixture = await self.request_fixture([response('{"tables":[]}')])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with patch.object(records, "request_history", side_effect=OSError("disk")):
                try:
                    await requester.request(version, "schema_filter_rc3", kwargs, validator)
                except Exception as exc:
                    caught = exc
                else:
                    caught = None
            self.assertIsInstance(caught, RuntimeError)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 0)
        finally:
            records.close()
            tmp.cleanup()

    async def test_unrecordable_local_request_data_pauses_before_http(self):
        fixture = await self.request_fixture([response('{"tables":[]}')])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        kwargs["unsupported"] = object()
        try:
            try:
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            except Exception as exc:
                caught = exc
            else:
                caught = None
            self.assertIsInstance(caught, RuntimeError)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 0)
        finally:
            records.close()
            tmp.cleanup()

    async def test_concurrent_requests_for_one_node_are_serial_and_cannot_cross_reuse(self):
        """Removing the per-node critical section must cause two sends/cross-talk."""

        Records, Requester = transport_api()
        with tempfile.TemporaryDirectory() as tmp:
            records = Records(Path(tmp) / "batch", manifest())
            version = records.begin(TaskKey("bird_dev", "0"))
            dispatcher = EchoDispatcher()
            requester = Requester(dispatcher, None, DinSettings(), records)
            second_requester = Requester(dispatcher, None, DinSettings(), records)
            one = {"model": "fixture", "messages": [{"role": "user", "content": "A"}]}
            two = {"model": "fixture", "messages": [{"role": "user", "content": "B"}]}
            original_append = records.append
            input_barrier = threading.Barrier(2)

            def overlap_inputs(version_id, kind, payload):
                if kind == "node_input":
                    try:
                        input_barrier.wait(0.2)
                    except threading.BrokenBarrierError:
                        pass
                return original_append(version_id, kind, payload)

            with patch.object(records, "append", side_effect=overlap_inputs):
                outcomes = await asyncio.gather(
                    requester.request(version, "schema_filter_rc3", one, parse_json),
                    second_requester.request(version, "schema_filter_rc3", two, parse_json),
                    return_exceptions=True,
                )

            successes = [value for value in outcomes if isinstance(value, dict)]
            failures = [value for value in outcomes if isinstance(value, BaseException)]
            self.assertEqual(len(successes), 1)
            self.assertEqual(successes[0]["parsed"], {"tables": ["A"]})
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], RuntimeError)
            self.assertEqual(dispatcher.calls, 1)
            records.close()

    async def test_success_outcome_must_match_the_bound_input_fingerprint(self):
        """A stored success for another prompt must never be replayed."""

        fixture = await self.request_fixture([])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            fingerprint = digest(kwargs)
            records.append(
                version,
                "node_input",
                {"node": "schema_filter_rc3", "kwargs": kwargs, "input_fingerprint": fingerprint},
            )
            response_ref = records.append(
                version,
                "request_result",
                {"node": "schema_filter_rc3", "body": response('{"tables":["wrong"]}')},
            )
            records.append(
                version,
                "request_outcome",
                {
                    "node": "schema_filter_rc3",
                    "status": "succeeded",
                    "response_ref": response_ref,
                    "input_fingerprint": "different",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "outcome_input_changed"):
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 0)
        finally:
            records.close()
            tmp.cleanup()

    async def test_post_send_unknown_local_error_pauses_batch(self):
        """A local response-conversion fault is not a legitimate question failure."""

        fixture = await self.request_fixture([BrokenResponse()])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with self.assertRaisesRegex(RuntimeError, "local/OSError"):
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 1)
        finally:
            records.close()
            tmp.cleanup()

    async def test_validator_local_io_error_pauses_instead_of_spending_retry_budget(self):
        """A validator local I/O failure is not a model-output retry."""

        def broken_validator(_content):
            raise OSError("metadata read failed")

        fixture = await self.request_fixture([response('{"tables":[]}')] * 5, broken_validator)
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with self.assertRaisesRegex(RuntimeError, "local/OSError"):
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 1)
        finally:
            records.close()
            tmp.cleanup()

    async def test_plain_validator_value_error_pauses_after_one_send(self):
        """A bare ValueError carries no evidence that the model output was bad."""

        def broken_validator(_content):
            raise ValueError("local metadata invariant failed")

        fixture = await self.request_fixture([response('{"tables":[]}')] * 5, broken_validator)
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with self.assertRaisesRegex(RuntimeError, "configuration/ValueError"):
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 1)
        finally:
            records.close()
            tmp.cleanup()

    async def test_explicit_output_validation_error_uses_the_five_send_budget(self):
        """Only the explicit model-output exception remains retryable."""

        from scripts.baseline_adapters.din_sql_linking.transport import OutputValidationError

        def invalid_model_output(_content):
            raise OutputValidationError("invalid model selection")

        fixture = await self.request_fixture(
            [response('{"tables":[]}')] * 5,
            invalid_model_output,
        )
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        try:
            with patch("scripts.baseline_adapters.din_sql_linking.transport.retry_delay", return_value=0):
                result = await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertEqual(result["error"]["category"], "retry_budget_exhausted")
            self.assertFalse(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 5)
        finally:
            records.close()
            tmp.cleanup()

    async def test_sensitive_nested_request_kwargs_are_rejected_before_record_or_send(self):
        """Credential-like keys must never cross the durable-recording boundary."""

        fixture = await self.request_fixture([response('{"tables":[]}')])
        tmp, records, version, dispatcher, requester, kwargs, validator = fixture
        kwargs["extra_headers"] = {"Authorization": "Bearer REVIEW_SECRET"}
        try:
            with self.assertRaisesRegex(RuntimeError, "sensitive_request"):
                await requester.request(version, "schema_filter_rc3", kwargs, validator)
            self.assertTrue(dispatcher.stopped)
            self.assertEqual(dispatcher.calls, 0)
            self.assertEqual(records.request_history(version)["inputs"], {})
        finally:
            records.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
