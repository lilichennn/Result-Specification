"""Five independent single-choice samples with durable per-slot budgets."""
import asyncio
import copy
from dataclasses import asdict
import time
import threading
import uuid

from scripts.baseline_adapters.shared.transport import RecordingError, RequestStopped
from .records import aggregate_observed_usage


class ChoiceProtocolError(RuntimeError):
    """A successful HTTP response did not contain exactly one choice."""


def _known_inspection_body(body):
    """Match only the provider's two observed sample-specific rejections."""
    if not isinstance(body, dict):
        return False
    body = body.get("error", body)
    return (isinstance(body, dict)
            and body.get("code") == "data_inspection_failed"
            and body.get("message") in (
                "Output data may contain inappropriate content.",
                "Input text data may contain inappropriate content."))


def classify_error(error):
    """Stable classification; never interpret or execute SQL at this boundary."""
    from openai import APIConnectionError, APITimeoutError
    status = getattr(error, "status_code", None)
    if isinstance(error, RecordingError):
        category, retryable, pause = "recording", False, True
    elif isinstance(error, (RequestStopped, asyncio.CancelledError)):
        category, retryable, pause = "interrupted", False, False
    elif status == 400 and _known_inspection_body(getattr(error, "body", None)):
        category, retryable, pause = "data_inspection", True, False
    elif status in (400, 401, 403, 404, 422) or isinstance(error, (TypeError, ValueError)):
        category, retryable, pause = "configuration", False, True
    elif isinstance(error, (TimeoutError, APITimeoutError)):
        category, retryable, pause = "timeout", True, False
    elif status == 429:
        category, retryable, pause = "rate_limit", True, False
    elif status is not None and (status >= 500 or status in (408, 409)):
        category, retryable, pause = "server", True, False
    elif isinstance(error, APIConnectionError):
        category, retryable, pause = "connection", True, False
    elif isinstance(error, ChoiceProtocolError):
        category, retryable, pause = "protocol", True, False
    else:
        category, retryable, pause = "local", False, False
    return {"category": category, "type": type(error).__name__, "status_code": status,
            "retryable": retryable, "pause": pause}


def _payload(response):
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return copy.deepcopy(response)
    raise ChoiceProtocolError("Response is not a structured chat completion")


class GroupRequester:
    def __init__(self, dispatcher, client, settings, records, *, resume_configuration_errors=False):
        self.dispatcher, self.client = dispatcher, client
        self.settings, self.records = settings.validate(), records
        # Only an explicit operator resume may reconsider a historical fatal
        # configuration error. New failures still stop this invocation below.
        self.resume_configuration_errors = resume_configuration_errors
        limits = getattr(dispatcher, "limits", None)
        if limits is not None and limits.request_timeout != settings.request_timeout_seconds:
            raise ValueError("DAIL dispatcher must explicitly use the 910-second request deadline")

    def _append(self, version, kind, payload):
        try:
            return self.records.append(version, kind, payload)
        except Exception as exc:
            self.dispatcher.stop(cancel_active=True)
            raise RecordingError("DAIL request recording failed") from exc

    async def generate(self, *, version_id: str, round_execution_id: str,
                       messages: list[dict], model: str) -> dict:
        samples = await self._run(version_id, round_execution_id, messages, model, range(5))
        return self._group(samples)

    async def preflight_sample(self, *, version_id: str, round_execution_id: str,
                               messages: list[dict], model: str) -> dict:
        """Diagnostic single-slot check through the same budget and HTTP path."""
        samples = await self._run(version_id, round_execution_id, messages, model, (0,))
        return samples[0]

    async def _run(self, version_id, round_execution_id, messages, model, positions):
        kwargs = self.settings.request_kwargs(model, messages)
        if not isinstance(round_execution_id, str) or not round_execution_id:
            raise ValueError("round_execution_id is required")
        identity = asdict(self.records.version_key(version_id))
        identity["round_execution_id"] = round_execution_id
        history = self.records.request_history(version_id, round_execution_id)
        for slot in history:
            if slot["attempts"]:
                source = self.records.get_event(version_id, slot["attempts"][0]["request_attempt_id"])
                if source.get("queued_event_id"):
                    queued = self.records.get_event(version_id, source["queued_event_id"])
                    if queued["request_kwargs"] != kwargs:
                        raise ValueError("Existing round model/messages/settings differ from this invocation")
        cancelled = threading.Event()
        tasks = [asyncio.create_task(self._sample(version_id, identity, history[pos], kwargs, cancelled))
                 for pos in positions]
        pending = asyncio.gather(*tasks)
        try:
            samples = await asyncio.shield(pending)
        except BaseException as exc:
            cancelled.set()
            if not isinstance(exc, asyncio.CancelledError):
                self.dispatcher.stop(cancel_active=True)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if pending.done() and not pending.cancelled():
                pending.exception()
            raise
        return samples

    @staticmethod
    def _group(samples):
        success = all(sample["status"] == "success" for sample in samples)
        return {"status": "success" if success else "failed",
                "choices": [sample["choice"] for sample in samples if sample["status"] == "success"],
                "samples": samples,
                "request_attempt_ids": [a for sample in samples for a in sample["request_attempt_ids"]],
                "successful_request_ids": [sample["successful_request_id"] for sample in samples
                                           if sample["status"] == "success"],
                "success_usage": aggregate_observed_usage([s["success_usage"] for s in samples]) if success else None,
                "error": None if success else {"category": "samples_failed", "failed_positions": [
                    s["sample_position"] for s in samples if s["status"] != "success"]}}

    async def _sample(self, version, identity, slot, kwargs, cancelled):
        sample = {"sample_position": slot["sample_position"], "status": "failed",
                  "request_attempt_ids": [a["request_attempt_id"] for a in slot["attempts"]],
                  "successful_request_id": None, "success_usage": None,
                  "error": {"category": "attempts_exhausted"}}
        for old in slot["attempts"]:
            if old["status"] == "success":
                result = self.records.get_event(version, old["request_result_id"])
                sample.update(status="success", successful_request_id=old["request_result_id"],
                              choice=result["choice"], success_usage=result.get("usage"), error=None)
                return sample
        # Success restoration takes precedence over earlier repaired failures.
        # Exhaustion is a terminal sample outcome, not a mutation of historical
        # cancellation/authentication evidence or unknown usage.
        if len(slot['attempts']) >= self.settings.max_attempts:
            return sample
        for old in slot['attempts']:
            if old["request_result_id"]:
                result = self.records.get_event(version, old["request_result_id"])
                sample["error"] = result.get("error") or sample["error"]
                repaired_configuration = (self.resume_configuration_errors
                    and sample['error'].get('category') == 'configuration'
                    and sample['error'].get('pause') is True)
                if (sample["error"].get("retryable") is False
                        and sample["error"].get("category") != "interrupted"
                        and not repaired_configuration):
                    return sample
        for number in range(len(slot["attempts"]) + 1, self.settings.max_attempts + 1):
            if self.dispatcher.stop_event.is_set():
                sample["error"] = {"category": "interrupted", "retryable": False}
                break
            request_identity = {**identity, "sample_position": slot["sample_position"],
                                "request_id": uuid.uuid4().hex, "attempt_no": number}
            queued_at = time.monotonic()
            queued = self._append(version, "attempt_queued", {**request_identity,
                "queued_at": queued_at, "request_kwargs": kwargs})
            attempt_id, started_at = None, None
            def started():
                nonlocal attempt_id, started_at
                if cancelled.is_set():
                    raise RequestStopped("Group cancelled before HTTP send")
                started_at = time.monotonic()
                attempt_id = self._append(version, "request_attempt", {**request_identity,
                    "lifecycle": "attempt_started", "queued_event_id": queued,
                    "queued_at": queued_at, "started_at": started_at})
                sample["request_attempt_ids"].append(attempt_id)
            def finished(request_identity, telemetry):
                self._append(version, "request_dispatch", {
                    **request_identity, **telemetry, "queued_event_id": queued,
                    "request_attempt_id": attempt_id})
            body = None
            try:
                response = await self.dispatcher.call_chat(self.client, identity=request_identity,
                    sdk_kwargs=kwargs, on_started=started, on_finished=finished)
                body = _payload(response)
                choices = body.get("choices")
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise ChoiceProtocolError("Exactly one choice is required")
            except (Exception, asyncio.CancelledError) as exc:
                error = classify_error(exc)
                # An unrecorded send is forbidden. Unsent failures cannot keep
                # looping over a budget that has not been spent.
                if attempt_id is None and error["category"] != "interrupted":
                    error.update(category="recording" if isinstance(exc, RecordingError) else "local",
                                 retryable=False, pause=True)
                usage = body.get("usage") if body else None
                provider_body = getattr(exc, "body", None)
                if usage is None and isinstance(provider_body, dict):
                    usage = provider_body.get("usage")
                payload = {**request_identity, "status": "failed", "usage": usage,
                           "error": error, "response": body,
                           "finished_at": time.monotonic()}
                if attempt_id is not None:
                    self._append(version, "request_result", {
                        **payload, "request_attempt_id": attempt_id})
                else:
                    self._append(version, "attempt_not_sent", {**payload, "queued_event_id": queued})
                sample["error"] = error
                if error["pause"]:
                    self.dispatcher.stop(cancel_active=True)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if not error["retryable"]:
                    break
            else:
                result_id = self._append(version, "request_result", {
                    **request_identity, "request_attempt_id": attempt_id, "status": "success",
                    "choice": choices[0], "usage": body.get("usage"), "response": body,
                    "finish_reason": choices[0].get("finish_reason"), "error": None,
                    "finished_at": time.monotonic()})
                sample.update(status="success", successful_request_id=result_id,
                              success_usage=body.get("usage"), choice=choices[0], error=None)
                return sample
        if len(sample['request_attempt_ids']) >= self.settings.max_attempts:
            sample['error'] = {'category': 'attempts_exhausted'}
        return sample
