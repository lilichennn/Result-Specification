"""Single-choice durable requester for the two DIN Linking RC3 nodes."""
from __future__ import annotations

import asyncio
from email.utils import parsedate_to_datetime
import inspect
import math
import threading
import time
import uuid

from scripts.baseline_adapters.dail_sql.transport import classify_error
from scripts.baseline_adapters.shared.transport import RecordingError, RequestStopped

from .records import NODES, digest


MAX_ATTEMPTS = 5
_SENSITIVE_KEY_NAMES = frozenset({
    "authorization",
    "proxyauthorization",
    "apikey",
    "xapikey",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "bearertoken",
    "token",
    "secret",
    "clientsecret",
    "cookie",
    "setcookie",
    "password",
    "passwd",
    "credential",
    "credentials",
})


class OutputValidationError(RuntimeError):
    """A provider response was delivered but did not satisfy the node contract."""


def retry_delay(attempt_no, retry_after=None):
    delay = 2 ** max(0, attempt_no - 2)
    if retry_after is not None and math.isfinite(retry_after):
        delay = max(delay, retry_after)
    return min(60, delay)


def parse_retry_after(value, *, now=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        try:
            result = parsedate_to_datetime(value).timestamp() - (
                time.time() if now is None else now
            )
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, result) if math.isfinite(result) else None


class LinkingRequester:
    def __init__(self, dispatcher, client, settings, records):
        self.dispatcher = dispatcher
        self.client = client
        self.settings = settings
        self.records = records
        configured_attempts = getattr(settings, "max_attempts", MAX_ATTEMPTS)
        if configured_attempts != MAX_ATTEMPTS:
            raise ValueError("DIN Linking requires exactly five request attempts")
        if (
            hasattr(dispatcher, "limits")
            and hasattr(settings, "request_timeout_seconds")
            and dispatcher.limits.request_timeout != settings.request_timeout_seconds
        ):
            raise ValueError(
                "DIN Linking dispatcher must use the configured total request deadline"
            )
        self._node_locks = {}
        self._node_locks_guard = threading.Lock()

    def _stop(self):
        self.dispatcher.stop(cancel_active=True)

    def _append(self, version, kind, payload):
        try:
            return self.records.append(version, kind, payload)
        except Exception as exc:
            self._stop()
            raise RecordingError("DIN Linking durable recording failed") from exc

    def _pause(self, category, exc=None):
        self._stop()
        error = RuntimeError(f"DIN Linking paused: {category}")
        if exc is None:
            raise error
        raise error from exc

    async def _replay(self, outcome):
        try:
            payload = await asyncio.to_thread(
                self.records.read_ref, outcome["response_ref"]
            )
            body = payload["body"]
            content = body["choices"][0]["message"]["content"]
        except Exception as exc:
            self._pause("recording/read", exc)
        return {**outcome, "content": content}

    @staticmethod
    def _content(body):
        choices = body.get("choices") if isinstance(body, dict) else None
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)
        ):
            raise OutputValidationError("Expected exactly one object choice")
        content = choices[0]["message"].get("content")
        if (
            choices[0].get("finish_reason") != "stop"
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise OutputValidationError("Empty or unfinished response")
        return content

    @staticmethod
    async def _validate(validator, content):
        parsed = validator(content)
        if inspect.isawaitable(parsed):
            parsed = await parsed
        return parsed

    @staticmethod
    def _contains_sensitive_key(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = "".join(character for character in str(key).casefold() if character.isalnum())
                if normalized in _SENSITIVE_KEY_NAMES or (
                    normalized != "maxtokens"
                    and normalized.endswith(("apikey", "token", "secret", "password", "cookie", "credential", "credentials"))
                ):
                    return True
                if LinkingRequester._contains_sensitive_key(item):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(LinkingRequester._contains_sensitive_key(item) for item in value)
        return False

    def _node_lock(self, version, node):
        shared = getattr(self.records, "request_lock", None)
        if callable(shared):
            return shared(version, node)
        key = (version, node)
        with self._node_locks_guard:
            return self._node_locks.setdefault(key, asyncio.Lock())

    async def request(self, version, node, kwargs, validator):
        if node not in NODES:
            self._pause("configuration/unknown_node")
        if not isinstance(kwargs, dict) or not kwargs or not callable(validator):
            self._pause("configuration/invalid_request")
        if self._contains_sensitive_key(kwargs):
            self._pause("configuration/sensitive_request")

        async with self._node_lock(version, node):
            return await self._request_locked(version, node, kwargs, validator)

    async def _request_locked(self, version, node, kwargs, validator):

        try:
            fingerprint = digest(kwargs)
        except Exception as exc:
            self._pause("local_data/unrecordable_request", exc)
        try:
            history = self.records.request_history(version)
        except Exception as exc:
            self._pause("recording/read", exc)
        old_input = history["inputs"].get(node)
        if old_input and old_input["input_fingerprint"] != fingerprint:
            self._pause("local_data/request_changed")
        if not old_input:
            await asyncio.to_thread(
                self._append,
                version,
                "node_input",
                {"node": node, "kwargs": kwargs, "input_fingerprint": fingerprint},
            )
            try:
                history = self.records.request_history(version)
            except Exception as exc:
                self._pause("recording/read", exc)

        for outcome in history["outcomes"].get(node, []):
            if outcome.get("input_fingerprint") != fingerprint:
                self._pause("local_data/outcome_input_changed")
            if outcome["status"] == "succeeded":
                return await self._replay(outcome)

        attempts = history["attempts"].get(node, [])
        retry_after = None
        for attempt_no in range(len(attempts) + 1, MAX_ATTEMPTS + 1):
            if attempt_no > 1:
                await asyncio.sleep(retry_delay(attempt_no, retry_after))
            try:
                key = self.records.key(version)
                batch_id = self.records.manifest["batch_id"]
            except Exception as exc:
                self._pause("recording/read", exc)
            identity = {
                "batch_id": batch_id,
                "group": key.group,
                "question_id": key.question_id,
                "round_execution_id": version + ":" + node,
                "sample_position": 0,
                "request_id": uuid.uuid4().hex,
                "attempt_no": attempt_no,
            }
            queued_ref = await asyncio.to_thread(
                self._append, version, "attempt_queued", {"node": node, **identity}
            )
            sent = False
            attempt_ref = None

            def started():
                nonlocal sent, attempt_ref
                attempt_ref = self._append(
                    version,
                    "request_attempt",
                    {"node": node, **identity, "queued_ref": queued_ref},
                )
                sent = True

            def finished(actual_identity, telemetry):
                self._append(
                    version,
                    "request_dispatch",
                    {
                        "node": node,
                        **actual_identity,
                        "attempt_ref": attempt_ref,
                        "telemetry": telemetry,
                    },
                )

            response_ref = None
            body = None
            usage = None
            response_model = None
            parsed = None
            try:
                response = await self.dispatcher.call_chat(
                    self.client,
                    identity=identity,
                    sdk_kwargs=kwargs,
                    on_started=started,
                    on_finished=finished,
                )
                body = (
                    response.model_dump(mode="json")
                    if hasattr(response, "model_dump")
                    else response
                )
                response_ref = await asyncio.to_thread(
                    self._append,
                    version,
                    "request_result",
                    {"node": node, **identity, "body": body},
                )
                if isinstance(body, dict):
                    usage = body.get("usage")
                    response_model = body.get("model")
                content = self._content(body)
                parsed = await self._validate(validator, content)
                outcome = {
                    "node": node,
                    "status": "succeeded",
                    "input_fingerprint": fingerprint,
                    "response_ref": response_ref,
                    "usage": usage,
                    "response_model": response_model,
                    "request_id": identity["request_id"],
                    "attempt_no": attempt_no,
                    "parsed": parsed,
                    "error": None,
                }
                await asyncio.to_thread(
                    self._append, version, "request_outcome", outcome
                )
                return {**outcome, "content": content}
            except (asyncio.CancelledError, RequestStopped):
                raise
            except Exception as exc:
                if isinstance(exc, OutputValidationError):
                    error = {
                        "category": "output_validation",
                        "type": type(exc.__cause__ or exc).__name__,
                        "retryable": True,
                        "pause": False,
                    }
                else:
                    error = classify_error(exc)
                if not sent:
                    error.update(
                        category=(
                            "recording" if isinstance(exc, RecordingError) else "local"
                        ),
                        retryable=False,
                        pause=True,
                    )
                elif error["category"] == "local":
                    error.update(retryable=False, pause=True)
                provider_body = getattr(exc, "body", None)
                if usage is None and isinstance(provider_body, dict):
                    usage = provider_body.get("usage")
                outcome = {
                    "node": node,
                    "status": "failed",
                    "input_fingerprint": fingerprint,
                    "response_ref": response_ref,
                    "usage": usage,
                    "response_model": response_model,
                    "request_id": identity["request_id"],
                    "attempt_no": attempt_no,
                    "parsed": parsed,
                    "error": error,
                }
                await asyncio.to_thread(
                    self._append, version, "request_outcome", outcome
                )
                if error["pause"]:
                    self._pause(f'{error["category"]}/{error.get("type", "unknown")}', exc)
                if not error["retryable"]:
                    return outcome
                try:
                    retry_after = parse_retry_after(
                        exc.response.headers.get("retry-after")
                    )
                except AttributeError:
                    retry_after = None

        return {
            "node": node,
            "status": "failed",
            "content": None,
            "parsed": None,
            "response_ref": None,
            "usage": None,
            "response_model": None,
            "input_fingerprint": fingerprint,
            "error": {
                "category": "retry_budget_exhausted",
                "attempts": MAX_ATTEMPTS,
            },
        }


DinLinkingRequester = LinkingRequester
