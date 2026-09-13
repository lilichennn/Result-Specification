"""Shared adaptive admission, bounded retries and cost audit for precomputation."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from time import monotonic, sleep
from typing import Any, Callable, TypeVar
import uuid

import httpx
import openai

_T = TypeVar("_T")
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class EmptyResponseError(ValueError):
    """A successful HTTP response without usable content can be retried."""


def _empty_usage() -> dict[str, Any]:
    return {"attempts": 0, "elapsed_seconds": 0., "usage_missing": 0, "usage_complete": True,
            **{field: None for field in _TOKEN_FIELDS}}


class AdaptiveAPI:
    """Synchronous chat and embedding requests share one adaptive concurrency limit.

    Each attempt is appended to ``api_calls.jsonl``. Token totals include only
    reported usage; unknown counts remain None, with usage_missing tracking gaps.
    close() stops admission and waits for admitted requests without cancelling them.
    """

    def __init__(self, values: dict, output_dir: Path, *, initial_concurrency=200,
                 step=50, max_concurrency=600, min_concurrency=1, window_seconds=60,
                 chat_max_tokens=2048, thinking_budget=1024):
        integer_options = (initial_concurrency, step, max_concurrency, min_concurrency,
                           chat_max_tokens, thinking_budget)
        if any(type(value) is not int or value < 1 for value in integer_options):
            raise ValueError("Concurrency and token limits must be positive integers")
        if not min_concurrency <= initial_concurrency <= max_concurrency:
            raise ValueError("Expected min_concurrency <= initial_concurrency <= max_concurrency")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        for key in ("DASH_BASE_URL", "DASH_API_KEY", "DASH_MODELS", "EMBEDDING_BASE_URL",
                    "EMBEDDING_API_KEY", "EMBEDDING_MODEL"):
            if not isinstance(values.get(key), str) or not values[key].strip():
                raise ValueError(f"Missing environment field: {key}")
        for key in ("DASH_MODELS", "EMBEDDING_MODEL"):
            if values[key].startswith("sk-") or not re.fullmatch(r"[\w./:-]+", values[key]):
                raise ValueError(f"Invalid model name in {key}")

        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._closed = False
        self._resources_closed = False
        self._limit = initial_concurrency
        self._in_flight = 0
        self._started = monotonic()
        self._last_increase = self._started
        self._last_decrease = float("-inf")
        self._failures_at_last_decrease = 0
        self._window: deque[tuple[float, bool]] = deque()
        self._changes: list[dict[str, Any]] = []
        self._counts = {"attempts": 0, "successes": 0, "failures": 0, "retries": 0}
        self._usage: dict[str, dict[str, Any]] = {}
        self._call_usage: dict[str, dict[str, Any]] = {}
        self._embedding_dimension: int | None = None
        self._session_id = uuid.uuid4().hex
        self._secrets = sorted({value for key, value in values.items()
                                if isinstance(value, str) and value
                                and ("KEY" in key or "PASSWORD" in key)}, key=len, reverse=True)
        self._config = {"initial_concurrency": initial_concurrency, "step": step,
                        "min_concurrency": min_concurrency, "max_concurrency": max_concurrency,
                        "window_seconds": window_seconds, "decrease_cooldown_seconds": 15,
                        "new_failures_required_for_decrease": 5,
                        "chat_max_tokens": chat_max_tokens, "thinking_budget": thinking_budget,
                        "temperature": 0.6, "chat_model": values["DASH_MODELS"],
                        "embedding_model": values["EMBEDDING_MODEL"], "embedding_timeout": 60,
                        "chat_timeout": 300, "max_retries": 4, "sdk_max_retries": 0}
        self.audit_path = Path(output_dir) / "api_calls.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit = self.audit_path.open("a", encoding="utf-8")
        self._clients: list[openai.OpenAI] = []
        try:
            for prefix, timeout in (("DASH", 300), ("EMBEDDING", 60)):
                transport = httpx.Client(timeout=timeout, limits=httpx.Limits(
                    max_connections=max_concurrency, max_keepalive_connections=max_concurrency))
                try:
                    client = openai.OpenAI(api_key=values[f"{prefix}_API_KEY"],
                                           base_url=values[f"{prefix}_BASE_URL"],
                                           max_retries=0, timeout=timeout, http_client=transport)
                except BaseException:
                    transport.close()
                    raise
                self._clients.append(client)
        except BaseException:
            self.close()
            raise
        self._chat_client, self._embedding_client = self._clients

    def _adjust(self, now: float) -> None:
        window = self._config["window_seconds"]
        while self._window and self._window[0][0] < now - window:
            self._window.popleft()
        successes = sum(success for _, success in self._window)
        failures = len(self._window) - successes
        old, reason = self._limit, None
        if (failures >= 5 and failures / len(self._window) >= .1
                and self._counts["failures"] - self._failures_at_last_decrease >= 5
                and now - self._last_decrease >= 15):
            self._limit = max(self._config["min_concurrency"], old - self._config["step"])
            self._last_decrease = now
            self._failures_at_last_decrease = self._counts["failures"]
            self._last_increase = now
            reason = "frequent_failures"
        elif successes >= 50 and not failures and now - self._last_increase >= window:
            self._limit = min(self._config["max_concurrency"], old + self._config["step"])
            self._last_increase = now
            reason = "stable_window"
        if old != self._limit:
            self._changes.append({"elapsed_seconds": now - self._started, "from": old,
                                  "to": self._limit, "reason": reason,
                                  "successes": successes, "failures": failures})
            self._condition.notify_all()

    def _acquire(self) -> None:
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("AdaptiveAPI is closed")
                self._adjust(monotonic())
                if self._in_flight < self._limit:
                    self._in_flight += 1
                    return
                self._condition.wait()

    def _redact(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED]")
        return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)

    def _finish(self, kind: str, input_hash: str, call_id: str, attempt: int, started: float,
                response: Any, error: BaseException | None) -> None:
        now = monotonic()
        usage = getattr(response, "usage", None)
        tokens = {field: getattr(usage, field, None) for field in _TOKEN_FIELDS}
        tokens = {field: value if type(value) is int and value >= 0 else None
                  for field, value in tokens.items()}
        expected = _TOKEN_FIELDS if kind == "chat" else ("prompt_tokens", "total_tokens")
        request_id = (getattr(error, "request_id", None) if error is not None
                      else getattr(response, "_request_id", None))
        if request_id is None:
            request_id = getattr(response, "_request_id", None)
        event = {"session_id": self._session_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "kind": kind, "input_hash": input_hash, "call_id": call_id, "attempt": attempt,
                 "elapsed_seconds": max(0., now - started), "status": "ok" if error is None else "error",
                 "http_status": 200 if response is not None else getattr(error, "status_code", None),
                 "error_class": type(error).__name__ if error is not None else None,
                 "request_id": self._redact(request_id) if isinstance(request_id, str) else None,
                 "usage_missing": int(any(tokens[field] is None for field in expected)), **tokens}
        with self._condition:
            try:
                self._window.append((now, error is None))
                self._counts["attempts"] += 1
                self._counts["successes" if error is None else "failures"] += 1
                self._counts["retries"] += int(attempt > 1)
                for cost in (
                    self._usage.setdefault(input_hash, _empty_usage()),
                    self._call_usage.setdefault(call_id, _empty_usage()),
                ):
                    for field in ("attempts", "elapsed_seconds", "usage_missing"):
                        cost[field] += 1 if field == "attempts" else event[field]
                    cost["usage_complete"] = cost["usage_complete"] and not event["usage_missing"]
                    for field, value in tokens.items():
                        if value is not None:
                            cost[field] = (cost[field] or 0) + value
                self._adjust(now)
                event["concurrency"] = self._limit
                self._audit.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                self._audit.flush()
            finally:
                self._in_flight -= 1
                self._condition.notify_all()

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, openai.APIStatusError):
            return error.status_code == 429 or error.status_code >= 500
        return isinstance(error, (openai.APIConnectionError, httpx.TransportError,
                                  TimeoutError, ConnectionError, EmptyResponseError))

    def _call(self, kind: str, inputs: Any, invoke: Callable[[], Any],
              validate: Callable[[Any], _T], *, return_metadata: bool = False) -> Any:
        encoded = json.dumps(inputs, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        input_hash = hashlib.sha256(encoded).hexdigest()
        call_id = uuid.uuid4().hex
        for attempt in range(1, 6):
            self._acquire()
            started, response, error = monotonic(), None, None
            try:
                try:
                    response = invoke()
                    result = validate(response)
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    self._finish(kind, input_hash, call_id, attempt, started, response, error)
            except Exception as exc:
                if attempt == 5 or not self._retryable(exc):
                    raise
                sleep(min(2 ** (attempt - 1), 30))
                continue
            if return_metadata:
                with self._condition:
                    usage = dict(self._call_usage[call_id])
                return {"content": result, "usage": usage, "call_id": call_id,
                        "input_hash": input_hash}
            return result
        raise RuntimeError("Unreachable retry state")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def validate(response: Any) -> list[list[float]]:
            rows = getattr(response, "data", None)
            if not rows:
                raise EmptyResponseError("Embedding response is empty")
            if len(rows) != len(texts):
                raise ValueError("Embedding response count does not match input count")
            vectors: dict[int, list[float]] = {}
            dimension = None
            for row in rows:
                index = getattr(row, "index", None)
                if type(index) is not int or not 0 <= index < len(texts) or index in vectors:
                    raise ValueError("Embedding response indices are incomplete or duplicated")
                raw = getattr(row, "embedding", None)
                if not isinstance(raw, list) or not raw:
                    raise ValueError("Embedding vector is missing")
                if any(type(value) not in (float, int) for value in raw):
                    raise ValueError("Embedding vector must contain numbers")
                vector = [float(value) for value in raw]
                if not all(math.isfinite(value) for value in vector) or not any(vector):
                    raise ValueError("Embedding vector must be finite and nonzero")
                if dimension is not None and len(vector) != dimension:
                    raise ValueError("Embedding dimensions differ within a batch")
                dimension = len(vector)
                vectors[index] = vector
            with self._condition:
                if self._embedding_dimension is not None and dimension != self._embedding_dimension:
                    raise ValueError("Embedding dimension changed between batches")
                self._embedding_dimension = dimension
            return [vectors[index] for index in range(len(texts))]

        return self._call("embedding", texts, lambda: self._embedding_client.embeddings.create(
            model=self._config["embedding_model"], input=texts, encoding_format="float", timeout=60), validate)

    def chat(self, messages: list[dict]) -> str:
        return self.chat_with_usage(messages)["content"]

    def chat_with_usage(self, messages: list[dict]) -> dict:
        """Return content and this call's cost, including only its own HTTP retries.

        Persist this whole result with the question. Tokens sum only reported usage;
        entirely unreported fields stay None. Missing usage in any attempt sets
        usage_complete=False. Input hashes are diagnostic, not cost identities.
        """
        def validate(response: Any) -> str:
            choices = getattr(response, "choices", None)
            content = getattr(choices[0].message, "content", None) if choices else None
            if not isinstance(content, str) or not content.strip():
                raise EmptyResponseError("Chat response has no text content")
            return content

        return self._call("chat", messages, lambda: self._chat_client.chat.completions.create(
            model=self._config["chat_model"], messages=messages, temperature=0.6,
            max_tokens=self._config["chat_max_tokens"], timeout=300,
            extra_body={"thinking_budget": self._config["thinking_budget"]}), validate,
            return_metadata=True)

    def usage_for(self, input_hash: str) -> dict:
        """Process-local diagnostic sum across all calls with this input, not per-question cost."""
        with self._condition:
            return dict(self._usage.get(input_hash, _empty_usage()))

    def summary(self) -> dict:
        with self._condition:
            return {"session_id": self._session_id, "config": dict(self._config), **self._counts,
                    "concurrency": self._limit, "in_flight": self._in_flight,
                    "embedding_dimension": self._embedding_dimension, "closed": self._closed,
                    "elapsed_seconds": max(0., monotonic() - self._started),
                    "adjustments": [dict(change) for change in self._changes],
                    "audit_path": str(self.audit_path)}

    def close(self) -> None:
        with self._close_lock:
            if self._resources_closed:
                return
            with self._condition:
                self._closed = True
                self._condition.notify_all()
                self._condition.wait_for(lambda: self._in_flight == 0)
            try:
                for client in self._clients:
                    client.close()
            finally:
                self._audit.close()
                self._resources_closed = True

    def __enter__(self) -> AdaptiveAPI:
        with self._condition:
            if self._closed:
                raise RuntimeError("AdaptiveAPI is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
