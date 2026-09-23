"""One process-wide preparation owner's cached, token-paced embedding service.

Embedding and map executors are separate: map functions may wait for embedding
requests without consuming the request workers. Retries live on the admission
queue, never asleep inside an in-flight request. VectorCache owns persistence.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .precompute_cache import fingerprint, validate_vectors
from .sampling import submit_owned


@dataclass(frozen=True)
class EmbeddingLimits:
    batch_size: int = 20
    dimension: int = 1024
    request_limit: int = 64
    request_workers: int = 64
    http_connections: int = 64
    start_rate: float = 200.0
    input_tokens_per_second: float = 12000.0
    input_tokens_per_minute: float = 720000.0
    request_timeout: float = 60.0
    max_attempts: int = 4
    retry_delay: float = 1.0
    max_input_tokens: int | None = None
    feedback_window_seconds: float = 60.0
    decrease_factor: float = .8
    recovery_window_seconds: float = 60.0
    recovery_factor: float = 1.05

    def __post_init__(self):
        for field in ('batch_size', 'dimension', 'request_limit', 'request_workers',
                      'http_connections', 'max_attempts'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f'{field} must be a positive integer')
        if self.batch_size > 20 or self.max_attempts > 4:
            raise ValueError('Embedding batches allow at most 20 texts and four attempts')
        for field in ('start_rate', 'input_tokens_per_second', 'input_tokens_per_minute',
                      'request_timeout', 'retry_delay', 'feedback_window_seconds', 'decrease_factor',
                      'recovery_window_seconds', 'recovery_factor'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{field} must be finite')
            if value < 0 or (value == 0 and field != 'retry_delay'):
                raise ValueError(f'{field} must be positive')
        if not 0 < self.decrease_factor < 1 or self.recovery_factor <= 1:
            raise ValueError('Feedback must decrease by a factor below one and recover by a factor above one')
        if self.max_input_tokens is not None and (type(self.max_input_tokens) is not int or self.max_input_tokens < 1):
            raise ValueError('max_input_tokens must be a positive integer or None')


def embedding_namespace(environment, *, dimension=1024):
    """Raw-vector identity, independent of batch size, RC version or index rules."""
    return {'model': environment['EMBEDDING_MODEL'],
            'endpoint': environment['EMBEDDING_BASE_URL'].rstrip('/'),
            'dimension': dimension, 'encoding_format': 'float', 'storage': 'float32',
            'input_policy': 'verbatim-utf8-v1'}


class EmbeddingClosedError(RuntimeError):
    """Queued work was canceled because its owning preparation service closed."""


class _AdmissionPolicy:
    """Clock-independent pacing and rolling Token accounting; caller serializes."""
    def __init__(self, limits):
        self.limits = limits
        self.request_scale = self.token_scale = 1.0
        self.next_request = self.next_token = 0.0
        self.window = deque()
        self._window_tokens = 0
        self.last_decrease = {'tokens': float('-inf'), 'requests': float('-inf')}
        self.last_feedback = dict(self.last_decrease)
        self.last_recovery = dict(self.last_decrease)

    def snapshot(self):
        return {'start_rate': self.limits.start_rate * self.request_scale,
                'input_tokens_per_second': self.limits.input_tokens_per_second * self.token_scale,
                'input_tokens_per_minute': self.limits.input_tokens_per_minute * self.token_scale}

    def _prune(self, now):
        while self.window and self.window[0][0] <= now - 60:
            reservation = self.window.popleft()
            self._window_tokens -= reservation[1]
            reservation[2] = False

    def _delays(self, now, token_counts):
        self.recover(now)
        self._prune(now)
        common = max(0., self.next_request - now, self.next_token - now)
        used = self._window_tokens
        budget = self.snapshot()['input_tokens_per_minute']
        required = [(used if tokens > budget else max(0., used + tokens - budget))
                    for tokens in token_counts]
        cumulative, expiries, total = [], [], 0
        if any(value > 0 for value in required):
            for started, charged, _ in self.window:
                total += charged
                cumulative.append(total)
                expiries.append(started + 60)
        delays = []
        for needed in required:
            window_delay = 0.
            if needed > 0:
                position = bisect_left(cumulative, needed)
                window_delay = max(0., expiries[position] - now)
            delays.append(max(common, window_delay))
        return delays

    def delay(self, now, tokens):
        return self._delays(now, [tokens])[0]

    def select(self, now, batches):
        """Choose one batch with bounded fairness for an unsplittable input."""
        if not batches:
            raise ValueError('Cannot select from an empty embedding queue')
        delays = self._delays(now, [batch.estimated_tokens for batch in batches])
        delays = [max(batch.due - now, delay) for batch, delay in zip(batches, delays)]
        budget = self.snapshot()['input_tokens_per_minute']
        # Once a due, unsplittable batch reaches the queue, younger ordinary
        # batches cannot perpetually refill the rolling window ahead of it.
        # A retry that is still backing off does not block unrelated requests.
        barrier = next((index for index, batch in enumerate(batches)
                        if batch.due <= now and batch.estimated_tokens > budget), None)
        candidates = range(len(batches) if barrier is None else barrier + 1)
        index = min(candidates, key=delays.__getitem__)
        return index, delays[index]

    def reserve(self, now, tokens):
        rates = self.snapshot()
        self.next_request = max(now, self.next_request) + 1 / rates['start_rate']
        self.next_token = max(now, self.next_token) + tokens / rates['input_tokens_per_second']
        reservation = [now, tokens, True]
        self.window.append(reservation)
        self._window_tokens += tokens
        return reservation

    def calibrate(self, reservation, actual, now):
        if actual is not None:
            difference = actual - reservation[1]
            reservation[1] = actual
            if reservation[2]:
                self._window_tokens += difference
            self.next_token = max(now, self.next_token + difference / self.snapshot()['input_tokens_per_second'])

    def throttle(self, kind, now):
        self.last_feedback[kind] = now
        self.last_recovery[kind] = now
        if now - self.last_decrease[kind] >= self.limits.feedback_window_seconds:
            field = 'token_scale' if kind == 'tokens' else 'request_scale'
            setattr(self, field, getattr(self, field) * self.limits.decrease_factor)
            self.last_decrease[kind] = now

    def recover(self, now):
        for kind, field in (('tokens', 'token_scale'), ('requests', 'request_scale')):
            if (getattr(self, field) < 1 and now - self.last_feedback[kind] >= self.limits.recovery_window_seconds
                    and now - self.last_recovery[kind] >= self.limits.recovery_window_seconds):
                setattr(self, field, min(1., getattr(self, field) * self.limits.recovery_factor))
                self.last_recovery[kind] = now

    def note_failure(self, now):
        # Transport/server failures also interrupt a stable recovery interval.
        for kind in self.last_recovery:
            self.last_recovery[kind] = now


@dataclass
class _Batch:
    texts: list[str]
    estimated_tokens: int
    purpose: str
    queued_at: float
    call_id: str
    attempt: int = 0
    due: float = 0.


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _usage(response):
    source = _field(response, 'usage')
    prompt = _field(source, 'prompt_tokens', _field(source, 'input_tokens'))
    total = _field(source, 'total_tokens')
    return {name: value if type(value) is int and value >= 0 else None
            for name, value in (('prompt_tokens', prompt), ('total_tokens', total))}


def _retryable(error):
    status = getattr(error, 'status_code', None)
    if status is not None:
        return status in (408, 409, 429) or 500 <= status < 600
    # OpenAI connection errors inherit APIConnectionError, not OSError.
    from openai import APIConnectionError
    return isinstance(error, (OSError, TimeoutError, APIConnectionError))


def _feedback_kind(error):
    if getattr(error, 'status_code', None) != 429:
        return None
    message = (str(error) + ' ' + str(getattr(error, 'body', ''))).lower()
    return 'tokens' if 'token' in message or 'tpm' in message else 'requests'


class EmbeddingService:
    """Shared client, pending-text deduplication and a durable float32 cache.

    ``embed`` and ``__call__`` return unnormalized NumPy float32 arrays, suitable
    for native ``np.asarray`` consumers. The caller owns and closes VectorCache
    after this service; the service owns and closes its injected/shared client.
    """
    manages_retries = True

    def __init__(self, environment, cache, log_path, *, limits=None, client=None):
        self.limits = limits or EmbeddingLimits()
        for key in ('EMBEDDING_MODEL', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY'):
            if not isinstance(environment.get(key), str) or not environment[key].strip():
                raise ValueError(f'Missing embedding environment field: {key}')
        self.model = environment['EMBEDDING_MODEL']
        self.cache = cache
        expected = embedding_namespace(environment, dimension=self.limits.dimension)
        if any(key not in cache.namespace_config for key in ('model', 'endpoint')):
            raise ValueError('Embedding cache namespace must identify its model and endpoint')
        # Old verified VectorCache namespaces did not have explicit dimension or
        # input_policy fields. Their stored dimension and verbatim input remain
        # compatible; do not rewrite their namespace or vector bytes.
        for key in ('model', 'endpoint', 'dimension', 'encoding_format', 'storage', 'input_policy'):
            if key in cache.namespace_config and cache.namespace_config[key] != expected[key]:
                raise ValueError(f'Embedding cache namespace mismatch: {key}')
        if cache.dimension is not None and cache.dimension != self.limits.dimension:
            raise ValueError('Embedding cache dimension does not match service')
        self._condition = threading.Condition(threading.RLock())
        self._close_lock = threading.Lock()
        self._closing = self._closed = False
        self._pending = {}
        self._queue = deque()
        self._active = self._peak = 0
        self._policy = _AdmissionPolicy(self.limits)
        self._map_slots = threading.BoundedSemaphore(self.limits.request_workers)
        self._request_pool = ThreadPoolExecutor(self.limits.request_workers, thread_name_prefix='deepeye-embedding-request')
        self._map_pool = ThreadPoolExecutor(self.limits.request_workers, thread_name_prefix='deepeye-embedding-map')
        self._client = client
        self._log = None
        self._scheduler = None
        try:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open('a', encoding='utf-8')
            if self._client is None:
                import httpx
                from openai import OpenAI
                transport = httpx.Client(timeout=self.limits.request_timeout, limits=httpx.Limits(
                    max_connections=self.limits.http_connections, max_keepalive_connections=self.limits.http_connections))
                try:
                    self._client = OpenAI(api_key=environment['EMBEDDING_API_KEY'],
                        base_url=environment['EMBEDDING_BASE_URL'], max_retries=0,
                        timeout=self.limits.request_timeout, http_client=transport)
                except BaseException:
                    transport.close()
                    raise
            self._client.max_retries = 0
            self._scheduler = threading.Thread(target=self._schedule, name='deepeye-embedding-admission')
            self._scheduler.start()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __call__(self, texts):
        return self.embed(texts)

    def embed(self, texts, *, purpose='embedding') -> np.ndarray:
        texts = list(texts)
        purpose = str(purpose)
        with self._condition:
            if self._closing:
                raise EmbeddingClosedError('Embedding service is closed')
            found = self.cache.get_many(texts)
            if not texts:
                return np.empty((0, self.limits.dimension), dtype='<f4')
            waiting, owned, estimates = {}, [], {}
            for text, vector in found.items():
                if vector is not None:
                    continue
                # UTF-8 byte length is a conservative tokenizer-independent
                # preflight estimate; returned usage recalibrates the charge.
                tokens = max(1, len(text.encode('utf-8')))
                if self.limits.max_input_tokens is not None and tokens > self.limits.max_input_tokens:
                    raise ValueError('Embedding input exceeds the configured estimated-token limit; text was not truncated')
                estimates[text] = tokens
            pending_hits = sum(text in self._pending for text in estimates)
            event = {'kind': 'embedding_cache', 'timestamp': datetime.now(timezone.utc).isoformat(),
                     'purpose': purpose, 'requested_texts': len(texts), 'unique_texts': len(found),
                     'cache_hits': sum(vector is not None for vector in found.values()),
                     'pending_hits': pending_hits, 'cache_misses': len(estimates) - pending_hits}
            try:
                self._write_event(event)
            except BaseException:
                self._closing = True
                self._condition.notify_all()
                raise
            # Validate the entire input list before creating any shared pending
            # entry; a later invalid input cannot orphan an earlier text.
            for text, tokens in estimates.items():
                if text not in self._pending:
                    self._pending[text] = Future()
                    owned.append((text, tokens))
                waiting[text] = self._pending[text]
            batch, estimate = [], 0
            target = min(self.limits.input_tokens_per_second, self.limits.input_tokens_per_minute)
            for text, tokens in owned:
                # The provider accepts empty text alone but rejects mixed batches.
                if text == '':
                    self._queue.append(_Batch([text], tokens, purpose, time.monotonic(), uuid.uuid4().hex))
                    continue
                if batch and (len(batch) == self.limits.batch_size or estimate + tokens > target):
                    self._queue.append(_Batch(batch, estimate, purpose, time.monotonic(), uuid.uuid4().hex))
                    batch, estimate = [], 0
                batch.append(text)
                estimate += tokens
            if batch:
                self._queue.append(_Batch(batch, estimate, purpose, time.monotonic(), uuid.uuid4().hex))
            self._condition.notify_all()
        try:
            found.update((text, future.result()) for text, future in waiting.items())
        finally:
            # Caller failures never detach another successful owned batch.
            wait(tuple(waiting.values()))
        return np.stack([found[text] for text in texts])

    def map(self, function, iterable):
        """Ordered concurrent map with at most request_workers submitted calls."""
        iterator, pending = iter(iterable), deque()
        def submit(value):
            self._map_slots.acquire()
            try:
                with self._condition:
                    if self._closing:
                        raise EmbeddingClosedError('Embedding service is closed')
                    future = submit_owned(self._map_pool, function, value)
            except BaseException:
                self._map_slots.release()
                raise
            future.add_done_callback(lambda _: self._map_slots.release())
            return future
        try:
            for _ in range(self.limits.request_workers):
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                pending.append(submit(value))
            while pending:
                yield pending.popleft().result()
                try:
                    value = next(iterator)
                except StopIteration:
                    continue
                pending.append(submit(value))
        finally:
            for future in pending:
                future.cancel()
            wait(tuple(pending))

    def _complete(self, batch, *, array=None, error=None):
        for index, text in enumerate(batch.texts):
            future = self._pending.pop(text)
            if error is None:
                future.set_result(array[index])
            else:
                future.set_exception(error)

    def _schedule(self):
        while True:
            with self._condition:
                if self._closing:
                    while self._queue:
                        self._complete(self._queue.popleft(), error=EmbeddingClosedError('Queued embedding canceled'))
                    if not self._active:
                        return
                    self._condition.wait()
                    continue
                cap = min(self.limits.request_limit, self.limits.request_workers, self.limits.http_connections)
                if self._active >= cap or not self._queue:
                    self._condition.wait()
                    continue
                now = time.monotonic()
                index, delay = self._policy.select(now, self._queue)
                if delay > 0:
                    self._condition.wait(delay)
                    continue
                batch = self._queue[index]
                del self._queue[index]
                reservation = self._policy.reserve(now, batch.estimated_tokens)
                self._active += 1
                self._peak = max(self._peak, self._active)
                batch.attempt += 1
                try:
                    submit_owned(self._request_pool, self._perform, batch, reservation, now)
                except BaseException as error:
                    self._active -= 1
                    self._policy.calibrate(reservation, 0, time.monotonic())
                    self._complete(batch, error=error)

    def _perform(self, batch, reservation, started):
        result, array, error = None, None, None
        try:
            result = self._client.embeddings.create(model=self.model, input=batch.texts,
                dimensions=self.limits.dimension, encoding_format='float', timeout=self.limits.request_timeout)
            rows = _field(result, 'data', [])
            indices = [_field(row, 'index') for row in rows]
            if (any(type(index) is not int for index in indices)
                    or sorted(indices) != list(range(len(batch.texts)))):
                raise ValueError('Embedding response indices/count are invalid')
            array = validate_vectors([_field(row, 'embedding') for row in sorted(rows, key=lambda row: _field(row, 'index'))],
                                     len(batch.texts), self.limits.dimension)
            self.cache.put_many(batch.texts, array)
        except BaseException as caught:
            error = caught
        with self._condition:
            now = time.monotonic()
            usage = _usage(result)
            self._policy.calibrate(reservation, usage['prompt_tokens'], now)
            feedback = _feedback_kind(error)
            if feedback:
                self._policy.throttle(feedback, now)
            elif error is not None:
                self._policy.note_failure(now)
            retry = error is not None and _retryable(error) and batch.attempt < self.limits.max_attempts and not self._closing
            event = {'kind': 'embedding_request', 'timestamp': datetime.now(timezone.utc).isoformat(),
                     'call_id': batch.call_id, 'purpose': batch.purpose, 'attempt': batch.attempt,
                     'model': self.model, 'batch_size': len(batch.texts),
                     'input_hash': fingerprint(batch.texts), 'estimated_input_tokens': batch.estimated_tokens,
                     'usage': usage, 'usage_complete': all(value is not None for value in usage.values()),
                     'queue_seconds': max(0., started - batch.queued_at), 'request_seconds': max(0., now - started),
                     'status': 'error' if error else 'ok', 'error_type': type(error).__name__ if error else None,
                     'http_status': getattr(error, 'status_code', None) or (200 if result is not None else None),
                     'rate_feedback': feedback, 'retry_scheduled': retry, 'rates': self._policy.snapshot()}
            try:
                self._write_event(event)
            except BaseException as log_error:
                error, retry, self._closing = log_error, False, True
            self._active -= 1
            if retry:
                delay = self.limits.retry_delay * 2 ** (batch.attempt - 1)
                headers = getattr(getattr(error, 'response', None), 'headers', {})
                try:
                    after = float(headers.get('retry-after', 0))
                    if math.isfinite(after):
                        delay = max(delay, after)
                except (ValueError, TypeError):
                    pass
                batch.due = now + delay
                batch.queued_at = now
                self._queue.append(batch)
            else:
                self._complete(batch, array=array, error=error)
            self._condition.notify_all()

    def _write_event(self, event):
        self._log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        self._log.flush()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            with self._condition:
                self._closing = True
                self._condition.notify_all()
            if self._scheduler is not None and self._scheduler.ident is not None:
                self._scheduler.join()
            self._map_pool.shutdown(wait=True, cancel_futures=True)
            self._request_pool.shutdown(wait=True, cancel_futures=True)
            try:
                if self._client is not None:
                    self._client.close()
            finally:
                if self._log is not None:
                    self._log.close()
                self._closed = True
