"""One run's bounded async transports behind a synchronous sample API.

No SDK retries and no executor/timer thread per transport. Deadlines cancel the
coroutine running the actual async HTTP operation, and admission is released
only after its cancellation cleanup finishes. Server-side work is unknowable.
"""
import asyncio
from collections import deque
from concurrent.futures import Future
import contextvars
from dataclasses import dataclass, field
import math
import threading
import time
import uuid
from types import SimpleNamespace

from .run_admission import AdaptiveAdmission

_TRANSPORT_REQUEST = contextvars.ContextVar('deepeye_async_transport_request', default=None)


@dataclass(frozen=True)
class RequestLimits:
    request_limit: int = 8000
    request_workers: int = 8000
    coordinator_workers: int = 6000
    http_connections: int = 8000
    start_rate: float = 50.0
    request_timeout: float = 660.0
    retry_delay: float = 0.0

    def __post_init__(self):
        for name in ('request_limit', 'request_workers', 'coordinator_workers', 'http_connections'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('start_rate', 'request_timeout', 'retry_delay'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or
                    (name != 'retry_delay' and value == 0)):
                raise ValueError(f'{name} must be finite and positive (retry_delay may be zero)')


@dataclass(eq=False)
class _Request:
    operation: object
    context: object
    group: str
    identity: dict
    submitted: float
    result: Future = field(default_factory=Future)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    telemetry: dict = field(default_factory=dict)
    task: object = None
    eligible_at: float = 0.0
    pacing_at: float | None = None
    http_managed: bool = False
    deadline: object = None


class RequestDispatcher:
    def __init__(self, limits=None, *, stop_event=None, emit=None):
        self.limits = limits or RequestLimits()
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.emit = emit
        self._lock = threading.Lock()
        self._accepting = True
        self._closed = False
        self._fatal = None
        self._active = self._peak = self._completed = self._submitted = 0
        self._thread = threading.Thread(target=self._serve, name='deepeye-http-loop', daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait()

    @property
    def fatal_error(self):
        return self._fatal

    def _serve(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._wake = asyncio.Event()
        self._groups = {}
        self._rotation = deque()
        self._work = set()
        self._clients = []
        self._http = None
        self._http_pace_lock = asyncio.Lock()
        self._next_http_start = 0.0
        self._wire_pace_lock = asyncio.Lock()
        self._next_wire_start = 0.0
        self._next_start = 0.0
        self._pump_task = self._loop.create_task(self._pump())
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def _enqueue(self, work):
        self._work.add(work)
        retry_delay = self.limits.retry_delay if work.identity.get('sample_attempt', 1) > 1 else 0.0
        work.eligible_at = work.submitted + retry_delay
        if work.group not in self._groups:
            self._groups[work.group] = deque()
            self._rotation.append(work.group)
        self._groups[work.group].append(work)
        self._wake.set()

    def _pop(self):
        group = self._rotation.popleft()
        queue = self._groups[group]
        work = queue.popleft()
        if queue:
            self._rotation.append(group)
        else:
            del self._groups[group]
        return work

    def _reject(self, work, error):
        work.telemetry.update(acquired=False, released=False,
                              queue_wait_seconds=time.monotonic() - work.submitted,
                              pacing_wait_seconds=0.0, retry_wait_seconds=0.0,
                              service_seconds=0.0, error_type=type(error).__name__)
        self._work.discard(work)
        work.result.set_exception(error)

    async def _pump(self):
        from app.llm.sampling import SamplingPaused
        while True:
            self._wake.clear()
            if self.stop_event.is_set():
                while self._rotation:
                    self._reject(self._pop(), SamplingPaused('Request admission stopped'))
            delay = .025  # Observe an externally supplied run stop Event too.
            cap = min(self.limits.request_limit, self.limits.http_connections)
            if self._rotation and self._active < cap:
                now = time.monotonic()
                # Backoff lives in this queue, not in a worker holding a permit
                # or in the dispatcher head, so other groups can still start.
                candidate = None
                for _ in range(len(self._rotation)):
                    queue = self._groups[self._rotation[0]]
                    for work in queue:
                        if work.eligible_at <= now:
                            candidate = work
                            break
                        delay = min(delay, work.eligible_at - now)
                    if candidate is not None:
                        # A delayed retry must not block ready first attempts
                        # from its own group. Preserve all other queue order.
                        if queue[0] is not candidate:
                            queue.remove(candidate)
                            queue.appendleft(candidate)
                        break
                    self._rotation.rotate(-1)
                if candidate is None:
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=delay)
                    except TimeoutError:
                        pass
                    continue
                if candidate.pacing_at is None:
                    candidate.pacing_at = now
                remaining = self._next_start - time.monotonic()
                if remaining <= 0:
                    work = self._pop()
                    ready = asyncio.Event()
                    work.task = self._loop.create_task(self._execute(work, ready), context=work.context)
                    def completed(task, work=work, ready=ready):
                        # Cancellation can happen before a coroutine enters its
                        # try/finally. Its awaiting thread still needs a terminal.
                        if not work.result.done():
                            error = (SamplingPaused('Request cancelled before transport entry')
                                     if task.cancelled() else task.exception())
                            self._reject(work, error or RuntimeError('Request task exited without a result'))
                        ready.set()
                    work.task.add_done_callback(completed)
                    # Actual coroutine start, not submit time, anchors pacing.
                    await ready.wait()
                    continue
                delay = min(delay, remaining)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _execute(self, work, ready):
        from app.llm.sampling import SamplingPaused
        from openai import APITimeoutError
        import httpx2
        error = None
        result = None
        started = None
        try:
            # Optional retry backoff already completed before admission.
            retry_wait = self.limits.retry_delay if work.identity.get('sample_attempt', 1) > 1 else 0.0
            if self.stop_event.is_set():
                raise SamplingPaused('Request admission stopped')
            started = time.monotonic()
            self._next_start = started + 1.0 / self.limits.start_rate
            self._active += 1
            self._peak = max(self._peak, self._active)
            waiting = max(0.0, work.pacing_at - work.submitted - retry_wait)
            work.telemetry.update(acquired=True, released=False, started_at=started,
                                  queue_wait_seconds=waiting,
                                  pacing_wait_seconds=started - work.pacing_at,
                                  retry_wait_seconds=retry_wait)
            ready.set()
            try:
                # Real HTTP starts its deadline at the pre-send hook, after
                # request construction and any final HTTP pacing wait.
                async with asyncio.timeout(None if work.http_managed else self.limits.request_timeout) as deadline:
                    work.deadline = deadline
                    token = _TRANSPORT_REQUEST.set(work)
                    try:
                        result = await work.operation()
                    finally:
                        _TRANSPORT_REQUEST.reset(token)
            except TimeoutError as exc:
                raise APITimeoutError(request=httpx2.Request('POST', 'https://deadline.invalid')) from exc
        except asyncio.CancelledError:
            error = SamplingPaused('Active request cancelled; client transport drained')
        except BaseException as exc:
            error = exc
            if (AdaptiveAdmission._status_code(exc) in (400, 401, 403, 404, 422)
                    or isinstance(exc, (TypeError, ValueError))):
                self._fatal = exc
                self.stop_event.set()
        finally:
            ended = time.monotonic()
            if started is not None:
                self._active -= 1
                self._completed += 1
                work.telemetry.update(released=True, finished_at=ended,
                                      service_seconds=ended - work.telemetry.get('http_started_at', started))
            else:
                work.telemetry.update(acquired=False, released=False, service_seconds=0.0,
                                      queue_wait_seconds=ended - work.submitted,
                                      pacing_wait_seconds=0.0, retry_wait_seconds=0.0)
            work.telemetry['error_type'] = type(error).__name__ if error else None
            self._work.discard(work)
            self._wake.set()
            ready.set()
            if error is not None:
                work.result.set_exception(error)
            else:
                work.result.set_result(result)

    def call(self, operation, *, _http_managed=False):
        """Execute one async API attempt; retry/parsing/persistence stay in C1/C2."""
        from app.llm.sampling import sampling_identity, SamplingPaused
        if threading.current_thread() is self._thread:
            raise RuntimeError('Synchronous dispatch cannot run on its HTTP loop')
        identity = sampling_identity()
        work = _Request(operation, contextvars.copy_context(),
                        identity.get('group_id') or uuid.uuid4().hex, identity, time.monotonic(),
                        http_managed=_http_managed)
        with self._lock:
            if not self._accepting or self.stop_event.is_set():
                raise SamplingPaused('Request dispatcher stopped')
            self._submitted += 1
            self._loop.call_soon_threadsafe(self._enqueue, work)
        try:
            return work.result.result()
        except BaseException:
            if not work.result.done():
                self._loop.call_soon_threadsafe(self._cancel, work)
                # Never treat Future cancellation as proof of HTTP termination.
                while not work.result.done():
                    try:
                        work.result.result()
                    except BaseException:
                        continue
            raise
        finally:
            if self.emit is not None:
                self.emit('request_dispatch', {**identity, **work.telemetry,
                          'request_id': work.request_id, 'submitted_at': work.submitted})

    def _cancel(self, work):
        from app.llm.sampling import SamplingPaused
        if work not in self._work:
            return
        if work.task is not None:
            if not work.task.cancelling():
                work.task.cancel()
        else:
            queue = self._groups[work.group]
            queue.remove(work)
            if not queue:
                del self._groups[work.group]
                self._rotation.remove(work.group)
            self._reject(work, SamplingPaused('Queued request cancelled'))

    def stop(self, *, cancel_active=False):
        self.stop_event.set()
        with self._lock:
            if self._closed:
                return
            def apply():
                if cancel_active:
                    for work in tuple(self._work):
                        self._cancel(work)
                self._wake.set()
            self._loop.call_soon_threadsafe(apply)

    def snapshot(self):
        return dict(request_limit=self.limits.request_limit,
                    http_connections=self.limits.http_connections,
                    in_flight=self._active, peak_in_flight=self._peak,
                    submitted=self._submitted, completed=self._completed)

    def make_client(self, *, api_key, base_url, api_type='openai', api_version=None):
        """Native-compatible chat facade; credentials and HTTP pools stay run-local.

        C4 installs the facade on the LLM before recorder instrumentation. The
        facade preserves request kwargs and returns the actual SDK response.
        Its close marks the facade closed; the runtime drains/closes all owned
        async clients together after sample checkpointing has completed.
        """
        import httpx2
        from openai import AsyncOpenAI, AsyncAzureOpenAI
        if api_type not in ('openai', 'azure'):
            self.stop_event.set()
            raise ValueError(f'Unsupported api type: {api_type}')
        async def create_client():
            if self._http is None:
                self._http = httpx2.AsyncClient(timeout=None, limits=httpx2.Limits(
                    max_connections=self.limits.http_connections,
                    max_keepalive_connections=self.limits.http_connections),
                    event_hooks={'request': [self._before_http_send]})
            params = dict(api_key=api_key, base_url=base_url, max_retries=0,
                          timeout=self.limits.request_timeout, http_client=self._http)
            try:
                client = (AsyncOpenAI(**params) if api_type == 'openai' else
                          AsyncAzureOpenAI(**params, api_version=api_version))
            except BaseException:
                self.stop_event.set()
                raise
            self._clients.append(client)
            return client
        with self._lock:
            if not self._accepting:
                raise RuntimeError('Request dispatcher closed')
            future = asyncio.run_coroutine_threadsafe(create_client(), self._loop)
        client = future.result()
        return _ClientFacade(self, client)

    async def _before_http_send(self, request):
        """Pace at httpx's pre-send boundary, never from submission/disk timing."""
        from app.llm.sampling import SamplingPaused
        work = _TRANSPORT_REQUEST.get()
        if work is None:
            raise RuntimeError('HTTP transport escaped request admission')
        waiting_at = time.monotonic()
        async with self._http_pace_lock:
            remaining = self._next_http_start - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            if self.stop_event.is_set():
                raise SamplingPaused('HTTP request stopped before send')
            now = time.monotonic()
            self._next_http_start = now + 1.0 / self.limits.start_rate
            work.telemetry.update(http_started_at=now,
                                  http_pacing_wait_seconds=now - waiting_at,
                                  request_prepare_seconds=waiting_at - work.telemetry['started_at'])
            work.deadline.reschedule(now + self.limits.request_timeout)
        previous_trace = request.extensions.get('trace')
        async def trace(name, info):
            if previous_trace is not None:
                await previous_trace(name, info)
            if name in ('http11.send_request_headers.started', 'http2.send_request_headers.started'):
                waiting_at = time.monotonic()
                async with self._wire_pace_lock:
                    remaining = self._next_wire_start - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    if self.stop_event.is_set():
                        raise SamplingPaused('Request stopped before headers send')
                    now = time.monotonic()
                    self._next_wire_start = now + 1.0 / self.limits.start_rate
                    work.telemetry.update(wire_started_at=now,
                                          wire_pacing_wait_seconds=now - waiting_at)
            elif name in ('http11.send_request_headers.complete', 'http2.send_request_headers.complete'):
                work.telemetry['wire_headers_sent_at'] = time.monotonic()
        request.extensions['trace'] = trace

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._accepting = False
            self._closed = True
        async def drain():
            while self._work:
                await asyncio.sleep(.01)
            for client in self._clients:
                await client.close()
            if self._http is not None:
                await self._http.aclose()
            self._pump_task.cancel()
            await asyncio.gather(self._pump_task, return_exceptions=True)
        try:
            asyncio.run_coroutine_threadsafe(drain(), self._loop).result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()


class _ClientFacade:
    def __init__(self, dispatcher, client):
        self._dispatcher, self._client = dispatcher, client
        self._closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        async def operation():
            if self._closed:
                raise ValueError('Run-owned model client is closed')
            if kwargs.get('stream'):
                raise ValueError('Streaming responses cannot leave the bounded transport lifetime')
            return await self._client.chat.completions.create(*args, **kwargs)
        return self._dispatcher.call(operation, _http_managed=True)

    def close(self):
        self._closed = True
