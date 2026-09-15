"""Real dispatch path with cancellable async attempts; no remote inference."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import importlib.util
import threading
import time
import unittest
from pathlib import Path
import json
import tempfile

from tests.test_deepeye_sampling import response
from app.llm.sampling import execute_group, sampling_identity
from tests.test_deepeye_sampling_parallel import parse_message


class RuntimeTestCase(unittest.TestCase):
    def runtime(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('scripts.baseline_adapters.deepeye.request_dispatch'),
                             'C3 cancel-safe request dispatch is missing')
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        runtime = SamplingRuntime(**kwargs)
        self.addCleanup(runtime.close)
        return runtime


class DispatchTests(RuntimeTestCase):

    def test_retry_queued_first_does_not_block_ready_first_attempt_in_same_group(self):
        runtime = self.runtime(request_limit=2, request_workers=2,
                               start_rate=10000, retry_delay=.30)
        retry_enqueued = threading.Event()
        starts = []
        original_enqueue = runtime.dispatch._enqueue
        def enqueue(work):
            original_enqueue(work)
            if work.identity.get('sample_index') == 0 and work.identity.get('sample_attempt') == 2:
                retry_enqueued.set()
        runtime.dispatch._enqueue = enqueue
        async def api():
            identity = sampling_identity()
            starts.append((identity['sample_index'], identity['sample_attempt']))
            return response('bad' if identity['sample_index'] == 0 and
                            identity['sample_attempt'] == 1 else 'SELECT 1')
        def request():
            if sampling_identity()['sample_index'] == 1:
                self.assertTrue(retry_enqueued.wait(2))
            return runtime.dispatch.call(api)
        with runtime.context():
            group = execute_group(request, parse_message, n=2)
        self.assertTrue(group.complete)
        self.assertEqual(starts, [(0, 1), (1, 1), (0, 2)])

    def test_defaults_are_explicit_and_invalid_limits_reject(self):
        runtime = self.runtime()
        self.assertEqual(runtime.limits.request_limit, 8000)
        self.assertEqual(runtime.limits.request_workers, 8000)
        self.assertEqual(runtime.limits.coordinator_workers, 6000)
        self.assertEqual(runtime.limits.http_connections, 8000)
        self.assertEqual(runtime.limits.start_rate, 50)
        self.assertEqual(runtime.limits.request_timeout, 660)
        for kwargs in ({'request_limit': 0}, {'request_workers': True},
                       {'start_rate': float('nan')}, {'request_timeout': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.runtime(**kwargs)

    def test_all_attempts_share_capacity_and_context_with_distinct_wait_metrics(self):
        observed = []
        runtime = self.runtime(request_workers=5, request_limit=2, start_rate=100,
                               emit=lambda kind, payload: observed.append((kind, payload)))
        marker = ContextVar('dispatch_rc_context')
        marker.set('exact rc')
        active = peak = 0
        async def api():
            nonlocal active, peak
            self.assertEqual(marker.get(), 'exact rc')
            active += 1
            peak = max(peak, active)
            identity = sampling_identity()
            try:
                await asyncio.sleep(.035)
                return response('bad' if identity['sample_index'] == 1 and
                                identity['sample_attempt'] == 1 else 'SELECT 1')
            finally:
                active -= 1
        with runtime.context():
            group = execute_group(lambda: runtime.dispatch.call(api), parse_message, n=5)
        self.assertTrue(group.complete)
        self.assertEqual(peak, 2)
        events = [p for k, p in observed if k == 'request_dispatch']
        self.assertEqual(len(events), 6)
        self.assertEqual(len({p['request_id'] for p in events}), 6)
        for payload in events:
            for key in ('queue_wait_seconds', 'pacing_wait_seconds', 'retry_wait_seconds', 'service_seconds'):
                self.assertGreaterEqual(payload[key], 0)
            self.assertTrue(payload['acquired'])
            self.assertTrue(payload['released'])
        self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 0)
        self.assertTrue(any(payload['pacing_wait_seconds'] > 0 for payload in events))

    def test_total_deadline_cancels_underlying_attempt_and_retry_succeeds(self):
        runtime = self.runtime(request_limit=1, request_workers=1, start_rate=10000,
                               request_timeout=.04)
        cancelled = threading.Event()
        calls = 0
        async def api():
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return response()
        with runtime.context():
            group = execute_group(lambda: runtime.dispatch.call(api), parse_message, n=1)
        self.assertTrue(group.complete)
        self.assertEqual(calls, 2)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(group.samples[0].attempts), 2)
        self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 0)

    def test_stop_cancels_inflight_and_rejects_new_submissions(self):
        from app.llm.sampling import SamplingPaused
        runtime = self.runtime(request_limit=1, start_rate=10000)
        started, ended = threading.Event(), threading.Event()
        async def api():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        with ThreadPoolExecutor(1) as callers:
            future = callers.submit(runtime.dispatch.call, api)
            self.assertTrue(started.wait(2))
            runtime.stop(cancel_active=True)
            with self.assertRaises(SamplingPaused):
                future.result(2)
        self.assertTrue(ended.is_set())
        self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 0)
        with self.assertRaises(SamplingPaused):
            runtime.dispatch.call(api)

    def test_production_pacing_has_no_catchup_burst_after_loop_delay(self):
        runtime = self.runtime(request_limit=8, start_rate=50)
        starts = []
        async def api():
            starts.append(time.monotonic())
            if len(starts) == 1:
                time.sleep(.07)  # Deliberate local scheduling pause, not network delay.
            return response()
        with ThreadPoolExecutor(8) as callers:
            futures = [callers.submit(runtime.dispatch.call, api) for _ in range(8)]
            for future in futures:
                future.result(3)
        self.assertGreaterEqual(min(b - a for a, b in zip(starts, starts[1:])), .017)

    def test_retry_wait_does_not_delay_other_ready_samples(self):
        runtime = self.runtime(request_limit=2, request_workers=3, start_rate=10000,
                               retry_delay=.15)
        starts = []
        async def api():
            identity = sampling_identity()
            starts.append((identity['sample_index'], identity['sample_attempt'], time.monotonic()))
            await asyncio.sleep(.01)
            return response('bad' if identity['sample_index'] == 0 and
                            identity['sample_attempt'] == 1 else 'SELECT 1')
        with runtime.context():
            group = execute_group(lambda: runtime.dispatch.call(api), parse_message, n=3)
        self.assertTrue(group.complete)
        retry_time = next(t for i, a, t in starts if i == 0 and a == 2)
        self.assertTrue(all(t < retry_time for i, a, t in starts if i != 0))

    def test_close_drains_active_transports_without_stopping_other_runs(self):
        runtime = self.runtime(request_limit=1, start_rate=10000)
        other = self.runtime(request_limit=1, start_rate=10000)
        started = threading.Event()
        finished = threading.Event()
        async def api():
            started.set()
            await asyncio.sleep(.06)
            finished.set()
            return response()
        with ThreadPoolExecutor(1) as callers:
            future = callers.submit(runtime.dispatch.call, api)
            self.assertTrue(started.wait(2))
            runtime.close()
            self.assertTrue(finished.is_set())
            self.assertEqual(future.result().id, 'fixture')
        self.assertEqual(other.dispatch.call(api).id, 'fixture')
        self.assertFalse(other.stop_event.is_set())

    def test_cancel_before_transport_coroutine_enters_does_not_orphan_permit_waiter(self):
        from app.llm.sampling import SamplingPaused
        runtime = self.runtime(request_limit=1, start_rate=10000)
        gate = threading.Event()
        captured = []
        original = runtime.dispatch._execute
        async def delayed(work, ready):
            captured.append((work, ready))
            gate.set()
            await asyncio.sleep(1)
            return await original(work, ready)
        runtime.dispatch._execute = delayed
        async def api():
            return response()
        with ThreadPoolExecutor(1) as callers:
            future = callers.submit(runtime.dispatch.call, api)
            self.assertTrue(gate.wait(2))
            runtime.stop(cancel_active=True)
            deadline = time.monotonic() + .25
            while not future.done() and time.monotonic() < deadline:
                time.sleep(.005)
            completed = future.done()
            if not completed:
                # Unstick the pre-fix implementation so RED cannot hang teardown.
                def rescue():
                    work, ready = captured[0]
                    runtime.dispatch._reject(work, SamplingPaused('fixture cleanup'))
                    ready.set()
                runtime.dispatch._loop.call_soon_threadsafe(rescue)
            with self.assertRaises(SamplingPaused):
                future.result(2)
        self.assertTrue(completed, 'Cancellation orphaned a request before coroutine entry')


class LoopbackServer:
    """One bounded async fixture thread, not a thread per accepted connection."""
    def __init__(self, modes, barrier=0):
        self.modes = list(modes)
        self.barrier = barrier
        self.requests, self.starts = [], []
        self.disconnected = []
        self.active = self.peak = 0
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()
        self.ready.wait(2)

    def serve(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.handlers = set()
        self.release = asyncio.Event()
        self.server = self.loop.run_until_complete(asyncio.start_server(self.handle, '127.0.0.1', 0))
        self.url = f'http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1'
        self.ready.set()
        self.loop.run_forever()
        self.loop.close()

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        index = None
        try:
            while True:
                try:
                    headers = await reader.readuntil(b'\r\n\r\n')
                except asyncio.IncompleteReadError:
                    break
                length = next(int(line.split(b':', 1)[1]) for line in headers.split(b'\r\n')
                              if line.lower().startswith(b'content-length:'))
                body = await reader.readexactly(length)
                index = len(self.requests)
                self.requests.append(json.loads(body))
                self.starts.append(time.monotonic())
                self.active += 1
                self.peak = max(self.peak, self.active)
                mode = self.modes[index] if index < len(self.modes) else 'ok'
                try:
                    if mode == 'barrier':
                        if self.active >= self.barrier:
                            self.release.set()
                        await self.release.wait()
                        await asyncio.sleep(.002)
                    if mode == 'no_headers':
                        await reader.read()
                        break
                    if mode == 'trickle':
                        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n')
                        while not reader.at_eof():
                            writer.write(b'1\r\n \r\n')
                            await writer.drain()
                            await asyncio.sleep(.01)
                        break
                    status = b'500 Internal Server Error' if mode == 'error' else (
                        b'401 Unauthorized' if mode == 'auth' else b'200 OK')
                    payload = (json.dumps({'error': {'message': mode, 'type': 'fixture'}}).encode()
                               if mode in ('error', 'auth') else response().model_dump_json().encode())
                    if mode in ('output_inspection', 'input_inspection', 'invalid_parameter'):
                        status = b'400 Bad Request'
                        payload = json.dumps({'error': {
                            'code': 'invalid_parameter' if mode == 'invalid_parameter' else 'data_inspection_failed',
                            'type': 'data_inspection_failed', 'param': None,
                            'message': ('Input' if mode == 'input_inspection' else 'Output')
                                       + ' data may contain inappropriate content.'}}).encode()
                    writer.write(b'HTTP/1.1 ' + status + b'\r\nContent-Type: application/json\r\nContent-Length: '
                                 + str(len(payload)).encode() + b'\r\n\r\n' + payload)
                    await writer.drain()
                finally:
                    self.active -= 1
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            if index is not None:
                self.disconnected.append(index)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.handlers.discard(task)

    def close(self):
        async def shutdown():
            self.server.close()
            await self.server.wait_closed()
            for task in tuple(self.handlers):
                task.cancel()
            await asyncio.gather(*self.handlers, return_exceptions=True)
        asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(3)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(3)


class LoopbackHTTPTests(RuntimeTestCase):
    def test_output_inspection_retries_without_stopping_runwide_admission(self):
        from app.llm.sampling import SamplingPaused
        server = LoopbackServer(['output_inspection', 'ok'])
        self.addCleanup(server.close)
        runtime = self.runtime(request_limit=1, request_workers=1, start_rate=10000)
        client = runtime.make_client(api_key='offline', base_url=server.url)
        params = dict(model='fixture', messages=[{'role': 'user', 'content': 'SELECT task'}],
                      temperature=.6, max_tokens=16384, n=1)
        try:
            with runtime.context():
                group = execute_group(lambda: client.chat.completions.create(**params), parse_message, n=1)
        except SamplingPaused:
            self.fail('Output inspection must not stop the shared dispatcher')
        self.assertTrue(group.complete)
        self.assertEqual(len(server.requests), 2)
        self.assertEqual(server.requests, [params, params])
        self.assertEqual(len(group.samples[0].attempts), 2)
        self.assertIsNone(runtime.dispatch.fatal_error)
        self.assertFalse(runtime.stop_event.is_set())
        self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 0)

    def test_four_output_rejections_exhaust_sample_but_allow_other_requests(self):
        from app.llm.sampling import SamplingPaused
        server = LoopbackServer(['output_inspection'] * 4 + ['ok'])
        self.addCleanup(server.close)
        runtime = self.runtime(request_limit=1, request_workers=1, start_rate=10000)
        client = runtime.make_client(api_key='offline', base_url=server.url)
        try:
            with runtime.context():
                group = execute_group(lambda: client.chat.completions.create(model='fixture', messages=[]),
                                      parse_message, n=1)
        except SamplingPaused:
            self.fail('Exhaustion of output-inspection retries must stay local to the sample')
        self.assertFalse(group.complete)
        self.assertEqual(len(server.requests), 4)
        self.assertEqual(len(group.samples[0].attempts), 4)
        self.assertFalse(group.samples[0].fatal)
        self.assertEqual(client.chat.completions.create(model='fixture', messages=[]).id, 'fixture')
        self.assertEqual(len(server.requests), 5)
        self.assertFalse(runtime.stop_event.is_set())

    def test_input_inspection_and_other_bad_requests_still_stop_admission(self):
        from app.llm.sampling import SamplingPaused
        for mode in ('input_inspection', 'invalid_parameter'):
            with self.subTest(mode=mode):
                server = LoopbackServer([mode, 'ok'])
                self.addCleanup(server.close)
                runtime = self.runtime(request_limit=1, request_workers=1, start_rate=10000)
                client = runtime.make_client(api_key='offline', base_url=server.url)
                with runtime.context():
                    group = execute_group(lambda: client.chat.completions.create(model='fixture', messages=[]),
                                          parse_message, n=1)
                self.assertFalse(group.complete)
                self.assertTrue(group.samples[0].fatal)
                self.assertEqual(len(server.requests), 1)
                self.assertTrue(runtime.stop_event.is_set())
                self.assertEqual(runtime.dispatch.fatal_error.status_code, 400)
                with self.assertRaises(SamplingPaused):
                    client.chat.completions.create(model='fixture', messages=[])

    def test_stalled_connect_does_not_block_other_ready_requests(self):
        server = LoopbackServer([])
        self.addCleanup(server.close)
        runtime = self.runtime(request_limit=2, start_rate=50)
        first_connect = threading.Event()
        release = asyncio.Event()
        intercepted = False
        original = runtime.dispatch._before_http_send
        async def before_send(request):
            await original(request)
            previous = request.extensions['trace']
            async def trace(name, info):
                nonlocal intercepted
                if name == 'connection.connect_tcp.started' and not intercepted:
                    intercepted = True
                    first_connect.set()
                    await release.wait()
                await previous(name, info)
            request.extensions['trace'] = trace
        runtime.dispatch._before_http_send = before_send
        client = runtime.make_client(api_key='offline', base_url=server.url)
        with ThreadPoolExecutor(1) as callers:
            first = callers.submit(client.chat.completions.create, model='fixture', messages=[])
            try:
                self.assertTrue(first_connect.wait(2))
                second = client.chat.completions.create(model='fixture', messages=[])
                self.assertEqual(second.id, 'fixture')
                self.assertFalse(first.done())
            finally:
                runtime.dispatch._loop.call_soon_threadsafe(release.set)
            self.assertEqual(first.result(2).id, 'fixture')
        self.assertEqual(len(server.requests), 2)

    def test_headers_and_trickling_body_have_total_deadline_and_release_socket(self):
        from openai import APITimeoutError
        for mode in ('no_headers', 'trickle'):
            with self.subTest(mode=mode):
                server = LoopbackServer([mode, 'ok'])
                self.addCleanup(server.close)
                runtime = self.runtime(request_limit=1, http_connections=1, request_workers=1,
                                       request_timeout=.15, start_rate=10000)
                self.assertTrue(hasattr(runtime, 'make_client'), 'Native async transport facade is missing')
                client = runtime.make_client(api_key='offline', base_url=server.url)
                # Warm up local imports/client construction outside the deadline.
                started = time.monotonic()
                with self.assertRaises(APITimeoutError):
                    client.chat.completions.create(model='fixture', messages=[],
                                                   timeout=10 if mode == 'no_headers' else .05)
                self.assertLess(time.monotonic() - started, .7)
                self.assertGreaterEqual(time.monotonic() - started, .12)
                deadline = time.monotonic() + 1
                while 0 not in server.disconnected and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertIn(0, server.disconnected)
                self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 0)
                value = client.chat.completions.create(model='fixture', messages=[], timeout=.05)
                self.assertEqual(value.id, 'fixture')
                self.assertEqual(len(server.requests), 2)
                runtime.close()

    def test_http_deadline_retries_once_and_successes_restore_without_new_requests(self):
        from scripts.baseline_adapters.deepeye.run_store import RunStore
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from tests.test_deepeye_sampling import llm_fixture, parse
        from app.llm_extractor import LLMExtractor
        server = LoopbackServer(['trickle', 'ok', 'ok', 'ok'])
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {'fixture': True}) as store:
            stop = threading.Event()
            recorder = TraceRecorder(store, stop_event=stop)
            runtime = self.runtime(request_limit=1, http_connections=1, request_workers=3,
                                   start_rate=10000, request_timeout=.12, stop_event=stop,
                                   emit=recorder.record_admission)
            self.assertTrue(hasattr(runtime, 'make_client'), 'Native async transport facade is missing')
            llm, _ = llm_fixture([])
            llm._client = runtime.make_client(api_key='offline', base_url=server.url)
            original = llm._client.chat.completions.create
            llm._client.chat.completions.create = recorder._api_wrapper(original)
            for invocation in range(2):
                attempt = store.begin_attempt('lite/one', 'schema_linking', 'input')
                with recorder.context(attempt), runtime.context():
                    values, usage = LLMExtractor().extract_with_retry(llm, [], parse, n=3)
                self.assertEqual((len(values), usage['total_tokens']), (3, 90))
                self.assertEqual(len(server.requests), 4)
                dispatch = list(store.iter_events(attempt, kinds='request_dispatch'))
                self.assertEqual(len(dispatch), 4 if invocation == 0 else 0)
                if invocation == 0:
                    self.assertTrue(all(e['payload']['call_id'] for e in dispatch))
                    self.assertTrue(all('sample_index' in e['payload'] for e in dispatch))
                    self.assertTrue(all(e['payload'].get('http_started_at') for e in dispatch))
            checkpoints = list(store.iter_events(kinds='sample_checkpoint'))
            self.assertEqual(len(checkpoints), 3)
            self.assertEqual(len({(e['payload']['group_id'], e['payload']['sample_index']) for e in checkpoints}), 3)
            self.assertEqual(len(list(store.iter_events(kinds='api_error'))), 1)
            timings = list(store.iter_events(kinds='sample_timing'))
            self.assertEqual(len(timings), 6)
            self.assertTrue(all(e['payload']['persistence_wait_seconds'] > 0 for e in timings))
            api_timings = list(store.iter_events(kinds='api_persistence_timing'))
            self.assertEqual(len(api_timings), 4)
            self.assertTrue(all(e['payload']['persistence_wait_seconds'] > 0 for e in api_timings))
            self.assertTrue(store.verify()['ok'])
            runtime.close()

    def test_sdk_does_not_retry_inside_one_permitted_attempt(self):
        from openai import InternalServerError
        server = LoopbackServer(['error', 'ok'])
        self.addCleanup(server.close)
        runtime = self.runtime(start_rate=10000)
        client = runtime.make_client(api_key='offline', base_url=server.url)
        with self.assertRaises(InternalServerError):
            client.chat.completions.create(model='fixture', messages=[])
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(runtime.dispatch.snapshot()['completed'], 1)

    def test_authentication_failure_stops_new_runwide_admission(self):
        from app.llm.sampling import SamplingPaused
        server = LoopbackServer(['auth'])
        self.addCleanup(server.close)
        stop = threading.Event()
        runtime = self.runtime(request_limit=1, request_workers=5, stop_event=stop)
        client = runtime.make_client(api_key='offline', base_url=server.url)
        with runtime.context(), self.assertRaises(SamplingPaused):
            execute_group(lambda: client.chat.completions.create(model='fixture', messages=[]),
                          parse_message, n=5)
        self.assertEqual(len(server.requests), 1)
        self.assertTrue(stop.is_set())
        self.assertEqual(runtime.dispatch.fatal_error.status_code, 401)
        self.assertEqual(runtime.dispatch.snapshot()['in_flight'], 0)

    def test_actual_http_start_pacing_and_original_kwargs_are_preserved(self):
        server = LoopbackServer([])
        self.addCleanup(server.close)
        emitted = []
        def emit(kind, payload):
            self.assertNotEqual(threading.current_thread().name, 'deepeye-http-loop')
            emitted.append(payload)
        runtime = self.runtime(request_workers=5, emit=emit)
        client = runtime.make_client(api_key='offline', base_url=server.url)
        params = dict(model='fixture', messages=[{'role': 'user', 'content': 'fixed input'}],
                      temperature=.6, max_tokens=17000, reasoning_effort='high',
                      extra_body={'enable_thinking': True}, n=1)
        with runtime.context():
            result = execute_group(lambda: client.chat.completions.create(**params), parse_message, n=5)
        self.assertTrue(result.complete)
        self.assertEqual(len(server.requests), 5)
        self.assertTrue(all(p['temperature'] == .6 and p['max_tokens'] == 17000 and
                            p['reasoning_effort'] == 'high' and p['enable_thinking'] is True
                            for p in server.requests))
        starts = sorted(p.get('wire_started_at', 0) for p in emitted)
        self.assertGreater(starts[0], 0)
        self.assertGreaterEqual(min(b - a for a, b in zip(starts, starts[1:])), .019)
        self.assertGreaterEqual(min(b - a for a, b in zip(server.starts, server.starts[1:])), .015)
