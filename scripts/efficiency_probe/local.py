"""Thread barriers and a one-event-loop local HTTP fixture (no external traffic)."""
import asyncio
import concurrent.futures as cf
import math
import threading
import time
from pathlib import Path
from aiohttp import web
from .core import resources, RunStore, fingerprint, utc, distribution, write_json


def thread_probe(target, hold_seconds=2):
    release = threading.Event()
    condition = threading.Condition()
    waiting = 0
    threads = []
    before = resources()
    error = None
    def worker():
        nonlocal waiting
        with condition:
            waiting += 1
            condition.notify_all()
        release.wait()
    try:
        for _ in range(target):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        with condition:
            if not condition.wait_for(lambda: waiting == target, timeout=30):
                error = "barrier_timeout"
        peak = resources()
        release.wait(hold_seconds)
    except (RuntimeError, OSError) as e:
        error = type(e).__name__+": "+str(e)
        peak = resources()
    finally:
        release.set()
        for t in threads:
            t.join(timeout=30)
    return {"target": target, "peak_waiting": waiting, "joined": sum(not t.is_alive() for t in threads),
            "error": error, "before": before, "at_barrier": peak, "after": resources()}


class LocalServer:
    """Holds the first wave until genuine server-side concurrency reaches target.

    The finite barrier timeout exposes inability to reach a configured level.
    Only a single event loop serves all clients; no per-request server threads.
    """
    def __init__(self, target, status=200, delay=.02, barrier_timeout=30):
        self.target, self.status, self.delay = target, status, delay
        self.barrier_timeout = barrier_timeout
        self.received, self.active, self.peak = 0, 0, 0
        self.barrier_timed_out = False

    async def handle(self, request):
        await request.json()
        self.received += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active >= self.target:
            self.gate.set()
        try:
            try:
                await asyncio.wait_for(self.gate.wait(), self.barrier_timeout)
            except TimeoutError:
                self.barrier_timed_out = True
                self.gate.set()
            await asyncio.sleep(self.delay)
            if self.status != 200:
                return web.json_response({"error": {"message": "local fixture", "type": "fixture"}}, status=self.status)
            return web.json_response({"id": f"local-{self.received}", "object": "chat.completion", "created": 1,
                "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "<reasoning>"+"local fixture "*100+"</reasoning><result>SELECT 1</result>"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
        finally:
            self.active -= 1

    async def setup(self):
        self.gate = asyncio.Event()
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0, backlog=8192)
        await self.site.start()
        self.url = f"http://127.0.0.1:{self.site._server.sockets[0].getsockname()[1]}/v1"

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="probe-local-server", daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.setup(), self.loop).result(10)
        return self

    def __exit__(self, *_):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(40)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def group_probe(out, *, workers, concurrency, coordinators, groups, samples=5, delay=.05):
    """Synthetic local groups, NOT DeepEye or a remote model benchmark.

    Separate blocking group coordinators and sample workers. Each tenth sample
    waits 20x longer; each fiftieth fails after waiting. No automatic retry.
    Real RunStore persistence exposes coordination/queue/record contention.
    """
    if any(type(v) is not int or v <= 0 for v in (workers, concurrency, coordinators, groups, samples)):
        raise ValueError('all counts must be positive integers')
    if not math.isfinite(delay) or delay <= 0:
        raise ValueError('delay must be finite and positive')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(kind='local_group_scheduling_fixture', workers=workers,
                    concurrency=concurrency, coordinators=coordinators, groups=groups,
                    samples=samples, delay=delay, slow_every=10, slow_multiplier=20,
                    fail_every=50, created_at=utc(), actual_model_calls=0)
    store = RunStore.create(out, manifest)
    attempt = store.begin_attempt('capacity', 'local_groups', fingerprint(manifest))
    write_json(out/'manifest.json', manifest)
    lock, permits, finished_monitor = threading.Lock(), threading.BoundedSemaphore(concurrency), threading.Event()
    state = dict(active=0, peak=0, coordinators=0, peak_coordinators=0)
    rows, group_rows, monitor_rows = [], [], []
    started = time.perf_counter()

    def monitor():
        while not finished_monitor.is_set():
            r = resources()
            monitor_rows.append(r)
            store.append_event(attempt, 'resource', r)
            finished_monitor.wait(.25)

    def sample(number, submitted):
        entered = time.perf_counter()
        group_no, sample_no = divmod(number, samples)
        row = dict(request_no=number, group_no=group_no, sample_no=sample_no,
                   executor_wait_seconds=entered-submitted, started_at=utc())
        store.append_event(attempt, 'request', row)
        gate_started = time.perf_counter()
        with permits:
            row['permit_wait_seconds'] = time.perf_counter()-gate_started
            with lock:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
            try:
                time.sleep(delay*(20 if number % 10 == 9 else 1))
                row['success'] = number % 50 != 49
            finally:
                with lock:
                    state['active'] -= 1
        row['finished_at'] = utc()
        row['elapsed_seconds'] = time.perf_counter()-entered
        commit_started = time.perf_counter()
        store.append_event(attempt, 'response' if row['success'] else 'error', dict(row))
        row['commit_seconds'] = time.perf_counter()-commit_started
        with lock:
            rows.append(row)
        return row

    def group(number, submitted, pool):
        began = time.perf_counter()
        with lock:
            state['coordinators'] += 1
            state['peak_coordinators'] = max(state['peak_coordinators'], state['coordinators'])
        try:
            futures = [pool.submit(sample, number*samples+i, time.perf_counter()) for i in range(samples)]
            results = [f.result() for f in futures]
            result = dict(group_no=number, target=samples,
                          successful=sum(r['success'] for r in results),
                          coordinator_wait_seconds=began-submitted,
                          elapsed_seconds=time.perf_counter()-began)
            store.append_event(attempt, 'group_result', result)
            return result
        finally:
            with lock:
                state['coordinators'] -= 1

    m = threading.Thread(target=monitor, name='group-probe-resource', daemon=True)
    m.start()
    try:
        # A coordinator never waits on a task submitted to its own executor.
        with cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix='fixture-sample') as pool:
            with cf.ThreadPoolExecutor(max_workers=coordinators, thread_name_prefix='fixture-group') as outer:
                futures = [outer.submit(group, i, time.perf_counter(), pool) for i in range(groups)]
                group_rows = [f.result() for f in futures]
    finally:
        finished_monitor.set()
        m.join()
    elapsed = time.perf_counter()-started
    success = sum(r['success'] for r in rows)
    summary = dict(manifest=manifest, elapsed_seconds=elapsed, requests=len(rows),
                   retained_successful_samples=success, simulated_failures=len(rows)-success,
                   complete_groups=sum(g['successful']==samples for g in group_rows),
                   peak_model_requests=state['peak'], peak_coordinators=state['peak_coordinators'],
                   outstanding=groups*samples-len(rows), successful_samples_per_second=success/elapsed,
                   resources_peak={k:max(r[k] for r in monitor_rows) for k in ('threads','rss_bytes','fds','swap_used_bytes')},
                   resources_after=resources(),
                   executor_wait_seconds=distribution([r['executor_wait_seconds'] for r in rows]),
                   permit_wait_seconds=distribution([r['permit_wait_seconds'] for r in rows]),
                   coordinator_wait_seconds=distribution([g['coordinator_wait_seconds'] for g in group_rows]),
                   commit_seconds=distribution([r['commit_seconds'] for r in rows]))
    store.finish_attempt(attempt, 'succeeded' if not summary['outstanding'] else 'failed', summary)
    summary['integrity'] = store.verify()
    store.close()
    write_json(out/'summary.json', summary)
    return summary
