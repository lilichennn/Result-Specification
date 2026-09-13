"""Repeatable, bounded offline capacity exercise through the real C3/C2 path.

CLI example: python -m tests.deepeye_runtime_capacity --requests 256 --coordinators 128
The deliberately raised start rate is ONLY for reaching a finite overlap barrier.
No model/PG credentials or remote URLs are accepted by this helper.
"""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time

from tests.test_deepeye_sampling import response
from tests.test_deepeye_sampling_parallel import parse_message
from app.llm.sampling import execute_group
from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder


def run_capacity(*, request_cap=64, coordinator_cap=32, transport='fake', deadline=60):
    if transport not in ('fake', 'http'):
        raise ValueError('Only offline fake or loopback http transports are accepted')
    if min(request_cap, coordinator_cap) < 1:
        raise ValueError('Capacity must be positive')
    budgets = [1, 4, 5]
    group_sizes = [budgets[i % 3] for i in range(coordinator_cap)]
    if sum(group_sizes) < request_cap:
        raise ValueError('Coordinator groups must provide enough samples to reach the request barrier')
    started = time.monotonic()
    server = None
    if transport == 'http':
        from tests.test_deepeye_request_dispatch import LoopbackServer
        server = LoopbackServer(['barrier'] * sum(group_sizes), barrier=request_cap)
    try:
        with tempfile.TemporaryDirectory(prefix='deepeye-c3-capacity-') as temp:
            with RunStore.create(Path(temp) / 'run', {'offline_capacity': True}) as store:
                stop = threading.Event()
                recorder = TraceRecorder(store, stop_event=stop)
                runtime = SamplingRuntime(request_limit=request_cap, request_workers=request_cap,
                    http_connections=request_cap, coordinator_workers=coordinator_cap,
                    start_rate=100000, request_timeout=deadline, stop_event=stop,
                    emit=recorder.record_admission)
                futures = []
                completed = False
                result = None
                try:
                    active = peak = 0
                    release = asyncio.Event()
                    barrier_threads = {}
                    async def fake():
                        nonlocal active, peak
                        active += 1
                        peak = max(peak, active)
                        try:
                            if active >= request_cap:
                                # One coroutine waits for the joint capacity
                                # condition. Releasing on requests alone would
                                # not prove that all coordinators coexist.
                                while runtime.snapshot()['coordinators']['active'] < coordinator_cap:
                                    await asyncio.sleep(.01)
                                names = [thread.name for thread in threading.enumerate()]
                                barrier_threads.update(
                                    sample_threads=sum(name.startswith('deepeye-sample') for name in names),
                                    coordinator_threads=sum(name.startswith('deepeye-coordinator') for name in names),
                                    http_loop_threads=sum(name == 'deepeye-http-loop' for name in names),
                                    joint_request_inflight=active,
                                    joint_coordinator_active=runtime.snapshot()['coordinators']['active'])
                                release.set()
                            await release.wait()
                            await asyncio.sleep(.002)
                            return response()
                        finally:
                            active -= 1
                    if server is None:
                        request = recorder._api_wrapper(lambda: runtime.dispatch.call(fake))
                    else:
                        client = runtime.make_client(api_key='offline', base_url=server.url)
                        request = recorder._api_wrapper(lambda: client.chat.completions.create(
                            model='fixture', messages=[], n=1, timeout=deadline))
                    attempts = [store.begin_attempt(f'lite/{i}', 'schema_linking', 'input')
                                for i in range(len(group_sizes))]
                    def coordinate(attempt, n):
                        with recorder.context(attempt), runtime.context():
                            return execute_group(request, parse_message, n=n,
                                recovery_identity={'offline_capacity_request': True})
                    for attempt, n in zip(attempts, group_sizes):
                        futures.append(runtime.submit_coordinator(coordinate, attempt, n))
                    outcomes = [future.result(deadline + 10) for future in futures]
                    snapshot = runtime.snapshot()
                    result = dict(transport=transport, request_cap=request_cap,
                        coordinator_cap=coordinator_cap,
                        actual_request_peak=snapshot['requests']['peak_in_flight'],
                        actual_connection_peak=server.peak if server else None,
                        sample_worker_peak=snapshot['samples']['peak'],
                        coordinator_peak=snapshot['coordinators']['peak'],
                        group_sizes=budgets, total_groups=len(group_sizes),
                        total_samples=sum(group_sizes), all_groups_complete=all(g.complete for g in outcomes),
                        checkpoint_successes=sum(1 for _ in store.iter_events(kinds='sample_checkpoint')),
                        request_trace_events=sum(1 for _ in store.iter_events(kinds='request_dispatch')),
                        final_inflight=snapshot['requests']['in_flight'],
                        store_verified=store.verify()['ok'],
                        capacity_only_start_rate=100000, production_start_rate=50,
                        elapsed_seconds=time.monotonic() - started,
                        barrier_threads=barrier_threads)
                    completed = True
                    return result
                finally:
                    # Partial submit failure can make the joint barrier
                    # unreachable. Stop on every abnormal exit, even when the
                    # failing submission did not return a Future to append.
                    if not completed:
                        runtime.stop(cancel_active=True)
                    runtime.close()
                    if result is not None:
                        result['cleanup'] = dict(
                            http_loop_stopped=not runtime.dispatch._thread.is_alive(),
                            sample_threads_stopped=all(not t.is_alive() for t in runtime.samples._pool._threads),
                            coordinator_threads_stopped=all(not t.is_alive() for t in runtime.coordinators._threads),
                            final_inflight=runtime.dispatch.snapshot()['in_flight'])
    finally:
        if server is not None:
            server.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=64)
    parser.add_argument('--coordinators', type=int, default=32)
    parser.add_argument('--transport', choices=('fake', 'http'), default='fake')
    parser.add_argument('--deadline', type=float, default=60)
    args = parser.parse_args()
    print(json.dumps(run_capacity(request_cap=args.requests, coordinator_cap=args.coordinators,
        transport=args.transport, deadline=args.deadline), indent=2, sort_keys=True))
