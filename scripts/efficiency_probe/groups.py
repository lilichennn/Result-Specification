"""Blocking group coordinators feeding the SAME bounded SDK admission queue.

This is a capacity probe, not a replacement DeepEye Pipeline. A group represents
one model-calling node and holds one coordinator until all its samples terminate.
"""
import concurrent.futures as cf
from collections import Counter
import queue
import threading
import time


def build_plan(workload, profile, count):
    """Repeat historical node inputs, retaining the official node sample size.

    Revision Syntax/Result nodes use five samples; its other model-calling
    checkers use one. This is a ready-node snapshot, NOT eight parallel checkers.
    """
    if profile not in ('generation', 'revision') or type(count) is not int or count <= 0:
        raise ValueError('choose generation/revision and a positive group count')
    choices = []
    for i, item in enumerate(workload):
        if item['stage'] != 'sql_'+profile:
            continue
        path = '.'.join(item.get('branch_path', []))
        n = 4 if profile == 'generation' else 5 if ('SyntaxChecker' in path or 'ResultChecker' in path) else 1
        choices.append(dict(workload_index=i, samples=n))
    if not choices:
        raise ValueError('workload has no matching model-calling nodes')
    return [dict(choices[i % len(choices)]) for i in range(count)]


def validate_plan(plan, workload, coordinators):
    if type(coordinators) is not int or coordinators <= 0:
        raise ValueError('coordinators must be a positive integer')
    if not isinstance(plan, list) or not plan:
        raise ValueError('group_plan must be a nonempty list')
    for spec in plan:
        i, n = spec.get('workload_index'), spec.get('samples')
        if type(i) is not int or not 0 <= i < len(workload):
            raise ValueError('invalid group workload_index')
        if type(n) is not int or n not in (1, 4, 5):
            raise ValueError('capacity groups must use 1, 4 or 5 independent samples')


def warm_pool(pool, count):
    """Create all OS threads before any paid request; cleanly unwind on failure.

    ThreadPoolExecutor.submit can enqueue before thread creation raises. Warming
    with inert tasks prevents that edge case from sending an untracked SDK call.
    """
    release = threading.Event()
    futures = []
    try:
        for _ in range(count):
            futures.append(pool.submit(release.wait))
    finally:
        release.set()
        for f in futures:
            f.result()


class GroupQueue:
    def __init__(self, plan, coordinators, record, stop):
        self.plan, self.record, self.stop = plan, record, stop
        self.pool = cf.ThreadPoolExecutor(max_workers=coordinators, thread_name_prefix='probe-group')
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.active = self.peak = 0
        self.results, self.futures = [], []

    def start(self, coordinators):
        warm_pool(self.pool, coordinators)
        for i, spec in enumerate(self.plan):
            if self.stop.is_set():
                break
            self.futures.append(self.pool.submit(self.group, i, spec, time.perf_counter()))

    def group(self, number, spec, submitted):
        began = time.perf_counter()
        samples = []
        with self.lock:
            if self.closed or self.stop.is_set():
                return
            self.active += 1
            self.peak = max(self.peak, self.active)
            for sample_no in range(spec['samples']):
                f = cf.Future()
                samples.append(f)
                self.queue.put((spec['workload_index'], dict(group_no=number,
                    sample_no=sample_no, group_target=spec['samples']), f))
        try:
            # result() handles cancelled futures as well as completed ones. All
            # samples are already enqueued, so this does not serialize sampling.
            rows, errors, cancelled = [], 0, 0
            for f in samples:
                try:
                    rows.append(f.result())
                except cf.CancelledError:
                    cancelled += 1
                except Exception:
                    errors += 1
            result = dict(group_no=number, workload_index=spec['workload_index'],
                target=spec['samples'], submitted=len(rows)+errors, cancelled=cancelled,
                worker_errors=errors, successful=sum(bool(r['http_success'] and r['parsed']) for r in rows),
                request_nos=[r['request_no'] for r in rows],
                coordinator_wait_seconds=began-submitted, elapsed_seconds=time.perf_counter()-began)
            self.record('group_result', result)
            with self.lock:
                self.results.append(result)
        except Exception:
            self.stop.set()  # Notify admission immediately, not only during close().
            raise
        finally:
            with self.lock:
                self.active -= 1

    def take(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def done(self):
        return all(f.done() for f in self.futures)

    def cancel_unsent(self):
        with self.lock:
            self.closed = True
            while True:
                job = self.take()
                if job is None:
                    break
                job[2].cancel()

    def close(self):
        self.cancel_unsent()
        self.pool.shutdown(wait=True, cancel_futures=True)
        for f in self.futures:
            if not f.cancelled():
                f.result()  # Do not silently swallow a recording failure.

    def summary(self):
        complete = sum(r['successful'] == r['target'] for r in self.results)
        failed = sum(r['submitted'] == r['target'] and r['successful'] < r['target'] for r in self.results)
        return dict(planned_groups=len(self.plan), started_groups=len(self.results),
                    complete_groups=complete, failed_groups=failed,
                    unfinished_groups=len(self.plan)-complete-failed,
                    peak_coordinators=self.peak, active_at_finish=self.active,
                    sample_count_composition=dict(Counter(str(s['samples']) for s in self.plan)))
