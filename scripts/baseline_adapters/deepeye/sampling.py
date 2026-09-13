"""Fair run-scoped execution of independent C1 samples, without another retry loop."""
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
import contextvars
import threading


class SampleScheduler:
    """Only sample leaves use this executor; their waiting parents run elsewhere.

    Keep at most ``workers`` submitted leaves. Pending groups rotate after each
    leaf admission, so a large group cannot monopolize newly available workers.
    Context and fixed sample indexes are assigned before any submission.
    """
    def __init__(self, workers, stop_event):
        self.workers, self.stop_event = workers, stop_event
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix='deepeye-sample')
        self._lock = threading.RLock()
        self._groups = deque()
        self._active = self._peak = 0
        self._closed = False

    def _pump(self):
        while self._groups and self._active < self.workers:
            group = self._groups.popleft()
            context, fn, args, kwargs, result = group.popleft()
            if group:
                self._groups.append(group)
            self._active += 1
            self._peak = max(self._peak, self._active)
            try:
                future = self._pool.submit(context.run, fn, *args, **kwargs)
            except BaseException as error:
                self._active -= 1
                result.set_exception(error)
                continue
            def finished(future, result=result):
                try:
                    result.set_result(future.result())
                except BaseException as error:
                    result.set_exception(error)
                finally:
                    with self._lock:
                        self._active -= 1
                        self._pump()
            future.add_done_callback(finished)

    def run_samples(self, request, parser, *, group_id, n, max_attempts):
        from app.llm.sampling import execute_sample, SamplingPaused
        pending = []
        with self._lock:
            if self._closed or self.stop_event.is_set():
                raise SamplingPaused('Sample scheduler stopped')
            group = deque()
            for index in range(n):
                result = Future()
                pending.append(result)
                group.append((contextvars.copy_context(), execute_sample, (request, parser),
                              dict(group_id=group_id, sample_index=index,
                                   max_attempts=max_attempts), result))
            self._groups.append(group)
            self._pump()
        try:
            return [future.result() for future in pending]
        finally:
            # BaseException includes pause/integrity signals and interruption.
            # No group may unwind its C2 session while siblings still commit.
            while any(not future.done() for future in pending):
                try:
                    wait(pending)
                except BaseException:
                    continue

    def snapshot(self):
        with self._lock:
            return dict(worker_cap=self.workers, active=self._active, peak=self._peak,
                        queued=sum(len(group) for group in self._groups))

    def close(self):
        with self._lock:
            self._closed = True
        # Group callers have drained all leaves before runtime close.
        self._pool.shutdown(wait=True)
