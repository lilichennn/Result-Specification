"""Run-scoped safety for shared, unmodified native DeepEye resources."""
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, ExitStack
import contextvars
import functools
import threading


_STAGE_WORK = contextvars.ContextVar('deepeye_native_stage_work', default=None)
_SUBMISSION_INDEX = contextvars.ContextVar('deepeye_native_submission_index', default=0)
_POOL_NAMES = ('_thread_pool_executor', '_inner_thread_pool_executor', '_column_query_executor')
_COORDINATOR_RUNTIME = contextvars.ContextVar('deepeye_coordinator_runtime', default=None)


class SamplingRuntime:
    """Run-owned coordinator, sample and cancellable request resources for C4.

    Use the same ``stop_event`` as TraceRecorder, bind ``context()`` around native
    stages, and drain this runtime before closing recorder/store resources.
    Coordinators may await samples, never other tasks on the coordinator pool.
    """
    def __init__(self, *, stop_event=None, emit=None, **limits):
        from .request_dispatch import RequestDispatcher, RequestLimits
        from .sampling import SampleScheduler
        self.limits = RequestLimits(**limits)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.samples = SampleScheduler(self.limits.request_workers, self.stop_event)
        self.coordinators = ThreadPoolExecutor(self.limits.coordinator_workers,
                                              thread_name_prefix='deepeye-coordinator')
        self.dispatch = RequestDispatcher(self.limits, stop_event=self.stop_event, emit=emit)
        self._closed = False
        self._closing = False
        self._coordinator_lock = threading.Lock()
        self._coordinator_active = self._coordinator_peak = 0

    @contextmanager
    def context(self):
        from app.llm.sampling import sampling_scheduler
        with sampling_scheduler(self.samples):
            yield self

    def submit_coordinator(self, function, /, *args, **kwargs):
        from app.llm.sampling import SamplingPaused
        from .sampling import submit_owned
        nested = _COORDINATOR_RUNTIME.get() is self
        if self._closed or (self._closing and not nested) or self.stop_event.is_set():
            raise SamplingPaused('Coordinator submissions stopped')
        if nested:
            # A waiting coordinator must not consume the final thread needed
            # by its descendants. Nested control work executes inline; sample
            # leaves still run concurrently on the separate sample pool.
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
            return future
        context = contextvars.copy_context()
        def run():
            token = _COORDINATOR_RUNTIME.set(self)
            with self._coordinator_lock:
                self._coordinator_active += 1
                self._coordinator_peak = max(self._coordinator_peak, self._coordinator_active)
            try:
                return function(*args, **kwargs)
            finally:
                with self._coordinator_lock:
                    self._coordinator_active -= 1
                _COORDINATOR_RUNTIME.reset(token)
        return submit_owned(self.coordinators, context.run, run)

    def stop(self, *, cancel_active=False):
        self.dispatch.stop(cancel_active=cancel_active)

    def make_client(self, **configuration):
        return self.dispatch.make_client(**configuration)

    def executor_view(self):
        return _CoordinatorView(self)

    def bind_runner(self, runner):
        """Replace unused constructor pools/clients before tracing or stage work."""
        pools = set()
        for name in _POOL_NAMES:
            pool = getattr(runner, name, None)
            if pool is not None:
                if id(pool) not in pools:
                    pool.shutdown(wait=True)
                    pools.add(id(pool))
                setattr(runner, name, self.executor_view())
        llm = getattr(runner, '_llm', None)
        if llm is not None:
            previous = getattr(llm, '_client', None)
            if previous is not None:
                previous.close()
                llm._client = None
            config = llm.llm_config
            llm._client = self.make_client(api_key=config.api_key, base_url=str(config.base_url),
                                          api_type=config.api_type,
                                          api_version=getattr(config, 'api_version', None))

    def close(self):
        if self._closed:
            return
        self._closing = True
        self.coordinators.shutdown(wait=True)
        self.samples.close()
        self.dispatch.close()
        self._closed = True

    def snapshot(self):
        from dataclasses import asdict
        with self._coordinator_lock:
            coordinators = dict(cap=self.limits.coordinator_workers,
                                active=self._coordinator_active, peak=self._coordinator_peak)
        return dict(limits=asdict(self.limits), coordinators=coordinators, samples=self.samples.snapshot(),
                    requests=self.dispatch.snapshot())


class _CoordinatorView:
    """Runner-owned drain scope; shutdown never closes the shared executor."""
    def __init__(self, runtime):
        self.runtime = runtime
        self._lock = threading.Lock()
        self._futures = set()
        self._shutdown = False

    def submit(self, fn, /, *args, **kwargs):
        # Do not hold the view lock while nested coordination executes inline.
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot schedule after runner shutdown')
        future = self.runtime.submit_coordinator(fn, *args, **kwargs)
        with self._lock:
            self._futures.add(future)
        def done(completed):
            with self._lock:
                self._futures.discard(completed)
        future.add_done_callback(done)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._lock:
            self._shutdown = True
            pending = tuple(self._futures)
        if cancel_futures:
            for future in pending:
                future.cancel()
        if wait:
            from concurrent.futures import wait as drain
            drain(pending)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown(wait=True)


@contextmanager
def native_stage_work():
    """Wait for this stage's own nested branches, including on parent failure.

    Keep errors/fallbacks owned by native methods: waiting does not re-raise an
    exception the native parent already handled. Descendant submissions append
    to the same list before their parent future can finish.
    """
    pending, lock = [], threading.Lock()
    token = _STAGE_WORK.set((pending, lock))
    submission_token = _SUBMISSION_INDEX.set(0)
    try:
        yield
    finally:
        try:
            index = 0
            while True:
                with lock:
                    if index == len(pending):
                        break
                    future = pending[index]
                wait((future,))
                index += 1
        finally:
            _STAGE_WORK.reset(token)
            _SUBMISSION_INDEX.reset(submission_token)


def instrument_native_pools(runner):
    """Reversibly track only submissions made inside a native stage context."""
    undo = ExitStack()
    try:
        for name in _POOL_NAMES:
            pool = getattr(runner, name, None)
            if pool is None:
                continue
            original = pool.submit
            had_instance_submit = 'submit' in vars(pool)
            tracked, tracked_lock = set(), threading.Lock()
            pool._deepeye_run_futures = (tracked, tracked_lock)
            undo.callback(lambda pool=pool: delattr(pool, '_deepeye_run_futures'))
            def restore(pool=pool, original=original, had_instance_submit=had_instance_submit):
                if had_instance_submit:
                    pool.submit = original
                else:
                    del pool.submit
            undo.callback(restore)
            @functools.wraps(original)
            def submit(fn, /, *args, _original=original, _tracked=tracked, _tracked_lock=tracked_lock, **kwargs):
                from app.llm.sampling import check_sampling_stop
                check_sampling_stop()
                work = _STAGE_WORK.get()
                if work is None:
                    future = _original(fn, *args, **kwargs)
                else:
                    from .run_sampling import stable_sampling_node
                    position = _SUBMISSION_INDEX.get()
                    _SUBMISSION_INDEX.set(position + 1)
                    with stable_sampling_node(('submission', position)):
                        context = contextvars.copy_context()
                    context.run(_SUBMISSION_INDEX.set, 0)
                    future = _original(context.run, fn, *args, **kwargs)
                    pending, lock = work
                    with lock:
                        pending.append(future)
                with _tracked_lock:
                    _tracked.add(future)
                def finished(completed):
                    with _tracked_lock:
                        _tracked.discard(completed)
                future.add_done_callback(finished)
                return future
            pool.submit = submit
    except BaseException:
        undo.close()
        raise
    return undo.close


@contextmanager
def protect_schema_profiles():
    """Disable only the id(dict)-keyed cache while transient stage copies coexist.

    Call after all native constructors (which replace the global service), and
    keep installed until every native pool has drained. Other service caches and
    their configuration stay intact. Never restore stale identity-only entries.
    """
    from app.services._bounded_cache import BoundedCache
    from app.services.schema_service import get_schema_service
    service = get_schema_service()
    with service._lock:
        original = service._schema_profile_cache
        original.clear()
        service._schema_profile_cache = BoundedCache(0)
    try:
        yield
    finally:
        with service._lock:
            service._schema_profile_cache = original


def close_runners(resources, *, question_pool=None, question_futures=()):
    """Drain all pools, then all native cleanup, then tracing and unique clients.

    A native cleanup resets process-wide services, so even another runner's
    detached inner futures must finish first. Complete every cleanup action even
    if an earlier action fails, then propagate the first error.
    """
    errors = []
    def perform(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    def wait_for_work(snapshot):
        # A SIGINT in Thread.join can mark a live thread stopped in CPython.
        # Future completion, rather than a second join/is_alive check, proves
        # that native work and its trace context have actually returned.
        while True:
            try:
                pending = [future for future in snapshot() if not future.done()]
                if not pending:
                    return
                wait(pending)
            except BaseException as exc:
                errors.append(exc)

    def drain(pool):
        tracking = getattr(pool, '_deepeye_run_futures', None)
        if tracking is not None:
            tracked, lock = tracking
            def snapshot():
                with lock:
                    return tuple(tracked)
            wait_for_work(snapshot)
        try:
            pool.shutdown(wait=True)
            return
        except BaseException as exc:
            errors.append(exc)
        # Native runners use stdlib pools. An interrupted/overridden instance
        # shutdown does not prove drain; finish via the stdlib implementation.
        # Preserve the first exception but never unwind while workers may live.
        while True:
            try:
                if isinstance(pool, _CoordinatorView):
                    _CoordinatorView.shutdown(pool, wait=True)
                else:
                    ThreadPoolExecutor.shutdown(pool, wait=True)
                return
            except BaseException:
                # A repeated interruption (or persistent shutdown failure)
                # leaves all run protections installed until drain succeeds.
                try:
                    threading.Event().wait(0.05)
                except BaseException:
                    pass

    pools = set()
    if question_pool is not None:
        wait_for_work(lambda: question_futures)
        drain(question_pool)
        pools.add(id(question_pool))
    for runner, _, _ in resources:
        for name in _POOL_NAMES:
            pool = getattr(runner, name, None)
            if pool is not None and id(pool) not in pools:
                pools.add(id(pool))
                drain(pool)
    for runner, _, _ in resources:
        perform(runner._clean_up)
    for _, undo, client in resources:
        if undo is not None:
            perform(undo)
    clients = set()
    for _, _, client in resources:
        if client is not None and id(client) not in clients:
            clients.add(id(client))
            perform(client.close)
    if errors:
        raise errors[0]
