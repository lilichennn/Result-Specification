"""Run-scoped safety for shared, unmodified native DeepEye resources."""
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager, ExitStack
import contextvars
import functools
import threading


_STAGE_WORK = contextvars.ContextVar('deepeye_native_stage_work', default=None)
_SUBMISSION_INDEX = contextvars.ContextVar('deepeye_native_submission_index', default=0)
_POOL_NAMES = ('_thread_pool_executor', '_inner_thread_pool_executor', '_column_query_executor')


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
