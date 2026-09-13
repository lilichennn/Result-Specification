"""Transactional outer orchestration around unchanged native DeepEye item methods.

Completion means a native stage produced its required artifact, NOT SQL correctness.
The caller installs PostgreSQL support and TraceRecorder for the whole invocation.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import copy
import importlib
import sys
import threading
import time

from .precompute_cache import fingerprint
from .run_store import restore_jsonable, to_jsonable
from .run_resources import close_runners, protect_schema_profiles, instrument_native_pools, native_stage_work

STAGE_METHODS = {
    'schema_linking': '_link_tables_and_columns',
    'sql_generation': '_generate_sql',
    'sql_revision': '_revise_sql',
    'sql_selection': '_select_best_sql',
}
STAGES = tuple(STAGE_METHODS)
_RUNNER_NAMES = ('SchemaLinkingRunner', 'SQLGenerationRunner', 'SQLRevisionRunner', 'SQLSelectionRunner')
_CONSTRUCTOR_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()


def task_key(variant, item):
    if variant not in ('lite', 'full'):
        raise ValueError('Unknown BIRD-Interact variant')
    return f'{variant}/{item.instance_id}'


def native_runner_factory(config):
    """Inject the current in-memory items at construction; never load native snapshots.

    Constructors, generators, checkers, comparison rules and inner pools are native.
    Their unused ArtifactStore is closed by normal cleanup; we never call run/save.
    One pipeline per process is required by the native global schema/SQL services.
    """
    def factory(stage, items):
        pipeline = importlib.import_module('app.pipeline')
        cls = getattr(pipeline, _RUNNER_NAMES[STAGES.index(stage)])
        module = importlib.import_module(cls.__module__)
        with _CONSTRUCTOR_LOCK:
            original = module.load_stage_dataset
            original_init = cls.__init__
            allocated = []
            def tracked_init(instance, *args, **kwargs):
                allocated.append(instance)
                original_init(instance, *args, **kwargs)
            cls.__init__ = tracked_init
            module.load_stage_dataset = lambda *args, **kwargs: (items, 'transactional-run-store')
            try:
                return cls.from_config(config)
            except BaseException as exc:
                partial = [(runner, None, getattr(getattr(runner, '_llm', None), '_client', None))
                    for runner in allocated]
                try:
                    close_runners(partial)
                except BaseException as cleanup_error:
                    exc.add_note(f'Partial native cleanup also failed: {type(cleanup_error).__name__}')
                raise
            finally:
                module.load_stage_dataset = original
                cls.__init__ = original_init
    return factory


def _checkpoint(item, stage):
    return {
        'artifact': to_jsonable(item.get_stage_artifact(stage).model_dump()),
        'metrics': to_jsonable(item.get_metrics_record().model_dump()),
        'completion_semantics': 'native_required_fields_present_not_sql_correctness',
    }


def _restore(item, stage, checkpoint):
    item.apply_stage_artifact(stage, restore_jsonable(checkpoint['artifact']))
    item.apply_metrics_record(restore_jsonable(checkpoint['metrics']))


def _valid_output(item, stage):
    if not item.is_stage_complete(stage):
        return False
    if stage in ('sql_generation', 'sql_revision'):
        field = 'sql_candidates' if stage == 'sql_generation' else 'sql_candidates_after_revision'
        values = getattr(item, field)
        return bool(values) and all(isinstance(sql, str) and bool(sql.strip()) for sql in values)
    if stage == 'sql_selection':
        return bool(item.final_selected_sql.strip()) and item.final_selected_sql != 'Error'
    return True


def run_pipeline(store, tasks, runner_factory, recorder, workers=4, slot_controller=None):
    """Run/resume four stages, appending attempts and committing each completed item.

    Inputs must be pristine, gold-free precomputed DataItems. Failed item mutations
    never leak to retry; errors of the recorder/store stop execution, not just that
    item. A crash can leave an unfinished attempt and already-paid calls; its next
    attempt reruns that stage. It cannot guarantee exactly-once remote inference.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise RuntimeError('Only one native DeepEye pipeline may run per process')
    try:
        if slot_controller is None:
            from .run_slots import PipelineSlots
            slot_controller = PipelineSlots(fixed_limit=workers)
        return _run_pipeline(store, list(tasks), runner_factory, recorder, slot_controller)
    finally:
        _PIPELINE_LOCK.release()


def _run_pipeline(store, tasks, runner_factory, recorder, slot_controller):
    states, identities = {}, {}
    for variant, item in tasks:
        key = task_key(variant, item)
        if key in states:
            raise ValueError('Duplicate variant/instance identifier')
        if item.gold_sql:
            raise ValueError('Target gold SQL must not be available to this pipeline')
        if any(getattr(item, field) is not None for field in
               ('final_linked_tables_and_columns', 'sql_candidates', 'sql_candidates_after_revision', 'final_selected_sql')):
            raise ValueError('Use pristine precomputed inputs; stage state comes only from RunStore')
        snapshot = item.model_dump(exclude={'gold_sql'})
        identities[key] = fingerprint({'manifest': store.manifest, 'input': to_jsonable(snapshot)})
        states[key] = copy.deepcopy(item)
    # Restore and validate every available prefix serially before constructing
    # resources or permitting paid work on any question.
    initial_identities = dict(identities)
    remaining, reused = {}, {}
    for key, state in states.items():
        reused[key] = []
        for index, stage in enumerate(STAGES):
            input_hash = fingerprint({'input': identities[key], 'stage': stage})
            prior = store.completed(key, stage, input_hash)
            if prior:
                _restore(state, stage, prior['payload'])
                if not _valid_output(state, stage):
                    raise ValueError(f'Invalid completed checkpoint for {key}/{stage}')
                identities[key] = fingerprint({'input': input_hash, 'output': prior['payload']})
                reused[key].append({'stage': stage, 'attempt_id': prior['attempt_id'], 'reused': True})
            else:
                remaining[key] = STAGES[index:]
                break

    failed, runners, resources = {}, {}, []
    pending = deque(remaining)
    halted = threading.Event()
    admission_lock = threading.Lock()
    failure_lock = threading.Lock()
    fatal_errors = []
    unreleased = {}
    guard = ExitStack()
    question_pool = None
    question_futures = []
    try:
        for stage in STAGES:
            stage_items = [states[key] for key in remaining if stage in remaining[key]]
            if not stage_items:
                continue
            recorder.raise_if_failed()
            runner = runner_factory(stage, stage_items)
            entry = [runner, None, None]
            resources.append(entry)
            runners[stage] = runner
            instrumentation = ExitStack()
            entry[1] = instrumentation.close
            instrumentation.callback(instrument_native_pools(runner))
            llm = getattr(runner, '_llm', None)
            if llm is not None:
                entry[2] = llm._get_client()
            instrumentation.callback(recorder.instrument_runner(runner, stage))
        if remaining:
            guard.enter_context(protect_schema_profiles())

        def execute(key, ticket):
            try:
                if halted.is_set():
                    raise fatal_errors[0]
                recorder.raise_if_failed()
                master = store.begin_attempt(key, 'pipeline', fingerprint({
                    'input': initial_identities[key], 'stage': 'pipeline'}))
                store.append_event(master, 'pipeline_admitted', {
                    'ticket': ticket, 'reused_stage_attempts': reused[key]})
                links = list(reused[key])
                status, failed_stage = 'succeeded', None
                for stage in remaining[key]:
                    if halted.is_set():
                        raise fatal_errors[0]
                    recorder.raise_if_failed()
                    input_hash = fingerprint({'input': identities[key], 'stage': stage})
                    attempt_id = store.begin_attempt(key, stage, input_hash)
                    link = {'stage': stage, 'attempt_id': attempt_id, 'reused': False}
                    links.append(link)
                    store.append_event(master, 'pipeline_stage_attempt', link)
                    target = copy.deepcopy(states[key])
                    with recorder.context(attempt_id):
                        started = time.monotonic()
                        error_type = error_message = None
                        try:
                            with native_stage_work():
                                getattr(runners[stage], STAGE_METHODS[stage])(target)
                        except Exception as exc:
                            error_type, error_message = type(exc).__name__, str(exc)
                            for secret in getattr(recorder, 'secrets', ()):
                                error_message = error_message.replace(secret, '[REDACTED]')
                        recorder.raise_if_failed()
                        payload = _checkpoint(target, stage)
                        payload['attempt_wall_seconds'] = time.monotonic() - started
                        status = 'succeeded' if error_type is None and _valid_output(target, stage) else 'failed'
                        if status == 'failed':
                            payload['error_type'] = error_type or 'IncompleteNativeStageOutput'
                            payload['error_message'] = error_message or 'Native required output fields are missing or empty'
                        store.finish_attempt(attempt_id, status, payload)
                    print(f'{stage}: committed; {key} {status}', flush=True)
                    if status == 'failed':
                        failed_stage = failed[key] = stage
                        break
                    states[key] = target
                    identities[key] = fingerprint({'input': input_hash, 'output': payload})
                terminal = {'ticket': ticket, 'stage_attempts': links, 'failed_stage': failed_stage}
                store.append_event(master, 'pipeline_release_intent', {**terminal, 'status': status})
                store.finish_attempt(master, status, terminal)
                # Release only after both the terminal native stage and master
                # outcome are durable.
                with admission_lock:
                    slot_controller.release(ticket, status=status)
                    unreleased.pop(key)
                return key
            except BaseException as exc:
                # Do not wait on a scheduler lock while it is filling slots:
                # failures must become visible during an admission burst too.
                with failure_lock:
                    if not fatal_errors:
                        fatal_errors.append(exc)
                    halted.set()
                raise

        if remaining:
            question_pool = ThreadPoolExecutor(max_workers=slot_controller.max_limit, thread_name_prefix='run-pipeline')
            with question_pool as pool:
                futures = {}
                while pending or futures:
                    # Surface all observed failures before refilling a free slot.
                    for future in list(futures):
                        if future.done():
                            key = future.result()
                            del futures[future]
                    recorder.raise_if_failed()
                    with admission_lock:
                        if not halted.is_set():
                            slot_controller.set_pending(len(pending))
                            while pending and not halted.is_set():
                                key = pending[0]
                                ticket = slot_controller.try_acquire(key)
                                if ticket is None:
                                    break
                                if halted.is_set():
                                    slot_controller.release(ticket, status='failed')
                                    break
                                pending.popleft()
                                unreleased[key] = ticket
                                future = pool.submit(execute, key, ticket)
                                question_futures.append(future)
                                futures[future] = key
                                slot_controller.set_pending(len(pending))
                    if futures:
                        wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                recorder.raise_if_failed()
    finally:
        # Executor context drains all questions before the first native cleanup;
        # close_runners additionally drains detached native inner work first.
        active_error = sys.exc_info()[1]
        try:
            close_runners(resources, question_pool=question_pool, question_futures=question_futures)
        except BaseException as exc:
            if active_error is None:
                raise
            active_error.add_note(f'Native cleanup also failed: {type(exc).__name__}')
        finally:
            try:
                guard.close()
            finally:
                slot_controller.set_pending(0)
                for ticket in unreleased.values():
                    # Fatal runs cannot promise a durable terminal record; these
                    # tickets are reclaimed only after all native work drains.
                    slot_controller.release(ticket, status='failed')

    items = {key: {
        'status': 'failed' if key in failed else 'succeeded',
        'failed_stage': failed.get(key),
        'final_selected_sql': state.final_selected_sql,
        'total_time': state.total_time,
        'total_llm_cost': to_jsonable(state.total_llm_cost),
        'metric_semantics': 'native_completed_lineage_including_reused_precompute_excluding_failed_attempts',
    } for key, state in states.items()}
    return {'succeeded': len(states) - len(failed), 'failed': len(failed),
        'accuracy_evaluated': False, 'rc_enabled': False, 'items': items}
