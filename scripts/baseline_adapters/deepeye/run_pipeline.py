"""Transactional outer orchestration around unchanged native DeepEye item methods.

Completion means a native stage produced its required artifact, NOT SQL correctness.
The caller installs PostgreSQL support and TraceRecorder for the whole invocation.
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, nullcontext
import copy
import importlib
import sys
import threading
import time

from .precompute_cache import fingerprint
from .run_store import restore_jsonable, to_jsonable
from .run_resources import close_runners, protect_schema_profiles, instrument_native_pools, native_stage_work
from .run_usage import sampling_completeness
from .workloads import task_key

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


def input_identity(manifest, manifest_digest, snapshot):
    manifest_value = (manifest_digest if manifest.get('fingerprint_algorithm') == 'manifest-digest-v2'
                      else manifest)
    return fingerprint({'manifest': manifest_value, 'input': snapshot})


@dataclass
class PreparedRun:
    store: object
    revision: tuple
    manifest: dict
    manifest_digest: str
    plans: dict
    attempts: dict
    verification: dict

    def check(self, store):
        if self.store is not store:
            raise ValueError('Prepared state belongs to a different RunStore')
        if self.revision != store._database_revision():
            raise ValueError('Prepared state is stale for the current RunStore')


def prepare_run(store, tasks, *, selected_keys=None, checkpoints=None):
    """Verify the whole store once and restore only selected canonical prefixes."""
    with store._read_snapshot():
        verification = store.verify()
        if not verification['ok']:
            raise ValueError('RunStore verification failed before preparation')
        manifest, rows = store.manifest, defaultdict(list)
        for row in store.attempts():
            rows[row['item_key']].append(row)
        plans, seen = {}, set()
        selected = None if selected_keys is None else set(selected_keys)
        for partition, original in tasks:
            key = task_key(partition, original)
            if key in seen:
                raise ValueError('Duplicate task identifier')
            seen.add(key)
            if selected is not None and key not in selected:
                continue
            if original.gold_sql:
                raise ValueError('Target gold SQL must not be available to this pipeline')
            if any(getattr(original, field) is not None for field in
                   ('final_linked_tables_and_columns', 'sql_candidates', 'sql_candidates_after_revision', 'final_selected_sql')):
                raise ValueError('Use pristine precomputed inputs; stage state comes only from RunStore')
            state = copy.deepcopy(original)
            initial = identity = input_identity(manifest, store.manifest_fingerprint,
                to_jsonable(original.model_dump(exclude={'gold_sql'})))
            completed = {(row['stage'], row['input_fingerprint']): row for row in rows[key]
                         if row['status'] == 'succeeded'}
            stage_inputs = defaultdict(set)
            for stage_name, recorded_hash in completed:
                stage_inputs[stage_name].add(recorded_hash)
            reused, remaining = [], ()
            for index, stage in enumerate(STAGES):
                stage_hash = fingerprint({'input': identity, 'stage': stage})
                if stage_inputs[stage] - {stage_hash}:
                    raise ValueError(f'Native checkpoint input fingerprint mismatch: {key}/{stage}')
                prior = completed.get((stage, stage_hash))
                if prior is None:
                    remaining = STAGES[index:]
                    if any(row['status'] == 'succeeded' and row['stage'] in STAGES[index + 1:] for row in rows[key]):
                        raise ValueError(f'Native successful stage outside canonical prefix: {key}')
                    break
                if checkpoints is not None:
                    checkpoints.validate_stage(prior)
                _restore(state, stage, prior['payload'])
                if not _valid_output(state, stage):
                    raise ValueError(f'Invalid completed checkpoint for {key}/{stage}')
                identity = fingerprint({'input': stage_hash, 'output': prior['payload']})
                reused.append({'stage': stage, 'attempt_id': prior['attempt_id'], 'reused': True})
            plans[key] = {'state': state, 'initial_identity': initial, 'identity': identity,
                          'reused': reused, 'remaining': remaining, 'original': original}
        if selected is not None and not selected.issubset(seen):
            raise ValueError('Execution item selection is outside the frozen inputs')
        return PreparedRun(store, store._database_revision(), manifest, store.manifest_fingerprint,
                           plans, dict(rows), verification)


def select_unfinished(store, tasks, *, prepared=None):
    """Validate canonical prefixes and seal only provable interrupted masters.

    Caller holds the RunStore writer lock and has checked the FULL manifest.
    Success takes precedence over older failures, as in native ``completed``.
    This changes execution membership only, never sampling or attempt budgets.
    """
    if store._lock_file is None:
        raise ValueError('Unfinished selection requires the RunStore writer lock')
    tasks = list(tasks)
    prepared = prepared or prepare_run(store, tasks)
    prepared.check(store)
    selected, report, seals = [], {'selected': [], 'terminal': {}, 'sealed': []}, []
    for variant, original in tasks:
        key = task_key(variant, original)
        plan = prepared.plans[key]
        rows = prepared.attempts.get(key, [])
        masters = [row for row in rows if row['stage'] == 'pipeline']
        master = masters[-1] if masters else None
        identity = plan['initial_identity']
        master_hash = fingerprint({'input': identity, 'stage': 'pipeline'})
        if master and master['input_fingerprint'] != master_hash:
            raise ValueError(f'Native master input mismatch: {key}')
        links, terminal, failed_stage = list(plan['reused']), 'succeeded', None
        identity = plan['identity']
        for stage in plan['remaining'][:1]:
            input_hash = fingerprint({'input': identity, 'stage': stage})
            failures = [row for row in rows if row['stage'] == stage and
                        row['input_fingerprint'] == input_hash and row['status'] == 'failed']
            if failures:
                terminal, failed_stage = 'failed', stage
                links.append({'stage': stage, 'attempt_id': failures[-1]['attempt_id'], 'reused': False})
            else:
                terminal = None
            break
        if terminal is None:
            if master and master['status'] != 'interrupted':
                raise ValueError(f'Terminal master has incomplete canonical lineage: {key}')
            selected.append((variant, original))
            report['selected'].append(key)
            continue
        if master is None:
            raise ValueError(f'Terminal native lineage has no master: {key}')
        if master['status'] == 'interrupted':
            recorded, ticket = [], None
            for event in store.iter_events(master['attempt_id'], kinds=('pipeline_admitted', 'pipeline_stage_attempt')):
                if event['kind'] == 'pipeline_admitted':
                    if ticket is not None:
                        raise ValueError(f'Duplicate native admission record: {key}')
                    ticket = event['payload']['ticket']
                    recorded.extend(event['payload']['reused_stage_attempts'])
                else:
                    recorded.append(event['payload'])
            if [(row['stage'], row['attempt_id']) for row in recorded] != [(row['stage'], row['attempt_id']) for row in links]:
                raise ValueError(f'Interrupted master does not own terminal canonical lineage: {key}')
            payload = {'ticket': ticket, 'stage_attempts': recorded, 'failed_stage': failed_stage}
            seals.append((master['attempt_id'], terminal, payload))
            report['sealed'].append({'task_key': key, 'attempt_id': master['attempt_id'], 'status': terminal})
        else:
            payload = master['payload']
            if master['status'] != terminal or payload.get('failed_stage') != failed_stage:
                raise ValueError(f'Native master disagrees with canonical lineage: {key}')
            referenced = payload.get('stage_attempts', [])
            if [(row['stage'], row['attempt_id']) for row in referenced] != [(row['stage'], row['attempt_id']) for row in links]:
                raise ValueError(f'Native master references differ from canonical lineage: {key}')
        report['terminal'][key] = terminal
    # Validate the whole cohort before sealing any item.
    for attempt_id, status, payload in seals:
        store.append_event(attempt_id, 'pipeline_recovery_seal', {
            'reason': 'committed_terminal_stage_chain_without_master_finish', 'status': status})
        store.finish_attempt(attempt_id, status, payload)
    prepared.revision = store._database_revision()
    return selected, report


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
    """Use DeepEye's required-field check, without extra sampling/SQL criteria.

    Native empty lists and fallback strings are not None. Later native stages
    decide whether they can continue; SQL correctness belongs to evaluation.
    """
    return item.is_stage_complete(stage)


def run_pipeline(store, tasks, runner_factory, recorder, workers=4, slot_controller=None, *, runtime=None, prepared=None):
    """Run/resume four stages, appending attempts and committing each completed item.

    Inputs must be pristine, gold-free precomputed DataItems. Failed item mutations
    never leak to retry; errors of the recorder/store stop execution, not just that
    item. A crash or cooperative pause leaves a resumable stage: rerunning its
    native control flow restores durable samples and their consumed budgets.
    Uncommitted responses remain uncertain, not exactly-once remote inference.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise RuntimeError('Only one native DeepEye pipeline may run per process')
    try:
        tasks = list(tasks)
        if runtime is not None:
            if runtime.stop_event is not recorder.stop_event:
                raise ValueError('Runtime and recorder must share the same stop event')
            from .run_slots import WorkflowSlots
            if slot_controller is not None and not isinstance(slot_controller, WorkflowSlots):
                raise ValueError('Legacy pipeline limits are incompatible with the shared runtime')
            slot_controller = slot_controller or WorkflowSlots(len(tasks))
        if slot_controller is None:
            from .run_slots import PipelineSlots
            slot_controller = PipelineSlots(fixed_limit=workers)
        prepared = prepared or prepare_run(store, tasks, checkpoints=getattr(recorder, 'sampling_checkpoints', None))
        prepared.check(store)
        return _run_pipeline(store, tasks, runner_factory, recorder, slot_controller, runtime, prepared)
    finally:
        _PIPELINE_LOCK.release()


def _run_pipeline(store, tasks, runner_factory, recorder, slot_controller, runtime=None, prepared=None):
    from app.llm.sampling import SamplingPaused
    stop = getattr(recorder, 'stop_event', threading.Event())
    keys = [task_key(partition, item) for partition, item in tasks]
    if len(keys) != len(set(keys)) or not set(keys).issubset(prepared.plans):
        raise ValueError('Duplicate or unprepared execution task')
    states = {key: prepared.plans[key]['state'] for key in keys}
    identities = {key: prepared.plans[key]['identity'] for key in keys}
    initial_identities = {key: prepared.plans[key]['initial_identity'] for key in keys}
    reused = {key: prepared.plans[key]['reused'] for key in keys}
    remaining = {key: prepared.plans[key]['remaining'] for key in keys if prepared.plans[key]['remaining']}

    failed, runners, resources, paused = {}, {}, [], set()
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
            if runtime is not None:
                runtime.bind_runner(runner)
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
                    if stop.is_set():
                        paused.add(key)
                        status = 'paused'
                        break
                    if halted.is_set():
                        raise fatal_errors[0]
                    recorder.raise_if_failed()
                    input_hash = fingerprint({'input': identities[key], 'stage': stage})
                    attempt_id = store.begin_attempt(key, stage, input_hash)
                    link = {'stage': stage, 'attempt_id': attempt_id, 'reused': False}
                    links.append(link)
                    store.append_event(master, 'pipeline_stage_attempt', link)
                    target = copy.deepcopy(states[key])
                    with recorder.context(attempt_id), runtime.context() if runtime else nullcontext():
                        started = time.monotonic()
                        error_type = error_message = None
                        try:
                            with native_stage_work():
                                getattr(runners[stage], STAGE_METHODS[stage])(target)
                        except SamplingPaused:
                            stop.set()
                        except Exception as exc:
                            error_type, error_message = type(exc).__name__, str(exc)
                            for secret in getattr(recorder, 'secrets', ()):
                                error_message = error_message.replace(secret, '[REDACTED]')
                        recorder.raise_if_failed()
                        if stop.is_set():
                            store.append_event(attempt_id, 'stage_paused', {'resumable': True})
                            paused.add(key)
                            status = 'paused'
                            break
                        sampling = sampling_completeness(store.iter_events(attempt_id,
                            kinds=('sampling_group_start', 'sampling_group_result')))
                        payload = _checkpoint(target, stage)
                        if getattr(recorder, 'sampling_checkpoints', None):
                            payload['sampling_implementation_version'] = recorder.sampling_checkpoints.source_version
                        if sampling['groups']:
                            payload['sampling'] = sampling
                        payload['attempt_wall_seconds'] = time.monotonic() - started
                        status = 'succeeded' if error_type is None and _valid_output(target, stage) else 'failed'
                        if status == 'failed':
                            payload['error_type'] = error_type or 'IncompleteNativeStageOutput'
                            payload['error_message'] = error_message or 'Native required fields or metrics are None'
                        store.finish_attempt(attempt_id, status, payload)
                    print(f'{stage}: committed; {key} {status}', flush=True)
                    if status == 'failed':
                        failed_stage = failed[key] = stage
                        break
                    states[key] = target
                    identities[key] = fingerprint({'input': input_hash, 'output': payload})
                terminal = {'ticket': ticket, 'stage_attempts': links, 'failed_stage': failed_stage}
                store.append_event(master, 'pipeline_release_intent', {**terminal, 'status': status})
                if status == 'paused':
                    store.append_event(master, 'pipeline_paused', terminal)
                else:
                    store.finish_attempt(master, status, terminal)
                # Release only after both the terminal native stage and master
                # outcome are durable.
                with admission_lock:
                    slot_controller.release(ticket, status='failed' if status == 'paused' else status)
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
            question_pool = (runtime.workflow_executor(len(remaining)) if runtime else
                             ThreadPoolExecutor(max_workers=slot_controller.max_limit, thread_name_prefix='run-pipeline'))
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
                        if stop.is_set():
                            paused.update(pending)
                            pending.clear()
                        if not halted.is_set() and not stop.is_set():
                            slot_controller.set_pending(len(pending))
                            while pending and not halted.is_set() and not stop.is_set():
                                key = pending[0]
                                ticket = slot_controller.try_acquire(key)
                                if ticket is None:
                                    break
                                if halted.is_set():
                                    slot_controller.release(ticket, status='failed')
                                    break
                                pending.popleft()
                                unreleased[key] = ticket
                                try:
                                    future = pool.submit(execute, key, ticket)
                                except SamplingPaused:
                                    stop.set()
                                    paused.add(key)
                                    slot_controller.release(ticket, status='failed')
                                    unreleased.pop(key)
                                    continue
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
        'status': 'paused' if key in paused else 'failed' if key in failed else 'succeeded',
        'failed_stage': failed.get(key),
        'final_selected_sql': state.final_selected_sql,
        'total_time': state.total_time,
        'total_llm_cost': to_jsonable(state.total_llm_cost),
        'metric_semantics': 'native_completed_lineage_including_reused_precompute_excluding_failed_attempts',
    } for key, state in states.items()}
    return {'succeeded': len(states) - len(failed) - len(paused), 'failed': len(failed), 'paused': len(paused),
        'accuracy_evaluated': False, 'rc_enabled': False, 'items': items}
