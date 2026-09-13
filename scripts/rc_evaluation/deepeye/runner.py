"""Run only the target stage and explicitly requested native downstream stages."""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, nullcontext
import copy
import threading
import time

from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, STAGE_METHODS, _PIPELINE_LOCK, _checkpoint, _restore, _valid_output
from scripts.baseline_adapters.deepeye.run_resources import close_runners, instrument_native_pools, native_stage_work, protect_schema_profiles
from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots
from scripts.baseline_adapters.deepeye.run_store import restore_jsonable
from .source import api_trace, digest, restore_seed, validate_manifest, TRACE_KINDS


class MissingRCParticipation(RuntimeError):
    """An actual target request (or recovered sample) did not contain its RC."""


def _semantic_artifact(payload):
    return {key: value for key, value in restore_jsonable(payload['artifact']).items()
            if not key.endswith(('_time', '_llm_cost', '_recall'))}


def _same_output(payload, source):
    return source is not None and _semantic_artifact(payload) == _semantic_artifact(source['payload'])


def _replay(stage, target, unchanged, source):
    return source is not None and unchanged and (stage != target or (
        source['api_trace']['requests'] == 0 and not source['api_trace'].get('restored_samples')))


def _stage_hash(manifest, key, stage, item, manifest_digest=None):
    identity = (manifest_digest if manifest.get('fingerprint_algorithm') == 'manifest-digest-v2' else manifest)
    if identity is None:
        identity = digest(manifest)
    return digest({'manifest': identity, 'task_key': key, 'stage': stage,
                   'input': item.model_dump(exclude={'gold_sql'})})


@dataclass
class PreparedExperiment:
    store: object
    revision: tuple
    manifest: dict
    manifest_digest: str
    plans: dict
    needed: set
    attempts: dict
    verification: dict

    def check(self, store):
        if self.store is not store:
            raise ValueError('Prepared experiment belongs to a different RunStore')
        if self.revision != store._database_revision():
            raise ValueError('Prepared experiment is stale for the current RunStore')


def prepare_experiment(store, checkpoints=None, *, item_keys=None):
    with store._read_snapshot():
        return _prepare_experiment(store, checkpoints, item_keys=item_keys)


def _prepare_experiment(store, checkpoints=None, *, item_keys=None):
    verification = store.verify()
    if not verification['ok']:
        raise ValueError('Experiment RunStore verification failed')
    manifest = store.manifest
    selected = set(manifest['source_checkpoints']) if item_keys is None else set(item_keys)
    if not selected.issubset(manifest['source_checkpoints']):
        raise ValueError('Execution item selection is outside the frozen manifest')
    seeds = {}
    validate_manifest(manifest, item_keys=selected, seeds=seeds)
    target = manifest['target_stage']
    stages = STAGES[STAGES.index(target):] if manifest['continue_downstream'] else (target,)
    successes = set()
    rows, completed, stage_inputs = defaultdict(list), {}, defaultdict(set)
    for attempt in store.attempts():
        rows[attempt['item_key']].append(attempt)
        if attempt['item_key'] not in manifest['source_checkpoints'] or attempt['stage'] not in stages:
            raise ValueError('Experiment attempt lies outside the configured lineage')
        if attempt['status'] == 'succeeded':
            successes.add((attempt['item_key'], attempt['stage']))
            completed[(attempt['item_key'], attempt['stage'], attempt['input_fingerprint'])] = attempt
            stage_inputs[attempt['item_key'], attempt['stage']].add(attempt['input_fingerprint'])
    plans, needed = {}, set()
    for key, snapshot in manifest['source_checkpoints'].items():
        if key not in selected:
            continue
        state = seeds[key]
        unchanged, remaining = True, []
        for index, stage in enumerate(stages):
            expected = _stage_hash(manifest, key, stage, state, store.manifest_fingerprint)
            if stage_inputs[key, stage] - {expected}:
                raise ValueError(f'RC checkpoint input fingerprint mismatch: {key}/{stage}')
            prior = completed.get((key, stage, expected))
            if prior is not None:
                if checkpoints is not None:
                    checkpoints.validate_stage(prior)
                _restore(state, stage, prior['payload'])
                if not _valid_output(state, stage):
                    raise ValueError(f'Invalid completed RC checkpoint: {key}/{stage}')
                unchanged = unchanged and _same_output(prior['payload'], snapshot['stages'].get(stage))
            else:
                remaining = list(stages[index:])
                if any((key, future) in successes for future in remaining):
                    raise ValueError('Successful downstream checkpoint is outside the completed prefix')
                possible_unchanged = unchanged
                for future in remaining:
                    if not _replay(future, target, possible_unchanged, snapshot['stages'].get(future)):
                        needed.add(future)
                        possible_unchanged = False
                break
        plans[key] = {'state': state, 'remaining': remaining, 'unchanged': unchanged}
    return PreparedExperiment(store, store._database_revision(), manifest, store.manifest_fingerprint,
                              plans, needed, dict(rows), verification)


def _preflight(store, checkpoints=None, *, item_keys=None, prepared=None):
    prepared = prepared or prepare_experiment(store, checkpoints, item_keys=item_keys)
    prepared.check(store)
    selected = set(prepared.plans) if item_keys is None else set(item_keys)
    if not selected.issubset(prepared.plans):
        raise ValueError('Execution item selection is outside the prepared manifest')
    plans = {key: plan for key, plan in prepared.plans.items() if key in selected}
    needed = set()
    for key, plan in plans.items():
        unchanged = plan['unchanged']
        for stage in plan['remaining']:
            if not _replay(stage, prepared.manifest['target_stage'], unchanged,
                           prepared.manifest['source_checkpoints'][key]['stages'].get(stage)):
                needed.add(stage)
                unchanged = False
    return prepared.manifest, plans, needed


def unfinished_keys(store, *, prepared=None):
    """Choose resumable canonical prefixes; committed failures stay terminal."""
    if store._lock_file is None:
        raise ValueError('Unfinished selection requires the RunStore writer lock')
    prepared = prepared or prepare_experiment(store)
    manifest, plans, _ = _preflight(store, prepared=prepared)
    result = []
    for key, plan in plans.items():
        if not plan['remaining']:
            continue
        stage = plan['remaining'][0]
        expected = _stage_hash(manifest, key, stage, plan['state'], prepared.manifest_digest)
        if not any(row['item_key'] == key and row['stage'] == stage and
                   row['input_fingerprint'] == expected and row['status'] == 'failed' for row in prepared.attempts.get(key, ())):
            result.append(key)
    return result


def run_experiment(store, runner_factory, recorder, *, workers=4, slot_controller=None, runtime=None, item_keys=None, prepared=None):
    """Append native checkpoints, retaining failed attempts and paid-call traces.

    ``runner_factory`` is already bounded in production (see cli.execute_run).
    Offline tests inject in-process runners; no production bypass flag exists.
    The caller installs PostgreSQL support, TraceRecorder and RC prompt wrappers.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    from app.llm.sampling import SamplingPaused
    stop = getattr(recorder, 'stop_event', threading.Event())
    paused = set()
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise RuntimeError('Only one native DeepEye pipeline may run per process')
    resources, question_pool, question_futures = [], None, []
    guard = ExitStack()
    slots = slot_controller or PipelineSlots(fixed_limit=workers)
    tickets, failed = {}, {}
    halted = threading.Event()
    try:
        prepared = prepared or prepare_experiment(store, getattr(recorder, 'sampling_checkpoints', None), item_keys=item_keys)
        manifest, plans, needed = _preflight(store, item_keys=item_keys, prepared=prepared)
        if runtime is not None:
            from scripts.baseline_adapters.deepeye.run_slots import WorkflowSlots
            if runtime.stop_event is not recorder.stop_event:
                raise ValueError('Runtime and recorder must share the same stop event')
            if slot_controller is not None and not isinstance(slot_controller, WorkflowSlots):
                raise ValueError('Legacy pipeline limits are incompatible with the shared runtime')
            slots = slot_controller or WorkflowSlots(len(plans))
        target = manifest['target_stage']
        runners = {}
        for stage in STAGES:
            if stage not in needed:
                continue
            recorder.raise_if_failed()
            items = [plan['state'] for plan in plans.values() if stage in plan['remaining']]
            runner = runner_factory(stage, items)
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
        if runners:
            guard.enter_context(protect_schema_profiles())

        def execute(key):
            plan, snapshot = plans[key], manifest['source_checkpoints'][key]
            state, unchanged = plan['state'], plan['unchanged']
            status = 'succeeded'
            try:
                for stage in plan['remaining']:
                    if stop.is_set():
                        paused.add(key)
                        return 'failed'
                    if halted.is_set():
                        return 'failed'
                    recorder.raise_if_failed()
                    source = snapshot['stages'].get(stage)
                    attempt = store.begin_attempt(key, stage, _stage_hash(manifest, key, stage, state, prepared.manifest_digest))
                    provenance = {'source_run': manifest['source_run'],
                                  'source_manifest_fingerprint': manifest['source_manifest_fingerprint'],
                                  'source_attempt_id': source['attempt_id'] if source else None,
                                  'source_payload_sha256': source['payload_sha256'] if source else None,
                                  'upstream_state_sha256': snapshot['upstream_state_sha256']}
                    if _replay(stage, target, unchanged, source):
                        payload = copy.deepcopy(source['payload'])
                        payload.update(execution_origin=('reused_no_native_llm_call' if stage == target
                                                         else 'reused_unchanged_upstream'),
                                       source_provenance=provenance, attempt_wall_seconds=0.0,
                                       rc_participation={'status': 'rc_not_participating', 'actual_request_count': 0, 'native_request_count': 0,
                                                         'reason': ('no_native_llm_call' if stage == target
                                                                    else 'unchanged_upstream')})
                        store.finish_attempt(attempt, 'succeeded', payload)
                        _restore(state, stage, payload)
                        continue
                    updated = copy.deepcopy(state)
                    contract = manifest.get('contracts', {}).get(key) if (
                        manifest['condition'] == 'rc' and stage == target) else None
                    if manifest['condition'] == 'rc' and stage == target and contract is None:
                        raise ValueError(f'Missing bound RC contract: {key}')
                    if contract is None:
                        context, block = nullcontext(), None
                    else:
                        from .injection import rc_context
                        from .contracts import render_rc_block
                        context, block = rc_context(stage, key, contract), render_rc_block(contract)
                    with recorder.context(attempt), context, runtime.context() if runtime else nullcontext():
                        started = time.monotonic()
                        error = None
                        try:
                            with native_stage_work():
                                getattr(runners[stage], STAGE_METHODS[stage])(updated)
                        except SamplingPaused:
                            stop.set()
                        except Exception as exc:
                            error = exc
                        recorder.raise_if_failed()
                        if stop.is_set():
                            store.append_event(attempt, 'stage_paused', {'resumable': True})
                            paused.add(key)
                            return 'failed'
                        payload = _checkpoint(updated, stage)
                        if getattr(recorder, 'sampling_checkpoints', None):
                            payload['sampling_implementation_version'] = recorder.sampling_checkpoints.source_version
                        trace = api_trace(store.iter_events(attempt, kinds=TRACE_KINDS))
                        if not trace['complete']:
                            raise RuntimeError('Stage finished with incomplete API trace')
                        if 'sampling' in trace:
                            payload['sampling'] = trace['sampling']
                        count = 0
                        restored = trace.get('restored_samples', 0)
                        restored_rc = trace.get('restored_rc_samples', 0)
                        if block is not None:
                            from .injection import count_rc_requests
                            count = count_rc_requests(list(store.iter_events(attempt, kinds='api_request')), block)
                            if count != trace['requests'] or restored_rc != restored:
                                error = MissingRCParticipation(
                                    f'RC present in {count}/{trace["requests"]} logical requests and '
                                    f'{restored_rc}/{restored} restored samples')
                        payload.update(execution_origin='executed', source_provenance=provenance,
                                       attempt_wall_seconds=time.monotonic() - started,
                                       rc_participation={'status': 'participating' if count or restored_rc else 'rc_not_participating',
                                                         'actual_request_count': count,
                                                         'restored_sample_count': restored,
                                                         'restored_rc_sample_count': restored_rc,
                                                         'native_request_count': trace['requests'],
                                                         'reason': ('participating' if count else
                                                                    'restored_rc_samples' if restored_rc else
                                                                    'restored_control_samples' if restored else
                                                                    'no_native_llm_call' if not trace['requests'] else
                                                                    'control' if contract is None else 'rc_not_in_request')})
                        status = 'succeeded' if error is None and _valid_output(updated, stage) else 'failed'
                        if status == 'failed':
                            message = str(error) if error is not None else 'Native required fields or metrics are None'
                            for secret in getattr(recorder, 'secrets', ()):
                                message = message.replace(secret, '[REDACTED]')
                            payload.update(error_type=type(error).__name__ if error else 'IncompleteNativeStageOutput',
                                           error_message=message)
                        store.finish_attempt(attempt, status, payload)
                    if status == 'failed':
                        failed[key] = stage
                        return status
                    unchanged = unchanged and _same_output(payload, source)
                    state = updated
                    plan['state'] = updated
                return status
            except BaseException:
                halted.set()
                raise

        pending = deque(key for key, plan in plans.items() if plan['remaining'])
        if pending:
            question_pool = (runtime.workflow_executor(len(pending)) if runtime else
                             ThreadPoolExecutor(max_workers=slots.max_limit, thread_name_prefix='rc-stage'))
            futures = {}
            while pending or futures:
                for future in tuple(futures):
                    if future.done():
                        key = futures.pop(future)
                        status = future.result()
                        slots.release(tickets.pop(key), status=status)
                recorder.raise_if_failed()
                slots.set_pending(len(pending))
                if stop.is_set():
                    paused.update(pending)
                    pending.clear()
                while pending and not halted.is_set() and not stop.is_set():
                    key = pending[0]
                    ticket = slots.try_acquire(key)
                    if ticket is None:
                        break
                    pending.popleft()
                    tickets[key] = ticket
                    try:
                        future = question_pool.submit(execute, key)
                    except SamplingPaused:
                        stop.set()
                        paused.add(key)
                        slots.release(tickets.pop(key), status='failed')
                        continue
                    futures[future] = key
                    question_futures.append(future)
                if futures:
                    wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
            recorder.raise_if_failed()
        return {'succeeded': len(plans) - len(failed) - len(paused), 'failed': len(failed), 'paused': len(paused),
                'accuracy_evaluated': False, 'target_stage': target, 'condition': manifest['condition'],
                'items': {key: {'status': 'paused' if key in paused else 'failed' if key in failed else 'succeeded',
                                'failed_stage': failed.get(key),
                                'final_selected_sql': plan['state'].final_selected_sql} for key, plan in plans.items()}}
    finally:
        halted.set()
        try:
            close_runners(resources, question_pool=question_pool, question_futures=question_futures)
        finally:
            try:
                guard.close()
            finally:
                for ticket in tickets.values():
                    slots.release(ticket, status='failed')
                slots.set_pending(0)
                _PIPELINE_LOCK.release()
