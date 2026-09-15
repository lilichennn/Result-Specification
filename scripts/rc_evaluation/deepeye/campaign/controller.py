"""One durable scheduler; independently locked supervisors own each run."""
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .configuration import validate_config
from .ledger import CampaignLedger
from .observations import read_run
from .planning import plan_tick
from .processes import atomic_json, exclusive_lock, lock_held, read_json, process_identity, verified_identity


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def control(path):
    return read_json(Path(path) / 'control.json') or {'mode': 'configured'}


def set_control(ledger, mode, reason=None):
    value = {'mode': mode, 'updated_at': now(), 'reason': reason}
    atomic_json(ledger.campaign_dir / 'control.json', value)
    ledger.append_event('campaign_control', value)


def job_paths(path, job):
    base = Path(path) / 'jobs' / job['job_id']
    return base.with_suffix('.lock'), base.with_suffix('.json')


def observe(job, *, reader=None):
    observation = (reader.read() if reader is not None else
                   read_run(job['run_dir'], kind='rc' if job['kind'] == 'rc' else 'native', target_stage=job['target_stage']))
    if sorted(observation['items']) != sorted(job['items']):
        raise ValueError('Job membership differs from frozen cohort')
    if job['kind'] == 'rc' and observation['manifest'].get('source_run') != job['source_run']:
        raise ValueError('RC source differs from claimed native source')
    return observation


def all_terminal(observation):
    return all(row['status'] in ('succeeded', 'failed') for row in observation['states'].values())


def postgres_fault(result):
    """Recognize adapter diagnostics, not ordinary generated-query errors."""
    if result.get('result_type') != 'execution_error':
        return None
    message = result.get('error_message', '')
    if message == 'PostgreSQL connection failed; check connection settings and server availability':
        return 'PostgreSQLConnectionFailure'
    for sqlstate, name in (('42501', 'PostgreSQLPermissionDenied'), ('3D000', 'PostgreSQLDatabaseMissing')):
        if message.startswith(f'PostgreSQL [{sqlstate}]:'):
            return name
    return None


def check_run_faults(ledger, job):
    """Inspect only new error events, including errors hidden by native fallback."""
    from app.llm.sampling import is_retryable_data_inspection_error
    from scripts.baseline_adapters.deepeye.run_store import RunStore
    path = ledger.campaign_dir / 'fault-cursors.json'
    cursors = read_json(path) or {}
    cursor = cursors.get(job['job_id'], 0)
    with RunStore.open(Path(job['run_dir']), read_only=True) as store, store._read_snapshot() as db:
        latest = db.execute('SELECT COALESCE(MAX(event_id),0) FROM events').fetchone()[0]
        rows = db.execute("SELECT * FROM events WHERE event_id>? AND event_id<=? AND "
                          "(kind='api_error' OR (kind='sql_execute_result' AND "
                          "json_extract(payload_json,'$.result.result_type')='execution_error')) ORDER BY event_id",
                          (cursor, latest)).fetchall()
        errors = [store._event_dict(row) for row in rows]
    faults = []
    for event in errors:
        if event['kind'] == 'sql_execute_result':
            name = postgres_fault(event['payload'].get('result', {}))
            if name:
                faults.append({'job_id': job['job_id'], 'event_id': event['event_id'], 'error_type': name})
            continue
        error = event['payload'].get('error', {})
        if is_retryable_data_inspection_error(error):
            continue  # Sample retry/exhaustion is not a run-wide infrastructure fault.
        name = error.get('type', {}).get('qualname')
        code = error.get('status_code')
        if code in (400, 401, 403, 404, 422) or name in ('AuthenticationError', 'PermissionDeniedError', 'ConfigurationError'):
            faults.append({'job_id': job['job_id'], 'event_id': event['event_id'], 'error_type': name, 'status_code': code})
    if faults:
        ledger.append_event('system_fault_evidence', {'faults': faults})
        set_control(ledger, 'blocked', faults[0])
    # A cursor acknowledges inspection. Never acknowledge a fault before its
    # evidence and block are durable: interruption may duplicate a report, but
    # cannot consume the only record that requires explicit repair/resume.
    if latest > cursor:
        cursors[job['job_id']] = latest
        atomic_json(path, cursors)
    if faults:
        raise RuntimeError('Provider or PostgreSQL infrastructure fault requires explicit repair and resume')


def reconcile(ledger, *, observer=observe):
    """No inferred ownership from PIDs; delayed acknowledgements remain pending."""
    result = {}
    for job in ledger.jobs():
        if job['state'] == 'finished' or not job['process']:
            result[job['job_id']] = job['state']
            continue
        lock, path = job_paths(ledger.campaign_dir, job)
        record = read_json(path)
        token = job['process']['token']
        if not record or record.get('token') != token:
            # A previous acknowledged launch may still be visible until the new
            # supervisor claims its lock. Intent alone never licenses relaunch.
            result[job['job_id']] = 'pending'
            continue
        held = lock_held(lock)
        phase = record.get('phase')
        if held:
            if verified_identity(record):
                result[job['job_id']] = phase if phase in ('prepared', 'running') else 'pending'
            else:
                result[job['job_id']] = 'blocked'
        elif phase == 'finished' and record.get('exit_code') in (0, 1):
            if all_terminal(observer(job)):
                ledger.update_job(job['job_id'], state='finished')
                result[job['job_id']] = 'finished'
            else:
                result[job['job_id']] = 'blocked'
        elif phase == 'paused':
            ledger.update_job(job['job_id'], state='paused')
            result[job['job_id']] = 'paused'
        else:
            result[job['job_id']] = 'blocked'
        if result[job['job_id']] == 'blocked':
            reason = {'job_id': job['job_id'], 'reason': record.get('error_type', 'unverified_or_nonterminal_exit')}
            if job['state'] != 'blocked':
                ledger.update_job(job['job_id'], state='blocked', detail=reason)
                set_control(ledger, 'blocked', reason)
    return result


def worker_command(config, campaign_dir, job_id, token):
    return [config['python'], '-E', '-B', str(Path(config['code_root']) / 'scripts/deepeye_campaign.py'),
            '_worker', '--campaign-dir', str(campaign_dir), '--job-id', job_id, '--token', token]


def launch(ledger, job):
    if control(ledger.campaign_dir)['mode'] != 'running':
        return None
    token = uuid.uuid4().hex
    # Committed before Popen; the worker can acknowledge without its parent.
    ledger.update_job(job['job_id'], state='running', process={'token': token, 'intent_at': now()})
    argv = worker_command(ledger.config, ledger.campaign_dir, job['job_id'], token)
    path = ledger.campaign_dir / 'logs' / (job['job_id'] + '-' + token + '.log')
    try:
        with path.open('ab', buffering=0) as stream:
            child = subprocess.Popen(argv, cwd=ledger.config['code_root'], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    except OSError as exc:
        ledger.update_job(job['job_id'], state='blocked', process={}, detail={'error_type': type(exc).__name__})
        set_control(ledger, 'blocked', 'supervisor_spawn_failed')
        raise
    ledger.append_event('supervisor_spawned', {'job_id': job['job_id'], 'token': token, 'pid': child.pid})
    return child


def status(path):
    """Read-only, offline, including an explicit timestamp for stale snapshots."""
    path = Path(path).resolve()
    with CampaignLedger.open(path, read_only=True) as ledger:
        from ..injection import rc_labels
        snapshot = read_json(path / 'status.json') or {}
        return {**snapshot, 'campaign_dir': str(path), **control(path), **rc_labels({'condition': 'rc', **ledger.config}),
                'jobs': ledger.jobs(), 'items': len(ledger.config['items']),
                'controller': read_json(path / 'controller.json')}


def pause(path):
    # Workers read this durable request and forward graceful stop to their own
    # Popen child. No bare-PID signalling, including during acknowledgement gaps.
    with CampaignLedger.open(path) as ledger:
        set_control(ledger, 'paused', 'explicit_pause')
    return status(path)


def run(path, *, resume=False):
    path = Path(path).resolve()
    with exclusive_lock(path / 'controller.lock'), CampaignLedger.open(path) as ledger:
        config = ledger.config
        atomic_json(path / 'controller.json', {'pid': os.getpid(), 'identity': process_identity(os.getpid()), 'started_at': now()})
        try:
            validate_config(config)
        except Exception as exc:
            set_control(ledger, 'blocked', {'error_type': type(exc).__name__, 'reason': 'frozen_configuration_validation'})
            raise
        states = reconcile(ledger)
        if control(path)['mode'] in ('paused', 'blocked') and not resume:
            raise ValueError('Campaign requires explicit resume after pause or fault')
        if resume:
            for job in ledger.jobs():
                if states[job['job_id']] not in ('blocked', 'paused'):
                    continue
                lock, identity = job_paths(path, job)
                record = read_json(identity)
                if lock_held(lock) or (record and (verified_identity(record) or verified_identity(record.get('child')))):
                    raise ValueError('Cannot resume an ambiguous or still-owned supervisor/child')
                ledger.update_job(job['job_id'], state='prepared' if Path(job['run_dir']).exists() else 'planned', process={})
        set_control(ledger, 'running')
        previous = {}
        def stopped(*unused):
            set_control(ledger, 'paused', 'controller_signal')
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stopped)
        children = []
        observed_finished = set()
        readers = {}
        try:
            while True:
                from .observations import RunObservationReader
                tick_observations = {}
                def current_observation(job):
                    key = job['job_id']
                    if key not in tick_observations:
                        if key not in readers:
                            readers[key] = RunObservationReader(job['run_dir'],
                                kind='rc' if job['kind'] == 'rc' else 'native', target_stage=job['target_stage'])
                        tick_observations[key] = observe(job, reader=readers[key])
                    return tick_observations[key]
                states = reconcile(ledger, observer=current_observation)
                observations = {}
                for job in ledger.jobs():
                    if job['job_id'] in observed_finished:
                        continue
                    _, identity = job_paths(path, job)
                    record = read_json(identity)
                    # Never inspect a partially created store during prepare.
                    if record and record.get('phase') in ('prepared', 'running', 'finished', 'paused', 'blocked') and Path(job['run_dir']).exists():
                        check_run_faults(ledger, job)
                        observations[job['job_id']] = current_observation(job)
                result = plan_tick(ledger, observations)
                jobs = ledger.jobs()
                observed_finished.update(job['job_id'] for job in jobs
                                         if job['state'] == 'finished' and job['job_id'] in observations)
                for key in observed_finished.intersection(readers):
                    readers.pop(key).close()
                from .monitoring import snapshot
                atomic_json(path / 'status.json', snapshot(ledger, result, states, jobs=jobs))
                if result['complete'] and all(job['state'] == 'finished' for job in jobs):
                    set_control(ledger, 'complete')
                    return status(path)
                mode = control(path)['mode']
                if mode == 'running':
                    for job in jobs:
                        if job['state'] in ('planned', 'prepared') and not job['process']:
                            validate_config(config, inputs=False)
                            child = launch(ledger, job)
                            if child is not None:
                                children.append(child)
                elif not any(value in ('pending', 'running', 'prepared') for value in states.values()):
                    return status(path)
                children[:] = [child for child in children if child.poll() is None]
                deadline = time.monotonic() + config['poll_seconds']
                while time.monotonic() < deadline:
                    time.sleep(min(.2, max(0, deadline - time.monotonic())))
                    if control(path)['mode'] != mode:
                        break
        except Exception as exc:
            set_control(ledger, 'blocked', {'error_type': type(exc).__name__})
            raise
        finally:
            for reader in readers.values():
                reader.close()
            # Normal loop exit only follows released job locks. Reap our direct
            # supervisors, which may still be completing Python interpreter exit.
            for child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass  # Ownership persists; no forced termination.
            for sig, handler in previous.items():
                signal.signal(sig, handler)
