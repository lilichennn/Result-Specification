"""A supervisor retains its exclusive job lock across prepare and execution."""
import os
from pathlib import Path
import signal
import subprocess
import time

from .configuration import validate_config
from .controller import all_terminal, control, job_paths, now, observe, set_control
from .ledger import CampaignLedger
from .processes import atomic_json, commands, exclusive_lock, process_identity


def run_worker(campaign_dir, job_id, token):
    path = Path(campaign_dir).resolve()
    with CampaignLedger.open(path) as ledger:
        job = next((row for row in ledger.jobs() if row['job_id'] == job_id), None)
        if job is None or not job['process'] or job['process'].get('token') != token:
            raise ValueError('Supervisor token does not match the durable launch intent')
        lock, identity_path = job_paths(path, job)
        with exclusive_lock(lock):
            # Recheck after acquiring ownership; delayed old processes must not
            # overwrite a newer explicit recovery's acknowledgement.
            job = next(row for row in ledger.jobs() if row['job_id'] == job_id)
            if job['process'].get('token') != token:
                raise ValueError('Supervisor launch intent changed before lock acquisition')
            record = {'pid': os.getpid(), 'identity': process_identity(os.getpid()),
                      'token': token, 'phase': 'acknowledged', 'started_at': now(), 'child': None}
            if record['identity'] is None:
                raise RuntimeError('Cannot verify supervisor OS identity')
            atomic_json(identity_path, record)
            stop = False
            previous = {}
            def stopping(*unused):
                nonlocal stop
                stop = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, stopping)

            def stopping_requested():
                return stop or control(path)['mode'] in ('paused', 'blocked')

            def update(phase, **fields):
                record.update(phase=phase, updated_at=now(), **fields)
                atomic_json(identity_path, record)

            def execute(argv, phase):
                update(phase, child=None)
                log = path / 'logs' / f'{job_id}-{token}-{phase}.log'
                with log.open('ab', buffering=0) as stream:
                    child = subprocess.Popen(argv, cwd=ledger.config['code_root'], stdin=subprocess.DEVNULL,
                                             stdout=stream, stderr=subprocess.STDOUT, close_fds=True)
                    update(phase, child={'pid': child.pid, 'identity': process_identity(child.pid)})
                    signalled = False
                    while child.poll() is None:
                        # Preparation has no paid requests and is allowed to
                        # finish atomically. Pause is checked before execution.
                        if phase == 'running' and stopping_requested() and not signalled:
                            child.send_signal(signal.SIGTERM)
                            signalled = True
                        time.sleep(.2)
                    return child.returncode

            try:
                validate_config(ledger.config, inputs=False)
                prepare, run = commands(ledger.config, job)
                if stopping_requested():
                    update('paused', exit_code=0)
                    return 0
                if not Path(job['run_dir']).exists():
                    code = execute(prepare, 'preparing')
                    if code != 0:
                        raise RuntimeError('Native/RC offline preparation failed')
                update('prepared', child=None)
                observation = observe(job)
                manifest = observation['manifest']
                if job['kind'] != 'rc' and 'native_manifest' in ledger.config:
                    frozen = dict(ledger.config['native_manifest'])
                    frozen['items'] = [row for row in frozen['items'] if row['task_key'] in job['items']]
                    frozen['item_count'] = len(frozen['items'])
                    if manifest != frozen:
                        raise ValueError('Prepared native manifest differs from frozen campaign cohort')
                if stopping_requested():
                    update('paused', child=None, exit_code=0)
                    return 0
                # The execution child alone selects unfinished work while it
                # owns the RunStore writer lock. This always runs so its full
                # preflight validates even observationally terminal stores; it
                # can also seal an interrupted master from a committed prefix.
                code = execute(run, 'running')
                if code in (0, 1) and all_terminal(observe(job)):
                    update('finished', child=None, exit_code=code)
                    return 0
                if stopping_requested():
                    update('paused', child=None, exit_code=code)
                    return 0
                raise RuntimeError('Execution exited without all configured items terminal')
            except Exception as exc:
                update('blocked', error_type=type(exc).__name__)
                set_control(ledger, 'blocked', {'job_id': job_id, 'error_type': type(exc).__name__})
                ledger.append_event('supervisor_fault', {'job_id': job_id, 'error_type': type(exc).__name__})
                return 2
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
