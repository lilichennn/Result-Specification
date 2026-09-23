"""POSIX ownership locks and shell-free, source-bound subprocess recipes."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile


@contextmanager
def exclusive_lock(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield descriptor
    finally:
        os.close(descriptor)


def lock_held(path):
    try:
        with exclusive_lock(path):
            return False
    except BlockingIOError:
        return True


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def commands(config, job):
    root = Path(config['code_root'])
    items = [value for key in job['items'] for value in ('--item', key)]
    if job['kind'] != 'rc':
        base = [config['python'], '-E', '-B', str(root / 'scripts/deepeye_run.py')]
        options = ['--run-dir', job['run_dir'], *config['native_args'], *items]
        return base + ['prepare', *options], base + ['resume', *options, '--unfinished-only']
    base = [config['python'], '-E', '-B', str(root / 'scripts/rc_evaluation/deepeye/cli.py')]
    paths = config.get('rc_sources') or {key: config['rc_' + key] for key in ('lite', 'full')}
    contracts = [value for partition, path in paths.items() for value in ('--rc', f'{partition}={path}')]
    prepare = base + ['prepare', '--source-run', job['source_run'], '--run-dir', job['run_dir'],
                      '--target-stage', job['target_stage'], '--condition', 'rc',
                      '--rc-version', str(config.get('rc_version', 2)),
                      *contracts,
                      '--env-file', config['env_file'], *items]
    return prepare, base + ['resume', '--run-dir', job['run_dir'], '--env-file', config['env_file'], '--unfinished-only']


def process_identity(pid):
    """Kernel-observed start time plus argv, never PID existence alone."""
    result = subprocess.run(['/bin/ps', '-ww', '-p', str(pid), '-o', 'lstart=', '-o', 'command='],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def verified_identity(record):
    return bool(record and type(record.get('pid')) is int and record.get('identity') and
                process_identity(record['pid']) == record['identity'])
