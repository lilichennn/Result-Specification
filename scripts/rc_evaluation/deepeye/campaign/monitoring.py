"""Read-only request/resource accounting and deterministic milestone samples."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.baseline_adapters.deepeye.run_store import RunStore
from .controller import job_paths, now
from .processes import atomic_json, read_json, verified_identity


def resource_observations(ledger):
    prior = (read_json(ledger.campaign_dir / 'status.json') or {}).get('resources', {}).get('runs', {})
    runs, processes = {}, []
    for job in ledger.jobs():
        directory = Path(job['run_dir'])
        _, identity = job_paths(ledger.campaign_dir, job)
        record = read_json(identity)
        if record:
            for process in (record, record.get('child')):
                if not verified_identity(process):
                    continue
                output = subprocess.run(['/bin/ps', '-p', str(process['pid']), '-o', 'rss=', '-o', 'pcpu=', '-o', 'stat='],
                                        text=True, capture_output=True, check=False).stdout.split()
                processes.append({'job_id': job['job_id'], 'pid': process['pid'],
                                  'rss_kib': int(output[0]) if output else None,
                                  'cpu_percent': float(output[1]) if len(output) > 1 else None,
                                  'threads': None})
        if not (directory / 'run.sqlite3').exists():
            continue
        # Do not open stores still undergoing creation/preparation.
        if record and record.get('phase') in ('acknowledged', 'preparing'):
            continue
        previous = prior.get(job['job_id'], {'cursor': 0, 'requests': 0, 'terminals': 0})
        with RunStore.open(directory, read_only=True) as store, store._read_snapshot() as db:
            cursor = db.execute('SELECT COALESCE(MAX(event_id),0) FROM events').fetchone()[0]
            counts = dict(db.execute('SELECT kind,COUNT(*) FROM events WHERE event_id>? AND event_id<=? '
                                     "AND kind IN ('api_request','api_response','api_error') GROUP BY kind",
                                     (previous['cursor'], cursor)).fetchall())
        runs[job['job_id']] = {'cursor': cursor, 'requests': previous['requests'] + counts.get('api_request', 0),
                              'terminals': previous['terminals'] + counts.get('api_response', 0) + counts.get('api_error', 0)}
    return {'occupancy_definition': 'Logical api_request events minus api_response/api_error events; not HTTP in-flight or TCP connections. Interrupted calls can remain outstanding.',
            'http_in_flight': None, 'logical_api_outstanding': sum(row['requests'] - row['terminals'] for row in runs.values()),
            'runs': runs, 'processes': processes, 'load_average': list(os.getloadavg()), 'new_resource_limits': None}


def audit_milestones(ledger):
    directory = ledger.campaign_dir / 'audits'
    directory.mkdir(exist_ok=True)
    jobs = {row['job_id']: row for row in ledger.jobs()}
    groups = {}
    for row in ledger._db.execute("SELECT job_id,item_key,status,attempt_id FROM observations WHERE status IN ('succeeded','failed')"):
        job = jobs[row['job_id']]
        group = job['target_stage'] if job['kind'] == 'rc' else job['kind']
        groups.setdefault(group, []).append(dict(row))
    outputs = []
    for group, rows in groups.items():
        previous = list(directory.glob(group + '-*.json'))
        used = {example['task_key'] for path in previous for example in read_json(path)['examples']}
        for milestone in range(100, len(rows) + 1, 100):
            path = directory / f'{group}-{milestone:06d}.json'
            if path.exists():
                outputs.append(str(path))
                continue
            fresh = sorted((row for row in rows if row['item_key'] not in used), key=lambda row:
                           hashlib.sha256(f"{group}/{milestone}/{row['item_key']}".encode()).hexdigest())[:5]
            examples = []
            for row in fresh:
                job = jobs[row['job_id']]
                with RunStore.open(Path(job['run_dir']), read_only=True) as store:
                    attempt = store.attempt(row['attempt_id'])
                    ids = ([link['attempt_id'] for link in attempt['payload']['stage_attempts']]
                           if job['kind'] != 'rc' else [attempt['attempt_id']])
                    stages = []
                    for identity in ids:
                        stage = store.attempt(identity)
                        with store._read_snapshot() as db:
                            counts = dict(db.execute('SELECT kind,COUNT(*) FROM events WHERE attempt_id=? GROUP BY kind', (identity,)).fetchall())
                        payload = stage['payload']
                        stages.append({'stage': stage['stage'], 'status': stage['status'],
                                       'structured_output': payload.get('artifact'), 'sampling': payload.get('sampling'),
                                       'trace_event_counts': counts, 'trace_available': bool(counts),
                                       'rc_participation': payload.get('rc_participation')})
                examples.append({'task_key': row['item_key'], 'job_id': row['job_id'], 'status': row['status'],
                                 'source_run': job['source_run'] or job['run_dir'], 'stages': stages})
                used.add(row['item_key'])
            audit = {'group': group, 'terminal_milestone': milestone, 'created_at': now(), 'examples': examples,
                     'assessment': 'Queued evidence review. Native completion, empty outputs and fallback do not establish SQL correctness. SQL evaluation and token analysis are separate.'}
            atomic_json(path, audit)
            ledger.append_event('audit_queued', {'group': group, 'milestone': milestone, 'path': str(path),
                                               'items': [row['task_key'] for row in examples]})
            outputs.append(str(path))
    return outputs


def snapshot(ledger, result, states):
    return {**{key: value for key, value in result.items() if key != 'jobs'}, 'observed_at': now(),
            'active_jobs': {key: value for key, value in states.items() if value not in ('finished', 'planned')},
            'resources': resource_observations(ledger), 'audit_files': audit_milestones(ledger),
            'anchors': {stage: ledger.anchor(stage) for stage in ledger.opened_stages()}}
