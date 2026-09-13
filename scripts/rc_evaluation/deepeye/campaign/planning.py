"""Pure scheduling decisions persisted as one transaction; never launch work."""
import math
from pathlib import Path

from scripts.baseline_adapters.deepeye.run_pipeline import STAGES
from .observations import manifest_items


_TERMINAL = frozenset(('succeeded', 'failed'))


def _counts(states):
    result = {status: 0 for status in ('pending', 'unfinished', 'succeeded', 'failed')}
    for state in states:
        result[state['status']] += 1
    result['total'] = sum(result.values())
    result['terminal'] = result['succeeded'] + result['failed']
    return result


def _ingest(ledger, jobs, observations):
    if not isinstance(observations, dict):
        raise ValueError('observations must be keyed by job id')
    for job_id, observed in observations.items():
        if job_id not in jobs:
            raise ValueError('observation belongs to an unknown job')
        job = jobs[job_id]
        if job['state'] == 'planned':
            raise ValueError('cannot observe an unprepared job')
        if not isinstance(observed, dict) or str(Path(observed['run_dir']).resolve()) != job['run_dir']:
            raise ValueError('observed run directory differs from job')
        if (set(observed['items']) != set(job['items']) or len(observed['items']) != len(job['items']) or
                set(manifest_items(observed['manifest'])) != set(job['items']) or
                set(observed['states']) != set(job['items'])):
            raise ValueError('observed run members differ from job')
        if job['kind'] == 'rc':
            manifest = observed['manifest']
            if (manifest.get('target_stage') != job['target_stage'] or
                    not isinstance(manifest.get('source_run'), str) or
                    str(Path(manifest['source_run']).resolve()) != job['source_run']):
                raise ValueError('observed RC source or target differs from job')
        for key, state in observed['states'].items():
            status = state.get('status')
            if status not in ('pending', 'unfinished', 'succeeded', 'failed'):
                raise ValueError('unknown observed item status')
            attempt_id, finished_at = state.get('attempt_id'), state.get('finished_at')
            if status in _TERMINAL and (not isinstance(attempt_id, str) or not attempt_id or not finished_at):
                raise ValueError('terminal item is missing its attempt identity/time')
            prior = ledger._db.execute('SELECT * FROM observations WHERE job_id=? AND item_key=?', (job_id, key)).fetchone()
            record = (status, attempt_id, finished_at)
            if prior:
                previous = (prior['status'], prior['attempt_id'], prior['finished_at'])
                if prior['status'] in _TERMINAL and record != previous:
                    raise ValueError('a previously terminal item outcome changed')
                if previous == record:
                    continue
            ledger._db.execute('INSERT INTO observations VALUES(?,?,?,?,?) ON CONFLICT(job_id,item_key) DO UPDATE SET status=excluded.status,attempt_id=excluded.attempt_id,finished_at=excluded.finished_at',
                               (job_id, key, status, attempt_id, finished_at))
            ledger.append_event('item_observed', {'job_id': job_id, 'item_key': key, 'status': status,
                                                  'attempt_id': attempt_id, 'finished_at': finished_at})


def plan_tick(ledger, observations):
    """Persist observations, open fixed-cohort gates and claim new work.

    Observations may omit unchanged or finished runs: compact statuses are
    durable. ``canonical_sources`` maps task keys to absolute native run paths.
    Counts contain pending/unfinished/succeeded/failed/total/terminal. Native
    counts are separated into ``first`` and ``retry``; RC counts use stage keys.
    """
    with ledger.transaction():
        jobs = {job['job_id']: job for job in ledger.jobs()}
        _ingest(ledger, jobs, observations)
        recorded = {(row['job_id'], row['item_key']): dict(row)
                    for row in ledger._db.execute('SELECT * FROM observations')}

        def state(job, key):
            return recorded.get((job['job_id'], key), {'status': 'pending', 'attempt_id': None})

        first_jobs = [job for job in jobs.values() if job['kind'] == 'native_first']
        if len(first_jobs) != 1:
            raise ValueError('campaign requires exactly one native first-pass run')
        first = first_jobs[0]
        retries = {key: job for job in jobs.values() if job['kind'] == 'native_retry' for key in job['items']}
        sources = {}
        excluded = []
        failed_first = []
        for key in ledger.config['items']:
            first_state = state(first, key)
            if first_state['status'] == 'succeeded':
                source = first
            elif first_state['status'] == 'failed':
                failed_first.append(key)
                retry = retries.get(key)
                if retry and state(retry, key)['status'] == 'succeeded':
                    source = retry
                else:
                    if retry and state(retry, key)['status'] == 'failed':
                        excluded.append(key)
                    continue
            else:
                continue
            prior = ledger._db.execute('SELECT * FROM canonical WHERE item_key=?', (key,)).fetchone()
            attempt_id = state(source, key)['attempt_id']
            if prior and (prior['job_id'] != source['job_id'] or prior['attempt_id'] != attempt_id):
                raise ValueError('canonical native source cannot be replaced')
            if not prior:
                ledger._db.execute('INSERT INTO canonical VALUES(?,?,?)', (key, source['job_id'], attempt_id))
                ledger.append_event('canonical_selected', {'item_key': key, 'job_id': source['job_id'], 'attempt_id': attempt_id})
            sources[key] = source['run_dir']

        fraction = ledger.config['tail_fraction']
        first_counts = _counts(state(first, key) for key in first['items'])
        opened_before = set(ledger.opened_stages())
        if first_counts['terminal'] >= math.ceil(fraction * first_counts['total']):
            ledger.open_stage(STAGES[0])
        new_jobs = []
        if STAGES[0] in ledger.opened_stages():
            retry_keys = [key for key in failed_first if key not in retries]
            if retry_keys:
                retry = ledger.claim_job(kind='native_retry', items=retry_keys)
                new_jobs.append(retry)
                jobs[retry['job_id']] = retry
                retries.update({key: retry for key in retry_keys})

        # Only already dispatched anchors may advance gates in this tick.
        for previous, following in zip(STAGES, STAGES[1:]):
            anchor = ledger.anchor(previous)
            if anchor:
                members = [(jobs[job_id], key) for job_id in anchor for key in jobs[job_id]['items']]
                terminal = sum(state(job, key)['status'] in _TERMINAL for job, key in members)
                if terminal >= math.ceil(fraction * len(members)):
                    ledger.open_stage(following)

        for stage in ledger.opened_stages():
            claimed = {key for job in jobs.values() if job['kind'] == 'rc' and job['target_stage'] == stage for key in job['items']}
            cohorts = {}
            for key, source_run in sources.items():
                if key not in claimed:
                    cohorts.setdefault(source_run, []).append(key)
            dispatched = []
            for source_run, keys in cohorts.items():
                job = ledger.claim_job(kind='rc', items=keys, source_run=source_run, target_stage=stage)
                dispatched.append(job['job_id'])
                new_jobs.append(job)
                jobs[job['job_id']] = job
            if dispatched and not ledger.anchor(stage):
                ledger.set_anchor(stage, dispatched)

        rc_counts = {stage: _counts(state(job, key) for job in jobs.values()
                                   if job['kind'] == 'rc' and job['target_stage'] == stage for key in job['items'])
                     for stage in STAGES}
        rc_terminal = {(job['target_stage'], key) for job in jobs.values() if job['kind'] == 'rc'
                       for key in job['items'] if state(job, key)['status'] in _TERMINAL}
        complete = (first_counts['terminal'] == first_counts['total'] and
                    all(key in retries and state(retries[key], key)['status'] in _TERMINAL for key in failed_first) and
                    all((stage, key) in rc_terminal for key in sources for stage in STAGES))
        return {
            'jobs': new_jobs,
            'opened_stages': [stage for stage in ledger.opened_stages() if stage not in opened_before],
            'canonical_sources': sources,
            'excluded': sorted(excluded),
            'native_counts': {'first': first_counts,
                              'retry': _counts(state(job, key) for key, job in retries.items())},
            'rc_counts': rc_counts,
            'complete': complete,
        }
