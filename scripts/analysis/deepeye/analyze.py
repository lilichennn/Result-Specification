"""Read-only RC3 analysis. No model calls or mutation of experiment RunStores.

Run as python -m scripts.analysis.deepeye.analyze extract --campaign-root PATH.
The evaluate action executes benchmark SQL; other tools read saved assessments.
Derived query results are compressed and committed in --analysis-dir.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
import zlib

from .paths import ROOT, DEFAULT_ANALYSIS, add_analysis_argument
from scripts import deepeye_run  # Register native serialized types using the public entry point.

HERE = DEFAULT_ANALYSIS
from scripts.baseline_adapters.deepeye.run_store import to_jsonable, restore_jsonable
from scripts.baseline_adapters.deepeye.run_usage import _effective_sampling
from scripts.rc_evaluation.deepeye.evaluation import _references, _reference, _executor, _selection_trace, schema_coverage
from scripts.rc_evaluation.deepeye.comparison import compare_results, SUCCESS_TYPES, _rows
from scripts.rc_evaluation.deepeye.injection import manifest_prompt

STAGES = ('schema_linking', 'sql_generation', 'sql_revision', 'sql_selection')
GROUPS = {'bird_dev': 1534, 'spider_dev': 1034, 'spider_test': 2147,
          'bird_interact_lite': 195, 'bird_interact_full': 410}


def sha(raw):
    return hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


def checked(raw, checksum):
    assert sha(raw) == checksum, 'source payload checksum mismatch'
    return json.loads(raw)


def read_db(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as handle:
        json.dump(to_jsonable(value), handle, ensure_ascii=False, indent=2)


def extract(group, campaign_root, campaign_pattern='{group}'):
    dest = HERE / group
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'offline.json').exists():
        print(group, 'reuse extracted inputs', flush=True)
        return
    campaign = Path(campaign_root) / campaign_pattern.format(group=group)
    with read_db(campaign / 'campaign.sqlite3') as db:
        jobs = [dict(row) for row in db.execute('select * from jobs order by kind desc, job_id')]
    assert all(j['state'] == 'finished' for j in jobs)
    assert Counter(j['kind'] for j in jobs).keys() == {'native_first', 'rc'}
    records, manifests, inputs, contracts = [], [], {}, {}
    native_manifest = None
    for index, job in enumerate(jobs):
        with read_db(Path(job['run_dir']) / 'run.sqlite3') as db:
            raw, checksum = db.execute('select payload_json,payload_checksum from manifest').fetchone()
            m = checked(raw, checksum)
            side = 'native' if job['kind'] == 'native_first' else 'rc'
            manifests.append({'job_id': job['job_id'], 'manifest_sha256': checksum,
                              'run_dir': job['run_dir'], 'condition': side, 'stage': job['target_stage']})
            if side == 'native':
                native_manifest = m
            else:
                assert m['rc_version'] == 3 and m['gold_corrected'] is True
                assert m['condition'] == 'rc' and not m['continue_downstream']
                manifest_prompt(m)
                for key, contract in m['contracts'].items():
                    assert contract['rc_version'] == 3 and contract['source_field'] == 'rc_round3'
                    if key in contracts:
                        assert contracts[key] == contract
                    contracts[key] = contract
                for key, snapshot in m['source_checkpoints'].items():
                    if key not in inputs:
                        inputs[key] = snapshot['input']
            rows, samples, selection_events, budgets = {}, defaultdict(list), defaultdict(list), defaultdict(list)
            for a in db.execute('select a.*,f.status,f.payload_json,f.payload_checksum,f.finished_at '
                                'from attempts a join finishes f using(attempt_id)'):
                if a['stage'] not in STAGES:
                    continue
                assert a['status'] == 'succeeded'
                if side == 'rc':
                    assert a['stage'] == m['target_stage']
                p = checked(a['payload_json'], a['payload_checksum'])
                artifact = restore_jsonable(p['artifact'])
                row = {k: a[k] for k in ('attempt_id', 'item_key', 'stage', 'status', 'started_at', 'finished_at')}
                row.update(condition=side, job_id=job['job_id'], payload_sha256=a['payload_checksum'],
                           origin=p.get('execution_origin', 'executed'), participation=p.get('rc_participation'),
                           sampling=p.get('sampling'), source_provenance=p.get('source_provenance'),
                           wall_seconds=p.get('attempt_wall_seconds'))
                if a['stage'] == 'schema_linking':
                    schema = artifact['database_schema_after_schema_linking']['tables']
                    row['linked'] = {t: list(v['columns']) for t, v in schema.items()}
                    row['linked_recall_recorded'] = artifact.get('final_linking_recall')
                else:
                    field = {'sql_generation': 'sql_candidates', 'sql_revision': 'sql_candidates_after_revision',
                             'sql_selection': 'final_selected_sql'}[a['stage']]
                    row['sqls'] = artifact[field] if a['stage'] != 'sql_selection' else [artifact[field]]
                    assert isinstance(row['sqls'], list)
                rows[a['attempt_id']] = row
            # Read compact successful-sample events, not huge raw API reasoning bodies.
            for e in db.execute("select attempt_id,kind,payload_json,payload_checksum from events where kind in "
                                "('sample_result','sampling_group_start') or (kind='component_result' and "
                                "json_extract(payload_json,'$.component') in "
                                "('selection.shortlist','selection.pairwise_comparison')) order by event_id"):
                if e['attempt_id'] not in rows:
                    continue
                p = checked(e['payload_json'], e['payload_checksum'])
                if e['kind'] == 'sample_result':
                    if side == 'rc' and p['succeeded']:
                        assert p['rc_applied'] is True
                    p.pop('result', None)
                    samples[e['attempt_id']].append({'attempt_id': e['attempt_id'], 'kind': 'sample_result', 'payload': p})
                elif e['kind'] == 'sampling_group_start':
                    budgets[e['attempt_id']].append({'branch_path': p.get('branch_path'), 'target_n': p['target_n']})
                else:
                    selection_events[e['attempt_id']].append({'kind': e['kind'], 'payload': p})
            api = defaultdict(Counter)
            for a in db.execute("select attempt_id,kind,count(*) n from events where kind in "
                                "('api_request','api_response','api_error') group by attempt_id,kind"):
                api[a['attempt_id']][a['kind']] = a['n']
            threshold = m['effective_config']['stages']['sql_selection']['shortcut_consistency_score_threshold']
            for aid, row in rows.items():
                row['effective'] = _effective_sampling(samples[aid])
                row['api'] = dict(api[aid])
                row['budgets'] = budgets[aid]
                row['failed_samples'] = sum(not e['payload']['succeeded'] for e in samples[aid])
                row['sample_attempts'] = sum(e['payload']['attempt_count'] for e in samples[aid])
                if row['stage'] == 'sql_selection' and row['origin'] == 'executed':
                    row['selection_trace'] = _selection_trace(selection_events[aid], [],
                        {'status': 'succeeded', 'payload': {'artifact': {'final_selected_sql': row['sqls'][0]}}}, threshold)
                records.append(row)
        if index % 10 == 0 or index + 1 == len(jobs):
            print(group, f'extract {index+1}/{len(jobs)} runs, {len(records)} stages', flush=True)
    assert native_manifest is not None
    bindings = native_manifest['items']
    lookup = {(r['condition'], r['stage'], r['item_key']): r for r in records}
    assert len(records) == len(lookup) == GROUPS[group] * 8
    assert {b['task_key'] for b in bindings} == set(contracts) == set(inputs)
    for row in records:
        if row['condition'] == 'rc':
            native = lookup['native', row['stage'], row['item_key']]
            p = row['source_provenance']
            assert p['source_attempt_id'] == native['attempt_id'] and p['source_payload_sha256'] == native['payload_sha256']
            if row['origin'] != 'executed':
                field = 'linked' if row['stage'] == 'schema_linking' else 'sqls'
                assert row[field] == native[field]
                if row['stage'] == 'sql_selection':
                    row['selection_trace'] = native['selection_trace']
    refs, sources = _references({}, bindings)
    references = {b['task_key']: _reference(b, refs) for b in bindings}
    db_type = bindings[0]['db_type']
    for row in records:
        if row['stage'] == 'schema_linking':
            linked = row['linked']
            if db_type == 'sqlite':
                linked = {t.lower(): [c.lower() for c in cols] for t, cols in linked.items()}
            row['coverage'] = schema_coverage(references[row['item_key']].get('sql'), linked,
                                              dialect='sqlite' if db_type == 'sqlite' else 'postgres')
    write_new(dest / 'offline.json', {'group': group, 'campaign': str(campaign), 'manifests': manifests,
        'bindings': bindings, 'config': native_manifest['effective_config'], 'references': references,
        'reference_sources': sources, 'inputs': inputs, 'contracts': contracts, 'records': records})
    print(group, 'extraction complete', flush=True)


def assess(predicted, reference):
    result = {**compare_results(predicted, reference),
              'execution_success': predicted.get('result_type') in SUCCESS_TYPES,
              'result_type': predicted.get('result_type'), 'set_equal': None}
    try:
        pc, pr = _rows(predicted)
        rc, rr = _rows(reference)
        result['set_equal'] = set(pr) == set(rr)  # BIRD public EX rule; empty shapes ignored.
    except (TypeError, ValueError, OverflowError, RecursionError):
        pass
    return result


def sql_targets(rows):
    targets = {r['condition'] + '/' + r['stage']: r['sqls'] for r in rows if r['stage'] != 'schema_linking'}
    for r in rows:
        if r['stage'] == 'sql_selection':
            targets[r['condition'] + '/shortlist'] = r['selection_trace']['shortlist_sqls'] or []
    return targets


def evaluate(group, workers, query_workers=0, env_file=ROOT / 'config/.env'):
    dest = HERE / group
    data = json.loads((dest / 'offline.json').read_text())
    bindings = data['bindings']
    postgres = bindings[0]['db_type'] == 'postgresql'
    if postgres and query_workers:
        raise ValueError('Nested query concurrency is SQLite-only; preserve PG admission')
    by_key = defaultdict(list)
    for r in data['records']:
        by_key[r['item_key']].append(r)
    db = sqlite3.connect(dest / 'evaluation.sqlite3', check_same_thread=False, timeout=120)
    db.executescript('PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);'
        'CREATE TABLE IF NOT EXISTS queries(cache_key TEXT PRIMARY KEY, task_key TEXT, sql TEXT, result BLOB, sha256 TEXT);'
        'CREATE TABLE IF NOT EXISTS items(task_key TEXT PRIMARY KEY,payload_json TEXT,sha256 TEXT);')
    # Only analysis artefacts are writable. Every source RunStore and database is read-only.
    local_stats = {b['database_path']: (Path(b['database_path']).stat().st_size, Path(b['database_path']).stat().st_mtime_ns)
                   for b in bindings} if not postgres else {}
    identity = {'offline_sha256': sha((dest / 'offline.json').read_bytes()), 'comparison': 'strict_bag_ordered_and_bird_set_v1',
                'workers': workers, 'local_database_stats': local_stats, 'postgres_snapshot': 'live_readonly_not_immutable',
                'timeout_seconds': data['config'].get('postgres', {}).get('statement_timeout_seconds', 30) if postgres
                                   else data['config']['dataset']['sql_execution_timeout_seconds']}
    raw_identity = json.dumps(identity, sort_keys=True)
    prior = db.execute("select value from metadata where key='identity'").fetchone()
    if prior:
        # Worker count affects evaluation scheduling, not query/comparison identity.
        # Keep the original identity and append each actual invocation below.
        old = json.loads(prior[0])
        assert {k: v for k, v in old.items() if k != 'workers'} == {
            k: v for k, v in json.loads(raw_identity).items() if k != 'workers'}
    else:
        db.execute('insert into metadata values (?,?)', ('identity', raw_identity)); db.commit()
    db.execute('insert into metadata values (?,?)',
        ('invocation_' + datetime.now(timezone.utc).isoformat(), json.dumps({'workers': workers,
            'query_workers': query_workers, 'timeout_seconds': identity['timeout_seconds'], 'resume': prior is not None})))
    db.commit()
    completed = {r[0] for r in db.execute('select task_key from items')}
    lock = threading.Lock()
    in_flight = {}
    from contextlib import nullcontext
    cm = _executor(env_file, data['config']) if postgres else nullcontext((None, None))
    with cm as (pg_execute, _), (ThreadPoolExecutor(max_workers=query_workers) if query_workers else nullcontext(None)) as query_pool:
        if postgres:
            probe = pg_execute('probe', bindings[0]['database_id'], 'SELECT 1 AS connection_check')
            if probe.get('result_type') not in SUCCESS_TYPES:
                raise RuntimeError('PostgreSQL connection unavailable; evaluation not started')
        else:
            from app.db_utils.execution import execute_sql_without_cache

        def query(binding, sql):
            if not isinstance(sql, str) or not sql.strip():
                return {'result_type': 'missing_sql'}
            key = binding['task_key']
            scope = key if postgres else binding['database_path']
            cache_key = sha(json.dumps([scope, sql]))
            with lock:
                cached = db.execute('select result,sha256 from queries where cache_key=?', (cache_key,)).fetchone()
                waiter = in_flight.get(cache_key)
                if not cached and waiter is None:
                    waiter = in_flight[cache_key] = threading.Event()
                    owner = True
                else:
                    owner = False
            if cached:
                raw = zlib.decompress(cached[0]); assert sha(raw) == cached[1]
                return restore_jsonable(json.loads(raw))
            if not owner:
                waiter.wait()
                return query(binding, sql)
            try:
                value = (pg_execute(key, binding['database_id'], sql) if postgres else
                         execute_sql_without_cache(binding['database_path'], sql, timeout=identity['timeout_seconds']).model_dump())
                raw = json.dumps(to_jsonable(value), ensure_ascii=False, separators=(',', ':')).encode()
                with lock:
                    db.execute('insert into queries values (?,?,?,?,?)', (cache_key, key, sql, zlib.compress(raw), sha(raw)))
                    db.commit()
                return value
            finally:
                with lock:
                    in_flight.pop(cache_key).set()

        def task(binding):
            key = binding['task_key']
            reference = data['references'][key]
            targets = sql_targets(by_key[key])
            if reference['status'] == 'available':
                gold = query(binding, reference['sql'])
                sqls = list(dict.fromkeys(s for pool in targets.values() for s in pool))
                if query_pool is None:
                    results = {sql: query(binding, sql) for sql in sqls}
                else:
                    submitted = {query_pool.submit(query, binding, sql): sql for sql in sqls}
                    results = {submitted[f]: f.result() for f in as_completed(submitted)}
                metrics = {sql: assess(result, gold) for sql, result in results.items()}
                pools = {label: [{'sql': sql, **metrics[sql]} for sql in sqls] for label, sqls in targets.items()}
            else:
                gold = {'result_type': reference['status']}
                pools = {label: [{'sql': sql, **assess({'result_type': 'not_executed'}, gold)} for sql in sqls]
                         for label, sqls in targets.items()}
            payload = {'item_key': key, 'reference_status': reference['status'],
                       'reference_success': gold.get('result_type') in SUCCESS_TYPES,
                       'reference_type': gold.get('result_type'), 'reference_error': gold.get('error_message'),
                       'reference_rows': len(gold.get('result_rows') or []),
                       'reference_columns': len(gold.get('result_cols') or []), 'targets': pools}
            raw = json.dumps(to_jsonable(payload), ensure_ascii=False, separators=(',', ':'))
            with lock:
                db.execute('insert into items values (?,?,?)', (key, raw, sha(raw))); db.commit()
            return key, payload['reference_type']

        pending = [b for b in bindings if b['task_key'] not in completed]
        print(group, f'evaluation begin: {len(completed)} cached, {len(pending)} pending, workers={workers}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, b) for b in pending]
            for n, future in enumerate(as_completed(futures), len(completed) + 1):
                key, status = future.result()
                if n % 25 == 0 or n == len(bindings) or status not in SUCCESS_TYPES:
                    print(datetime.now(timezone.utc).isoformat(), group, n, '/', len(bindings), key, status, flush=True)
    for path, before in local_stats.items():
        st = Path(path).stat()
        assert (st.st_size, st.st_mtime_ns) == before, 'source database changed during evaluation'
    assert db.execute('select count(*) from items').fetchone()[0] == len(bindings)
    print(group, 'evaluation complete;', db.execute('select count(*) from queries').fetchone()[0], 'distinct queries', flush=True)
    db.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    add_analysis_argument(ap)
    ap.add_argument('action', choices=('extract', 'evaluate'))
    ap.add_argument('--group', choices=list(GROUPS))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--query-workers', type=int, default=0)
    ap.add_argument('--campaign-root', type=Path, help='Required for extract: parent of campaign directories.')
    ap.add_argument('--campaign-pattern', default='{group}', help='Campaign child name template, with {group}.')
    ap.add_argument('--env-file', type=Path, default=ROOT / 'config/.env', help='PostgreSQL credentials for evaluate (default: config/.env).')
    args = ap.parse_args()
    HERE = args.analysis_dir.resolve()
    if args.action == 'extract' and args.campaign_root is None:
        ap.error('extract requires --campaign-root')
    for group in [args.group] if args.group else GROUPS:
        extract(group, args.campaign_root, args.campaign_pattern) if args.action == 'extract' else evaluate(group, args.workers, args.query_workers, args.env_file)
