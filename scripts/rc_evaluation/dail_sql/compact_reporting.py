"""Resumable, compressed, current-version SQL evaluation for DAIL.

The raw DAIL RunStores remain the source for generation, selection, requests,
and tokens.  This module stores only post-hoc correctness evidence: one compact
item per question and zlib-compressed query results.  SQLite queries are shared
across questions by physical database + SQL; PostgreSQL observations retain
DeepEye's per-question scope.  Timeouts are observations and are never retried.
"""

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import uuid
import zlib

from scripts.baseline_adapters.dail_sql.config import MODES, TaskKey
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from scripts.baseline_adapters.dail_sql.execution import execute_sql
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.deepeye.run_store import restore_jsonable, to_jsonable
from scripts.rc_evaluation.deepeye import comparison
from .evaluation import evaluate_candidate
from .reporting import _bindings, _expected, _read_json, _versions, _write_json


FORMAT = 'dail-compact-evaluation-v1'
QUERY_RESULT_ENCODING = 'run-store-typed-json-once'


def _canonical(value):
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def evaluation_policy(keys, bindings, timeout, profile):
    if profile not in ('batch', 'deepeye'):
        raise ValueError('Unknown SQL evaluation profile')
    groups = {}
    for group in dict.fromkeys(key.group for key in keys):
        if profile == 'batch':
            groups[group] = {'timeout_seconds':timeout, 'question_workers':1, 'query_workers':0}
            continue
        dialects = {bindings[key.group][key.question_id]['database']['dialect']
                    for key in keys if key.group == group}
        if dialects == {'postgresql'}:
            groups[group] = {'timeout_seconds':30, 'question_workers':5, 'query_workers':0}
        elif dialects == {'sqlite'}:
            groups[group] = {'timeout_seconds':600,
                             'question_workers':16 if group == 'bird_dev' else 4,
                             'query_workers':16 if group == 'bird_dev' else 0}
        else:
            raise ValueError('Evaluation profile requires one supported dialect per group')
    return {'profile':profile, 'parallel_groups':profile == 'deepeye', 'groups':groups}


def _task_identity(key):
    return json.dumps(asdict(key), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _query_identity(key, database, database_version, sql):
    dialect = database['dialect']
    if dialect == 'sqlite':
        scope = {'dialect':'sqlite', 'path':str(Path(database['path']).resolve(strict=True)),
                 'database_version':database_version}
    elif dialect == 'postgresql':
        scope = {'dialect':'postgresql', 'task_key':asdict(key),
                 'database_id':database['database_id'], 'database_version':database_version}
    else:
        raise ValueError('Unsupported evaluation dialect')
    scope_json = json.dumps(to_jsonable(scope), ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
    return _sha(json.dumps([scope_json, sql], ensure_ascii=False,
                           separators=(',', ':')).encode()), scope_json


def _result_signature(result):
    value = {key:item for key,item in result.items() if key not in ('elapsed_seconds', 'execution_id')}
    return _sha(_canonical(value))


class EvaluationStore:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._inflight = {}
        self._db = sqlite3.connect(self.path, timeout=120, isolation_level=None, check_same_thread=False)
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.execute('PRAGMA synchronous=NORMAL')
        self._db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS queries(
                cache_key TEXT PRIMARY KEY, scope_json TEXT NOT NULL, sql TEXT,
                status TEXT NOT NULL, result BLOB NOT NULL, sha256 TEXT NOT NULL,
                raw_bytes INTEGER NOT NULL, result_signature TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS items(
                batch TEXT NOT NULL, grp TEXT NOT NULL, question TEXT NOT NULL,
                version_id TEXT, state TEXT NOT NULL, payload_json TEXT NOT NULL, sha256 TEXT NOT NULL,
                PRIMARY KEY(batch,grp,question));
        ''')
        frozen = json.dumps(to_jsonable(metadata), ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
        row = self._db.execute("SELECT value FROM metadata WHERE key='frozen'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO metadata VALUES('frozen',?)", (frozen,))
        elif row[0] != frozen:
            raise ValueError('Existing compact evaluation has different frozen inputs or policy')

    def close(self):
        with self._lock:
            self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            self._db.close()

    def quick_check(self):
        with self._lock:
            value = self._db.execute('PRAGMA quick_check').fetchone()[0]
        if value != 'ok':
            raise ValueError('Compact evaluation SQLite integrity check failed: ' + value)

    def _decode(self, row):
        raw = zlib.decompress(row[0])
        if _sha(raw) != row[1]:
            raise ValueError('Compressed SQL result checksum mismatch')
        return restore_jsonable(json.loads(raw))

    def get_query(self, cache_key):
        with self._lock:
            row = self._db.execute('SELECT result,sha256 FROM queries WHERE cache_key=?', (cache_key,)).fetchone()
        if row is None:
            return None
        value = self._decode(row)
        value['execution_id'] = cache_key
        return value

    def _put_query(self, cache_key, scope_json, sql, result):
        value = {key:item for key,item in result.items() if key != 'execution_id'}
        raw = _canonical(value)
        signature = _result_signature(value)
        compressed = zlib.compress(raw, 6)
        with self._lock:
            row = self._db.execute('SELECT result,sha256,result_signature FROM queries WHERE cache_key=?',
                                   (cache_key,)).fetchone()
            if row is not None:
                if row[2] != signature:
                    raise ValueError('Conflicting observations for the same SQL cache identity')
                return
            self._db.execute('INSERT INTO queries VALUES(?,?,?,?,?,?,?,?)',
                (cache_key, scope_json, sql, value.get('status','unknown'), compressed,
                 _sha(raw), len(raw), signature))

    def query(self, key, binding, sql, timeout_seconds):
        database = binding['database']
        cache_key, scope_json = _query_identity(key, database, binding.get('database_version'), sql)
        cached = self.get_query(cache_key)
        if cached is not None:
            return cached
        with self._lock:
            future = self._inflight.get(cache_key)
            owner = future is None
            if owner:
                future = self._inflight[cache_key] = Future()
        if not owner:
            return future.result()
        try:
            result = execute_sql(database, sql, timeout_seconds=timeout_seconds)
            self._put_query(cache_key, scope_json, sql, result)
            value = self.get_query(cache_key)
            future.set_result(value)
            return value
        except BaseException as error:
            future.set_exception(error)
            future.exception()
            raise
        finally:
            with self._lock:
                self._inflight.pop(cache_key, None)

    def import_seed(self, seed, policy):
        seed = Path(seed)
        compact = (seed/'evaluation.sqlite3').is_file()
        description = _read_json(seed/'versions.json') if compact else _read_json(seed/'progress.json')
        if description.get('evaluation_policy') != policy:
            raise ValueError('Uncompressed seed uses a different evaluation policy')
        source = seed/('evaluation.sqlite3' if compact else 'executions.jsonl')
        stat = source.stat()
        marker = json.dumps({'kind':'compact' if compact else 'jsonl', 'path':str(seed.resolve()), 'bytes':stat.st_size,
                             'mtime_ns':stat.st_mtime_ns}, sort_keys=True)
        with self._lock:
            prior = self._db.execute("SELECT value FROM metadata WHERE key='seed'").fetchone()
        if prior is not None:
            if prior[0] != marker:
                raise ValueError('A different seed was already imported')
            return {'already_imported':True, 'rows':0}
        rows = inserted = 0
        if compact:
            legacy_double_encoded = description.get('query_result_encoding') != QUERY_RESULT_ENCODING
            with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as connection:
                for cache_key, scope, sql, blob, expected_hash in connection.execute(
                        'SELECT cache_key,scope_json,sql,result,sha256 FROM queries'):
                    rows += 1
                    raw = zlib.decompress(blob)
                    if _sha(raw) != expected_hash:
                        raise ValueError('Compact seed SQL result checksum mismatch')
                    result = restore_jsonable(json.loads(raw))
                    if legacy_double_encoded:
                        result = restore_jsonable(result)
                    before = self.get_query(cache_key)
                    self._put_query(cache_key,scope,sql,result)
                    inserted += before is None
        else:
            with source.open(encoding='utf-8') as stream:
                for line in stream:
                    rows += 1
                    try:
                        row = restore_jsonable(json.loads(line))
                    except json.JSONDecodeError:
                        if not line.endswith('\n'):
                            rows -= 1
                            break
                        raise
                    key = TaskKey(**row['task_key'])
                    expected = policy['groups'][key.group]['timeout_seconds']
                    if row['result'].get('timeout_seconds') != expected:
                        raise ValueError('Seed SQL timeout differs from evaluation policy')
                    cache_key, scope = _query_identity(key, row['database'], row.get('database_version'), row['sql'])
                    before = self.get_query(cache_key)
                    self._put_query(cache_key, scope, row['sql'], row['result'])
                    inserted += before is None
        with self._lock:
            self._db.execute("INSERT INTO metadata VALUES('seed',?)", (marker,))
        return {'already_imported':False, 'rows':rows, 'inserted':inserted}

    def item(self, key):
        with self._lock:
            row = self._db.execute('SELECT version_id,state FROM items WHERE batch=? AND grp=? AND question=?',
                                   (key.batch_id,key.group,key.question_id)).fetchone()
        return None if row is None else {'version_id':row[0], 'state':row[1]}

    def put_item(self, key, version_id, state, payload):
        raw = json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
        with self._lock:
            row = self._db.execute('SELECT version_id,payload_json,sha256 FROM items WHERE batch=? AND grp=? AND question=?',
                                   (key.batch_id,key.group,key.question_id)).fetchone()
            if row is not None:
                if row != (version_id, raw, _sha(raw.encode())):
                    raise ValueError('Conflicting compact item for frozen question version')
                return
            self._db.execute('INSERT INTO items VALUES(?,?,?,?,?,?,?)',
                (key.batch_id,key.group,key.question_id,version_id,state,raw,_sha(raw.encode())))

    def completed_by_group(self):
        with self._lock:
            return dict(self._db.execute('SELECT grp,count(*) FROM items GROUP BY grp'))

    def counts(self):
        with self._lock:
            return {'items':self._db.execute('SELECT count(*) FROM items').fetchone()[0],
                    'queries':self._db.execute('SELECT count(*) FROM queries').fetchone()[0],
                    'query_raw_bytes':self._db.execute('SELECT coalesce(sum(raw_bytes),0) FROM queries').fetchone()[0],
                    'query_compressed_bytes':self._db.execute('SELECT coalesce(sum(length(result)),0) FROM queries').fetchone()[0],
                    'query_statuses':dict(self._db.execute('SELECT status,count(*) FROM queries GROUP BY status'))}

    def iter_payloads(self):
        with self._lock:
            rows = list(self._db.execute('SELECT payload_json FROM items ORDER BY grp,question'))
        for row in rows:
            yield restore_jsonable(json.loads(row[0]))


def _after_item(_key):
    """Test seam after durable item publication; no production side effect."""


def _compact_item(key, version_id, version, rounds, reference):
    candidates = []
    by_id = {}
    compact_rounds = []
    for rid, round_ in rounds.items():
        current = []
        for candidate in round_['candidates']:
            evaluation = candidate['evaluation']
            value = {'round_execution_id':rid, 'choice_position':candidate.get('choice_position'), **evaluation}
            candidates.append(value)
            by_id[candidate['candidate_id']] = value
            current.append(value)
        matched = sum(c['bag_equal'] is True for c in current)
        evaluable = sum(c['bag_equal'] is not None for c in current)
        compact_rounds.append({key_:round_.get(key_) for key_ in (
            'round_execution_id','round_no','status','rc_injected','actual_parent_round_id','example_ids','next_example_ids')}
            | {'candidate_ids':[c['candidate_id'] for c in round_['candidates']],
               'candidate_count':len(current), 'matched':matched, 'evaluable':evaluable,
               'unknown':len(current)-evaluable,
               'selected_candidate_id':(round_.get('selection') or {}).get('candidate_id')})
    modes = []
    for mode in MODES:
        value = version['modes'][mode]
        selected = by_id.get(value.get('final_candidate_id'))
        modes.append({**value, 'final_match':selected.get('bag_equal') if selected else None,
                      'final_evaluation_status':selected.get('status') if selected else None})
    return {'task_key':asdict(key), 'version_id':version_id, 'state':'evaluated',
            'reference':{'execution_id':reference.get('execution_id'), 'status':reference.get('status'),
                         'error':reference.get('error'), 'rows':len(reference.get('rows') or []),
                         'columns':len(reference.get('columns') or [])},
            'modes':modes, 'rounds':compact_rounds, 'candidates':candidates}


def _summary(store, policy, started, seed_result):
    groups = Counter()
    modes = Counter()
    candidates = Counter()
    for item in store.iter_payloads():
        group = item['task_key']['group']
        groups[(group,item['state'])] += 1
        for mode in item.get('modes', []):
            modes[(group,mode['mode'],mode['status'])] += 1
            final = mode.get('final_match')
            modes[(group,mode['mode'],'final_true' if final is True else 'final_false' if final is False else 'final_unknown')] += 1
        for candidate in item.get('candidates', []):
            value = candidate.get('bag_equal')
            candidates[(group,'true' if value is True else 'false' if value is False else 'unknown')] += 1
    return {'format':FORMAT, 'evaluation_observation':{
                'started_at':started, 'finished_at':datetime.now(timezone.utc).isoformat(),
                'boundary':'SQL queries observe database contents during this export; no global database snapshot',
                'evaluation_policy':policy, 'timeouts_are_retried':False},
            'storage':store.counts(), 'seed_import':seed_result,
            'groups':[{'group':g, 'state':s, 'count':n} for (g,s),n in sorted(groups.items())],
            'mode_counts':[{'group':g, 'mode':m, 'status':s, 'count':n} for (g,m,s),n in sorted(modes.items())],
            'candidate_matches':[{'group':g, 'outcome':o, 'count':n} for (g,o),n in sorted(candidates.items())],
            'metric_boundary':'diagnostic counts only; paper metrics are not preselected'}


def compact_current_export(batch_root, output_root=None):
    root = Path(batch_root)
    parent = Path(output_root) if output_root is not None else root/'exports/compact'
    latest = parent/'latest.json'
    if not latest.exists():
        return None
    name = _read_json(latest)['directory']
    if Path(name).name != name:
        raise ValueError('Invalid compact export pointer')
    directory = latest.parent/name
    versions = _read_json(directory/'versions.json')
    manifest = _read_json(root/'manifest.json')
    with CurrentIndex(root/'current.sqlite3', read_only=True) as index:
        current = _versions(_expected(manifest), index.snapshot())
    return directory if versions['versions'] == current else None


def export_compact_current(batch_root, *, evaluation_profile='deepeye', seed=None, output_root=None):
    root = Path(batch_root)
    manifest = _read_json(root/'manifest.json')
    keys = _expected(manifest)
    bindings, source = _bindings(root, manifest)
    timeout = manifest.get('sql_timeout_seconds', 60)
    if not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('SQL timeout must be positive and finite')
    policy = evaluation_policy(keys, bindings, timeout, evaluation_profile)
    with CurrentIndex(root/'current.sqlite3', read_only=True) as index:
        snapshot = index.snapshot()
    version_list = _versions(keys, snapshot)
    manifest_sha = _sha((root/'manifest.json').read_bytes())
    frozen = {'format':FORMAT, 'manifest_sha256':manifest_sha, 'versions':version_list,
              'evaluation_source':source, 'evaluation_policy':policy,
              'comparison_source_sha256':_sha(Path(comparison.__file__).read_bytes()),
              'exporter_source_sha256':_sha(Path(__file__).read_bytes()),
              'query_result_encoding':QUERY_RESULT_ENCODING}
    parent = Path(output_root) if output_root is not None else root/'exports/compact'
    existing = compact_current_export(root, parent)
    if existing is not None:
        saved = _read_json(existing/'versions.json')
        if all(saved.get(key) == value for key,value in frozen.items()):
            return existing
    parent.mkdir(parents=True, exist_ok=True)
    frozen_id = _sha(_canonical(frozen))[:16]
    work_dir = parent/('.working-' + evaluation_profile + '-' + frozen_id)
    work_dir.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    store = EvaluationStore(work_dir/'evaluation.sqlite3', frozen)
    try:
        seed_result = store.import_seed(seed, policy) if seed is not None else None
        completed = store.completed_by_group()
        progress = {'phase':'sql_evaluation', 'evaluation_policy':policy,
                    'groups':{g:{'total':sum(k.group == g for k in keys),
                                 'completed':completed.get(g,0)} for g in policy['groups']}}
        _write_json(work_dir/'progress.json', progress)
        with DailRecords(root, manifest, read_only=True) as records:
            def evaluate_question(key, group_policy, query_pool):
                version_id = snapshot.get(key)
                prior = store.item(key)
                if prior is not None:
                    if prior['version_id'] != version_id:
                        raise ValueError('Compact item version differs from frozen current version')
                    return
                if version_id is None:
                    store.put_item(key, None, 'pending', {'task_key':asdict(key), 'version_id':None,
                                                         'state':'pending','modes':[],'rounds':[],'candidates':[]})
                    _after_item(key)
                    return
                if not records.is_sealed(key, version_id):
                    raise ValueError('Current version must be sealed')
                version = records.get_version(version_id)
                rounds = {}
                for rid in version['round_ids']:
                    rounds[rid] = records.find_source(version_id, 'round_result', rid)['payload']
                binding = bindings[key.group][key.question_id]
                sqls = list(dict.fromkeys([binding['reference_sql']] +
                    [candidate.get('candidate_sql') for round_ in rounds.values()
                     for candidate in round_['candidates']]))
                identity = json.dumps([binding['database'],binding.get('database_version')], sort_keys=True)
                local = {}
                if query_pool is None:
                    values = {sql:store.query(key,binding,sql,group_policy['timeout_seconds']) for sql in sqls}
                else:
                    futures = {sql:query_pool.submit(store.query,key,binding,sql,group_policy['timeout_seconds'])
                               for sql in sqls}
                    values = {sql:future.result() for sql,future in futures.items()}
                local.update({(identity,sql):value for sql,value in values.items()})
                def execute(_database, sql):
                    return store.query(key,binding,sql,group_policy['timeout_seconds'])
                for round_ in rounds.values():
                    for candidate in round_['candidates']:
                        candidate['evaluation'] = evaluate_candidate(candidate,binding,execute=execute,cache=local)
                payload = _compact_item(key,version_id,version,rounds,values[binding['reference_sql']])
                store.put_item(key,version_id,'evaluated',payload)
                _after_item(key)

            progress_lock = threading.Lock()
            def evaluate_group(group):
                group_policy = policy['groups'][group]
                pending = [key for key in keys if key.group == group and store.item(key) is None]
                query_context = (ThreadPoolExecutor(max_workers=group_policy['query_workers'])
                                 if group_policy['query_workers'] else nullcontext(None))
                with query_context as query_pool, ThreadPoolExecutor(max_workers=group_policy['question_workers']) as pool:
                    def work(key):
                        evaluate_question(key,group_policy,query_pool)
                        with progress_lock:
                            progress['groups'][group]['completed'] += 1
                            count = progress['groups'][group]['completed']
                            if count % 25 == 0 or count == progress['groups'][group]['total']:
                                _write_json(work_dir/'progress.json', progress)
                                print(f"Compact SQL export {group}: {count}/{progress['groups'][group]['total']}", flush=True)
                    list(pool.map(work,pending))

            with ThreadPoolExecutor(max_workers=len(policy['groups']) if policy['parallel_groups'] else 1) as pool:
                list(pool.map(evaluate_group,policy['groups']))
        if store.counts()['items'] != len(keys):
            raise ValueError('Compact export did not cover every manifest question')
        store.quick_check()
        summary = _summary(store,policy,started,seed_result)
        _write_json(work_dir/'summary.json', summary)
        _write_json(work_dir/'versions.json', {**frozen, 'sql_timeout_seconds':None,
            'scope':'full', 'note':'Per-group limits are authoritative; null is not unlimited.'})
        progress['phase'] = 'complete'
        _write_json(work_dir/'progress.json', progress)
    finally:
        store.close()
    destination = parent/uuid.uuid4().hex
    os.replace(work_dir,destination)
    pointer = parent/('.latest-' + uuid.uuid4().hex)
    _write_json(pointer,{'directory':destination.name})
    os.replace(pointer,parent/'latest.json')
    return destination
