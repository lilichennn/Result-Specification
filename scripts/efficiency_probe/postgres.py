"""Independent, bounded PostgreSQL replay probe; not a production runner."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import sqlglot
from sqlglot import exp
import psycopg

from .core import distribution, fingerprint, resources, utc, write_json, RunStore


def read_query(sql):
    """Conservative workload filter only; never installed in production SQL execution."""
    try:
        parsed=sqlglot.parse(sql, read='postgres')
    except Exception:
        return False
    if len(parsed)!=1 or not isinstance(parsed[0], exp.Query):
        return False
    root=parsed[0]
    blocked={'Insert','Update','Delete','Merge','Create','Drop','Alter','Command',
             'Copy','Into','Lock','Transaction','Grant','Revoke'}
    if any(type(node).__name__ in blocked for node in root.walk()):
        return False
    safe_anonymous={'JSONB_EXTRACT_PATH_TEXT','JSONB_EXTRACT_PATH','JSONB_ARRAY_LENGTH',
                    'JSONB_ARRAY_ELEMENTS','JSONB_ARRAY_ELEMENTS_TEXT','JSONB_EACH',
                    'JSONB_EACH_TEXT','TO_CHAR','TO_DATE','DATE_TRUNC'}
    if any(f.name.upper() not in safe_anonymous for f in root.find_all(exp.Anonymous)):
        return False
    return not any(isinstance(node,exp.Dot) and isinstance(node.expression,exp.Func)
                   for node in root.walk())


def select_workload(rows):
    unique={}
    for row in rows:
        key=(row['db_id'],row['sql'])
        if key not in unique and 0<=row['historical_rows']<=1000 and read_query(row['sql']):
            unique[key]=row
    per_db=defaultdict(list)
    for row in unique.values():
        per_db[row['db_id']].append(row)
    chosen={}
    for db,items in sorted(per_db.items()):
        items.sort(key=lambda r:(r['historical_seconds'],r['source_event_id']))
        middle=len(items)//2
        second=len(items)-1 if middle!=len(items)-1 else 0
        chosen[db]=[items[middle]]+([items[second]] if second!=middle else [])
    return [items[i] for i in range(2) for items in chosen.values() if i<len(items)]


def dispatch(concurrency, count, execute, record, *, stop=None):
    if type(concurrency) is not int or type(count) is not int or not 1<=concurrency<=50 or not concurrency<=count<=200:
        raise ValueError('probe requires 1..50 concurrent operations and C..200 queries')
    stop=stop or threading.Event()
    rows=[]; issued=0; failures=0; started=time.monotonic()
    def one(index):
        record('query_start',{'request_no':index,'at':utc()})
        row=execute(index)
        record('query_result',row)
        return row
    with cf.ThreadPoolExecutor(max_workers=concurrency,thread_name_prefix='pg-probe') as pool:
        pending=set()
        try:
            while pending or (issued<count and not stop.is_set()):
                if time.monotonic()-started>120:
                    stop.set()
                while not stop.is_set() and issued<count and len(pending)<concurrency:
                    pending.add(pool.submit(one,issued)); issued+=1
                if not pending:
                    break
                done,pending=cf.wait(pending,timeout=.2,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    row=future.result(); rows.append(row)
                    failures+=not row['success']
                if failures>=5:
                    stop.set()
        finally:
            # Context manager drains already submitted work on exceptions too.
            stop.set()
    return rows


def prepare(source, output):
    source=Path(source).resolve()
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as db:
        actual={r[0] for r in db.execute("SELECT json_extract(payload_json,'$.component_call_id') FROM events WHERE kind='postgres_completion'")}
        cursor=db.execute("SELECT event_id,json_extract(payload_json,'$.component_call_id'),"
            "json_extract(payload_json,'$.data_item.database_id'),json_extract(payload_json,'$.sql'),"
            "json_extract(payload_json,'$.result.execution_time'),json_array_length(json_extract(payload_json,'$.result.result_rows')) ,"
            "json_extract(payload_json,'$.data_item.instance_id') FROM events WHERE kind='sql_execute_result' "
            "AND json_extract(payload_json,'$.result.result_type')='success'")
        rows=[dict(source_event_id=e,db_id=d,sql=s,historical_seconds=t,historical_rows=n,item_key=q)
              for e,c,d,s,t,n,q in cursor if c in actual and n is not None and n<=1000]
        latest=db.execute('SELECT MAX(event_id) FROM events').fetchone()[0]
    workload=select_workload(rows)
    result={'source':str(source),'source_max_event':latest,'created_at':utc(),
            'selection':'per database: median and slowest distinct successful actual PG queries; historical rows <=1000; no gold or SQL rewriting',
            'workload':workload,'workload_sha256':fingerprint(workload)}
    write_json(output,result)
    return {'queries':len(workload),'databases':len({r['db_id'] for r in workload})}


class Observer:
    """One extra read-only connection, excluded from the tested connection limit."""
    def __init__(self, environment, database, application):
        self.application=application
        self.lock=threading.Lock()
        self.connection=psycopg.connect(host=environment['PG_HOST'],port=int(environment['PG_PORT']),
            user=environment['PG_USER'],password=environment['PG_PASSWORD'],dbname=database,
            application_name=application+'_watch',connect_timeout=5,autocommit=True,
            options='-c default_transaction_read_only=on -c statement_timeout=3000')
        try:
            pairs=self.connection.execute("SELECT name,setting FROM pg_settings WHERE name IN "
                "('max_connections','superuser_reserved_connections','reserved_connections','work_mem',"
                "'shared_buffers','max_parallel_workers','max_parallel_workers_per_gather')").fetchall()
            self.settings={k:int(v) for k,v in pairs}
            self.settings['server_version']=self.connection.execute('SHOW server_version').fetchone()[0]
            self.settings['role_limit']=self.connection.execute('SELECT rolconnlimit FROM pg_roles WHERE rolname=current_user').fetchone()[0]
        except BaseException:
            self.connection.close()
            raise

    def snapshot(self):
        with self.lock:
            row=self.connection.execute("SELECT COUNT(*) FILTER (WHERE application_name=%s),"
                "COUNT(*) FILTER (WHERE application_name=%s AND state='active'),"
                "COUNT(*) FILTER (WHERE application_name<>%s AND backend_type='client backend') "
                "FROM pg_stat_activity",(self.application,self.application,self.application)).fetchone()
        return dict(at=utc(),probe_connections=row[0],probe_active=row[1],other_clients=row[2])

    def close(self):
        self.connection.close()


def run_batch(run_dir, bundle, environment, concurrency, count):
    from scripts.baseline_adapters.deepeye import postgres_execution
    from scripts.deepeye_bird_interact_run import code_source_hashes
    if type(concurrency) is not int or type(count) is not int or not 1<=concurrency<=50 or not concurrency<=count<=200:
        raise ValueError('probe requires 1..50 concurrent operations and C..200 queries')
    work=bundle['workload']
    if not work or any(not read_query(r['sql']) for r in work):
        raise ValueError('workload must contain only supported read queries')
    run_dir=Path(run_dir)
    application='rc_pg_'+hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:16]
    manifest={'kind':'independent_postgres_capacity','created_at':utc(),'source':bundle['source'],
              'workload_sha256':fingerprint(work),'concurrency':concurrency,'max_queries':count,
              'sql_timeout_seconds':30,'max_admission_seconds':120,'retries':0,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'production_hashes':code_source_hashes(),'application_name':application,
              'semantics':'unchanged adapter; new connection per query; first wave waits until C connections ready; one additional monitor connection; no LLM; no gold scoring'}
    store=RunStore.create(run_dir,manifest)
    observer=None; monitor=None; stop=threading.Event(); monitor_stop=threading.Event()
    lock=threading.Lock(); local=threading.local(); snapshots=[]; handlers={}; monitor_errors=[]
    state={'connected':0,'peak_connected':0,'executing':0,'peak_executing':0,'ready_connections':0}
    attempt=store.begin_attempt('pg_capacity','probe',fingerprint(manifest))
    def record(kind,row):
        try:
            store.append_event(attempt,kind,row)
        except BaseException:
            stop.set()
            raise
    def observe(kind):
        snap=observer.snapshot()
        snapshots.append(snap)
        record(kind,snap)
        return snap
    def first_ready():
        snap=observe('first_wave_ready')
        state['ready_connections']=snap['probe_connections']
        if snap['probe_connections']!=concurrency:
            stop.set()
            raise RuntimeError('server did not confirm all first-wave connections')
    barrier=threading.Barrier(concurrency,action=first_ready,timeout=15)
    connector=psycopg.connect
    def connect(**kwargs):
        row=local.row
        began=time.perf_counter()
        try:
            connection=connector(**{**kwargs,'application_name':application})
        except BaseException:
            if row['request_no']<concurrency:
                stop.set(); barrier.abort()
            raise
        row['connect_seconds']=time.perf_counter()-began
        row['backend_pid']=connection.info.backend_pid
        row['_connected']=True
        with lock:
            state['connected']+=1
            state['peak_connected']=max(state['peak_connected'],state['connected'])
        try:
            if row['request_no']<concurrency:
                began=time.perf_counter()
                barrier.wait()
                row['barrier_wait_seconds']=time.perf_counter()-began
            with lock:
                state['executing']+=1
                state['peak_executing']=max(state['peak_executing'],state['executing'])
            row['_executing']=True
            row['query_ready_at']=utc()
            return connection
        except BaseException:
            stop.set(); barrier.abort()
            connection.close()
            raise
    def execute(index):
        task=work[index%len(work)]
        row={'request_no':index,'workload_index':index%len(work),'db_id':task['db_id'],
             'source_event_id':task['source_event_id'],'sql_sha256':fingerprint(task['sql']),
             'started_at':utc(),'barrier_wait_seconds':0.0}
        local.row=row
        began=time.perf_counter()
        try:
            result=postgres_execution.execute_postgres_sql(SimpleNamespace(database_id=task['db_id']),task['sql'],timeout=30)
            row.update(result_type=result.result_type,success=result.result_type in {'success','empty_result','all_null_result'},
                       result_cols=result.result_cols,row_count=len(result.result_rows or []),error_message=result.error_message)
        finally:
            row['finished_at']=utc()
            row['operation_seconds']=max(0,time.perf_counter()-began-row['barrier_wait_seconds'])
            with lock:
                state['connected']-=bool(row.pop('_connected',False))
                state['executing']-=bool(row.pop('_executing',False))
            local.row=None
        return row
    def monitoring():
        try:
            while not monitor_stop.wait(.05):
                observe('server_sample')
        except Exception as exc:
            stop.set(); barrier.abort()
            error={'type':type(exc).__name__}
            monitor_errors.append(error)
            record('monitor_error',error)
    try:
        write_json(run_dir/'manifest.json',manifest)
        write_json(run_dir/'workload.json',bundle)
        observer=Observer(environment,work[0]['db_id'],application)
        before=observe('server_before')
        settings=observer.settings
        free=settings['max_connections']-settings['superuser_reserved_connections']-settings.get('reserved_connections',0)-before['other_clients']
        if free<concurrency+5 or (0<=settings.get('role_limit',-1)<concurrency+1):
            raise RuntimeError('insufficient connection headroom for requested probe')
        record('server_settings',settings)
        record('resources_before',resources())
        for sig in (signal.SIGINT,signal.SIGTERM):
            if threading.current_thread() is threading.main_thread():
                def stopped(number,frame):
                    stop.set(); barrier.abort()
                handlers[sig]=signal.signal(sig,stopped)
        monitor=threading.Thread(target=monitoring,name='pg-probe-monitor',daemon=True)
        monitor.start()
        began=time.perf_counter()
        proxy=SimpleNamespace(connect=connect,Error=psycopg.Error,errors=psycopg.errors)
        with patch.dict(os.environ,{k:v for k,v in environment.items() if k.startswith('PG_')}), \
             patch.object(postgres_execution,'psycopg',proxy):
            rows=dispatch(concurrency,count,execute,record,stop=stop)
        elapsed=time.perf_counter()-began
        monitor_stop.set(); monitor.join()
        after=observe('server_after')
        summary={'concurrency':concurrency,'requests':len(rows),'planned_queries':count,
                 'success':sum(r['success'] for r in rows),'errors':dict(Counter(r['result_type'] for r in rows if not r['success'])),
                 'elapsed_seconds':elapsed,'completed_per_second':len(rows)/elapsed,
                 'operation_seconds_excluding_barrier':distribution([r['operation_seconds'] for r in rows]),
                 'connect_seconds':distribution([r['connect_seconds'] for r in rows if 'connect_seconds' in r]),
                 'barrier_wait_seconds':distribution([r['barrier_wait_seconds'] for r in rows]),
                 'ready_connections':state['ready_connections'],'peak_client_connections':state['peak_connected'],
                 'peak_client_dispatched_operations':state['peak_executing'],
                 'peak_server_connections':max(s['probe_connections'] for s in snapshots),
                 'peak_server_active':max(s['probe_active'] for s in snapshots),
                 'client_connections_at_finish':state['connected'],'server_connections_at_finish':after['probe_connections'],
                 'server_settings':settings,'monitor_samples':len(snapshots),'monitor_errors':monitor_errors,
                 'resources_after':resources()}
        status='succeeded' if len(rows)==count and all(r['success'] for r in rows) and not after['probe_connections'] and not monitor_errors else 'failed'
        summary['status']=status
        store.finish_attempt(attempt,status,summary)
        summary['integrity']=store.verify()
        write_json(run_dir/'summary.json',summary)
        return summary
    finally:
        stop.set(); monitor_stop.set(); barrier.abort()
        if monitor is not None:
            monitor.join()
        if observer is not None:
            observer.close()
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--workload',type=Path)
    parser.add_argument('--env-file',type=Path,default=Path('config/.env'))
    parser.add_argument('--concurrency',type=int,default=50)
    parser.add_argument('--queries',type=int,default=200)
    parser.add_argument('--authorize-postgres',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.source is None:
            parser.error('prepare requires --source')
        result=prepare(args.source,args.output)
    else:
        if not args.authorize_postgres or args.workload is None:
            parser.error('run requires --authorize-postgres and --workload')
        from scripts.deepeye_bird_interact_smoke import read_environment
        result=run_batch(args.output,json.loads(args.workload.read_text()),read_environment(args.env_file),args.concurrency,args.queries)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
