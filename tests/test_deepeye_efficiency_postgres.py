"""Offline checks for the independent PostgreSQL capacity probe."""
import concurrent.futures as cf
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class PostgresCapacityTests(unittest.TestCase):
    def test_only_supported_read_queries_are_selected(self):
        from scripts.efficiency_probe.postgres import read_query
        for sql in ('SELECT a FROM t', 'WITH x AS (SELECT 1) SELECT * FROM x',
                    "SELECT COUNT(*), SUM(a), jsonb_extract_path_text(b, 'x') FROM t"):
            self.assertTrue(read_query(sql), sql)
        for sql in ('DELETE FROM t', 'SELECT 1; SELECT 2', 'SELECT 1 INTO t',
                    'WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x',
                    'SELECT pg_terminate_backend(1)', 'SELECT custom_function()',
                    'SELECT * FROM t FOR UPDATE'):
            self.assertFalse(read_query(sql), sql)

    def test_workload_is_deduplicated_and_database_balanced(self):
        from scripts.efficiency_probe.postgres import select_workload
        def row(db, sql, duration, event):
            return dict(db_id=db, sql=sql, historical_seconds=duration,
                        source_event_id=event, historical_rows=1, item_key=db+'/q')
        rows=[row('a','SELECT 1',.1,1),row('a','SELECT 1',.1,2),
              row('a','SELECT 2',.2,3),row('a','SELECT 3',1.,4),
              row('b','SELECT 4',.1,5),row('b','SELECT 5',.3,6),
              row('b','DELETE FROM t',2.,7)]
        chosen=select_workload(rows)
        self.assertEqual([r['source_event_id'] for r in chosen], [3,6,4,5])
        self.assertEqual(len({(r['db_id'],r['sql']) for r in chosen}),4)

    def test_executor_limit_exact_budget_and_terminal_pairing(self):
        from scripts.efficiency_probe.postgres import dispatch
        barrier=threading.Barrier(3, timeout=2)
        lock=threading.Lock(); active=0; peak=0; calls=[]; records=[]
        def execute(index):
            nonlocal active,peak
            with lock:
                active+=1; peak=max(peak,active); calls.append(index)
            if index<3:
                barrier.wait()
            time.sleep(.005)
            with lock:
                active-=1
            return {'request_no':index,'success':True}
        rows=dispatch(3,8,execute,lambda kind,row:records.append((kind,row)))
        self.assertEqual(sorted(calls),list(range(8)))
        self.assertEqual(peak,3)
        self.assertEqual(active,0)
        self.assertEqual(len(rows),8)
        self.assertEqual(sorted(r['request_no'] for k,r in records if k=='query_start'),list(range(8)))
        self.assertEqual(sorted(r['request_no'] for k,r in records if k=='query_result'),list(range(8)))

    def test_errors_stop_new_work_without_retry_and_drain_sent(self):
        from scripts.efficiency_probe.postgres import dispatch
        calls=[]
        def execute(index):
            calls.append(index)
            return {'request_no':index,'success':False}
        rows=dispatch(1,20,execute,lambda kind,row:None)
        self.assertEqual(calls,list(range(5)))
        self.assertEqual(len(rows),5)

    def test_recording_error_is_not_swallowed(self):
        from scripts.efficiency_probe.postgres import dispatch
        def record(kind,row):
            if kind=='query_result':
                raise OSError('disk failure')
        with self.assertRaisesRegex(OSError,'disk failure'):
            dispatch(2,8,lambda i:dict(request_no=i,success=True),record)

    def test_invalid_capacity_is_rejected_before_queries(self):
        from scripts.efficiency_probe.postgres import dispatch
        calls=[]
        for concurrency,count in ((0,10),(51,100),(10,201),(10,0),(50,10)):
            with self.assertRaises(ValueError):
                dispatch(concurrency,count,lambda i:calls.append(i),lambda k,r:None)
        self.assertEqual(calls,[])

    def test_adapter_replay_preserves_sql_and_releases_every_connection(self):
        self._adapter_replay()

    def test_first_wave_connect_failure_stops_before_any_sql_or_replacement(self):
        self._adapter_replay(fail_first=True)

    def test_monitor_failure_cannot_report_success_when_queries_all_succeed(self):
        self._adapter_replay(monitor_fail=True)

    def _adapter_replay(self, fail_first=False, monitor_fail=False):
        from scripts.efficiency_probe.postgres import run_batch
        state={'open':0,'next_pid':0,'sql':[],'connect_calls':0}
        lock=threading.Lock()
        class Connection:
            def __init__(self, **kwargs):
                with lock:
                    state['connect_calls']+=1
                    if fail_first and state['connect_calls']==1:
                        import psycopg
                        raise psycopg.OperationalError('fixture connection failure')
                self.closed=False; self.read_only=False
                assert 'default_transaction_read_only=on' in kwargs['options']
                assert 'statement_timeout=30000' in kwargs['options']
                with lock:
                    state['open']+=1; state['next_pid']+=1
                    self.info=SimpleNamespace(backend_pid=state['next_pid'])
            def cursor(self):
                parent=self
                class Cursor:
                    description=[('value',)]
                    def execute(self, sql, prepare):
                        assert parent.read_only and prepare
                        with lock:
                            state['sql'].append(sql)
                        time.sleep(.15 if monitor_fail else .005)
                    def fetchall(self): return [(1,)]
                    def close(self): pass
                return Cursor()
            def rollback(self): pass
            def close(self):
                if not self.closed:
                    with lock: state['open']-=1
                    self.closed=True
        class Observer:
            def __init__(self,*args):
                self.settings={'max_connections':80,'superuser_reserved_connections':3,'reserved_connections':0}
            def snapshot(self):
                if monitor_fail and threading.current_thread().name=='pg-probe-monitor':
                    raise RuntimeError('fixture monitor failure')
                with lock: n=state['open']
                return dict(at='fixture',probe_connections=n,probe_active=0,other_clients=0)
            def close(self): pass
        bundle={'workload':[dict(db_id='fixture',sql='SELECT 1',source_event_id=1,
                                historical_rows=1,historical_seconds=.1)],'source':'fixture'}
        with tempfile.TemporaryDirectory() as td, patch('scripts.efficiency_probe.postgres.Observer',Observer,create=True), \
             patch('psycopg.connect',Connection):
            summary=run_batch(Path(td)/'run',bundle,{},2,2 if monitor_fail else 7)
            if monitor_fail:
                self.assertEqual(summary['success'],2)
                self.assertEqual(summary['status'],'failed')
                self.assertTrue(summary['monitor_errors'])
                self.assertEqual(state['open'],0)
                self.assertTrue(summary['integrity']['ok'])
                return
            if fail_first:
                self.assertLessEqual(summary['requests'],2)
                self.assertEqual(summary['ready_connections'],0)
                self.assertEqual(summary['status'],'failed')
                self.assertEqual(state['sql'],[])
                self.assertEqual(state['open'],0)
                self.assertTrue(summary['integrity']['ok'])
                return
            self.assertEqual(summary.get('requests'),7)
            self.assertEqual(summary['success'],7)
            self.assertEqual(summary['ready_connections'],2)
            self.assertEqual(state['open'],0)
            self.assertEqual(state['sql'],['SELECT 1']*7)
            self.assertTrue(summary['integrity']['ok'])
            import sqlite3
            with sqlite3.connect(Path(td)/'run/run.sqlite3') as db:
                counts=dict(db.execute("SELECT kind,COUNT(*) FROM events WHERE kind IN ('query_start','query_result') GROUP BY kind"))
            self.assertEqual(counts,{'query_start':7,'query_result':7})


if __name__=='__main__':
    unittest.main()
