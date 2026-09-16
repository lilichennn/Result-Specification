import contextlib
import io
import json
import os
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.din_sql.inputs import PreparedInputs, DinSettings, NODES
from scripts.baseline_adapters.din_sql.records import DinRecords, hydrate_batch, load_prepared
from scripts.rc_evaluation.din_sql.cli import main
from din_sql_fixtures import minimal_manifest, make_task, terminal


class CliTests(unittest.TestCase):
    def test_actual_run_entrypoint_scoped_rerun_then_explicit_full_resume(self):
        from scripts.rc_evaluation.din_sql.campaign import run_batch
        class OfflineDispatcher:
            def __init__(self,*args): pass
            def make_client(self,**kwargs): return None
            def snapshot(self): return {}
            def stop(self,**kwargs): pass
            def close(self): pass
        first,second=make_task(question_id='0'),make_task(question_id='1')
        tasks={t.key:t for t in (first,second)}
        manifest={**minimal_manifest(),'settings':asdict(DinSettings())}
        prepared=PreparedInputs(tasks,{}, {},{}, {},{})
        calls=[]
        async def execute(node,task,*args):
            calls.append((task.key,node))
            return terminal(node)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'b'
            with DinRecords(root,manifest) as records:
                hydrate_batch(root,prepared,records)
                v=records.unfinished(first.key)
                for node in NODES:
                    records.save_node(v,node,terminal(node))
                records.seal(v)
            with patch.dict(os.environ,{'DASH_MODELS':DinSettings().model,'DASH_API_KEY':'fixture','DASH_BASE_URL':'http://offline.invalid'}),\
                 patch('scripts.rc_evaluation.din_sql.campaign.RequestDispatcher',OfflineDispatcher),\
                 patch('scripts.rc_evaluation.din_sql.campaign.NodeExecutor',return_value=execute):
                env=Path(tmp)/'absent.env'
                run_batch(root,env,operation='rerun',targets=[first.key])
                run_batch(root,env,operation='resume')
                self.assertEqual(calls,[])
                run_batch(root,env,operation='resume',all_pending=True)
                self.assertEqual({k for k,n in calls},{second.key})
            with DinRecords(root,manifest,read_only=True) as records:
                self.assertEqual(len(records.current_rows()),2)

    def test_prepare_hydration_is_idempotent_and_missing_results_remain_unfinished(self):
        task=make_task()
        prepared=PreparedInputs({task.key:task},{'scores':{}},{},{},
                                {'accepted':{'bird_dev/0':{n:terminal(n) for n in
                                  ('linking','decomposition','generation_base')}}},{})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'b'
            with DinRecords(root,minimal_manifest()) as records:
                first=hydrate_batch(root,prepared,records)
                v=records.unfinished(task.key)
                events_before=len(records.stores['bird_dev'].events(v))
                second=hydrate_batch(root,prepared,records)
                self.assertEqual(events_before,len(records.stores['bird_dev'].events(v)))
                self.assertIsNone(records.current(task.key))
                self.assertEqual(second['counts']['bird_dev']['imported'],0)
                self.assertEqual(load_prepared(root).tasks[task.key],task)

    def test_status_has_no_llm_or_schema_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'b'
            with DinRecords(root,minimal_manifest()):
                pass
            with patch('scripts.baseline_adapters.shared.transport.RequestDispatcher',side_effect=AssertionError('LLM')):
                output=io.StringIO()
                with contextlib.redirect_stdout(output):
                    main(['status','--batch',str(root)])
                self.assertEqual(json.loads(output.getvalue())['groups']['bird_dev']['total'],2)
