import asyncio
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from scripts.baseline_adapters.din_sql.inputs import PreparedInputs, TaskKey, OUTPUT_NODES
from scripts.baseline_adapters.din_sql.records import DinRecords
from scripts.rc_evaluation.din_sql.campaign import schedule, status, resolve_scope
from din_sql_fixtures import make_task, terminal


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    def test_resume_preserves_targeted_rerun_scope(self):
        scope=[{'group':'bird_dev','question_id':'0'}]
        self.assertEqual(resolve_scope('resume',{'targets':scope},None),[TaskKey('bird_dev','0')])
        self.assertIsNone(resolve_scope('resume',{'targets':None},None))
        self.assertIsNone(resolve_scope('resume',{'targets':scope},None,all_pending=True))
        with self.assertRaises(ValueError):
            resolve_scope('resume',None,None)
    async def test_resume_interrupted_rerun_recovers_parent_references_without_requests(self):
        manifest={'format':'din-sql-v1','batch_id':'fixture','groups':{'bird_dev':{'ids':['0']}}}
        task=make_task()
        prepared=PreparedInputs({task.key:task},{},{},{},{},{})
        async def execute(node,*args):
            return terminal(node)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'b'
            with DinRecords(root,manifest) as records:
                await schedule(prepared,records,execute)
                old=records.current(task.key)
                new=records.begin(task.key,parent_version=old)
                # Simulate crash between durable begin and first reusable copy.
            async def forbidden(*args):
                raise AssertionError('Unnecessary model request')
            with DinRecords(root,manifest) as records:
                await schedule(prepared,records,forbidden)
                self.assertEqual(records.current(task.key),new)

    async def test_crash_before_question_start_recovers_latest_finished_parent(self):
        manifest={'format':'din-sql-v1','batch_id':'fixture','groups':{'bird_dev':{'ids':['0']}}}
        task=make_task()
        prepared=PreparedInputs({task.key:task},{},{},{},{},{})
        async def execute(node,*args):
            return terminal(node)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'b'
            with DinRecords(root,manifest) as records:
                await schedule(prepared,records,execute)
                old=records.current(task.key)
                with patch.object(records,'append',side_effect=OSError('crash')):
                    with self.assertRaises(OSError):
                        records.begin(task.key,parent_version=old)
            async def forbidden(*args):
                raise AssertionError('Unnecessary model request')
            with DinRecords(root,manifest) as records:
                await schedule(prepared,records,forbidden)
                self.assertNotEqual(records.current(task.key),old)

    async def test_next_group_starts_at_eighty_percent_without_waiting_for_tail(self):
        manifest = {'format':'din-sql-v1','batch_id':'fixture','groups':
                    {'bird_dev':{'ids':[str(i) for i in range(5)]},'bird_interact_full':{'ids':['0']}}}
        tasks = [make_task(group='bird_dev',question_id=str(i)) for i in range(5)]
        tasks.append(make_task(group='bird_interact_full'))
        prepared = PreparedInputs({t.key:t for t in tasks},{},{},{},{},{})
        tail, next_started = asyncio.Event(),asyncio.Event()
        async def execute(node, task, parents, version):
            if task.key==TaskKey('bird_dev','4'):
                await tail.wait()
            if task.key.group=='bird_interact_full':
                next_started.set()
            return terminal(node)
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',manifest) as records:
                pending = asyncio.create_task(schedule(prepared,records,execute))
                await asyncio.wait_for(next_started.wait(),3)
                self.assertFalse(pending.done())
                tail.set()
                result = await pending
                self.assertEqual(result['started_groups'],['bird_dev','bird_interact_full'])
                self.assertEqual(result['terminal']['bird_dev'],dict.fromkeys(OUTPUT_NODES,5))
                self.assertEqual(len(records.current_rows()),6)

    async def test_cancel_retains_completed_nodes_and_resume_only_missing(self):
        manifest = {'format':'din-sql-v1','batch_id':'fixture','groups':{'bird_dev':{'ids':['0']}}}
        task = make_task()
        prepared = PreparedInputs({task.key:task},{},{},{},{},{})
        entered = asyncio.Event()
        async def blocked(node,task,parents,version):
            if node=='generation_rc3':
                entered.set()
                await asyncio.Event().wait()
            return terminal(node)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'b'
            with DinRecords(root,manifest) as records:
                pending = asyncio.create_task(schedule(prepared,records,blocked))
                await asyncio.wait_for(entered.wait(),3)
                # Wait for the independent baseline Revision to finish, no sleeps.
                while not records.node(records.unfinished(task.key),'revision_base'):
                    await asyncio.sleep(0)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertIsNone(records.current(task.key))
                before = set(records.view(records.unfinished(task.key))['nodes'])
            calls=[]
            async def resume(node,*args):
                calls.append(node)
                return terminal(node)
            with DinRecords(root,manifest) as records:
                await schedule(prepared,records,resume)
                self.assertTrue(set(calls).isdisjoint(before))
                self.assertEqual(len(records.current_rows()),1)
