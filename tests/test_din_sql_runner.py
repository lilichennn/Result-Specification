import asyncio
import tempfile
import unittest
from pathlib import Path
from scripts.baseline_adapters.din_sql.inputs import OUTPUT_NODES
from scripts.baseline_adapters.din_sql.records import DinRecords
from scripts.rc_evaluation.din_sql.runner import dependencies, run_question, copy_reusable
from scripts.rc_evaluation.din_sql.campaign import ready_for_next
from din_sql_fixtures import make_task, minimal_manifest, terminal


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_threshold_and_dependencies(self):
        self.assertEqual(dependencies('revision_rc3','NESTED'),('generation_base',))
        self.assertEqual(dependencies('decomposition','EASY'),())
        counts = dict.fromkeys(OUTPUT_NODES,828)
        self.assertTrue(ready_for_next(1034,counts))
        counts['revision_rc3']=827
        self.assertFalse(ready_for_next(1034,counts))

    async def test_revision_does_not_wait_for_rc_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                task = make_task()
                version = records.begin(task.key)
                revision_started, release = asyncio.Event(),asyncio.Event()
                async def execute(node, task, parents, version):
                    if node=='generation_rc3':
                        await release.wait()
                    if node=='revision_rc3':
                        revision_started.set()
                    return terminal(node)
                pending = asyncio.create_task(run_question(task,version,records,execute))
                await asyncio.wait_for(revision_started.wait(),1)
                self.assertFalse(pending.done())
                release.set()
                await pending
                self.assertEqual(records.current(task.key),version)

    async def test_failed_base_only_blocks_its_revisions_and_rerun_invalidates_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                task = make_task()
                v = records.begin(task.key)
                async def execute(node, task, parents, version):
                    return terminal(node,'failed' if node=='generation_base' else 'succeeded')
                await run_question(task,v,records,execute)
                self.assertEqual(records.node(v,'generation_rc3')['status'],'succeeded')
                self.assertEqual(records.node(v,'revision_rc3')['status'],'dependency_failed')
                new = records.begin(task.key,parent_version=v)
                copy_reusable(task,v,new,records)
                self.assertIsNotNone(records.node(new,'generation_rc3'))
                self.assertIsNone(records.node(new,'revision_base'))
                self.assertIsNone(records.node(new,'revision_rc3'))
