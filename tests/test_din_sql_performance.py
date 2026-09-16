import asyncio
from collections import Counter
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.din_sql.inputs import PreparedInputs, OUTPUT_NODES
from scripts.baseline_adapters.din_sql.records import DinRecords
from scripts.rc_evaluation.din_sql.campaign import schedule
from din_sql_fixtures import make_task, terminal


class PerformanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_does_not_scan_manifest_or_responses_per_node(self):
        sizes = {'spider_dev':1034,'bird_dev':1534,'bird_interact_full':410,
                 'bird_interact_lite':195,'spider_test':2147}
        if os.environ.get('DIN_FULL_SCALE_TEST')!='1':
            sizes=dict.fromkeys(sizes,5)
        manifest={'format':'din-sql-v1','batch_id':'scale','groups':
                  {g:{'ids':[str(i) for i in range(n)]} for g,n in sizes.items()}}
        tasks=[make_task(group=g,question_id=str(i)) for g,n in sizes.items() for i in range(n)]
        prepared=PreparedInputs({t.key:t for t in tasks},{},{},{},{},{})
        calls=Counter()
        async def execute(node,task,parents,version):
            calls[node]+=1
            return terminal(node)
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',manifest) as records:
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for store in records.stores.values():
                        stack.enter_context(patch.object(store,'attempts',side_effect=AssertionError('whole scan')))
                        stack.enter_context(patch.object(store,'event',side_effect=AssertionError('body read')))
                    result=await schedule(prepared,records,execute)
                self.assertEqual(result['started_groups'],list(sizes))
                self.assertEqual(len(records.current_rows()),sum(sizes.values()))
                for group,n in sizes.items():
                    self.assertEqual(result['terminal'][group],dict.fromkeys(OUTPUT_NODES,n))
                self.assertEqual(calls['generation_base'],sum(sizes.values()))
                self.assertEqual(calls['decomposition'],0)
