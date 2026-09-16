import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.baseline_adapters.din_sql.transport import DinRequester, retry_delay, parse_retry_after
from scripts.baseline_adapters.din_sql.inputs import DinSettings, TaskKey
from scripts.baseline_adapters.din_sql.records import DinRecords
from din_sql_fixtures import minimal_manifest


class FakeDispatcher:
    def __init__(self, failures=0, *, started=True):
        self.failures, self.calls, self.started = failures,0,started
        self.stopped = False

    async def call_chat(self, client, *, identity, sdk_kwargs, on_started, on_finished):
        if not self.started:
            raise asyncio.CancelledError()
        on_started()
        self.calls += 1
        on_finished(identity,{'request_wall_seconds':0.01})
        if self.calls <= self.failures:
            raise TimeoutError('fixture')
        return {'choices':[{'index':0,'message':{'content':'SELECT 1'},'finish_reason':'stop'}],
                'model':'qwen3.8-2.4t-a95b','usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}

    def stop(self, **kwargs):
        self.stopped = True


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_choices_are_protocol_retries(self):
        class Malformed(FakeDispatcher):
            async def call_chat(self,client,**kwargs):
                result=await super().call_chat(client,**kwargs)
                if self.calls==1:
                    result['choices']=[None]
                elif self.calls==2:
                    result['choices'][0]['message']='not an object'
                return result
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                v=records.begin(TaskKey('bird_dev','0'))
                dispatcher=Malformed()
                with patch('scripts.baseline_adapters.din_sql.transport.retry_delay',return_value=0):
                    result=await DinRequester(dispatcher,None,DinSettings(),records).request(
                        v,'linking',[{'role':'user','content':'SQL'}],'bird')
                self.assertEqual(result['status'],'succeeded')
                self.assertEqual(dispatcher.calls,3)

    async def test_crashed_fifth_send_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                v=records.begin(TaskKey('bird_dev','0'))
                for i in range(5):
                    records.append(v,'request_attempt',{'node':'linking','attempt_no':i+1})
                dispatcher=FakeDispatcher()
                result=await DinRequester(dispatcher,None,DinSettings(),records).request(
                    v,'linking',[{'role':'user','content':'SQL'}],'bird')
                self.assertEqual(result['status'],'failed')
                self.assertEqual(dispatcher.calls,0)

    async def test_moderation_and_timeout_share_budget(self):
        from tests.test_dail_sql_transport import inspection_error
        class Mixed(FakeDispatcher):
            async def call_chat(self, client, **kwargs):
                kwargs['on_started']()
                self.calls+=1
                if self.calls%2:
                    raise inspection_error('Output data may contain inappropriate content.')
                raise TimeoutError()
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                v=records.begin(TaskKey('bird_dev','0'))
                dispatcher=Mixed()
                with patch('scripts.baseline_adapters.din_sql.transport.retry_delay',return_value=0):
                    result=await DinRequester(dispatcher,None,DinSettings(),records).request(
                        v,'linking',[{'role':'user','content':'SQL'}],'bird')
                self.assertEqual(dispatcher.calls,5)
                self.assertEqual(result['error']['category'],'retry_budget_exhausted')

    async def test_success_on_fifth_and_exhaustion_without_sixth(self):
        for failures,want in [(4,'succeeded'),(5,'failed')]:
            with tempfile.TemporaryDirectory() as tmp:
                with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                    v = records.begin(TaskKey('bird_dev','0'))
                    dispatcher = FakeDispatcher(failures)
                    requester = DinRequester(dispatcher,None,DinSettings(),records)
                    with patch('scripts.baseline_adapters.din_sql.transport.retry_delay',return_value=0):
                        result = await requester.request(v,'linking',[{'role':'user','content':'SQL'}],'bird')
                    self.assertEqual(result['status'],want)
                    self.assertEqual(dispatcher.calls,5)
                    again = await requester.request(v,'linking',[{'role':'user','content':'SQL'}],'bird')
                    self.assertEqual(again['status'],want)
                    self.assertEqual(dispatcher.calls,5)

    async def test_cancel_before_send_does_not_spend_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                v = records.begin(TaskKey('bird_dev','0'))
                requester = DinRequester(FakeDispatcher(started=False),None,DinSettings(),records)
                with self.assertRaises(asyncio.CancelledError):
                    await requester.request(v,'linking',[{'role':'user','content':'SQL'}],'bird')
                self.assertEqual(records.view(v)['attempts'].get('linking',[]),[])

    async def test_retry_delay_bounded(self):
        self.assertEqual([retry_delay(i) for i in range(2,6)],[1,2,4,8])
        self.assertEqual(retry_delay(2,999),60)
        self.assertEqual(parse_retry_after('Thu, 01 Jan 1970 00:00:30 GMT',now=0),30)
        self.assertIsNone(parse_retry_after('unknown',now=0))
