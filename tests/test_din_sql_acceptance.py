import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits
from scripts.baseline_adapters.din_sql.inputs import DinSettings, TaskKey
from scripts.baseline_adapters.din_sql.records import DinRecords
from scripts.baseline_adapters.din_sql.transport import DinRequester
from din_sql_fixtures import minimal_manifest


class HttpAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_success_deadline_and_request_fields(self):
        observed=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                observed.append(body)
                if body['messages'][0]['content']=='slow':
                    time.sleep(.3)
                response={'id':'local','object':'chat.completion','created':0,'model':body['model'],
                          'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'SELECT 1'}}],
                          'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}
                raw=json.dumps(response).encode()
                try:
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError,ConnectionResetError):
                    pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with DinRecords(Path(tmp)/'b',minimal_manifest()) as records:
                    settings=replace(DinSettings(),request_timeout_seconds=.1,max_attempts=1)
                    dispatcher=RequestDispatcher(RequestLimits(request_limit=2,http_connections=2,start_rate=100,
                                                                request_timeout=.1))
                    try:
                        client=dispatcher.make_client(api_key='offline-fixture',base_url=f'http://127.0.0.1:{server.server_port}/v1')
                        requester=DinRequester(dispatcher,client,settings,records)
                        v=records.begin(TaskKey('bird_dev','0'))
                        result=await requester.request(v,'generation_base',[{'role':'user','content':'fast'}],'bird')
                        self.assertEqual(result['status'],'succeeded')
                        self.assertEqual(result['usage']['total_tokens'],5)
                        v2=records.begin(TaskKey('bird_dev','1'))
                        result=await requester.request(v2,'generation_base',[{'role':'user','content':'slow'}],'bird')
                        self.assertEqual(result['status'],'failed')
                        history=records.view(v2)
                        self.assertEqual(len(history['attempts']['generation_base']),1)
                        self.assertEqual(history['outcomes']['generation_base'][0]['error']['category'],'timeout')
                        self.assertEqual(dispatcher.snapshot()['in_flight'],0)
                    finally:
                        await asyncio.to_thread(dispatcher.close)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join()
        self.assertEqual(len(observed),2)
        for request in observed:
            self.assertEqual(request['n'],1)
            self.assertFalse(request['stream'])
            self.assertEqual(request['temperature'],0)
            self.assertNotIn('max_tokens',request)
            self.assertNotIn('enable_thinking',request)
