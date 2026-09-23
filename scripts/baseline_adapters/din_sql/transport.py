"""One single-choice request per node; durable retries share one five-send budget."""
import asyncio
import math
import uuid
import time
from email.utils import parsedate_to_datetime

from scripts.baseline_adapters.shared.transport import RecordingError, RequestStopped
from scripts.baseline_adapters.dail_sql.transport import classify_error, ChoiceProtocolError
from .prompts import request_kwargs
from .inputs import digest, validate_settings


def retry_delay(attempt_no, retry_after=None):
    delay = 2**max(0,attempt_no-2)
    if retry_after is not None and math.isfinite(retry_after):
        delay = max(delay,retry_after)
    return min(60,delay)


def parse_retry_after(value, *, now=None):
    try:
        result = float(value)
    except (TypeError,ValueError):
        try:
            result = parsedate_to_datetime(value).timestamp()-(time.time() if now is None else now)
        except (TypeError,ValueError,OverflowError):
            return None
    return max(0,result) if math.isfinite(result) else None


class DinRequester:
    def __init__(self, dispatcher, client, settings, records):
        self.dispatcher, self.client = dispatcher,client
        self.settings, self.records = validate_settings(settings),records
        if hasattr(dispatcher,'limits') and dispatcher.limits.request_timeout != settings.request_timeout_seconds:
            raise ValueError('DIN dispatcher must use the configured total request deadline')

    def _append(self, version, kind, payload):
        try:
            return self.records.append(version,kind,payload)
        except Exception as exc:
            self.dispatcher.stop(cancel_active=True)
            raise RecordingError('DIN durable recording failed') from exc

    async def _success(self, outcome):
        payload = await asyncio.to_thread(self.records.read_ref,outcome['response_ref'])
        body = payload['body']
        return {**outcome,'content':body['choices'][0]['message']['content']}

    async def request(self, version, node, messages, family):
        if not messages:
            raise ValueError('DIN request requires messages')
        kwargs = request_kwargs(node,family,messages,self.settings)
        view = self.records.request_history(version)
        old_input = view['inputs'].get(node)
        fingerprint = digest(kwargs)
        if old_input and old_input['input_fingerprint'] != fingerprint:
            raise ValueError('Existing node request differs; create a new batch')
        if not old_input:
            await asyncio.to_thread(self._append,version,'node_input',
                {'node':node,'kwargs':kwargs,'input_fingerprint':fingerprint})
        outcomes = view['outcomes'].get(node,[])
        for outcome in outcomes:
            if outcome['status']=='succeeded':
                return await self._success(outcome)
        attempts = view['attempts'].get(node,[])
        retry_after = None
        for attempt_no in range(len(attempts)+1,self.settings.max_attempts+1):
            if attempt_no > 1:
                await asyncio.sleep(retry_delay(attempt_no,retry_after))
            key = self.records.key(version)
            identity = {'batch_id':self.records.manifest['batch_id'],'group':key.group,'question_id':key.question_id,
                        'round_execution_id':version+':'+node,'sample_position':0,'request_id':uuid.uuid4().hex,
                        'attempt_no':attempt_no}
            await asyncio.to_thread(self._append,version,'attempt_queued',{'node':node,**identity})
            sent = False
            def started():
                nonlocal sent
                self._append(version,'request_attempt',{'node':node,**identity})
                sent = True
            def finished(actual_identity, telemetry):
                self._append(version,'request_dispatch',{'node':node,**actual_identity,'telemetry':telemetry})
            response_ref = None
            usage = None
            try:
                response = await self.dispatcher.call_chat(self.client,identity=identity,sdk_kwargs=kwargs,
                                                          on_started=started,on_finished=finished)
                body = response.model_dump(mode='json') if hasattr(response,'model_dump') else response
                response_ref = await asyncio.to_thread(self._append,version,'request_result',
                                                      {'node':node,**identity,'body':body})
                choices = body.get('choices') if isinstance(body,dict) else None
                usage = body.get('usage') if isinstance(body,dict) else None
                if (not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict)
                        or not isinstance(choices[0].get('message'),dict)):
                    raise ChoiceProtocolError('Expected exactly one choice')
                content = (choices[0].get('message') or {}).get('content')
                if choices[0].get('finish_reason') != 'stop' or not isinstance(content,str) or not content.strip():
                    raise ChoiceProtocolError('Empty or unfinished response')
                outcome = {'node':node,'status':'succeeded','response_ref':response_ref,'usage':usage,
                           'response_model':body.get('model'),'request_id':identity['request_id'],
                           'attempt_no':attempt_no,'error':None}
                await asyncio.to_thread(self._append,version,'request_outcome',outcome)
                return {**outcome,'content':content}
            except (asyncio.CancelledError,RequestStopped):
                raise
            except Exception as exc:
                error = classify_error(exc)
                if not sent:
                    error.update(retryable=False,pause=True)
                # Do not serialize provider exception strings containing URLs/keys.
                outcome = {'node':node,'status':'failed','response_ref':response_ref,'usage':usage,
                           'request_id':identity['request_id'],'attempt_no':attempt_no,'error':error}
                await asyncio.to_thread(self._append,version,'request_outcome',outcome)
                if error['pause']:
                    self.dispatcher.stop(cancel_active=True)
                    raise RuntimeError(f'DIN paused: {error["category"]}/{error["type"]}') from exc
                if not error['retryable']:
                    return outcome
                try:
                    retry_after = parse_retry_after(exc.response.headers.get('retry-after'))
                except AttributeError:
                    retry_after = None
        return {'node':node,'status':'failed','content':None,'response_ref':None,'usage':None,
                'error':{'category':'retry_budget_exhausted','attempts':len(view['attempts'].get(node,[]))}}
