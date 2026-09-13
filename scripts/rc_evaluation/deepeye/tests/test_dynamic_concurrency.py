"""Production RC entrypoint uses the shared runtime, never adaptive slots."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

from . import OfflineTestCase
from .test_cli import ENV, native_args
from .test_runner import experiment_manifest
from tests.test_deepeye_run_inheritance import make_item, manifest, complete_stage, cost
from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, _checkpoint
from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from scripts.deepeye_bird_interact_run import build_effective_config, code_source_hashes
from scripts.rc_evaluation.deepeye import cli
from scripts.rc_evaluation.deepeye.source import snapshot_source


class AsyncClientFixture:
    """Fake only the SDK transport below the real facade/admission/trace."""
    def __init__(self, create):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.closed = False

    async def close(self):
        self.closed = True


def completion(content='<result>SELECT x FROM t</result>'):
    from openai.types.chat import ChatCompletion
    return ChatCompletion(id='offline', created=0, model='offline', object='chat.completion',
        choices=[{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}],
        usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5})


class DynamicConcurrencyTests(OfflineTestCase):
    def test_actual_rc_stop_between_admission_and_submit_returns_paused(self):
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        original = SamplingRuntime.submit_coordinator
        def stop_at_submission(runtime, *args, **kwargs):
            runtime.stop()
            return original(runtime, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, keys='ab') as store, \
                patch.object(SamplingRuntime, 'submit_coordinator', stop_at_submission):
            result = cli.execute_run(store, ENV)
            self.assertEqual((result['succeeded'], result['failed'], result['paused']), (0, 0, 2))
            self.assertEqual(store.attempts(), [])

    def test_actual_rc_authentication_failure_propagates_and_drains_without_new_work(self):
        import httpx
        from openai import AuthenticationError
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        calls = []
        response = httpx.Response(401, request=httpx.Request('POST', 'https://invalid.test/v1'))
        original = AuthenticationError('offline invalid credential', response=response, body=None)
        async def create(**kwargs):
            calls.append(kwargs)
            raise original
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, calls=1, keys='ab') as store:
            client = AsyncClientFixture(create)
            with patch('openai.AsyncOpenAI', return_value=client), \
                 patch.object(LLM, '_create_client', side_effect=AssertionError('sync client forbidden')), \
                 patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)):
                with self.assertRaises(AuthenticationError) as failure:
                    cli.execute_run(store, ENV)
            self.assertIs(failure.exception, original)
            self.assertLessEqual(len(calls), 2)
            self.assertTrue(client.closed)
            self.assertTrue(all(row['status'] == 'interrupted' for row in store.attempts()))
            self.assertFalse(any(t.name.startswith(('deepeye-http', 'deepeye-coordinator', 'deepeye-sample'))
                                 for t in threading.enumerate()))

    def test_actual_rc_manual_stop_preserves_unfinished_sampling_and_same_event(self):
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        runtimes, calls = [], []
        original_init = SamplingRuntime.__init__
        def initialize(runtime, **kwargs):
            original_init(runtime, **kwargs)
            runtimes.append(runtime)
        async def create(**kwargs):
            calls.append(kwargs)
            runtimes[0].stop(cancel_active=False)
            await asyncio.sleep(.01)
            return completion()
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, calls=1, keys='ab') as store:
            client = AsyncClientFixture(create)
            with patch.object(SamplingRuntime, '__init__', initialize), \
                 patch('openai.AsyncOpenAI', return_value=client), \
                 patch.object(LLM, '_create_client', side_effect=AssertionError('sync client forbidden')), \
                 patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)):
                result = cli.execute_run(store, ENV)
            self.assertEqual((result['succeeded'], result['failed'], result['paused']), (0, 0, 2))
            self.assertTrue(runtimes[0].stop_event.is_set())
            self.assertLessEqual(len(calls), 2)
            self.assertTrue(client.closed)
            self.assertEqual(runtimes[0].dispatch.snapshot()['in_flight'], 0)
            self.assertTrue(all(row['status'] == 'interrupted' for row in store.attempts()))
            self.assertTrue(store.verify()['ok'])

    def prepared(self, temporary, *, calls=0, keys='a', stage='sql_generation', condition='rc',
                 downstream=False, coordinator_workers=4):
        root = Path(temporary)
        args = native_args()
        args.coordinator_workers = coordinator_workers
        config = build_effective_config(ENV, args)
        tasks = [('lite', make_item(key)) for key in keys]
        source = manifest(tasks, upgrade=True)
        source['effective_config'] = config
        source['sources']['code'] = code_source_hashes()
        with RunStore.create(root / 'source', source) as store:
            for _, original in tasks:
                item = copy.deepcopy(original)
                identity = fingerprint({'manifest': source, 'input': to_jsonable(item.model_dump(exclude={'gold_sql'}))})
                for current in STAGES:
                    input_hash = fingerprint({'input': identity, 'stage': current})
                    complete_stage(item, current)
                    count = calls if current == stage else 1
                    setattr(item, current + '_llm_cost', cost(count))
                    payload = _checkpoint(item, current)
                    payload['attempt_wall_seconds'] = 1.0
                    attempt = store.begin_attempt(f'lite/{item.instance_id}', current, input_hash)
                    for number in range(count):
                        call = f'{attempt}-{number}'
                        store.append_event(attempt, 'api_request', {'call_id': call})
                        store.append_event(attempt, 'api_response', {'call_id': call, 'response': {'usage': cost(1)}})
                    store.finish_attempt(attempt, 'succeeded', payload)
                    identity = fingerprint({'input': input_hash, 'output': payload})
        data = experiment_manifest(snapshot_source(root / 'source', tasks, stage, continue_downstream=downstream),
                                   condition=condition, downstream=downstream)
        data['target_stage'] = stage
        data['sources'] = {**data['sources'], 'rc_evaluation_code_sha256': cli.production_hash()}
        if condition == 'rc':
            from .test_contracts import _record
            from scripts.rc_evaluation.deepeye.contracts import load_contracts
            records = [_record(row.instance_id, db_id=row.database_id, question=row.question, evidence=row.evidence,
                rc_round2={field: f'Unique contract for {row.instance_id}' for field in (
                    'population', 'row_grain', 'column_role', 'derivation', 'filter_policy', 'meta_review')})
                for _, row in tasks]
            path = root / 'contracts.json'
            path.write_text(json.dumps(records))
            data['contracts'] = load_contracts({'lite': path}, tasks)
        return RunStore.create(root / 'run', data)

    def test_replay_keeps_shared_runtime_limits_without_creating_native_clients(self):
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary) as store:
            with patch.object(cli, 'build_runtime_config', side_effect=AssertionError('replay built native config')), \
                 patch.object(cli, 'bounded_runner_factory', side_effect=AssertionError('replay built model clients')), \
                 patch('openai.AsyncOpenAI', side_effect=AssertionError('replay opened client')):
                result = cli.execute_run(store, ENV)
            self.assertEqual(result['succeeded'], 1)
            self.assertEqual(result['admission']['pipeline']['mode'], 'all_questions')
            self.assertEqual(result['runtime']['requests']['submitted'], 0)
            self.assertEqual(store.attempts()[0]['payload']['execution_origin'], 'reused_no_native_llm_call')
            self.assertTrue(store.verify()['ok'])

    def test_actual_rc_generation_shares_caps_preserves_context_on_retries_and_closes(self):
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from app.llm.sampling import sampling_identity
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID
        from scripts.baseline_adapters.deepeye.run_usage import observed_usage
        from scripts.rc_evaluation.deepeye.injection import render_rc_block
        for coordinators in (1, 2):
            with self.subTest(coordinators=coordinators), tempfile.TemporaryDirectory() as temporary, \
                    self.prepared(temporary, calls=1, keys='ab', coordinator_workers=coordinators) as store:
                clients, seen, active, peak = [], [], 0, 0
                async def create(**kwargs):
                    nonlocal active, peak
                    row = store.attempt(_ATTEMPT_ID.get())
                    self.assertEqual(row['stage'], 'sql_generation')
                    key = row['item_key']
                    block = render_rc_block(store.manifest['contracts'][key])
                    prompt = kwargs['messages'][0]['content']
                    self.assertIn(block, prompt)
                    other = 'lite/b' if key == 'lite/a' else 'lite/a'
                    self.assertNotIn(render_rc_block(store.manifest['contracts'][other]), prompt)
                    self.assertEqual((kwargs['n'], kwargs['max_tokens'], kwargs['temperature'], kwargs['timeout']),
                                     (1, 16384, 0.6, 660))
                    self.assertNotIn('extra_body', kwargs)
                    # Decide before the await, while this is still the first request.
                    bad = key == 'lite/a' and not any(r[0] == key for r in seen)
                    seen.append((key, sampling_identity().copy(), threading.current_thread().name))
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(.005)
                    active -= 1
                    return completion('not parseable' if bad else '<result>SELECT x FROM t</result>')
                def client(**kwargs):
                    self.assertEqual(kwargs['max_retries'], 0)
                    fixture = AsyncClientFixture(create)
                    clients.append(fixture)
                    return fixture
                with patch('openai.AsyncOpenAI', side_effect=client), \
                     patch.object(LLM, '_create_client', side_effect=AssertionError('unused sync client created')), \
                     patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)):
                    result = cli.execute_run(store, ENV)
                self.assertEqual((result['succeeded'], result['failed']), (2, 0))
                self.assertEqual(peak, 2)
                self.assertEqual(len(seen), 25)
                self.assertEqual(result['runtime']['requests']['peak_in_flight'], 2)
                self.assertLessEqual(result['runtime']['coordinators']['peak'], coordinators)
                self.assertTrue(all(c.closed for c in clients))
                self.assertEqual({name for _, _, name in seen}, {'deepeye-http-loop'})
                usage = observed_usage(store)
                self.assertEqual(usage['effective_sampling']['retained_samples'], 24)
                self.assertEqual(usage['effective_sampling']['known_tokens']['total_tokens'], 120)
                self.assertEqual(usage['reported_tokens']['total_tokens'], 125)
                groups = [e['payload'] for e in store.events() if e['kind'] == 'sampling_group_result']
                self.assertEqual([g['target_n'] for g in groups], [4] * 6)
                self.assertTrue(all(g['complete'] for g in groups))
                self.assertTrue(all(a['payload']['rc_participation']['status'] == 'participating' for a in store.attempts()))
                self.assertTrue(store.verify()['ok'])

    def test_explicit_downstream_uses_new_generation_and_rc_stays_in_target(self):
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from app.db_utils.execution import SQLExecutionResult
        from scripts.rc_evaluation.deepeye.injection import render_rc_block
        seen_sql = []
        async def create(**kwargs):
            return completion('<result>SELECT x FROM t WHERE x = 7</result>')
        def pg(item, sql, timeout=None):
            seen_sql.append(sql)
            return SQLExecutionResult(result_type='success', db_path='db', sql=sql,
                execution_time=.01, result_rows=[(7,)], result_cols=['x'])
        with tempfile.TemporaryDirectory() as temporary, self.prepared(temporary, calls=1, downstream=True) as store:
            client = AsyncClientFixture(create)
            with patch('openai.AsyncOpenAI', return_value=client), \
                 patch.object(LLM, '_create_client', side_effect=AssertionError('sync client forbidden')), \
                 patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)), \
                 patch('scripts.baseline_adapters.deepeye.backend_hooks.execute_postgres_sql', side_effect=pg):
                result = cli.execute_run(store, ENV)
            self.assertEqual((result['succeeded'], result['failed']), (1, 0))
            rows = {a['stage']: a for a in store.attempts()}
            self.assertEqual(rows['sql_revision']['payload']['artifact']['sql_candidates_after_revision'],
                             ['SELECT x FROM t WHERE x = 7'] * 12)
            self.assertEqual(rows['sql_selection']['payload']['artifact']['final_selected_sql'], 'SELECT x FROM t WHERE x = 7')
            self.assertTrue(all(a['payload']['execution_origin'] == 'executed' for a in rows.values()))
            self.assertTrue(seen_sql)
            starts = [e['payload'] for e in store.events() if e['kind'] == 'component_start'
                      and e['attempt_id'] == rows['sql_revision']['attempt_id']]
            checkers = [p for p in starts if p['component'].startswith('revision.')
                        and p['component'] != 'revision.candidate']
            self.assertEqual([p['component'] for p in checkers], [
                'revision.SyntaxChecker', 'revision.JoinChecker', 'revision.OrderByLimitChecker',
                'revision.TimeChecker', 'revision.SelectChecker', 'revision.MaxMinChecker',
                'revision.OrderByNullChecker', 'revision.ResultChecker'])
            self.assertTrue(all(p['inputs']['sampling_budget'] == 5 for p in checkers))
            requests = [e for e in store.events() if e['kind'] == 'api_request']
            self.assertEqual({e['attempt_id'] for e in requests}, {rows['sql_generation']['attempt_id']})
            block = render_rc_block(store.manifest['contracts']['lite/a'])
            self.assertTrue(all(block in e['payload']['kwargs']['messages'][0]['content'] for e in requests))
