"""All shared routes exercise real native stages and RC model-node injection offline."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_deepeye_workloads import WorkloadTests
from scripts import deepeye_run as entry
from scripts.rc_evaluation.deepeye import cli
from scripts.baseline_adapters.deepeye.run_store import RunStore


class SharedNativeNodesTests(unittest.TestCase):
    def test_real_native_four_stages_and_rc_targets_preserve_backend_and_prompts(self):
        from app.llm import LLM
        from app.llm.sampling import sampling_identity
        from app.db_utils.execution import SQLExecutionResult
        from app.services.schema_service import SchemaService
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID, _BRANCH_PATH
        from scripts.rc_evaluation.deepeye.tests.test_dynamic_concurrency import AsyncClientFixture, completion
        from scripts.rc_evaluation.deepeye.tests.test_contracts import _record
        from scripts.rc_evaluation.deepeye.injection import render_rc_block
        cases = [('bird', 'dev', False), ('spider', 'dev', False), ('spider', 'test', False),
                 ('spider2', 'lite', False), ('spider2', 'lite', True),
                 ('bird_interact', 'lite', False), ('bird_interact', 'full', False)]
        for benchmark, split, cloud in cases:
            with self.subTest(benchmark=benchmark, split=split, cloud=cloud), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                workload, _ = WorkloadTests().fixture(root, benchmark, split, cloud=cloud)
                with sqlite3.connect(root/'resources/db.sqlite') as db:
                    db.executemany('INSERT INTO visible(id) VALUES(?)', [(0,), (1,)])
                env = {'DASH_MODELS': 'offline', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'offline'}
                if benchmark == 'bird_interact':
                    env.update(PG_HOST='invalid.test', PG_PORT='5432', PG_USER='reader', PG_PASSWORD='offline')
                envfile = root/'env'; envfile.write_text('\n'.join(f'{k}={v}' for k,v in env.items()))
                first = json.loads((root/'questions.json').read_text())[0]
                (root/'rc.json').write_text(json.dumps([_record(first['index'], db_id='db', question=first['question'], evidence='')]))
                args = entry._build_parser().parse_args(['run', '--workload', str(workload), '--run-dir', str(root/'native'),
                    '--request-limit', '8', '--request-workers', '8', '--coordinator-workers', '8', '--http-connections', '8',
                    '--request-start-rate', '10000', '--direct-linking-budget', '1', '--reversed-linking-budget', '1',
                    '--dc-generation-budget', '2', '--skeleton-generation-budget', '2', '--icl-generation-budget', '2',
                    '--revision-checker-budget', '1', '--selection-evaluator-budget', '1'])
                tasks, bindings, sources = entry.prepare_inputs(workload=workload)
                tasks, bindings = tasks[:1], bindings[:1]
                key = bindings[0]['task_key']
                active_store, sent, backend_calls = None, [], []
                async def create(**kwargs):
                    stage = active_store.attempt(_ATTEMPT_ID.get())['stage']
                    prompt = kwargs['messages'][0]['content']
                    sent.append((stage, prompt))
                    if stage == 'schema_linking':
                        content = ('<table table_name="visible"><column column_name="id" /></table>'
                            if any('direct' in branch for branch in _BRANCH_PATH.get()) else 'SELECT id FROM visible')
                    elif stage == 'sql_generation':
                        content = f"SELECT broken_{sampling_identity()['sample_index'] % 2} FROM visible"
                    elif stage == 'sql_revision':
                        content = 'SELECT id FROM visible WHERE id = ' + ('1' if 'broken_1' in prompt else '0')
                    else:
                        content = 'A'
                    return completion(f'<result>{content}</result>')
                def remote_sql(kind, sql):
                    backend_calls.append(kind)
                    if 'broken_' in sql:
                        return SQLExecutionResult(result_type='execution_error', db_path='db', sql=sql, error_message='missing column')
                    return SQLExecutionResult(result_type='success', db_path='db', sql=sql,
                        result_cols=['id'], result_rows=[(1 if '= 1' in sql else 0,)], execution_time=.001)
                def pg(item, sql, timeout=None):
                    self.assertEqual(item.db_type, 'postgresql')
                    return remote_sql('postgresql', sql)
                def bq(sql, db_path, credential_path=None, timeout=None):
                    self.assertEqual(db_path, 'db')
                    return remote_sql('bigquery', sql)
                with patch('openai.AsyncOpenAI', side_effect=lambda **kwargs: AsyncClientFixture(create)), \
                     patch.object(LLM, '_create_client', side_effect=AssertionError('sync client forbidden')), \
                     patch.object(SchemaService, '_get_encoding', return_value=SimpleNamespace(encode=list)), \
                     patch('scripts.baseline_adapters.deepeye.backend_hooks.execute_postgres_sql', side_effect=pg), \
                     patch('app.db_utils.cloud_execution.execute_bigquery_sql', side_effect=bq), \
                     patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), redirect_stdout(io.StringIO()):
                    manifest = entry.build_manifest(entry.build_effective_config(env, args), sources, bindings)
                    with RunStore.create(root/'native', manifest) as store:
                        active_store = store
                        result = entry._execute_pipeline(store, tasks, env, args)
                        self.assertEqual(result['succeeded'], 1)
                    native_prompts = {stage: [p for s,p in sent if s == stage] for stage in entry.STAGES}
                    self.assertTrue(all(native_prompts.values()), {s:len(p) for s,p in native_prompts.items()})
                    expected_dialect = 'postgresql' if benchmark == 'bird_interact' else ('bigquery' if cloud else 'sqlite')
                    self.assertTrue(any(expected_dialect in p.lower() for p in native_prompts['sql_generation']))
                    for stage in entry.STAGES:
                        rcpath = root/('rc-'+stage)
                        self.assertEqual(cli.main(['prepare', '--source-run', str(root/'native'), '--run-dir', str(rcpath),
                            '--target-stage', stage, '--condition', 'rc', '--env-file', str(envfile)]), 0)
                        with RunStore.open(rcpath) as store:
                            active_store = store
                            offset = len(sent)
                            result = cli.execute_run(store, env)
                            self.assertEqual(result['succeeded'], 1)
                            block = render_rc_block(store.manifest['contracts'][key])
                            current = sent[offset:]
                            self.assertTrue(current)
                            self.assertTrue(all(s == stage and block in p for s,p in current))
                            # Native ask() strips trailing whitespace in some
                            # branches; RC adds its block before that stripping.
                            self.assertEqual(sorted(p.removesuffix('\n\n'+block).rstrip() for _,p in current),
                                             sorted(p.rstrip() for p in native_prompts[stage]))
                            self.assertTrue(store.verify()['ok'])
                if expected_dialect == 'sqlite':
                    self.assertEqual(backend_calls, [])
                else:
                    self.assertEqual(set(backend_calls), {expected_dialect})
