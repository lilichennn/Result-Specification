"""Offline integration through shared native, RC, and campaign boundaries."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_deepeye_workloads import WorkloadTests
from scripts import deepeye_run as native
from scripts.rc_evaluation.deepeye import cli


class SharedExecutionTests(unittest.TestCase):
    def test_generic_frozen_configuration_round_trips_without_postgres(self):
        with tempfile.TemporaryDirectory() as directory:
            workload, _ = WorkloadTests().fixture(Path(directory), 'spider', 'dev')
            args = native._build_parser().parse_args(['prepare', '--run-dir', directory, '--workload', str(workload)])
            env = {'DASH_MODELS': 'qwen3.6', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'offline'}
            effective = native.build_effective_config(env, args)
            restored = cli.runtime_args(effective)
            self.assertEqual(native.build_effective_config(env, restored), effective)
            self.assertNotIn('postgres', effective)

    def test_six_workloads_native_to_rc_resume_are_offline_and_preserve_types(self):
        from types import SimpleNamespace
        from scripts.baseline_adapters.deepeye.run_store import RunStore
        from scripts.baseline_adapters.deepeye.run_pipeline import STAGE_METHODS
        from scripts.rc_evaluation.deepeye.tests.test_source import complete_stage, cost
        from scripts.rc_evaluation.deepeye.tests.test_contracts import _record
        selections = [('bird', 'dev', False), ('spider', 'dev', False), ('spider', 'test', False),
                      ('spider2', 'lite', False), ('spider2', 'lite', True),
                      ('bird_interact', 'lite', False), ('bird_interact', 'full', False)]
        for benchmark, split, cloud in selections:
            with self.subTest(benchmark=benchmark, split=split, cloud=cloud), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workload, originals = WorkloadTests().fixture(root, benchmark, split, cloud=cloud)
                env = {'DASH_MODELS': 'qwen3.6', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'offline'}
                if benchmark == 'bird_interact':
                    env.update(PG_HOST='invalid.test', PG_PORT='5432', PG_USER='reader', PG_PASSWORD='offline')
                envfile = root/'env'; envfile.write_text('\n'.join(f'{key}={value}' for key,value in env.items()))
                rows = json.loads((root/'questions.json').read_text())
                (root/'rc.json').write_text(json.dumps([_record(row['index'], db_id='db', question=row['question'], evidence='') for row in rows]))
                observed_types = []
                def factory(stage, items):
                    def execute(item):
                        observed_types.append(type(item))
                        self.assertEqual(item.gold_sql, '')
                        complete_stage(item, stage)
                        setattr(item, stage+'_llm_cost', cost())
                    return SimpleNamespace(**{STAGE_METHODS[stage]: execute, '_clean_up': lambda: None})
                original_execute = native._execute_pipeline
                def execute(store, tasks, environment, args, **kwargs):
                    return original_execute(store, tasks, environment, args,
                        runner_factory_fn=lambda config: factory, **kwargs)
                base = ['--run-dir', str(root/'native'), '--workload', str(workload), '--env-file', str(envfile),
                        '--request-limit', '2', '--request-workers', '4', '--coordinator-workers', '4', '--http-connections', '4']
                with patch('socket.socket.connect', side_effect=AssertionError('offline integration called network')), \
                     patch('scripts.baseline_adapters.deepeye.run_trace.TraceRecorder.instrument_runner',
                           return_value=lambda: None), \
                     patch.object(native, '_execute_pipeline', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(native.main(['run', *base]), 0)
                    self.assertEqual(native.main(['resume', *base, '--unfinished-only']), 0)
                    self.assertEqual(cli.main(['prepare', '--source-run', str(root/'native'), '--run-dir', str(root/'rc-run'),
                        '--target-stage', 'sql_revision', '--condition', 'rc', '--env-file', str(envfile)]), 0)
                    self.assertEqual(cli.main(['resume', '--run-dir', str(root/'rc-run'), '--env-file', str(envfile), '--unfinished-only']), 0)
                self.assertEqual(set(observed_types), {type(originals[0])})
                with RunStore.open(root/'rc-run', read_only=True) as store:
                    self.assertEqual(len(store.attempts()), len(originals))
                    self.assertTrue(all(row['payload']['rc_participation']['status'] == 'rc_not_participating' for row in store.attempts()))
                    self.assertEqual(store.manifest['fingerprint_algorithm'], 'manifest-digest-v2')
