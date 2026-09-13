"""Offline commands, runtime preflight and checkout-independent entry points."""
import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import OfflineTestCase

from .test_source import CODE, make_source
from .test_runner import experiment_manifest
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.deepeye_bird_interact_run import build_effective_config, code_source_hashes
try:
    from scripts.rc_evaluation.deepeye import cli
except ImportError:
    cli = None


ENV = {'DASH_MODELS': 'offline', 'DASH_BASE_URL': 'https://invalid.test/v1',
       'DASH_API_KEY': 'test-key', 'EMBEDDING_MODEL': 'offline',
       'EMBEDDING_BASE_URL': 'https://invalid.test/v1', 'EMBEDDING_API_KEY': 'test-key',
       'PG_HOST': 'invalid.test', 'PG_PORT': '5432', 'PG_USER': 'reader', 'PG_PASSWORD': 'test-password'}


def native_args():
    return SimpleNamespace(workers=1, inner_workers=1, pg_concurrency=1, max_tokens=6144,
        thinking_budget=None, chat_timeout=1200, pg_sslmode='prefer', extractor_retries=2,
        direct_linking_budget=1, reversed_linking_budget=1, dc_generation_budget=1,
        skeleton_generation_budget=1, icl_generation_budget=1, revision_checker_budget=1,
        selection_evaluator_budget=1, adaptive_concurrency=False)


class CliTests(OfflineTestCase):
    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                return cli.main(arguments)
            except SystemExit as error:
                self.fail(f'Expected supported CLI arguments, parser exited with {error.code}')

    def test_saved_dynamic_policy_round_trips_without_falling_back_to_fixed(self):
        # Break caught: run/resume silently rebuilds a fixed controller or defaults.
        args = native_args()
        args.adaptive_concurrency = True
        args.concurrency_initial, args.concurrency_step = 3, 2
        args.concurrency_min, args.concurrency_max = 1, 9
        args.concurrency_window = 0.75
        effective = build_effective_config(ENV, args)
        restored = cli.runtime_args(effective)
        self.assertTrue(restored.adaptive_concurrency)
        rebuilt = build_effective_config(ENV, restored)
        self.assertEqual(rebuilt, effective)
        self.assertEqual(rebuilt['admission']['pipeline']['initial_limit'], 3)
        self.assertEqual(rebuilt['admission']['pipeline']['step'], 2)
        self.assertEqual(rebuilt['admission']['pipeline']['max_limit'], 9)
        self.assertEqual(rebuilt['admission']['pipeline']['stable_window_s'], 0.75)
        defaults = cli.runtime_args(build_effective_config(ENV, native_args()),
                                    SimpleNamespace(adaptive_concurrency=True))
        self.assertEqual((defaults.concurrency_initial, defaults.concurrency_step,
                          defaults.concurrency_min, defaults.concurrency_max, defaults.concurrency_window),
                         (50, 10, 10, 100, 60.0))

    def test_prepare_freezes_custom_dynamic_policy_and_replay_uses_it_offline(self):
        # Break caught: options accepted but not used/frozen, or reuse bypasses slots.
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, arguments, environment = self.fixture(temporary)
            arguments += ['--adaptive-concurrency', '--concurrency-initial', '200',
                          '--concurrency-step', '10', '--concurrency-min', '5',
                          '--concurrency-max', '400', '--concurrency-window', '2.5']
            with patch.object(cli, 'prepare_inputs', return_value=inputs):
                self.assertEqual(self.invoke(arguments), 0)
            with RunStore.open(root / 'run') as store:
                policy = store.manifest['effective_config']['admission']['pipeline']
                self.assertEqual({key: policy[key] for key in ('initial_limit', 'step', 'min_limit', 'max_limit', 'stable_window_s')},
                                 {'initial_limit': 200, 'step': 10, 'min_limit': 5, 'max_limit': 400, 'stable_window_s': 2.5})
                restored = cli._check_frozen(store, ENV)
                self.assertTrue(restored.adaptive_concurrency)
                with patch.object(cli, 'build_runtime_config', side_effect=AssertionError('reuse built native resources')):
                    result = cli.execute_run(store, ENV)
                    self.assertIn('admission', result)
                    self.assertEqual(result['admission']['pipeline']['current_limit'], 200)
                    self.assertEqual(result['admission']['pipeline']['active'], 0)
                    self.assertEqual(result['admission']['pipeline']['model']['requested'], 0)
                    count = len(store.attempts())
                    repeated = cli.execute_run(store, ENV)
                self.assertEqual(repeated['succeeded'], 1)
                self.assertEqual(len(store.attempts()), count)
                self.assertEqual(store.attempts()[0]['payload']['execution_origin'], 'reused_no_native_llm_call')
                self.assertEqual(store.events(), [])

    def test_prepare_from_dynamic_baseline_still_requires_explicit_opt_in(self):
        # Break caught: baseline scheduling unexpectedly enables RC adaptive mode.
        source_args = native_args()
        source_args.adaptive_concurrency = True
        source_args.concurrency_initial, source_args.concurrency_step = 200, 10
        source_args.concurrency_min, source_args.concurrency_max = 10, 400
        source_args.concurrency_window = 60.0
        effective = build_effective_config(ENV, source_args)
        basic = ['prepare', '--source-run', '/unused-source', '--run-dir', '/unused-run',
                 '--target-stage', 'sql_revision', '--condition', 'none']
        options = cli.build_parser().parse_args(basic)
        fixed = cli.runtime_args(effective, options)
        self.assertFalse(fixed.adaptive_concurrency)
        options = cli.build_parser().parse_args(basic + ['--adaptive-concurrency'])
        adaptive = cli.runtime_args(effective, options)
        self.assertTrue(adaptive.adaptive_concurrency)
        self.assertEqual(build_effective_config(ENV, adaptive)['admission'], effective['admission'])
        partial = cli.build_parser().parse_args(basic + ['--adaptive-concurrency', '--concurrency-step', '20'])
        overridden = build_effective_config(ENV, cli.runtime_args(effective, partial))['admission']['pipeline']
        self.assertEqual({key: overridden[key] for key in ('initial_limit', 'step', 'min_limit', 'max_limit', 'stable_window_s')},
                         {'initial_limit': 200, 'step': 20, 'min_limit': 10, 'max_limit': 400, 'stable_window_s': 60.0})

    def test_invalid_dynamic_options_rejected_before_directory_creation(self):
        # Break caught: malformed or silently ignored policies reach paid execution.
        invalid = [
            ['--concurrency-initial', '2'],
            ['--adaptive-concurrency', '--concurrency-initial', '0'],
            ['--adaptive-concurrency', '--concurrency-step', '0'],
            ['--adaptive-concurrency', '--concurrency-min', '0'],
            ['--adaptive-concurrency', '--concurrency-max', '0'],
            ['--adaptive-concurrency', '--concurrency-min', '60'],
            ['--adaptive-concurrency', '--concurrency-initial', '101'],
            ['--adaptive-concurrency', '--concurrency-window', '0'],
            ['--adaptive-concurrency', '--concurrency-window', 'nan'],
            ['--adaptive-concurrency', '--concurrency-window', 'inf'],
        ]
        for options in invalid:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                root, inputs, arguments, _ = self.fixture(temporary)
                with patch.object(cli, 'prepare_inputs', return_value=inputs):
                    self.assertEqual(self.invoke(arguments + options), 2)
                self.assertFalse((root / 'run').exists())

    def test_help_works_from_repo_and_code_without_network(self):
        self.assertIsNotNone(cli, 'CLI implementation is required')
        for cwd, relative in [(CODE, 'scripts/rc_evaluation/deepeye/cli.py'),
                              (CODE.parent, 'code/scripts/rc_evaluation/deepeye/cli.py')]:
            completed = subprocess.run([str(CODE / '.venv/bin/python'), relative, '--help'],
                                       cwd=cwd, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('prepare', completed.stdout)
            self.assertIn('compare', completed.stdout)

    def fixture(self, temporary):
        root = Path(temporary)
        config = build_effective_config(ENV, native_args())
        tasks = make_source(root / 'source', calls=0, source_config=config, code_hashes=code_source_hashes())
        with RunStore.open(root / 'source', read_only=True) as source:
            baseline = source.manifest
        environment = root / '.env'
        environment.write_text('\n'.join(f'{key}={value}' for key, value in ENV.items()))
        inputs = (tasks, baseline['items'], baseline['sources'])
        args = ['prepare', '--source-run', str(root / 'source'), '--run-dir', str(root / 'run'),
                '--target-stage', 'sql_revision', '--condition', 'none', '--env-file', str(environment)]
        return root, inputs, args, environment

    def test_prepare_inspect_export_and_replay_are_offline(self):
        self.assertIsNotNone(cli)
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, args, environment = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs), patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(args), 0)
                    self.assertEqual(cli.main(['run', '--run-dir', str(root / 'run'), '--env-file', str(environment)]), 0)
                    self.assertEqual(cli.main(['inspect', '--run-dir', str(root / 'run')]), 0)
                    self.assertEqual(cli.main(['export', '--run-dir', str(root / 'run'), '--export-dir', str(root / 'export')]), 0)
            with RunStore.open(root / 'run', read_only=True) as store:
                self.assertEqual(store.attempts()[0]['payload']['execution_origin'], 'reused_no_native_llm_call')
            self.assertTrue((root / 'export' / 'COMPLETE.json').exists())

    def test_changed_budget_rejected_before_run_directory_exists(self):
        self.assertIsNotNone(cli)
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, args, _ = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args + ['--max-tokens', '999']), 2)
            self.assertFalse((root / 'run').exists())

    def test_timeout_above_1200_rejected_before_prepare(self):
        self.assertIsNotNone(cli)
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, args, _ = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args + ['--chat-timeout', '1201']), 2)
            self.assertFalse((root / 'run').exists())

    def test_changed_experiment_code_rejects_resume_before_native_calls(self):
        self.assertIsNotNone(cli)
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, args, environment = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
            with patch.object(cli, 'production_hash', return_value='changed'), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['resume', '--run-dir', str(root / 'run'), '--env-file', str(environment)]), 2)
            with RunStore.open(root / 'run', read_only=True) as store:
                self.assertEqual(store.attempts(), [])

    def test_failed_stage_returns_nonzero_and_keeps_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, args, environment = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
            def failed_stage(store, environment):
                # Exercise the CLI outcome boundary with an actual committed failure.
                attempt = store.begin_attempt('lite/a', 'sql_revision', 'synthetic-failure')
                store.finish_attempt(attempt, 'failed', {'error_type': 'SyntheticFailure'})
                return {'succeeded': 0, 'failed': 1}
            with patch.object(cli, 'execute_run', side_effect=failed_stage), redirect_stdout(io.StringIO()):
                self.assertNotEqual(cli.main(['run', '--run-dir', str(root / 'run'), '--env-file', str(environment)]), 0)
            with RunStore.open(root / 'run', read_only=True) as store:
                self.assertEqual(store.attempts()[0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
