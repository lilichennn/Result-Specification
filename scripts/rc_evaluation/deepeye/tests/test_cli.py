"""Offline commands, runtime preflight and checkout-independent entry points."""
import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import OfflineTestCase

from .test_source import CODE, make_source
from .test_runner import experiment_manifest
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.deepeye_run import build_effective_config, code_source_hashes
try:
    from scripts.rc_evaluation.deepeye import cli
except ImportError:
    cli = None


ENV = {'DASH_MODELS': 'offline', 'DASH_BASE_URL': 'https://invalid.test/v1',
       'DASH_API_KEY': 'test-key', 'EMBEDDING_MODEL': 'offline',
       'EMBEDDING_BASE_URL': 'https://invalid.test/v1', 'EMBEDDING_API_KEY': 'test-key',
       'PG_HOST': 'invalid.test', 'PG_PORT': '5432', 'PG_USER': 'reader', 'PG_PASSWORD': 'test-password'}


def native_args():
    return SimpleNamespace(pg_concurrency=1, max_tokens=16384,
        thinking_budget=None, chat_timeout=660, pg_sslmode='prefer', extractor_retries=2,
        direct_linking_budget=4, reversed_linking_budget=4, dc_generation_budget=4,
        skeleton_generation_budget=4, icl_generation_budget=4, revision_checker_budget=5,
        selection_evaluator_budget=5, request_limit=2, request_workers=4,
        coordinator_workers=4, http_connections=4, start_rate=10000.0, retry_delay=0.0)


class CliTests(OfflineTestCase):
    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                return cli.main(arguments)
            except SystemExit as error:
                self.fail(f'Expected supported CLI arguments, parser exited with {error.code}')

    def test_saved_runtime_round_trips_without_falling_back_to_legacy_pools(self):
        args = native_args()
        effective = build_effective_config(ENV, args)
        restored = cli.runtime_args(effective)
        rebuilt = build_effective_config(ENV, restored)
        self.assertEqual(rebuilt, effective)
        self.assertEqual(rebuilt['runtime']['request_limit'], 2)
        self.assertEqual(rebuilt['runtime']['coordinator_workers'], 4)
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            cli.runtime_args(effective, SimpleNamespace(adaptive_concurrency=True))

    def test_prepare_freezes_custom_runtime_and_replay_needs_no_native_clients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, arguments, environment = self.fixture(temporary)
            arguments += ['--request-limit', '5', '--request-workers', '7',
                          '--coordinator-workers', '3', '--request-start-rate', '12.5']
            with patch.object(cli, 'prepare_inputs', return_value=inputs):
                self.assertEqual(self.invoke(arguments), 0)
            with RunStore.open(root / 'run') as store:
                policy = store.manifest['effective_config']['runtime']
                self.assertEqual((policy['request_limit'], policy['request_workers'], policy['coordinator_workers'], policy['start_rate']),
                                 (5, 7, 3, 12.5))
                restored = cli._check_frozen(store, ENV)
                self.assertEqual(restored.request_limit, 5)
                with patch.object(cli, 'build_runtime_config', side_effect=AssertionError('reuse built native resources')):
                    result = cli.execute_run(store, ENV)
                    self.assertIn('admission', result)
                    self.assertEqual(result['admission']['pipeline']['current_limit'], 1)
                    self.assertEqual(result['admission']['pipeline']['active'], 0)
                    self.assertEqual(result['runtime']['requests']['submitted'], 0)
                    count = len(store.attempts())
                    repeated = cli.execute_run(store, ENV)
                self.assertEqual(repeated['succeeded'], 1)
                self.assertEqual(len(store.attempts()), count)
                self.assertEqual(store.attempts()[0]['payload']['execution_origin'], 'reused_no_native_llm_call')
                self.assertEqual(store.events(), [])

    def test_replay_only_execution_creates_no_runtime_or_threads(self):
        # Break caught: an all-replay invocation still allocates native scheduling resources.
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, arguments, _ = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs):
                self.assertEqual(self.invoke(arguments), 0)
            with RunStore.open(root / 'run') as store, \
                    patch.object(cli, 'sampling_runtime', side_effect=AssertionError('sampling runtime created')), \
                    patch.object(cli, 'admission_context', side_effect=AssertionError('admission context created')), \
                    patch.object(threading.Thread, 'start', side_effect=AssertionError('thread started')):
                result = cli.execute_run(store, ENV)
            self.assertEqual(result['succeeded'], 1)
            self.assertEqual(result['runtime']['requests']['submitted'], 0)
            self.assertEqual(result['admission']['pipeline']['active'], 0)

    def test_old_runtime_manifest_cannot_start_new_paid_work(self):
        effective = build_effective_config(ENV, native_args())
        effective.pop('runtime')
        effective['profile'] = 'bounded-config'
        with self.assertRaisesRegex(ValueError, 'Legacy native runtime'):
            cli.runtime_args(effective)

    def test_generic_reference_paths_preserve_partition_and_reject_conflicts(self):
        args = cli.build_parser().parse_args([
            'evaluate', '--run-dir', 'run', '--output-dir', 'out', '--database-version', 'v1',
            '--reference', 'spider/dev=dev.json', '--reference', 'spider/test=test.json',
        ])
        self.assertEqual(cli._evaluation_paths(args), {
            'spider/dev': Path('dev.json'), 'spider/test': Path('test.json')})
        args = cli.build_parser().parse_args([
            'evaluate', '--run-dir', 'run', '--output-dir', 'out', '--database-version', 'v1',
            '--reference', 'lite=generic.json', '--reference-lite', 'legacy.json',
        ])
        with self.assertRaisesRegex(ValueError, 'Duplicate reference source'):
            cli._evaluation_paths(args)

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
        entry = CODE / 'scripts/rc_evaluation/deepeye/cli.py'
        for cwd in (CODE, CODE.parent):
            completed = subprocess.run([sys.executable, str(entry), '--help'],
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

    def test_resume_prepares_full_experiment_once(self):
        from scripts.rc_evaluation.deepeye import runner, source as source_module
        with tempfile.TemporaryDirectory() as temporary:
            root, inputs, args, environment = self.fixture(temporary)
            with patch.object(cli, 'prepare_inputs', return_value=inputs), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
            with patch.object(runner, 'prepare_experiment', wraps=runner.prepare_experiment) as prepare, \
                 patch.object(runner, 'validate_manifest', wraps=runner.validate_manifest) as validate, \
                 patch.object(source_module, 'restore_seed', wraps=source_module.restore_seed) as restore, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['resume', '--run-dir', str(root / 'run'),
                                           '--env-file', str(environment), '--unfinished-only']), 0)
            self.assertEqual((prepare.call_count, validate.call_count, restore.call_count), (1, 1, 1))

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
            def failed_stage(store, environment, **unused):
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
