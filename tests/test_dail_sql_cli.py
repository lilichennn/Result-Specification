from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.rc_evaluation.dail_sql import cli, campaign


class CliTests(unittest.TestCase):
    def test_prepare_literal_all_uses_all_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resource = root / 'resources.json'
            resource.write_text('{}')
            captured = []
            def prepare(manifest, output, resources):
                captured.append(resources)
                return {'status': 'ready', 'groups': {'a': {'status': 'ready'}}}
            with patch('scripts.baseline_adapters.dail_sql.preparation.prepare', prepare), redirect_stdout(io.StringIO()):
                result = cli.main(['prepare', '--manifest', 'inputs.json', '--resources', str(resource),
                                   '--output', str(root), '--groups', 'all'])
            self.assertEqual(result, 0)
            self.assertNotIn('groups', captured[0])

    def test_empty_rerun_rejected_before_runner_and_invalid_status_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'batch_id': 'batch'}))
            targets = root / 'targets.jsonl'
            targets.write_text('')
            with patch.object(campaign, 'rerun_questions', create=True, side_effect=AssertionError('must reject before run')), redirect_stdout(io.StringIO()):
                try:
                    result = cli.main(['rerun', '--batch', str(root), '--targets', str(targets)])
                except SystemExit:
                    self.fail('rerun command missing')
            self.assertEqual(result, 1)
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                cli.main(['status', '--batch', str(root), '--preparation', str(root)])

    def test_smoke_freezes_small_resources_and_target_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}')
            targets = root / 'targets.jsonl'
            targets.write_text('{"group":"a","question_id":"0"}\n')
            received = []
            def run(value, prepared, output):
                received.append(value)
                return {'status': 'paused'}
            with patch.object(campaign, 'run_batch', run), redirect_stdout(io.StringIO()):
                try:
                    result = cli.main(['smoke', '--config', str(config), '--prepared', str(root),
                        '--env-file', str(root / 'env'), '--targets', str(targets), '--per-group', '1'])
                except SystemExit:
                    self.fail('smoke command missing')
            self.assertEqual(result, 1)
            self.assertEqual(received[0]['purpose'], 'smoke')
            self.assertEqual(received[0]['resources']['request_limit'], 20)
            self.assertLessEqual(received[0]['resources']['sql_workers'], 20)
            self.assertEqual(received[0]['smoke_per_group'], 1)
            self.assertEqual(received[0]['targets'][0].question_id, '0')
