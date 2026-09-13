"""Shared preparation must not repeat cohort work along the execution chain."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_deepeye_run_pipeline import item, Factory, FakeTrace
from scripts.baseline_adapters.deepeye import run_pipeline as pipeline
from scripts.baseline_adapters.deepeye.run_store import RunStore


class PreparedRunTests(unittest.TestCase):
    def test_prepared_prefix_reused_for_selection_and_execution_at_scale(self):
        for size in (5, 50, 500):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp:
                tasks = [('lite', item(str(i))) for i in range(size)]
                data = {'format': 'deepeye-run-v2', 'fingerprint_algorithm': 'manifest-digest-v2',
                        'items': [{'task_key': 'lite/' + str(i)} for i in range(size)]}
                with RunStore.create(Path(temp) / 'run', data) as store:
                    with patch('builtins.print'):
                        pipeline.run_pipeline(store, tasks, Factory(), FakeTrace(), workers=2)
                    with patch.object(store, 'verify', wraps=store.verify) as verify, \
                         patch.object(pipeline, '_restore', wraps=pipeline._restore) as restore, \
                         patch.object(store, 'attempts', wraps=store.attempts) as attempts:
                        prepared = pipeline.prepare_run(store, tasks, selected_keys=['lite/0'])
                        selected, report = pipeline.select_unfinished(store, tasks[:1], prepared=prepared)
                        result = pipeline.run_pipeline(store, tasks[:1],
                            lambda *args: self.fail('completed item constructed resources'), FakeTrace(), prepared=prepared)
                    self.assertEqual(selected, [])
                    self.assertEqual(result['succeeded'], 1)
                    self.assertEqual(report['terminal'], {'lite/0': 'succeeded'})
                    self.assertEqual((verify.call_count, restore.call_count, attempts.call_count), (1, 4, 1))

    def test_prepared_state_rejects_intervening_writes(self):
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {}) as store:
            tasks = [('lite', item())]
            prepared = pipeline.prepare_run(store, tasks)
            store.begin_attempt('unselected', 'pipeline', 'x')
            with self.assertRaisesRegex(ValueError, 'stale'):
                pipeline.run_pipeline(store, tasks, Factory(), FakeTrace(), prepared=prepared)

    def test_unselected_corruption_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp, RunStore.create(Path(temp) / 'run', {}) as store:
            attempt = store.begin_attempt('unselected', 'pipeline', 'x')
            store.finish_attempt(attempt, 'succeeded', {'answer': 1})
            store._connection.execute('DROP TRIGGER finishes_no_update')
            store._connection.execute("UPDATE finishes SET payload_json='{}'")
            with self.assertRaisesRegex(ValueError, 'verification'):
                pipeline.prepare_run(store, [('lite', item())])
