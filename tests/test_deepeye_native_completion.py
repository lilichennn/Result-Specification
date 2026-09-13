"""Keep experimental completion aligned with native DeepEye, not full sampling."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tests.test_deepeye_run_pipeline import Factory, FakeTrace, item, cost
from tests.test_deepeye_run_inheritance import complete_stage
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, _valid_output, run_pipeline
from scripts.baseline_adapters.deepeye.run_store import RunStore


class NativeCompletionTests(unittest.TestCase):
    def test_native_validation_not_content_quality_defines_completion(self):
        cases = [
            ('sql_generation', 'sql_candidates', [], True),
            ('sql_generation', 'sql_candidates', [''], True),
            ('sql_generation', 'sql_candidates', None, False),
            ('sql_revision', 'sql_candidates_after_revision', [], True),
            ('sql_revision', 'sql_candidates_after_revision', None, False),
            ('sql_selection', 'final_selected_sql', 'Error', True),
            ('sql_selection', 'final_selected_sql', '', True),
            ('sql_selection', 'final_selected_sql', None, False),
        ]
        for stage, field, value, expected in cases:
            with self.subTest(stage=stage, value=value):
                target = item()
                complete_stage(target, stage)
                setattr(target, field, value)
                self.assertEqual(target.is_stage_complete(stage), expected)
                self.assertEqual(_valid_output(target, stage), expected)

    def test_real_native_generation_distinguishes_empty_channel_and_failed_channel(self):
        from app.pipeline.sql_generation.sql_generation import SQLGenerationRunner
        from app.pipeline.sql_revision.sql_revision import SQLRevisionRunner
        cases = [
            ([['SELECT 1'], ['SELECT 2'], []], 'succeeded', None),
            ([['SELECT 1'], ['SELECT 2'], None], 'failed', 'sql_generation'),
            ([[], [], []], 'failed', 'sql_revision'),
        ]
        for outputs, status, failed_stage in cases:
            with self.subTest(outputs=outputs), tempfile.TemporaryDirectory() as temp:
                base = Factory()
                def factory(stage, items):
                    runner = base(stage, items)
                    if stage == 'sql_generation':
                        runner._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=3)
                        runner._stage_config = SimpleNamespace(dc_sampling_budget=4,
                            skeleton_sampling_budget=4, icl_sampling_budget=4)
                        runner._llm = None
                        for name, value in zip(('dc', 'skeleton', 'icl'), outputs):
                            setattr(runner, '_' + name + '_generator', SimpleNamespace(
                                generate=lambda target, llm, budget, value=value: (value, cost())))
                        runner._generate_sql = lambda target: SQLGenerationRunner._generate_sql(runner, target)
                    elif stage == 'sql_revision' and all(value == [] for value in outputs):
                        runner._revise_sql = lambda target: SQLRevisionRunner._revise_sql(runner, target)
                    return runner
                with RunStore.create(Path(temp) / 'run', {}) as store:
                    report = run_pipeline(store, [('lite', item())], factory, FakeTrace())
                    self.assertEqual(report['items']['lite/one']['status'], status)
                    self.assertEqual(report['items']['lite/one']['failed_stage'], failed_stage)
                    generation = next(row for row in store.attempts() if row['stage'] == 'sql_generation')
                    self.assertEqual(generation['status'], 'failed' if failed_stage == 'sql_generation' else 'succeeded')
                    if failed_stage is None:
                        self.assertEqual(generation['payload']['artifact']['sql_candidates'], ['SELECT 1', 'SELECT 2'])
                        self.assertEqual(len([row for row in store.attempts() if row['stage'] in STAGES]), 4)


if __name__ == '__main__':
    unittest.main()
