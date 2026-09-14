"""Observe real native preliminary selection without repeating generation/SQL."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_deepeye_workloads  # native import bootstrap
from app.few_shot.preliminary_sql import PreliminarySQLGenerator
from app.db_utils.execution import SQLExecutionResult
from app.pipeline.utils import get_execution_result_hash


class ContextPool:
    def __init__(self, pool):
        self.pool = pool

    def submit(self, function, *args):
        return self.pool.submit(copy_context().run, function, *args)


class PreliminaryDiagnosticsTests(unittest.TestCase):
    def item(self, identity=1):
        return SimpleNamespace(question_id=identity, database_schema={},
            get_item_id=lambda: identity,
            model_copy=lambda deep: SimpleNamespace(question_id=identity))

    def generator(self, pool, dc, skeleton, service):
        generator = object.__new__(PreliminarySQLGenerator)
        generator._executor = ContextPool(pool)
        generator._config = SimpleNamespace(dc_sampling_budget=4, skeleton_sampling_budget=4)
        generator._llm = None
        generator._dc_generator = SimpleNamespace(generate=dc)
        generator._skeleton_generator = SimpleNamespace(generate=skeleton)
        generator._execution_service = service
        return generator

    def test_branch_failure_and_successful_empty_candidates_have_distinct_fallbacks(self):
        from scripts.baseline_adapters.deepeye.preliminary_diagnostics import attach_preliminary_diagnostics
        zero = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        def failed(*args):
            raise RuntimeError('exhausted retries')
        with ThreadPoolExecutor(2) as pool, patch('app.few_shot.preliminary_sql.traceback.print_exc'), \
             patch('app.few_shot.preliminary_sql.logger.error'):
            generator = self.generator(pool, failed, lambda *a: ([], zero), None)
            result = attach_preliminary_diagnostics(generator)(self.item())
            self.assertEqual(result['diagnostics']['fallback_reason'], 'generation_error')
            self.assertEqual(result['diagnostics']['branches']['dc']['error_type'], 'RuntimeError')
            self.assertEqual(result['diagnostics']['branches']['skeleton']['status'], 'succeeded')
            generator = self.generator(pool, lambda *a: ([], zero), lambda *a: ([], zero), None)
            result = attach_preliminary_diagnostics(generator)(self.item())
            self.assertEqual(result['diagnostics']['fallback_reason'], 'no_candidates')

    def test_native_selection_is_unchanged_with_bounded_execution_details_and_no_repeat_calls(self):
        from scripts.baseline_adapters.deepeye.preliminary_diagnostics import attach_preliminary_diagnostics
        calls, hashed, events = [], [], []
        def execute(item, sql):
            calls.append(sql)
            return SQLExecutionResult(result_type='execution_error' if sql == 'BAD' else 'success',
                db_path='unused', sql=sql, execution_time=.1 if sql == 'SELECT 1' else .2,
                result_cols=['x'], result_rows=None if sql == 'BAD' else ([(1,)] * 30 if sql == 'SELECT 1' else [(2,)]),
                error_message='syntax error' if sql == 'BAD' else None)
        def hash_result(item, rows):
            hashed.append(rows)
            return get_execution_result_hash(item, rows)
        zero = {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}
        with ThreadPoolExecutor(2) as pool:
            generator = self.generator(pool, lambda *a: (['SELECT 1', 'BAD'], zero),
                lambda *a: (['SELECT 2'], zero), SimpleNamespace(execute=execute, hash_result=hash_result))
            expected = asdict(generator.generate(self.item()))
            calls.clear(); hashed.clear()
            run = attach_preliminary_diagnostics(generator, SimpleNamespace(emit=events.append))
            self.assertIs(attach_preliminary_diagnostics(generator), run)
            result = run(self.item())
        diagnostics = result.pop('diagnostics')
        self.assertEqual(result, expected)
        self.assertEqual(calls, ['SELECT 1', 'BAD', 'SELECT 2'])
        self.assertEqual(len(hashed), 2)
        self.assertEqual(result['sql'], 'SELECT 1')
        self.assertIsNone(diagnostics['fallback_reason'])
        rows = diagnostics['executions']
        self.assertEqual(rows[0]['row_count'], 30)
        self.assertEqual(len(rows[0]['rows_preview']), 20)
        self.assertTrue(rows[0]['preview_truncated'])
        self.assertIsNotNone(rows[0]['result_hash'])
        self.assertEqual(rows[1]['error_message'], 'syntax error')
        self.assertEqual(rows[1]['result_type'], 'execution_error')
        self.assertEqual(events[-1]['kind'], 'preliminary_diagnostics')

    def test_concurrent_questions_do_not_share_diagnostics(self):
        from scripts.baseline_adapters.deepeye.preliminary_diagnostics import attach_preliminary_diagnostics
        def generate(item, llm, budget):
            return [f'SELECT {item.question_id}'], {'total_tokens': 1}
        def execute(item, sql):
            return SQLExecutionResult(result_type='success', db_path='unused', sql=sql,
                result_cols=['x'], result_rows=[(item.question_id,)], execution_time=.1)
        with ThreadPoolExecutor(4) as branches, ThreadPoolExecutor(2) as questions:
            generator = self.generator(branches, generate, generate,
                SimpleNamespace(execute=execute, hash_result=get_execution_result_hash))
            run = attach_preliminary_diagnostics(generator)
            results = list(questions.map(run, [self.item(7), self.item(19)]))
        for identity, result in zip((7, 19), results):
            self.assertEqual([row['sql'] for row in result['diagnostics']['executions']], [f'SELECT {identity}'])
            self.assertEqual(result['diagnostics']['branches']['dc']['candidates'], [f'SELECT {identity}'])

    def test_native_execution_exception_propagates_and_is_audited(self):
        from scripts.baseline_adapters.deepeye.preliminary_diagnostics import attach_preliminary_diagnostics
        events = []
        def execute(*args):
            raise TimeoutError('simulated execution failure')
        with ThreadPoolExecutor(2) as pool:
            generator = self.generator(pool, lambda *a: (['SELECT 1'], {}), lambda *a: ([], {}),
                SimpleNamespace(execute=execute, hash_result=lambda *a: None))
            run = attach_preliminary_diagnostics(generator, SimpleNamespace(emit=events.append))
            with self.assertRaises(TimeoutError):
                run(self.item())
        self.assertEqual(events[-1]['diagnostics']['executions'][0]['error_type'], 'TimeoutError')

    def test_real_sampling_runtime_propagates_question_context_and_native_execution_failure_fallback(self):
        from scripts.baseline_adapters.deepeye.preliminary_diagnostics import attach_preliminary_diagnostics
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        def generate(item, llm, budget):
            self.assertEqual(budget, 4)
            return [f'BAD {item.question_id}'], {'total_tokens': 2}
        def execute(item, sql):
            return SQLExecutionResult(result_type='execution_error', db_path='unused', sql=sql,
                result_cols=None, result_rows=None, error_message='bad SQL')
        runtime = SamplingRuntime(coordinator_workers=2, request_workers=2)
        try:
            generator = self.generator(None, generate, generate,
                SimpleNamespace(execute=execute, hash_result=lambda *a: self.fail('cannot hash failed SQL')))
            generator._executor = runtime.executor_view()
            run = attach_preliminary_diagnostics(generator)
            with runtime.context():
                workflow = runtime.workflow_executor(2)
                futures = [workflow.submit(run, self.item(identity)) for identity in (7, 19)]
                results = [future.result() for future in futures]
            for identity, result in zip((7, 19), results):
                diagnostics = result['diagnostics']
                self.assertEqual(diagnostics['fallback_reason'], 'no_executable_candidates')
                self.assertEqual(set(diagnostics['branches']), {'dc', 'skeleton'})
                self.assertEqual(diagnostics['branches']['dc']['candidates'], [f'BAD {identity}'])
                self.assertEqual([row['sql'] for row in diagnostics['executions']], [f'BAD {identity}'])
        finally:
            runtime.close()

    def test_native_hash_with_postgres_special_values_and_nonfinite_preview_is_portable(self):
        from scripts.baseline_adapters.deepeye.preliminary_diagnostics import attach_preliminary_diagnostics
        from datetime import datetime, timedelta
        from decimal import Decimal
        import json
        cells = (float('nan'), float('inf'), Decimal('NaN'), datetime(2026, 9, 14),
                 timedelta(days=2, microseconds=1), memoryview(b'bytes'))
        calls = []
        def execute(item, sql):
            calls.append(sql)
            return SQLExecutionResult(result_type='success', db_path='unused', sql=sql,
                result_rows=[cells], result_cols=['nan', 'infinity', 'decimal', 'timestamp', 'interval', 'bytea'],
                execution_time=.1)
        with ThreadPoolExecutor(2) as pool:
            generator = self.generator(pool, lambda *a: (['SELECT special'], {}), lambda *a: ([], {}),
                SimpleNamespace(execute=execute, hash_result=get_execution_result_hash))
            expected = asdict(generator.generate(self.item()))
            calls.clear()
            result = attach_preliminary_diagnostics(generator)(self.item())
        diagnostic = result.pop('diagnostics')
        self.assertEqual(result, expected)
        self.assertEqual(calls, ['SELECT special'])
        self.assertEqual(diagnostic['executions'][0]['result_type'], 'success')
        self.assertEqual(len(diagnostic['executions'][0]['result_hash']), 64)
        json.dumps(diagnostic, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
