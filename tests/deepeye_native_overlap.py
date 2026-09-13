"""Fake transport/PG boundaries for real native branch-overlap regressions."""
import asyncio
from collections import defaultdict

from scripts.rc_evaluation.deepeye.tests.test_dynamic_concurrency import completion


GENERATION_SQL = [
    'SELECT broken_a FROM t', 'SELECT  broken_b FROM t',
    'SELECT   broken_a FROM t', 'SELECT    broken_b FROM t',
    'SELECT     broken_a FROM t', 'SELECT      broken_b FROM t',
    'SELECT       broken_a FROM t', 'SELECT        broken_b FROM t',
    'SELECT         broken_a FROM t', 'SELECT          broken_b FROM t',
    'SELECT           broken_a FROM t', 'SELECT            broken_b FROM t',
]
REVISED_SQL = ['SELECT x FROM t WHERE x = 0', 'SELECT x FROM t WHERE x = 10']
CHECKERS = ['SyntaxChecker', 'JoinChecker', 'OrderByLimitChecker', 'TimeChecker',
            'SelectChecker', 'MaxMinChecker', 'OrderByNullChecker', 'ResultChecker']


class NativeOverlapTransport:
    def __init__(self, store, *, coordinators=32):
        self.store = store
        self.coordinators = coordinators
        self.active = defaultdict(int)
        self.peak = defaultdict(int)
        self.started = defaultdict(list)
        self.finished = defaultdict(list)
        self.barriers = {}
        self.timed_out = set()

    async def create(self, **kwargs):
        from app.llm.sampling import sampling_identity
        from scripts.baseline_adapters.deepeye.run_trace import _ATTEMPT_ID, _BRANCH_PATH
        row = self.store.attempt(_ATTEMPT_ID.get())
        stage = row['stage']
        if stage == 'schema_linking':
            return completion('<result><table table_name="t"><column column_name="x" /></table></result>')
        index = sampling_identity()['sample_index']
        if stage == 'sql_generation':
            branch = next(name for name in _BRANCH_PATH.get() if name.startswith('generation.'))
            position = ['generation.dc', 'generation.skeleton', 'generation.icl'].index(branch) * 4 + index
            expected, per_branch, sql = 12, 4, GENERATION_SQL[position]
        elif stage == 'sql_revision':
            candidate = 0 if 'broken_a' in kwargs['messages'][0]['content'] else 1
            position = candidate * 5 + index
            expected, per_branch, sql = 10, 5, f'SELECT x FROM t WHERE x = {candidate * 10 + index}'
        else:
            raise AssertionError(f'Unexpected model call in {stage}')
        key = (row['item_key'], stage)
        barrier = self.barriers.setdefault(key, asyncio.Event())
        self.started[key].append(position)
        self.active[key] += 1
        self.peak[key] = max(self.peak[key], self.active[key])
        if self.active[key] == min(expected, self.coordinators * per_branch):
            barrier.set()
        try:
            try:
                await asyncio.wait_for(barrier.wait(), 2)
            except TimeoutError:
                # Let a serialized implementation finish, then fail on evidence.
                self.timed_out.add(key)
                barrier.set()
            await asyncio.sleep((expected - position) * .01)
            self.finished[key].append(position)
            return completion(f'<result>{sql}</result>')
        finally:
            self.active[key] -= 1


def execute_sql(target, sql, timeout=None):
    from app.db_utils.execution import SQLExecutionResult
    if 'broken_' in sql:
        return SQLExecutionResult(result_type='execution_error', db_path='db', sql=sql,
                                  error_message='offline missing column')
    return SQLExecutionResult(result_type='success', db_path='db', sql=sql,
                             execution_time=.01, result_rows=[(1,)], result_cols=['x'])


def assert_overlap(test, transport, key, stage, expected, *, peak=None):
    identity = (key, stage)
    test.assertEqual(transport.peak[identity], expected if peak is None else peak,
                     f'{stage} independent branches serialized: starts={transport.started[identity]}')
    test.assertNotIn(identity, transport.timed_out)
    test.assertEqual(sorted(transport.started[identity]), list(range(expected)))
    test.assertNotEqual(transport.finished[identity], list(range(expected)))
    test.assertEqual(transport.active[identity], 0)


def assert_sequential_checkers(test, store, attempt):
    events = list(store.iter_events(attempt['attempt_id']))
    candidates = [e['payload']['component_call_id'] for e in events
                  if e['kind'] == 'component_start' and e['payload']['component'] == 'revision.candidate']
    test.assertEqual(len(candidates), 2)
    for candidate in candidates:
        checks = [e for e in events if e['kind'] in ('component_start', 'component_result')
                  and e['payload'].get('parent_component_call_id') == candidate]
        test.assertEqual([(e['kind'], e['payload']['component']) for e in checks],
                         [(kind, f'revision.{checker}') for checker in CHECKERS
                          for kind in ('component_start', 'component_result')])
        test.assertTrue(all(e['payload']['inputs']['sampling_budget'] == 5 for e in checks
                            if e['kind'] == 'component_start'))
