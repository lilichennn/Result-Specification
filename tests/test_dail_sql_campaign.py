import importlib.util
import unittest
import asyncio
from pathlib import Path
import tempfile
import json
import hashlib
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import patch
import numpy as np

from scripts.baseline_adapters.dail_sql.config import TaskKey
from scripts.baseline_adapters.dail_sql.config import DailSettings
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.dail_sql.composite import run_composite, CompositePaused
from scripts.rc_evaluation.dail_sql import campaign

from scripts.baseline_adapters.dail_sql.config import MODES


class ProgressTests(unittest.TestCase):
    def test_question_preparation_progresses_while_sql_workers_are_busy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, tasks, pool = prepared_fixture(root / 'prepared')
            released = threading.Event()
            observed = []
            timers = []
            def busy_pool(*args, **kwargs):
                executor = ThreadPoolExecutor(*args, **kwargs)
                entered = [threading.Event() for _ in range(kwargs['max_workers'])]
                def block(event):
                    event.set()
                    released.wait()
                for event in entered:
                    executor.submit(block, event)
                self.assertTrue(all(event.wait(1) for event in entered))
                timer = threading.Timer(1, released.set)
                timers.append(timer)
                timer.start()
                return executor
            def factory(*args, **kwargs):
                observed.append(not released.is_set())
                released.set()
                async def pause(*args, **kwargs):
                    raise CompositePaused('offline')
                return pause
            try:
                with patch.dict('os.environ', {'DASH_MODELS': 'model', 'DASH_API_KEY': 'unused',
                                               'DASH_BASE_URL': 'https://offline.invalid'}, clear=True), \
                     patch.object(campaign, 'load_prepared_group', side_effect=lambda root, g: tasks[g]), \
                     patch.object(campaign, 'load_training_pool', return_value=pool), \
                     patch.object(campaign, 'make_round_runner', factory), \
                     patch.object(campaign, 'ThreadPoolExecutor', busy_pool), \
                     patch.object(campaign.RequestDispatcher, 'make_client', return_value=object()):
                    campaign.run_batch(config, root / 'prepared', root / 'batch')
                self.assertTrue(any(observed), 'question preparation waited behind busy SQL workers')
            finally:
                released.set()
                for timer in timers:
                    timer.cancel()

    def test_only_explicit_campaign_resume_authorizes_repaired_configuration(self):
        from scripts.baseline_adapters.dail_sql.transport import GroupRequester
        from tests.test_dail_sql_transport import FakeDispatcher
        import httpx2
        from openai import AuthenticationError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, tasks, pool = prepared_fixture(root / 'prepared')
            config['selected_groups'] = ['a']
            config['env_file'] = str(root / 'env')
            Path(config['env_file']).write_text('DASH_API_KEY=broken\nDASH_MODELS=model\nDASH_BASE_URL=https://offline.invalid\n')
            dispatchers, authorizations = [], []
            error = AuthenticationError('auth', response=httpx2.Response(401,
                request=httpx2.Request('POST', 'https://offline.invalid')), body={})
            class Dispatcher(FakeDispatcher):
                def __init__(self, limits, **kwargs):
                    super().__init__({0: [error]} if not dispatchers else {})
                    self.limits = limits
                    dispatchers.append(self)
                def make_client(self, **kwargs):
                    return object()
                def close(self):
                    pass
            def requester(*args, **kwargs):
                authorizations.append(kwargs.get('resume_configuration_errors', False))
                return GroupRequester(*args, **kwargs)
            def factory(task, *, version_id, requester, records, **kwargs):
                async def round_(number, rc, ids, **kw):
                    rid = 'B' if rc else 'A'
                    saved = records.find_source(version_id, 'round_result', rid)
                    if saved:
                        return saved['payload']
                    generation = await requester.generate(version_id=version_id, round_execution_id=rid,
                        model='model', messages=[{'role': 'user', 'content': 'SELECT 1'}])
                    if any(s.get('error', {}).get('pause') for s in generation['samples'] if s.get('error')):
                        raise CompositePaused('authentication')
                    payload = {**generation, 'round_execution_id': rid, 'round_no': 1,
                        'rc_injected': rc, 'actual_parent_round_id': None, 'example_ids': ids,
                        'status': 'failed', 'success_usage': None, 'candidates': [], 'selection': None,
                        'next_example_ids': [], 'error': {'category': 'retrieval_parse'}}
                    records.append(version_id, 'round_result', payload)
                    return payload
                return round_
            with patch.dict('os.environ', {}, clear=True), \
                 patch.object(campaign, 'load_prepared_group', side_effect=lambda root, g: tasks[g]), \
                 patch.object(campaign, 'load_training_pool', return_value=pool), \
                 patch.object(campaign, 'make_round_runner', factory), \
                 patch.object(campaign, 'RequestDispatcher', Dispatcher), \
                 patch.object(campaign, 'GroupRequester', requester):
                self.assertEqual(campaign.run_batch(config, root / 'prepared', root / 'batch')['status'], 'paused')
                Path(config['env_file']).write_text('DASH_API_KEY=repaired\nDASH_MODELS=model\nDASH_BASE_URL=https://offline.invalid\n')
                result = campaign.resume_batch(root / 'batch')
                self.assertEqual(result['status'], 'success', 'explicit repair resume remains trapped')
                self.assertEqual(authorizations, [False, True])
                self.assertTrue(dispatchers[1].calls)
                self.assertTrue(all(i['attempt_no'] <= 2 for i, _ in dispatchers[1].calls))

    def test_later_ready_group_is_bound_before_work_and_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, tasks, pool = prepared_fixture(root / 'prepared')
            marker = root / 'prepared' / 'groups' / 'b' / 'ready.json'
            marker.unlink()
            values = {'DASH_MODELS': 'model', 'DASH_API_KEY': 'secret', 'DASH_BASE_URL': 'https://example.invalid'}
            def runner(*args, **kwargs):
                async def pause(*args, **kwargs):
                    raise CompositePaused('offline')
                return pause
            with patch.dict('os.environ', values, clear=True), \
                 patch.object(campaign, 'load_prepared_group', side_effect=lambda root, g: tasks[g]), \
                 patch.object(campaign, 'load_training_pool', return_value=pool), \
                 patch.object(campaign, 'make_round_runner', runner), \
                 patch.object(campaign.RequestDispatcher, 'make_client', return_value=object()):
                campaign.run_batch(config, root / 'prepared', root / 'batch')
                marker.write_text('{}')
                campaign.resume_batch(root / 'batch')
                marker.write_text('{"changed":true}')
                with self.assertRaises(ValueError):
                    campaign.resume_batch(root / 'batch')

    def test_sigterm_cancels_and_drains_before_returning_paused(self):
        self.assertTrue(hasattr(campaign, '_interruptible'), 'safe signal shutdown missing')
        drained = []
        previous = signal.getsignal(signal.SIGTERM)
        async def operation():
            try:
                signal.raise_signal(signal.SIGTERM)
                await asyncio.Event().wait()
            finally:
                drained.append(True)
        result = asyncio.run(campaign._interruptible(operation()))
        self.assertEqual(result, {'status': 'paused'})
        self.assertEqual(drained, [True])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_owner_lock_refuses_second_owner_and_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with campaign._owner(root):
                with self.assertRaises(ValueError):
                    with campaign._owner(root):
                        self.fail('second owner entered')
            with campaign._owner(root):
                pass

    def test_smoke_enforces_per_group_and_request_bound_before_writes(self):
        from scripts.rc_evaluation.dail_sql import campaign
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, _, _ = prepared_fixture(root / 'prepared')
            config.update(purpose='smoke', smoke_per_group=1, batch_id='smoke')
            environment = {'DASH_MODELS': 'model', 'DASH_API_KEY': 'secret', 'DASH_BASE_URL': 'https://example.invalid'}
            with patch.dict('os.environ', environment, clear=True):
                manifest, _, _ = campaign._configuration(config, root / 'prepared', root / 'batch')
                self.assertEqual(manifest['groups'], {'a': {'ids': ['0']}, 'b': {'ids': ['0']}})
                config['targets'] = [TaskKey('smoke', 'a', '0')]
                with self.assertRaises(ValueError):
                    campaign._configuration(config, root / 'prepared', root / 'batch')
                config.pop('targets')
                config['resources']['request_limit'] = 21
                with self.assertRaises(ValueError):
                    campaign._configuration(config, root / 'prepared', root / 'batch')

    def test_all_four_modes_need_their_own_threshold(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.rc_evaluation.dail_sql.campaign'),
                             'campaign missing')
        from scripts.rc_evaluation.dail_sql.campaign import ready_for_next
        counts = {mode: 828 for mode in MODES}
        self.assertTrue(ready_for_next(1034, counts))
        counts['rc_both'] = 827
        self.assertFalse(ready_for_next(1034, counts))
        self.assertFalse(ready_for_next(0, counts))

    def test_frozen_configuration_and_model_rejected_before_experiment_writes(self):
        from scripts.rc_evaluation.dail_sql import campaign
        self.assertTrue(hasattr(campaign, 'run_batch'), 'public campaign missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, tasks, pool = prepared_fixture(root / 'prepared')
            config['model'] = 'different'
            environment = {'DASH_MODELS': 'model', 'DASH_API_KEY': 'secret', 'DASH_BASE_URL': 'https://example.invalid'}
            with patch.dict('os.environ', environment, clear=True), self.assertRaises(ValueError):
                campaign.run_batch(config, root / 'prepared', root / 'batch')
            self.assertFalse((root / 'batch').exists())
            config.pop('model')
            config['settings']['n'] = 5
            with patch.dict('os.environ', environment, clear=True), self.assertRaises(ValueError):
                campaign.run_batch(config, root / 'prepared', root / 'batch')
            self.assertFalse((root / 'batch').exists())

    def test_bulk_runtime_loads_once_and_status_never_reads_records(self):
        from scripts.rc_evaluation.dail_sql import campaign
        self.assertTrue(hasattr(campaign, 'run_batch'), 'public campaign missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config, tasks, pool = prepared_fixture(root / 'prepared')
            environment = {'DASH_MODELS': 'model', 'DASH_API_KEY': 'secret', 'DASH_BASE_URL': 'https://example.invalid'}
            loads = []
            def group_loader(path, group):
                loads.append(('group', group))
                return tasks[group]
            def pool_loader(path, name):
                loads.append(('pool', name))
                return pool
            def factory(task, **kwargs):
                loads.append(('factory', task['task']['group'], task['task']['question_id']))
                self.assertEqual(kwargs['first_messages'], [{'role': 'user', 'content': 'SELECT '}])
                self.assertEqual(kwargs['distance_ids'], [str(i) for i in range(9)])
                async def paused(*args, **kw):
                    raise CompositePaused('offline')
                return paused
            with patch.dict('os.environ', environment, clear=True), \
                 patch.object(campaign, 'load_prepared_group', group_loader), \
                 patch.object(campaign, 'load_training_pool', pool_loader), \
                 patch.object(campaign, 'make_round_runner', factory), \
                 patch.object(campaign.RequestDispatcher, 'make_client', return_value=object()), \
                 patch.object(campaign, '_read_json', wraps=campaign._read_json) as reads:
                result = campaign.run_batch(config, root / 'prepared', root / 'batch')
                self.assertEqual(result['status'], 'paused')
                self.assertEqual(loads.count(('pool', 'pool')), 1)
                self.assertEqual(loads.count(('group', 'a')), 1)
                self.assertEqual(loads.count(('group', 'b')), 1)
                read_paths = [str(call.args[0]) for call in reads.call_args_list]
                self.assertEqual(read_paths.count(str(root / 'prepared' / 'identity.json')), 1)
                self.assertEqual(sum(path.endswith('prompt.json') for path in read_paths), 2)
                manifest = json.loads((root / 'batch' / 'manifest.json').read_text())
                self.assertNotIn('secret', json.dumps(manifest))
                self.assertEqual(manifest['resources']['request_limit'], 20)
                self.assertEqual(manifest['resources']['sql_workers'], 2)
                self.assertIn('question_queue', result, 'compact queue statistics missing')
                self.assertEqual(result['question_queue']['paused'], 2)
                self.assertEqual(result['question_queue']['unstarted'], 2)
                before = json.loads((root / 'batch' / 'manifest.json').read_text())
                versions_before = None
                with CurrentIndex(root / 'batch' / 'current.sqlite3', read_only=True) as index:
                    versions_before = [index.assignment(TaskKey('batch', 'a', q))['version'] for q in ('0', '1')]
                campaign.resume_batch(root / 'batch')
                with CurrentIndex(root / 'batch' / 'current.sqlite3', read_only=True) as index:
                    self.assertEqual([index.assignment(TaskKey('batch', 'a', q))['version'] for q in ('0', '1')], versions_before)
                with patch.dict('os.environ', {'DASH_MODELS': 'changed'}), self.assertRaises(ValueError):
                    campaign.resume_batch(root / 'batch')
                original_hash = campaign._hash
                def changed_upstream(path):
                    return 'changed' if str(path).endswith('DAIL-SQL/utils/post_process.py') else original_hash(path)
                with patch.object(campaign, '_hash', changed_upstream), self.assertRaises(ValueError):
                    campaign.resume_batch(root / 'batch')
                pool_marker = root / 'prepared' / 'pools' / 'pool' / 'ready.json'
                original_marker = pool_marker.read_text()
                pool_marker.write_text('{"changed":true}')
                with self.assertRaises(ValueError):
                    campaign.resume_batch(root / 'batch')
                pool_marker.write_text(original_marker)
                self.assertEqual(json.loads((root / 'batch' / 'manifest.json').read_text()), before)
                with patch.object(campaign, 'DailRecords', side_effect=AssertionError('large record read')):
                    for _ in range(3):
                        self.assertEqual(campaign.status(root / 'batch')['expected_questions'], 4)
                config['model'] = 'other'
                with self.assertRaises(ValueError):
                    campaign.run_batch(config, root / 'prepared', root / 'batch')


def prepared_fixture(root):
    root.mkdir()
    settings = asdict(DailSettings())
    config = {'format': 'dail-sql-inputs-v1', 'settings': settings,
              'groups': [{'name': group, 'training_pool': 'pool', 'compute_cv_link': False,
                          'expected_count': 2} for group in ('a', 'b')], 'training_pools': {'pool': {}},
              'resources': {'request_limit': 20, 'http_connections': 20, 'sql_workers': 2}}
    groups = {g: {'ids': ['0', '1'], 'count': 2, 'training_pool': 'pool',
                  'compute_cv_link': False, 'sources': {}} for g in ('a', 'b')}
    identity = {'manifest': {'settings': settings, 'group_order': ['a', 'b'], 'groups': groups,
                            'training_pools': {'pool': {}}}}
    (root / 'identity.json').write_text(json.dumps(identity))
    (root / 'plan.json').write_text(json.dumps({'cache_key': 'fixture'}))
    (root / 'evaluation').mkdir()
    (root / 'evaluation' / 'source.json').write_text('{}')
    (root / 'schemas.ready.json').write_text('{}')
    tasks = {}
    for group in groups:
        path = root / 'groups' / group
        path.mkdir(parents=True)
        (path / 'ready.json').write_text('{}')
        tasks[group] = {}
        for q in groups[group]['ids']:
            target = path / q
            target.mkdir()
            (target / 'prompt.json').write_text(json.dumps([{'role': 'user', 'content': 'SELECT '}]))
            np.save(target / 'order.npy', np.arange(9))
            tasks[group][q] = {'task': {'group': group, 'question_id': q, 'training_pool': 'pool',
                'rc3_ref': {'content': {}}}, 'first_example_ids': [str(i) for i in range(9)],
                'first_prompt_ref': str((target / 'prompt.json').relative_to(root)),
                'distance_order_ref': str((target / 'order.npy').relative_to(root))}
    path = root / 'pools' / 'pool'
    path.mkdir(parents=True)
    (path / 'ready.json').write_text('{}')
    (path / 'ids.json').write_text(json.dumps([str(i) for i in range(9)]))
    return config, tasks, [{'example_id': str(i), 'query_skeleton': 'select _'} for i in range(9)]


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    async def test_crash_sent_attempt_becomes_interrupted_without_budget_reset(self):
        from scripts.rc_evaluation.dail_sql import campaign
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'batch_id': 'batch', 'group_order': ['a'], 'groups': {'a': {'ids': ['0']}}}
            key = TaskKey('batch', 'a', '0')
            with DailRecords(root, manifest) as records, CurrentIndex(root / 'current.sqlite3') as index:
                version = records.begin_version(key)
                index.claim_assignment(key, version)
                attempt = records.append(version, 'request_attempt', {'round_execution_id': 'A', 'sample_position': 0, 'attempt_no': 1})
                async def resume(key, resumed, notify):
                    self.assertEqual(resumed, version)
                    history = records.request_history(version, 'A')[0]['attempts']
                    self.assertEqual(len(history), 1)
                    self.assertIsNotNone(history[0]['request_result_id'], 'unanswered send not reconciled')
                    outcome = records.get_event(version, history[0]['request_result_id'])
                    self.assertEqual(outcome['error']['category'], 'interrupted')
                    self.assertIsNone(outcome['usage'])
                    self.assertEqual(outcome['request_attempt_id'], attempt)
                    raise CompositePaused('remaining budget can resume')
                await campaign._drive(manifest, records, index, resume, {'a'})

    async def test_cancel_drains_children_and_leaves_resumable_versions(self):
        from scripts.rc_evaluation.dail_sql import campaign
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'batch_id': 'batch', 'group_order': ['a'], 'groups': {'a': {'ids': ['0', '1']}}}
            started = asyncio.Event()
            drained = []
            with DailRecords(root, manifest) as records, CurrentIndex(root / 'current.sqlite3') as index:
                async def running(key, version, notify):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        drained.append(key)
                owner = asyncio.create_task(campaign._drive(manifest, records, index, running, {'a'}))
                await started.wait()
                owner.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await owner
                self.assertEqual(len(drained), 2)
                for q in ('0', '1'):
                    assignment = index.assignment(TaskKey('batch', 'a', q))
                    self.assertEqual(assignment['state'], 'paused')
                    self.assertIsNotNone(assignment['lease'])
                self.assertTrue(all(v is None for v in index.snapshot().values()))

    async def test_five_group_overlap_failure_counts_and_exact_resume(self):
        from scripts.rc_evaluation.dail_sql import campaign
        self.assertTrue(hasattr(campaign, '_drive'), 'campaign scheduler missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'batch_id': 'batch', 'group_order': ['a', 'b', 'c', 'd', 'e'],
                        'groups': {g: {'ids': [str(i) for i in range(5)]} for g in 'abcde'}}
            started = []
            release = asyncio.Event()
            calls = []
            with DailRecords(root, manifest) as records, CurrentIndex(root / 'current.sqlite3') as index:
                async def run(key, version, notify):
                    started.append((key.group, key.question_id))
                    if key.group == 'b':
                        self.assertNotEqual(index.assignment(TaskKey('batch', 'a', '4'))['state'], 'done')
                        release.set()
                    if key.group == 'a' and key.question_id == '4':
                        await release.wait()
                    async def round_(number, rc, ids, **kwargs):
                        rid = 'B' if rc else 'A'
                        previous = records.find_source(version, 'round_result', rid)
                        if previous:
                            return previous['payload']
                        calls.append((version, rid))
                        payload = {'round_execution_id': rid, 'round_no': 1, 'rc_injected': rc,
                            'actual_parent_round_id': None, 'example_ids': ids, 'status': 'failed',
                            'samples': [{'sample_position': p, 'status': 'failed', 'request_attempt_ids': [],
                                'successful_request_id': None, 'success_usage': None, 'error': {'category': 'local'}} for p in range(5)],
                            'request_attempt_ids': [], 'successful_request_ids': [], 'success_usage': None,
                            'candidates': [], 'selection': None, 'next_example_ids': [], 'error': {'category': 'local'}}
                        records.append(version, 'round_result', payload)
                        return payload
                    return await run_composite({'first_example_ids': [str(i) for i in range(9)]},
                        version_id=version, run_round=round_, records=records, on_mode_result=notify)
                await asyncio.wait_for(campaign._drive(manifest, records, index, run, set('abcd')), 4)
                self.assertEqual(len(started), 20)
                self.assertEqual(index.campaign_counts('batch', 'e')['waiting'], 'preparation_not_ready')
                self.assertFalse(index.group_started('batch', 'e'))
                self.assertFalse((root / 'group-65').exists())
                await campaign._drive(manifest, records, index, run, set('abcde'))
                self.assertEqual(len(started), 25)
                self.assertEqual(len(index.snapshot()), 25)
                self.assertEqual(index.campaign_counts('batch', 'a')['failed']['native'], 5)
                before = index.snapshot()
                await campaign._drive(manifest, records, index, run, set('abcde'))
                self.assertEqual(len(calls), 50)
                target = TaskKey('batch', 'c', '2')
                await campaign._drive(manifest, records, index, run, set('abcde'), [target])
                self.assertNotEqual(index.current(target), before[target])
                self.assertEqual({k: v for k, v in index.snapshot().items() if k != target},
                                 {k: v for k, v in before.items() if k != target})
                self.assertEqual(index.campaign_counts('batch', 'c')['terminal']['native'], 5)
                self.assertEqual(len(calls), 52)
                with patch.object(index, 'finish_assignment', side_effect=RuntimeError('crash after seal')):
                    with self.assertRaises(RuntimeError):
                        await campaign._drive(manifest, records, index, run, set('abcde'), [target])
                sealed = index.assignment(target)['version']
                self.assertTrue(records.is_sealed(target, sealed))
                self.assertNotEqual(index.current(target), sealed)
                prior_calls = len(calls)
                await campaign._drive(manifest, records, index, run, set('abcde'))
                self.assertEqual(index.current(target), sealed)
                self.assertEqual(len(calls), prior_calls)

    async def test_pause_resume_retains_assignment_and_waiting_group(self):
        from scripts.rc_evaluation.dail_sql import campaign
        self.assertTrue(hasattr(campaign, '_drive'), 'campaign scheduler missing')
        with tempfile.TemporaryDirectory() as directory:
            manifest = {'batch_id': 'batch', 'group_order': ['a', 'b'],
                        'groups': {'a': {'ids': ['0']}, 'b': {'ids': ['0']}}}
            root = Path(directory)
            with DailRecords(root, manifest) as records, CurrentIndex(root / 'current.sqlite3') as index:
                versions = []
                async def pause(key, version, notify):
                    versions.append(version)
                    self.assertEqual(index.assignment(key)['state'], 'active')
                    raise CompositePaused('test interruption')
                await campaign._drive(manifest, records, index, pause, {'a'})
                key = TaskKey('batch', 'a', '0')
                token = index.assignment(key)['lease']['token']
                await campaign._drive(manifest, records, index, pause, {'a'})
                self.assertEqual(versions[0], versions[1])
                self.assertNotEqual(token, index.assignment(key)['lease']['token'])
                self.assertFalse(index.group_started('batch', 'b'))
                self.assertIsNone(index.current(key))
