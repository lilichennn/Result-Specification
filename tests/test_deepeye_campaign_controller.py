"""Recovery and supervisor boundaries use real stores and real OS processes."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tests.test_deepeye_run_pipeline import Factory, FakeTrace, item
from scripts.baseline_adapters.deepeye.run_pipeline import run_pipeline
from scripts.baseline_adapters.deepeye.run_store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def fixture_tasks(config):
    return [(key.split('/')[0], item(key.split('/')[1])) for key in config['items']]


def zero_factory(failure=None):
    from scripts.baseline_adapters.deepeye.run_pipeline import STAGE_METHODS
    from tests.test_deepeye_run_pipeline import cost
    base = Factory(failure)
    def factory(stage, items):
        runner = base(stage, items)
        method = STAGE_METHODS[stage]
        original = getattr(runner, method)
        def execute(target):
            original(target)
            setattr(target, stage + '_llm_cost', cost())
            target.total_llm_cost = cost()
        setattr(runner, method, execute)
        return runner
    return factory


def fixture_commands(config, job):
    prefix = [config['python'], '-E', '-B', '-c',
              'import sys;sys.path.insert(0,sys.argv[1]);from tests.test_deepeye_campaign_controller import fixture_cli;sys.exit(fixture_cli(sys.argv[2],sys.argv[3],sys.argv[4]))',
              str(ROOT), str(Path(job['run_dir']).parents[1]), job['job_id']]
    return prefix + ['prepare'], prefix + ['resume']


def fixture_worker(path, job_id, token):
    from scripts.rc_evaluation.deepeye.campaign import supervisor
    with patch.object(supervisor, 'validate_config', side_effect=fixture_tasks), patch.object(supervisor, 'commands', side_effect=fixture_commands):
        return supervisor.run_worker(path, job_id, token)


def fixture_worker_command(config, path, job_id, token):
    return [config['python'], '-E', '-B', '-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from tests.test_deepeye_campaign_controller import fixture_worker;sys.exit(fixture_worker(*sys.argv[2:]))',
            str(ROOT), str(path), job_id, token]


def fixture_cli(path, job_id, command):
    from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
    from scripts.rc_evaluation.deepeye.campaign.processes import read_json
    from scripts import deepeye_bird_interact_run as native
    from scripts.rc_evaluation.deepeye.tests.test_cli import ENV
    from scripts.rc_evaluation.deepeye.source import snapshot_source
    from scripts.rc_evaluation.deepeye.runner import run_experiment, unfinished_keys
    from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, select_unfinished
    from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
    with CampaignLedger.open(path, read_only=True) as ledger:
        config = ledger.config
        job = next(j for j in ledger.jobs() if j['job_id'] == job_id)
    identity = read_json(Path(path) / 'jobs' / (job_id + '.json'))
    assert identity['token'] == job['process']['token'] and identity['identity']
    tasks = fixture_tasks({'items': job['items']})
    if command == 'prepare':
        delay = config.get('fixture_prepare_delay', 0)
        time.sleep(delay)
        if job['kind'] != 'rc':
            args = native._build_parser().parse_args(['prepare', '--run-dir', job['run_dir'], '--precompute-dir', '/tmp/offline'])
            manifest = native.build_manifest(native.build_effective_config(ENV, args), {},
                                            [{'task_key': key, 'database_id': 'db'} for key in job['items']])
        else:
            snapshot = snapshot_source(Path(job['source_run']), tasks, job['target_stage'])
            manifest = {'format': 'deepeye-rc-evaluation-run-v1', 'target_stage': job['target_stage'],
                        'condition': 'rc', 'repeat_id': '1', 'continue_downstream': False,
                        'effective_config': snapshot['source_manifest']['effective_config'], 'sources': {},
                        'contracts': {key: {'task_key': key, 'db_id': 'db', 'question': 'Return x', 'evidence': '',
                            'round2': {field: 'Meaning' for field in ('population', 'row_grain', 'column_role', 'derivation', 'filter_policy', 'meta_review')}}
                                      for key in job['items']}, **snapshot}
        with RunStore.create(Path(job['run_dir']), manifest):
            pass
        return 0
    with RunStore.open(Path(job['run_dir'])) as store:
        if job['kind'] == 'rc':
            time.sleep(config.get('fixture_delay', 0))
            return int(bool(run_experiment(store, lambda *a: (_ for _ in ()).throw(AssertionError('No model for replay')),
                                           FakeTrace(), item_keys=unfinished_keys(store))['failed']))
        tasks, _ = select_unfinished(store, tasks)
        recorder = FakeTrace()
        recorder.stop_event = threading.Event()
        @contextlib.contextmanager
        def graceful():
            old = signal.signal(signal.SIGTERM, lambda *a: recorder.stop_event.set())
            try:
                yield
            finally:
                signal.signal(signal.SIGTERM, old)
        with graceful():
            results = []
            for variant, current in tasks:
                if recorder.stop_event.is_set():
                    break
                time.sleep(config.get('fixture_delay', 0))
                fail = current.instance_id in config.get('fixture_permanent', []) or (
                    current.instance_id in config.get('fixture_retry', []) and job['kind'] == 'native_first')
                results.append(run_pipeline(store, [(variant, current)], zero_factory(('sql_generation', current.instance_id) if fail else None), recorder))
        return int(any(result['failed'] for result in results))


class RecoveryTests(unittest.TestCase):
    def test_rc_filter_keeps_failed_target_terminal_and_allows_empty_selection(self):
        from scripts.rc_evaluation.deepeye import runner
        self.assertTrue(hasattr(runner, 'unfinished_keys'), 'RC unfinished-only recovery is required')
        from scripts.rc_evaluation.deepeye.tests.test_runner import experiment_manifest, Factory as RCFactory, OfflineTrace
        from scripts.rc_evaluation.deepeye.tests.test_source import make_source
        from scripts.rc_evaluation.deepeye.source import snapshot_source
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            tasks = make_source(source, calls=1)
            manifest = experiment_manifest(snapshot_source(source, tasks, 'sql_revision'))
            with RunStore.create(Path(directory) / 'run', manifest) as store:
                runner.run_experiment(store, RCFactory('sql_revision'), OfflineTrace())
                before = store.attempts()
                keys = runner.unfinished_keys(store)
                self.assertEqual(keys, [])
                runner.run_experiment(store, lambda *args: self.fail('terminal failure rerun'), OfflineTrace(), item_keys=keys)
                self.assertEqual(store.attempts(), before)

    def test_native_resume_checks_full_manifest_before_filtering(self):
        import scripts.deepeye_bird_interact_run as cli
        parser = cli._build_parser()
        try:
            args = parser.parse_args(['resume', '--run-dir', '/tmp/run', '--precompute-dir', '/tmp/pre', '--unfinished-only'])
        except SystemExit:
            self.fail('Native resume must accept unfinished-only')
        self.assertTrue(args.unfinished_only)
        self.assertEqual((args.request_limit, args.request_workers, args.coordinator_workers,
                          args.http_connections, args.start_rate, args.pg_concurrency), (8000, 8000, 6000, 8000, 50, 10))

    def selector(self):
        module = importlib.import_module('scripts.baseline_adapters.deepeye.run_pipeline')
        self.assertTrue(hasattr(module, 'select_unfinished'), 'Native unfinished-only recovery is required')
        return module.select_unfinished

    def test_terminal_failures_and_successes_are_not_executed_again(self):
        select = self.selector()
        with tempfile.TemporaryDirectory() as directory, RunStore.create(Path(directory) / 'run', {}) as store:
            tasks = [('lite', item(k)) for k in 'abc']
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, tasks[:2], Factory(('sql_generation', 'b')), FakeTrace())
            before = store.attempts()
            selected, report = select(store, tasks)
            self.assertEqual([x.instance_id for _, x in selected], ['c'])
            self.assertEqual(report['terminal'], {'lite/a': 'succeeded', 'lite/b': 'failed'})
            self.assertEqual(store.attempts(), before)
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, selected, Factory(), FakeTrace())
            self.assertEqual(sum(a['item_key'] == 'lite/b' for a in store.attempts()), 3)

    def test_crash_after_success_or_failure_stage_seals_master_without_new_attempt(self):
        select = self.selector()
        for failure in (None, ('sql_generation', 'a')):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                with RunStore.create(Path(directory) / 'run', {}) as store:
                    finish = store.finish_attempt
                    def crash(master, status, payload):
                        if store.attempt(master)['stage'] == 'pipeline':
                            raise RuntimeError('crash before master finish')
                        return finish(master, status, payload)
                    with patch.object(store, 'finish_attempt', side_effect=crash), self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                        run_pipeline(store, [('lite', item('a'))], Factory(failure), FakeTrace())
                    before = len(store.attempts())
                    selected, report = select(store, [('lite', item('a'))])
                    self.assertEqual(selected, [])
                    self.assertEqual(len(store.attempts()), before)
                    self.assertEqual(store.attempts()[0]['status'], 'failed' if failure else 'succeeded')
                    self.assertEqual(len(report['sealed']), 1)
                    self.assertTrue(any(e['kind'] == 'pipeline_recovery_seal' for e in store.events()))
                    self.assertEqual(select(store, [('lite', item('a'))])[1]['sealed'], [])

    def test_earlier_failure_followed_by_valid_success_is_success(self):
        select = self.selector()
        with tempfile.TemporaryDirectory() as directory, RunStore.create(Path(directory) / 'run', {}) as store:
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, [('lite', item('a'))], Factory(('sql_generation', 'a')), FakeTrace())
                run_pipeline(store, [('lite', item('a'))], Factory(), FakeTrace())
            self.assertEqual(select(store, [('lite', item('a'))])[1]['terminal'], {'lite/a': 'succeeded'})

    def test_interrupted_prefix_resumes_only_remaining_stages(self):
        select = self.selector()
        with tempfile.TemporaryDirectory() as directory, RunStore.create(Path(directory) / 'run', {}) as store:
            finish = store.finish_attempt
            def crash(attempt, status, payload):
                finish(attempt, status, payload)
                if store.attempt(attempt)['stage'] == 'schema_linking':
                    raise RuntimeError('power loss')
            with patch.object(store, 'finish_attempt', side_effect=crash), self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, [('lite', item('a'))], Factory(), FakeTrace())
            selected, _ = select(store, [('lite', item('a'))])
            factory = Factory()
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, selected, factory, FakeTrace())
            self.assertEqual(factory.calls, [('sql_generation', 'a'), ('sql_revision', 'a'), ('sql_selection', 'a')])


class ProcessTests(unittest.TestCase):
    def modules(self):
        try:
            return importlib.import_module('scripts.rc_evaluation.deepeye.campaign.processes')
        except ModuleNotFoundError as exc:
            self.fail(f'Recoverable process ownership is required: {exc}')

    def test_exclusive_lock_survives_parent_process_boundary(self):
        module = self.modules()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner.lock'
            with module.exclusive_lock(path):
                child = subprocess.run([sys.executable, '-E', '-B', '-c',
                    'import sys;sys.path.insert(0,sys.argv[1]);from scripts.rc_evaluation.deepeye.campaign.processes import exclusive_lock;\nwith exclusive_lock(sys.argv[2]): pass',
                    str(ROOT), str(path)], cwd='/', capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0)
            with module.exclusive_lock(path):
                pass

    def test_command_recipes_bind_exact_members_source_and_safe_resume(self):
        module = self.modules()
        config = {'python': sys.executable, 'code_root': str(ROOT), 'env_file': '/tmp/env',
                  'rc_lite': '/tmp/lite', 'rc_full': '/tmp/full',
                  'native_args': ['--precompute-dir', '/tmp/pre', '--few-shot-source', '/tmp/train', '--env-file', '/tmp/env']}
        job = {'kind': 'native_first', 'run_dir': '/tmp/run', 'items': ['lite/a', 'full/b']}
        prepare, execute = module.commands(config, job)
        self.assertEqual(execute[4], 'resume')
        self.assertIn('--unfinished-only', execute)
        self.assertEqual(execute[-5:], ['--item', 'lite/a', '--item', 'full/b', '--unfinished-only'])
        job.update(kind='rc', source_run='/tmp/source', target_stage='sql_selection')
        prepare, execute = module.commands(config, job)
        self.assertEqual(prepare[prepare.index('--source-run') + 1], '/tmp/source')
        self.assertEqual(prepare[prepare.index('--condition') + 1], 'rc')
        self.assertEqual(execute[-1], '--unfinished-only')
        for argv in (prepare, execute):
            for forbidden in ('none', '--continue-downstream', '--renew-samples', '--inherit-from'):
                self.assertNotIn(forbidden, argv)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('scripts.rc_evaluation.deepeye.campaign.controller')
        except ModuleNotFoundError as exc:
            self.fail(f'Campaign controller is required: {exc}')
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'campaign'
        self.config = dict(items=['lite/a'], native_args=[], tail_fraction=.8, poll_seconds=.05,
                           python=sys.executable, code_root=str(ROOT), env_file='/tmp/env', rc_lite='/tmp/lite', rc_full='/tmp/full')
        self.ledger = CampaignLedger.create(self.path, self.config)
        self.addCleanup(self.ledger.close)
        (self.path / 'jobs').mkdir()
        (self.path / 'logs').mkdir()

    def wait_until(self, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.05)
        self.fail('Timed out awaiting real subprocess progress')

    def spawn(self, job):
        with patch.object(self.module, 'worker_command', side_effect=fixture_worker_command):
            child = self.module.launch(self.ledger, job)
        self.addCleanup(lambda: child.wait(timeout=15) if child.poll() is None else None)
        self.addCleanup(lambda: self.module.pause(self.path) if child.poll() is None else None)
        return child

    def test_real_supervisor_owns_launch_before_ack_and_completed_resume_does_zero_work(self):
        self.assertTrue((ROOT / 'scripts/rc_evaluation/deepeye/campaign/supervisor.py').exists(), 'Supervisor is required')
        self.module.set_control(self.ledger, 'running')
        first = self.ledger.jobs()[0]
        child = self.spawn(first)
        # Parent acknowledgement is deliberately absent; only worker identity
        # and the persisted pre-spawn token can attach this launched process.
        self.assertNotIn('pid', self.ledger.jobs()[0]['process'])
        self.wait_until(lambda: child.poll() is not None)
        self.assertEqual(child.returncode, 0, '\n'.join(p.read_text() for p in (self.path / 'logs').glob('*')))
        self.assertEqual(self.module.reconcile(self.ledger)[first['job_id']], 'finished')
        with RunStore.open(Path(first['run_dir']), read_only=True) as store:
            self.assertEqual(len(store.attempts()), 5)
        self.ledger.update_job(first['job_id'], state='prepared', process={})
        second = self.spawn(self.ledger.jobs()[0])
        self.wait_until(lambda: second.poll() is not None)
        self.assertEqual(second.returncode, 0)
        with RunStore.open(Path(first['run_dir']), read_only=True) as store:
            self.assertEqual(len(store.attempts()), 5)

    def test_duplicate_supervisor_cannot_make_another_attempt(self):
        self.assertTrue((ROOT / 'scripts/rc_evaluation/deepeye/campaign/supervisor.py').exists(), 'Supervisor is required')
        from scripts.rc_evaluation.deepeye.campaign.processes import lock_held
        self.module.set_control(self.ledger, 'running')
        child = self.spawn(self.ledger.jobs()[0])
        job = self.ledger.jobs()[0]
        lock, _ = self.module.job_paths(self.path, job)
        self.wait_until(lambda: lock_held(lock))
        duplicate = subprocess.run(fixture_worker_command(self.config, self.path, job['job_id'], job['process']['token']),
                                   cwd='/', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertNotEqual(duplicate.returncode, 0)
        self.wait_until(lambda: child.poll() is not None)
        with RunStore.open(Path(job['run_dir']), read_only=True) as store:
            self.assertEqual(len(store.attempts()), 5)

    def test_monitoring_is_explicit_about_unobservable_http_occupancy(self):
        try:
            from scripts.rc_evaluation.deepeye.campaign.monitoring import snapshot
        except ImportError as exc:
            self.fail(f'Campaign monitoring is required: {exc}')
        from scripts.rc_evaluation.deepeye.campaign.planning import plan_tick
        first = self.ledger.jobs()[0]
        with RunStore.create(Path(first['run_dir']), {'items': [{'task_key': 'lite/a'}]}) as store:
            attempt = store.begin_attempt('lite/a', 'pipeline', 'pending')
            store.append_event(attempt, 'api_request', {'call_id': 'logical-one'})
            self.ledger.update_job(first['job_id'], state='prepared')
            result = plan_tick(self.ledger, {first['job_id']: self.module.observe(first)})
            status = snapshot(self.ledger, result, {})
        self.assertIsNone(status['resources']['http_in_flight'])
        self.assertEqual(status['resources']['logical_api_outstanding'], 1)
        self.assertIn('not HTTP', status['resources']['occupancy_definition'])

    def test_auth_error_blocks_even_if_native_fallback_finishes_successfully(self):
        self.assertTrue(hasattr(self.module, 'check_run_faults'), 'System faults must pause new launches')
        job = self.ledger.jobs()[0]
        with RunStore.create(Path(job['run_dir']), {'items': [{'task_key': 'lite/a'}]}) as store, contextlib.redirect_stdout(io.StringIO()):
            finish = store.finish_attempt
            def auth_then_finish(attempt, status, payload):
                if store.attempt(attempt)['stage'] == 'schema_linking':
                    store.append_event(attempt, 'api_error', {'call_id': 'auth', 'error': {'type': {'qualname': 'AuthenticationError'}, 'status_code': 401}})
                return finish(attempt, status, payload)
            with patch.object(store, 'finish_attempt', side_effect=auth_then_finish):
                run_pipeline(store, [('lite', item('a'))], Factory(), FakeTrace())
        with self.assertRaises(RuntimeError):
            self.module.check_run_faults(self.ledger, job)
        self.assertEqual(self.module.control(self.path)['mode'], 'blocked')

    def test_full_real_subprocess_campaign_completes_retry_exclusion_and_four_rc_targets(self):
        # No controller/supervisor/store/planning mocks: only native/PG execution
        # and source loading are replaced in child fixtures.
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        second_path = Path(self.temp.name) / 'end_to_end'
        config = {**self.config, 'items': ['lite/a', 'lite/b', 'lite/c'],
                  'fixture_retry': ['a'], 'fixture_permanent': ['b']}
        with CampaignLedger.create(second_path, config):
            pass
        (second_path / 'jobs').mkdir()
        (second_path / 'logs').mkdir()
        with patch.object(self.module, 'validate_config', side_effect=fixture_tasks), \
                patch.object(self.module, 'worker_command', side_effect=fixture_worker_command):
            result = self.module.run(second_path)
        self.assertEqual(result['mode'], 'complete', '\n'.join(p.read_text() for p in (second_path / 'logs').glob('*')))
        self.assertEqual(result['excluded'], ['lite/b'])
        self.assertEqual(set(result['canonical_sources']), {'lite/a', 'lite/c'})
        self.assertEqual(len([j for j in result['jobs'] if j['kind'] == 'native_retry']), 1)
        for stage in ('schema_linking', 'sql_generation', 'sql_revision', 'sql_selection'):
            self.assertEqual(result['rc_counts'][stage]['succeeded'], 2)
        count = len(result['jobs'])
        with patch.object(self.module, 'validate_config', side_effect=fixture_tasks):
            resumed = self.module.run(second_path, resume=True)
        self.assertEqual(len(resumed['jobs']), count)

    def test_parent_dies_after_spawn_before_ack_supervisor_continues(self):
        parent_code = '''
import os,sys
sys.path.insert(0,sys.argv[1])
from unittest.mock import patch
from scripts.rc_evaluation.deepeye.campaign import controller
from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
from tests.test_deepeye_campaign_controller import fixture_worker_command
with CampaignLedger.open(sys.argv[2]) as ledger:
    controller.set_control(ledger,'running')
    append = ledger.append_event
    def crash(kind,payload):
        if kind == 'supervisor_spawned': os._exit(17)
        append(kind,payload)
    with patch.object(controller,'worker_command',side_effect=fixture_worker_command), patch.object(ledger,'append_event',side_effect=crash):
        controller.launch(ledger,ledger.jobs()[0])
'''
        parent = subprocess.run([sys.executable, '-E', '-B', '-c', parent_code, str(ROOT), str(self.path)],
                                cwd='/', capture_output=True, text=True)
        self.assertEqual(parent.returncode, 17, parent.stderr)
        job = self.ledger.jobs()[0]
        self.assertTrue(job['process']['token'])
        self.assertNotIn('pid', job['process'])
        self.wait_until(lambda: self.module.reconcile(self.ledger)[job['job_id']] == 'finished')
        with RunStore.open(Path(job['run_dir']), read_only=True) as store:
            self.assertEqual(len(store.attempts()), 5)

    def test_pause_during_prepare_stops_before_any_native_attempt(self):
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        from scripts.rc_evaluation.deepeye.campaign.processes import read_json
        path = Path(self.temp.name) / 'preparing'
        with CampaignLedger.create(path, {**self.config, 'fixture_prepare_delay': 2}) as ledger:
            (path / 'jobs').mkdir()
            (path / 'logs').mkdir()
            self.module.set_control(ledger, 'running')
            with patch.object(self.module, 'worker_command', side_effect=fixture_worker_command):
                child = self.module.launch(ledger, ledger.jobs()[0])
            job = ledger.jobs()[0]
            _, identity = self.module.job_paths(path, job)
            self.wait_until(lambda: (read_json(identity) or {}).get('phase') == 'preparing')
            self.module.pause(path)
            child.wait(timeout=15)
            self.assertEqual(child.returncode, 0)
            self.assertEqual(read_json(identity)['phase'], 'paused')
            with RunStore.open(Path(job['run_dir']), read_only=True) as store:
                self.assertEqual(store.attempts(), [])

    def test_partial_cohort_pause_and_resume_never_reruns_terminal_failure(self):
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        path = Path(self.temp.name) / 'partial'
        config = {**self.config, 'items': ['lite/a', 'lite/c'], 'fixture_permanent': ['a'], 'fixture_delay': 1}
        with CampaignLedger.create(path, config) as ledger:
            (path / 'jobs').mkdir()
            (path / 'logs').mkdir()
            self.module.set_control(ledger, 'running')
            with patch.object(self.module, 'worker_command', side_effect=fixture_worker_command):
                child = self.module.launch(ledger, ledger.jobs()[0])
            job = ledger.jobs()[0]
            def failed():
                if not (Path(job['run_dir']) / 'run.sqlite3').exists():
                    return False
                return self.module.observe(job)['states']['lite/a']['status'] == 'failed'
            self.wait_until(failed)
            self.module.pause(path)
            child.wait(timeout=15)
            self.assertEqual(child.returncode, 0)
            self.module.reconcile(ledger)
            with RunStore.open(Path(job['run_dir']), read_only=True) as store:
                before = len([a for a in store.attempts() if a['item_key'] == 'lite/a'])
            with patch.object(self.module, 'validate_config', side_effect=fixture_tasks), \
                    patch.object(self.module, 'worker_command', side_effect=fixture_worker_command):
                result = self.module.run(path, resume=True)
            self.assertEqual(result['mode'], 'complete')
            with RunStore.open(Path(job['run_dir']), read_only=True) as store:
                self.assertEqual(len([a for a in store.attempts() if a['item_key'] == 'lite/a']), before)

    def test_five_real_jobs_overlap_independently_with_fixed_anchors(self):
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        from scripts.rc_evaluation.deepeye.campaign.planning import plan_tick
        from scripts.rc_evaluation.deepeye.campaign.processes import atomic_json, read_json
        from scripts.rc_evaluation.deepeye.runner import run_experiment
        from scripts import deepeye_bird_interact_run as native
        from scripts.rc_evaluation.deepeye.tests.test_cli import ENV
        path = Path(self.temp.name) / 'overlap'
        config = {**self.config, 'items': ['lite/' + k for k in 'abcdef'], 'fixture_delay': 5}
        with CampaignLedger.create(path, config) as ledger:
            (path / 'jobs').mkdir()
            (path / 'logs').mkdir()
            first = ledger.jobs()[0]
            args = native._build_parser().parse_args(['prepare', '--run-dir', first['run_dir'], '--precompute-dir', '/tmp/pre'])
            manifest = native.build_manifest(native.build_effective_config(ENV, args), {},
                                            [{'task_key': key, 'database_id': 'db'} for key in config['items']])
            with RunStore.create(Path(first['run_dir']), manifest) as store, contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, fixture_tasks({'items': config['items'][:5]}), zero_factory(), FakeTrace())
            ledger.update_job(first['job_id'], state='prepared')
            result = plan_tick(ledger, {first['job_id']: self.module.observe(first)})
            anchor_jobs = []
            for stage in ('schema_linking', 'sql_generation', 'sql_revision', 'sql_selection'):
                job = next(j for j in result['jobs'] if j['target_stage'] == stage)
                ledger.update_job(job['job_id'], state='prepared', process={'token': 'seed'})
                _, identity = self.module.job_paths(path, job)
                atomic_json(identity, {'token': 'seed', 'identity': 'fixture_preparation'})
                fixture_cli(str(path), job['job_id'], 'prepare')
                with RunStore.open(Path(job['run_dir'])) as store:
                    run_experiment(store, lambda *a: self.fail('No model expected'), FakeTrace(), item_keys=config['items'][:4])
                ledger.update_job(job['job_id'], state='prepared', process={})
                anchor_jobs.append(job['job_id'])
                result = plan_tick(ledger, {job['job_id']: self.module.observe(job)})
            self.module.set_control(ledger, 'running')
            with patch.object(self.module, 'worker_command', side_effect=fixture_worker_command):
                children = [self.module.launch(ledger, job) for job in ledger.jobs()]
            try:
                def overlapping():
                    records = [read_json(self.module.job_paths(path, j)[1]) or {} for j in ledger.jobs()]
                    return all(record.get('phase') == 'running' for record in records)
                self.wait_until(overlapping)
                self.assertEqual(len(children), 5)
                self.assertEqual(len({child.pid for child in children}), 5)
                for stage, anchor in zip(('schema_linking', 'sql_generation', 'sql_revision', 'sql_selection'), anchor_jobs):
                    self.assertEqual(ledger.anchor(stage), [anchor])
            finally:
                for child in children:
                    child.wait(timeout=20)
            self.assertTrue(all(child.returncode == 0 for child in children))
            self.assertTrue(all(value == 'finished' for value in self.module.reconcile(ledger).values()))

    def test_unacknowledged_launch_is_pending_even_when_lock_not_yet_acquired(self):
        job = self.ledger.jobs()[0]
        self.ledger.update_job(job['job_id'], state='running', process={'token': 'delayed'})
        reconciled = self.module.reconcile(self.ledger)
        self.assertEqual(reconciled[job['job_id']], 'pending')
        self.assertEqual(len(self.ledger.jobs()), 1)
        self.assertEqual(self.ledger.jobs()[0]['process']['token'], 'delayed')

    def test_dead_or_reused_supervisor_identity_blocks_new_launch(self):
        from scripts.rc_evaluation.deepeye.campaign.processes import atomic_json
        job = self.ledger.jobs()[0]
        self.ledger.update_job(job['job_id'], state='running', process={'token': 'stale'})
        atomic_json(self.path / 'jobs' / (job['job_id'] + '.json'),
                    {'token': 'stale', 'pid': os.getpid(), 'identity': 'unrelated prior process', 'phase': 'running'})
        reconciled = self.module.reconcile(self.ledger)
        self.assertEqual(reconciled[job['job_id']], 'blocked')
        self.assertEqual(self.ledger.jobs()[0]['state'], 'blocked')
        self.assertEqual(self.module.control(self.path)['mode'], 'blocked')

    def test_status_does_not_create_control_or_claim_work(self):
        before = self.ledger.jobs()
        result = self.module.status(self.path)
        self.assertEqual(result['mode'], 'configured')
        self.assertEqual(self.ledger.jobs(), before)
        self.assertFalse((self.path / 'control.json').exists())

    def test_configuration_mismatch_blocks_before_launch(self):
        with patch.object(self.module, 'validate_config', side_effect=ValueError('changed input')):
            with self.assertRaises(ValueError):
                self.module.run(self.path)
        self.assertEqual(self.module.control(self.path)['mode'], 'blocked')
        self.assertIsNone(self.ledger.jobs()[0]['process'])

    def test_duplicate_controller_is_rejected_before_claim_or_launch(self):
        from scripts.rc_evaluation.deepeye.campaign.processes import exclusive_lock
        before = self.ledger.jobs()
        with exclusive_lock(self.path / 'controller.lock'), self.assertRaises(BlockingIOError):
            self.module.run(self.path)
        self.assertEqual(self.ledger.jobs(), before)

    def test_completed_batch_exit_one_is_finished_not_global_fault(self):
        from scripts.rc_evaluation.deepeye.campaign.processes import atomic_json
        job = self.ledger.jobs()[0]
        with RunStore.create(Path(job['run_dir']), {'items': [{'task_key': 'lite/a'}]}) as store, contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(store, [('lite', item('a'))], Factory(('sql_generation', 'a')), FakeTrace())
        self.ledger.update_job(job['job_id'], state='running', process={'token': 'normal'})
        atomic_json(self.path / 'jobs' / (job['job_id'] + '.json'), {'token': 'normal', 'phase': 'finished', 'exit_code': 1})
        self.assertEqual(self.module.reconcile(self.ledger)[job['job_id']], 'finished')
        self.assertNotEqual(self.module.control(self.path)['mode'], 'blocked')

    def test_cli_help_and_status_work_from_unrelated_directory(self):
        for args in (['--help'], ['status', '--campaign-dir', str(self.path)]):
            child = subprocess.run([sys.executable, '-E', '-B', str(ROOT / 'scripts/deepeye_bird_interact_campaign.py'), *args],
                                   cwd='/', capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)


class ConfigurationTests(unittest.TestCase):
    def test_configure_is_offline_freezes_defaults_and_venv_entry(self):
        try:
            from scripts.rc_evaluation.deepeye.campaign import configuration
        except ImportError as exc:
            self.fail(f'Offline configuration is required: {exc}')
        import socket
        from types import SimpleNamespace
        from scripts import deepeye_bird_interact_run as native
        from scripts.rc_evaluation.deepeye.tests.test_cli import ENV
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('env', 'lite', 'full', 'train'):
                (root / name).write_text('fixture')
            args = SimpleNamespace(campaign_dir=root / 'campaign', precompute_dir=root / 'pre',
                                   few_shot_source=root / 'train', env_file=root / 'env', rc_lite=root / 'lite',
                                   rc_full=root / 'full', item_keys=['lite/a'], tail_fraction=.8, poll_seconds=60)
            inputs = ([('lite', item('a'))], [{'task_key': 'lite/a'}], {'code': native.code_source_hashes()})
            with patch.object(native, 'prepare_inputs', return_value=inputs), \
                    patch('scripts.deepeye_bird_interact_smoke.read_environment', return_value=ENV), \
                    patch('scripts.rc_evaluation.deepeye.contracts.load_contracts', return_value={'lite/a': {}}), \
                    patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
                configuration.configure(args)
            with CampaignLedger.open(root / 'campaign', read_only=True) as ledger:
                config = ledger.config
                self.assertEqual(config['python'], os.path.abspath(sys.executable))
                self.assertEqual(config['native_manifest']['effective_config']['runtime']['request_limit'], 8000)
                self.assertEqual(config['native_manifest']['effective_config']['chat']['sample_max_attempts'], 4)
                self.assertEqual(len(ledger.jobs()), 1)
                self.assertFalse(Path(ledger.jobs()[0]['run_dir']).exists())
                self.assertNotIn('test-key', json.dumps(config))
                self.assertNotIn('test-password', json.dumps(config))


class FaultHandlingTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        with tempfile.TemporaryDirectory() as directory:
            config = dict(items=['lite/a'], native_args=[], tail_fraction=.8, poll_seconds=60,
                          python=sys.executable, code_root=str(ROOT), env_file='/tmp/env', rc_lite='/tmp/lite', rc_full='/tmp/full')
            with CampaignLedger.create(Path(directory) / 'campaign', config) as ledger:
                job = ledger.jobs()[0]
                with RunStore.create(Path(job['run_dir']), {'items': [{'task_key': 'lite/a'}]}) as store:
                    attempt = store.begin_attempt('lite/a', 'sql_generation', 'input')
                yield ledger, job, attempt

    def append_api_error(self, job, attempt, call_id='unauthorized'):
        with RunStore.open(Path(job['run_dir'])) as store:
            store.append_event(attempt, 'api_error', {'call_id': call_id,
                'error': {'type': {'module': 'openai', 'qualname': 'AuthenticationError'}, 'status_code': 401}})

    def record_postgres_error(self, job, attempt, error, *, connecting):
        from types import SimpleNamespace
        from scripts.baseline_adapters.deepeye.postgres_execution import execute_postgres_sql
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        def execute(*args, **kwargs):
            raise error
        connection = SimpleNamespace(read_only=False, rollback=lambda: None, close=lambda: None,
            cursor=lambda: SimpleNamespace(execute=execute, close=lambda: None))
        # The adapter and its real trace wrapper serialize the actual
        # SQLExecutionResult; only connection establishment is replaced.
        with RunStore.open(Path(job['run_dir'])) as store:
            recorder = TraceRecorder(store)
            wrapped = recorder._execution_wrapper(lambda service, data, sql: execute_postgres_sql(data, sql),
                                                   'sql_execute', 'execute')
            with patch('scripts.baseline_adapters.deepeye.postgres_execution.psycopg.connect',
                       side_effect=error if connecting else None, return_value=connection), recorder.context(attempt):
                wrapped(None, item('a'), 'SELECT x FROM t')
            event = next(store.iter_events(attempt, kinds='sql_execute_result'))
            return event['payload']['result']

    def test_actual_postgres_auth_permission_and_database_failures_block(self):
        import psycopg
        from scripts.rc_evaluation.deepeye.campaign import controller
        cases = [(psycopg.OperationalError('password authentication failed'), True),
                 (psycopg.errors.InsufficientPrivilege('permission denied for table t'), False),
                 (psycopg.errors.InvalidCatalogName('database missing does not exist'), False)]
        for error, connecting in cases:
            with self.subTest(error=type(error).__name__), self.fixture() as (ledger, job, attempt):
                result = self.record_postgres_error(job, attempt, error, connecting=connecting)
                self.assertEqual(result['result_type'], 'execution_error')
                if connecting:
                    self.assertEqual(result['error_message'], 'PostgreSQL connection failed; check connection settings and server availability')
                with self.assertRaises(RuntimeError):
                    controller.check_run_faults(ledger, job)
                self.assertEqual(controller.control(ledger.campaign_dir)['mode'], 'blocked')

    def test_ordinary_generated_sql_errors_and_timeouts_do_not_block(self):
        import psycopg
        from scripts.rc_evaluation.deepeye.campaign import controller
        for error in (psycopg.errors.SyntaxError('syntax error near FROM'),
                      psycopg.errors.UndefinedTable('relation missing does not exist'),
                      psycopg.errors.QueryCanceled('statement timeout')):
            with self.subTest(error=type(error).__name__), self.fixture() as (ledger, job, attempt):
                self.record_postgres_error(job, attempt, error, connecting=False)
                controller.check_run_faults(ledger, job)
                self.assertEqual(controller.control(ledger.campaign_dir)['mode'], 'configured')

    def test_crash_after_cursor_write_cannot_lose_durable_block(self):
        from scripts.rc_evaluation.deepeye.campaign import controller
        with self.fixture() as (ledger, job, attempt):
            self.append_api_error(job, attempt)
            atomic = controller.atomic_json
            def crash(path, value):
                atomic(path, value)
                if Path(path).name == 'fault-cursors.json':
                    raise SystemExit('crash immediately after durable cursor')
            with patch.object(controller, 'atomic_json', side_effect=crash), self.assertRaises(SystemExit):
                controller.check_run_faults(ledger, job)
            self.assertEqual(controller.control(ledger.campaign_dir)['mode'], 'blocked')
            # Acknowledged evidence does not re-block an explicit repair/resume,
            # but a later independent event must still block.
            controller.set_control(ledger, 'running', 'explicit_resume_after_repair')
            controller.check_run_faults(ledger, job)
            self.assertEqual(controller.control(ledger.campaign_dir)['mode'], 'running')
            self.append_api_error(job, attempt, 'later-unauthorized')
            with self.assertRaises(RuntimeError):
                controller.check_run_faults(ledger, job)
            self.assertEqual(controller.control(ledger.campaign_dir)['mode'], 'blocked')

    def test_crash_before_block_does_not_acknowledge_fault(self):
        from scripts.rc_evaluation.deepeye.campaign import controller
        with self.fixture() as (ledger, job, attempt):
            self.append_api_error(job, attempt)
            with patch.object(controller, 'set_control', side_effect=SystemExit('before block')), self.assertRaises(SystemExit):
                controller.check_run_faults(ledger, job)
            with self.assertRaises(RuntimeError):
                controller.check_run_faults(ledger, job)
            self.assertEqual(controller.control(ledger.campaign_dir)['mode'], 'blocked')


class AuditMilestoneTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, population):
        from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
        config = dict(items=[f'lite/{number:03d}' for number in range(population)], native_args=[],
                      tail_fraction=.8, poll_seconds=60, python=sys.executable, code_root=str(ROOT),
                      env_file='/tmp/env', rc_lite='/tmp/lite', rc_full='/tmp/full')
        with tempfile.TemporaryDirectory() as directory, CampaignLedger.create(Path(directory) / 'campaign', config) as ledger:
            yield ledger

    def fill_native(self, ledger, job, keys, *, failed):
        from scripts.baseline_adapters.deepeye.run_pipeline import STAGE_METHODS
        manifest = {'items': [{'task_key': key} for key in job['items']]}
        path = Path(job['run_dir'])
        with (RunStore.open(path) if path.exists() else RunStore.create(path, manifest)) as store:
            base = zero_factory()
            def factory(stage, tasks):
                runner = base(stage, tasks)
                if failed and stage == 'schema_linking':
                    def failure(target):
                        raise ValueError('controlled native failure')
                    setattr(runner, STAGE_METHODS[stage], failure)
                return runner
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(store, fixture_tasks({'items': keys}), factory, FakeTrace())
        ledger.update_job(job['job_id'], state='prepared')

    def tick(self, ledger, job):
        from scripts.rc_evaluation.deepeye.campaign.controller import observe
        from scripts.rc_evaluation.deepeye.campaign.planning import plan_tick
        return plan_tick(ledger, {job['job_id']: observe(job)})

    def test_first_failures_pending_retry_do_not_form_native_milestone(self):
        from scripts.rc_evaluation.deepeye.campaign.monitoring import audit_milestones
        with self.fixture(100) as ledger:
            first = ledger.jobs()[0]
            self.fill_native(ledger, first, first['items'], failed=True)
            self.tick(ledger, first)
            self.assertEqual(audit_milestones(ledger), [])
            self.assertEqual(list((ledger.campaign_dir / 'audits').glob('*.json')), [])

    def test_retry_successes_use_canonical_source_and_unique_fresh_milestones(self):
        from scripts.rc_evaluation.deepeye.campaign.monitoring import audit_milestones
        from scripts.rc_evaluation.deepeye.campaign.processes import read_json
        with self.fixture(200) as ledger:
            first = ledger.jobs()[0]
            self.fill_native(ledger, first, first['items'], failed=True)
            result = self.tick(ledger, first)
            retry = next(job for job in result['jobs'] if job['kind'] == 'native_retry')
            self.fill_native(ledger, retry, retry['items'][:100], failed=False)
            result = self.tick(ledger, retry)
            paths = audit_milestones(ledger)
            self.assertEqual([Path(path).name for path in paths], ['native-000100.json'])
            first_audit = read_json(paths[0])
            self.assertEqual(len(first_audit['examples']), 5)
            for example in first_audit['examples']:
                self.assertEqual(example['source_run'], result['canonical_sources'][example['task_key']])
                self.assertEqual(example['job_id'], retry['job_id'])
                self.assertEqual(example['status'], 'succeeded')
            self.fill_native(ledger, retry, retry['items'][100:], failed=False)
            self.tick(ledger, retry)
            paths = audit_milestones(ledger)
            self.assertEqual([Path(path).name for path in paths], ['native-000100.json', 'native-000200.json'])
            examples = [example for path in paths for example in read_json(path)['examples']]
            self.assertEqual(len(examples), 10)
            self.assertEqual(len({row['task_key'] for row in examples}), 10)
            before = {path: Path(path).read_bytes() for path in paths}
            events = ledger._db.execute("SELECT COUNT(*) FROM events WHERE kind='audit_queued'").fetchone()[0]
            self.assertEqual(audit_milestones(ledger), paths)
            self.assertEqual({path: Path(path).read_bytes() for path in paths}, before)
            self.assertEqual(ledger._db.execute("SELECT COUNT(*) FROM events WHERE kind='audit_queued'").fetchone()[0], events)

    def test_permanent_exclusions_count_once_and_rc_milestones_are_unchanged(self):
        from scripts.rc_evaluation.deepeye.campaign.monitoring import audit_milestones
        from scripts.rc_evaluation.deepeye.campaign.processes import read_json
        with self.fixture(200) as ledger:
            first = ledger.jobs()[0]
            self.fill_native(ledger, first, first['items'][:100], failed=False)
            self.fill_native(ledger, first, first['items'][100:], failed=True)
            result = self.tick(ledger, first)
            retry = next(job for job in result['jobs'] if job['kind'] == 'native_retry')
            rc = next(job for job in result['jobs'] if job['kind'] == 'rc')
            self.fill_native(ledger, retry, retry['items'], failed=True)
            result = self.tick(ledger, retry)
            paths = audit_milestones(ledger)
            self.assertEqual([Path(path).name for path in paths], ['native-000100.json', 'native-000200.json'])
            examples = [row for path in paths for row in read_json(path)['examples']]
            excluded_examples = [row for row in examples if row['task_key'] in result['excluded']]
            self.assertTrue(excluded_examples)
            for example in excluded_examples:
                self.assertEqual(example['status'], 'failed')
                self.assertEqual(example['source_run'], retry['run_dir'])
            manifest = {'items': [{'task_key': key} for key in rc['items']],
                        'target_stage': rc['target_stage'], 'source_run': rc['source_run']}
            with RunStore.create(Path(rc['run_dir']), manifest) as store:
                for key in rc['items']:
                    attempt = store.begin_attempt(key, rc['target_stage'], 'rc')
                    store.finish_attempt(attempt, 'succeeded', {'artifact': {}, 'rc_participation': {'status': 'rc_not_participating'}})
            ledger.update_job(rc['job_id'], state='prepared')
            self.tick(ledger, rc)
            paths = audit_milestones(ledger)
            self.assertEqual(set(Path(path).name for path in paths), {'native-000100.json', 'native-000200.json', 'schema_linking-000100.json'})
            rc_audit = read_json(ledger.campaign_dir / 'audits/schema_linking-000100.json')
            self.assertEqual(len(rc_audit['examples']), 5)
            self.assertTrue(all(row['stages'][0]['rc_participation']['status'] == 'rc_not_participating' for row in rc_audit['examples']))


if __name__ == '__main__':
    unittest.main()
