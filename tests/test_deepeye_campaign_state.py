"""Rolling policy tests: real SQLite ledgers and native RunStore records."""
import contextlib
import importlib
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tests.test_deepeye_run_pipeline import Factory, FakeTrace, item
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, run_pipeline
from scripts.baseline_adapters.deepeye.run_store import RunStore


class CampaignStateTest(unittest.TestCase):
    def test_generic_typed_keys_can_be_frozen_without_lite_full_assumptions(self):
        ledger = self.ledger(['spider/dev/i:7', 'spider/dev/i:19'])
        self.assertEqual(ledger.jobs()[0]['items'], ['spider/dev/i:19', 'spider/dev/i:7'])

    def test_incremental_reader_decodes_only_new_attempts_and_finishes(self):
        from scripts.rc_evaluation.deepeye.campaign import observations
        self.assertTrue(hasattr(observations, 'RunObservationReader'))
        ledger = self.ledger(['lite/a'])
        job = ledger.jobs()[0]
        store = self.prepare(ledger, job)
        started = self.native(store, 'lite/a', 'unfinished')
        with observations.RunObservationReader(job['run_dir'], kind='native') as reader:
            with patch.object(reader.store, '_attempt_dict', wraps=reader.store._attempt_dict) as decode:
                self.assertEqual(reader.read()['states']['lite/a']['status'], 'unfinished')
                self.assertEqual(decode.call_count, 1)
                decode.reset_mock()
                for _ in range(3):
                    self.assertEqual(reader.read()['states']['lite/a']['status'], 'unfinished')
                self.assertEqual(decode.call_count, 0)
                stage = store.begin_attempt('lite/a', 'schema_linking', 'input')
                store.finish_attempt(stage, 'failed', {'error': 'offline fixture'})
                store.finish_attempt(started, 'failed', {'failed_stage': 'schema_linking',
                    'stage_attempts': [{'stage': 'schema_linking', 'attempt_id': stage}]})
                self.assertEqual(reader.read()['states']['lite/a']['status'], 'failed')
                self.assertEqual(decode.call_count, 2)
                decode.reset_mock()
                reader.read()
                self.assertEqual(decode.call_count, 0)

    def test_jobs_read_all_memberships_in_one_query(self):
        ledger = self.ledger()
        for key in 'abcde':
            ledger.claim_job(kind='native_retry', items=['lite/' + key])
        queries = []
        ledger._db.set_trace_callback(queries.append)
        try:
            jobs = ledger.jobs()
        finally:
            ledger._db.set_trace_callback(None)
        self.assertEqual(len(jobs), 6)
        member_queries = [query for query in queries if 'FROM claims' in query]
        self.assertEqual(len(member_queries), 1)

    def setUp(self):
        try:
            self.Ledger = importlib.import_module('scripts.rc_evaluation.deepeye.campaign.ledger').CampaignLedger
            self.read = importlib.import_module('scripts.rc_evaluation.deepeye.campaign.observations').read_run
            self.tick = importlib.import_module('scripts.rc_evaluation.deepeye.campaign.planning').plan_tick
        except ModuleNotFoundError as exc:
            self.fail(f'Campaign state implementation is required: {exc}')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(items=['lite/' + k for k in 'abcde'], tail_fraction=0.8,
                           poll_seconds=60, native_args=[], env_file='/tmp/env',
                           rc_lite='/tmp/lite', rc_full='/tmp/full', python='/usr/bin/python3',
                           code_root=str(Path(__file__).resolve().parents[1]))

    def ledger(self, keys=None):
        config = dict(self.config)
        if keys is not None:
            config['items'] = keys
        ledger = self.Ledger.create(self.root / 'campaign', config)
        self.addCleanup(ledger.close)
        return ledger

    def prepare(self, ledger, job):
        manifest = {'items': [{'task_key': key} for key in job['items']]}
        if job['kind'] == 'rc':
            manifest.update(target_stage=job['target_stage'], source_run=job['source_run'])
        store = RunStore.create(Path(job['run_dir']), manifest)
        self.addCleanup(store.close)
        ledger.update_job(job['job_id'], state='prepared')
        return store

    def native(self, store, key, status='succeeded'):
        identity = key.split('/', 1)[1]
        if status == 'unfinished':
            return store.begin_attempt(key, 'pipeline', 'pending')
        with contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(store, [('lite', item(identity))],
                         Factory(('sql_generation', identity) if status == 'failed' else None), FakeTrace())

    def rc(self, store, key, stage, status='succeeded'):
        attempt = store.begin_attempt(key, stage, 'rc-input')
        if status != 'unfinished':
            store.finish_attempt(attempt, status, {'rc_participated': False})

    def observe(self, job):
        return self.read(job['run_dir'], kind='rc' if job['kind'] == 'rc' else 'native',
                         target_stage=job['target_stage'])

    def test_four_of_five_terminal_opens_gate_and_only_failed_gets_retry(self):
        ledger = self.ledger()
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        for key in 'abc':
            self.native(store, 'lite/' + key)
        self.native(store, 'lite/e', 'unfinished')
        below = self.tick(ledger, {first['job_id']: self.observe(first)})
        self.assertEqual(below['opened_stages'], [])
        self.assertEqual(below['jobs'], [])
        self.native(store, 'lite/d', 'failed')
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        self.assertEqual(result['opened_stages'], ['schema_linking'])
        self.assertEqual([j['items'] for j in result['jobs'] if j['kind'] == 'native_retry'], [['lite/d']])
        self.assertEqual([j['items'] for j in result['jobs'] if j['kind'] == 'rc'], [['lite/a', 'lite/b', 'lite/c']])
        ledger.close()
        with self.Ledger.open(self.root / 'campaign') as reopened:
            self.assertEqual(self.tick(reopened, {})['jobs'], [])
            self.assertFalse(self.tick(reopened, {})['complete'])

    def test_fixed_anchor_late_items_and_independent_stages(self):
        ledger = self.ledger()
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        for key in 'abc':
            self.native(store, 'lite/' + key)
        self.native(store, 'lite/d', 'failed')
        self.native(store, 'lite/e', 'unfinished')
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        linking = next(j for j in result['jobs'] if j['kind'] == 'rc')
        retry = next(j for j in result['jobs'] if j['kind'] == 'native_retry')
        rcstore = self.prepare(ledger, linking)
        for key in 'bc':
            self.rc(rcstore, 'lite/' + key, 'schema_linking')
        result = self.tick(ledger, {linking['job_id']: self.observe(linking)})
        self.assertNotIn('sql_generation', ledger.opened_stages())
        self.native(store, 'lite/e')
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        self.assertEqual([(j['target_stage'], j['items']) for j in result['jobs']], [('schema_linking', ['lite/e'])])
        self.assertEqual(ledger.anchor('schema_linking'), [linking['job_id']])
        self.rc(rcstore, 'lite/a', 'schema_linking', 'failed')
        result = self.tick(ledger, {linking['job_id']: self.observe(linking)})
        self.assertEqual(result['opened_stages'], ['sql_generation'])
        self.assertEqual(result['jobs'][0]['items'], ['lite/a', 'lite/b', 'lite/c', 'lite/e'])
        retry_store = self.prepare(ledger, retry)
        self.native(retry_store, 'lite/d')
        result = self.tick(ledger, {retry['job_id']: self.observe(retry)})
        self.assertEqual([(j['target_stage'], j['items']) for j in result['jobs']],
                         [('schema_linking', ['lite/d']), ('sql_generation', ['lite/d'])])
        self.assertEqual(result['canonical_sources']['lite/d'], retry['run_dir'])

    def test_failed_retry_is_excluded_without_third_native_attempt(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        self.native(self.prepare(ledger, first), 'lite/a', 'failed')
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        retry = result['jobs'][0]
        self.native(self.prepare(ledger, retry), 'lite/a', 'failed')
        result = self.tick(ledger, {retry['job_id']: self.observe(retry)})
        self.assertEqual(result['excluded'], ['lite/a'])
        self.assertEqual(result['canonical_sources'], {})
        self.assertEqual(result['jobs'], [])
        self.assertTrue(result['complete'])
        self.assertEqual(ledger.opened_stages(), ['schema_linking'])

    def test_completion_requires_all_four_targets_even_when_zero_call_reused(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        self.native(self.prepare(ledger, first), 'lite/a')
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        for stage in STAGES:
            self.assertFalse(result['complete'])
            job = next(j for j in result['jobs'] if j['target_stage'] == stage)
            self.rc(self.prepare(ledger, job), 'lite/a', stage)
            result = self.tick(ledger, {job['job_id']: self.observe(job)})
        self.assertTrue(result['complete'])

    def test_all_first_dispatch_source_cohorts_belong_to_anchor(self):
        ledger = self.ledger(['lite/a', 'lite/b'])
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        self.native(store, 'lite/a')
        self.native(store, 'lite/b', 'failed')
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        retry = next(j for j in result['jobs'] if j['kind'] == 'native_retry')
        linking = next(j for j in result['jobs'] if j['kind'] == 'rc')
        self.native(self.prepare(ledger, retry), 'lite/b')
        self.rc(self.prepare(ledger, linking), 'lite/a', 'schema_linking')
        result = self.tick(ledger, {retry['job_id']: self.observe(retry), linking['job_id']: self.observe(linking)})
        generation = [j for j in result['jobs'] if j['target_stage'] == 'sql_generation']
        self.assertEqual([j['items'] for j in generation], [['lite/a'], ['lite/b']])
        self.assertEqual(set(ledger.anchor('sql_generation')), {j['job_id'] for j in generation})

    def test_config_freezing_invalid_numbers_and_members(self):
        ledger = self.ledger()
        config = ledger.config
        config['items'].append('lite/extra')
        self.assertEqual(len(ledger.config['items']), 5)
        for field, value in [('tail_fraction', True), ('tail_fraction', float('nan')),
                             ('tail_fraction', 0), ('tail_fraction', 1.01),
                             ('poll_seconds', 0), ('poll_seconds', False), ('poll_seconds', float('inf')),
                             ('items', []), ('items', ['lite/a', 'lite/a']), ('items', ['../a'])]:
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, TypeError)):
                self.Ledger.create(self.root / 'invalid', {**self.config, field: value})
        self.assertFalse((self.root / 'invalid').exists())

    def test_whole_cohort_conflict_rolls_back_and_read_only_cannot_claim(self):
        ledger = self.ledger()
        first = ledger.jobs()[0]
        ledger.claim_job(kind='native_retry', items=['lite/b'])
        before = ledger.jobs()
        with self.assertRaises((ValueError, sqlite3.IntegrityError)):
            ledger.claim_job(kind='native_retry', items=['lite/a', 'lite/b'])
        self.assertEqual(ledger.jobs(), before)
        self.assertEqual(ledger.claim_job(kind='native_retry', items=['lite/a'])['items'], ['lite/a'])
        for kwargs in [dict(kind='native_retry', items=['lite/z']),
                       dict(kind='rc', items=['lite/z'], source_run=first['run_dir'], target_stage=STAGES[0]),
                       dict(kind='rc', items=['lite/a'], source_run='/tmp/unknown', target_stage=STAGES[0])]:
            with self.assertRaises(ValueError):
                ledger.claim_job(**kwargs)
        with self.Ledger.open(self.root / 'campaign', read_only=True) as reader:
            self.assertEqual(reader.jobs(), ledger.jobs())
            with self.assertRaises(PermissionError):
                reader.update_job(first['job_id'], state='paused')

    def test_unknown_observation_or_member_mismatch_rolls_back_tick(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        self.native(self.prepare(ledger, first), 'lite/a')
        observed = self.observe(first)
        for observations in [{'unseen': observed}, {first['job_id']: {**observed, 'items': ['lite/z']}},
                             {first['job_id']: {**observed, 'run_dir': '/wrong/path'}}]:
            with self.assertRaises(ValueError):
                self.tick(ledger, observations)
            self.assertEqual(ledger.opened_stages(), [])
            self.assertEqual(len(ledger.jobs()), 1)

    def test_canonical_success_cannot_be_replaced_or_regressed(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        self.native(store, 'lite/a')
        self.tick(ledger, {first['job_id']: self.observe(first)})
        self.native(store, 'lite/a', 'failed')
        # Native completed prefixes are reused; an explicit conflicting master
        # represents a malformed resumed writer, not a legitimate retry.
        store.begin_attempt('lite/a', 'pipeline', 'conflict')
        with self.assertRaises(ValueError):
            self.tick(ledger, {first['job_id']: self.observe(first)})

    def test_rc_source_mismatch_is_fatal(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        self.native(self.prepare(ledger, first), 'lite/a')
        job = self.tick(ledger, {first['job_id']: self.observe(first)})['jobs'][0]
        self.prepare(ledger, job)
        observed = self.observe(job)
        observed['manifest']['source_run'] = '/wrong/source'
        with self.assertRaises(ValueError):
            self.tick(ledger, {job['job_id']: observed})

    def test_native_observation_rejects_forged_or_wrong_item_success_links(self):
        for cross_item in [False, True]:
            with self.subTest(cross_item=cross_item):
                path = self.root / str(cross_item)
                with RunStore.create(path, {'items': [{'task_key': 'lite/a'}, {'task_key': 'lite/b'}]}) as store:
                    links = []
                    for stage in STAGES:
                        attempt = store.begin_attempt('lite/b' if cross_item else 'lite/a', stage, 'input')
                        store.finish_attempt(attempt, 'succeeded', {})
                        links.append({'stage': stage, 'attempt_id': attempt})
                    master = store.begin_attempt('lite/a', 'pipeline', 'input')
                    store.finish_attempt(master, 'succeeded', {'stage_attempts': links if cross_item else links[:3]})
                with self.assertRaises(ValueError):
                    self.read(path, kind='native')

    def test_failed_master_requires_actual_failed_stage_reference(self):
        path = self.root / 'bad-failure'
        with RunStore.create(path, {'items': [{'task_key': 'lite/a'}]}) as store:
            master = store.begin_attempt('lite/a', 'pipeline', 'input')
            store.finish_attempt(master, 'failed', {'failed_stage': 'sql_generation', 'stage_attempts': []})
        with self.assertRaises(ValueError):
            self.read(path, kind='native')

    def test_native_success_rejects_conflicting_successful_stage_fingerprints(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                path = self.root / stage
                with RunStore.create(path, {'items': [{'task_key': 'lite/a'}]}) as store:
                    self.native(store, 'lite/a')
                    original = next(row for row in store.attempts() if row['stage'] == stage)
                    conflicting = store.begin_attempt('lite/a', stage, 'different-input-fingerprint')
                    store.finish_attempt(conflicting, 'succeeded', original['payload'])
                    # Match the source-resume contract enforced by RunStore.
                    with self.assertRaises(ValueError):
                        store.completed('lite/a', stage, original['input_fingerprint'])
                    with self.assertRaises(ValueError):
                        self.read(path, kind='native')

    def test_native_fingerprint_validation_allows_same_input_success_and_different_input_failure(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        self.native(store, 'lite/a')
        original = next(row for row in store.attempts() if row['stage'] == 'schema_linking')
        repeated = store.begin_attempt('lite/a', 'schema_linking', original['input_fingerprint'])
        store.finish_attempt(repeated, 'succeeded', original['payload'])
        failed = store.begin_attempt('lite/a', 'schema_linking', 'different-input-fingerprint')
        store.finish_attempt(failed, 'failed', {'error_type': 'ControlledFailure'})
        result = self.tick(ledger, {first['job_id']: self.observe(first)})
        self.assertEqual(result['canonical_sources'], {'lite/a': first['run_dir']})
        self.assertEqual([job['kind'] for job in result['jobs']], ['rc'])

    def test_unfinished_master_and_pending_items_are_not_terminal(self):
        path = self.root / 'unfinished'
        with RunStore.create(path, {'items': [{'task_key': 'lite/a'}, {'task_key': 'lite/b'}]}) as store:
            store.begin_attempt('lite/a', 'pipeline', 'input')
            attempt = store.begin_attempt('lite/a', 'sql_selection', 'input')
            store.finish_attempt(attempt, 'succeeded', {})
        observed = self.read(path, kind='native')
        self.assertEqual(observed['states']['lite/a']['status'], 'unfinished')
        self.assertEqual(observed['states']['lite/b']['status'], 'pending')

    def test_partial_sampling_success_is_canonical_without_event_reads(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        from tests.test_deepeye_sampling import llm_fixture, response, parse
        from app.llm_extractor import LLMExtractor
        from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
        from unittest.mock import patch
        base = Factory()
        llm, calls = llm_fixture([response()] * 3 + [response('bad')] * 4 + [response()])

        def factory(stage, items):
            runner = base(stage, items)
            if stage == 'schema_linking':
                original = runner._link_tables_and_columns

                def process(target):
                    LLMExtractor().extract_with_retry(llm, [], parse, n=5)
                    original(target)

                runner._link_tables_and_columns = process
                runner._llm = llm
            return runner

        recorder = TraceRecorder(store)
        with patch.object(recorder, 'instrument_runner', return_value=lambda: None), \
                patch('app.llm_extractor.extractor.logger.warning'), contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(store, [('lite', item('a'))], factory, recorder, workers=1)
        linking = next(row for row in store.attempts() if row['stage'] == 'schema_linking')
        self.assertFalse(linking['payload']['sampling']['complete'])
        self.assertEqual(len(calls), 8)
        with patch.object(RunStore, 'events', side_effect=AssertionError('No event scan')), \
                patch.object(RunStore, 'iter_events', side_effect=AssertionError('No event scan')):
            result = self.tick(ledger, {first['job_id']: self.observe(first)})
        self.assertEqual(result['canonical_sources'], {'lite/a': first['run_dir']})
        self.assertEqual([j['kind'] for j in result['jobs']], ['rc'])

    def test_rc_four_of_five_anchor_threshold_is_exact(self):
        ledger = self.ledger()
        first = ledger.jobs()[0]
        store = self.prepare(ledger, first)
        for key in first['items']:
            self.native(store, key)
        linking = self.tick(ledger, {first['job_id']: self.observe(first)})['jobs'][0]
        rcstore = self.prepare(ledger, linking)
        for key in ['lite/a', 'lite/b', 'lite/c']:
            self.rc(rcstore, key, STAGES[0])
        self.assertEqual(self.tick(ledger, {linking['job_id']: self.observe(linking)})['opened_stages'], [])
        self.rc(rcstore, 'lite/d', STAGES[0], 'failed')
        result = self.tick(ledger, {linking['job_id']: self.observe(linking)})
        self.assertEqual(result['opened_stages'], ['sql_generation'])
        self.assertEqual(result['rc_counts']['schema_linking'],
                         {'pending': 1, 'unfinished': 0, 'succeeded': 3, 'failed': 1, 'total': 5, 'terminal': 4})

    def test_rc_observation_reads_only_target_and_rejects_unlisted_target_item(self):
        path = self.root / 'target-only'
        with RunStore.create(path, {'items': [{'task_key': 'lite/a'}], 'target_stage': STAGES[0]}) as store:
            self.rc(store, 'lite/a', STAGES[0])
            store.begin_attempt('lite/a', STAGES[1], 'unrelated')
        with sqlite3.connect(path / 'run.sqlite3') as connection:
            connection.execute('DROP TRIGGER attempts_no_update')
            connection.execute("UPDATE attempts SET record_checksum='bad' WHERE stage='sql_generation'")
        self.assertEqual(self.read(path, kind='rc', target_stage=STAGES[0])['states']['lite/a']['status'], 'succeeded')
        with RunStore.open(path) as store:
            self.rc(store, 'lite/unseen', STAGES[0])
        with self.assertRaises(ValueError):
            self.read(path, kind='rc', target_stage=STAGES[0])

    def test_observation_batch_rollback_does_not_remember_earlier_valid_member(self):
        ledger = self.ledger(['lite/a'])
        first = ledger.jobs()[0]
        self.native(self.prepare(ledger, first), 'lite/a')
        with self.assertRaises(ValueError):
            self.tick(ledger, {first['job_id']: self.observe(first), 'unknown-job': {}})
        result = self.tick(ledger, {})
        self.assertEqual(result['canonical_sources'], {})
        self.assertEqual(result['jobs'], [])

    def test_missing_store_and_corrupt_checksums_do_not_look_empty(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read(self.root / 'absent', kind='native')
        path = self.root / 'corrupt'
        with RunStore.create(path, {'items': [{'task_key': 'lite/a'}]}) as store:
            store.begin_attempt('lite/a', 'pipeline', 'input')
        with sqlite3.connect(path / 'run.sqlite3') as connection:
            connection.execute('DROP TRIGGER attempts_no_update')
            connection.execute("UPDATE attempts SET record_checksum = 'bad'")
        with self.assertRaises(ValueError):
            self.read(path, kind='native')

    def test_anchor_is_frozen_and_wrong_stage_members_rejected(self):
        ledger = self.ledger()
        first = ledger.jobs()[0]
        linking = ledger.claim_job(kind='rc', items=['lite/a'], source_run=first['run_dir'], target_stage=STAGES[0])
        generation = ledger.claim_job(kind='rc', items=['lite/a'], source_run=first['run_dir'], target_stage=STAGES[1])
        ledger.open_stage(STAGES[0])
        with self.assertRaises(ValueError):
            ledger.set_anchor(STAGES[0], [generation['job_id']])
        ledger.set_anchor(STAGES[0], [linking['job_id']])
        with self.assertRaises(ValueError):
            ledger.set_anchor(STAGES[0], [])


if __name__ == '__main__':
    unittest.main()
