import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.dail_sql.config import MODES, TaskKey
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from tests.test_dail_sql_records import complete_round, manifest


def four_rounds(records, key, *, total_values=(20, 24, 40, 44)):
    version = records.begin_version(key)
    rounds = {}
    for rid, number, parent, rc, total in zip('ABCD', (1,1,2,2), (None,None,'A','A'),
                                             (False,True,False,True), total_values):
        usage = {'prompt_tokens': total // 2, 'completion_tokens': total // 2,
                 'total_tokens': total, 'completion_tokens_details': {'reasoning_tokens': 2}}
        aggregate = {'prompt_tokens': total // 2 * 5, 'completion_tokens': total // 2 * 5,
                     'total_tokens': total * 5, 'completion_tokens_details': {'reasoning_tokens': 10}}
        value = complete_round(records, version, rid, number, parent, rc, usage, aggregate, persist=False)
        value['selection'].update(tie=False, fallback=False, selection_ref='fixture-selection')
        records.append(version, 'round_result', value)
        rounds[rid] = value
    refs = {}
    for mode, first, second in zip(MODES, 'ABAB', 'CCDD'):
        refs[mode] = records.append(version, 'mode_result', {'mode': mode, 'status': 'succeeded',
            'first_round_id': first, 'second_round_id': second,
            'final_candidate_id': second + '-0', 'failure_origin': None})
    records.seal(version, refs)
    for value in rounds.values():
        for candidate in value['candidates']:
            candidate['evaluation'] = {'bag_equal': True, 'comparable': True, 'status': 'comparable'}
    return version, rounds


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.rc_evaluation.dail_sql.reporting'),
                             'Task7 reporting missing')
        from scripts.rc_evaluation.dail_sql import reporting
        self.reporting = reporting
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = TaskKey('batch', 'spider_dev', '1')
        database_path = self.root / 'query.sqlite'
        with sqlite3.connect(database_path) as db:
            db.execute('CREATE TABLE t(a)')
        binding = {'database': {'dialect': 'sqlite', 'database_id': 'query', 'path': str(database_path)},
                   'reference_sql': 'SELECT 1'}
        evaluation = self.root / 'preparation' / 'evaluation'
        evaluation.mkdir(parents=True)
        bindings = {group: {qid: binding for qid in group_spec['ids']}
                    for group, group_spec in manifest()['groups'].items()}
        raw = json.dumps(bindings).encode()
        (evaluation / 'bindings.json').write_bytes(raw)
        (evaluation / 'source.json').write_text(json.dumps({'bindings': 'evaluation/bindings.json',
                                                          'sha256': hashlib.sha256(raw).hexdigest()}))
        self.manifest = {**manifest(), 'evaluation_source': str(evaluation / 'source.json'),
                         'sql_timeout_seconds': 1}
        self.records = DailRecords(self.root / 'batch', self.manifest)
        self.addCleanup(self.records.close)
        self.batch = self.records.root
        self.index = CurrentIndex(self.batch / 'current.sqlite3')
        self.addCleanup(self.index.close)

    def publish(self, key):
        version, rounds = four_rounds(self.records, key)
        self.index.publish(key, version, self.index.claim(key), self.records.is_sealed)
        return version, rounds

    def load_rows(self, path):
        return [json.loads(line) for line in (path / 'modes.jsonl').read_text().splitlines()]

    def publish_failed(self, key, *, dependency=True):
        version = self.records.begin_version(key)
        refs = {}
        for rc in (False, True):
            first = None
            if not dependency:
                first = 'first-' + str(rc)
                complete_round(self.records, version, first, rc=rc)
            rid = 'failed-' + str(rc)
            self.records.append(version, 'round_result', {
                'round_execution_id': rid, 'round_no': 1 if dependency else 2,
                'status': 'failed', 'rc_injected': rc, 'actual_parent_round_id': first,
                'example_ids': [str(i) for i in range(9)], 'next_example_ids': [],
                'samples': [{'sample_position': i, 'status': 'failed', 'request_attempt_ids': [],
                    'successful_request_id': None, 'success_usage': None, 'error': 'fixture failure'} for i in range(5)],
                'request_attempt_ids': [], 'successful_request_ids': [], 'success_usage': None,
                'candidates': [], 'selection': None, 'error': 'fixture failure'})
        for mode in MODES:
            first_rc = mode in ('rc_first', 'rc_both')
            second_rc = mode in ('rc_second', 'rc_both')
            first = ('failed-' if dependency else 'first-') + str(first_rc)
            origin = 'failed-' + str(first_rc if dependency else second_rc)
            refs[mode] = self.records.append(version, 'mode_result', {'mode': mode,
                'status': 'dependency_failed' if dependency else 'failed', 'first_round_id': first,
                'second_round_id': None if dependency else origin, 'final_candidate_id': None, 'failure_origin': origin})
        self.records.seal(version, refs)
        self.index.publish(key, version, self.index.claim(key), self.records.is_sealed)

    def test_vote_failure_is_counted_from_persisted_vote_not_independent_evaluation(self):
        append = self.records.append
        vote_number = 0
        def append_with_vote_failure(version, kind, payload):
            nonlocal vote_number
            if kind == 'vote_execution':
                vote_number += 1
                if vote_number == 1:
                    payload = {'status': 'error', 'rows': [], 'columns': [],
                        'error': {'type': 'vote_preprocessing', 'message': 'native transform rejected SQL'}}
            return append(version, kind, payload)
        with patch.object(self.records, 'append', side_effect=append_with_vote_failure):
            self.publish(self.key)
        exported = self.reporting.export_current(self.batch)
        summary = json.loads((exported / 'summary.json').read_text())
        voting = next(r for r in summary['voting'] if r['group'] == 'spider_dev' and r['mode'] == 'native' and r['round_no'] == 1)
        self.assertEqual(voting['execution_errors'], 1)
        evaluation = next(r for r in summary['candidate_matches'] if r['group'] == 'spider_dev' and r['mode'] == 'native' and r['round_no'] == 1)
        self.assertEqual(evaluation['execution_errors'], 0)
        self.assertEqual(evaluation['matched'], 5)
        candidate = next(json.loads(line) for line in (exported / 'candidates.jsonl').read_text().splitlines()
                         if json.loads(line)['candidate_id'] == 'A-0')
        self.assertEqual(candidate['vote_execution_summary'], {'source_event_id': candidate['vote_execution_ref'],
            'status': 'error', 'classification': 'vote_preprocessing',
            'error': {'type': 'vote_preprocessing', 'message': 'native transform rejected SQL'}})
        self.assertTrue(candidate['evaluation']['bag_equal'])

    def test_failed_question_targets_are_unique_current_terminal_and_diagnostic_scoped(self):
        other = TaskKey('batch', 'bird_dev', '1')
        self.publish_failed(self.key)
        self.publish_failed(other, dependency=False)
        exported = self.reporting.export_current(self.batch)
        path = exported / 'failed_questions.jsonl'
        self.assertTrue(path.exists(), 'atomic export lacks failed-question targets')
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],
                         [{'group': 'bird_dev', 'question_id': '1'}, {'group': 'spider_dev', 'question_id': '1'}])
        diagnostic = self.reporting.export_current(self.batch, diagnostic_targets=[self.key])
        self.assertEqual([json.loads(line) for line in (diagnostic / 'failed_questions.jsonl').read_text().splitlines()],
                         [{'group': 'spider_dev', 'question_id': '1'}])
        self.assertEqual(self.reporting.current_export(self.batch), exported)
        self.publish(self.key)
        after = self.reporting.export_current(self.batch, diagnostic_targets=[self.key])
        self.assertEqual((after / 'failed_questions.jsonl').read_bytes(), b'')

    def test_failed_question_targets_exclude_mismatch_unknown_accuracy_and_pending(self):
        self.publish(self.key)
        source = Path(self.manifest['evaluation_source'])
        bindings_path = source.parent / 'bindings.json'
        bindings = json.loads(bindings_path.read_text())
        for reference in ('SELECT 2', 'SELECT missing FROM t'):
            bindings['spider_dev']['1']['reference_sql'] = reference
            raw = json.dumps(bindings).encode()
            bindings_path.write_bytes(raw)
            source.write_text(json.dumps({'bindings': 'evaluation/bindings.json', 'sha256': hashlib.sha256(raw).hexdigest()}))
            exported = self.reporting.export_current(self.batch)
            path = exported / 'failed_questions.jsonl'
            self.assertTrue(path.exists(), 'empty failed-question target list must be published')
            self.assertEqual(path.read_bytes(), b'')

    def test_fixed_tokens_reused_rounds_and_actual_dedup(self):
        version, rounds = self.publish(self.key)
        rows = self.reporting.expand_modes(self.records.get_version(version), rounds)
        self.assertEqual([r['success_usage']['total_tokens'] for r in rows], [300,320,320,340])
        self.assertEqual([r['second_round_reused'] for r in rows], [False,True,False,True])
        self.assertEqual(rows[0]['success_usage']['completion_tokens'], 150)
        requests = list(self.reporting.iter_actual_requests(self.records))
        cost = self.reporting.aggregate_actual_requests(requests + requests)
        self.assertEqual(cost['attempt_count'], 20)
        self.assertEqual(cost['known_usage']['total_tokens'], 640)
        self.assertEqual(cost['unknown_attempt_count'], 0)
        self.assertNotIn('raw_text', json.dumps(rows))
        self.assertNotIn('message', json.dumps(rows))

    def test_failed_fifth_slot_retains_partial_cost_and_unknown_attempts(self):
        version = self.records.begin_version(self.key)
        samples = []
        for pos in range(5):
            ids = []
            success = None
            for retry in range(1, 6 if pos == 4 else 2):
                attempt = self.records.append(version, 'request_attempt', {
                    'round_execution_id': 'partial', 'sample_position': pos, 'attempt_no': retry})
                ids.append(attempt)
                result = self.records.append(version, 'request_result', {'request_attempt_id': attempt,
                    'status': 'failed' if pos == 4 else 'success',
                    'choice': {'index': 0, 'message': {'content': 'SELECT 1'}} if pos != 4 else None,
                    'usage': {'total_tokens': 20} if pos != 4 else {'total_tokens': 7} if retry == 1 else None})
                if pos != 4:
                    success = result
            samples.append({'sample_position': pos, 'status': 'success' if success else 'failed',
                'successful_request_id': success, 'request_attempt_ids': ids,
                'success_usage': {'total_tokens': 20} if success else None,
                'choice': {'index': 0, 'message': {'content': 'SELECT 1'}} if success else None,
                'error': None if success else 'exhausted'})
        self.records.append(version, 'round_result', {'round_execution_id': 'partial', 'round_no': 1,
            'status': 'failed', 'rc_injected': False, 'actual_parent_round_id': None,
            'example_ids': [str(i) for i in range(9)], 'next_example_ids': [], 'samples': samples,
            'request_attempt_ids': [a for s in samples for a in s['request_attempt_ids']],
            'successful_request_ids': [s['successful_request_id'] for s in samples if s['successful_request_id']],
            'success_usage': None, 'candidates': [], 'selection': None, 'error': 'exhausted'})
        partial = self.records.get_round(version, 'partial')
        self.assertIsNone(partial['success_usage'])
        self.assertEqual(len(partial['successful_request_ids']), 4)
        cost = self.reporting.aggregate_actual_requests(self.reporting.iter_actual_requests(self.records))
        self.assertEqual(cost['attempt_count'], 9)
        self.assertEqual(cost['known_usage']['total_tokens'], 87)
        self.assertEqual(cost['successful']['known_usage']['total_tokens'], 80)
        self.assertEqual(cost['failed_or_unfinished']['known_usage']['total_tokens'], 7)
        self.assertEqual(cost['unknown_attempt_count'], 4)
        self.assertIsNone(cost['usage'])
        failed_rc = copy.deepcopy(partial)
        failed_rc.update(round_execution_id='failed-rc', rc_injected=True, request_attempt_ids=[], successful_request_ids=[])
        for sample in failed_rc['samples']:
            sample.update(status='failed', request_attempt_ids=[], successful_request_id=None, success_usage=None, choice=None)
        self.records.append(version, 'round_result', failed_rc)
        refs = {}
        for mode in MODES:
            first = 'failed-rc' if mode in ('rc_first','rc_both') else 'partial'
            refs[mode] = self.records.append(version, 'mode_result', {'mode': mode, 'status': 'dependency_failed',
                'first_round_id': first, 'second_round_id': None, 'final_candidate_id': None, 'failure_origin': first})
        self.records.seal(version, refs)
        self.index.publish(self.key, version, self.index.claim(self.key), self.records.is_sealed)
        exported = self.reporting.export_current(self.batch)
        native = next(r for r in self.load_rows(exported) if r['task_key'] == asdict(self.key) and r['mode'] == 'native')
        self.assertEqual(native['status'], 'dependency_failed')
        self.assertIsNone(native['success_usage'])
        self.assertEqual(len(native['rounds'][0]['successful_request_ids']), 4)
        self.assertIsNone(native['final_match'])

    def test_missing_usage_leaves_and_reasoning_are_not_zero_or_double_counted(self):
        usage = self.reporting.usage_summary([{'total_tokens': 20, 'completion_tokens': 10,
            'completion_tokens_details': {'reasoning_tokens': 4}}, {'total_tokens': 24}])
        self.assertEqual(usage['usage']['total_tokens'], 44)
        self.assertIsNone(usage['usage']['completion_tokens'])
        self.assertIsNone(usage['usage']['completion_tokens_details'])
        self.assertEqual(usage['known_usage']['completion_tokens'], 10)
        self.assertEqual(usage['known_usage']['total_tokens'], 44)
        self.assertIsNone(self.reporting.usage_summary([None, {}])['known_usage'])

    def test_all_five_groups_export_full_21280_positions_without_creating_stores(self):
        full_manifest = {**self.manifest, 'batch_id': 'full', 'groups': {
            g: {'ids': [str(i) for i in range(n)]} for g,n in
            [('spider_dev',1034), ('bird_dev',1534), ('bird_interact_full',410),
             ('bird_interact_lite',195), ('spider_test',2147)]}}
        full_root = self.root / 'full'
        with DailRecords(full_root, full_manifest), CurrentIndex(full_root / 'current.sqlite3'):
            exported = self.reporting.export_current(full_root)
        self.assertEqual(len(self.load_rows(exported)), 21280)
        self.assertEqual(json.loads((exported / 'summary.json').read_text())['expected_questions'], 5320)
        self.assertEqual(list(full_root.glob('group-*')), [])

    def test_cross_group_same_question_has_distinct_current_and_cost_sources(self):
        first, _ = self.publish(self.key)
        other_key = TaskKey('batch', 'bird_dev', '1')
        second, _ = self.publish(other_key)
        exported = self.reporting.export_current(self.batch)
        rows = self.load_rows(exported)
        self.assertEqual({r['version_id'] for r in rows if r['task_key'] == asdict(self.key)}, {first})
        self.assertEqual({r['version_id'] for r in rows if r['task_key'] == asdict(other_key)}, {second})
        summary = json.loads((exported / 'summary.json').read_text())
        self.assertEqual(summary['actual_cost']['known_usage']['total_tokens'], 1280)

    def test_readonly_export_typed_codec_and_unfinished_attempt(self):
        from decimal import Decimal
        from scripts.baseline_adapters.deepeye.run_store import RunStore
        version, _ = self.publish(self.key)
        unfinished = self.records.begin_version(self.key)
        attempt = self.records.append(unfinished, 'request_attempt', {
            'round_execution_id': 'paused', 'sample_position': 0, 'attempt_no': 1})
        with patch.object(RunStore, 'create', side_effect=AssertionError('read must not create a store')):
            exported = self.reporting.export_current(self.batch)
        requests = [json.loads(line) for line in (exported / 'requests.jsonl').read_text().splitlines()]
        self.assertEqual(next(r for r in requests if r['request_attempt_id'] == attempt)['request_status'], 'unfinished')
        with DailRecords(self.batch, self.manifest, read_only=True) as reader:
            vote_ref = reader.get_round(version, 'A')['candidates'][0]['vote_execution_ref']
            self.assertIsInstance(reader.get_event(version, vote_ref)['rows'][0][0], Decimal)

    def test_export_retains_independent_execution_evidence_once_and_candidate_references(self):
        self.publish(self.key)
        exported = self.reporting.export_current(self.batch)
        self.assertTrue((exported / 'executions.jsonl').exists(), 'independent SQL evidence is missing')
        executions = [json.loads(line) for line in (exported / 'executions.jsonl').read_text().splitlines()]
        self.assertEqual(len(executions), 1)  # All candidates and reference are SELECT 1.
        self.assertEqual(executions[0]['sql'], 'SELECT 1')
        self.assertEqual(executions[0]['result']['status'], 'success')
        candidates = [json.loads(line) for line in (exported / 'candidates.jsonl').read_text().splitlines()]
        self.assertEqual(len(candidates), 20)
        self.assertEqual({c['evaluation']['prediction_execution_id'] for c in candidates}, {executions[0]['execution_id']})
        self.assertEqual({c['evaluation']['reference_execution_id'] for c in candidates}, {executions[0]['execution_id']})

    def test_full_denominator_unknown_paired_tables_and_candidate_traceability(self):
        version, rounds = self.publish(self.key)
        rounds['B']['candidates'][0]['evaluation'].update(bag_equal=False)
        rounds['D']['candidates'][0]['evaluation'].update(bag_equal=None, comparable=False, status='unknown')
        rows = self.reporting.expand_modes(self.records.get_version(version), rounds)
        keys = [self.key, TaskKey('batch', 'bird_dev', '1')]
        summary = self.reporting.aggregate(rows, keys)
        self.assertEqual(summary['expected_mode_positions'], 8)
        finals = {r['mode']: r for r in summary['final_outputs'] if r['group'] == 'spider_dev'}
        self.assertEqual(finals['rc_second']['unknown'], 1)
        pair = next(r for r in summary['candidate_changes'] if r['mode'] == 'rc_first' and r['round_no'] == 1)
        self.assertEqual(pair['decreased'], 1)
        self.assertEqual(pair['incomplete'], 0)
        self.assertEqual(rows[0]['rounds'][0]['candidates'][0]['candidate_id'], 'A-0')
        self.assertEqual(rows[0]['rounds'][0]['candidate_count'], 5)
        bird = [r for r in summary['final_outputs'] if r['group'] == 'bird_dev']
        self.assertTrue(all(r['missing'] == 1 and r['total'] == 1 for r in bird))
        self.assertEqual(len(summary['coverage']), 16)
        self.assertEqual(len(summary['voting']), 16)
        self.assertEqual(len(summary['examples']), 8)

    def test_export_three_question_middle_rerun_and_cross_group_pending(self):
        keys = [TaskKey('batch', 'spider_dev', str(i)) for i in range(3)]
        for key in keys:
            self.publish(key)
        first = self.reporting.export_current(self.batch)
        before = self.load_rows(first)
        self.assertEqual(len(before), 16)
        self.assertEqual(sum(r['status'] == 'pending' for r in before), 4)
        self.assertFalse((self.batch / ('group-' + 'bird_dev'.encode().hex())).exists())
        self.publish(keys[1])
        self.assertIsNone(self.reporting.current_export(self.batch))
        second = self.reporting.export_current(self.batch)
        after = self.load_rows(second)
        self.assertEqual([r for r in before if r['task_key']['question_id'] != '1'],
                         [r for r in after if r['task_key']['question_id'] != '1'])
        self.assertEqual(self.reporting.current_export(self.batch), second)
        cost = json.loads((second / 'summary.json').read_text())['actual_cost']
        self.assertEqual(cost['known_usage']['total_tokens'], 2560)
        self.assertEqual(len((second / 'requests.jsonl').read_text().splitlines()), 80)
        self.assertTrue(first.exists())
        diagnostic = self.reporting.export_current(self.batch, diagnostic_targets=[keys[1]])
        self.assertEqual(len(self.load_rows(diagnostic)), 4)
        self.assertIn('diagnostic', diagnostic.parts)
        self.assertEqual(self.reporting.current_export(self.batch), second)

    def test_switch_during_export_is_consistent_but_not_fresh(self):
        old, _ = self.publish(self.key)
        original = self.reporting.evaluate_candidate
        switched = []
        def switch(*args, **kwargs):
            if not switched:
                switched.append(self.publish(self.key)[0])
            return original(*args, **kwargs)
        with patch.object(self.reporting, 'evaluate_candidate', side_effect=switch):
            exported = self.reporting.export_current(self.batch)
        rows = [r for r in self.load_rows(exported) if r['task_key'] == asdict(self.key)]
        self.assertEqual({r['version_id'] for r in rows}, {old})
        self.assertIsNone(self.reporting.current_export(self.batch))
        saved = json.loads((exported / 'versions.json').read_text())['versions']
        self.assertEqual(next(v['version_id'] for v in saved if v['task_key'] == asdict(self.key)), old)

    def test_source_hash_failure_never_publishes(self):
        self.publish(self.key)
        source = Path(self.manifest['evaluation_source'])
        (source.parent / 'bindings.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.reporting.export_current(self.batch)
        self.assertFalse((self.batch / 'exports/full/latest.json').exists())

    def test_task4_original_input_manifest_source_resolves_once(self):
        self.publish(self.key)
        source = Path(self.manifest['evaluation_source'])
        bindings = json.loads((source.parent / 'bindings.json').read_text())
        original = {'groups': {group: {'rows': [{'question_id': qid, 'evaluation_binding': binding}
                                                for qid,binding in values.items()]}
                               for group,values in bindings.items()}}
        raw = json.dumps(original).encode()
        path = self.root / 'original-inputs.json'
        path.write_bytes(raw)
        source.write_text(json.dumps({'input_manifest': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}))
        exported = self.reporting.export_current(self.batch)
        native = next(r for r in self.load_rows(exported) if r['task_key'] == asdict(self.key) and r['mode'] == 'native')
        self.assertTrue(native['final_match'])

    def test_deepeye_profile_uses_sqlite_timeout_without_changing_batch(self):
        self.publish(self.key)
        self.publish(TaskKey('batch', 'bird_dev', '1'))
        before = (self.batch / 'manifest.json').read_bytes()
        original = self.reporting.execute_sql
        timeouts = []
        def execute(database, sql, *, timeout_seconds):
            timeouts.append(timeout_seconds)
            return original(database, sql, timeout_seconds=timeout_seconds)
        with patch.object(self.reporting, 'execute_sql', side_effect=execute):
            exported = self.reporting.export_current(self.batch, evaluation_profile='deepeye')
        self.assertEqual(timeouts, [600, 600])
        self.assertEqual((self.batch / 'manifest.json').read_bytes(), before)
        self.assertEqual(sum(r['final_match'] is True for r in self.load_rows(exported)), 8)
        saved = json.loads((exported / 'versions.json').read_text())
        self.assertEqual(saved['evaluation_policy']['groups']['bird_dev'],
                         {'timeout_seconds':600, 'question_workers':16, 'query_workers':16})
        self.assertEqual(saved['evaluation_policy']['groups']['spider_dev'],
                         {'timeout_seconds':600, 'question_workers':4, 'query_workers':0})

    def test_parallel_questions_make_progress_without_a_slow_question_blocking(self):
        for qid in ('0', '1', '2'):
            self.publish(TaskKey('batch', 'spider_dev', qid))
        barrier = threading.Barrier(3)
        original = self.reporting.execute_sql
        def execute(database, sql, *, timeout_seconds):
            barrier.wait(timeout=3)
            return original(database, sql, timeout_seconds=timeout_seconds)
        with patch.object(self.reporting, 'execute_sql', side_effect=execute):
            exported = self.reporting.export_current(self.batch, evaluation_profile='deepeye')
        rows = self.load_rows(exported)
        self.assertEqual(sum(r['final_match'] is True for r in rows), 12)
        self.assertEqual(len((exported / 'executions.jsonl').read_text().splitlines()), 3)

    def test_bird_candidates_and_reference_use_shared_query_parallelism(self):
        key = TaskKey('batch', 'bird_dev', '1')
        self.publish(key)
        source = Path(self.manifest['evaluation_source'])
        binding_path = source.parent / 'bindings.json'
        bindings = json.loads(binding_path.read_text())
        bindings['bird_dev']['1']['reference_sql'] = 'SELECT 2'
        raw = json.dumps(bindings).encode()
        binding_path.write_bytes(raw)
        source.write_text(json.dumps({'bindings':'evaluation/bindings.json',
                                     'sha256':hashlib.sha256(raw).hexdigest()}))
        barrier = threading.Barrier(2)
        original = self.reporting.execute_sql
        def execute(database, sql, *, timeout_seconds):
            barrier.wait(timeout=3)
            return original(database, sql, timeout_seconds=timeout_seconds)
        with patch.object(self.reporting, 'execute_sql', side_effect=execute):
            exported = self.reporting.export_current(self.batch, evaluation_profile='deepeye')
        rows = [r for r in self.load_rows(exported) if r['task_key']['group'] == 'bird_dev']
        self.assertEqual([r['final_match'] for r in rows], [False] * 4)
        self.assertEqual(len((exported / 'executions.jsonl').read_text().splitlines()), 2)

    def test_profile_runs_groups_independently_and_pg_timeout_is_not_retried(self):
        # The two group queries must overlap; a serial group loop breaks the barrier.
        self.publish(self.key)
        self.publish(TaskKey('batch', 'bird_dev', '1'))
        source = Path(self.manifest['evaluation_source'])
        binding_path = source.parent / 'bindings.json'
        bindings = json.loads(binding_path.read_text())
        bindings['bird_dev']['1']['database'] = {'dialect':'postgresql', 'database_id':'pg_fixture'}
        raw = json.dumps(bindings).encode()
        binding_path.write_bytes(raw)
        source.write_text(json.dumps({'bindings':'evaluation/bindings.json',
                                     'sha256':hashlib.sha256(raw).hexdigest()}))
        barrier = threading.Barrier(2)
        original = self.reporting.execute_sql
        seen = []
        def execute(database, sql, *, timeout_seconds):
            seen.append((database['dialect'], timeout_seconds))
            barrier.wait(timeout=3)
            if database['dialect'] == 'postgresql':
                return {'status':'timeout', 'rows':[], 'columns':[], 'error':{'type':'QueryCanceled'}}
            return original(database, sql, timeout_seconds=timeout_seconds)
        with patch.object(self.reporting, 'execute_sql', side_effect=execute):
            exported = self.reporting.export_current(self.batch, evaluation_profile='deepeye')
        self.assertCountEqual(seen, [('sqlite',600), ('postgresql',30)])
        rows = [r for r in self.load_rows(exported) if r['task_key']['group'] == 'bird_dev']
        self.assertTrue(all(r['final_match'] is None for r in rows))
        saved = json.loads((exported / 'versions.json').read_text())
        self.assertEqual(saved['evaluation_policy']['groups']['bird_dev'],
                         {'timeout_seconds':30, 'question_workers':5, 'query_workers':0})
        self.assertIsNone(saved['sql_timeout_seconds'])

    def test_parallel_execution_failure_cannot_publish_a_partial_export(self):
        self.publish(self.key)
        with patch.object(self.reporting, 'execute_sql', side_effect=RuntimeError('injected export failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected export failure'):
                self.reporting.export_current(self.batch, evaluation_profile='deepeye')
        self.assertFalse((self.batch / 'exports/full/latest.json').exists())


if __name__ == '__main__':
    unittest.main()
