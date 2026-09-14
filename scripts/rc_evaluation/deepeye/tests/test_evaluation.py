import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.rc_evaluation.deepeye.evaluation import evaluate_run, compare_evaluations, schema_coverage


def result(value):
    return {'result_type': 'success', 'result_cols': ['x'], 'result_rows': [[value]]}


class EvaluationTests(unittest.TestCase):
    def test_spider2_gold_subset_matches_external_ids_not_original_row_positions(self):
        from copy import deepcopy
        from scripts.rc_evaluation.deepeye.evaluation import _references, _reference
        original = [{'index': f'local{i:03d}', 'db_id': 'demo'} for i in range(280)]
        selected = original[159:]
        path = self.root / 'spider2-gold.json'
        path.write_text(json.dumps([{**row, 'gold_sql': f'SELECT {row["index"][5:]}'}
                                    for row in reversed(selected)]))
        bindings = [{'task_key': f'spider2/lite/s:{row["index"]}', 'benchmark': 'spider2',
            'partition': 'spider2/lite', 'external_id': row['index'], 'database_id': 'demo',
            'reference': {'path': str(path), 'source_row': position}}
            for position, row in enumerate(original) if position >= 159]
        before = deepcopy(bindings)
        records, _ = _references({}, bindings)
        self.assertEqual(len(records), 121)
        for binding in bindings:
            reference = _reference(binding, records)
            self.assertEqual(reference['status'], 'available')
            self.assertEqual(reference['sql'], f'SELECT {binding["external_id"][5:]}')
        self.assertEqual(bindings, before, 'Reference answers must not be copied into run bindings')

    def test_spider2_existing_sql_reference_fields_remain_supported(self):
        from scripts.rc_evaluation.deepeye.evaluation import _reference
        binding = {'task_key': 'spider2/lite/s:local003', 'benchmark': 'spider2', 'database_id': 'demo'}
        for field in ('sql', 'SQL', 'query'):
            with self.subTest(field=field):
                row = {'db_id': 'demo', field: 'SELECT 1', 'gold_sql': 'SELECT 2'}
                result = _reference(binding, {binding['task_key']: [row]})
                self.assertEqual(result['status'], 'available')
                self.assertEqual(result['sql'], 'SELECT 1')

    def test_supplied_spider2_profile_binds_all_121_reference_sqls_by_identity(self):
        from scripts.baseline_adapters.deepeye.workloads import load_workload, question_rows
        from scripts.rc_evaluation.deepeye.evaluation import _references, _reference
        root = Path(__file__).resolve().parents[4]
        workload = load_workload(root / 'config/deepeye/spider2_lite_rc3.json')
        self.assertEqual(Path(workload.get('reference_questions', '')).resolve(),
                         root / 'scripts/spider2_lite/gold_sql.json')
        if not Path(workload['questions']).is_file() or not Path(workload['reference_questions']).is_file():
            self.skipTest('Supplied Spider2 source/reference artifacts are not installed')
        rows = question_rows(workload)
        bindings = [{'task_key': f'spider2/lite/s:{row["external_id"]}', 'benchmark': 'spider2',
            'partition': 'spider2/lite', 'external_id': row['external_id'],
            'database_id': row['database_id'], 'reference': {'path': workload['reference_questions'],
                                                          'source_row': row['source_row']}}
            for row in rows]
        records, _ = _references({}, bindings)
        self.assertEqual(len(rows), 121)
        self.assertTrue(any(row['source_row'] >= 121 for row in rows))
        self.assertTrue(all(_reference(binding, records)['status'] == 'available' for binding in bindings))

    def test_rc_evaluations_of_different_versions_cannot_be_paired(self):
        for version in (2, 3):
            run = self.make_run(f'rc{version}', condition='rc',
                mutate=lambda m, version=version: m.update(rc_version=version))
            report = self.evaluate(run, f'eval{version}')
            self.assertEqual(report['rc_version'], version)
            self.assertEqual(report['gold_corrected'], version == 3)
        with self.assertRaisesRegex(ValueError, 'version'):
            compare_evaluations(self.root / 'eval2', self.root / 'eval3', self.root / 'paired')

    def test_token_pairing_rejects_contract_version_mismatch(self):
        from scripts.rc_evaluation.deepeye.evaluation import stage_token_pairs
        manifest = {'condition': 'rc', 'target_stage': 'sql_generation', 'rc_version': 3,
                    'contracts': {'a': {'rc_version': 2}}, 'source_checkpoints': {}}
        with self.assertRaisesRegex(ValueError, 'version'):
            stage_token_pairs(manifest, [], [])

    def test_retained_stage_token_pair_excludes_failures_and_missing_usage_without_zero_fill(self):
        from scripts.rc_evaluation.deepeye import evaluation
        from scripts.rc_evaluation.deepeye.source import api_trace
        self.assertTrue(hasattr(evaluation, 'stage_token_pairs'), 'Native-source token pairing is missing')
        def events(aid, tokens=10, complete=True):
            return [
                {'attempt_id': aid, 'kind': 'sampling_group_start', 'payload': {'group_id': aid, 'target_n': 1}},
                {'attempt_id': aid, 'kind': 'sampling_group_result', 'payload': {'group_id': aid, 'target_n': 1,
                    'success_count': int(complete), 'complete': complete}},
                {'attempt_id': aid, 'kind': 'sample_result', 'payload': {'group_id': aid, 'sample_index': 0,
                    'succeeded': complete, 'rc_applied': True, 'usage': None if tokens is None else {
                        'prompt_tokens': tokens - 2, 'completion_tokens': 2, 'total_tokens': tokens,
                        'reasoning_tokens': 1}}}]
        native = api_trace(events('native'))
        manifest = {'target_stage': 'sql_generation', 'condition': 'rc', 'source_checkpoints': {
            key: {'stages': {'sql_generation': {'api_trace': native}}}
            for key in ('ok', 'failed', 'unknown', 'partial', 'missing')}}
        rows, trace = [], []
        for key, tokens, status, complete in [('ok', 15, 'succeeded', True), ('failed', 12, 'failed', True),
                                             ('unknown', None, 'succeeded', True), ('partial', 8, 'succeeded', False)]:
            rows.append({'attempt_id': key, 'item_key': key, 'stage': 'sql_generation', 'status': status,
                         'payload': {'rc_participation': {'status': 'participating'}}})
            trace.extend(events(key, tokens, complete))
        report = evaluation.stage_token_pairs(manifest, rows, trace)
        self.assertEqual(report['summary']['eligible_pairs'], 1)
        self.assertEqual(report['summary']['native_tokens']['total_tokens'], 10)
        self.assertEqual(report['summary']['rc_tokens']['total_tokens'], 15)
        self.assertEqual(report['items']['ok']['delta_tokens']['total_tokens'], 5)
        for key, reason in [('failed', 'rc_stage_failed'), ('unknown', 'rc_usage_incomplete'),
                            ('partial', 'rc_sampling_incomplete'), ('missing', 'rc_stage_missing')]:
            self.assertIn(reason, report['items'][key]['exclusions'])
            self.assertIsNone(report['items'][key]['delta_tokens'])
        self.assertEqual(report['items']['ok']['rc']['reasoning_tokens'], 1)

    def test_legacy_and_zero_call_native_sources_do_not_claim_full_budget_pairs(self):
        from scripts.rc_evaluation.deepeye import evaluation
        self.assertTrue(hasattr(evaluation, 'stage_token_pairs'), 'Native-source token pairing is missing')
        manifest = {'target_stage': 'sql_revision', 'condition': 'rc', 'source_checkpoints': {
            'legacy': {'stages': {'sql_revision': {'api_trace': {'requests': 1, 'complete': True}}}},
            'zero': {'stages': {'sql_revision': {'api_trace': {'requests': 0, 'complete': True}}}}}}
        rows = [{'attempt_id': key, 'item_key': key, 'stage': 'sql_revision', 'status': 'succeeded',
                 'payload': {'execution_origin': 'reused_no_native_llm_call'}} for key in ('legacy', 'zero')]
        report = evaluation.stage_token_pairs(manifest, rows, [])
        self.assertIn('native_effective_sampling_unavailable', report['items']['legacy']['exclusions'])
        self.assertIn('native_zero_call_target', report['items']['zero']['exclusions'])
        self.assertEqual(report['summary']['eligible_pairs'], 0)
        self.assertIsNone(report['summary']['native_tokens'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reference = self.root / 'reference.jsonl'
        self.reference.write_text(json.dumps({'instance_id': 'a', 'selected_database': 'demo', 'category': 'Query',
                                             'sol_sql': ['SELECT x FROM t'], 'preprocess_sql': [], 'clean_up_sqls': [],
                                             'test_cases': []}) + '\n')
        self.network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def make_run(self, name='run', target='sql_selection', include_missing=True, revision=None, condition='none', mutate=None):
        from scripts.rc_evaluation.deepeye.evaluation import _hash
        keys = ['lite/a', 'lite/missing'] if include_missing else ['lite/a']
        generation = {'artifact': {'sql_candidates': ['bad', 'good', 'good']}}
        revised = {'artifact': {'sql_candidates_after_revision': revision or ['good', 'bad', 'good']}}
        manifest = {'format': 'deepeye-rc-evaluation-run-v1', 'target_stage': target, 'condition': condition,
                    'repeat_id': '1', 'source_manifest_fingerprint': 'baseline', 'effective_config': {},
                    'items': [{'task_key': key, 'database_id': 'demo', 'question_sha256': 'q'} for key in keys],
                    'source_checkpoints': {key: {'upstream_state_sha256': 'prefix', 'stages': {
                        'sql_generation': {'attempt_id': 'g', 'payload': generation},
                        'sql_revision': {'attempt_id': 'r', 'payload': revised}}} for key in keys}}
        for checkpoint in manifest['source_checkpoints'].values():
            for stage in checkpoint['stages'].values():
                stage['payload_sha256'] = _hash(stage['payload'])
        if mutate:
            mutate(manifest)
        path = self.root / name
        with RunStore.create(path, manifest) as store:
            for key in keys:
                attempt = store.begin_attempt(key, target, 'input')
                payload = {'sql_generation': generation, 'sql_revision': revised,
                           'sql_selection': {'artifact': {'final_selected_sql': 'good'}}}[target]
                store.finish_attempt(attempt, 'succeeded', payload)
        return path

    def evaluate(self, run, output='eval', executor=None):
        return evaluate_run(run, self.root / output, {'lite': self.reference}, env_file=self.root / 'no-env',
                            database_version='fixture-v1', execute_fn=executor or
                            (lambda key, db, sql: result(1 if sql in ('good', 'SELECT x FROM t') else 0)))

    def test_fixed_denominator_missing_reference_and_separate_append_only_store(self):
        run = self.make_run()
        with RunStore.open(run, read_only=True) as store:
            before = store.attempts()
        report = self.evaluate(run)
        self.assertEqual(report['summary']['tasks'], 2)
        self.assertEqual(report['summary']['bag_equal'], 1)
        self.assertEqual(report['summary']['unknown'], 1)
        self.assertEqual(report['summary']['bag_equal_rate'], 0.5)
        self.assertEqual(report['items']['lite/a']['selection']['selected_when_pool_correct'], True)
        with RunStore.open(run, read_only=True) as store:
            self.assertEqual(store.attempts(), before)
        with RunStore.open(self.root / 'eval', read_only=True) as store:
            self.assertTrue(store.verify()['ok'])
            count = len(store.attempts())
        self.evaluate(run, executor=lambda *args: self.fail('completed evaluation executed again'))
        with RunStore.open(self.root / 'eval', read_only=True) as store:
            self.assertEqual(len(store.attempts()), count)

    def test_revision_fixed_slots_repair_harm_unchanged_and_missing(self):
        report = self.evaluate(self.make_run(target='sql_revision', revision=['good', 'bad']))
        revision = report['items']['lite/a']['revision']
        self.assertEqual(revision['input_slots'], 3)
        self.assertEqual([slot['change'] for slot in revision['slots']], ['repair', 'harm', 'unknown'])
        self.assertEqual(revision['unknown'], 1)
        self.assertTrue(revision['before']['has_correct'])
        self.assertTrue(revision['after']['has_correct'])

    def test_execution_errors_remain_unknown_and_records_bind_sql_database_version(self):
        run = self.make_run(include_missing=False)
        report = self.evaluate(run, executor=lambda key, db, sql: {'result_type': 'timeout'})
        self.assertIsNone(report['items']['lite/a']['bag_equal'])
        with RunStore.open(self.root / 'eval', read_only=True) as store:
            executions = [e for e in store.events() if e['kind'] == 'sql_execution']
            self.assertTrue(executions)
            self.assertEqual(executions[0]['payload']['database_version'], 'fixture-v1')
            self.assertEqual(executions[0]['payload']['database_id'], 'demo')
            self.assertIn('sql_sha256', executions[0]['payload'])

    def test_reference_cleanup_and_duplicates_rejected_without_execution(self):
        run = self.make_run(include_missing=False)
        row = json.loads(self.reference.read_text())
        row['clean_up_sqls'] = ['DELETE FROM t']
        self.reference.write_text(json.dumps(row))
        report = self.evaluate(run, executor=lambda *a: self.fail('unsafe reference executed'))
        self.assertEqual(report['items']['lite/a']['reference']['status'], 'unsupported_reference_setup')
        self.reference.write_text(json.dumps(row) + '\n' + json.dumps(row))
        report = self.evaluate(run, output='eval-duplicates', executor=lambda *a: self.fail('ambiguous reference executed'))
        self.assertEqual(report['items']['lite/a']['reference']['status'], 'duplicate_reference')

    def test_current_api_usage_counts_failed_attempts_and_nested_native_response(self):
        run = self.make_run(include_missing=False)
        with RunStore.open(run) as store:
            attempt = store.begin_attempt('lite/a', 'sql_selection', 'retry')
            store.append_event(attempt, 'api_request', {'call_id': 'one'})
            store.append_event(attempt, 'api_response', {'call_id': 'one', 'response': {
                'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}}})
            store.finish_attempt(attempt, 'failed', {'error': 'parse failed'})
        report = self.evaluate(run)
        self.assertEqual(report['items']['lite/a']['usage']['api_requests'], 1)
        self.assertEqual(report['items']['lite/a']['usage']['usage']['total_tokens'], 18)

    def test_rc_usage_counts_interrupted_and_resumed_requests_from_current_target_events(self):
        from scripts.rc_evaluation.deepeye.injection import render_rc_block
        contract = {'task_key': 'lite/a', 'db_id': 'demo', 'question': 'Return x', 'evidence': '',
                    'round2': {field: 'fixed meaning' for field in ('population', 'row_grain',
                               'column_role', 'derivation', 'filter_policy', 'meta_review')}}
        block = render_rc_block(contract)
        run = self.make_run(include_missing=False, condition='rc',
                            mutate=lambda manifest: manifest.update(contracts={'lite/a': contract}))
        with RunStore.open(run) as store:
            interrupted = store.begin_attempt('lite/a', 'sql_selection', 'interrupted')
            store.append_event(interrupted, 'api_request', {'call_id': 'interrupted-call',
                'kwargs': {'messages': [{'role': 'user', 'content': block}]}})
        self.assertEqual(self.evaluate(run)['summary']['usage']['rc_actual_requests'], 1)
        with RunStore.open(run) as store:
            resumed = store.begin_attempt('lite/a', 'sql_selection', 'resumed')
            store.append_event(resumed, 'api_request', {'call_id': 'resumed-call',
                'kwargs': {'messages': [{'role': 'user', 'content': block}]}})
            store.append_event(resumed, 'api_response', {'call_id': 'resumed-call', 'response': {
                'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}}})
            store.finish_attempt(resumed, 'succeeded', {'artifact': {'final_selected_sql': 'good'},
                'rc_participation': {'actual_request_count': 99}})
            other_stage = store.begin_attempt('lite/a', 'sql_generation', 'not-target')
            store.append_event(other_stage, 'api_request', {'call_id': 'wrong-stage',
                'kwargs': {'messages': [{'role': 'user', 'content': block}]}})
            other_task = store.begin_attempt('lite/other', 'sql_selection', 'not-this-task')
            store.append_event(other_task, 'api_request', {'call_id': 'wrong-task',
                'kwargs': {'messages': [{'role': 'user', 'content': block}]}})
        usage = self.evaluate(run, output='resumed-eval')['items']['lite/a']['usage']
        self.assertEqual(usage['rc_actual_requests'], 2)
        self.assertEqual(usage['api_requests'], 3)
        self.assertEqual(usage['unanswered_requests'], 2)

    def test_rc_usage_does_not_count_source_wrong_contract_or_none_metadata(self):
        from scripts.rc_evaluation.deepeye.injection import render_rc_block
        from scripts.rc_evaluation.deepeye.evaluation import _usage
        contract = {'round2': {field: 'bound meaning' for field in ('population', 'row_grain',
                               'column_role', 'derivation', 'filter_policy', 'meta_review')}}
        other = {'round2': {field: 'different meaning' for field in contract['round2']}}
        attempts = [{'attempt_id': 'current', 'item_key': 'lite/a', 'stage': 'sql_selection',
                     'payload': {'rc_participation': {'actual_request_count': 99}}}]
        def request(attempt_id, block):
            return {'attempt_id': attempt_id, 'kind': 'api_request', 'payload': {'call_id': attempt_id,
                    'kwargs': {'messages': [{'role': 'user', 'content': block}]}}}
        events = [request('current', render_rc_block(other)), request('source-only', render_rc_block(contract))]
        manifest = {'condition': 'rc', 'target_stage': 'sql_selection', 'contracts': {'lite/a': contract}}
        self.assertEqual(_usage(attempts, events, manifest)['rc_actual_requests'], 0)
        events[0] = request('current', render_rc_block(contract))
        manifest['condition'] = 'none'
        self.assertEqual(_usage(attempts, events, manifest)['rc_actual_requests'], 0)

    def test_rc_evaluation_rejects_changed_bound_prompt_production_before_execution(self):
        run = self.make_run(include_missing=False, condition='rc', mutate=lambda manifest:
            manifest.update(sources={'rc_evaluation_code_sha256': 'older-production-version'}))
        with self.assertRaisesRegex(ValueError, 'corresponding production version'):
            self.evaluate(run, executor=lambda *args: self.fail('mismatched prompt version reached executor'))
        self.assertFalse((self.root / 'eval').exists())

    def test_native_shortlist_retention_and_conditional_rate_use_known_pool_only(self):
        run = self.make_run(include_missing=True)
        with RunStore.open(run) as store:
            attempt = store.begin_attempt('lite/a', 'sql_selection', 'fresh')
            store.append_event(attempt, 'component_result', {'component': 'selection.shortlist',
                'result': [['good', 'rendered result', 1.0, 0.01]]})
            store.finish_attempt(attempt, 'succeeded', {'artifact': {'final_selected_sql': 'good'}})
        report = self.evaluate(run)
        selection = report['items']['lite/a']['selection']
        self.assertEqual(selection['branch'], 'single_shortlist_candidate')
        self.assertTrue(selection['shortlist_retains_correct'])
        self.assertEqual(report['summary']['selection']['eligible_tasks'], 1)
        self.assertEqual(report['summary']['selection']['unknown_eligibility_tasks'], 1)
        self.assertEqual(report['summary']['selection']['selected_when_pool_correct_rate'], 1.0)
        self.assertEqual(report['summary']['generation']['pool_has_correct_tasks'], 1)
        self.assertEqual(report['summary']['generation']['pool_has_correct_rate'], 0.5)

    def test_generation_slots_unique_sql_and_ordered_pool_metrics(self):
        report = self.evaluate(self.make_run(target='sql_generation', include_missing=False))
        pool = report['items']['lite/a']['generation']
        self.assertEqual((pool['slots'], pool['unique_sql'], pool['correct']), (3, 2, 2))
        self.assertTrue(report['items']['lite/a']['ordered_equal'])

    def test_changed_database_version_or_source_config_rejects_resume_and_pairing(self):
        run = self.make_run(include_missing=False)
        self.evaluate(run)
        with self.assertRaises(ValueError):
            evaluate_run(run, self.root / 'eval', {'lite': self.reference}, env_file=self.root / 'unused',
                         database_version='different', execute_fn=lambda *a: result(1))
        evaluate_run(run, self.root / 'different', {'lite': self.reference}, env_file=self.root / 'unused',
                     database_version='different', execute_fn=lambda *a: result(1))
        with self.assertRaises(ValueError):
            compare_evaluations(self.root / 'eval', self.root / 'different', self.root / 'invalid-pair')

    def test_consistency_shortcut_requires_success_and_recorded_threshold(self):
        from scripts.rc_evaluation.deepeye.evaluation import _selection_trace
        events = [{'kind': 'component_result', 'payload': {'component': 'selection.shortlist',
                  'result': [['good', 'result', 0.8, 0.1], ['bad', 'result', 0.2, 0.1]]}}]
        attempt = {'status': 'succeeded', 'payload': {'artifact': {'final_selected_sql': 'good'}}}
        self.assertEqual(_selection_trace(events, [], attempt, 0.8)['branch'], 'consistency_shortcut')
        self.assertEqual(_selection_trace(events, [], attempt, 0.9)['branch'], 'unknown')
        attempt['status'] = 'failed'
        self.assertEqual(_selection_trace(events, [], attempt, 0.8)['branch'], 'unknown')

    def test_reused_trace_comes_from_bound_source_and_cost_stays_zero(self):
        from scripts.rc_evaluation.deepeye.evaluation import _selection_trace
        events = [{'kind': 'component_result', 'payload': {'component': 'selection.shortlist',
                  'result': [['good', 'result', 1.0, 0.1]]}}]
        attempt = {'status': 'succeeded', 'payload': {'execution_origin': 'reused_no_native_llm_call',
                                                     'artifact': {'final_selected_sql': 'good'}}}
        got = _selection_trace([], events, attempt, 0.8)
        self.assertEqual(got['branch'], 'single_shortlist_candidate')
        self.assertEqual(got['trace_provenance'], 'source_attempt')
        attempt['payload']['execution_origin'] = 'reused_unchanged_upstream'
        self.assertEqual(_selection_trace([], events, attempt, 0.8)['trace_provenance'], 'source_attempt')

    def test_missing_usage_metadata_is_explicit_not_reported_zero_cost(self):
        run = self.make_run(include_missing=False)
        with RunStore.open(run) as store:
            attempt = store.begin_attempt('lite/a', 'sql_selection', 'partial')
            store.append_event(attempt, 'api_request', {'call_id': 'pending'})
        report = self.evaluate(run)
        usage = report['items']['lite/a']['usage']
        self.assertEqual(usage['api_requests'], 1)
        self.assertFalse(usage['usage_complete'])
        self.assertEqual(usage['unanswered_requests'], 1)

    def test_schema_missing_or_ambiguous_relations_never_claim_column_coverage(self):
        self.assertEqual(schema_coverage('WITH q AS (SELECT a FROM t) SELECT a FROM q', {'t': ['a']})['status'], 'unknown')
        self.assertEqual(schema_coverage('SELECT count(*) FROM t', {'t': []})['status'], 'unknown')
        self.assertEqual(schema_coverage('SELECT "T"."A" FROM "T"', {'T': ['A']})['column_coverage'], 1.0)
        self.assertEqual(schema_coverage('SELECT t.a FROM t', {'t': ['A']})['column_coverage'], 0.0)

    def test_using_and_natural_joins_preserve_table_diagnostics_but_columns_are_unknown(self):
        for sql in ('SELECT a.x FROM a JOIN b USING (id)',
                    'SELECT a.x FROM a NATURAL JOIN b',
                    'SELECT a.x FROM a NATURAL LEFT JOIN b'):
            with self.subTest(sql=sql):
                coverage = schema_coverage(sql, {'a': ['x'], 'b': []})
                self.assertEqual(coverage['status'], 'unknown')
                self.assertIsNone(coverage['column_coverage'])
                self.assertEqual(coverage['reason'], 'implicit_join_columns_unresolved')
                self.assertEqual(coverage['table_coverage'], 1.0)
                self.assertEqual(coverage['retained_tables'], 2)
                self.assertEqual(coverage['retained_columns'], 1)

    def test_reference_multiple_sql_and_database_mismatch_stay_unknown(self):
        run = self.make_run(include_missing=False)
        row = json.loads(self.reference.read_text())
        row['sol_sql'] = ['SELECT 1', 'SELECT 2']
        self.reference.write_text(json.dumps(row))
        report = self.evaluate(run, executor=lambda *a: self.fail('ambiguous reference executed'))
        self.assertEqual(report['items']['lite/a']['reference']['status'], 'invalid_reference_sql')
        row['selected_database'] = 'wrong'
        self.reference.write_text(json.dumps(row))
        report = self.evaluate(run, output='wrong-db', executor=lambda *a: self.fail('wrong database executed'))
        self.assertEqual(report['items']['lite/a']['reference']['status'], 'reference_database_mismatch')

    def test_paired_comparison_requires_source_and_input_identity_and_keeps_unknown(self):
        self.evaluate(self.make_run('left'), 'left-eval')
        self.evaluate(self.make_run('right', condition='rc', mutate=lambda m: m.update(contracts={'lite/a': {'round2': 'frozen'}})), 'right-eval')
        report = compare_evaluations(self.root / 'left-eval', self.root / 'right-eval', self.root / 'paired')
        self.assertEqual(report['summary']['tasks'], 2)
        self.assertEqual(report['summary']['unknown'], 1)
        self.assertEqual(report['summary']['unchanged_correct'], 1)
        self.assertEqual(report['summary']['ordered']['unchanged_correct'], 1)
        altered = self.make_run('altered', include_missing=False)
        self.evaluate(altered, 'altered-eval')
        with self.assertRaises(ValueError):
            compare_evaluations(self.root / 'left-eval', self.root / 'altered-eval', self.root / 'bad-paired')

    def test_schema_coverage_is_best_effort_and_ambiguous_columns_unknown(self):
        known = schema_coverage('SELECT t.a FROM t WHERE t.b > 0', {'t': ['a']})
        self.assertEqual(known['table_coverage'], 1.0)
        self.assertEqual(known['column_coverage'], 0.5)
        unknown = schema_coverage('SELECT a FROM t JOIN u ON t.id=u.id', {'t': ['a', 'id'], 'u': ['id']})
        self.assertIsNone(unknown['column_coverage'])
        self.assertEqual(unknown['status'], 'unknown')
        self.assertIsNone(schema_coverage('SELECT * FROM t', {'t': ['a']})['column_coverage'])

    def test_tampered_embedded_source_payload_rejected_before_executor(self):
        def mutate(manifest):
            manifest['source_checkpoints']['lite/a']['stages']['sql_generation']['payload']['artifact']['sql_candidates'] = ['poison']
        run = self.make_run(include_missing=False, mutate=mutate)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.evaluate(run, executor=lambda *a: self.fail('tampered checkpoint reached executor'))
        self.assertFalse((self.root / 'eval').exists())

    def test_paired_comparison_rejects_different_pg_identity_upstream_and_source_code(self):
        self.evaluate(self.make_run('first', include_missing=False), 'first-eval')
        mutations = [lambda m: m['effective_config'].update(postgres={'host': 'another'}),
                     lambda m: m['source_checkpoints']['lite/a'].update(upstream_state_sha256='different'),
                     lambda m: m.update(sources={'code': {'current': 'different'}})]
        for index, mutate in enumerate(mutations):
            run = self.make_run('other' + str(index), include_missing=False, mutate=mutate)
            self.evaluate(run, 'other-eval' + str(index))
            with self.assertRaises(ValueError):
                compare_evaluations(self.root / 'first-eval', self.root / ('other-eval' + str(index)), self.root / ('pair' + str(index)))

    def test_real_controller_snapshot_and_native_tagged_payload_evaluate_offline(self):
        from scripts.rc_evaluation.deepeye.tests.test_source import make_source
        from scripts.rc_evaluation.deepeye.tests.test_runner import experiment_manifest
        from scripts.rc_evaluation.deepeye.source import snapshot_source
        source = self.root / 'native-source'
        tasks = make_source(source, calls=0)
        snapshot = snapshot_source(source, tasks, 'sql_revision')
        manifest = experiment_manifest(snapshot)
        record = snapshot['source_checkpoints']['lite/a']['stages']['sql_revision']
        with RunStore.create(self.root / 'native-run', manifest) as store:
            attempt = store.begin_attempt('lite/a', 'sql_revision', 'replay')
            store.finish_attempt(attempt, 'succeeded', {**record['payload'], 'execution_origin': 'reused_no_native_llm_call'})
        reference = json.loads(self.reference.read_text())
        reference['selected_database'] = 'db'
        self.reference.write_text(json.dumps(reference))
        report = self.evaluate(self.root / 'native-run', executor=lambda *args: result(1))
        self.assertTrue(report['items']['lite/a']['bag_equal'])
        self.assertEqual(report['items']['lite/a']['usage']['api_requests'], 0)
        self.assertEqual(report['items']['lite/a']['revision']['input_slots'], 1)

    def test_native_evaluator_binds_actual_connection_and_refuses_ssl_mismatch_before_sql(self):
        from scripts.rc_evaluation.deepeye.evaluation import _executor
        env = self.root / 'pg.env'
        env.write_text('PG_HOST=fixture\nPG_PORT=5433\nPG_USER=reader\nPG_PASSWORD=secret\nPG_SSLMODE=require\n')
        with _executor(env, {}) as (_, identity):
            self.assertEqual(identity['host'], 'fixture')
            self.assertEqual(identity['port'], 5433)
            self.assertEqual(identity['principal'], 'reader')
            self.assertEqual(identity['sslmode'], 'require')
            self.assertNotIn('secret', json.dumps(identity))
        with self.assertRaises(ValueError):
            with _executor(env, {'postgres': {'sslmode': 'disable'}}):
                self.fail('different TLS identity accepted')


if __name__ == '__main__':
    unittest.main()
