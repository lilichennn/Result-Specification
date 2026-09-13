"""Offline behavioral tests for explicit successful-checkpoint inheritance."""
from contextlib import ExitStack
import copy
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'baselines/DeepEye-SQL'))

from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataItem
from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
from scripts.baseline_adapters.deepeye.precompute_pipeline import question_row
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, STAGE_METHODS, _checkpoint
from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
from scripts.baseline_adapters.deepeye.run_usage import observed_usage

try:
    from scripts.baseline_adapters.deepeye.run_inheritance import inherit_checkpoints
except ImportError:
    inherit_checkpoints = None


def cost(tokens=0):
    return {'prompt_tokens': tokens, 'completion_tokens': 0, 'total_tokens': tokens}


def make_item(identity):
    schema = {'db_id': 'db', 'db_path': 'db', 'db_type': 'postgresql',
              'tables': {'t': {'table_name': 't', 'columns': {'x': {
                  'column_type': 'INTEGER', 'primary_key': False, 'foreign_keys': []}}}}}
    examples = [{'question': f'Example {number}', 'SQL': 'SELECT 1', 'db_id': f'train{number}'}
                for number in range(3)]
    return BirdInteractDataItem(question_id=0, instance_id=identity, question='Return x',
        evidence='Ascending order', database_id='db', database_path='db', database_schema=schema,
        database_schema_after_value_retrieval=copy.deepcopy(schema), question_keywords=['x'],
        retrieved_values={}, value_retrieval_time=2.0, value_retrieval_llm_cost=cost(2),
        total_time=2.0, total_llm_cost=cost(2), few_shot_examples=examples,
        few_shot_preparation_metadata={'mode': 'static_independent_bird_train', 'num_examples': 3})


def manifest(tasks, upgrade=False):
    config = {
        'profile': 'bounded-config',
        'chat': {'model': 'offline', 'endpoint': 'https://invalid.test/v1', 'max_tokens': 6144,
                 'temperature': 0.6, 'thinking_budget': None, 'timeout_seconds': 300,
                 'sdk_max_retries': 0, 'native_ask_attempts': 2, 'extractor_max_retries': 2,
                 'n_call_strategy': 'split', 'max_request_n': 1},
        'postgres': {'host': 'invalid.test', 'port': 5432, 'principal': 'reader',
                     'sslmode': 'prefer', 'read_only': True, 'statement_timeout_seconds': 30,
                     'execution_policy': {
                         'version': 'postgres-original-sql-v1', 'meta_fence': False,
                         'sql_rewrite': False, 'result_row_limit': None,
                         'search_path': 'pg_catalog,public',
                         'single_statement': 'extended_protocol', 'read_only': True}},
        'dataset': {'sql_execution_timeout_seconds': 30, 'max_value_example_length': 100},
        'few_shot': {'mode': 'static_independent_bird_train', 'examples_per_item': 3},
        'workers': 4, 'inner_workers': 8,
        'admission': {'enabled': True, 'model': {'initial_limit': 50}, 'postgres_limit': 10},
        'stages': {
            'schema_linking': {'direct_linking_sampling_budget': 1,
                               'reversed_linking_sampling_budget': 1, 'value_distance_threshold': 0.05},
            'sql_generation': {'dc_sampling_budget': 1, 'skeleton_sampling_budget': 1,
                               'icl_sampling_budget': 1},
            'sql_revision': {'checker_sampling_budget': 1, 'checkers': ['SyntaxChecker']},
            'sql_selection': {'evaluator_sampling_budget': 1, 'filter_top_k_sql': 2,
                              'shortcut_consistency_score_threshold': 0.8, 'timing_refine_repeat': 2},
        },
    }
    code = {'baseline_python_sha256': 'native', 'adapter_python_sha256': 'adapter-before',
            'entrypoints_sha256': 'entry-before', 'dependency_locks_sha256': 'dependencies'}
    if upgrade:
        config.update(workers=50, inner_workers=12,
                      scheduler={'mode': 'pipeline_slots', 'concurrency_unit': 'question_pipeline'},
                      admission={'enabled': True, 'adaptive': True,
                                 'pipeline': {'initial_limit': 50}, 'postgres_limit': 10})
        code.update(adapter_python_sha256='adapter-after', entrypoints_sha256='entry-after')
    bindings = [{
        'task_key': f'{variant}/{item.instance_id}', 'database_id': item.database_id,
        'question_sha256': fingerprint(question_row(item)),
        'schema_sha256': fingerprint(item.database_schema),
        'keywords_sha256': f'keywords-{item.instance_id}',
        'retrieval_sha256': f'retrieval-{item.instance_id}',
        'few_shot_sha256': fingerprint(item.few_shot_examples),
        'few_shot_source_rows': [{'source_row': number, 'db_id': f'train{number}',
                                 'dialect_conversion': 'postgresql'} for number in range(3)],
    } for variant, item in tasks]
    return {
        'format': 'deepeye-bird-interact-run-v1',
        'scope': 'gold-free bounded four-stage PostgreSQL workflow from frozen value retrieval',
        'workflow': list(STAGES), 'accuracy_evaluated': False, 'dynamic_few_shot_retrieval': False,
        'effective_config': config,
        'sources': {'precompute_inputs_content_hash': 'inputs',
                    'precompute_config_content_hash': 'precompute-config',
                    'precompute_semantic_config': {'embedding': {'model': 'frozen-embedding'}},
                    'few_shot_source_sha256': 'examples',
                    'precompute_population': {'lite': 195, 'full': 410, 'databases': 40},
                    'locators': {'precompute_dir': '/offline/unused', 'few_shot_source': '/offline/examples'},
                    'code': code},
        'item_count': len(tasks), 'items': sorted(bindings, key=lambda row: row['task_key']),
    }


def complete_stage(item, stage):
    if stage == 'schema_linking':
        for prefix in ('direct', 'reversed', 'value', 'final'):
            setattr(item, prefix + '_linked_tables_and_columns', {'t': ['x']})
        item.database_schema_after_schema_linking = copy.deepcopy(item.database_schema)
    elif stage == 'sql_generation':
        item.sql_candidates = ['SELECT x FROM t ORDER BY x']
    elif stage == 'sql_revision':
        item.sql_candidates_after_revision = ['SELECT x FROM t ORDER BY x']
    else:
        item.final_selected_sql = 'SELECT x FROM t ORDER BY x'
    setattr(item, stage + '_time', 3.0)
    setattr(item, stage + '_llm_cost', cost(3))
    item.total_time += 3
    item.total_llm_cost = cost(item.total_llm_cost['total_tokens'] + 3)


def seed_source(store, tasks, lengths=(2, 4), *, failed=None, unfinished=None, invalid=None):
    """Build canonical source chains independently of the import implementation."""
    for (variant, original), length in zip(tasks, lengths):
        key = f'{variant}/{original.instance_id}'
        state = copy.deepcopy(original)
        identity = fingerprint({'manifest': store.manifest,
                                'input': to_jsonable(original.model_dump(exclude={'gold_sql'}))})
        for stage in STAGES[:length]:
            input_hash = fingerprint({'input': identity, 'stage': stage})
            complete_stage(state, stage)
            payload = _checkpoint(state, stage)
            payload['attempt_wall_seconds'] = 1.25
            if invalid == (key, stage):
                payload['artifact']['sql_candidates'] = None
            attempt = store.begin_attempt(key, stage, input_hash)
            store.append_event(attempt, 'api_request', {'call_id': attempt, 'logical_sdk_call': True})
            store.append_event(attempt, 'api_response', {'call_id': attempt, 'response': {'usage': cost(3)}})
            if unfinished != (key, stage):
                status = 'failed' if failed == (key, stage) else 'succeeded'
                store.finish_attempt(attempt, status, payload)
            identity = fingerprint({'input': input_hash, 'output': payload})
    master = store.begin_attempt('lite/a', 'pipeline', 'orchestration-only')
    store.finish_attempt(master, 'succeeded', {'not_a_native_stage': True})


class OfflineTrace(TraceRecorder):
    def instrument_runner(self, runner, stage):
        return lambda: None


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch('socket.socket.connect', side_effect=AssertionError('network forbidden')))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.tasks = [('lite', make_item('a')), ('full', make_item('b'))]

    def inherit(self, source, destination, tasks=None):
        self.assertIsNotNone(inherit_checkpoints, 'The checkpoint inheritance implementation is missing')
        return inherit_checkpoints(source, destination, self.tasks if tasks is None else tasks)

    def stores(self, *, lengths=(2, 4), source_manifest=None, destination_manifest=None, **seed_options):
        source_path = self.root / 'source'
        with RunStore.create(source_path, source_manifest or manifest(self.tasks)) as source:
            seed_source(source, self.tasks, lengths, **seed_options)
        source = self.stack.enter_context(RunStore.open(source_path, read_only=True))
        destination = self.stack.enter_context(RunStore.create(
            self.root / 'destination', destination_manifest or manifest(self.tasks, upgrade=True)))
        return source, destination

    def test_inherits_six_checkpoints_with_historical_costs_and_verifiable_source_links(self):
        # Catches copying the source fingerprint chain, dropping provenance, or copying API/master events.
        source, destination = self.stores()
        source_before = source.summary(), source.attempts(), source.events(), source.verify()
        report = self.inherit(source, destination)
        self.assertEqual(report, {'imported': 6, 'reused': 0, 'stages': {
            'schema_linking': {'imported': 2, 'reused': 0},
            'sql_generation': {'imported': 2, 'reused': 0},
            'sql_revision': {'imported': 1, 'reused': 0},
            'sql_selection': {'imported': 1, 'reused': 0}}})
        source_by_id = {row['attempt_id']: row for row in source.attempts()}
        for row in destination.attempts():
            payload = row['payload']
            link = payload['inheritance']
            original = source_by_id[link['source_attempt_id']]
            self.assertEqual(payload['artifact'], original['payload']['artifact'])
            self.assertEqual(payload['metrics'], original['payload']['metrics'])
            self.assertEqual(payload['execution_origin'], 'inherited_successful_checkpoint')
            self.assertEqual(payload['attempt_wall_seconds'], 0.0)
            self.assertEqual(payload['source_attempt_wall_seconds'], 1.25)
            self.assertEqual(link['source_run'], str(source.run_dir.resolve()))
            self.assertEqual(link['source_manifest_fingerprint'], fingerprint(source.manifest))
            self.assertEqual(link['source_input_fingerprint'], original['input_fingerprint'])
            self.assertEqual(link['source_payload_fingerprint'], fingerprint(original['payload']))
            self.assertNotEqual(row['input_fingerprint'], original['input_fingerprint'])
        self.assertEqual(destination.events(), [])
        self.assertEqual(observed_usage(source)['requests'], 6)
        self.assertEqual(observed_usage(destination)['requests'], 0)
        self.assertEqual(observed_usage(destination)['reported_tokens'], cost())
        self.assertEqual(source_before, (source.summary(), source.attempts(), source.events(), source.verify()))
        self.assertTrue(destination.verify()['ok'])
        self.assertEqual(self.tasks[0][1].total_time, 2.0)

    def test_scheduler_restores_imports_and_executes_only_the_two_missing_stages(self):
        # Catches destination lineage hashes that look valid but cannot actually resume.
        source, destination = self.stores()
        self.inherit(source, destination)
        pipeline = importlib.import_module('scripts.baseline_adapters.deepeye.run_pipeline')
        def factory(stage, items):
            def process(target):
                if (target.instance_id, stage) not in {('a', 'sql_revision'), ('a', 'sql_selection')}:
                    raise AssertionError('An inherited stage was recomputed')
                complete_stage(target, stage)
            return SimpleNamespace(**{STAGE_METHODS[stage]: process, '_clean_up': lambda: None})
        result = pipeline.run_pipeline(destination, self.tasks, factory, OfflineTrace(destination), workers=2)
        self.assertEqual(result['succeeded'], 2)
        self.assertEqual(result['items']['lite/a']['total_time'], 14.0)
        self.assertEqual(result['items']['full/b']['total_time'], 14.0)
        stage_rows = [row for row in destination.attempts() if row['stage'] in STAGES]
        self.assertEqual(len(stage_rows), 8)
        self.assertEqual([row['stage'] for row in stage_rows
                          if row['payload'].get('execution_origin') != 'inherited_successful_checkpoint'],
                         ['sql_revision', 'sql_selection'])

    def test_repeated_import_is_idempotent(self):
        # Catches creating duplicate attempts or rebuilding destination chain from source payloads on replay.
        source, destination = self.stores()
        self.inherit(source, destination)
        rows = destination.attempts()
        report = self.inherit(source, destination)
        self.assertEqual(report['imported'], 0)
        self.assertEqual(report['reused'], 6)
        self.assertEqual(report['stages']['sql_generation'], {'imported': 0, 'reused': 2})
        self.assertEqual(destination.attempts(), rows)

    def test_chained_inheritance_keeps_older_provenance_and_original_metrics(self):
        # Catches overwriting an earlier inheritance claim during a second upgrade.
        source, middle = self.stores()
        self.inherit(source, middle)
        old_payload = middle.attempts()[0]['payload']
        middle_manifest = middle.manifest
        middle.close()
        middle = self.stack.enter_context(RunStore.open(self.root / 'destination', read_only=True))
        changed = copy.deepcopy(middle_manifest)
        changed['effective_config']['workers'] = 75
        changed['sources']['code']['entrypoints_sha256'] = 'entry-third'
        final = self.stack.enter_context(RunStore.create(self.root / 'third', changed))
        self.assertEqual(self.inherit(middle, final)['imported'], 6)
        payload = final.attempts()[0]['payload']
        self.assertEqual(payload['inheritance']['prior_inheritance'], old_payload['inheritance'])
        self.assertEqual(payload['metrics'], old_payload['metrics'])
        self.assertEqual(payload['source_attempt_wall_seconds'], 0.0)
        self.assertEqual(observed_usage(final)['requests'], 0)

    def test_interrupted_import_resumes_its_unfinished_attempt_without_duplicates(self):
        # Catches getting stuck after begin_attempt is committed but finish_attempt fails.
        source, destination = self.stores()
        original_finish = destination.finish_attempt
        count = 0
        def fail_third_finish(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError('controlled interruption before finish')
            return original_finish(*args, **kwargs)
        with patch.object(destination, 'finish_attempt', side_effect=fail_third_finish):
            with self.assertRaises(OSError):
                self.inherit(source, destination)
        partial = destination.attempts()
        self.assertEqual([row['status'] for row in partial], ['succeeded', 'succeeded', 'interrupted'])
        try:
            report = self.inherit(source, destination)
        except ValueError as error:
            self.fail(f'A consistent interrupted import could not resume: {error}')
        self.assertEqual((report['imported'], report['reused']), (4, 2))
        self.assertEqual(len(destination.attempts()), 6)
        self.assertTrue({row['attempt_id'] for row in partial}.issubset(
            row['attempt_id'] for row in destination.attempts()))

    def test_changed_semantic_config_or_source_is_rejected_before_any_import(self):
        # Catches an allowlist that accidentally permits model/data/SQL semantics to change.
        changes = [
            ('effective_config', 'chat', 'model'), ('effective_config', 'chat', 'timeout_seconds'),
            ('effective_config', 'chat', 'max_tokens'), ('effective_config', 'dataset', 'sql_execution_timeout_seconds'),
            ('effective_config', 'postgres', 'principal'),
            ('effective_config', 'stages', 'sql_generation', 'dc_sampling_budget'),
            ('effective_config', 'stages', 'sql_revision', 'checker_sampling_budget'),
            ('effective_config', 'stages', 'sql_selection', 'filter_top_k_sql'),
            ('sources', 'code', 'baseline_python_sha256'), ('sources', 'code', 'dependency_locks_sha256'),
            ('sources', 'precompute_inputs_content_hash'), ('sources', 'few_shot_source_sha256'),
            ('sources', 'precompute_semantic_config', 'embedding', 'model'),
        ]
        changes.extend(('effective_config', 'postgres', 'execution_policy', field) for field in (
            'version', 'meta_fence', 'sql_rewrite', 'result_row_limit',
            'search_path', 'single_statement', 'read_only'))
        source, _ = self.stores()
        for number, path in enumerate(changes):
            with self.subTest(path=path):
                changed = manifest(self.tasks, upgrade=True)
                owner = changed
                for key in path[:-1]:
                    owner = owner[key]
                owner[path[-1]] = 'changed'
                with RunStore.create(self.root / f'changed-{number}', changed) as destination:
                    with self.assertRaises(ValueError):
                        self.inherit(source, destination)
                    self.assertEqual(destination.attempts(), [])

    def test_missing_or_malformed_execution_policy_is_rejected_on_either_side(self):
        # Removing validation must not allow two legacy manifests to inherit together.
        missing = object()
        invalid_policies = [missing, None, {}, [], 'legacy', {'meta_fence': False},
                            {'version': ''}, {'version': ' '}, {'version': None}, {'version': 1}]
        for number, policy in enumerate(invalid_policies):
            for sides in (('source',), ('destination',), ('source', 'destination')):
                with self.subTest(policy=number, sides=sides):
                    manifests = {'source': manifest(self.tasks),
                                 'destination': manifest(self.tasks, upgrade=True)}
                    for side in sides:
                        postgres = manifests[side]['effective_config']['postgres']
                        if policy is missing:
                            postgres.pop('execution_policy')
                        else:
                            postgres['execution_policy'] = copy.deepcopy(policy)
                    prefix = f'policy-{number}-{"-".join(sides)}'
                    source_path = self.root / f'{prefix}-source'
                    with RunStore.create(source_path, manifests['source']) as source:
                        seed_source(source, self.tasks)
                    with RunStore.open(source_path, read_only=True) as source, \
                         RunStore.create(self.root / f'{prefix}-destination',
                                         manifests['destination']) as destination:
                        source_before = source.manifest, source.attempts(), source.events()
                        with self.assertRaisesRegex(ValueError, 'execution policy'):
                            self.inherit(source, destination)
                        self.assertEqual(destination.attempts(), [])
                        self.assertEqual(destination.events(), [])
                        self.assertEqual(source_before,
                                         (source.manifest, source.attempts(), source.events()))

    def test_changed_pristine_input_is_rejected_before_any_import(self):
        # Catches binding-only validation that misses evidence, few shots, or reconstructed precompute state.
        source, destination = self.stores()
        changes = {
            'question': 'Different question', 'evidence': 'Different evidence',
            'database_schema': {'tables': {}}, 'few_shot_examples': [],
            'question_keywords': ['different'], 'retrieved_values': {'t': {'x': ['other']}},
            'database_schema_after_value_retrieval': {'tables': {}},
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(self.tasks)
                setattr(changed[1][1], field, value)
                with self.assertRaises(ValueError):
                    self.inherit(source, destination, changed)
                self.assertEqual(destination.attempts(), [])

    def test_unstarted_source_item_without_full_input_binding_is_rejected(self):
        # Catches accepting retrieval/view drift for a target with no source stage fingerprint.
        source, destination = self.stores(lengths=(2, 0))
        for mutate in (False, True):
            with self.subTest(mutate=mutate):
                tasks = copy.deepcopy(self.tasks)
                if mutate:
                    tasks[1][1].retrieved_values = {'t': {'x': ['drift']}}
                with self.assertRaises(ValueError):
                    self.inherit(source, destination, tasks)
                self.assertEqual(destination.attempts(), [])

    def test_active_source_writer_is_rejected_even_through_read_only_handle(self):
        # Catches treating a read-only connection as evidence that the source run is quiescent.
        with RunStore.create(self.root / 'active', manifest(self.tasks)) as writer:
            seed_source(writer, self.tasks)
            with RunStore.open(self.root / 'active', read_only=True) as source, \
                 RunStore.create(self.root / 'inactive', manifest(self.tasks, upgrade=True)) as destination:
                with self.assertRaises(ValueError):
                    self.inherit(source, destination)
                self.assertEqual(destination.attempts(), [])

    def test_target_set_dirty_inputs_and_gold_are_rejected(self):
        # Catches inheriting a selected subset or checkpoint-mutated caller input under a full-run manifest.
        source, destination = self.stores()
        candidates = [self.tasks[:1], self.tasks + [self.tasks[0]]]
        for field, value in [('sql_candidates', ['SELECT 1']), ('gold_sql', 'SELECT secret')]:
            changed = copy.deepcopy(self.tasks)
            setattr(changed[0][1], field, value)
            candidates.append(changed)
        for tasks in candidates:
            with self.subTest(tasks=len(tasks)):
                with self.assertRaises(ValueError):
                    self.inherit(source, destination, tasks)
                self.assertEqual(destination.attempts(), [])

    def test_failed_or_unfinished_stage_stops_prefix_even_if_later_successes_exist(self):
        # Catches leaping across a missing generation checkpoint to later successes.
        for stop in ('failed', 'unfinished'):
            with self.subTest(stop=stop):
                with RunStore.create(self.root / f'source-{stop}', manifest(self.tasks)) as seeded:
                    seed_source(seeded, self.tasks, (4, 4), **{stop: ('lite/a', 'sql_generation')})
                with RunStore.open(self.root / f'source-{stop}', read_only=True) as source, \
                     RunStore.create(self.root / f'dest-{stop}', manifest(self.tasks, upgrade=True)) as destination:
                    self.assertEqual(self.inherit(source, destination)['imported'], 5)
                    self.assertEqual([row['stage'] for row in destination.attempts() if row['item_key'] == 'lite/a'],
                                     ['schema_linking'])

    def test_invalid_success_payload_rejects_before_writing_other_valid_prefixes(self):
        # Catches validating lazily after importing A while B has an unusable successful record.
        source, destination = self.stores(invalid=('full/b', 'sql_generation'))
        with self.assertRaises(ValueError):
            self.inherit(source, destination)
        self.assertEqual(destination.attempts(), [])

    def test_conflicting_destination_rejects_before_other_imports(self):
        # Catches silently accepting an independently executed or incompatible destination checkpoint.
        source, destination = self.stores()
        item = self.tasks[1][1]
        initial = fingerprint({'manifest': destination.manifest,
                               'input': to_jsonable(item.model_dump(exclude={'gold_sql'}))})
        incoming = fingerprint({'input': initial, 'stage': 'schema_linking'})
        attempt = destination.begin_attempt('full/b', 'schema_linking', incoming)
        destination.finish_attempt(attempt, 'succeeded', {'unrelated': True})
        before = destination.attempts()
        with self.assertRaises(ValueError):
            self.inherit(source, destination)
        self.assertEqual(destination.attempts(), before)

    def test_corrupt_store_verification_prevents_imports(self):
        # Catches omitting either source or destination integrity verification before appending.
        source, destination = self.stores()
        for store in (source, destination):
            with self.subTest(store=store.run_dir.name), patch.object(store, 'verify', return_value={'ok': False}):
                with self.assertRaises(ValueError):
                    self.inherit(source, destination)
                self.assertEqual(destination.attempts(), [])


if __name__ == '__main__':
    unittest.main()
