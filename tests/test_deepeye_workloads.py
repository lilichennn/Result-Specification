"""Offline workload boundaries: native identity, Meta scope and backend config."""
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/DeepEye-SQL'))


class WorkloadTests(unittest.TestCase):
    def modules(self):
        self.assertTrue((ROOT / 'scripts/baseline_adapters/deepeye/workloads.py').exists(),
                        'shared workload boundary is missing')
        return (importlib.import_module('scripts.baseline_adapters.deepeye.workloads'),
                importlib.import_module('scripts.deepeye_run'))

    def fixture(self, root, benchmark, split, *, cloud=False):
        from app.dataset.dataset import DataItem, BirdDataset, SpiderDataset
        from app.dataset.spider2_dataset import Spider2DataItem, Spider2LiteDataset
        from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataItem, BirdInteractDataset
        from app.dataset.utils import save_dataset
        meta = root / 'meta/db'
        meta.mkdir(parents=True)
        if benchmark in ('bird', 'bird_interact'):
            (meta / 'visible.csv').write_text('original_column_name,column_name,column_description,data_format,value_description\nid,id,identifier,INTEGER,\n')
        else:
            (meta / 'visible.csv').write_text('column_name,column_type,sample_value,ref_key\nid,INTEGER,[],\n')
        db = root / 'resources/db.sqlite'
        db.parent.mkdir()
        with sqlite3.connect(db) as conn:
            conn.executescript('create table visible(id integer primary key, hidden text); create table secret(x);')
        ids = ['bq001' if cloud else 'local001'] if benchmark == 'spider2' else (
            ['7'] if benchmark == 'bird_interact' else [7, 19])
        rows = [{'index': identity, 'db_id': 'db', 'question': f'question {identity}',
                 'evidence': '', 'SQL': 'SELECT SECRET_GOLD', 'query': 'SELECT SECRET_GOLD',
                 'sol_sql': 'SELECT SECRET_GOLD'} for identity in ids]
        (root / 'questions.json').write_text(json.dumps(rows))
        (root / 'rc.json').write_text('[]')
        schema = {'db_id': 'db', 'db_path': str(db), 'db_type': 'sqlite', 'tables': {
            'visible': {'table_name': 'visible', 'columns': {'id': {
                'column_name': 'id', 'column_type': 'INTEGER', 'is_unuseful': False,
                'primary_key': True, 'foreign_keys': [], 'description': '',
                'value_examples': [], 'value_statistics': None}}}}}
        item_type = {'bird': DataItem, 'spider': DataItem, 'spider2': Spider2DataItem,
                     'bird_interact': BirdInteractDataItem}[benchmark]
        dataset_type = {'bird': BirdDataset, 'spider': SpiderDataset, 'spider2': Spider2LiteDataset,
                        'bird_interact': BirdInteractDataset}[benchmark]
        items = []
        for i, row in enumerate(rows):
            extra = {}
            if benchmark in ('spider2', 'bird_interact'):
                extra = {'instance_id': row['index'], 'db_type': 'postgresql' if benchmark == 'bird_interact'
                         else ('bigquery' if cloud else 'sqlite')}
            item = item_type(question_id=row['index'] if type(row['index']) is int else i,
                question=row['question'], gold_sql='SELECT SECRET_GOLD', database_id='db',
                database_path='db' if cloud or benchmark == 'bird_interact' else str(db),
                database_schema=schema, question_keywords=[], retrieved_values={},
                value_retrieval_time=0.0,
                value_retrieval_llm_cost={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                total_time=0.0, total_llm_cost={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                database_schema_after_value_retrieval=schema,
                few_shot_examples=[{'question': 'independent', 'sql': 'SELECT 1'}],
                few_shot_preparation_metadata={'mode': 'native', 'num_examples': 1}, **extra)
            items.append(item)
        dataset = object.__new__(dataset_type)
        dataset._data = items
        dataset._config = SimpleNamespace(type=benchmark, split=split, root_path=str(root / 'resources'))
        save_dataset(dataset, str(root / 'prepared.snapshot'))
        config = {'benchmark': benchmark, 'split': split, 'questions': 'questions.json',
                  'meta': 'meta', 'resource_root': 'resources', 'rc': 'rc.json',
                  'prepared_dataset': 'prepared.snapshot'}
        path = root / 'workload.json'
        path.write_text(json.dumps(config))
        return path, items

    def test_six_selections_keep_original_identity_native_type_and_gold_out(self):
        workloads, entry = self.modules()
        for benchmark, split in [('bird', 'dev'), ('spider', 'dev'), ('spider', 'test'),
                                 ('spider2', 'lite'), ('bird_interact', 'lite'), ('bird_interact', 'full')]:
            with self.subTest(benchmark=benchmark, split=split), tempfile.TemporaryDirectory() as temporary:
                path, original = self.fixture(Path(temporary), benchmark, split)
                key = f'{benchmark}/{split}/' + ('i:19' if benchmark in ('bird', 'spider') else
                                                ('s:local001' if benchmark == 'spider2' else 's:7'))
                tasks, bindings, sources = entry.prepare_inputs(workload=path, item_keys=[key], code_source_hasher=lambda: {})
                item = tasks[0][1]
                self.assertEqual(bindings[0]['task_key'], key)
                self.assertEqual(bindings[0]['partition'], f'{benchmark}/{split}')
                self.assertEqual(bindings[0]['external_id'], 19 if benchmark in ('bird', 'spider') else workloads.external_id(item))
                self.assertIs(type(item), type(original[0]))
                self.assertEqual(item.gold_sql, '')
                self.assertNotIn('SECRET_GOLD', json.dumps([workloads.dump_item(item), bindings, sources]))
                restored = workloads.restore_item(workloads.dump_item(item))
                self.assertIs(type(restored), type(item))
                if benchmark in ('bird', 'spider'):
                    self.assertEqual(restored.question_id, 19)
                    self.assertFalse(hasattr(restored, 'instance_id'))
                    self.assertFalse(hasattr(restored, 'db_type'))

    def test_typed_keys_escape_string_ids_and_reject_arbitrary_classes(self):
        workloads, _ = self.modules()
        from app.dataset.dataset import DataItem
        from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataItem
        fields = dict(question_id=7, question='q', gold_sql='secret', database_id='d', database_path='d', database_schema={})
        self.assertEqual(workloads.task_key('bird/dev', DataItem(**fields)), 'bird/dev/i:7')
        self.assertEqual(workloads.task_key('bird_interact/lite', BirdInteractDataItem(instance_id='7/a %', **fields)),
                         'bird_interact/lite/s:7%2Fa%20%25')
        with self.assertRaises(ValueError):
            workloads.restore_item({'type': 'os.system', 'data': {}})

    def test_native_bird_meta_encoding_is_decoded_without_dropping_characters(self):
        workloads, _ = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.fixture(root, 'bird', 'dev')
            csv = root/'meta/db/visible.csv'
            csv.write_bytes('original_column_name,column_name,column_description,data_format,value_description\nid,id,identifier – official,INTEGER,\n'.encode('cp1252'))
            tables, _ = workloads._meta_tables(workloads.load_workload(path), 'db')
            self.assertEqual(tables['visible']['id']['column_description'], 'identifier – official')

    def test_config_and_backend_allow_sqlite_and_bigquery_without_postgres(self):
        _, entry = self.modules()
        for cloud in (False, True):
            with self.subTest(cloud=cloud), tempfile.TemporaryDirectory() as temporary:
                path, _ = self.fixture(Path(temporary), 'spider2' if cloud else 'spider', 'lite' if cloud else 'dev', cloud=cloud)
                args = entry._build_parser().parse_args(['prepare', '--run-dir', str(Path(temporary)/'run'), '--workload', str(path)])
                environment = {'DASH_MODELS': 'qwen3.6', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'secret'}
                config = entry.build_effective_config(environment, args)
                runtime = entry.build_runtime_config(environment, args, Path(temporary)/'run')
                self.assertNotIn('postgres', config)
                self.assertEqual(runtime.dataset_config.type, 'spider2' if cloud else 'spider')
                self.assertEqual(runtime.dataset_config.sql_execution_timeout, 600)
                self.assertEqual(runtime.dataset_config.max_value_example_length, 50)
                self.assertEqual(runtime.schema_linking_config.direct_linking_sampling_budget, 4)
                self.assertEqual(runtime.sql_generation_config.icl_sampling_budget, 4)
                self.assertEqual(runtime.sql_revision_config.checker_sampling_budget, 5)
                self.assertEqual(runtime.sql_selection_config.evaluator_sampling_budget, 5)
                self.assertEqual(runtime.sql_selection_config.shortcut_consistency_score_threshold, .6)
                if cloud:
                    self.assertEqual(runtime.sql_revision_config.checkers, ['SyntaxChecker', 'ResultChecker'])
                with entry.backend_context(environment, args):
                    pass
                from scripts.baseline_adapters.deepeye import backend_hooks
                execute_pg = backend_hooks.execute_postgres_sql
                with entry.admission_context(SimpleNamespace(record_admission=lambda event: None), args, population=1) as gates:
                    self.assertIs(backend_hooks.execute_postgres_sql, execute_pg)
                    self.assertNotIn('postgres', gates)

    def test_native_config_paths_resolve_from_config_and_effective_values_are_frozen(self):
        _, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'dev')
            native = path.parent/'settings/native.json'; native.parent.mkdir()
            native.write_text(json.dumps({'dataset': {'sql_execution_timeout': 91},
                'vector_database': {'store_root_path':'vectors'},
                'schema_linking': {'value_distance_threshold':.123},
                'sql_revision': {'checkers': ['SyntaxChecker']},
                'few_shot_index': {'save_path':'examples'}}))
            workload = json.loads(path.read_text()); workload['native_config']='settings/native.json'; path.write_text(json.dumps(workload))
            args = entry._build_parser().parse_args(['prepare','--workload',str(path),'--run-dir',str(path.parent/'run')])
            env = {'DASH_MODELS':'test','DASH_BASE_URL':'https://example.test/v1','DASH_API_KEY':'secret'}
            config = entry.build_runtime_config(env,args,path.parent/'run')
            self.assertEqual(config.vector_database_config.store_root_path,str((native.parent/'vectors').resolve()))
            self.assertEqual(config.few_shot_index_config.save_path,str((native.parent/'examples').resolve()))
            frozen = entry.build_effective_config(env,args)
            self.assertEqual(frozen['dataset']['sql_execution_timeout_seconds'],91)
            self.assertEqual(frozen['stages']['schema_linking']['value_distance_threshold'],.123)
            self.assertEqual(config.sql_revision_config.checkers, ['SyntaxChecker'])

    def test_fresh_vector_build_and_retrieval_backend_defaults_agree(self):
        _, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'bird', 'dev')
            args = entry._build_parser().parse_args(['prepare', '--workload', str(path), '--run-dir', temporary])
            config = entry.build_runtime_config({'DASH_MODELS': 'fixture', 'DASH_BASE_URL': 'https://example.test/v1'}, args, path.parent/'run')
            self.assertIn(config.vector_database_config.build_backend, ('both', config.value_retrieval_config.backend))

    def test_preparation_resolves_native_shared_embedding_and_llm_profile(self):
        _, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.fixture(root, 'bird', 'dev')
            raw = {'llm_profiles': {'masker': {'model': 'mask-fixture', 'base_url': 'https://example.test/v1', 'api_key': 'offline'}},
                   'run': {'default_llm_profile': 'masker'},
                   'embedding': {'embedding_model_name_or_path': 'embedding-fixture'},
                   'few_shot_index': {'preliminary_sql': {'enabled': True}}}
            (root/'native.json').write_text(json.dumps(raw))
            spec = json.loads(path.read_text()); spec['native_config'] = 'native.json'; path.write_text(json.dumps(spec))
            args = entry._build_parser().parse_args(['prepare', '--workload', str(path), '--run-dir', temporary])
            config = entry.build_runtime_config({'DASH_MODELS': 'fixture', 'DASH_BASE_URL': 'https://example.test/v1'}, args, root/'run')
            self.assertEqual(config.few_shot_index_config.llm.model, 'mask-fixture')
            self.assertEqual(config.few_shot_index_config.embedding.embedding_model_name_or_path, 'embedding-fixture')
            self.assertEqual(config.few_shot_index_config.preliminary_sql.llm.model, 'mask-fixture')

    def test_missing_preparation_points_to_explicit_native_command(self):
        _, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'dev')
            config = json.loads(path.read_text()); config.pop('prepared_dataset')
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'prepare-native'):
                entry.prepare_inputs(workload=path, code_source_hasher=lambda: {})

    def test_snapshot_class_and_identity_mismatch_are_rejected(self):
        _, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'test')
            snapshot = path.parent/'prepared.snapshot'
            manifest = json.loads(snapshot.read_text()); manifest['item_class_module'] = 'os'
            snapshot.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'class|type'):
                entry.prepare_inputs(workload=path, code_source_hasher=lambda: {})

    def test_prepared_schema_must_not_expand_visible_meta_columns(self):
        _, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'dev')
            records = path.parent/'prepared.snapshot.data/items.jsonl'
            rows = [json.loads(line) for line in records.read_text().splitlines()]
            rows[0]['input']['database_schema']['tables']['secret'] = {'table_name':'secret','columns':{}}
            records.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'Meta|schema'):
                entry.prepare_inputs(workload=path, code_source_hasher=lambda: {})

    def test_prepared_foreign_keys_cannot_reveal_hidden_meta_objects(self):
        from app.dataset.utils import save_dataset
        from app.dataset.dataset import BirdDataset
        workloads, _ = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, items = self.fixture(root, 'bird', 'dev')
            for item in items:
                item.database_schema['tables']['visible']['columns']['id']['foreign_keys'] = [['secret', 'x']]
                item.database_schema_after_value_retrieval['tables']['visible']['columns']['id']['foreign_keys'] = [['secret', 'x']]
            dataset = object.__new__(BirdDataset)
            dataset._data, dataset._config = items, SimpleNamespace(type='bird', split='dev', root_path=str(root/'resources'))
            save_dataset(dataset, str(root/'prepared.snapshot'))
            tasks, _, _ = workloads.load_items(path)
            for _, item in tasks:
                for schema in (item.database_schema, item.database_schema_after_value_retrieval):
                    self.assertEqual(schema['tables']['visible']['columns']['id']['foreign_keys'], [])

    def test_fewshot_client_closes_if_index_construction_fails(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import _prepare_examples
        from unittest.mock import Mock
        client = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            config = SimpleNamespace(few_shot_index_config=SimpleNamespace(llm=object(),
                save_path=temporary, embedding=object(), mask_cache_path=None, max_samples=None,
                max_samples_per_db=None), run_config=SimpleNamespace(embedding_batch_size=1,
                parallelism=1, llm_timeout=1, progress_log_interval=1))
            with patch('app.llm.LLM', return_value=SimpleNamespace(_client=client)), patch(
                    'app.few_shot.index_builder.build_few_shot_index', side_effect=RuntimeError('index failed')):
                with self.assertRaisesRegex(RuntimeError, 'index failed'):
                    _prepare_examples({'benchmark': 'bird', 'few_shot_source': temporary}, [], config)
            client.close.assert_called_once()

    def test_explicit_preparation_reuses_native_snapshots_without_new_requests(self):
        workloads, entry = self.modules()
        self.assertTrue((ROOT / 'scripts/baseline_adapters/deepeye/workload_preparation.py').exists(),
                        'explicit native preparation command is missing')
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'test')
            output = Path(temporary)/'finished.snapshot'
            environment = {'DASH_MODELS':'test', 'DASH_BASE_URL':'https://example.test/v1', 'DASH_API_KEY':'secret'}
            report = prepare_native(path, output, environment, workers=1)
            self.assertEqual(report['prepared'], 2)
            config = workloads.load_workload(path); config['prepared_dataset'] = str(output)
            tasks, _, sources = entry.prepare_inputs(workload=config, code_source_hasher=lambda: {})
            self.assertEqual([item.question_id for _, item in tasks], [7, 19])
            self.assertNotIn('SECRET_GOLD', output.with_name(output.name+'.data').joinpath('items.jsonl').read_text())
            self.assertNotIn('secret', json.dumps(report))

    def test_raw_native_loading_clips_meta_without_reindexing(self):
        workloads, _ = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'dev')
            config = workloads.load_workload(path); config.pop('prepared_dataset')
            tasks, _, _ = workloads.load_items(config, require_prepared=False)
            self.assertEqual([item.question_id for _, item in tasks], [7, 19])
            self.assertEqual(set(tasks[0][1].database_schema['tables']), {'visible'})
            self.assertEqual(set(tasks[0][1].database_schema['tables']['visible']['columns']), {'id'})
            self.assertTrue(tasks[0][1].database_schema['tables']['visible']['columns']['id']['primary_key'])

    def test_fresh_preparation_uses_native_vr_and_snapshot_with_mocked_services(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner
        from app.config.config import EmbeddingConfig
        from unittest.mock import Mock
        workloads, entry = self.modules()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.fixture(root, 'bird', 'dev')
            workload = workloads.load_workload(path); workload.pop('prepared_dataset')
            args = entry._build_parser().parse_args(['prepare', '--workload', str(path), '--run-dir', temporary])
            env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'offline'}
            config = entry.build_runtime_config(env, args, root/'preparation')
            config.run_config.parallelism = 1
            config.few_shot_index_config.embedding = EmbeddingConfig(embedding_model_name_or_path='fixture')
            config.few_shot_index_config.llm = config.value_retrieval_config.llm
            index = Path(config.few_shot_index_config.save_path); index.mkdir(parents=True)
            (index/'manifest.json').write_text('{}')
            def examples(_, items, __):
                for item in items:
                    item.few_shot_examples = [{'question': 'independent', 'sql': 'SELECT 1'}]
            fake_llm = SimpleNamespace(llm_config=config.value_retrieval_config.llm, _client=Mock())
            with patch.object(entry, 'build_runtime_config', return_value=config), \
                 patch('runner.create_vector_db_parallel.run_vector_db_creation', autospec=True) as build, \
                 patch('app.pipeline.value_retrieval.value_retrieval.get_embedding_function', return_value=lambda **kw: []), \
                 patch('app.pipeline.value_retrieval.value_retrieval.LLM', return_value=fake_llm), \
                 patch.object(ValueRetrievalRunner, '_extract_keywords', return_value=([], {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2})), \
                 patch.object(ValueRetrievalRunner, '_get_local_value_index', return_value=object()), \
                 patch('scripts.baseline_adapters.deepeye.workload_preparation._prepare_examples', side_effect=examples), \
                 patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                result = prepare_native(workload, root/'ready.snapshot', env, workers=1)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(result['prepared'], 2)
            tasks, _, _ = workloads.load_items({**workload, 'prepared_dataset': str(root/'ready.snapshot')})
            self.assertEqual([item.question_id for _, item in tasks], [7, 19])
            for _, item in tasks:
                self.assertTrue(item.is_stage_complete('value_retrieval'))
                self.assertEqual(item.value_retrieval_llm_cost['total_tokens'], 2)
                self.assertEqual(set(item.database_schema_after_value_retrieval['tables']), {'visible'})

    def test_preparation_reports_missing_fewshot_before_any_paid_work(self):
        self.modules()
        self.assertTrue((ROOT / 'scripts/baseline_adapters/deepeye/workload_preparation.py').exists(),
                        'explicit native preparation command is missing')
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider', 'dev')
            config = json.loads(path.read_text()); config.pop('prepared_dataset'); path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'few.shot'):
                prepare_native(path, Path(temporary)/'out.snapshot',
                    {'DASH_MODELS':'test', 'DASH_BASE_URL':'https://example.test/v1', 'DASH_API_KEY':'secret'}, workers=1)

    def test_spider2_native_skip_does_not_require_fewshot_resources(self):
        workloads, _ = self.modules()
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'spider2', 'lite')
            config = json.loads(path.read_text()); config.pop('prepared_dataset')
            path.write_text(json.dumps(config))
            output = Path(temporary)/'skipped.snapshot'
            with patch('socket.socket.connect', side_effect=AssertionError('native skip called network')):
                result = prepare_native(path, output,
                    {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'secret'})
                tasks, _, _ = workloads.load_items({**workloads.load_workload(path), 'prepared_dataset': str(output)})
            self.assertEqual(result['prepared'], 1)
            self.assertTrue(tasks[0][1].is_stage_complete('value_retrieval'))
            self.assertFalse(tasks[0][1].few_shot_examples)

    def test_explicit_interact_workload_uses_existing_precompute_without_605_gate(self):
        workloads, entry = self.modules()
        from scripts.baseline_adapters.deepeye.precompute_pipeline import write_record, question_row, file_hash
        from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'bird_interact', 'lite')
            config = workloads.load_workload(path); config.pop('prepared_dataset')
            tasks, _, _ = workloads.load_items(config, require_prepared=False)
            item = tasks[0][1]
            from scripts.baseline_adapters.deepeye.dataset import _load_meta_table
            native_table = _load_meta_table(path.parent/'meta/db/visible.csv')
            self.assertEqual(item.database_schema['tables']['visible'], native_table)
            precompute = path.parent/'precompute'
            directory = precompute/'questions/lite/7'; directory.mkdir(parents=True)
            schema_path = precompute/'databases/db/schema.json'; schema_path.parent.mkdir(parents=True)
            schema_path.write_text(json.dumps(item.database_schema))
            keywords = write_record(directory/'keywords.json', {'row':question_row(item),'keywords':['visible']})
            np.save(directory/'keywords.npy',np.asarray([[1.,0.]],dtype='float32'))
            write_record(directory/'retrieval.json', {'row':question_row(item),'variant':'lite','question_id':0,
                'schema_hash':fingerprint(item.database_schema),'keywords_hash':keywords['content_hash'],
                'vectors_hash':file_hash(directory/'keywords.npy'),'dimension':2,'retrieved_values':{},
                'max_values_per_column':5,'retrieved_schema_hash':fingerprint(item.database_schema),
                'value_retrieval_time':0.,'value_retrieval_llm_cost':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
            write_record(precompute/'inputs.json',{'questions':[question_row(item)]})
            write_record(precompute/'run_config.json',{'config':{'model':'fixture'}})
            source = path.parent/'train.json'
            source.write_text(json.dumps([{'db_id':f'train{i}','question':f'train {i}','SQL':'SELECT 1'} for i in range(3)]))
            config.update(precompute_dir=str(precompute),few_shot_source=str(source))
            tasks, bindings, sources = entry.prepare_inputs(workload=config,code_source_hasher=lambda:{})
            self.assertEqual(bindings[0]['task_key'],'bird_interact/lite/s:7')
            self.assertEqual(len(tasks[0][1].few_shot_examples),3)
            self.assertIsNotNone(tasks[0][1].database_schema_after_value_retrieval)
            self.assertIn('precompute_config_content_hash',sources)


if __name__ == '__main__':
    unittest.main()
