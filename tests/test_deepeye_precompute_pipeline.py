import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))
try:
    pipeline = importlib.import_module('scripts.baseline_adapters.deepeye.precompute_pipeline')
except ModuleNotFoundError as exc:
    if exc.name != 'scripts.baseline_adapters.deepeye.precompute_pipeline':
        raise
    pipeline = None


class PipelineTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(pipeline, 'Precompute pipeline is missing')
        return pipeline

    def reply(self, content):
        return {'content': content, 'call_id': 'fixture-call', 'input_hash': 'fixture-input',
                'usage': {'prompt_tokens': 7, 'completion_tokens': 4, 'total_tokens': 11,
                          'attempts': 1, 'elapsed_seconds': .2, 'usage_missing': 0}}

    def test_keywords_are_native_postprocessed_and_reused_without_model(self):
        module = self.module()
        row = {'index': 'a_1', 'db_id': 'a', 'question': 'Find red stars', 'evidence': 'color means hue'}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'keywords.json'
            def chat(messages):
                self.assertIn('Find red stars', messages[0]['content'])
                self.assertNotIn('gold_sql', messages[0]['content'])
                return self.reply('<result>["red stars"]</result>')
            record = module.prepare_keywords(row, {'model': 'test'}, path, chat)
            self.assertEqual(record['keywords'], ['red', 'red stars', 'stars'])
            self.assertEqual(record['usage']['total_tokens'], 11)
            second = module.prepare_keywords(row, {'model': 'test'}, path, lambda _: self.fail('Unexpected API'))
            self.assertEqual(record, second)

    def test_changed_question_or_model_cannot_reuse_cached_keywords(self):
        module = self.module()
        row = {'index': 'a_1', 'db_id': 'a', 'question': 'Find red stars', 'evidence': ''}
        for field in ['question', 'model']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'keywords.json'
                module.prepare_keywords(row, {'model': 'first'}, path, lambda _: self.reply('<result>["red"]</result>'))
                changed = {**row, 'question': 'Find blue stars'} if field == 'question' else row
                config = {'model': 'second' if field == 'model' else 'first'}
                with self.assertRaises(ValueError):
                    module.prepare_keywords(changed, config, path, lambda _: self.fail('Must reject stale artifact'))

    def test_malformed_keyword_responses_do_not_create_success_record(self):
        module = self.module()
        row = {'index': 'a_1', 'db_id': 'a', 'question': 'stars', 'evidence': ''}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'keywords.json'
            with self.assertRaises(ValueError):
                module.prepare_keywords(row, {'model': 'test'}, path, lambda _: self.reply('not a result'))
            self.assertFalse(path.exists())
            self.assertTrue(path.with_suffix('.failed.json').exists())

    def test_reconstruction_matches_native_update_and_does_not_mutate_common_schema(self):
        module = self.module()
        schema = {'db_id': 'a', 'db_type': 'postgresql', 'tables': {'stars': {'columns': {
            'name': {'column_type': 'TEXT', 'value_examples': ['old']},
            'size': {'column_type': 'INTEGER'},
        }}}}
        retrieved = {'stars': {'name': [{'value': 'new', 'distance': 0.1}]}}
        actual = module.schema_with_values(schema, retrieved, 1)
        self.assertEqual(actual['tables']['stars']['columns']['name']['value_examples'], ['new'])
        self.assertEqual(schema['tables']['stars']['columns']['name']['value_examples'], ['old'])
        self.assertEqual(actual['tables']['stars']['columns']['size'], {'column_type': 'INTEGER'})

    def test_content_tampering_is_not_a_valid_keyword_cache(self):
        module = self.module()
        row = {'index': 'a_1', 'db_id': 'a', 'question': 'stars', 'evidence': ''}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'keywords.json'
            module.prepare_keywords(row, {'model': 'test'}, path, lambda _: self.reply('<result>["stars"]</result>'))
            record = json.loads(path.read_text())
            record['keywords'] = ['unrelated']
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                module.prepare_keywords(row, {'model': 'test'}, path, lambda _: self.fail('Unexpected call'))

    def test_native_index_roundtrip_and_corruption_are_checked(self):
        import numpy as np
        from app.vector_db.local_index import LocalValueIndex
        from scripts.baseline_adapters.deepeye.precompute_cache import VectorCache
        module = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with VectorCache(root / 'cache.sqlite', {'model': 'test'}) as cache:
                cache.embed(['red', 'blue'], lambda _: [[1., 0.], [0., 1.]])
                manifest = {'schema_hash': 'schema', 'columns': [
                    {'table_name': 'stars', 'column_name': 'color', 'documents': ['red', 'blue'], 'status': 'collected'}]}
                record = module.build_native_index(root / 'value_index/a', manifest, cache)
                self.assertEqual(record['dimension'], 2)
                index = LocalValueIndex(root / 'value_index/a/local_index', device='cpu')
                result = index.retrieve_values_for_column([[0., 1.]], 'stars', 'color', 1, True)
                self.assertEqual(result['values'][0]['value'], 'blue')
                self.assertEqual(result['values'][0]['distance'], 0.)
                first_file = next((root / 'value_index/a/local_index/columns').glob('*.npy'))
                np.save(first_file, np.array([[99., 0.], [0., 99.]], dtype='float32'))
                with self.assertRaises(ValueError):
                    module.verify_native_index(root / 'value_index/a')

    def test_native_index_identity_binds_complete_sample_manifest(self):
        """Reducing identity to schema or namespace must not preserve equality."""
        from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
        module = self.module()
        self.assertTrue(
            hasattr(module, 'native_index_identity'),
            'shared native-index identity helper is missing',
        )
        manifest = {'schema_hash': 'same-schema', 'columns': [
            {'table_name': 'stars', 'column_name': 'color',
             'documents': ['red'], 'status': 'collected'}]}
        changed = {**manifest, 'columns': [
            {**manifest['columns'][0], 'documents': ['blue']}]}
        expected = fingerprint({
            'sample_manifest': manifest,
            'embedding_namespace': 'same-namespace',
            'format': 'native-local-index-v1',
        })
        self.assertEqual(
            module.native_index_identity(manifest, 'same-namespace'), expected
        )
        self.assertNotEqual(
            module.native_index_identity(manifest, 'same-namespace'),
            module.native_index_identity(changed, 'same-namespace'),
        )

    def test_cli_verify_rejects_index_from_different_complete_manifest(self):
        """Matching schema and embedding namespace must not accept stale samples."""
        script = importlib.import_module('scripts.deepeye_bird_interact_precompute')
        manifest = {
            'collection_fingerprint': 'sample-fingerprint',
            'schema_hash': 'same-schema',
            'columns': [],
            'policy': {'max_values_per_column': 1000},
            'provenance': {'source': {'host': 'pg', 'port': 5432, 'database': 'a'}},
        }
        config = {
            'sampling': script.sampling_identity({'a': manifest}),
            'embedding': {'model': 'test'},
        }
        stale_index = {
            'identity': 'wrong-full-manifest-identity',
            'schema_hash': 'same-schema',
            'embedding_namespace': 'same-namespace',
            'documents': 0,
        }

        class FakeCache:
            namespace = 'same-namespace'
            dimension = 2

            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _documents):
                self.fail('No documents expected')

            def count(self):
                return 0

        args = SimpleNamespace(output_dir=Path('/unused'), sample_cap=1000)
        databases = {'a': SimpleNamespace(database_schema={})}
        with patch.object(script, 'read_record', return_value={'config': config}), \
             patch.object(script, 'freeze_inventory'), \
             patch.object(script, 'collect_manifests', return_value={'a': manifest}), \
             patch.object(script, 'verify_native_index', return_value=stale_index), \
             patch.object(script, 'VectorCache', FakeCache), \
             patch.object(script, 'atomic_json'), \
             self.assertRaisesRegex(ValueError, 'Index.*sample'):
            script.verify(args, [], databases)

    def test_missing_usage_cannot_be_frozen_as_success(self):
        module = self.module()
        row = {'index': 'a_1', 'db_id': 'a', 'question': 'stars', 'evidence': ''}
        reply = self.reply('<result>["stars"]</result>')
        reply['usage']['total_tokens'] = None
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'keywords.json'
            with self.assertRaises(ValueError):
                module.prepare_keywords(row, {'model': 'test'}, path, lambda _: reply)
            self.assertFalse(path.exists())

    def test_parse_retry_costs_are_combined_in_the_same_atomic_keyword_record(self):
        module = self.module()
        row = {'index': 'a_1', 'db_id': 'a', 'question': 'stars', 'evidence': ''}
        responses = iter([self.reply('invalid'), self.reply('<result>["stars"]</result>')])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'keywords.json'
            record = module.prepare_keywords(row, {'model': 'test'}, path, lambda _: next(responses))
            self.assertEqual(record['usage']['total_tokens'], 22)
            self.assertEqual(len(record['calls']), 2)
            self.assertEqual(module.read_record(path)['usage'], record['usage'])

    def test_shared_reader_reads_common_schema_and_question_records_once_without_aliasing(self):
        """Removing instance caches or defensive copies must increase reads or leak mutations."""
        module = self.module()
        schema = {'db_id': 'a', 'db_type': 'postgresql', 'tables': {'stars': {'columns': {
            'name': {'column_type': 'TEXT', 'value_examples': ['public']},
        }}}}
        usage = {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}
        expected = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schema_path = root / 'databases' / 'a' / 'schema.json'
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text(json.dumps(schema))
            for number in (1, 2):
                instance_id = f'a_{number}'
                row = {'index': instance_id, 'db_id': 'a', 'question': f'question {number}',
                       'evidence': f'evidence {number}'}
                expected.append(SimpleNamespace(instance_id=instance_id, database_id='a',
                    question=row['question'], evidence=row['evidence'], database_schema=schema))
                directory = module.item_directory(root, 'lite', instance_id)
                keywords = module.write_record(directory / 'keywords.json', {
                    'row': row, 'keywords': [f'term-{number}'],
                })
                vectors_path = directory / 'keywords.npy'
                np.save(vectors_path, np.array([[float(number), 1.]], dtype='float32'))
                retrieved = {'stars': {'name': [
                    {'value': f'value-{number}', 'distance': 0.1},
                ]}}
                reconstructed = {
                    **schema,
                    'tables': {'stars': {'columns': {
                        'name': {'column_type': 'TEXT',
                                 'value_examples': [f'value-{number}', 'public']},
                    }}},
                }
                module.write_record(directory / 'retrieval.json', {
                    'row': row, 'variant': 'lite', 'question_id': number,
                    'schema_hash': module.fingerprint(schema),
                    'keywords_hash': keywords['content_hash'],
                    'vectors_hash': module.file_hash(vectors_path), 'dimension': 2,
                    'retrieved_values': retrieved, 'max_values_per_column': 5,
                    'retrieved_schema_hash': module.fingerprint(reconstructed),
                    'value_retrieval_time': float(number),
                    'value_retrieval_llm_cost': usage,
                })

            reads = []
            schema_hashes = 0
            native_read_text = Path.read_text
            native_fingerprint = module.fingerprint

            def traced_read_text(path, *args, **kwargs):
                try:
                    reads.append(Path(path).relative_to(root))
                except ValueError:
                    pass
                return native_read_text(path, *args, **kwargs)

            def traced_fingerprint(value):
                nonlocal schema_hashes
                if value == schema:
                    schema_hashes += 1
                return native_fingerprint(value)

            with patch.object(Path, 'read_text', traced_read_text), \
                 patch.object(module, 'fingerprint', traced_fingerprint):
                reader = module.PrecomputedInputReader(root)
                first = reader.load('lite', 'a_1', expected_item=expected[0])
                second = reader.load('lite', 'a_2', expected_item=expected[1])
                first_keywords, first_retrieval = reader.records('lite', 'a_1')
                again_keywords, again_retrieval = reader.records('lite', 'a_1')

            self.assertEqual(reads.count(Path('databases/a/schema.json')), 1)
            # Once for the frozen schema plus one current-dataset check per question.
            self.assertEqual(schema_hashes, 3)
            for instance_id in ('a_1', 'a_2'):
                self.assertEqual(reads.count(Path(f'questions/lite/{instance_id}/keywords.json')), 1)
                self.assertEqual(reads.count(Path(f'questions/lite/{instance_id}/retrieval.json')), 1)

            first.database_schema['tables']['stars']['columns']['name']['value_examples'].append('private')
            first.question_keywords.append('private')
            first.retrieved_values['stars']['name'][0]['value'] = 'private'
            first_keywords['keywords'].append('private')
            first_retrieval['retrieved_values']['stars']['name'][0]['value'] = 'private'
            self.assertEqual(second.database_schema['tables']['stars']['columns']['name']['value_examples'], ['public'])
            self.assertEqual(second.question_keywords, ['term-2'])
            self.assertEqual(second.retrieved_values['stars']['name'][0]['value'], 'value-2')
            self.assertEqual(again_keywords['keywords'], ['term-1'])
            self.assertEqual(again_retrieval['retrieved_values']['stars']['name'][0]['value'], 'value-1')


if __name__ == '__main__':
    unittest.main()
