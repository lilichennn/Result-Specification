"""Regression checks for shared preparation, identities and expensive-step reuse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_deepeye_workloads as fixtures


class DynamicPreparationTests(unittest.TestCase):
    def fixture(self, root, benchmark='bird', split='dev'):
        return fixtures.WorkloadTests().fixture(root, benchmark, split)

    def test_native_value_builder_separates_database_from_column_concurrency(self):
        from runner.create_vector_db_parallel import run_vector_db_creation
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        pools = []
        def pool_factory(max_workers):
            pools.append(max_workers)
            return ThreadPoolExecutor(max_workers=max_workers)
        with patch('runner.create_vector_db_parallel.load_dataset', return_value=[]), \
             patch('runner.create_vector_db_parallel._collect_sqlite_db_paths', return_value=['a.sqlite']), \
             patch('runner.create_vector_db_parallel.ThreadPoolExecutor', side_effect=pool_factory), \
             patch('runner.create_vector_db_parallel.make_vector_db_for_db_path', return_value=True) as build:
            run_vector_db_creation('unused', 'bird', SimpleNamespace(), 16, 20, 50, database_parallelism=4)
        self.assertEqual(pools, [4])
        self.assertEqual(build.call_args.kwargs['parallelism'], 16)

    def test_incomplete_native_value_index_stops_before_keyword_requests(self):
        from app.dataset.utils import load_dataset, save_dataset
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'key'}
        for index_succeeds in (False, True):
            with self.subTest(index_succeeds=index_succeeds), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, _ = self.fixture(root)
                source = root/'prepared.snapshot'
                dataset = load_dataset(str(source))
                for item in dataset:
                    item.database_schema_after_value_retrieval = None
                save_dataset(dataset, str(source))
                output = root/'ready.snapshot'
                # Keep the real native database wrapper and its success_flag
                # handling; replace only the expensive column-index builder.
                with patch('runner.create_vector_db_parallel.make_vector_db', return_value=index_succeeds), \
                     patch('app.pipeline.value_retrieval.value_retrieval.ValueRetrievalRunner.from_config',
                           side_effect=RuntimeError('reached value retrieval')), \
                     patch('socket.socket.connect', side_effect=AssertionError('no paid calls')):
                    expected = 'reached value retrieval' if index_succeeds else 'incomplete.*db'
                    with self.assertRaisesRegex(RuntimeError, expected):
                        prepare_native(path, output, env, workers=1, embedding_service=lambda texts: [])
                self.assertFalse(output.exists(), 'unfinished preparation must not be published')

    def test_preparation_cli_accepts_several_workloads_without_requiring_one_output(self):
        from scripts.deepeye_run import _build_parser
        args = _build_parser().parse_args(['prepare-native', '--workload', 'a.json',
            '--workload', 'b.json', '--output-dir', 'ready'])
        self.assertEqual(args.workload, [Path('a.json'), Path('b.json')])
        self.assertEqual(args.output_dir, Path('ready'))

    def test_multi_workload_command_owns_one_shared_embedding_service(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_many
        from scripts.baseline_adapters.deepeye.embedding_service import EmbeddingService
        env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'key',
               'EMBEDDING_MODEL': 'embed', 'EMBEDDING_BASE_URL': 'https://example.test/v1', 'EMBEDDING_API_KEY': 'key'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self.fixture(root/'bird', 'bird', 'dev')[0], self.fixture(root/'spider', 'spider', 'dev')[0]]
            for path in paths:
                value = json.loads(path.read_text()); value['preparation_root'] = str(root/'shared')
                path.write_text(json.dumps(value))
            seen = []
            def prepare(workload, output, environment, **kwargs):
                seen.append((kwargs['embedding_service'], output))
                return {'prepared': 2}
            with patch('socket.socket.connect', side_effect=AssertionError('no paid calls in lifecycle test')), \
                 patch('scripts.deepeye_run.read_environment', return_value=env), \
                 patch('scripts.baseline_adapters.deepeye.workload_preparation.prepare_native', side_effect=prepare), \
                 patch('scripts.baseline_adapters.deepeye.embedding_service.EmbeddingService', wraps=EmbeddingService) as service:
                prepare_many(paths, None, root/'outputs', root/'unused.env')
            self.assertEqual(service.call_count, 1)
            self.assertIs(seen[0][0], seen[1][0])
            self.assertEqual([p.relative_to((root/'outputs').resolve()).as_posix() for _, p in seen],
                             ['bird/dev.snapshot', 'spider/dev.snapshot'])

    def test_workload_rejects_ambiguous_rc_version(self):
        from scripts.baseline_adapters.deepeye.workloads import load_workload
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.fixture(Path(tmp))
            for value in (True, '3', 4):
                config = json.loads(path.read_text()); config['rc_version'] = value
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'rc_version'):
                    load_workload(dict(config, _path=str(path)))

    def test_preparation_audit_accepts_real_dispatcher_callback(self):
        from scripts.baseline_adapters.deepeye.preparation_store import PreparationAudit
        from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
        with tempfile.TemporaryDirectory() as tmp:
            audit = PreparationAudit(Path(tmp)/'chat.jsonl')
            runtime = SamplingRuntime(request_workers=1, request_limit=1, emit=audit.emit)
            try:
                async def endpoint():
                    return 'unchanged successful response'
                with runtime.context(), audit.label({'step': 'keywords', 'item': 0}):
                    result = runtime.dispatch.call(endpoint)
                self.assertEqual(result, 'unchanged successful response')
            finally:
                runtime.close()
            rows = [json.loads(x) for x in audit.path.read_text().splitlines()]
            self.assertEqual(rows[0]['kind'], 'request_dispatch')
            self.assertEqual(rows[0]['label'], {'step': 'keywords', 'item': 0})

    def test_preparation_refuses_unbound_dimension_override_before_network(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_many
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = self.fixture(root, 'spider2', 'lite')
            limits = root/'limits.json'; limits.write_text('{"dimension": 512}')
            with patch('scripts.deepeye_run.read_environment', return_value={}), \
                 self.assertRaisesRegex(ValueError, '1024|dimension'):
                prepare_many([path], root/'ready.snapshot', None, root/'unused.env', embedding_config=limits)

    def test_selected_question_ids_are_typed_and_validated_before_resource_loading(self):
        from scripts.baseline_adapters.deepeye.workloads import load_workload, question_rows
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.fixture(Path(tmp))
            config = load_workload(path); config['question_ids'] = [19]
            self.assertEqual([r['external_id'] for r in question_rows(config)], [19])
            for ids in ([100], ['19'], [19, 19], []):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    question_rows(dict(config, question_ids=ids))

    def test_dynamic_profile_rejects_static_snapshot_but_preparation_can_replace_examples(self):
        from scripts.baseline_adapters.deepeye.workloads import load_workload, load_items
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.fixture(Path(tmp))
            config = load_workload(path); config['few_shot_strategy'] = 'native_dynamic'
            with self.assertRaisesRegex(ValueError, 'dynamic|strategy'):
                load_items(config)
            tasks, _, _ = load_items(config, require_prepared=False)
            self.assertTrue(all(not item.few_shot_examples for _, item in tasks))
            self.assertTrue(all(item.is_stage_complete('value_retrieval') for _, item in tasks))

    def test_dynamic_config_uses_one_embedding_profile_and_native_budgets(self):
        from scripts import deepeye_run as entry
        from scripts.baseline_adapters.deepeye.workloads import load_workload
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.fixture(Path(tmp))
            config = load_workload(path); config['few_shot_strategy'] = 'native_dynamic'
            args = entry._build_parser().parse_args(['prepare', '--workload', str(path), '--run-dir', tmp])
            args.workload = config
            env = {'DASH_MODELS': 'chat', 'DASH_BASE_URL': 'https://chat.test/v1', 'DASH_API_KEY': 'key',
                   'EMBEDDING_MODEL': 'embed', 'EMBEDDING_BASE_URL': 'https://embed.test/v1', 'EMBEDDING_API_KEY': 'key'}
            runtime = entry.build_runtime_config(env, args, Path(tmp)/'run')
            self.assertEqual(runtime.vector_database_config.embedding_model_name_or_path, 'embed')
            self.assertEqual(runtime.few_shot_index_config.embedding.embedding_model_name_or_path, 'embed')
            self.assertEqual(runtime.few_shot_index_config.llm.model, 'chat')
            self.assertEqual(runtime.few_shot_index_config.num_examples, 7)
            self.assertEqual(runtime.few_shot_index_config.question_weight, .6)
            self.assertEqual(runtime.few_shot_index_config.sql_weight, .4)
            self.assertTrue(runtime.few_shot_index_config.preliminary_sql.enabled)
            self.assertEqual(runtime.few_shot_index_config.preliminary_sql.dc_sampling_budget, 4)
            self.assertEqual(runtime.few_shot_index_config.preliminary_sql.skeleton_sampling_budget, 4)
            self.assertEqual(runtime.run_config.embedding_batch_size, 20)

    def test_complete_preparation_is_idempotent_and_does_not_use_network(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.fixture(Path(tmp), 'spider2', 'lite')
            output = Path(tmp)/'finished.snapshot'
            env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'key'}
            with patch('socket.socket.connect', side_effect=AssertionError('preparation repeated a network call')):
                first = prepare_native(path, output, env, workers=1)
                original = output.read_bytes()
                second = prepare_native(path, output, env, workers=1)
            self.assertTrue(second['reused'])
            self.assertEqual(first['sha256'], second['sha256'])
            self.assertEqual(original, output.read_bytes())

    def test_step_cache_keeps_preliminary_result_if_later_step_fails(self):
        from scripts.baseline_adapters.deepeye.preparation_store import PreparationStore
        with tempfile.TemporaryDirectory() as tmp:
            store = PreparationStore(Path(tmp), {'question': 'q', 'policy': 1})
            result = store.step('preliminary', lambda: {'sql': 'SELECT 1', 'token_usage': {'total_tokens': 12}})
            with self.assertRaises(RuntimeError):
                store.step('examples', lambda: (_ for _ in ()).throw(RuntimeError('interrupted')))
            restored = PreparationStore(Path(tmp), {'question': 'q', 'policy': 1})
            self.assertEqual(restored.step('preliminary', lambda: self.fail('must reuse preliminary')), result)
            self.assertEqual(restored.step('examples', lambda: {'ids': ['train:1']}), {'ids': ['train:1']})
            with self.assertRaisesRegex(ValueError, 'identity'):
                PreparationStore(Path(tmp), {'question': 'different', 'policy': 1})

    def test_corrupt_step_is_not_treated_as_success(self):
        from scripts.baseline_adapters.deepeye.preparation_store import PreparationStore
        with tempfile.TemporaryDirectory() as tmp:
            store = PreparationStore(Path(tmp), {'q': 1})
            store.step('preliminary', lambda: {'sql': 'SELECT 1'})
            (Path(tmp)/'preliminary.json').write_text('{incomplete')
            with self.assertRaises(ValueError):
                store.step('preliminary', lambda: self.fail('corrupt record must not be silently blessed'))

    def test_published_preparation_checks_content_and_database_changes(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        from scripts.baseline_adapters.deepeye.workloads import load_items
        env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'key'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = self.fixture(root, 'spider2', 'lite')
            output = root/'ready.snapshot'
            prepare_native(path, output, env, workers=1)
            manifest = json.loads(output.read_text())
            records = output.parent/manifest['snapshot_root']/'items.jsonl'
            original = records.read_bytes()
            records.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'content|checksum'):
                load_items(dict(json.loads(path.read_text()), _path=str(path), prepared_dataset=str(output)))
            records.write_bytes(original)
            with (root/'resources/db.sqlite').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity|database'):
                prepare_native(path, output, env, workers=1)

    def test_publish_failure_never_exposes_a_success_manifest_and_can_resume(self):
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        from scripts.baseline_adapters.deepeye.precompute_cache import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = self.fixture(root, 'spider2', 'lite')
            output = root/'ready.snapshot'
            env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'key'}
            def fail_final(target, value):
                if Path(target).resolve() == output.resolve():
                    raise OSError('simulated disk failure at final manifest publication')
                return atomic_json(target, value)
            with patch('scripts.baseline_adapters.deepeye.precompute_cache.atomic_json', side_effect=fail_final):
                with self.assertRaises(OSError):
                    prepare_native(path, output, env, workers=1)
            self.assertFalse(output.exists(), 'an unfinished output must not look published')
            result = prepare_native(path, output, env, workers=1)
            self.assertEqual(result['prepared'], 1)

    def test_native_dynamic_build_resume_and_ranking_with_no_real_endpoint(self):
        import numpy as np
        from app.few_shot.masker import MaskResult, TargetMaskResult
        from app.few_shot.preliminary_sql import PreliminarySQLResult
        from app.few_shot.retriever import FewShotRetriever
        from scripts.baseline_adapters.deepeye.workload_preparation import prepare_native
        from scripts.baseline_adapters.deepeye.workloads import load_workload, load_items
        class Embeddings:
            manages_retries = True
            def __call__(self, texts):
                return np.asarray([[len(text) + 1, sum(map(ord, text)) % 31 + 1] + [0.] * 1022
                                   for text in texts], dtype=np.float32)
            def map(self, function, items):
                return map(function, items)
        embeddings = Embeddings()
        env = {'DASH_MODELS': 'test', 'DASH_BASE_URL': 'https://example.test/v1', 'DASH_API_KEY': 'key',
               'EMBEDDING_MODEL': 'embed', 'EMBEDDING_BASE_URL': 'https://example.test/v1', 'EMBEDDING_API_KEY': 'key'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = self.fixture(root)
            training = root/'training'; training.mkdir()
            (training/'train.json').write_text(json.dumps([
                {'question_id': n, 'db_id': 'independent', 'question': f'train {n}',
                 'SQL': f'SELECT {n}', 'evidence': 'train evidence'} for n in range(9)]))
            spec = json.loads(path.read_text())
            spec.update(few_shot_strategy='native_dynamic', few_shot_source='training', few_shot_dataset='bird')
            path.write_text(json.dumps(spec))
            workload = load_workload(path); output = root/'ready.snapshot'
            real_retrieve = FewShotRetriever.retrieve_by_texts
            count = {'preliminary': 0, 'mask': 0}
            def preliminary(generator, item):
                count['preliminary'] += 1
                return PreliminarySQLResult('SELECT 1', ['SELECT 1'], 1, 1, 1.0,
                    {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30})
            def mask(**kwargs):
                count['mask'] += 1
                return TargetMaskResult(f"masked {kwargs['data_item'].question_id}", 'SELECT <number>', 'llm')
            def interrupted(retriever, **kwargs):
                if kwargs['masked_question'] == 'masked 19':
                    raise RuntimeError('simulated later retrieval interruption')
                return real_retrieve(retriever, **kwargs)
            with patch('socket.socket.connect', side_effect=AssertionError('must stay offline')), \
                 patch('app.few_shot.index_builder.mask_training_example',
                       side_effect=lambda example, *a, **kw: MaskResult(example.question, example.sql, 'llm')), \
                 patch('app.few_shot.preliminary_sql.PreliminarySQLGenerator.generate', preliminary), \
                 patch('app.few_shot.runtime._get_or_create_target_mask', side_effect=mask) as target_mask:
                # Mask injection inside the native formatter should honor its
                # already saved cache rather than count a second extraction.
                original_mask = target_mask.side_effect
                target_mask.side_effect = lambda **kw: (kw['cache'].get('unused') if
                    type(kw.get('cache')).__name__ == 'SimpleNamespace' else original_mask(**kw))
                with patch.object(FewShotRetriever, 'retrieve_by_texts', interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'question preparations failed'):
                        prepare_native(workload, output, env, workers=2, embedding_service=embeddings)
                self.assertFalse(output.exists())
                before = dict(count)
                result = prepare_native(workload, output, env, workers=2, embedding_service=embeddings)
                self.assertEqual(count, before, 'successful preliminary SQL and masks must survive retrieval failure')
                self.assertEqual(result['prepared'], 2)
                prepared, _, _ = load_items(dict(workload, prepared_dataset=str(output)))
                # Inspect the stored native index, not a fake ranking result.
                from scripts import deepeye_run as entry
                args = entry._build_parser().parse_args(['prepare', '--workload', str(path), '--run-dir', str(output)+'.preparation'])
                args.workload = workload; args.coordinator_workers = 2
                config = entry.build_runtime_config(env, args, Path(str(output)+'.preparation'))
                retriever = FewShotRetriever.from_index_path(config.few_shot_index_config.save_path,
                    config.few_shot_index_config.embedding, embedding_function=embeddings)
                for _, item in prepared:
                    expected = retriever.retrieve_by_texts(masked_question=f'masked {item.question_id}',
                        masked_sql='SELECT <number>', top_k=7, question_weight=.6, sql_weight=.4)
                    self.assertEqual([e['source_example_id'] for e in item.few_shot_examples],
                                     [r.example['example_id'] for r in expected])
                    self.assertEqual(item.few_shot_preparation_metadata['mode'], 'native_dynamic')
                    self.assertIn('diagnostics', item.few_shot_preparation_metadata['preliminary_sql'])
                    self.assertNotIn('SECRET_GOLD', json.dumps(item.few_shot_examples))
                before_bytes = output.read_bytes()
                self.assertTrue(prepare_native(workload, output, env, workers=2, embedding_service=embeddings)['reused'])
                self.assertEqual(before_bytes, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
