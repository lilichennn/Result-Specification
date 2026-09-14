"""Offline PG precompute integration through the real cached embedding service."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import openai

from scripts import deepeye_bird_interact_precompute as pg
from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataItem
from scripts.baseline_adapters.deepeye.embedding_service import (
    EmbeddingLimits, EmbeddingService, embedding_namespace,
)
from scripts.baseline_adapters.deepeye.precompute_cache import VectorCache, atomic_json, fingerprint
from scripts.baseline_adapters.deepeye.precompute_pipeline import PrecomputedInputReader, read_record
from tests.test_deepeye_precompute_api import VALUES, chat_reply, embed_reply, reply


ENV = {**VALUES, 'PG_HOST': 'offline-pg', 'PG_PORT': '5432'}


class PrecomputeServiceTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, dimension=2, embedding_handler=None):
        requests = []
        original_client = httpx.Client

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if 'input' in body:
                if embedding_handler is not None:
                    return embedding_handler(body)
                return embed_reply([[float(len(text)), 1.] + [0.] * (dimension - 2)
                                    for text in body['input']])
            return chat_reply('<result>["alpha", "gamma"]</result>')

        class OfflineClient(original_client):
            def __init__(self, *args, **kwargs):
                kwargs['transport'] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, patch.object(httpx, 'Client', OfflineClient):
            root = Path(temp)
            output = root / 'pg'
            schema = {'db_id': 'a', 'db_type': 'postgresql', 'tables': {
                'stars': {'columns': {'name': {'column_type': 'TEXT'}}}}}
            item = BirdInteractDataItem(question_id=1, instance_id='a_1', question='Find alpha',
                evidence='', database_id='a', database_path='a', database_schema=schema,
                gold_sql='', db_type='postgresql')
            atomic_json(output / 'databases' / 'a' / 'schema.json', schema)
            manifest = {'collection_fingerprint': 'offline-sample', 'schema_hash': fingerprint(schema),
                'columns': [{'table_name': 'stars', 'column_name': 'name',
                             'documents': ['alpha', 'beta'], 'status': 'collected'}],
                'policy': {'max_values_per_column': 1000},
                'provenance': {'source': {'host': ENV['PG_HOST'], 'port': 5432, 'database': 'a'}}}
            args = SimpleNamespace(output_dir=output, sample_cap=1000, limit=None,
                initial_concurrency=2, max_concurrency=2, window_seconds=60)
            with patch.object(pg, 'collect_manifests', return_value={'a': manifest}):
                yield root, args, [('lite', item)], {'a': item}, requests, manifest

    def test_injected_service_is_the_only_vector_ledger_and_keeps_reader_compatibility(self):
        with self.fixture() as (root, args, tasks, dbs, requests, _):
            namespace = embedding_namespace(ENV, dimension=2)
            with VectorCache(root / 'shared.sqlite', namespace) as cache, EmbeddingService(
                    ENV, cache, root / 'embedding.jsonl', limits=EmbeddingLimits(dimension=2)) as service:
                pg.compute(args, tasks, dbs, ENV, embedding_service=service)
                self.assertEqual(read_record(args.output_dir / 'run_config.json')['config']['embedding'], namespace)
                self.assertFalse((args.output_dir / 'vectors.sqlite').exists())
                sent = [text for request in requests for text in request.get('input', [])]
                self.assertCountEqual(sent, ['alpha', 'beta', 'gamma'])
                self.assertEqual(sent.count('alpha'), 1)
                loaded = PrecomputedInputReader(args.output_dir).load('lite', 'a_1', expected_item=tasks[0][1])
                self.assertEqual(loaded.question_keywords, ['alpha', 'gamma'])
                self.assertEqual(loaded.value_retrieval_llm_cost['total_tokens'], 5)
                self.assertTrue(loaded.is_stage_complete('value_retrieval'))
                self.assertTrue(pg.verify(args, tasks, dbs, embedding_service=service)['complete'])
                previous = list(requests)
                pg.compute(args, tasks, dbs, ENV, embedding_service=service)
                self.assertEqual(requests, previous)
                self.assertEqual(service.embed(['still-open']).shape, (1, 2))

    def test_standalone_compute_uses_shared_service_defaults_and_resumes_without_requests(self):
        with self.fixture(dimension=1024) as (_, args, tasks, dbs, requests, _):
            pg.compute(args, tasks, dbs, ENV)
            self.assertEqual(read_record(args.output_dir / 'run_config.json')['config']['embedding'],
                             embedding_namespace(ENV))
            self.assertTrue(pg.verify(args, tasks, dbs)['complete'])
            logs = list((args.output_dir / 'api_runs').glob('*/embedding_calls.jsonl'))
            self.assertEqual(len(logs), 1)
            summary = json.loads(next((args.output_dir / 'api_runs').glob('*/summary.json')).read_text())
            self.assertEqual(summary['config']['embedding_policy']['request_limit'], 64)
            self.assertEqual(summary['config']['embedding_policy']['batch_size'], 20)
            events = [json.loads(line) for line in logs[0].read_text().splitlines()]
            self.assertEqual({event['purpose'] for event in events}, {'database_values', 'question_keywords'})
            previous = list(requests)
            pg.compute(args, tasks, dbs, ENV)
            self.assertEqual(requests, previous)

    def test_standalone_resume_keeps_verified_legacy_namespace_and_vectors(self):
        with self.fixture(dimension=1024) as (_, args, tasks, dbs, requests, manifest):
            config = pg.semantic_config(ENV, {'a': manifest})
            legacy = config['embedding']
            self.assertNotIn('dimension', legacy)
            pg.freeze_config(args.output_dir, config)
            with VectorCache(args.output_dir / 'vectors.sqlite', legacy) as cache:
                cache.put_many(['alpha', 'beta', 'gamma'], [[1., 2.] + [0.] * 1022] * 3)
            pg.compute(args, tasks, dbs, ENV)
            self.assertFalse(any('input' in request for request in requests))
            self.assertEqual(read_record(args.output_dir / 'run_config.json')['config']['embedding'], legacy)
            self.assertTrue(pg.verify(args, tasks, dbs)['complete'])

    def test_adaptive_api_delegates_embedding_retries_and_does_not_close_injected_service(self):
        from scripts.baseline_adapters.deepeye.precompute_api import AdaptiveAPI
        with self.fixture() as (root, _, _, _, requests, _):
            with VectorCache(root / 'shared.sqlite', embedding_namespace(ENV, dimension=2)) as cache, \
                    EmbeddingService(ENV, cache, root / 'embedding.jsonl',
                        limits=EmbeddingLimits(dimension=2, retry_delay=0)) as service:
                with AdaptiveAPI(ENV, root / 'chat', embedding_service=service) as api:
                    self.assertEqual(api.embed(['alpha']), [[5., 1.]])
                    self.assertEqual(api.summary()['attempts'], 0)
                    self.assertEqual(api.embed(['alpha']), [[5., 1.]])
                self.assertEqual(len(requests), 1)
                self.assertEqual(service.embed(['gamma']).shape, (1, 2))
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    api.embed(['after-close'])

    def test_document_preparation_keeps_only_bounded_vector_batches_in_memory(self):
        with self.fixture() as (root, args, tasks, dbs, _, manifest):
            manifest['columns'][0]['documents'] = [f'text-{i}' for i in range(45)]
            with VectorCache(root / 'shared.sqlite', embedding_namespace(ENV, dimension=2)) as cache, \
                    EmbeddingService(ENV, cache, root / 'embedding.jsonl',
                        limits=EmbeddingLimits(dimension=2)) as service:
                original = service.embed

                def bounded(texts, **kwargs):
                    self.assertLessEqual(len(texts), service.limits.batch_size)
                    return original(texts, **kwargs)

                with patch.object(service, 'embed', side_effect=bounded):
                    pg.compute(args, tasks, dbs, ENV, embedding_service=service)
                self.assertEqual(cache.count(), 47)

    def test_embedding_failures_have_four_service_attempts_not_nested_chat_retries(self):
        from scripts.baseline_adapters.deepeye.precompute_api import AdaptiveAPI
        with self.fixture(embedding_handler=lambda _: reply({'error': {'message': 'temporary'}}, 503)) as (
                root, _, _, _, requests, _):
            with VectorCache(root / 'shared.sqlite', embedding_namespace(ENV, dimension=2)) as cache, \
                    EmbeddingService(ENV, cache, root / 'embedding.jsonl',
                        limits=EmbeddingLimits(dimension=2, retry_delay=0)) as service, \
                    AdaptiveAPI(ENV, root / 'chat', embedding_service=service) as api:
                with self.assertRaises(openai.InternalServerError):
                    api.embed(['failed'])
                self.assertEqual(len(requests), 4)
                self.assertEqual(api.summary()['attempts'], 0)
                self.assertEqual(cache.count(), 0)
                events = [json.loads(line) for line in (root / 'embedding.jsonl').read_text().splitlines()]
                events = [event for event in events if event['kind'] == 'embedding_request']
                self.assertEqual([event['attempt'] for event in events], [1, 2, 3, 4])
                self.assertTrue(all(event['status'] == 'error' for event in events))

    def test_failed_database_batch_resumes_successful_cache_without_rebilling(self):
        failing = [True]

        def handler(body):
            if failing[0] and any(text.startswith('z-failed') for text in body['input']):
                return reply({'error': {'message': 'temporary'}}, 503)
            return embed_reply([[float(len(text)), 1.] for text in body['input']])

        with self.fixture(embedding_handler=handler) as (root, args, tasks, dbs, requests, manifest):
            successful = [f'good-{i:02}' for i in range(20)]
            manifest['columns'][0]['documents'] = successful + [f'z-failed-{i}' for i in range(5)]
            with VectorCache(root / 'shared.sqlite', embedding_namespace(ENV, dimension=2)) as cache, \
                    EmbeddingService(ENV, cache, root / 'embedding.jsonl', limits=EmbeddingLimits(
                        dimension=2, retry_delay=0, request_workers=2, request_limit=2,
                        http_connections=2)) as service:
                with self.assertRaises(openai.InternalServerError):
                    pg.compute(args, tasks, dbs, ENV, embedding_service=service)
                self.assertEqual(cache.count(), 20)
                self.assertFalse(any('messages' in request for request in requests))
                self.assertFalse((args.output_dir / 'value_index').exists())
                failing[0] = False
                pg.compute(args, tasks, dbs, ENV, embedding_service=service)
                sent = [text for request in requests for text in request.get('input', [])]
                self.assertTrue(all(sent.count(text) == 1 for text in successful))
                self.assertEqual(cache.count(), 27)
                self.assertTrue(pg.verify(args, tasks, dbs, embedding_service=service)['complete'])


if __name__ == '__main__':
    unittest.main()
