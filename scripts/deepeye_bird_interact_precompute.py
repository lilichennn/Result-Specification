"""Precompute reusable DeepEye values, keyword embeddings and retrieval artifacts.

Use --stage collect, compute, or verify. No baseline source modifications.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = CODE_ROOT / 'baselines/DeepEye-SQL'
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))

import numpy as np

from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataset, BirdInteractDatasetConfig
from scripts.baseline_adapters.deepeye.precompute_cache import VectorCache, atomic_json, fingerprint
from scripts.baseline_adapters.deepeye.precompute_pipeline import (
    build_native_index, file_hash, item_directory, load_precomputed_item,
    native_index_identity, prepare_keywords, question_row, read_record, retrieve_item, schema_with_values,
    utc_now, verify_native_index, write_record,
    validate_token_usage,
)


def load_inputs():
    tasks = []
    databases = {}
    for variant in ('lite', 'full'):
        source = CODE_ROOT / 'scripts' / f'bird_interact_{variant}' / 'preprocessed_data'
        dataset = BirdInteractDataset(BirdInteractDatasetConfig(split=variant, root_path=str(source)))
        for item in dataset:
            tasks.append((variant, item))
            prior = databases.setdefault(item.database_id, item)
            if fingerprint(prior.database_schema) != fingerprint(item.database_schema):
                raise ValueError('One database ID refers to conflicting Meta schemas')
    validate_inventory(tasks, databases)
    return tasks, databases


def validate_inventory(tasks, databases):
    identities = [(variant, item.instance_id) for variant, item in tasks]
    counts = {variant: sum(v == variant for v, _ in tasks) for variant in ('lite', 'full')}
    db_counts = {variant: len({item.database_id for v, item in tasks if v == variant}) for variant in counts}
    if counts != {'lite': 195, 'full': 410} or db_counts != {'lite': 18, 'full': 22} or len(databases) != 40:
        raise ValueError('Expected exactly Lite 195/18 DBs and Full 410/22 DBs')
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate question identity in precompute input')


def freeze_inventory(root, tasks, databases):
    payload = {'questions': [{'variant': variant, **question_row(item)} for variant, item in tasks],
               'schemas': {db: fingerprint(item.database_schema) for db, item in databases.items()}}
    path = root / 'inputs.json'
    if path.exists():
        prior = read_record(path)
        if any(prior.get(key) != value for key, value in payload.items()):
            raise ValueError('Input inventory changed; refusing to overwrite the frozen population')
        return prior
    return write_record(path, payload)


def parallel_map(label, jobs, function, *, workers, output):
    results, failures = {}, {}
    started = time.monotonic()
    last_log = started
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=label) as pool:
        futures = {pool.submit(function, job): key for key, job in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                failures[key] = {'error_type': type(exc).__name__}
                print(f'{label} FAILED {key}: {type(exc).__name__}', flush=True)
            completed = len(results) + len(failures)
            now = time.monotonic()
            if completed % 25 == 0 or completed == len(jobs) or now - last_log >= 15:
                print(f'{label} {completed}/{len(jobs)}; failed={len(failures)}; elapsed={now-started:.1f}s', flush=True)
                atomic_json(output / f'{label}_progress.json', {'total': len(jobs), 'completed': completed,
                    'succeeded': len(results), 'failures': failures, 'updated_at': utc_now()})
                last_log = now
    if failures:
        raise RuntimeError(f'{label}: {len(failures)} failed items; successful work is checkpointed')
    return results


def collect(args, databases, env):
    from scripts.baseline_adapters.deepeye.precompute_values import collect_database_values
    kwargs = {'host': env['PG_HOST'], 'port': int(env['PG_PORT']), 'user': env['PG_USER'], 'password': env['PG_PASSWORD']}
    manifests = parallel_map('collect', list(databases.items()), lambda item:
        collect_database_values(item, args.output_dir / 'databases' / item.database_id,
            connection_kwargs=kwargs, max_values_per_column=args.sample_cap, timeout_seconds=60),
        workers=4, output=args.output_dir)
    atomic_json(args.output_dir / 'collection_summary.json', {'complete': True, 'updated_at': utc_now(),
        'databases': {db: manifest['stats'] for db, manifest in manifests.items()}, 'errors': {}})


def collect_manifests(root, databases, expected_cap=1000):
    from scripts.baseline_adapters.deepeye.precompute_values import verify_collection
    manifests = {}
    for db, item in databases.items():
        manifest = verify_collection(root / 'databases' / db)
        if manifest['status'] != 'complete' or manifest['schema_hash'] != fingerprint(item.database_schema):
            raise ValueError(f'Database sampling incomplete or Meta changed: {db}')
        expected_columns = {(table, column) for table, info in item.database_schema['tables'].items()
            for column, col in info['columns'].items() if col['column_type'].upper() == 'TEXT'
            or col['column_type'].upper().startswith(('VARCHAR', 'CHAR'))}
        actual_columns = {(col['table_name'], col['column_name']) for col in manifest['columns']}
        if actual_columns != expected_columns or len(actual_columns) != len(manifest['columns']):
            raise ValueError(f'Sampling is missing or duplicates declared columns: {db}')
        if any(col['status'] not in ('collected', 'empty', 'filtered_uuid', 'filtered_numeric') for col in manifest['columns']):
            raise ValueError(f'Sampling has unfinished columns: {db}')
        if manifest['policy']['max_values_per_column'] != expected_cap:
            raise ValueError(f'Sampling cap does not match requested cap: {db}')
        manifests[db] = manifest
    return manifests


def sampling_identity(manifests):
    return {db: {'collection_fingerprint': manifest['collection_fingerprint'],
                 'sample_content_hash': fingerprint({'schema_hash': manifest['schema_hash'], 'columns': manifest['columns'],
                    'policy': manifest['policy'], 'source': manifest['provenance']['source']})}
            for db, manifest in manifests.items()}


def semantic_config(env, manifests):
    from app.pipeline.value_retrieval import utils
    return {'version': 2,
        'embedding': {'model': env['EMBEDDING_MODEL'], 'endpoint': env['EMBEDDING_BASE_URL'].rstrip('/'),
                      'encoding_format': 'float', 'storage': 'float32', 'local_metric': 'cosine'},
        'chat': {'model': env['DASH_MODELS'], 'endpoint': env['DASH_BASE_URL'].rstrip('/'),
                 'temperature': 0.6, 'max_tokens': 2048, 'thinking_budget': 1024,
                 'parser_source_hash': file_hash(utils.__file__), 'parse_attempts': 3, 'fallback': 'fail'},
        'retrieval': {'max_values_per_column': 5, 'lower_meta_data': True, 'device': 'cpu'},
        'sampling': sampling_identity(manifests), 'keyword_postprocess': 'sorted-native-nonempty-v1',
        'keyword_cost': 'atomic-per-logical-call-reported-tokens-v1'}


def freeze_config(root, config):
    path = root / 'run_config.json'
    if path.exists():
        previous = read_record(path)
        if previous['config'] != config:
            raise ValueError('Precompute configuration changed; choose a new output directory')
        return previous
    return write_record(path, {'config': config, 'created_at': utc_now()})


def compute(args, tasks, databases, env):
    import resource
    import torch
    from app.vector_db.local_index import LocalValueIndex
    from scripts.baseline_adapters.deepeye.precompute_api import AdaptiveAPI
    # HTTP concurrency needs sufficient sockets; only raise this process's soft limit.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired = min(4096, hard) if hard != resource.RLIM_INFINITY else 4096
    if soft < desired:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    torch.set_num_threads(1)
    manifests = collect_manifests(args.output_dir, databases, args.sample_cap)
    config = semantic_config(env, manifests)
    for manifest in manifests.values():
        source = manifest['provenance']['source']
        if source['host'] != env['PG_HOST'] or source['port'] != int(env['PG_PORT']):
            raise ValueError('Database source changed since sample collection')
    freeze_config(args.output_dir, config)
    freeze_inventory(args.output_dir, tasks, databases)
    chosen = tasks[:args.limit] if args.limit else tasks
    unique_documents = sorted({value for manifest in manifests.values() for col in manifest['columns'] for value in col['documents']})
    print(f'COMPUTE inputs={len(tasks)} databases={len(databases)} unique_database_texts={len(unique_documents)}', flush=True)
    api_dir = args.output_dir / 'api_runs' / utc_now().replace(':', '-').replace('.', '-')
    with AdaptiveAPI(env, api_dir, initial_concurrency=args.initial_concurrency,
            max_concurrency=args.max_concurrency, window_seconds=args.window_seconds) as api, \
            VectorCache(args.output_dir / 'vectors.sqlite', config['embedding']) as cache:
        try:
            missing = [text for text in unique_documents if cache.get(text) is None]
            batches = [(str(i // 20), missing[i:i+20]) for i in range(0, len(missing), 20)]
            parallel_map('database_embeddings', batches,
                lambda texts: len(cache.embed(texts, api.embed)), workers=args.max_concurrency, output=args.output_dir)
            index_records = {}
            for db, manifest in manifests.items():
                index_records[db] = build_native_index(args.output_dir / 'value_index' / db, manifest, cache)
            print(f'DATABASE_INDEXES_READY {len(index_records)} dimension={cache.dimension}', flush=True)

            def keywords_for(task):
                variant, item = task
                directory = item_directory(args.output_dir, variant, item.instance_id)
                record = prepare_keywords(question_row(item), config['chat'], directory / 'keywords.json', api.chat_with_usage)
                return record
            keyword_records = parallel_map('keywords', [(f'{variant}/{item.instance_id}', (variant, item)) for variant, item in chosen],
                keywords_for, workers=args.max_concurrency, output=args.output_dir)
            unique_keywords = sorted({text for record in keyword_records.values() for text in record['keywords']})
            missing = [text for text in unique_keywords if cache.get(text) is None]
            batches = [(str(i // 20), missing[i:i+20]) for i in range(0, len(missing), 20)]
            parallel_map('keyword_embeddings', batches,
                lambda texts: len(cache.embed(texts, api.embed)), workers=args.max_concurrency, output=args.output_dir)

            indexes = {db: LocalValueIndex(args.output_dir / 'value_index' / db / 'local_index', device='cpu') for db in databases}
            def finish_item(task):
                variant, item = task
                directory = item_directory(args.output_dir, variant, item.instance_id)
                if (directory / 'retrieval.json').exists():
                    existing = read_record(directory / 'retrieval.json')
                    if existing['index_hash'] != index_records[item.database_id]['content_hash'] or existing['embedding_namespace'] != cache.namespace:
                        raise ValueError('Saved retrieval refers to a different index or embedding model')
                    load_precomputed_item(args.output_dir, variant, item.instance_id, expected_item=item)
                    return True
                started = time.monotonic()
                keyword_record = keyword_records[f'{variant}/{item.instance_id}']
                vectors = cache.read(keyword_record['keywords'])
                np.save(directory / 'keywords.npy', vectors, allow_pickle=False)
                retrieved = retrieve_item(item, vectors, indexes[item.database_id])
                enriched = schema_with_values(item.database_schema, retrieved)
                usage = validate_token_usage(keyword_record['usage'])
                tokens = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                write_record(directory / 'retrieval.json', {'variant': variant, 'row': question_row(item),
                    'question_id': item.question_id, 'schema_hash': fingerprint(item.database_schema),
                    'keywords_hash': keyword_record['content_hash'], 'vectors_hash': file_hash(directory / 'keywords.npy'),
                    'index_hash': index_records[item.database_id]['content_hash'], 'embedding_namespace': cache.namespace,
                    'dimension': cache.dimension, 'max_values_per_column': 5, 'retrieved_values': retrieved,
                    'retrieved_schema_hash': fingerprint(enriched), 'value_retrieval_llm_cost': tokens,
                    'cost_semantics': 'reported_tokens_only', 'usage_missing': usage.get('usage_missing', 0),
                    'value_retrieval_time': time.monotonic() - started + keyword_record['precompute_seconds'],
                    'created_at': utc_now()})
                load_precomputed_item(args.output_dir, variant, item.instance_id, expected_item=item)
                return True
            parallel_map('retrieval', [(f'{variant}/{item.instance_id}', (variant, item)) for variant, item in chosen],
                finish_item, workers=4, output=args.output_dir)
        finally:
            atomic_json(api_dir / 'summary.json', api.summary())
            print('API_SUMMARY', json.dumps(api.summary(), ensure_ascii=False), flush=True)


def verify(args, tasks, databases):
    started = time.monotonic()
    config = read_record(args.output_dir / 'run_config.json')['config']
    freeze_inventory(args.output_dir, tasks, databases)
    manifests = collect_manifests(args.output_dir, databases, args.sample_cap)
    if config['sampling'] != sampling_identity(manifests):
        raise ValueError('Frozen database sample contents/policy/source changed')
    indexes = {db: verify_native_index(args.output_dir / 'value_index' / db) for db in databases}
    errors = {}
    counts = {'lite': 0, 'full': 0}
    with VectorCache(args.output_dir / 'vectors.sqlite', config['embedding']) as cache:
        for db, manifest in manifests.items():
            if (indexes[db]['identity'] != native_index_identity(manifest, cache.namespace)
                    or indexes[db]['schema_hash'] != manifest['schema_hash']
                    or indexes[db]['embedding_namespace'] != cache.namespace):
                raise ValueError('Index does not match current complete sample manifest/model')
            for column in manifest['columns']:
                cache.read(column['documents'])
        for variant, item in tasks:
            try:
                loaded = load_precomputed_item(args.output_dir, variant, item.instance_id, expected_item=item)
                directory = item_directory(args.output_dir, variant, item.instance_id)
                record = read_record(directory / 'retrieval.json')
                if record['index_hash'] != indexes[item.database_id]['content_hash']:
                    raise ValueError('Retrieval index version mismatch')
                if not np.array_equal(cache.read(loaded.question_keywords), np.load(directory / 'keywords.npy', allow_pickle=False)):
                    raise ValueError('Keyword vector row order does not match text cache')
                if not loaded.is_stage_complete('value_retrieval'):
                    raise ValueError('Reconstructed native value-retrieval stage is incomplete')
                counts[variant] += 1
            except Exception as exc:
                errors[f'{variant}/{item.instance_id}'] = type(exc).__name__
        summary = {'complete': sum(counts.values()) == len(tasks) and not errors, 'question_counts': counts,
            'expected_questions': len(tasks), 'database_count': len(indexes), 'dimension': cache.dimension,
            'unique_cached_texts': cache.count(), 'indexed_column_value_pairs': sum(rec['documents'] for rec in indexes.values()),
            'errors': errors, 'verification_seconds': time.monotonic()-started, 'verified_at': utc_now(), 'network_calls': 0}
    atomic_json(args.output_dir / 'verification.json', summary)
    print('VERIFY', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('collect', 'compute', 'verify'), required=True)
    parser.add_argument('--output-dir', type=Path, default=CODE_ROOT / 'baselines_reproduce/deepeye_bird_interact/precomputed')
    parser.add_argument('--env-file', type=Path, default=CODE_ROOT / 'config/.env')
    parser.add_argument('--sample-cap', type=int, default=1000)
    parser.add_argument('--initial-concurrency', type=int, default=200)
    parser.add_argument('--max-concurrency', type=int, default=600)
    parser.add_argument('--window-seconds', type=float, default=60)
    parser.add_argument('--limit', type=int, help='Compute only this many questions; all databases still prepared')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, databases = load_inputs()
    if args.stage == 'verify':
        return 0 if verify(args, tasks, databases)['complete'] else 1
    from scripts.deepeye_bird_interact_smoke import read_environment
    env = read_environment(args.env_file)
    if args.stage == 'collect':
        collect(args, databases, env)
    else:
        compute(args, tasks, databases, env)
        if not args.limit:
            return 0 if verify(args, tasks, databases)['complete'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
