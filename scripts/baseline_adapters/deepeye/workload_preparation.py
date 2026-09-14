"""Explicit preparation using DeepEye's existing vector and few-shot implementations."""
from __future__ import annotations

import json
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile

from .workloads import load_workload, load_items, file_sha256, _prepared_items


def _dataset(workload, items, config):
    from app.dataset.dataset import BirdDataset, SpiderDataset
    from app.dataset.spider2_dataset import Spider2LiteDataset
    from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataset
    cls = {'bird': BirdDataset, 'spider': SpiderDataset, 'spider2': Spider2LiteDataset,
           'bird_interact': BirdInteractDataset}[workload['benchmark']]
    dataset = object.__new__(cls)
    dataset._config = config.dataset_config
    dataset._data = items
    dataset._database_schema_cache = {}
    return dataset


def prepare_native(workload, output, environment, *, workers=200, embedding_service=None):
    from .preparation_store import preparation_lock
    if getattr(getattr(embedding_service, 'limits', None), 'dimension', 1024) != 1024:
        raise ValueError('Shared native preparation currently requires dimension 1024')
    with preparation_lock(Path(output).resolve().with_name(Path(output).name + '.lock')):
        return _prepare_native(workload, output, environment, workers=workers, embedding_service=embedding_service)


def _prepare_native(workload, output, environment, *, workers, embedding_service):
    """Materialize a native snapshot, failing if any required preparation is absent.

Network work happens only in this explicit command, never in prepare_inputs.
Existing successful native VR and few-shot inputs are retained on continuation.
"""
    from app.dataset.utils import save_dataset
    from scripts import deepeye_run as entry
    if workers < 1:
        raise ValueError('Preparation workers must be positive')
    workload = load_workload(workload)
    output = Path(output).resolve()
    # Independent output snapshots must not silently reuse each other's native
    # indexes. Explicit native_config paths still allow intentional sharing.
    preparation_root = output.with_name(output.name + '.preparation')
    args = entry._build_parser().parse_args(['prepare', '--workload', workload['_path'],
                                          '--run-dir', str(preparation_root)])
    args.workload = workload
    args.coordinator_workers = workers
    config = entry.build_runtime_config(environment, args, preparation_root)
    tasks, _, sources = load_items(workload, require_prepared=False)
    items = [item for _, item in tasks]
    dynamic = workload.get('few_shot_strategy') == 'native_dynamic'
    sources['database_sha256'] = {path: entry._file_sha256(Path(path)) for path in sorted({
        item.database_path for item in items if Path(item.database_path).is_file()})}
    identity = _preparation_identity(workload, config, sources)
    if output.exists():
        manifest = json.loads(output.read_text())
        if manifest.get('preparation', {}).get('identity') != identity:
            raise ValueError('Completed preparation identity changed; use a new output')
        checked, _, _ = load_items(dict(workload, prepared_dataset=str(output)))
        return {'prepared': len(checked), 'prepared_dataset': str(output), 'sha256': file_sha256(output),
                'workload': workload['_path'], 'reused': True}
    needs_vr = any(item.database_schema_after_value_retrieval is None for item in items)
    # The native Spider2 template has no few-shot index; do not invent one.
    needs_examples = workload['benchmark'] != 'spider2' and any(not item.few_shot_examples for item in items)
    few = config.few_shot_index_config
    if needs_vr and workload['benchmark'] == 'bird_interact' and not dynamic:
        raise ValueError('BIRD-Interact requires precompute_dir from deepeye_bird_interact_precompute.py')
    if needs_examples and (workload['benchmark'] != 'bird_interact' or dynamic):
        if few.embedding is None or few.llm is None:
            raise ValueError('Native few-shot preparation requires native_config [few_shot_index.embedding] and [few_shot_index.llm]')
        _validate_few_shot_index(workload, few, config.run_config)
        if not (Path(few.save_path) / 'manifest.json').exists() and not workload.get('few_shot_source'):
            raise ValueError('Native few-shot index missing; supply few_shot_source training root to build it')
        if dynamic and not (Path(few.save_path) / 'manifest.json').exists():
            # Own the parent before native mask-cache markers create its contents.
            _bind_resource_identity(few.save_path, _training_identity(workload, few, config.run_config))
    if needs_examples and workload['benchmark'] == 'bird_interact' and not workload.get('few_shot_source'):
        raise ValueError('BIRD-Interact requires few_shot_source training JSON')
    dataset = _dataset(workload, items, config)
    preparation = {'format': 'deepeye-native-preparation-v1', 'sources': sources,
                   'effective_config': entry.build_effective_config(environment, args), 'identity': identity}
    resources = _bind_preparation_resources(items, config, needs_vr=needs_vr and workload['benchmark'] != 'bird_interact',
        needs_examples=needs_examples and workload['benchmark'] != 'spider2',
        database_hashes=sources['database_sha256'])
    if needs_vr:
        _bind_resource_identity(output.with_name(output.name + '.vr.artifacts'),
                                {'preparation': preparation, 'resources': resources})
    with ExitStack() as stack:
        stack.enter_context(entry.backend_context(environment, args))
        runtime, audit = None, None
        if dynamic and (needs_vr or needs_examples):
            from .preparation_store import PreparationAudit
            from .run_resources import SamplingRuntime
            from .embedding_service import EmbeddingService, embedding_namespace
            from .precompute_cache import VectorCache
            if embedding_service is None:
                shared = Path(workload.get('preparation_root', preparation_root))
                cache = stack.enter_context(VectorCache(shared/'vectors.sqlite', embedding_namespace(environment)))
                embedding_service = stack.enter_context(EmbeddingService(environment, cache, shared/'embedding_calls.jsonl'))
            audit = PreparationAudit(preparation_root/'chat_calls.jsonl')
            runtime = SamplingRuntime(**entry.runtime_limits(args), emit=audit.emit)
            stack.callback(runtime.close)
            stack.enter_context(runtime.context())
            # PostgreSQL execution uses its established independent admission cap.
            stack.enter_context(entry.admission_context(SimpleNamespace(record_admission=audit.emit), args, population=len(items)))
        if needs_vr and workload['benchmark'] == 'bird_interact':
            from scripts import deepeye_bird_interact_precompute as pg
            pg_root = Path(workload.get('precompute_dir') or preparation_root/'postgres')
            pg_root.mkdir(parents=True, exist_ok=True)
            pg_args = SimpleNamespace(output_dir=pg_root, sample_cap=1000, limit=None,
                initial_concurrency=min(workers, 200), max_concurrency=workers, window_seconds=60)
            pg_tasks = [(workload['split'], item) for item in items]
            databases = {item.database_id: item for item in items}
            pg.collect(pg_args, databases, environment)
            pg.compute(pg_args, pg_tasks, databases, environment, embedding_service=embedding_service)
            from .precompute_pipeline import PrecomputedInputReader
            reader = PrecomputedInputReader(pg_root)
            items = [reader.load(workload['split'], item.instance_id, expected_item=item) for item in items]
            dataset = _dataset(workload, items, config)
            needs_vr = False
        if needs_vr:
            # Native runners consume their existing structured snapshot format.
            input_path = output.with_name(output.name + '.input.snapshot')
            config.dataset_config.save_path = str(input_path)
            config.value_retrieval_config.save_path = str(output.with_name(output.name + '.vr.snapshot'))
            save_dataset(dataset, str(input_path))
            from runner.create_vector_db_parallel import run_vector_db_creation
            from .preparation_store import preparation_lock
            vector_root = Path(config.vector_database_config.store_root_path)
            with preparation_lock(vector_root.with_name(vector_root.name + '.build.lock')):
                run_vector_db_creation(str(input_path), workload['benchmark'], config.vector_database_config,
                    min(workers, 16), config.run_config.embedding_batch_size, config.run_config.progress_log_interval,
                    database_parallelism=min(workers, 4),
                    **({'embedding_function': embedding_service} if embedding_service is not None else {}))
            from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner
            runner = ValueRetrievalRunner.from_config(config,
                **({'embedding_function': embedding_service} if embedding_service is not None else {}))
            runner._llm.sample_max_attempts = 4
            if runtime is not None:
                runtime.bind_runner(runner)
                audit.bind_llm(runner._llm, runtime, bind_client=False)
                from .preparation_store import bind_keyword_checkpoints
                bind_keyword_checkpoints(runner, preparation_root/'keywords', audit=audit,
                    label={'benchmark': workload['benchmark'], 'split': workload['split']})
            try:
                runner.run()
            except BaseException:
                # Native run() already cleans up on success.
                runner._clean_up()
                raise
            finally:
                client = getattr(runner._llm, '_client', None)
                if client is not None:
                    client.close()
            loaded, _ = _prepared_items(config.value_retrieval_config.save_path, workload['benchmark'])
            items = list(loaded.values())
            dataset = _dataset(workload, items, config)
        if needs_examples:
            if workload['benchmark'] == 'bird_interact' and not dynamic:
                from scripts.deepeye_bird_interact_smoke import IndependentExampleReader
                reader = IndependentExampleReader(Path(workload['few_shot_source']))
                for item in items:
                    if not item.few_shot_examples:
                        item.few_shot_examples, provenance = reader.select(item, count=3)
                        item.few_shot_preparation_metadata = provenance
            else:
                extra = {'embedding_service': embedding_service, 'runtime': runtime, 'audit': audit,
                         'checkpoint_root': preparation_root/'questions', 'workers': workers} if dynamic else {}
                _prepare_examples(workload, items, config, **extra)
        if any(not item.is_stage_complete('value_retrieval') or
               (workload['benchmark'] != 'spider2' and not item.few_shot_examples) for item in items):
            raise ValueError('Native preparation incomplete; no executable snapshot produced')
        for item in items:
            item.gold_sql = ''
        _publish_snapshot(dataset, output, preparation, workload)
    return {'prepared': len(items), 'prepared_dataset': str(output), 'sha256': file_sha256(output),
            'workload': workload['_path'], 'reused': False, 'next': 'Set prepared_dataset in the workload to this native snapshot'}


def _publish_snapshot(dataset, output, preparation, workload):
    """Use native serialization; atomically publish a verified final manifest last."""
    from app.dataset.utils import save_dataset
    from .precompute_cache import atomic_json
    output = Path(output)
    data = output.with_name(output.name + '.data')
    _bind_resource_identity(data, {'output': str(output), 'identity': preparation['identity']})
    with tempfile.TemporaryDirectory(prefix='.deepeye-publish-', dir=output.parent) as tmp:
        staging = Path(tmp)/output.name
        save_dataset(dataset, str(staging))
        load_items(dict(workload, prepared_dataset=str(staging)))
        manifest = json.loads(staging.read_text())
        items_path = staging.parent/manifest['snapshot_root']/'items.jsonl'
        manifest['preparation'] = dict(preparation, items_sha256=file_sha256(items_path))
        manifest['snapshot_root'] = data.name
        data.mkdir(parents=True, exist_ok=True)
        os.replace(items_path, data/'items.jsonl')
        atomic_json(output, manifest)


def _preparation_identity(workload, config, sources):
    from .precompute_cache import fingerprint
    def public(value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json')
        if isinstance(value, dict):
            return {key: public(item) for key, item in value.items()
                    if key not in ('api_key', 'password', 'parallelism', 'embedding_batch_size')}
        if isinstance(value, list):
            return [public(item) for item in value]
        return value
    training = (_training_identity(workload, config.few_shot_index_config, config.run_config)
                if workload.get('few_shot_strategy') == 'native_dynamic' and workload.get('few_shot_source') else None)
    return fingerprint({'questions': sources['questions_sha256'], 'meta': sources['meta_sha256'],
        'databases': sources.get('database_sha256'), 'training': training,
        'benchmark': workload['benchmark'], 'split': workload['split'], 'selection': workload.get('question_ids'),
        'source': workload.get('few_shot_source'), 'strategy': workload.get('few_shot_strategy'),
        'dataset': public(config.dataset_config), 'embedding': public(config.vector_database_config),
        'examples': public(config.few_shot_index_config), 'vr': public(config.value_retrieval_config),
        'llm_timeout': config.run_config.llm_timeout})


def _bind_resource_identity(resource, identity):
    """A small provenance marker, never a second copy of cached data.

    Native success flags/mask keys do not bind model or database versions.
    Refuse unowned data; do not delete or retroactively bless old caches.
    """
    from .precompute_cache import fingerprint
    resource = Path(resource)
    marker = resource.with_name(resource.name + '.preparation_identity.json')
    expected = {'format': 'deepeye-native-resource-v1', 'sha256': fingerprint(identity)}
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != expected:
            raise ValueError('Native preparation resource identity changed; select a new resource path')
        return
    if resource.exists() and (not resource.is_dir() or any(resource.iterdir())):
        raise ValueError('Existing native resource has no verifiable provenance; select a new resource path')
    marker.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: interrupted/competing writes fail closed on validation.
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(expected, stream, sort_keys=True)


def _bind_preparation_resources(items, config, *, needs_vr, needs_examples, database_hashes=None):
    if not needs_vr and not needs_examples:
        return {}
    from scripts import deepeye_run as entry
    from app.few_shot.index_builder import _redact_config
    baseline = entry.code_source_hashes()['baseline_python_sha256']
    resources = {'baseline': baseline, 'databases': {}}
    def public_config(value):
        return {key: item for key, item in _redact_config(value).items() if key != 'api_key'}
    if needs_vr:
        # One streaming digest per distinct SQLite file, not one per question.
        databases = {item.database_path for item in items if Path(item.database_path).is_file()}
        for database in sorted(databases):
            path = Path(database)
            identity = {'baseline': baseline, 'database_path': str(path.resolve()),
                        'database_sha256': (database_hashes[str(path)] if database_hashes is not None else entry._file_sha256(path)),
                        'vector_config': public_config(config.vector_database_config)}
            resources['databases'][str(path.resolve())] = identity['database_sha256']
            _bind_resource_identity(Path(config.vector_database_config.store_root_path)/path.stem, identity)
    if needs_examples:
        few = config.few_shot_index_config
        identity = {'baseline': baseline, 'llm': public_config(few.llm),
                    'timeout': config.run_config.llm_timeout}
        index = Path(few.save_path)
        # Completed native training indexes have their own verified manifest;
        # this guard is for mask caches whose native keys omit the model.
        if not (index/'manifest.json').exists():
            _bind_resource_identity(Path(few.mask_cache_path) if few.mask_cache_path else index/'mask_cache.jsonl', identity)
        target = Path(few.target_mask_cache_path) if few.target_mask_cache_path else index/'target_mask_cache.jsonl'
        _bind_resource_identity(target, identity)
    return resources


def _validate_few_shot_index(workload, few, run_config):
    """Check native index provenance before any paid preparation or reuse."""
    from app.few_shot.index_builder import _redact_config
    path = Path(few.save_path) / 'manifest.json'
    if few.force_rebuild:
        raise ValueError('force_rebuild is unsupported here; select a new native index path instead of overwriting')
    if not path.exists():
        if path.parent.is_dir() and any(path.parent.iterdir()):
            from .precompute_cache import fingerprint
            marker = path.parent.with_name(path.parent.name + '.preparation_identity.json')
            if (not marker.is_file() or not workload.get('few_shot_source') or
                    json.loads(marker.read_text()).get('sha256') != fingerprint(_training_identity(workload, few, run_config))):
                raise ValueError('Cannot verify an incomplete native few-shot index; select a new index path')
        return
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if workload.get('few_shot_strategy') == 'native_dynamic' and any(
            manifest.get('embedding', {}).get(field) != 1024
            for field in ('question_embedding_dim', 'sql_embedding_dim')):
        raise ValueError('Native dynamic training index must have 1024-dimensional vectors')
    training_type = workload.get('few_shot_dataset', 'bird' if workload['benchmark'] == 'bird_interact' else workload['benchmark'])
    def config_without_key(value):
        return {key: item for key, item in value.items() if key != 'api_key'}
    if (manifest.get('dataset_type') != training_type or
            manifest.get('max_samples') != few.max_samples or
            manifest.get('max_samples_per_db') != few.max_samples_per_db or
            manifest.get('masking', {}).get('skip_mask_llm') is not False or
            manifest.get('masking', {}).get('llm_timeout') != run_config.llm_timeout or
            config_without_key(manifest.get('embedding', {}).get('config', {})) !=
                config_without_key(_redact_config(few.embedding)) or
            config_without_key(manifest.get('masking', {}).get('llm') or {}) !=
                config_without_key(_redact_config(few.llm))):
        raise ValueError('Native few-shot index configuration mismatch; use a new index path')
    if workload.get('few_shot_source'):
        source = Path(workload['few_shot_source']).resolve()
        if not manifest.get('root_path') or Path(manifest['root_path']).resolve() != source:
            raise ValueError('Native few-shot index training root mismatch; use a new index path')
        # The original manifest has no content hash for its training inputs.
        # Compare existing native records, without remasking or embedding.
        from app.few_shot.train_loader import load_training_examples
        expected = load_training_examples(training_type, source, few.max_samples, few.max_samples_per_db)
        examples = Path(few.save_path) / 'examples.jsonl'
        actual = [json.loads(line) for line in examples.read_text(encoding='utf-8').splitlines() if line.strip()]
        fields = ('example_id', 'dataset', 'db_id', 'question', 'sql', 'evidence', 'metadata')
        if len(expected) != len(actual) or any(
                any(row.get(field) != getattr(example, field) for field in fields)
                for example, row in zip(expected, actual)):
            raise ValueError('Native few-shot index training contents changed; use a new index path')


def _training_identity(workload, few, run):
    from app.few_shot.train_loader import load_training_examples
    from app.few_shot.index_builder import _redact_config
    from .precompute_cache import fingerprint
    training_type = workload.get('few_shot_dataset', 'bird' if workload['benchmark'] == 'bird_interact' else workload['benchmark'])
    examples = load_training_examples(training_type, workload['few_shot_source'], few.max_samples, few.max_samples_per_db)
    return {'format': 'native-training-inputs-v2', 'dataset': training_type,
            'examples': fingerprint([vars(example) for example in examples]),
            'embedding': {k: v for k, v in _redact_config(few.embedding).items() if k != 'api_key'},
            'mask_llm': {k: v for k, v in _redact_config(few.llm).items() if k != 'api_key'},
            'timeout': run.llm_timeout}


def _prepare_examples(workload, items, config, *, embedding_service=None, runtime=None,
                      audit=None, checkpoint_root=None, workers=None):
    from app.few_shot.index_builder import build_few_shot_index
    from app.few_shot.retriever import FewShotRetriever
    from app.few_shot.runtime import TargetMaskCache, prepare_few_shot_examples_for_item, _get_or_create_target_mask
    from app.few_shot.masker import TargetMaskResult
    from app.llm import LLM
    from .preparation_store import PreparationStore, preparation_lock
    from .precompute_cache import fingerprint
    from .workloads import external_id, dump_item
    few = config.few_shot_index_config
    llm = LLM(few.llm)
    llm.sample_max_attempts = 4
    generator = preliminary_call = None
    try:
        if runtime is not None:
            audit.bind_llm(llm, runtime, default_label={'step': 'training_mask',
                'training_dataset': workload.get('few_shot_dataset', workload['benchmark']), 'index': str(few.save_path)})
        with preparation_lock(Path(few.save_path).with_suffix('.build.lock')):
            if not (Path(few.save_path) / 'manifest.json').exists():
                training_type = workload.get('few_shot_dataset', 'bird' if workload['benchmark'] == 'bird_interact' else workload['benchmark'])
                if training_type not in ('bird', 'spider'):
                    raise ValueError('Specify few_shot_dataset bird or spider for native index construction')
                if embedding_service is not None:
                    _bind_resource_identity(few.save_path, _training_identity(workload, few, config.run_config))
                build_few_shot_index(dataset_type=training_type, root_path=workload['few_shot_source'],
                    save_path=few.save_path, embedding_config=few.embedding, llm=llm,
                    mask_cache_path=few.mask_cache_path, embedding_batch_size=config.run_config.embedding_batch_size,
                    parallelism=config.run_config.parallelism, llm_timeout=config.run_config.llm_timeout,
                    progress_log_interval=config.run_config.progress_log_interval,
                    max_samples=few.max_samples, max_samples_per_db=few.max_samples_per_db,
                    **({'embedding_function': embedding_service, 'embedding_map': embedding_service.map}
                       if embedding_service is not None else {}))
        retriever = FewShotRetriever.from_index_path(index_path=few.save_path, embedding_config=few.embedding,
            embedding_batch_size=config.run_config.embedding_batch_size, similarity_device=few.similarity_device,
            **({'embedding_function': embedding_service} if embedding_service is not None else {}))
        index_digest = fingerprint({name: file_sha256(Path(few.save_path)/name) for name in
                                   ('manifest.json', 'examples.jsonl', 'question_embeddings.npy', 'sql_embeddings.npy')})
        cache = TargetMaskCache(few.target_mask_cache_path or str(Path(few.save_path) / 'target_mask_cache.jsonl'))
        if few.preliminary_sql.enabled:
            from app.few_shot.preliminary_sql import PreliminarySQLGenerator
            generator = PreliminarySQLGenerator(few.preliminary_sql, config.dataset_config,
                extractor_max_retry=config.llm_extractor_config.max_retry, parallelism=config.run_config.parallelism)
            if runtime is not None:
                generator._executor.shutdown(wait=True)
                generator._executor = runtime.executor_view()
                audit.bind_llm(generator._llm, runtime)
            from .preliminary_diagnostics import attach_preliminary_diagnostics
            preliminary_call = attach_preliminary_diagnostics(generator, audit)
        def prepare_item(item):
            if item.few_shot_examples:
                return
            identity = {'input': fingerprint(dump_item(item)), 'index': index_digest,
                        'policy': {'examples': few.num_examples, 'weights': [few.question_weight, few.sql_weight],
                                   'preliminary': few.preliminary_sql.model_dump(exclude={'llm'}),
                                   'preliminary_llm': _public_llm(few.preliminary_sql.llm),
                                   'mask_llm': _public_llm(few.llm), 'timeout': config.run_config.llm_timeout}}
            store = PreparationStore(Path(checkpoint_root)/fingerprint(str(external_id(item))), identity) if checkpoint_root else None
            def step(name, function):
                with audit.label({'benchmark': workload['benchmark'], 'split': workload.get('split'),
                                  'item': external_id(item), 'step': name}) if audit else ExitStack():
                    return store.step(name, function) if store else function()
            preliminary = step('preliminary', lambda: preliminary_call(item) if preliminary_call else {'sql': None})
            sql = preliminary['sql']
            mask = step('mask', lambda: asdict(_get_or_create_target_mask(data_item=item,
                preliminary_sql=sql, llm=llm, cache=cache, skip_mask_llm=False,
                llm_timeout=config.run_config.llm_timeout)))
            # Reuse the native example formatting/ranking; only the already saved
            # target mask is injected so retrieval failure cannot repeat masking.
            saved_mask = SimpleNamespace(get=lambda key: TargetMaskResult(**mask))
            def retrieve():
                prepared = prepare_few_shot_examples_for_item(data_item=item, retriever=retriever,
                    llm=llm, top_k=few.num_examples, question_weight=few.question_weight,
                    sql_weight=few.sql_weight, preliminary_sql=sql, cache=saved_mask,
                    llm_timeout=config.run_config.llm_timeout)
                return {'examples': prepared.examples}
            examples = step('retrieval', retrieve)['examples']
            if workload['benchmark'] == 'bird_interact':
                examples = step('postgres_examples', lambda: {'examples': _postgres_examples(examples)})['examples']
            item.few_shot_examples = examples
            item.few_shot_preliminary_sql = sql
            item.few_shot_preparation_metadata = {'mode': 'native_dynamic',
                'index_content_sha256': index_digest, 'target_mask_source': mask['source'],
                'used_sql_similarity': mask['masked_sql'] is not None,
                'retrieved_example_count': len(examples), 'num_examples': few.num_examples,
                'weights': [few.question_weight, few.sql_weight],
                'preliminary_sql': preliminary, 'preparation_identity': fingerprint(identity),
                'request_audit': str(audit.path) if audit else None}
        count = min(workers or config.run_config.parallelism, max(1, len(items)))
        with ExitStack() as stack:
            pool = runtime.workflow_executor(count) if runtime else stack.enter_context(ThreadPoolExecutor(count))
            if runtime:
                stack.callback(pool.shutdown, wait=True)
            futures = [pool.submit(prepare_item, item) for item in items]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise RuntimeError(f'{len(errors)} question preparations failed; successful steps are saved; '
                                   f'first error: {type(errors[0]).__name__}') from errors[0]
    finally:
        if generator is not None:
            generator.close()
        client = getattr(llm, '_client', None)
        if client is not None:
            client.close()


def _public_llm(config):
    return {key: value for key, value in config.model_dump(mode='json').items() if key != 'api_key'} if config else None


def _postgres_examples(examples):
    from copy import deepcopy
    import sqlglot
    converted = []
    for example in examples:
        row = deepcopy(example)
        queries = sqlglot.transpile(row['sql'], read='sqlite', write='postgres', unsupported_level=sqlglot.ErrorLevel.RAISE)
        if len(queries) != 1:
            raise ValueError('Selected training example cannot be presented as one PostgreSQL statement')
        row['original_sql'] = row['sql']
        row['sql'] = queries[0]
        row['dialect_conversion'] = 'sqlglot sqlite -> postgres'
        converted.append(row)
    return converted


def prepare_many(paths, output, output_dir, env_file, *, workers=200, embedding_config=None):
    """One explicit command/lifecycle; each workload remains independently runnable."""
    from scripts import deepeye_run as entry
    from .precompute_cache import VectorCache
    from .embedding_service import EmbeddingLimits, EmbeddingService, embedding_namespace
    from .workloads import question_rows
    workloads = [load_workload(path) for path in paths]
    if workers < 1 or (output is None) == (output_dir is None) or (output is not None and len(workloads) != 1):
        raise ValueError('Use --output for one workload or --output-dir for one or more; workers must be positive')
    partitions = [w['partition'] for w in workloads]
    if len(partitions) != len(set(partitions)):
        raise ValueError('Duplicate benchmark/split in preparation command')
    outputs, environments = [], []
    for workload in workloads:
        question_rows(workload)
        target = Path(output) if output is not None else Path(output_dir)/workload['benchmark']/(workload['split']+'.snapshot')
        outputs.append(target.resolve())
        environments.append(entry.read_environment(env_file, args=SimpleNamespace(workload=workload)))
    limits = EmbeddingLimits(**json.loads(Path(embedding_config).read_text())) if embedding_config else EmbeddingLimits()
    if limits.dimension != 1024:
        raise ValueError('Shared native preparation currently requires dimension 1024; limits overrides cannot change vector semantics')
    results = []
    with ExitStack() as stack:
        service = None
        relevant = [i for i, w in enumerate(workloads) if w['benchmark'] != 'spider2' and not outputs[i].exists()]
        if relevant:
            environment = environments[relevant[0]]
            identity = embedding_namespace(environment, dimension=limits.dimension)
            if any(embedding_namespace(environments[i], dimension=limits.dimension) != identity for i in relevant):
                raise ValueError('Shared preparation requires one Embedding profile')
            roots = {str(Path(workloads[i].get('preparation_root', outputs[i].parent/'shared')).resolve()) for i in relevant}
            if len(roots) != 1:
                raise ValueError('Multiple workloads must specify the same preparation_root to share resources')
            shared = Path(next(iter(roots)))
            cache = stack.enter_context(VectorCache(shared/'vectors.sqlite', identity))
            service = stack.enter_context(EmbeddingService(environment, cache, shared/'embedding_calls.jsonl', limits=limits))
        for workload, target, environment in zip(workloads, outputs, environments):
            results.append(prepare_native(workload, target, environment, workers=workers, embedding_service=service))
    return {'workloads': results}
