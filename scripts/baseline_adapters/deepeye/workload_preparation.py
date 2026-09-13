"""Explicit preparation using DeepEye's existing vector and few-shot implementations."""
from __future__ import annotations

import json
from pathlib import Path

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


def prepare_native(workload, output, environment, *, workers=2):
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
    if output.exists():
        raise FileExistsError(f'Preparation output already exists: {output}')
    args = entry._build_parser().parse_args(['prepare', '--workload', workload['_path'],
                                          '--run-dir', str(output.parent)])
    args.workload = workload
    args.coordinator_workers = workers
    config = entry.build_runtime_config(environment, args, output.parent)
    tasks, _, sources = load_items(workload, require_prepared=False)
    items = [item for _, item in tasks]
    needs_vr = any(item.database_schema_after_value_retrieval is None for item in items)
    # The native Spider2 template has no few-shot index; do not invent one.
    needs_examples = workload['benchmark'] != 'spider2' and any(not item.few_shot_examples for item in items)
    few = config.few_shot_index_config
    if needs_vr and workload['benchmark'] == 'bird_interact':
        raise ValueError('BIRD-Interact requires precompute_dir from deepeye_bird_interact_precompute.py')
    if needs_examples and workload['benchmark'] != 'bird_interact':
        if few.embedding is None or few.llm is None:
            raise ValueError('Native few-shot preparation requires native_config [few_shot_index.embedding] and [few_shot_index.llm]')
        if not (Path(few.save_path) / 'manifest.json').exists() and not workload.get('few_shot_source'):
            raise ValueError('Native few-shot index missing; supply few_shot_source training root to build it')
    if needs_examples and workload['benchmark'] == 'bird_interact' and not workload.get('few_shot_source'):
        raise ValueError('BIRD-Interact requires few_shot_source training JSON')
    dataset = _dataset(workload, items, config)
    preparation = {'format': 'deepeye-native-preparation-v1', 'sources': sources,
                   'effective_config': entry.build_effective_config(environment, args)}
    with entry.backend_context(environment, args):
        if needs_vr:
            # Native runners consume their existing structured snapshot format.
            input_path = output.with_name(output.name + '.input.snapshot')
            config.dataset_config.save_path = str(input_path)
            config.value_retrieval_config.save_path = str(output.with_name(output.name + '.vr.snapshot'))
            save_dataset(dataset, str(input_path))
            from runner.create_vector_db_parallel import run_vector_db_creation
            run_vector_db_creation(str(input_path), workload['benchmark'], config.vector_database_config,
                workers, config.run_config.embedding_batch_size, config.run_config.progress_log_interval)
            from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner
            runner = ValueRetrievalRunner.from_config(config)
            runner._llm.sample_max_attempts = 4
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
            if workload['benchmark'] == 'bird_interact':
                from scripts.deepeye_bird_interact_smoke import IndependentExampleReader
                reader = IndependentExampleReader(Path(workload['few_shot_source']))
                for item in items:
                    if not item.few_shot_examples:
                        item.few_shot_examples, provenance = reader.select(item, count=3)
                        item.few_shot_preparation_metadata = provenance
            else:
                _prepare_examples(workload, items, config)
        if any(not item.is_stage_complete('value_retrieval') or
               (workload['benchmark'] != 'spider2' and not item.few_shot_examples) for item in items):
            raise ValueError('Native preparation incomplete; no executable snapshot produced')
        for item in items:
            item.gold_sql = ''
        save_dataset(dataset, str(output))
    # Bind actual outputs and validate visibility/identity through the same run boundary.
    checked = dict(workload, prepared_dataset=str(output))
    load_items(checked)
    manifest = json.loads(output.read_text())
    manifest['preparation'] = preparation
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'prepared': len(items), 'prepared_dataset': str(output), 'sha256': file_sha256(output),
            'workload': workload['_path'], 'next': 'Set prepared_dataset in the workload to this native snapshot'}


def _prepare_examples(workload, items, config):
    from app.few_shot.index_builder import build_few_shot_index
    from app.few_shot.retriever import FewShotRetriever
    from app.few_shot.runtime import TargetMaskCache, prepare_few_shot_examples_for_item
    from app.llm import LLM
    few = config.few_shot_index_config
    llm = LLM(few.llm)
    llm.sample_max_attempts = 4
    generator = None
    try:
        if not (Path(few.save_path) / 'manifest.json').exists():
            training_type = workload.get('few_shot_dataset', workload['benchmark'])
            if training_type not in ('bird', 'spider'):
                raise ValueError('Specify few_shot_dataset bird or spider for native index construction')
            build_few_shot_index(dataset_type=training_type, root_path=workload['few_shot_source'],
                save_path=few.save_path, embedding_config=few.embedding, llm=llm,
                mask_cache_path=few.mask_cache_path, embedding_batch_size=config.run_config.embedding_batch_size,
                parallelism=config.run_config.parallelism, llm_timeout=config.run_config.llm_timeout,
                progress_log_interval=config.run_config.progress_log_interval,
                max_samples=few.max_samples, max_samples_per_db=few.max_samples_per_db)
        retriever = FewShotRetriever.from_index_path(index_path=few.save_path, embedding_config=few.embedding,
            embedding_batch_size=config.run_config.embedding_batch_size, similarity_device=few.similarity_device)
        cache = TargetMaskCache(few.target_mask_cache_path or str(Path(few.save_path) / 'target_mask_cache.jsonl'))
        if few.preliminary_sql.enabled:
            from app.few_shot.preliminary_sql import PreliminarySQLGenerator
            generator = PreliminarySQLGenerator(few.preliminary_sql, config.dataset_config,
                extractor_max_retry=config.llm_extractor_config.max_retry, parallelism=config.run_config.parallelism)
        for item in items:
            if item.few_shot_examples:
                continue
            preliminary = generator.generate(item) if generator is not None else None
            sql = preliminary.sql if preliminary is not None else None
            prepared = prepare_few_shot_examples_for_item(data_item=item, retriever=retriever,
                llm=llm, top_k=few.num_examples, question_weight=few.question_weight,
                sql_weight=few.sql_weight, preliminary_sql=sql, cache=cache,
                llm_timeout=config.run_config.llm_timeout)
            item.few_shot_examples = prepared.examples
            item.few_shot_preliminary_sql = sql
            item.few_shot_preparation_metadata = {'mode': 'native_dynamic',
                'index_manifest_sha256': file_sha256(Path(few.save_path) / 'manifest.json'),
                'target_mask_source': prepared.mask_source, 'used_sql_similarity': prepared.masked_sql is not None,
                'retrieved_example_count': len(prepared.examples),
                'preliminary_sql': {'source': 'generated' if generator else None, 'selected': sql is not None}}
    finally:
        if generator is not None:
            generator.close()
        client = getattr(llm, '_client', None)
        if client is not None:
            client.close()
