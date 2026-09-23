"""Native-compatible precomputation records, without RC, gold SQL or database writes."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from .precompute_cache import atomic_json, fingerprint, validate_vectors


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path, record):
    record = dict(record)
    record.pop('content_hash', None)
    record['content_hash'] = fingerprint(record)
    atomic_json(path, record)
    return record


def read_record(path):
    record = json.loads(Path(path).read_text())
    stored = record.pop('content_hash', None)
    if stored != fingerprint(record):
        raise ValueError(f'Artifact checksum mismatch: {Path(path).name}')
    return {**record, 'content_hash': stored}


def question_row(item):
    return {'index': item.instance_id, 'db_id': item.database_id,
            'question': item.question, 'evidence': item.evidence}


def item_directory(root, variant, instance_id):
    if variant not in ('lite', 'full') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', instance_id):
        raise ValueError('Unsafe dataset or instance identifier')
    return Path(root) / 'questions' / variant / instance_id


def prepare_keywords(row, chat_config, path, chat):
    from app.prompt import PromptFactory
    from app.pipeline.value_retrieval.utils import _parse_keywords_response, _post_process_keywords
    prompt = PromptFactory.format_keywords_extraction_prompt(row['question'], row['evidence'])
    messages = [{'role': 'user', 'content': prompt}]
    identity = fingerprint({'row': row, 'config': chat_config, 'messages': messages})
    path = Path(path)
    if path.exists():
        record = read_record(path)
        if record['identity'] != identity or record['status'] != 'complete':
            raise ValueError('Keyword cache does not match current question/model/prompt')
        validate_token_usage(record.get('usage'))
        return record
    responses = []
    calls = []
    started = time.monotonic()
    for _ in range(3):
        result = chat(messages)
        if not isinstance(result, dict) or not isinstance(result.get('content'), str):
            raise ValueError('Keyword service must return content with independent call usage')
        validate_token_usage(result.get('usage'))
        response = result['content']
        calls.append({key: value for key, value in result.items() if key != 'content'})
        responses.append(response)
        parsed = _parse_keywords_response(response)
        if parsed:
            # Sorting changes only set iteration order; the native term set is preserved.
            keywords = sorted(word for word in _post_process_keywords(parsed) if word)
            if not keywords:
                continue
            usage = {key: sum(call['usage'][key] for call in calls)
                     for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            for key in ('attempts', 'elapsed_seconds', 'usage_missing'):
                usage[key] = sum(call['usage'].get(key, 0) for call in calls)
            usage['usage_complete'] = usage['usage_missing'] == 0
            return write_record(path, {'status': 'complete', 'identity': identity,
                'row': row, 'chat_config': chat_config, 'messages': messages,
                'input_hash': fingerprint(messages), 'responses': responses,
                'keywords': keywords, 'postprocess': 'sorted-native-nonempty-v1',
                'calls': calls, 'usage': usage, 'cost_semantics': 'reported_tokens_only',
                'precompute_seconds': time.monotonic() - started,
                'fallback_used': False, 'created_at': utc_now()})
    write_record(path.with_suffix('.failed.json'), {'status': 'failed', 'identity': identity,
        'row': row, 'responses': responses, 'error': 'No valid keyword response', 'created_at': utc_now()})
    raise ValueError('Keyword extraction returned no valid list; no fallback accepted')


def validate_token_usage(usage):
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        raise ValueError('Native token metrics require reported nonnegative integer counts')
    return usage


def schema_with_values(schema, retrieved_values, max_values_per_column=5):
    from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner
    runner = object.__new__(ValueRetrievalRunner)
    runner._stage_config = SimpleNamespace(max_values_per_column=max_values_per_column)
    item = SimpleNamespace(database_schema=schema, retrieved_values=retrieved_values)
    runner._update_database_schema(item, schema)
    return item.database_schema_after_value_retrieval


def retrieve_item(item, vectors, index, max_values_per_column=5):
    retrieved = {}
    for table, table_info in item.database_schema['tables'].items():
        for column, info in table_info['columns'].items():
            kind = info['column_type'].upper()
            # Keep the native ValueRetrievalRunner's exact column inclusion rule.
            if kind == 'TEXT' or kind.startswith('VARCHAR') or kind.startswith('CHAR'):
                result = index.retrieve_values_for_column(vectors.tolist(), table, column,
                    max_values_per_column=max_values_per_column, lower_meta_data=True)
                retrieved.setdefault(table, {})[column] = result['values']
    return retrieved


def verify_native_index(directory):
    directory = Path(directory)
    record = read_record(directory / 'index_record.json')
    for relative, checksum in record['files'].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or file_hash(path) != checksum:
            raise ValueError('Database index file checksum mismatch')
    return record


def native_index_identity(sample_manifest, embedding_namespace):
    """Bind a native index to the complete frozen samples and vector namespace."""
    return fingerprint({'sample_manifest': sample_manifest,
        'embedding_namespace': embedding_namespace, 'format': 'native-local-index-v1'})


def build_native_index(directory, manifest, cache):
    from app.vector_db.local_index import write_local_index_column, write_local_index_manifest
    directory = Path(directory)
    identity = native_index_identity(manifest, cache.namespace)
    if (directory / 'index_record.json').exists():
        record = verify_native_index(directory)
        if record['identity'] != identity:
            raise ValueError('Index was built from different samples or embedding model')
        return record
    index_dir = directory / 'local_index'
    entries = []
    for column in manifest['columns']:
        if column['status'] not in ('collected', 'empty', 'filtered_uuid', 'filtered_numeric'):
            raise ValueError('Cannot build database index from unfinished sampling')
        docs = column['documents']
        if not docs:
            continue
        vectors = cache.read(docs)
        entry = write_local_index_column(index_dir, column['table_name'].lower(), column['column_name'].lower(), docs, vectors)
        if entry is None:
            raise ValueError('Native writer unexpectedly omitted populated column')
        entries.append(entry)
    write_local_index_manifest(index_dir, entries)
    files = {str(path.relative_to(directory)): file_hash(path) for path in sorted(index_dir.rglob('*')) if path.is_file()}
    return write_record(directory / 'index_record.json', {'identity': identity, 'schema_hash': manifest['schema_hash'],
        'embedding_namespace': cache.namespace, 'dimension': cache.dimension,
        'columns': len(entries), 'documents': sum(len(col['documents']) for col in manifest['columns']),
        'files': files, 'created_at': utc_now()})


class PrecomputedInputReader:
    """Instance-scoped snapshot reader for shared schemas and question records."""

    def __init__(self, root):
        self.root = Path(root)
        self._record_cache = {}
        self._schema_cache = {}

    def _records(self, variant, instance_id):
        key = (variant, instance_id)
        if key not in self._record_cache:
            directory = item_directory(self.root, variant, instance_id)
            self._record_cache[key] = (
                read_record(directory / 'keywords.json'),
                read_record(directory / 'retrieval.json'),
            )
        return self._record_cache[key]

    def records(self, variant, instance_id):
        """Return independent mutable views of cached keyword and retrieval records."""
        keywords, retrieval = self._records(variant, instance_id)
        return deepcopy(keywords), deepcopy(retrieval)

    def _schema(self, database_id):
        if database_id not in self._schema_cache:
            schema_path = self.root / 'databases' / database_id / 'schema.json'
            schema = json.loads(schema_path.read_text())
            self._schema_cache[database_id] = (schema, fingerprint(schema))
        return self._schema_cache[database_id]

    def load(self, variant, instance_id, *, expected_item=None):
        """Offline load: common schema + per-question delta -> native DataItem."""
        from .dataset import BirdInteractDataItem

        keywords, record = self._records(variant, instance_id)
        row = record['row']
        validate_token_usage(record.get('value_retrieval_llm_cost'))
        if row['index'] != instance_id or record['variant'] != variant or keywords['row'] != row:
            raise ValueError('Question artifact identity mismatch')
        frozen_schema, frozen_schema_hash = self._schema(row['db_id'])
        if frozen_schema_hash != record['schema_hash']:
            raise ValueError('Common Meta schema changed after precomputation')
        if keywords['content_hash'] != record['keywords_hash']:
            raise ValueError('Keywords changed after retrieval')
        directory = item_directory(self.root, variant, instance_id)
        vector_path = directory / 'keywords.npy'
        if file_hash(vector_path) != record['vectors_hash']:
            raise ValueError('Keyword vector file checksum mismatch')
        validate_vectors(np.load(vector_path, allow_pickle=False), len(keywords['keywords']), record['dimension'])
        if expected_item is not None:
            if (question_row(expected_item) != row
                    or fingerprint(expected_item.database_schema) != record['schema_hash']):
                raise ValueError('Current dataset no longer matches the frozen question/Meta')

        schema = deepcopy(frozen_schema)
        item = BirdInteractDataItem(question_id=record['question_id'], instance_id=instance_id,
            question=row['question'], evidence=row['evidence'], database_id=row['db_id'],
            database_path=row['db_id'], database_schema=schema, gold_sql='', db_type='postgresql')
        item.question_keywords = deepcopy(keywords['keywords'])
        item.retrieved_values = deepcopy(record['retrieved_values'])
        item.database_schema_after_value_retrieval = schema_with_values(
            schema, item.retrieved_values, record['max_values_per_column'])
        if fingerprint(item.database_schema_after_value_retrieval) != record['retrieved_schema_hash']:
            raise ValueError('Reconstructed schema does not match recorded native result')
        item.value_retrieval_time = record['value_retrieval_time']
        item.value_retrieval_llm_cost = deepcopy(record['value_retrieval_llm_cost'])
        item.total_time = record['value_retrieval_time']
        item.total_llm_cost = dict(record['value_retrieval_llm_cost'])
        return item


def load_precomputed_item(root, variant, instance_id, *, expected_item=None):
    """Load one item through a fresh reader so source changes stay observable."""
    return PrecomputedInputReader(root).load(
        variant, instance_id, expected_item=expected_item)
