"""Finite workload bindings, preserving native item types and original identities.

This boundary reads public questions/Meta and prepared native snapshots. Answers
remain in their source files; only their locators are attached to run bindings.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote

SELECTIONS = {('bird', 'dev'), ('spider', 'dev'), ('spider', 'test'),
              ('spider2', 'lite'), ('bird_interact', 'lite'), ('bird_interact', 'full')}
PATH_FIELDS = ('questions', 'meta', 'resource_root', 'rc', 'precompute_dir',
               'few_shot_source', 'prepared_dataset', 'native_config', 'reference_questions',
               'bigquery_credential_path')


def _types():
    from app.dataset.dataset import DataItem
    from app.dataset.spider2_dataset import Spider2DataItem
    from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataItem
    return {'native': DataItem, 'spider2': Spider2DataItem, 'bird_interact': BirdInteractDataItem}


def external_id(item) -> int | str:
    types = _types()
    if type(item) in (types['spider2'], types['bird_interact']):
        return item.instance_id
    if type(item) is types['native']:
        return item.question_id
    raise ValueError(f'Unsupported native item type: {type(item).__name__}')


def task_key(partition: str, item) -> str:
    identity = external_id(item)
    if partition in ('lite', 'full') and type(item) is _types()['bird_interact']:
        return f'{partition}/{identity}'
    if not partition or any(part in ('', '.', '..') for part in partition.split('/')):
        raise ValueError('Invalid workload partition')
    suffix = f'i:{identity}' if type(identity) is int else f's:{quote(identity, safe="")}'
    return f'{partition}/{suffix}'


def dump_item(item) -> dict:
    tag = next((name for name, cls in _types().items() if type(item) is cls), None)
    if tag is None:
        raise ValueError(f'Unsupported native item type: {type(item).__name__}')
    data = item.model_dump(mode='json')
    data['gold_sql'] = ''
    return {'type': tag, 'data': data}


def restore_item(snapshot: dict):
    types = _types()
    if not isinstance(snapshot, dict) or snapshot.get('type') not in types:
        raise ValueError('Unsupported native item snapshot type')
    data = deepcopy(snapshot['data'])
    data['gold_sql'] = ''
    cls = types[snapshot['type']]
    if set(data).difference(cls.model_fields):
        raise ValueError('Snapshot contains fields outside the native item type')
    return cls(**data)


def load_workload(path) -> dict:
    if isinstance(path, dict):
        result = deepcopy(path)
        base = Path(result.get('_path', '.')).resolve().parent
    else:
        path = Path(path).resolve()
        result = json.loads(path.read_text(encoding='utf-8-sig'))
        result['_path'] = str(path)
        base = path.parent
    if (result.get('benchmark'), result.get('split')) not in SELECTIONS:
        raise ValueError('Unsupported workload benchmark/split')
    for field in ('questions', 'meta', 'resource_root', 'rc'):
        if not result.get(field):
            raise ValueError(f'Workload requires {field}')
    for field in PATH_FIELDS:
        if result.get(field):
            value = Path(result[field])
            result[field] = str((base / value).resolve())
    result['partition'] = f"{result['benchmark']}/{result['split']}"
    return result


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_rows(workload):
    path = Path(workload['questions'])
    payload = path.read_text(encoding='utf-8-sig')
    rows = ([json.loads(line) for line in payload.splitlines() if line.strip()]
            if path.suffix == '.jsonl' else json.loads(payload))
    if not isinstance(rows, list):
        raise ValueError('Workload questions must be an array or JSONL records')
    result, seen = [], set()
    special = workload['benchmark'] in ('spider2', 'bird_interact')
    for position, row in enumerate(rows):
        identity = row.get('index', row.get('instance_id' if special else 'question_id', position))
        expected_type = str if special else int
        if type(identity) is not expected_type or (special and not identity):
            raise ValueError(f'Question identity must be {expected_type.__name__}: {identity!r}')
        if identity in seen:
            raise ValueError(f'Duplicate original question identity: {identity!r}')
        seen.add(identity)
        db_id = row.get('db_id', row.get('db'))
        if not isinstance(db_id, str) or not db_id or '/' in db_id or '\\' in db_id or db_id in ('.', '..'):
            raise ValueError('Question requires a safe database id')
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError('Question requires nonempty question text')
        result.append({'external_id': identity, 'question_id': position if special else identity,
                       'database_id': db_id, 'question': row['question'],
                       'evidence': row.get('evidence') or '', 'source_row': position,
                       'external_knowledge_path': row.get('external_knowledge')})
    return result


def _meta_tables(workload, db_id):
    root = Path(workload['meta'])
    directory = root / db_id
    if not directory.is_dir():
        directory = root / db_id.casefold()
    if not directory.is_dir():
        raise FileNotFoundError(f'Meta directory missing for {db_id}: {directory}')
    tables, hashes = {}, {}
    for path in sorted(directory.glob('*.csv')):
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            # Original BIRD description files also include Windows encodings;
            # use the same detector as DeepEye's native description reader.
            import chardet
            encoding = chardet.detect(raw)['encoding']
            if not encoding:
                raise ValueError(f'Meta CSV encoding could not be identified: {path}')
            text = raw.decode(encoding)
        rows = list(csv.DictReader(io.StringIO(text, newline='')))
        if not rows:
            raise ValueError(f'Meta table has no columns: {path}')
        bird = workload['benchmark'] in ('bird', 'bird_interact')
        required = {'original_column_name', 'data_format'} if bird else {'column_name', 'column_type'}
        if not required.issubset(rows[0]):
            raise ValueError(f'Meta table has incompatible columns: {path}')
        columns = {}
        for row in rows:
            name = row['original_column_name' if bird else 'column_name'].strip()
            if not name or name in columns:
                raise ValueError(f'Meta table has empty or duplicate column: {path}')
            columns[name] = row
        tables[path.stem] = columns
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    if not tables:
        raise FileNotFoundError(f'Meta CSV tables missing for {db_id}')
    return tables, hashes


def _scope_schema(schema, meta, *, validate=False):
    result = deepcopy(schema)
    tables = result.get('tables', {})
    lookup = {name.casefold(): name for name in tables}
    allowed = {name.casefold() for name in meta}
    if validate and set(lookup) != allowed:
        raise ValueError('Prepared schema tables differ from the public Meta scope')
    selected = {}
    for table, columns in meta.items():
        native_name = lookup.get(table.casefold())
        if native_name is None:
            raise ValueError(f'Meta table absent from native schema: {table}')
        native = deepcopy(tables[native_name])
        col_lookup = {name.casefold(): name for name in native['columns']}
        allowed_columns = {name.casefold() for name in columns}
        if validate and set(col_lookup) != allowed_columns:
            raise ValueError(f'Prepared schema columns differ from Meta: {table}')
        selected_columns = {}
        for name in columns:
            if name.casefold() not in col_lookup:
                raise ValueError(f'Meta column absent from native schema: {table}.{name}')
            native_col = col_lookup[name.casefold()]
            selected_columns[native_col] = native['columns'][native_col]
            # Meta sets visibility here, not a replacement description format.
            # Retain the native loader's expanded/value descriptions and flags.
        native['columns'] = selected_columns
        selected[native_name] = native
    result['tables'] = selected
    # References to columns outside the visible inventory must not expose hidden schema.
    visible = {(t.casefold(), c.casefold()) for t, v in selected.items() for c in v['columns']}
    for table in selected.values():
        for column in table['columns'].values():
            column['foreign_keys'] = [pair for pair in column.get('foreign_keys', [])
                                      if tuple(str(v).casefold() for v in pair) in visible]
    return result


def _prepared_items(path, benchmark):
    """Read native structured snapshots without the native arbitrary class importer."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    classes = {'bird': ('app.dataset.dataset', 'BirdDataset', 'app.dataset.dataset', 'DataItem'),
               'spider': ('app.dataset.dataset', 'SpiderDataset', 'app.dataset.dataset', 'DataItem'),
               'spider2': ('app.dataset.spider2_dataset', 'Spider2LiteDataset', 'app.dataset.spider2_dataset', 'Spider2DataItem'),
               'bird_interact': ('scripts.baseline_adapters.deepeye.dataset', 'BirdInteractDataset',
                                 'scripts.baseline_adapters.deepeye.dataset', 'BirdInteractDataItem')}
    actual = tuple(manifest.get(key) for key in ('dataset_class_module', 'dataset_class_name', 'item_class_module', 'item_class_name'))
    if actual != classes[benchmark]:
        raise ValueError('Prepared dataset class/type does not match workload')
    if manifest.get('format') != 'structured_dataset_snapshot' or manifest.get('version') != 1:
        raise ValueError('Unsupported prepared native snapshot format')
    root = path.parent / manifest['snapshot_root']
    if root.resolve().parent != path.parent.resolve():
        raise ValueError('Prepared native snapshot root must be adjacent to manifest')
    items_path = root / 'items.jsonl'
    cls = _types()[benchmark if benchmark in ('spider2', 'bird_interact') else 'native']
    items = {}
    for line in items_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        data = dict(record['input'], gold_sql='')
        # Native DataItemInput contains nullable specialized fields for all classes.
        data = {key: value for key, value in data.items() if key in cls.model_fields}
        item = cls(**data)
        artifacts = record.get('pipeline_artifacts', {})
        # Only value retrieval is a prepared input; downstream results are not imported.
        if artifacts.get('value_retrieval'):
            item.apply_stage_artifact('value_retrieval', artifacts['value_retrieval'])
            # Later native stages are deliberately not restored. Aggregate
            # metrics must describe this retained preparation prefix only.
            item.total_time = item.value_retrieval_time
            item.total_llm_cost = deepcopy(item.value_retrieval_llm_cost)
        identity = external_id(item)
        if identity in items:
            raise ValueError('Duplicate identity in prepared native dataset')
        items[identity] = item
    if manifest.get('num_items') != len(items):
        raise ValueError('Prepared native snapshot item count mismatch')
    return items, {'manifest_sha256': file_sha256(path), 'items_sha256': file_sha256(items_path)}


def _database_path(workload, db_id, db_type):
    if db_type != 'sqlite':
        return db_id
    root = Path(workload['resource_root'])
    benchmark, split = workload['benchmark'], workload['split']
    if benchmark == 'bird':
        relative = Path(split) / f'{split}_databases' / db_id / f'{db_id}.sqlite'
    elif benchmark == 'spider':
        relative = Path('database' if split == 'dev' else 'test_database') / db_id / f'{db_id}.sqlite'
    else:
        relative = Path('databases/spider2-localdb') / f'{db_id}.sqlite'
    candidates = (root / relative, root / db_id / f'{db_id}.sqlite', root / f'{db_id}.sqlite')
    result = next((p for p in candidates if p.is_file()), None)
    if result is None:
        raise FileNotFoundError(f'SQLite database resource missing: {candidates[0]}')
    return str(result.resolve())


def load_items(workload, *, require_prepared=True):
    """Prepare the complete typed inventory before applying any execution filter."""
    from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
    workload = load_workload(workload)
    rows = question_rows(workload)
    benchmark = workload['benchmark']
    precomputed = benchmark == 'bird_interact' and workload.get('precompute_dir') and not workload.get('prepared_dataset')
    if require_prepared and not workload.get('prepared_dataset') and not precomputed:
        raise ValueError('Prepared native inputs are missing; use deepeye_run.py prepare-native --workload PATH --output PATH')
    prepared, snapshot_hashes = (_prepared_items(workload['prepared_dataset'], benchmark)
                                 if workload.get('prepared_dataset') else ({}, {}))
    tables, meta_hashes, schemas = {}, {}, {}
    precompute_sources = {}
    if precomputed:
        from .precompute_pipeline import PrecomputedInputReader, read_record
        from scripts.deepeye_bird_interact_smoke import IndependentExampleReader
        if not workload.get('few_shot_source'):
            raise ValueError('BIRD-Interact precomputation requires few_shot_source')
        reader = PrecomputedInputReader(workload['precompute_dir'])
        examples = IndependentExampleReader(Path(workload['few_shot_source']))
        root = Path(workload['precompute_dir'])
        inputs_record = read_record(root / 'inputs.json')
        config_record = read_record(root / 'run_config.json')
        precompute_sources = {'precompute_inputs_content_hash': inputs_record['content_hash'],
            'precompute_config_content_hash': config_record['content_hash'],
            'precompute_semantic_config': config_record['config'],
            'few_shot_source_sha256': examples.source_sha256}
    tasks, bindings = [], []
    for row in rows:
        identity, db_id = row['external_id'], row['database_id']
        if db_id not in tables:
            tables[db_id], hashes = _meta_tables(workload, db_id)
            meta_hashes.update(hashes)
        if benchmark == 'bird_interact':
            db_type = 'postgresql'
        elif benchmark == 'spider2':
            from app.dataset.spider2_dataset import get_db_type_from_instance_id
            db_type = get_db_type_from_instance_id(identity)
            if db_type not in ('sqlite', 'bigquery'):
                raise ValueError('This Spider2 workload supports only SQLite and BigQuery')
        else:
            db_type = 'sqlite'
        db_path = _database_path(workload, db_id, db_type)
        if prepared:
            if identity not in prepared:
                raise ValueError(f'Prepared dataset lacks original identity: {identity!r}')
            item = prepared[identity]
            if (item.question != row['question'] or item.evidence != row['evidence'] or
                    item.database_id != db_id or (hasattr(item, 'db_type') and item.db_type != db_type)):
                raise ValueError(f'Prepared input identity mismatch: {identity!r}')
            if ((db_type == 'sqlite' and Path(item.database_path).resolve() != Path(db_path).resolve()) or
                    (db_type != 'sqlite' and item.database_path != db_path)):
                raise ValueError(f'Prepared database resource does not match workload: {identity!r}')
            item.database_schema = _scope_schema(item.database_schema, tables[db_id], validate=True)
            if item.database_schema_after_value_retrieval is not None:
                item.database_schema_after_value_retrieval = _scope_schema(
                    item.database_schema_after_value_retrieval, tables[db_id], validate=True)
        else:
            cache_key = (db_id, db_type)
            if cache_key not in schemas:
                if db_type == 'sqlite':
                    from app.db_utils.schema import load_database_schema_dict
                    schema = load_database_schema_dict(db_path)
                elif db_type == 'bigquery':
                    from app.db_utils.cloud_schema import load_cloud_database_schema_dict
                    schema = load_cloud_database_schema_dict(db_id, db_type, workload['resource_root'], max_value_example_length=50)
                else:
                    from scripts.baseline_adapters.deepeye.dataset import _load_meta_table
                    schema = {'db_id': db_id, 'db_path': db_id, 'db_type': db_type,
                              'tables': {name: _load_meta_table(Path(name + '.csv'), rows=list(columns.values()))
                                         for name, columns in tables[db_id].items()}}
                schemas[cache_key] = _scope_schema(schema, tables[db_id], validate=db_type == 'postgresql')
            cls = _types()[benchmark if benchmark in ('spider2', 'bird_interact') else 'native']
            fields = {key: row[key] for key in ('question_id', 'question', 'evidence', 'database_id')}
            if benchmark in ('spider2', 'bird_interact'):
                fields.update(instance_id=identity, db_type=db_type)
            if benchmark == 'spider2':
                fields['external_knowledge_path'] = row['external_knowledge_path']
            item = cls(**fields, database_path=db_path, database_schema=deepcopy(schemas[cache_key]), gold_sql='')
        if precomputed:
            item = reader.load(workload['split'], identity, expected_item=item)
            item.few_shot_examples, provenance = examples.select(item, count=3)
            item.few_shot_preparation_metadata = {'mode': 'static_independent_bird_train',
                                                 'num_examples': 3, 'provenance': provenance}
        if benchmark == 'spider2' and item.database_schema_after_value_retrieval is None:
            item.question_keywords, item.retrieved_values = [], {}
            item.database_schema_after_value_retrieval = deepcopy(item.database_schema)
            item.value_retrieval_time = 0.0
            item.value_retrieval_llm_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            item.total_time = 0.0
            item.total_llm_cost = dict(item.value_retrieval_llm_cost)
        if require_prepared and (not item.is_stage_complete('value_retrieval') or
                                 (benchmark != 'spider2' and not item.few_shot_examples)):
            raise ValueError(f'Incomplete native preparation for {identity!r}; use prepare-native')
        partition = workload['partition']
        binding = {'task_key': task_key(partition, item), 'partition': partition,
                   'benchmark': benchmark, 'split': workload['split'], 'external_id': identity,
                   'database_id': db_id, 'database_path': db_path, 'db_type': db_type,
                   'question_sha256': fingerprint({'question': item.question, 'evidence': item.evidence}),
                   'schema_sha256': fingerprint(item.database_schema),
                   'input_sha256': fingerprint(dump_item(item)),
                   'reference': {'path': workload.get('reference_questions', workload['questions']),
                                 'source_row': row['source_row'], 'external_id': identity}}
        tasks.append((partition, item)); bindings.append(binding)
    sources = {'workload': {key: value for key, value in workload.items() if key != 'bigquery_credential_path'},
               'questions_sha256': file_sha256(workload['questions']), 'meta_sha256': fingerprint(meta_hashes),
               'prepared_dataset': snapshot_hashes, 'locators': {key: workload[key] for key in PATH_FIELDS if key in workload and key != 'bigquery_credential_path'}}
    sources.update(precompute_sources)
    return tasks, bindings, sources
