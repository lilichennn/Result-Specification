"""One-time local DAIL Linking comparison using previously generated Round-3 RS filters.

Run with python -m scripts.rc_evaluation.dail_sql.linking_handoff --help.
--smoke-per-group 2 checks the original linking and writes only a small subset.
Rerunning without that option resumes the remaining questions and exports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from scripts.baseline_adapters.dail_sql import native
from scripts.baseline_adapters.dail_sql.linking_comparison import crop_schema, normalized_linking
from scripts.baseline_adapters.dail_sql.preparation import load_prepared_group, _file_hash
from scripts.baseline_adapters.dail_sql.tokenizer import LocalCoreNLP
from scripts.rc_evaluation.din_sql_linking.reporting import _canonical_gold, _usage, _ratio

ROOT = Path(__file__).resolve().parents[3]
ZERO = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def key(row):
    identity = row.get('task_key', row)
    if not isinstance(identity, dict):
        identity = row
    return str(identity['group']), str(identity['question_id'])


def indexed(rows, label):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError(f'Duplicate {label} identity: {identity}')
        result[identity] = row
    return result


def catalog(schema):
    result = {table.strip(): [] for table in schema['table_names_original']}
    if len(result) != len(schema['table_names_original']):
        raise ValueError('Ambiguous trimmed table names')
    for table_id, column in schema['column_names_original']:
        if table_id >= 0:
            name = schema['table_names_original'][table_id].strip()
            if column.strip() in result[name]:
                raise ValueError('Ambiguous trimmed column names')
            result[name].append(column.strip())
    return result


def load_inputs(source, batch, filter_source):
    manifest = read(batch / 'manifest.json')
    preparation = Path(manifest['prepared_root'])
    for relative, expected in manifest['prepared_files'].items():
        if _file_hash(preparation / relative) != expected:
            raise ValueError(f'Frozen DAIL prepared file changed: {relative}')
    # Only the relevant native linking sources are consumed by this local job.
    for relative, expected in manifest['implementation'].items():
        if relative.endswith('native.py') or '/linking_utils/' in relative:
            if _file_hash(ROOT / relative) != expected:
                raise ValueError(f'Native DAIL implementation changed: {relative}')
    seal, index = read(filter_source / 'COMPLETE.json'), read(filter_source / 'index.json')
    if not seal['complete'] or seal['index_sha256'] != _file_hash(filter_source / 'index.json'):
        raise ValueError('Filter source seal mismatch')
    expected_files = {entry['path']: entry for entry in index['files']}
    for name in ('records/questions.jsonl', 'records/nodes.jsonl', 'evaluation/linking_details.jsonl',
                 'raw_records/linking_extension_prepared_inputs.json'):
        if _file_hash(filter_source / name) != expected_files[name]['sha256']:
            raise ValueError(f'Filter source changed: {name}')
    source_questions = indexed(lines(filter_source / 'records/questions.jsonl'), 'filter question')
    filters = indexed((row for row in lines(filter_source / 'records/nodes.jsonl')
                       if row['node'] == 'schema_filter_rc3'), 'schema filter')
    annotations = indexed(lines(filter_source / 'evaluation/linking_details.jsonl'), 'filter annotation')
    versions = indexed(lines(source / 'record_export/versions.jsonl'), 'source version')
    prepared_filter = read(filter_source / 'raw_records/linking_extension_prepared_inputs.json')
    tasks, schemas = {}, {}
    group_config = {entry['name']: entry for entry in manifest['config']['groups']}
    selected = set()
    for group, binding in manifest['groups'].items():
        ids = [str(value) for value in binding['ids']]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate DAIL selected question: {group}')
        selected.update((group, question_id) for question_id in ids)
    if set(versions) != selected:
        raise ValueError('Source versions differ from selected DAIL questions')
    for label, records in (('filter questions', source_questions), ('schema filters', filters),
                           ('filter annotations', annotations)):
        if not selected <= set(records):
            raise ValueError(f'{label} omit selected DAIL questions')
    for group, binding in manifest['groups'].items():
        group_tasks = load_prepared_group(preparation, group)
        available = {str(question_id): item for question_id, item in group_tasks.items()}
        if len(available) != len(group_tasks) or not set(map(str, binding['ids'])) <= set(available):
            raise ValueError(f'DAIL group membership mismatch: {group}')
        for question_id in map(str, binding['ids']):
            item = available[question_id]
            identity = group, str(question_id)
            original, other = item['task'], source_questions[identity]
            if (original['question'], original.get('evidence', ''), original['rc3_ref']['content'],
                original['database']['database_id']) != (
                other['question'], other.get('evidence', ''), other['rc3'], other['database']['database_id']):
                raise ValueError(f'Filter question or RC identity mismatch: {identity}')
            physical = catalog(item['schema'])
            metadata = prepared_filter['metadata'][other['schema_ref']]
            filter_catalog = {table['table_name'].strip(): sorted(
                col.get('original_column_name', col.get('column_name')).strip() for col in table['columns'])
                for table in metadata}
            if {name: sorted(columns) for name, columns in physical.items()} != filter_catalog:
                raise ValueError(f'Full schema membership differs: {identity}')
            schema_id = digest(item['schema'])
            schemas[schema_id] = item['schema']
            tasks[identity] = {**item, 'schema_id': schema_id, 'filter': filters[identity],
                               'annotation': annotations[identity], 'version': versions[identity],
                               'compute_cv_link': group_config[group]['compute_cv_link']}
    if set(tasks) != selected:
        raise ValueError('Question sets differ')
    # Match current SQLite content against the original value-linking input once.
    frozen = read(preparation / 'identity.json')
    cv_paths = sorted({item['task']['database']['path'] for item in tasks.values() if item['compute_cv_link']})
    def check_database(path):
        if Path(path + '-wal').exists() or _file_hash(path) != frozen['cv_database_sha256'][path]:
            raise ValueError(f'Frozen value-linking database differs: {path}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_database, cv_paths))
    return tasks, schemas, {
        'format': 'dail-linking-comparison-v1', 'questions': len(tasks),
        'source_analysis': str(source), 'filter_source': str(filter_source),
        'dail_manifest_sha256': _file_hash(batch / 'manifest.json'),
        'source_versions_sha256': _file_hash(source / 'record_export/versions.jsonl'),
        'filter_index_sha256': _file_hash(filter_source / 'index.json'),
        'preparation_identity_sha256': _file_hash(preparation / 'identity.json'),
        'primary_metric': 'column_macro_recall', 'new_model_requests': 0,
        'execution': 'native.link_question over cropped schema; original question tokens reused',
    }


class QuestionTokens:
    def __init__(self, linking):
        self.linking = linking

    def tokenize_for_copying(self, text):
        return list(self.linking['question']), list(self.linking['question_for_copying'])


def native_run(item, schema, tokens, stopwords_path):
    task = item['task']
    question = task['question'] + (' ' + task['evidence'] if task.get('evidence') else '')
    connection = None
    try:
        if item['compute_cv_link']:
            connection = sqlite3.connect(Path(task['database']['path']).resolve().as_uri() + '?mode=ro', uri=True)
            connection.execute('PRAGMA query_only=ON')
        return native.link_question(question, schema, QuestionTokens(item['linking']),
            compute_cv_link=item['compute_cv_link'], connection=connection,
            tokenized_schema=tokens, stopwords_path=stopwords_path)
    finally:
        if connection is not None:
            connection.close()


def evaluate(identity, item, cropped, raw, status, error):
    group, question_id = identity
    annotation, filter_node = item['annotation'], item['filter']
    gold = annotation['gold']
    gt, gc = set(gold['tables']), set(map(tuple, gold['columns']))
    # Validate gold names using the exact same catalog convention as DIN.
    _canonical_gold({'status': annotation['annotation_status'], 'required_tables': gold['tables'],
                     'required_columns': gold['columns']}, catalog(item['schema']))
    base = normalized_linking(item['linking'], item['schema'])
    rc = normalized_linking(raw, cropped['schema']) if status == 'succeeded' else {'tables': [], 'columns': []}
    def side(value, state):
        return {**value, 'status': state,
                'table_recall': _ratio(len(gt & set(value['tables'])), len(gt)),
                'column_recall': _ratio(len(gc & set(map(tuple, value['columns']))), len(gc))}
    usage = _usage(filter_node.get('usage'))
    return {'group': group, 'question_id': question_id, 'version_id': item['version']['version_id'],
            'task_key': annotation['task_key'], 'annotation_status': annotation['annotation_status'],
            'gold': gold, 'base': side(base, 'succeeded'), 'rc3': side(rc, status),
            'filter': annotation['filter'], 'tokens': {
                'base_linking': dict(ZERO), 'schema_filter_rc3': usage,
                'linking_rc3': dict(ZERO), 'rc3_combined': usage},
            'error': error}


def run_item(identity, item, all_tokens, verify_base=False, *, stopwords_path):
    start = time.monotonic()
    filter_node = item['filter']
    source_tokens = all_tokens[item['schema_id']]
    if verify_base:
        baseline = native_run(item, item['schema'], source_tokens, stopwords_path)
        if baseline != item['linking']:
            raise ValueError(f'Original native Linking cannot be reproduced: {identity}')
    cropped, raw, error = None, None, None
    status = 'dependency_failed'
    if filter_node['status'] == 'succeeded':
        cropped = crop_schema(item['schema'], filter_node['result']['filtered_metadata'])
        tokens = {'columns': [source_tokens['columns'][i] for i in cropped['column_local_to_full']],
                  'tables': [source_tokens['tables'][i] for i in cropped['table_local_to_full']]}
        try:
            raw = native_run(item, cropped['schema'], tokens, stopwords_path)
            status = 'succeeded'
        except Exception as exc:
            status, error = 'failed', {'type': type(exc).__name__, 'message': str(exc)}
    else:
        error = {'type': 'filter_dependency_failed', 'source_reason': filter_node.get('reason')}
    evaluation = evaluate(identity, item, cropped, raw, status, error)
    return {'task_key': item['version']['task_key'], 'version_id': item['version']['version_id'],
            'linking': {
                'status': status, 'compute_cv_link': item['compute_cv_link'],
                'full_schema_id': item['schema_id'], 'full_schema_store': 'raw_records/linking.sqlite3#schemas',
                'base': {'status': 'succeeded', 'raw': item['linking'], 'mask': item['mask'],
                         'model_requests': 0, 'usage': dict(ZERO), 'origin': 'original_preparation'},
                'schema_filter_rc3': {**filter_node, 'origin': 'reused_DIN_schema_filter'},
                'rc3': {'status': status, 'raw': raw,
                        'mask': native.mask_question(raw) if raw is not None else None,
                        'cropped_schema': cropped, 'model_requests': 0, 'usage': dict(ZERO), 'error': error},
                'elapsed_seconds': time.monotonic() - start, 'baseline_recomputed_equal': verify_base,
                'generation_uses_filtered_linking': False,
            }, 'evaluation': evaluation}


def schema_tokens(schemas, pending, work, resources):
    path = work / 'schema_tokens.json'
    cached = read(path) if path.exists() else {}
    required = {item['schema_id']: schemas[item['schema_id']] for item in pending}
    names = sorted({name for schema in required.values() for name in
                    [*(value for _, value in schema['column_names']), *schema['table_names']]})
    missing = [name for name in names if name not in cached]
    if missing:
        with LocalCoreNLP(resources, work / 'corenlp', root=ROOT) as tokenizer:
            for number, name in enumerate(missing, 1):
                cached[name] = tokenizer.tokenize(name)
                if number % 500 == 0:
                    print(encoded({'tokenized_schema_names': number, 'new_names': len(missing)}), flush=True)
        temporary = path.with_suffix('.pending')
        temporary.write_text(encoded(cached), encoding='utf-8')
        temporary.replace(path)
    return {sid: {'columns': [cached[name] for _, name in schema['column_names']],
                  'tables': [cached[name] for name in schema['table_names']]}
            for sid, schema in required.items()}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Sealed Generation analysis directory')
    parser.add_argument('--output', type=Path, required=True, help='New handoff directory; must not exist')
    parser.add_argument('--batch', type=Path, required=True, help='Frozen DAIL Generation batch')
    parser.add_argument('--filter-handoff', type=Path, required=True, help='Sealed DIN schema-filter handoff')
    parser.add_argument('--work-dir', type=Path, required=True, help='Resumable local comparison directory')
    parser.add_argument('--resources', type=Path, help='Verified local resource manifest; required unless exporting only')
    parser.add_argument('--nltk-data', type=Path, help='Stopwords directory; otherwise use resource manifest or cache/dail_sql/assets/nltk_data')
    parser.add_argument('--guide', type=Path, help='Optional handoff guide; defaults to docs/evaluation.md')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--smoke-per-group', type=int, default=0)
    parser.add_argument('--export-only', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1 or args.smoke_per_group < 0:
        parser.error('workers must be positive and smoke-per-group nonnegative')
    if not args.export_only and args.resources is None:
        parser.error('--resources is required unless --export-only is used')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    store_path = args.work_dir / 'linking.sqlite3'
    if not args.export_only:
        resources = read(args.resources)
        stopwords = args.nltk_data or Path(resources.get('nltk_data', ROOT / 'cache/dail_sql/assets/nltk_data'))
        tasks, schemas, frozen = load_inputs(args.source, args.batch, args.filter_handoff)
        print(encoded({'phase': 'validated', 'questions': len(tasks), 'schemas': len(schemas)}), flush=True)
        with sqlite3.connect(store_path) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS items(grp TEXT,question TEXT,payload_json TEXT,sha256 TEXT,
                PRIMARY KEY(grp,question));
                CREATE TABLE IF NOT EXISTS schemas(schema_id TEXT PRIMARY KEY,payload_json TEXT,sha256 TEXT);''')
            prior = db.execute("SELECT value FROM metadata WHERE key='frozen'").fetchone()
            if prior and prior[0] != encoded(frozen):
                raise ValueError('Input identity changed since this comparison began')
            db.execute("INSERT OR IGNORE INTO metadata VALUES('frozen',?)", (encoded(frozen),))
            for sid, schema in schemas.items():
                raw = encoded(schema)
                db.execute('INSERT OR IGNORE INTO schemas VALUES(?,?,?)', (sid, raw, hashlib.sha256(raw.encode()).hexdigest()))
            db.commit()
            done = set()
            for group, question, raw, checksum in db.execute('SELECT * FROM items'):
                if hashlib.sha256(raw.encode()).hexdigest() != checksum:
                    raise ValueError('Stored comparison row checksum mismatch')
                done.add((group, question))
            selected = list(tasks)
            if args.smoke_per_group:
                counts = {}
                selected = []
                for identity in tasks:
                    group = identity[0]
                    if counts.get(group, 0) < args.smoke_per_group:
                        selected.append(identity)
                        counts[group] = counts.get(group, 0) + 1
            pending = [identity for identity in selected if identity not in done]
            tokens = schema_tokens(schemas, [tasks[identity] for identity in pending], args.work_dir, resources)
            native._functions(str(stopwords.resolve()))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_item, identity, tasks[identity], tokens,
                                       bool(args.smoke_per_group), stopwords_path=stopwords): identity for identity in pending}
                for future in as_completed(futures):
                    identity, result = futures[future], future.result()
                    raw = encoded(result)
                    db.execute('INSERT INTO items VALUES(?,?,?,?)', (*identity, raw, hashlib.sha256(raw.encode()).hexdigest()))
                    db.commit()
                    done.add(identity)
                    if len(done) % 100 == 0 or args.smoke_per_group or len(done) == len(tasks):
                        print(encoded({'phase': 'linking', 'completed': len(done), 'total': len(tasks),
                                       'last': identity, 'status': result['linking']['status']}), flush=True)
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if args.smoke_per_group:
            return
    from scripts.rc_evaluation.dail_sql.linking_package import export_package
    print(export_package(args.source, args.output, store_path, guide=args.guide), flush=True)


if __name__ == '__main__':
    main()
