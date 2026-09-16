"""Frozen inputs and one-time, read-only preparation. Gold stays out of DinTask."""
from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3

import sqlglot
from sqlglot.optimizer.scope import traverse_scope
from scripts.rc_evaluation.dail_sql.contracts import select_rc3
from scripts.baseline_adapters.dail_sql.execution import execute_sql

OUTPUT_NODES = ('generation_base', 'generation_rc3', 'revision_base', 'revision_rc3')
NODES = ('linking', 'decomposition') + OUTPUT_NODES
LEGACY_COMMIT = '51cded06f5de1c6e787a659c0a4f3c6dfc2bd3d5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class TaskKey:
    group: str
    question_id: str

    def __post_init__(self):
        object.__setattr__(self, 'question_id', str(self.question_id))


@dataclass(frozen=True)
class DinSettings:
    model: str = 'qwen3.8-2.4t-a95b'
    n: int = 1
    temperature: float = 0
    max_attempts: int = 5
    request_timeout_seconds: float = 910
    request_limit: int = 8000
    start_rate: float = 50
    sql_workers: int = 20
    sql_timeout_seconds: float = 180


def validate_settings(settings):
    if settings.n != 1 or settings.temperature != 0:
        raise ValueError('DIN requires n=1 and temperature=0')
    for name in ('max_attempts', 'request_limit', 'sql_workers'):
        if type(getattr(settings, name)) is not int or getattr(settings, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('request_timeout_seconds', 'start_rate', 'sql_timeout_seconds'):
        value = getattr(settings, name)
        if not 0 < value < float('inf'):
            raise ValueError(f'{name} must be positive and finite')
    return settings


@dataclass(frozen=True)
class DinTask:
    key: TaskKey
    question: str
    evidence: str
    database: dict
    schema_ref: str
    rc3: dict
    label: str
    source_refs: dict = field(default_factory=dict)


@dataclass
class PreparedInputs:
    tasks: dict
    schemas: dict
    evaluation: dict
    templates: dict
    legacy: dict
    identities: dict


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_settings(DinSettings(**config.get('settings', {})))
    if len({g['name'] for g in config['groups']}) != len(config['groups']):
        raise ValueError('Duplicate groups')
    return config


def literal_values(path):
    source = path.read_text()
    # Official Spider source contains an intentionally blank API_KEY assignment.
    # Read literal strings without parsing/executing its unrelated runtime code.
    if path.parent.name == 'DIN-SQL' and path.name.startswith('DIN-SQL'):
        return {m.group(1): ast.literal_eval(m.group(2)+m.group(3)+m.group(2))
                for m in re.finditer(r'^([A-Za-z_]\w*)\s*=\s*(\'\'\'|""")(.*?)\2', source, re.M|re.S)}
    tree = ast.parse(source)
    result = {}
    for item in tree.body:
        if isinstance(item, ast.Assign):
            try:
                value = ast.literal_eval(item.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, str):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = value
    return result


def load_templates(code_root):
    official = code_root/'baselines/DIN-SQL'
    legacy = code_root/'baselines_reproduce/DIN-SQL'
    spider = literal_values(official/'DIN-SQL.py')
    source = (official/'DIN-SQL.py').read_text()
    match = re.search(r'def debuger\(.*?instruction = (""".*?""")', source, re.S)
    if not match:
        raise ValueError('Official Spider debugger instruction not found')
    spider['debug_instruction'] = ast.literal_eval(match.group(1))
    return {'spider': spider,
            'bird': literal_values(official/'DIN-SQL_BIRD.py'),
            'colleague': {**literal_values(legacy/'schema_linking.py'),
                          **literal_values(legacy/'difficulty_decomposition.py'),
                          **literal_values(legacy/'sql_generation.py')}}


def legacy_pure_functions(code_root, filename, names, extra=None):
    """Load ONLY named pure functions, not imports, main, or legacy runners.

    Used for already-tested context formatting and parsing; the file is frozen
    by the batch source hashes. This avoids a second copy of long method code.
    """
    path = code_root/'baselines_reproduce/DIN-SQL'/filename
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise ValueError(f'Missing pure function in {filename}')
    namespace = dict(ast=ast, csv=csv, io=io, json=json, re=re, sqlite3=sqlite3, Path=Path)
    namespace.update(extra or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def physical_tables(sql, dialect):
    tree = sqlglot.parse_one(sql, read=dialect)
    if not isinstance(tree, sqlglot.exp.Query):
        raise ValueError('Gold must be a read-only query')
    return sorted({source.name for scope in traverse_scope(tree)
                   for _, source in scope.selected_sources.values()
                   if isinstance(source, sqlglot.exp.Table)})


def classify_gold(sql, tables):
    # Tokenizer, unlike substring matching, excludes quoted words/comments.
    tokens = sqlglot.Tokenizer().tokenize(sql)
    words = [t.token_type.name for t in tokens]
    if words.count('SELECT') > 1 or any(w in words for w in ('UNION', 'INTERSECT', 'EXCEPT')):
        return 'NESTED'
    return 'NON-NESTED' if len(tables) > 1 or 'JOIN' in words else 'EASY'


def public_pg_context(database, meta_dir, *, execute=execute_sql):
    parts = []
    for path in sorted(meta_dir.glob('*.csv')):
        columns = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
        names = [row.get('original_column_name') or row.get('column_name') for row in columns]
        if not names or any(not name for name in names):
            raise ValueError(f'Invalid public Meta: {path}')
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        query = f'SELECT {", ".join(map(quote, names))} FROM {quote(path.stem)} LIMIT 3'
        sample = execute(database, query, timeout_seconds=180)
        if sample['status'] != 'success':
            raise ValueError(f'Cannot sample {database["database_id"]}.{path.stem}: {sample.get("error")}')
        parts.append(f'Table {path.stem}, columns = [*,{",".join(names)}]')
        for row, name in zip(columns, names):
            parts.append(f'Column {name}: type -> {row.get("data_format", row.get("data_type", ""))}, '
                         f'column description -> {row.get("column_description", "")}, '
                         f'value description -> {row.get("value_description", "")}')
        parts.append('Sample rows: ' + json.dumps(sample['rows'], ensure_ascii=False, default=str))
    if not parts:
        raise ValueError(f'No public Meta: {meta_dir}')
    return '\n'.join(parts)


def spider_context(schema):
    tables = schema['table_names_original']
    columns = {t: ['*'] for t in tables}
    for idx, col in schema['column_names_original']:
        if idx >= 0:
            columns[tables[idx]].append(col)
    lines = [f'Table {t}, columns = [{",".join(columns[t])}]' for t in sorted(tables)]
    keys = []
    for a, b in schema['foreign_keys']:
        ti, ci = schema['column_names_original'][a]
        tj, cj = schema['column_names_original'][b]
        keys.append(f'{tables[ti]}.{ci} = {tables[tj]}.{cj}')
    return '\n'.join(lines + ['Foreign_keys = [' + ','.join(keys) + ']'])


def prepare_inputs(config, code_root):
    code_root = Path(code_root).resolve()
    identities = {}
    def read(path):
        path = (code_root/path).resolve()
        raw = path.read_bytes()
        identities[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    templates = load_templates(code_root)
    for directory in ('baselines/DIN-SQL', 'baselines_reproduce/DIN-SQL'):
        for path in (code_root/directory).glob('*.py'):
            identities[str(path)] = file_hash(path)
    sqlite_context = legacy_pure_functions(code_root, 'schema_linking.py',
                                          ['quote_identifier','column_descriptions','database_context'])['database_context']
    spider = {}
    if any(g['name'].startswith('spider') for g in config['groups']):
        for filename in ('tables.json', 'test_tables.json'):
            for schema in read('../Spider/data/'+filename):
                spider[schema['db_id']] = spider_context(schema)
                primary = [schema['column_names_original'][i] for i in schema.get('primary_keys',[])]
                spider[schema['db_id']+':primary'] = 'Primary_keys = [' + ','.join(
                    schema['table_names_original'][i]+'.'+col for i,col in primary) + ']'
    schemas = {'spider:college_2': spider['college_2']} if spider else {}
    tasks, evaluation = {}, {}
    for group in config['groups']:
        name = group['name']
        base = Path('scripts')/name
        rows = read(str(base/'preprocessed_data'/f'{name}.json'))
        rc_rows = read(str(base/'rc.json'))
        rc = {str(r['index']): select_rc3(r) for r in rc_rows}
        if len(rc) != len(rc_rows):
            raise ValueError(f'Duplicate RC IDs: {name}')
        gold_rows = read(group['gold'])
        gold = {str(r.get('index', r.get('question_id', i))): r for i, r in enumerate(gold_rows)}
        dependencies = {}
        if group['dialect'] == 'sqlite':
            dependencies = {str(r['index']): r for r in read(str(base/'gold_sql_schema_linking.json'))}
        for row in rows:
            key = TaskKey(name, row['index'])
            if key in tasks:
                raise ValueError(f'Duplicate task: {key}')
            contract, reference = rc[key.question_id], gold[key.question_id]
            for other in (contract, reference):
                db = other.get('database_id', other.get('db_id'))
                if db != row['db_id'] or other['question'] != row['question']:
                    raise ValueError(f'Input identity mismatch: {key}')
            if contract['evidence'] != row.get('evidence', ''):
                raise ValueError(f'Evidence mismatch: {key}')
            sql = reference.get('gold_sql', reference.get('SQL', reference.get('query')))
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError(f'Missing gold SQL: {key}')
            database = {'dialect':group['dialect'], 'database_id':row['db_id']}
            if group['dialect'] == 'sqlite':
                database['path'] = str((code_root/group['database_root']/row['db_id']/(row['db_id']+'.sqlite')).resolve(strict=True))
                dep = dependencies[key.question_id]
                if dep['db_id'] != row['db_id'] or dep['question'] != row['question'] or dep['gold_sql'].strip() != sql.strip():
                    raise ValueError(f'Gold dependency mismatch: {key}')
                tables = dep['schemalinking']
            else:
                tables = physical_tables(sql, 'postgres')
            ref = f'{name}:{row["db_id"]}'
            if ref not in schemas:
                meta = code_root/base/'preprocessed_data/meta'/row['db_id']
                for path in meta.glob('*.csv'):
                    identities[str(path)] = file_hash(path)
                context = (public_pg_context(database, meta) if group['dialect'] == 'postgresql'
                           else sqlite_context(Path(database['path']), meta))
                schemas[ref] = {'context':context, 'spider':spider.get(row['db_id']),
                                'primary':spider.get(row['db_id']+':primary','Primary_keys = []')}
            tasks[key] = DinTask(key, row['question'], row.get('evidence',''), database, ref,
                                contract['rc_round3'], classify_gold(sql, tables), {'input':str(base)})
            evaluation[f'{name}/{key.question_id}'] = {'gold_sql':sql, 'database':database}
    prepared = PreparedInputs(tasks, schemas, evaluation, templates, {}, identities)
    if config.get('reuse_legacy', True):
        prepared.legacy = read_legacy(code_root/'baselines_reproduce/DIN-SQL', prepared)
    return prepared


def sub_questions_bound(questions, prompt):
    return all(q in prompt or json.dumps(q,ensure_ascii=False) in prompt for q in questions)


def read_legacy(source, prepared):
    accepted, rejected, hashes = {}, [], {}
    stage_paths = {'linking':('schema_linking','prompts','qwen38_result.json'),
                   'decomposition':('difficulty_decomposition','prompts','qwen38_result.json'),
                   'generation_base':('sql_generation','prompt','qwen38_result.json'),
                   'generation_rc3':('sql_generation','prompt_rc','qwen38_result_rc.json'),
                   'revision_base':('self_correction','prompt','qwen38_result.json'),
                   'revision_rc3':('self_correction','prompt_rc','qwen38_result_rc.json')}
    for group in dict.fromkeys(k.group for k in prepared.tasks):
        group_tasks = [t for k,t in prepared.tasks.items() if k.group == group]
        for node, (stage, folder, filename) in stage_paths.items():
            path = source/group/stage/filename
            if not path.exists():
                continue
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            rows = json.loads(raw)
            for task in group_tasks:
                key = f'{group}/{task.key.question_id}'
                row = rows.get(task.key.question_id)
                if row is None:
                    continue
                reason = None
                prompt_path = source/group/stage/folder/(task.key.question_id+'.txt')
                deterministic = node == 'decomposition' and task.label != 'NESTED'
                prompt = prompt_path.read_text() if prompt_path.exists() else ''
                parents = accepted.get(key, {})
                if row.get('status',{}).get('success') is not True:
                    reason = 'legacy_failure_observation'
                elif not deterministic and (not prompt or task.question not in prompt):
                    reason = 'missing_question_binding'
                elif node == 'linking' and not isinstance(row.get('result'), list):
                    reason = 'invalid_linking'
                elif node == 'decomposition' and (not isinstance(row.get('result'), dict) or row['result'].get('label') != task.label):
                    reason = 'label_mismatch'
                elif node in OUTPUT_NODES and not isinstance(row.get('result'), str):
                    reason = 'invalid_sql'
                elif node.startswith('generation') or (node == 'decomposition' and not deterministic):
                    links = parents.get('linking',{}).get('result')
                    if links is None or '['+','.join(links)+']' not in prompt:
                        reason = 'linking_parent_unconfirmed'
                    elif node.startswith('generation'):
                        dec = parents.get('decomposition',{}).get('result')
                        if dec is None or not sub_questions_bound(dec.get('sub_questions',[]),prompt):
                            reason = 'decomposition_parent_unconfirmed'
                elif node.startswith('revision'):
                    sql = parents.get('generation_base',{}).get('result')
                    if not sql or sql not in prompt:
                        reason = 'revision_parent_unconfirmed'
                if reason is None and node.endswith('rc3'):
                    marker = prompt.rfind('Result Contract:\n')
                    try:
                        content, _ = json.JSONDecoder().raw_decode(prompt[marker+len('Result Contract:\n'):].lstrip())
                        if marker < 0 or content != task.rc3:
                            reason = 'rc3_mismatch'
                    except ValueError:
                        reason = 'rc3_unconfirmed'
                if reason:
                    rejected.append({'key':key,'node':node,'reason':reason})
                    continue
                if prompt:
                    hashes[str(prompt_path)] = file_hash(prompt_path)
                resource = row.get('resource') or [None, None]
                accepted.setdefault(key,{})[node] = {
                    'node':node, 'status':'succeeded', 'result':row['result'],
                    'reason':row['status'].get('reason'),
                    'usage':({'prompt_tokens':0,'completion_tokens':0,'total_tokens':0} if deterministic else resource[0]),
                    'origin':'legacy_import', 'input_fingerprint':digest([key,node,prompt]),
                    'parent_refs':{}, 'response_ref':None, 'fallback_used':bool(row['status'].get('reason')),
                    'source_refs':{'path':str(path),'sha256':hashes[str(path)],'commit':LEGACY_COMMIT,
                                   'record_id':task.key.question_id,'prompt_path':str(prompt_path) if prompt else None,
                                   'model_evidence':'legacy_alias_unverified', 'response_model':None,
                                   'attempt_count':None,'elapsed_seconds':resource[1]}}
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise ValueError(f'Legacy source changed during import: {path}')
    prepared.identities.update(hashes)
    return {'accepted':accepted,'rejected':rejected}
