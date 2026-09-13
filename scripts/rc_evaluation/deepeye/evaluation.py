"""Reference-only evaluation in a separate append-only RunStore.

Result agreement is a diagnostic, not official BIRD-Interact accuracy. Database
version is an operator declaration, not a detected remote database snapshot.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
from types import SimpleNamespace

from scripts.baseline_adapters.deepeye.run_store import RunStore, restore_jsonable, to_jsonable
from .comparison import compare_results, SUCCESS_TYPES

STAGES = ('schema_linking', 'sql_generation', 'sql_revision', 'sql_selection')
EVALUATION_VERSION = 'deepeye-independent-evaluation-v1'


def _hash(value):
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _read_run(path):
    with RunStore.open(path, read_only=True) as store:
        with store._read_snapshot():
            return store.manifest, store.attempts(), store.events()


def _references(paths):
    records, sources = {}, {}
    for split, path in sorted(paths.items()):
        path = Path(path).resolve()
        raw = path.read_bytes()
        sources[split] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        content = raw.decode('utf-8-sig').strip()
        rows = json.loads(content) if content.startswith('[') else [json.loads(line) for line in content.splitlines() if line.strip()]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError('Reference must contain JSON object records')
        for row in rows:
            if row.get('category') == 'Query':
                key = f"{split}/{row.get('instance_id')}"
                records.setdefault(key, []).append(row)
    return records, sources


def _reference(binding, records):
    matches = records.get(binding['task_key'], [])
    if len(matches) != 1:
        return {'status': 'missing_reference' if not matches else 'duplicate_reference'}
    row = matches[0]
    provenance = {'record_sha256': _hash(row)}
    database = row.get('selected_database', row.get('db_id'))
    if database != binding['database_id'] or ('db_id' in row and row['db_id'] != database):
        return {**provenance, 'status': 'reference_database_mismatch'}
    # Refuse any setup/cleanup requirement; evaluator never executes mutations.
    if any(value for key, value in row.items() if 'preprocess' in key.lower().replace('_', '')
           or 'cleanup' in key.lower().replace('_', '') or key.lower() in ('pre_sql', 'post_sql')):
        return {**provenance, 'status': 'unsupported_reference_setup'}
    sql = row.get('sol_sql')
    if isinstance(sql, list) and len(sql) == 1:
        sql = sql[0]
    if not isinstance(sql, str) or not sql.strip():
        return {**provenance, 'status': 'invalid_reference_sql'}
    return {**provenance, 'status': 'available', 'sql': sql, 'sql_sha256': _hash(sql)}


def _artifact(attempt):
    if not attempt or attempt.get('status', 'succeeded') != 'succeeded':
        return {}
    return restore_jsonable(attempt.get('payload', {}).get('artifact', {}))


def _validate_source(manifest, source_manifest, source_attempts):
    checkpoints = manifest.get('source_checkpoints', {})
    bindings = {row['task_key']: row for row in manifest.get('items', [])}
    if checkpoints and set(checkpoints) != set(bindings):
        raise ValueError('Source checkpoint task set differs from experiment')
    embedded = manifest.get('source_manifest')
    if embedded is not None:
        if _hash(embedded) != manifest.get('source_manifest_fingerprint'):
            raise ValueError('Embedded source manifest fingerprint mismatch')
        if source_manifest and embedded != source_manifest:
            raise ValueError('Source manifest differs from embedded snapshot')
        original = {row['task_key']: row for row in embedded['items']}
        if any(binding != original.get(key) for key, binding in bindings.items()):
            raise ValueError('Source and experiment item bindings differ')
    original_attempts = {row['attempt_id']: row for row in source_attempts}
    for key, checkpoint in checkpoints.items():
        if not checkpoint.get('upstream_state_sha256'):
            raise ValueError('Source checkpoint has no upstream identity')
        if 'input' in checkpoint and _hash(checkpoint['input']) != checkpoint.get('input_sha256'):
            raise ValueError('Source checkpoint input fingerprint mismatch')
        for stage, record in checkpoint.get('stages', {}).items():
            if _hash(record.get('payload')) != record.get('payload_sha256'):
                raise ValueError('Source checkpoint payload fingerprint mismatch')
            if source_manifest:
                original = original_attempts.get(record.get('attempt_id'))
                if (not original or original['status'] != 'succeeded' or original['item_key'] != key
                        or original['stage'] != stage or original['payload'] != record['payload']
                        or original['input_fingerprint'] != record.get('input_fingerprint')):
                    raise ValueError('Source checkpoint does not match its original attempt')


def _pool(sqls, evaluate):
    if not isinstance(sqls, list):
        return {'status': 'missing', 'slots': None, 'unique_sql': None, 'execution_success': 0,
                'correct': 0, 'unknown': None, 'has_correct': None, 'results': []}
    results = [evaluate(sql) for sql in sqls]
    values = [row['bag_equal'] for row in results]
    has_correct = True if True in values else (None if None in values else False)
    return {'status': 'available', 'slots': len(sqls),
            'unique_sql': len({sql for sql in sqls if isinstance(sql, str) and sql.strip()}),
            'execution_success': sum(row['execution_success'] for row in results),
            'correct': values.count(True), 'unknown': values.count(None), 'has_correct': has_correct,
            'ordered_has_correct': True if any(row['ordered_equal'] is True for row in results)
            else (None if any(row['ordered_equal'] is None for row in results) else False), 'results': results}


def _transition(before, after):
    if before is None or after is None:
        return 'unknown'
    return {(False, True): 'repair', (True, False): 'harm',
            (True, True): 'unchanged_correct', (False, False): 'unchanged_incorrect'}[(before, after)]


def schema_coverage(sql, linked):
    """Conservative syntactic gold table/column coverage, without Meta/RC truth.

    No schema catalog is used to guess wildcard expansion, ambiguous unqualified
    columns, CTE/subquery lineage or quoted-name case folding.
    """
    output = {'status': 'unknown', 'table_coverage': None, 'column_coverage': None,
              'retained_tables': None, 'retained_columns': None, 'reason': 'missing_linking_or_reference'}
    if not isinstance(linked, dict):
        return output
    output.update(retained_tables=len(linked), retained_columns=sum(len(c) for c in linked.values() if isinstance(c, list)))
    if not isinstance(sql, str):
        return output
    try:
        import sqlglot
        from sqlglot import exp
        parsed = sqlglot.parse(sql, read='postgres')
        if len(parsed) != 1 or not isinstance(parsed[0], exp.Select):
            return {**output, 'reason': 'unsupported_query_structure'}
        tree = parsed[0]
        if any(tree.find_all(exp.CTE, exp.Subquery)) or len(list(tree.find_all(exp.Select))) != 1:
            return {**output, 'reason': 'unresolved_query_lineage'}
        tables, aliases = set(), {}
        def name(identifier):
            if identifier is None:
                return ''
            return identifier.name if identifier.args.get('quoted') else identifier.name.lower()
        for table in tree.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                return {**output, 'reason': 'unresolved_table'}
            table_name = name(table.this)
            if table.db or table.catalog:
                return {**output, 'reason': 'schema_qualified_table_unresolved'}
            tables.add(table_name)
            alias = table.args.get('alias')
            aliases[name(alias.this) if alias else table_name] = table_name
        retained = {str(t): set(cols) for t, cols in linked.items() if isinstance(cols, list)}
        output.update(reference_tables=sorted(tables), table_coverage=len(tables & retained.keys()) / len(tables) if tables else 1.0)
        columns, ambiguous = set(), False
        for column in tree.find_all(exp.Column):
            if column.is_star:
                ambiguous = True
                continue
            qualifier = name(column.args.get('table'))
            table = aliases.get(qualifier) if qualifier else (next(iter(tables)) if len(tables) == 1 else None)
            # SELECT aliases in ORDER/GROUP cannot be safely mapped without lineage.
            if not table or (not qualifier and any(alias.alias == column.name for alias in tree.find_all(exp.Alias))):
                ambiguous = True
                continue
            columns.add((table, name(column.this)))
        if any(tree.find_all(exp.Star)):
            ambiguous = True
        output['reference_columns'] = [list(pair) for pair in sorted(columns)]
        if any(join.args.get('using') or join.args.get('method') == 'NATURAL'
               for join in tree.find_all(exp.Join)):
            return {**output, 'reason': 'implicit_join_columns_unresolved'}
        if ambiguous:
            return {**output, 'reason': 'unresolved_or_wildcard_column'}
        covered = sum(column in retained.get(table, set()) for table, column in columns)
        return {**output, 'status': 'available', 'reason': None,
                'column_coverage': covered / len(columns) if columns else 1.0}
    except Exception as error:
        return {**output, 'reason': 'parse_unavailable:' + type(error).__name__}


def _selection_trace(events, source_events, attempt, threshold=None):
    origin = attempt.get('payload', {}).get('execution_origin') if attempt else None
    trace = events
    provenance = 'current_attempt'
    if origin in ('reused_no_native_llm_call', 'reused_unchanged_upstream'):
        trace, provenance = source_events, 'source_attempt'
    shortlists = [restore_jsonable(e['payload'].get('result')) for e in trace
                  if e['kind'] == 'component_result' and e['payload'].get('component') == 'selection.shortlist']
    shortlist = shortlists[-1] if shortlists else None
    sqls = [row[0] for row in shortlist] if isinstance(shortlist, (list, tuple)) and all(
        isinstance(row, (list, tuple)) and row and isinstance(row[0], str) for row in shortlist) else None
    pairs = any(e['payload'].get('component') == 'selection.pairwise_comparison' for e in trace)
    success = attempt and attempt.get('status') == 'succeeded'
    selected = _artifact(attempt).get('final_selected_sql')
    if sqls is None or not success:
        branch = 'unknown'
    elif not sqls:
        branch = 'fallback_no_valid_candidates'
    elif len(sqls) == 1:
        branch = 'single_shortlist_candidate'
    elif pairs:
        branch = 'pairwise_comparison'
    elif (type(threshold) in (int, float) and len(shortlist[0]) > 2
          and type(shortlist[0][2]) in (int, float) and shortlist[0][2] >= threshold
          and selected == sqls[0]):
        branch = 'consistency_shortcut'
    else:
        branch = 'unknown'  # no trace is not evidence of a consistency shortcut
    return {'branch': branch, 'trace_provenance': provenance, 'shortlist_sqls': sqls}


def _usage(attempts, events, manifest):
    from scripts.baseline_adapters.deepeye.run_usage import observed_usage
    from .injection import count_rc_requests, render_rc_block
    ids = {a['attempt_id'] for a in attempts}
    current = [e for e in events if e['attempt_id'] in ids]
    observed = observed_usage(SimpleNamespace(iter_events=lambda kinds: (e for e in current if e['kind'] in kinds)))
    participation = [a.get('payload', {}).get('rc_participation', {}) for a in attempts if a.get('payload')]
    rc_requests = 0
    if manifest.get('condition') == 'rc':
        blocks = {}
        for attempt in attempts:
            if attempt['stage'] != manifest['target_stage']:
                continue
            requests = [event for event in current if event['attempt_id'] == attempt['attempt_id']
                        and event['kind'] == 'api_request']
            if not requests:
                continue
            key = attempt['item_key']
            if key not in blocks:
                blocks[key] = render_rc_block(manifest['contracts'][key])
            rc_requests += count_rc_requests(requests, blocks[key])
    return {**observed, 'api_requests': observed['requests'], 'usage': observed['reported_tokens'],
            'rc_actual_requests': rc_requests,
            'rc_participation': participation, 'scope': 'current_run_attempts_only_including_failed_retries'}


def stage_token_pairs(manifest, attempts, events):
    """Compare retained target samples against the frozen native source, offline.

    This does not execute SQL, manufacture a control run, or relabel the existing
    all-attempt usage ledger. Legacy traces cannot prove the new sampling budget.
    """
    from .source import api_trace
    from .injection import count_rc_requests, render_rc_block
    target = manifest['target_stage']
    if manifest['condition'] != 'rc':
        raise ValueError('Native-source token pairing requires condition=rc')
    latest = {row['item_key']: row for row in attempts if row['stage'] == target}
    by_attempt = {}
    for event in events:
        by_attempt.setdefault(event['attempt_id'], []).append(event)
    items = {}
    def inspect(trace, label, exclusions):
        sampling, metric = trace.get('sampling', {}), trace.get('effective_sampling')
        if not trace.get('complete', False) or sampling.get('complete') is False:
            exclusions.append(label + '_sampling_incomplete')
        if metric is None or not sampling.get('groups'):
            exclusions.append(label + '_effective_sampling_unavailable')
        elif not metric['usage_complete']:
            exclusions.append(label + '_usage_incomplete')
        return metric
    for key, snapshot in manifest['source_checkpoints'].items():
        native_trace = snapshot['stages'][target]['api_trace']
        exclusions = []
        native = inspect(native_trace, 'native', exclusions)
        if not (native_trace['requests'] or native_trace.get('restored_samples')
                or (native and native['retained_samples'])):
            exclusions.append('native_zero_call_target')
        row = latest.get(key)
        rc = None
        if row is None:
            exclusions.append('rc_stage_missing')
        else:
            if row['status'] != 'succeeded':
                exclusions.append('rc_stage_' + row['status'])
            current = by_attempt.get(row['attempt_id'], [])
            rc_trace = api_trace(current)
            rc = inspect(rc_trace, 'rc', exclusions)
            requests = [e for e in current if e['kind'] == 'api_request']
            samples = [e['payload'] for e in current if e['kind'] == 'sample_result'
                       and e['payload'].get('succeeded')]
            applied = (all(p.get('rc_applied') is True for p in samples)
                       and bool(samples or requests))
            if requests:
                contract = manifest.get('contracts', {}).get(key)
                applied = applied and contract is not None and count_rc_requests(
                    requests, render_rc_block(contract)) == len(requests)
            if not applied:
                exclusions.append('rc_participation_unproven')
        eligible = not exclusions
        fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        items[key] = {'eligible': eligible, 'exclusions': exclusions, 'native': native, 'rc': rc,
                      'delta_tokens': {field: rc['known_tokens'][field] - native['known_tokens'][field]
                                       for field in fields} if eligible else None}
    eligible = [row for row in items.values() if row['eligible']]
    summary = {'tasks': len(items), 'eligible_pairs': len(eligible), 'excluded_pairs': len(items) - len(eligible),
               'exclusion_counts': dict(Counter(reason for row in items.values() for reason in row['exclusions']))}
    for side in ('native', 'rc'):
        summary[side + '_tokens'] = ({field: sum(row[side]['known_tokens'][field] for row in eligible)
                                     for field in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                                    if eligible else None)
    return {'target_stage': target, 'metric': 'finally_retained_successful_samples_only_v1',
            'control': 'frozen_native_source_no_extra_calls',
            'reasoning_semantics': 'subset_of_completion_tokens_not_added_to_total',
            'items': items, 'summary': summary}


@contextmanager
def _executor(env_file, effective_config):
    """Install only PG environment values for native read-only execution."""
    values = {}
    for line in Path(env_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, raw = line.split('=', 1)
        key = key.strip()
        if key in ('PG_HOST', 'PG_PORT', 'PG_USER', 'PG_PASSWORD', 'PG_SSLMODE'):
            parts = shlex.split(raw, comments=True)
            if len(parts) != 1 or key in values:
                raise ValueError('Invalid or duplicate PostgreSQL environment field')
            values[key] = parts[0]
    config = effective_config.get('postgres', {})
    for env_key, field in (('PG_HOST', 'host'), ('PG_PORT', 'port'), ('PG_USER', 'principal')):
        if field in config and str(config[field]) != values.get(env_key, os.environ.get(env_key)):
            raise ValueError('Evaluation database connection differs from inference identity: ' + field)
    if config.get('sslmode'):
        supplied_ssl = values.get('PG_SSLMODE', os.environ.get('PG_SSLMODE'))
        if supplied_ssl is not None and supplied_ssl != config['sslmode']:
            raise ValueError('Evaluation database connection differs from inference identity: sslmode')
        values.setdefault('PG_SSLMODE', config['sslmode'])
    prior = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        from scripts.baseline_adapters.deepeye.postgres_execution import execute_postgres_sql
        def execute(task_key, database_id, sql):
            return execute_postgres_sql(SimpleNamespace(database_id=database_id), sql,
                                        timeout=config.get('statement_timeout_seconds', 30)).model_dump()
        yield execute, {'host': os.environ.get('PG_HOST', 'localhost'),
                        'port': int(os.environ.get('PG_PORT', '5432')),
                        'principal': os.environ.get('PG_USER', 'postgres'),
                        'sslmode': os.environ.get('PG_SSLMODE', 'prefer'),
                        'statement_timeout_seconds': min(600, max(1, int(config.get('statement_timeout_seconds', 30)))),
                        'read_only': True, 'search_path': 'pg_catalog,public',
                        'single_statement': 'extended_protocol', 'row_limit': None}
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _summary(items):
    count = len(items)
    bag = sum(row['bag_equal'] is True for row in items.values())
    ordered = sum(row['ordered_equal'] is True for row in items.values())
    summary = {'tasks': count, 'bag_equal': bag, 'ordered_equal': ordered,
            'incorrect': sum(row['bag_equal'] is False for row in items.values()),
            'unknown': sum(row['bag_equal'] is None for row in items.values()),
            'bag_equal_rate': bag / count if count else None,
            'ordered_equal_rate': ordered / count if count else None,
            'denominator': 'all_fixed_tasks_including_missing_failed_unknown',
            'metric': 'strict_result_agreement_not_official_BIRD_Interact_accuracy'}
    for name in ('generation', 'revision'):
        pools = [row[name] if name == 'generation' else row[name]['after'] for row in items.values()]
        correct = sum(pool['has_correct'] is True for pool in pools)
        summary[name] = {'tasks': count, 'pool_has_correct_tasks': correct,
                         'pool_has_correct_rate': correct / count if count else None,
                         'pool_unknown_tasks': sum(pool['has_correct'] is None for pool in pools),
                         'ordered_pool_has_correct_tasks': sum(pool.get('ordered_has_correct') is True for pool in pools),
                         'candidate_slots': sum(pool['slots'] or 0 for pool in pools),
                         'missing_pool_tasks': sum(pool['slots'] is None for pool in pools),
                         'execution_success_candidates': sum(pool['execution_success'] for pool in pools),
                         'correct_candidates': sum(pool['correct'] for pool in pools),
                         'unknown_candidates': sum(pool['unknown'] or 0 for pool in pools)}
    summary['revision'].update({change: sum(row['revision'][change] for row in items.values())
                               for change in ('repair', 'harm', 'unchanged_correct', 'unchanged_incorrect', 'unknown')})
    eligible = [row['selection'] for row in items.values() if row['selection']['conditional_eligible'] is True]
    selected = sum(row['selected_when_pool_correct'] is True for row in eligible)
    summary['selection'] = {'tasks': count, 'eligible_tasks': len(eligible),
        'unknown_eligibility_tasks': sum(row['selection']['conditional_eligible'] is None for row in items.values()),
        'selected_when_pool_correct': selected,
        'selected_when_pool_correct_unknown': sum(row['selected_when_pool_correct'] is None for row in eligible),
        'selected_when_pool_correct_rate': selected / len(eligible) if eligible else None,
        'shortlist_retains_correct': sum(row['shortlist_retains_correct'] is True for row in eligible),
        'shortlist_retention_unknown': sum(row['shortlist_retains_correct'] is None for row in eligible),
        'branches': dict(Counter(row['selection']['branch'] for row in items.values()))}
    summary['coverage'] = {'available_tasks': sum(row['coverage']['status'] == 'available' for row in items.values()),
                           'unknown_tasks': sum(row['coverage']['status'] != 'available' for row in items.values())}
    summary['usage'] = {'api_requests': sum(row['usage']['api_requests'] for row in items.values()),
                        'rc_actual_requests': sum(row['usage']['rc_actual_requests'] for row in items.values()),
                        'usage_complete': all(row['usage']['usage_complete'] for row in items.values()),
                        'usage': dict(sum((Counter(row['usage']['usage']) for row in items.values()), Counter()))}
    return summary


def evaluate_run(run_dir: Path, output_dir: Path, reference_paths: dict[str, Path], *,
                 env_file: Path, database_version: str, source_run_dir: Path | None = None, execute_fn=None) -> dict:
    if not isinstance(database_version, str) or not database_version.strip():
        raise ValueError('An explicit database_version declaration is required')
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if run_dir == output_dir:
        raise ValueError('Evaluation requires a separate RunStore')
    manifest, attempts, events = _read_run(run_dir)
    bound_production = manifest.get('sources', {}).get('rc_evaluation_code_sha256')
    if manifest.get('condition') == 'rc' and bound_production is not None:
        from .cli import production_hash
        if production_hash() != bound_production:
            raise ValueError('RC evaluation production/prompt hash mismatch; use the corresponding production version')
    records, reference_sources = _references(reference_paths)
    bindings = manifest.get('items', [])
    keys = [row['task_key'] for row in bindings]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate task bindings')
    target = manifest.get('target_stage', 'sql_selection')
    if target not in STAGES:
        raise ValueError('Unknown target stage')
    source_manifest, source_attempts, source_events = {}, [], []
    source_path = source_run_dir or manifest.get('source_run')
    if source_path:
        source_manifest, source_attempts, source_events = _read_run(Path(source_path))
        expected = manifest.get('source_manifest_fingerprint')
        if expected and _hash(source_manifest) != expected:
            # Legacy source fingerprint uses the same canonical convention as its producer.
            from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
            if fingerprint(source_manifest) != expected:
                raise ValueError('Source manifest identity mismatch')
    _validate_source(manifest, source_manifest, source_attempts)
    identity = {'format': EVALUATION_VERSION, 'run_dir': str(run_dir), 'run_manifest': manifest,
                'run_snapshot_sha256': _hash({'attempts': attempts, 'events': events}),
                'source_run': str(Path(source_path).resolve()) if source_path else None,
                'reference_sources': reference_sources, 'database_version': database_version,
                'database_version_provenance': 'operator_declared_not_remote_snapshot_verified',
                'executor': 'injected' if execute_fn else 'native_read_only_postgres',
                'comparison': 'strict-finite-exact-numeric-bag-and-ordered-v1'}
    if execute_fn is None:
        with _executor(env_file, manifest.get('effective_config', {})) as (execute, database_identity):
            identity['database_execution'] = database_identity
            return _evaluate(identity, output_dir, records, attempts, events, source_events, execute)
    return _evaluate(identity, output_dir, records, attempts, events, source_events, execute_fn)


def _evaluate(identity, output_dir, records, attempts, events, source_events, execute):
    manifest = identity['run_manifest']
    target = manifest.get('target_stage', 'sql_selection')
    latest = {}
    for attempt in attempts:
        if attempt['status'] == 'succeeded':
            latest[(attempt['item_key'], attempt['stage'])] = attempt
    store = RunStore.open(output_dir, expected_manifest=identity) if output_dir.exists() else RunStore.create(output_dir, identity)
    items = {}
    with store:
        for binding in manifest['items']:
            key = binding['task_key']
            input_hash = _hash({'identity': identity, 'task_key': key})
            completed = store.completed(key, 'evaluation', input_hash)
            if completed:
                items[key] = completed['payload']
                continue
            aid = store.begin_attempt(key, 'evaluation', input_hash)
            reference = _reference(binding, records)
            cache = {}
            def execute_sql(sql, role):
                if not isinstance(sql, str) or not sql.strip():
                    return {'result_type': 'missing_sql'}
                if sql in cache:
                    return cache[sql]
                started = time.perf_counter()
                store.append_event(aid, 'sql_execution_start', {'sql': sql, 'sql_sha256': _hash(sql),
                    'database_id': binding['database_id'], 'database_version': identity['database_version'], 'role': role})
                try:
                    result = execute(key, binding['database_id'], sql)
                    if not isinstance(result, dict):
                        result = {'result_type': 'invalid_executor_result'}
                except Exception as error:
                    result = {'result_type': 'execution_error', 'error_type': type(error).__name__}
                try:
                    to_jsonable(result)
                except TypeError:
                    result = {'result_type': 'unsupported_result_serialization'}
                store.append_event(aid, 'sql_execution', {'sql': sql, 'sql_sha256': _hash(sql),
                    'database_id': binding['database_id'], 'database_version': identity['database_version'],
                    'role': role, 'elapsed_seconds': time.perf_counter() - started, 'result': result})
                cache[sql] = result
                return result
            gold = execute_sql(reference['sql'], 'reference') if reference['status'] == 'available' else {}
            if reference['status'] == 'available' and gold.get('result_type') not in SUCCESS_TYPES:
                reference = {**reference, 'status': 'reference_execution_unavailable', 'result_type': gold.get('result_type')}
            def assess(sql):
                result = execute_sql(sql, 'candidate') if reference['status'] == 'available' else {}
                return {'sql': sql, 'execution_success': result.get('result_type') in SUCCESS_TYPES,
                        **compare_results(result, gold)}
            checkpoint = manifest.get('source_checkpoints', {}).get(key, {})
            source_stages = checkpoint.get('stages', {})
            artifacts = {}
            for stage in STAGES:
                attempt = latest.get((key, stage))
                # Source prefix is allowed only before target; never fill missing intervention outputs.
                if attempt is None and STAGES.index(stage) < STAGES.index(target):
                    attempt = source_stages.get(stage)
                artifacts[stage] = _artifact(attempt)
            generation = _pool(artifacts['sql_generation'].get('sql_candidates'), assess)
            revised = _pool(artifacts['sql_revision'].get('sql_candidates_after_revision'), assess)
            target_attempt = latest.get((key, target))
            row = {'task_key': key, 'database_id': binding['database_id'], 'reference': reference,
                   'target_stage': target, 'target_status': 'succeeded' if target_attempt else 'missing_or_failed',
                   'bag_equal': None, 'ordered_equal': None,
                   'generation': generation, 'revision': {'before': generation, 'after': revised},
                   'coverage': schema_coverage(reference.get('sql'), artifacts['schema_linking'].get('final_linked_tables_and_columns')),
                   'usage': _usage([a for a in attempts if a['item_key'] == key], events, manifest)}
            slots = []
            for index, before in enumerate(generation['results']):
                after = revised['results'][index] if index < len(revised['results']) else None
                slots.append({'slot': index, 'before': before, 'after': after,
                              'sql_unchanged': before['sql'] == after['sql'] if after else None,
                              'change': _transition(before['bag_equal'], after['bag_equal'] if after else None),
                              'ordered_change': _transition(before['ordered_equal'], after['ordered_equal'] if after else None)})
            row['revision'].update(input_slots=generation['slots'], output_slots=revised['slots'], slots=slots,
                                   unknown=sum(slot['change'] == 'unknown' for slot in slots),
                                   extra_output_slots=max(0, len(revised['results']) - len(generation['results'])))
            for change in ('repair', 'harm', 'unchanged_correct', 'unchanged_incorrect'):
                row['revision'][change] = sum(slot['change'] == change for slot in slots)
            selection_attempt = latest.get((key, 'sql_selection'))
            selection_events = [e for e in events if selection_attempt and e['attempt_id'] == selection_attempt['attempt_id']]
            source_id = source_stages.get('sql_selection', {}).get('attempt_id')
            threshold = manifest.get('effective_config', {}).get('stages', {}).get('sql_selection', {}).get('shortcut_consistency_score_threshold')
            trace = _selection_trace(selection_events, [e for e in source_events if e['attempt_id'] == source_id], selection_attempt, threshold)
            selected = assess(artifacts['sql_selection'].get('final_selected_sql'))
            shortlist = _pool(trace.pop('shortlist_sqls'), assess)
            row['selection'] = {**trace, 'selected': selected, 'pool': revised, 'shortlist': shortlist,
                'shortlist_retains_correct': shortlist['has_correct'] if revised['has_correct'] is True else None,
                'selected_when_pool_correct': selected['bag_equal'] if revised['has_correct'] is True else None,
                'conditional_eligible': revised['has_correct']}
            if target in ('sql_generation', 'sql_revision'):
                pool = generation if target == 'sql_generation' else revised
                row.update(bag_equal=pool['has_correct'], ordered_equal=pool.get('ordered_has_correct'))
            elif target == 'sql_selection':
                row.update(bag_equal=selected['bag_equal'], ordered_equal=selected['ordered_equal'])
            # Downstream result exists only when explicitly present in the new run.
            row['downstream_final'] = selected
            store.finish_attempt(aid, 'succeeded', row)
            items[key] = row
    return {'format': EVALUATION_VERSION, 'items': items, 'summary': _summary(items)}


def compare_evaluations(left_dir: Path, right_dir: Path, output_dir: Path) -> dict:
    left, la, _ = _read_run(left_dir)
    right, ra, _ = _read_run(right_dir)
    if left.get('format') != EVALUATION_VERSION or right.get('format') != EVALUATION_VERSION:
        raise ValueError('Expected independent evaluation RunStores')
    def pairing(manifest):
        run = manifest['run_manifest']
        return {'target_stage': run.get('target_stage', 'sql_selection'), 'items': run['items'],
                'source_manifest_fingerprint': run.get('source_manifest_fingerprint'),
                'effective_config': run.get('effective_config'), 'repeat_id': run.get('repeat_id'),
                'sources': run.get('sources'),
                'source_checkpoints': run.get('source_checkpoints'),
                'continue_downstream': run.get('continue_downstream', False),
                'upstream': {key: value.get('upstream_state_sha256') for key, value in run.get('source_checkpoints', {}).items()},
                'database_version': manifest['database_version'], 'executor': manifest['executor'],
                'database_execution': manifest.get('database_execution'),
                'reference_sources': manifest['reference_sources'], 'comparison': manifest['comparison']}
    if pairing(left) != pairing(right):
        raise ValueError('Paired evaluations have incompatible source, input, database, budget or reference identities')
    lrows = {a['item_key']: a['payload'] for a in la if a['stage'] == 'evaluation' and a['status'] == 'succeeded'}
    rrows = {a['item_key']: a['payload'] for a in ra if a['stage'] == 'evaluation' and a['status'] == 'succeeded'}
    items = {}
    for binding in left['run_manifest']['items']:
        key = binding['task_key']
        l, r = lrows.get(key, {}), rrows.get(key, {})
        items[key] = {'left_bag_equal': l.get('bag_equal'), 'right_bag_equal': r.get('bag_equal'),
                      'change': _transition(l.get('bag_equal'), r.get('bag_equal')),
                      'ordered_change': _transition(l.get('ordered_equal'), r.get('ordered_equal'))}
    report = {'items': items, 'summary': {'tasks': len(items), **{status: sum(row['change'] == status for row in items.values())
              for status in ('repair', 'harm', 'unchanged_correct', 'unchanged_incorrect', 'unknown')}}}
    report['summary']['ordered'] = {status: sum(row['ordered_change'] == status for row in items.values())
                                   for status in ('repair', 'harm', 'unchanged_correct', 'unchanged_incorrect', 'unknown')}
    identity = {'format': 'deepeye-paired-evaluation-v1', 'left': _hash(left), 'right': _hash(right),
                'left_snapshot': _hash(la), 'right_snapshot': _hash(ra), 'pairing': pairing(left)}
    with RunStore.create(Path(output_dir), identity) as store:
        attempt = store.begin_attempt('paired', 'comparison', _hash(identity))
        store.finish_attempt(attempt, 'succeeded', report)
    return report
