"""Compact current-version statistics and separately observed historical costs.

Batch layout: DailRecords root, current.sqlite3, manifest.evaluation_source
(Task4 source.json), manifest.sql_timeout_seconds (default 60). Call synchronous
export_current on a caller-owned worker; it executes at most one bounded SQL at
a time and keeps execution tables only for the current question.
"""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from scripts.baseline_adapters.dail_sql.config import MODES, TaskKey
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from scripts.baseline_adapters.dail_sql.execution import execute_sql
from scripts.baseline_adapters.dail_sql.records import DailRecords, aggregate_observed_usage
from scripts.baseline_adapters.deepeye.run_store import restore_jsonable, to_jsonable
from scripts.rc_evaluation.deepeye import comparison
from .evaluation import evaluate_candidate


def _key(value):
    return value if isinstance(value, TaskKey) else TaskKey(**value)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _vote_summary(event, source_event_id):
    status, error = event.get('status'), event.get('error')
    error_type = error.get('type') if isinstance(error, dict) else None
    classification = ('success' if status == 'success' else
                      'timeout' if status == 'timeout' else
                      error_type if status == 'error' and error_type in ('extraction', 'vote_preprocessing') else
                      'execution_error' if status == 'error' else 'unknown')
    return {'source_event_id': source_event_id, 'status': status,
            'classification': classification, 'error': error}


def _round_summary(round_):
    candidates = []
    samples = {s['successful_request_id']: s for s in round_['samples'] if s.get('successful_request_id')}
    for candidate in round_['candidates']:
        evaluation = candidate.get('evaluation', {})
        candidates.append({name: candidate.get(name) for name in
                           ('candidate_id', 'choice_position', 'source_request_id', 'candidate_sql',
                            'vote_execution_ref')} | {
            'evaluation': evaluation,
            'vote_execution_summary': candidate.get('vote_execution_summary',
                _vote_summary({}, candidate.get('vote_execution_ref'))),
            'success_usage': samples[candidate['source_request_id']]['success_usage']})
    matched = sum(c['evaluation'].get('bag_equal') is True for c in candidates)
    evaluable = sum(c['evaluation'].get('bag_equal') is not None for c in candidates)
    selection = round_.get('selection') or {}
    selected = next((c for c in candidates if c['candidate_id'] == selection.get('candidate_id')), None)
    complete = round_['status'] == 'success' and len(candidates) == 5 and evaluable == 5
    return {name: round_.get(name) for name in (
        'round_execution_id', 'round_no', 'status', 'rc_injected', 'actual_parent_round_id',
        'example_ids', 'next_example_ids', 'successful_request_ids', 'request_attempt_ids', 'success_usage', 'error',
        'round_event_id')} | {
        'candidates': candidates, 'candidate_count': len(candidates), 'matched': matched,
        'evaluable': evaluable, 'unknown': len(candidates) - evaluable,
        'observed_match_ratio': _ratio(matched, evaluable), 'complete': complete,
        'covered': True if matched else False if complete else None,
        'execution_errors': sum(c['evaluation'].get('status') == 'prediction_error' for c in candidates),
        'reference_errors': sum(c['evaluation'].get('status') == 'reference_error' for c in candidates),
        'vote_execution_errors': sum(c['vote_execution_summary']['status'] in ('error', 'timeout') for c in candidates),
        'vote_execution_unknown': sum(c['vote_execution_summary']['classification'] == 'unknown' for c in candidates),
        'selection': {name: selection.get(name) for name in ('candidate_id', 'selection_ref', 'tie', 'fallback')},
        'selected_match': selected['evaluation'].get('bag_equal') if selected else None}


def expand_modes(version: dict, rounds_by_id: dict) -> list[dict]:
    """Hydrated sealed version + original rounds with candidate.evaluation -> four rows.

    Evaluation and compact vote_execution_summary are optional (missing evidence remains unknown). Source rounds
    are compacted once, then shared by mode rows. No requests or votes run here.
    """
    compact = {rid: _round_summary(value) for rid, value in rounds_by_id.items()}
    rows = []
    for mode in MODES:
        value = version['modes'][mode]
        rounds = [compact.get(value[f'{position}_round_id']) for position in ('first', 'second')]
        first, second = rounds
        final = next((c for c in (second['candidates'] if second else [])
                      if c['candidate_id'] == value['final_candidate_id']), None)
        rows.append({'task_key': version['task_key'], 'version_id': version['version_id'],
                     'mode_event_id': version.get('mode_event_ids', {}).get(mode), **value,
                     'rounds': rounds, 'final_candidate_sql': final['candidate_sql'] if final else None,
                     'final_match': final['evaluation'].get('bag_equal') if final else None,
                     'final_evaluation_status': final['evaluation'].get('status') if final else None,
                     'success_usage': aggregate_observed_usage([r['success_usage'] if r else None for r in rounds]),
                     'successful_request_ids': [rid for r in rounds if r for rid in r['successful_request_ids']],
                     'second_round_reused': bool(second and second['actual_parent_round_id'] != value['first_round_id']),
                     'examples_changed_between_rounds': (first['example_ids'] != second['example_ids'])
                                                        if first and second else None})
    return rows


def _pending(key, mode):
    return {'task_key': asdict(key), 'version_id': None, 'mode': mode, 'status': 'pending',
            'rounds': [None, None], 'final_match': None, 'success_usage': None,
            'successful_request_ids': [], 'second_round_reused': False}


def aggregate(rows: list[dict], expected_keys: list[TaskKey]) -> dict:
    """Seven tables with full manifest denominators and same-question pairing.

    Ratios over observed/evaluable outputs explicitly name their denominator;
    coverage and final-match ratios use all expected questions. Paired candidate
    changes require two complete five-candidate sets. Unknown never means false.
    """
    keys = [_key(key) for key in expected_keys]
    key_set = set(keys)
    if len(key_set) != len(keys):
        raise ValueError('duplicate expected key')
    positions = {}
    for row in rows:
        identity = (_key(row['task_key']), row['mode'])
        if identity[0] not in key_set or identity[1] not in MODES or identity in positions:
            raise ValueError('foreign or duplicate mode row')
        positions[identity] = row
    tables = {name: [] for name in ('candidate_matches', 'candidate_changes', 'coverage',
                                     'final_outputs', 'voting', 'examples', 'usage')}
    for group in dict.fromkeys(key.group for key in keys):
        members = [key for key in keys if key.group == group]
        for mode in MODES:
            current = [positions.get((key, mode), _pending(key, mode)) for key in members]
            native = [positions.get((key, 'native'), _pending(key, 'native')) for key in members]
            base = {'group': group, 'mode': mode, 'total': len(members)}
            final = [row.get('final_match') for row in current]
            matched = sum(value is True for value in final)
            evaluable = sum(value is not None for value in final)
            transition = Counter((n.get('final_match'), r.get('final_match')) for n, r in zip(native, current))
            tables['final_outputs'].append({**base, 'matched': matched, 'evaluable': evaluable,
                'unknown': len(members) - evaluable, 'missing': sum(r['status'] == 'pending' for r in current),
                'failed': sum(r['status'] in ('failed', 'dependency_failed') for r in current),
                'prediction_errors': sum(r.get('final_evaluation_status') == 'prediction_error' for r in current),
                'reference_errors': sum(r.get('final_evaluation_status') == 'reference_error' for r in current),
                'incomparable': sum(r.get('final_evaluation_status') == 'unknown' for r in current),
                'match_ratio': _ratio(matched, len(members)), 'observed_match_ratio': _ratio(matched, evaluable),
                'wrong_to_right': transition[(False, True)], 'right_to_wrong': transition[(True, False)],
                'paired_unknown': sum(a is None or b is None for a, b in
                                      ((n.get('final_match'), r.get('final_match')) for n,r in zip(native,current)))})
            for number in (1, 2):
                selected = [r['rounds'][number - 1] for r in current]
                controls = [r['rounds'][number - 1] for r in native]
                existing = [r for r in selected if r is not None]
                counts = {field: sum(r[field] for r in existing) for field in
                          ('candidate_count', 'matched', 'evaluable', 'unknown', 'execution_errors', 'reference_errors')}
                tables['candidate_matches'].append({**base, 'round_no': number, **counts,
                    'missing_rounds': len(selected) - len(existing),
                    'failed_rounds': sum(r['status'] != 'success' for r in existing),
                    'observed_match_ratio': _ratio(counts['matched'], counts['evaluable'])})
                changes = Counter()
                coverage_changes = Counter()
                for control, round_ in zip(controls, selected):
                    if not control or not round_ or not control['complete'] or not round_['complete']:
                        changes['incomplete'] += 1
                    else:
                        delta = round_['observed_match_ratio'] - control['observed_match_ratio']
                        changes['increased' if delta > 0 else 'decreased' if delta < 0 else 'unchanged'] += 1
                    coverage_changes[(control['covered'] if control else None, round_['covered'] if round_ else None)] += 1
                tables['candidate_changes'].append({**base, 'round_no': number,
                    **{field: changes[field] for field in ('increased','decreased','unchanged','incomplete')}})
                covered = sum(r['covered'] is True for r in existing)
                unknown = len(selected) - sum(r['covered'] is not None for r in existing)
                tables['coverage'].append({**base, 'round_no': number, 'covered': covered,
                    'unknown': unknown, 'evaluable': len(selected) - unknown,
                    'coverage_ratio': _ratio(covered, len(selected)),
                    'none_to_some': coverage_changes[(False, True)], 'some_to_none': coverage_changes[(True, False)],
                    'paired_unknown': sum(v for (a,b),v in coverage_changes.items() if a is None or b is None)})
                correct_available = [r for r in existing if r['covered'] is True]
                tables['voting'].append({**base, 'round_no': number, 'correct_available': len(correct_available),
                    'selected_correct': sum(r['selected_match'] is True for r in correct_available),
                    'selected_unknown': sum(r['selected_match'] is None for r in correct_available),
                    'selection_ratio': _ratio(sum(r['selected_match'] is True for r in correct_available), len(correct_available)),
                    'ties': sum(r['selection']['tie'] is True for r in existing),
                    'fallbacks': sum(r['selection']['fallback'] is True for r in existing),
                    'selection_flags_unknown': sum(r['selection']['tie'] is None or r['selection']['fallback'] is None for r in existing),
                    'all_incorrect': sum(r['covered'] is False for r in existing),
                    'execution_errors': sum(r['vote_execution_errors'] for r in existing),
                    'execution_unknown': sum(r['vote_execution_unknown'] for r in existing)})
            examples = {**base, **{field: 0 for field in ('first_same', 'first_different', 'first_unknown',
                         'second_changed', 'second_unchanged', 'second_unknown', 'reused', 'reused_correct', 'reused_unknown',
                         'changed_correct', 'changed_unknown', 'unchanged_correct', 'unchanged_unknown')}}
            for row, control in zip(current, native):
                for index, label in ((0, 'first'), (1, 'second')):
                    a, b = row['rounds'][index], control['rounds'][index]
                    suffix = 'unknown' if not a or not b else ('same' if label == 'first' else 'unchanged') if a['example_ids'] == b['example_ids'] else ('different' if label == 'first' else 'changed')
                    examples[label + '_' + suffix] += 1
                    if label == 'second' and suffix != 'unknown':
                        examples[suffix + '_correct'] += row.get('final_match') is True
                        examples[suffix + '_unknown'] += row.get('final_match') is None
                if row.get('second_round_reused'):
                    examples['reused'] += 1
                    examples['reused_correct'] += row.get('final_match') is True
                    examples['reused_unknown'] += row.get('final_match') is None
            tables['examples'].append(examples)
            tables['usage'].append({**base, 'success_usage': aggregate_observed_usage([r['success_usage'] for r in current]),
                'complete_modes': sum(r['status'] == 'succeeded' for r in current),
                'rounds': [usage_summary([r['rounds'][i]['success_usage'] if r['rounds'][i] else None for r in current]) for i in (0,1)],
                'accounting': 'logical source usage; shared requests appear in each consuming mode'})
    return {'expected_questions': len(keys), 'expected_mode_positions': len(keys) * 4,
            'supplied_mode_positions': len(rows), **tables}


def _numeric_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _numeric_leaves(child, (*path, key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield path, value


class _UsageTotals:
    """Streaming raw-usage reductions; does not retain historical payloads."""
    def __init__(self):
        self.count, self.unknown = 0, 0
        self.known, self.usage = {}, None

    def add(self, value):
        self.usage = aggregate_observed_usage([self.usage, value] if self.count else [value])
        self.count += 1
        total = value.get('total_tokens') if isinstance(value, dict) else None
        self.unknown += (not isinstance(total, (int, float)) or isinstance(total, bool) or not math.isfinite(total))
        for path, number in _numeric_leaves(value):
            target = self.known
            for field in path[:-1]:
                target = target.setdefault(field, {})
            target[path[-1]] = target.get(path[-1], 0) + number

    def result(self):
        return {'usage': self.usage, 'known_usage': self.known or None,
                'unknown_attempt_count': self.unknown, 'source_count': self.count}


def usage_summary(values):
    """Strict aggregate plus known numeric subtotals, never fill unknowns with zero."""
    totals = _UsageTotals()
    for value in values:
        totals.add(value)
    return totals.result()


def iter_actual_requests(records: DailRecords):
    """Stream compact attempts across all versions, including failed/paused/old.

    Each version's request events have one bounded RunStore snapshot. Version
    lists are observed per group; this is not a global instantaneous cost view.
    Only one version's compact attempt ledger is retained at a time.
    """
    for version in records.iter_versions():
        attempts = {}
        for event in records.iter_events(version['version_id'], ('request_attempt', 'request_result')):
            value = event['payload']
            if event['kind'] == 'request_attempt':
                attempts[event['event_id']] = {**version, 'request_attempt_id': event['event_id'],
                    'round_execution_id': value['round_execution_id'], 'sample_position': value['sample_position'],
                    'request_attempt_no': value['attempt_no'], 'request_result_id': None,
                    'request_status': 'unfinished', 'usage': None}
            else:
                attempt = attempts[value['request_attempt_id']]
                if attempt['request_result_id'] is not None:
                    raise ValueError('duplicate request result')
                attempt.update(request_result_id=event['event_id'], request_status=value['status'], usage=value.get('usage'))
        yield from attempts.values()


def aggregate_actual_requests(requests):
    """Deduplicate actual attempt IDs; response completion already includes reasoning."""
    seen = {}
    all_usage, successful, failed = _UsageTotals(), _UsageTotals(), _UsageTotals()
    for request in requests:
        identity = request['request_attempt_id']
        signature = hashlib.sha256(json.dumps(to_jsonable([request['request_status'], request['usage']]), sort_keys=True).encode()).digest()
        if identity in seen:
            if seen[identity] != signature:
                raise ValueError('conflicting observations for actual request')
            continue
        seen[identity] = signature
        all_usage.add(request['usage'])
        (successful if request['request_status'] == 'success' else failed).add(request['usage'])
    return {'attempt_count': len(seen), **all_usage.result(),
            'successful': successful.result(), 'failed_or_unfinished': failed.result(),
            'accounting': 'all observed versions; actual attempt IDs deduplicated; reasoning is within completion'}


def _read_json(path):
    return restore_jsonable(json.loads(Path(path).read_text()))


def _write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(to_jsonable(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _write_line(stream, value):
    stream.write(json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def _expected(manifest):
    return [TaskKey(manifest['batch_id'], group, qid) for group, spec in manifest['groups'].items() for qid in spec['ids']]


def _versions(keys, snapshot):
    return [{'task_key': asdict(key), 'version_id': snapshot.get(key)} for key in keys]


def _bindings(batch_root, manifest):
    source_path = Path(manifest['evaluation_source'])
    if not source_path.is_absolute():
        source_path = batch_root / source_path
    source = _read_json(source_path)
    if ('input_manifest' in source) == ('bindings' in source):
        raise ValueError('evaluation source requires exactly one binding source')
    path = Path(source.get('input_manifest', source.get('bindings')))
    if not path.is_absolute():
        path = source_path.parent.parent / path  # Task4 refs are preparation-root-relative.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('evaluation source checksum mismatch')
    value = restore_jsonable(json.loads(raw))
    if 'input_manifest' in source:
        value = {group: {row['question_id']: row['evaluation_binding'] for row in spec['rows']}
                 for group, spec in value['groups'].items()}
    return value, {'source_path': str(source_path.resolve()), **source}


def current_export(batch_root: Path) -> Path | None:
    """Resolve latest full publication only if its version map is still current.

    A latest.json pointer alone is never a freshness assertion. Call this for
    status/UI consumers; the result is current as of this index snapshot.
    """
    root = Path(batch_root)
    latest = root / 'exports/full/latest.json'
    if not latest.exists():
        return None
    pointer = _read_json(latest)
    name = pointer['directory']
    if Path(name).name != name:
        raise ValueError('invalid export directory')
    directory = latest.parent / name
    saved = _read_json(directory / 'versions.json')
    manifest = _read_json(root / 'manifest.json')
    with CurrentIndex(root / 'current.sqlite3', read_only=True) as index:
        now = _versions(_expected(manifest), index.snapshot())
    return directory if saved['versions'] == now and saved['manifest_sha256'] == hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() else None


def export_current(batch_root: Path, *, diagnostic_targets: list[TaskKey] | None = None) -> Path:
    """Atomically publish a derived directory from one immutable mode snapshot.

    Files: versions.json, summary.json, modes.jsonl, rounds.jsonl,
    candidates.jsonl, executions.jsonl, requests.jsonl, failed_questions.jsonl.
    The failed-question list contains unique current terminal failures only.
    Full/diagnostic trees never overwrite each
    other. Full latest.json is published last; current_export validates freshness.
    Historical requests have sequential group/version observation windows,
    described in summary.json, and are deliberately not current-only costs.
    """
    root = Path(batch_root)
    manifest = _read_json(root / 'manifest.json')
    keys = _expected(manifest)
    if diagnostic_targets is not None:
        targets = [_key(key) for key in diagnostic_targets]
        if not targets or len(set(targets)) != len(targets) or set(targets) - set(keys):
            raise ValueError('diagnostic targets must be unique manifest members')
        keys = [key for key in keys if key in set(targets)]
    bindings, source = _bindings(root, manifest)
    timeout = manifest.get('sql_timeout_seconds', 60)
    if not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('SQL timeout must be positive and finite')
    with CurrentIndex(root / 'current.sqlite3', read_only=True) as index:
        snapshot = index.snapshot()
    version_list = _versions(keys, snapshot)
    parent = root / 'exports' / ('full' if diagnostic_targets is None else 'diagnostic')
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.staging-', dir=parent))
    evaluation_started = datetime.now(timezone.utc).isoformat()
    rows = []
    with DailRecords(root, manifest, read_only=True) as records:
        with (staging / 'rounds.jsonl').open('w') as round_file, (staging / 'candidates.jsonl').open('w') as candidate_file, (staging / 'executions.jsonl').open('w') as execution_file:
            for key in keys:
                version_id = snapshot.get(key)
                if version_id is None:
                    rows.extend(_pending(key, mode) for mode in MODES)
                    continue
                if not records.is_sealed(key, version_id):
                    raise ValueError('current version must be sealed for its manifest key')
                version = records.get_version(version_id)
                rounds, cache = {}, {}  # One question's bounded, fresh execution observation.
                binding = bindings[key.group][key.question_id]
                def execute(database, sql):
                    identity = [asdict(key), version_id, database, binding.get('database_version'), sql]
                    execution_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    result = execute_sql(database, sql, timeout_seconds=timeout)
                    result['execution_id'] = execution_id
                    _write_line(execution_file, {'execution_id': execution_id, 'task_key': asdict(key),
                        'version_id': version_id, 'database': database, 'database_version': binding.get('database_version'),
                        'sql': sql, 'result': result})
                    return result
                for rid in version['round_ids']:
                    event = records.find_source(version_id, 'round_result', rid)
                    round_ = event['payload']
                    round_['round_event_id'] = event['event_id']
                    for candidate in round_['candidates']:
                        vote_ref = candidate['vote_execution_ref']
                        candidate['vote_execution_summary'] = _vote_summary(records.get_event(version_id, vote_ref), vote_ref)
                        candidate['evaluation'] = evaluate_candidate(candidate, binding, execute=execute, cache=cache)
                    rounds[rid] = round_
                mode_rows = expand_modes(version, rounds)
                emitted = set()
                for row in mode_rows:
                    for round_ in row['rounds']:
                        if round_ is None or round_['round_execution_id'] in emitted:
                            continue
                        emitted.add(round_['round_execution_id'])
                        provenance = {'task_key': asdict(key), 'version_id': version_id,
                                      'round_execution_id': round_['round_execution_id']}
                        _write_line(round_file, {**provenance, **{k:v for k,v in round_.items() if k != 'candidates'}})
                        for candidate in round_['candidates']:
                            _write_line(candidate_file, {**provenance, **candidate})
                rows.extend(mode_rows)
        evaluation_finished = datetime.now(timezone.utc).isoformat()
        observed_start = evaluation_finished
        with (staging / 'requests.jsonl').open('w') as request_file:
            def observed():
                for request in iter_actual_requests(records):
                    _write_line(request_file, request)
                    yield request
            cost = aggregate_actual_requests(observed())
        summary = aggregate(rows, keys)
        summary['actual_cost'] = cost
        summary['evaluation_observation'] = {'started_at': evaluation_started, 'finished_at': evaluation_finished,
            'boundary': 'SQL queries observe database contents during this export; no global database snapshot',
            'cache': 'one question at a time; exact database identity/version and SQL; historical votes not reused'}
        summary['cost_observation'] = {'started_at': observed_start, 'finished_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'whole batch, including historical and unfinished versions, even for diagnostic exports',
            'boundary': 'sequential per-group version lists and bounded per-version request-event snapshots; not a global transaction'}
    with (staging / 'modes.jsonl').open('w') as stream:
        for row in rows:
            _write_line(stream, row)
    failed_questions = {(_key(row['task_key']).group, _key(row['task_key']).question_id)
                        for row in rows if row['status'] in ('failed', 'dependency_failed')}
    with (staging / 'failed_questions.jsonl').open('w') as stream:
        for group, question_id in sorted(failed_questions):
            _write_line(stream, {'group': group, 'question_id': question_id})
    _write_json(staging / 'summary.json', summary)
    _write_json(staging / 'versions.json', {'format': 'dail-current-export-v1', 'versions': version_list,
        'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        'evaluation_source': source, 'sql_timeout_seconds': timeout,
        'comparison': {'rule': 'compare_results', 'module': comparison.__name__,
                       'source_sha256': hashlib.sha256(Path(comparison.__file__).read_bytes()).hexdigest()},
        'scope': 'full' if diagnostic_targets is None else 'diagnostic'})
    for path in staging.iterdir():
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    staging_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(staging_fd)
    finally:
        os.close(staging_fd)
    destination = parent / uuid.uuid4().hex
    os.replace(staging, destination)
    if diagnostic_targets is None:
        pointer = parent / ('.latest-' + uuid.uuid4().hex)
        _write_json(pointer, {'directory': destination.name})
        os.replace(pointer, parent / 'latest.json')
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination
