"""Read-only Schema Linking analysis; no model/SQL execution or runtime edits.

Run with python -m scripts.analysis.deepeye.schema_linking_metrics.
This is the conservative SQL-parser diagnostic; use scripts.analysis.column_recall
for the final annotation-based column macro recall.
Outputs a new analysis artifact without overwriting the previous summaries.
"""
from collections import Counter
import hashlib
import json
import math
import statistics

from .analyze import GROUPS, HERE, write_new, restore_jsonable
from .summarize import normalized_coverage
from .paths import analysis_directory
import argparse
from datetime import date


def sets(schema, sqlite):
    norm = str.lower if sqlite else str
    return {
        'table': {norm(t) for t in schema},
        'column': {(norm(t), norm(c)) for t, cols in schema.items() for c in cols},
    }


def score(gold, predicted):
    assert gold, 'Zero reference elements are undefined for this recall metric'
    tp = len(gold & predicted)
    return {
        'tp': tp, 'fp': len(predicted - gold), 'fn': len(gold - predicted),
        'predicted_count': len(predicted), 'reference_count': len(gold),
        'precision': tp / len(predicted) if predicted else 0.0,
        'recall': tp / len(gold),
        'f1': 2 * tp / (len(predicted) + len(gold)),
        'exact_match': gold == predicted, 'full_recall': gold <= predicted,
    }


def aggregate(rows, level):
    valid = [r for r in rows if r[level]['status'] == 'available']
    out = {'n': len(valid), 'excluded': dict(Counter(
        r[level]['reason'] for r in rows if r[level]['status'] != 'available'))}
    for side in ('native', 'rc'):
        vals = [r[level][side] for r in valid]
        counts = {f: sum(v[f] for v in vals) for f in
                  ('tp', 'fp', 'fn', 'predicted_count', 'reference_count')}
        tp, p, g = (counts[f] for f in ('tp', 'predicted_count', 'reference_count'))
        out[side] = {
            'counts': counts,
            'micro': {'precision': tp / p if p else 0.0, 'recall': tp / g,
                      'f1': 2 * tp / (p + g)},
            'macro': {f: statistics.mean(v[f] for v in vals)
                      for f in ('precision', 'recall', 'f1')},
            'exact_match_questions': sum(v['exact_match'] for v in vals),
            'full_recall_questions': sum(v['full_recall'] for v in vals),
        }
    out['change_pp'] = {avg: {f: 100 * (out['rc'][avg][f] - out['native'][avg][f])
                            for f in ('precision', 'recall', 'f1')}
                        for avg in ('micro', 'macro')}
    out['paired_directions'] = {f: dict(Counter(
        'improved' if r[level]['rc'][f] > r[level]['native'][f] else
        'worsened' if r[level]['rc'][f] < r[level]['native'][f] else 'unchanged'
        for r in valid)) for f in ('precision', 'recall', 'exact_match', 'full_recall')}
    return out


def analyze_group(group):
    raw = (HERE / group / 'offline.json').read_bytes()
    data = json.loads(raw)
    sqlite = data['bindings'][0]['db_type'] == 'sqlite'
    records = {(r['condition'], r['item_key']): r for r in data['records']
               if r['stage'] == 'schema_linking'}
    assert len(records) == 2 * GROUPS[group]
    rows = []
    for key, reference in sorted(data['references'].items()):
        assert reference['status'] == 'available'
        schema = {t: list(v['columns']) for t, v in
                  data['inputs'][key]['database_schema']['tables'].items()}
        universe = sets(schema, sqlite)
        cov = normalized_coverage(reference['sql'], {}, sqlite)
        row = {'item_key': key}
        for level in ('table', 'column'):
            result = row[level] = {'status': 'excluded', 'reason': cov['reason']}
            if cov[level + '_coverage'] is None:
                continue
            gold = (set(cov['reference_tables']) if level == 'table' else
                    set(map(tuple, cov['reference_columns'])))
            if not gold:
                result['reason'] = 'zero_reference_elements'
                continue
            if not gold <= universe[level]:
                result.update(reason='reference_elements_not_in_frozen_schema',
                              unresolved_elements=sorted(gold - universe[level]))
                continue
            result.update(status='available', reason=None, reference=sorted(gold))
            for side in ('native', 'rc'):
                record = records[side, key]
                predicted = sets(record['linked'], sqlite)[level]
                assert predicted <= universe[level], (key, side, predicted - universe[level])
                result[side] = {**score(gold, predicted), 'predicted': sorted(predicted)}
                old = normalized_coverage(reference['sql'], record['linked'], sqlite)
                assert math.isclose(result[side]['recall'], old[level + '_coverage'])
        rows.append(row)
    assert len(rows) == GROUPS[group]
    summary = {level: aggregate(rows, level) for level in ('table', 'column')}
    return {'group': group, 'total_questions': len(rows),
            'source_offline_sha256': hashlib.sha256(raw).hexdigest(),
            'summary': summary, 'items': rows}


def verify(result):
    """Recompute with element membership lists, independently of score()."""
    pairs = 0
    for group in result['groups'].values():
        for level in ('table', 'column'):
            included = [r[level] for r in group['items'] if r[level]['status'] == 'available']
            assert len(included) == group['summary'][level]['n']
            for side in ('native', 'rc'):
                totals = Counter()
                for r in included:
                    freeze = lambda values: [tuple(v) if isinstance(v, list) else v for v in values]
                    gold, pred = freeze(r['reference']), freeze(r[side]['predicted'])
                    tp = sum(p in gold for p in pred)
                    fp = sum(p not in gold for p in pred)
                    fn = sum(g not in pred for g in gold)
                    assert (tp, fp, fn) == tuple(r[side][k] for k in ('tp', 'fp', 'fn'))
                    assert len(pred) == len(set(pred)) and len(gold) == len(set(gold))
                    totals.update(tp=tp, fp=fp, fn=fn,
                                  predicted_count=len(pred), reference_count=len(gold))
                    pairs += 1
                assert dict(totals) == group['summary'][level][side]['counts']
                s = group['summary'][level][side]
                assert math.isclose(s['micro']['precision'], totals['tp'] / totals['predicted_count'])
                assert math.isclose(s['micro']['recall'], totals['tp'] / totals['reference_count'])
    return {'verified_question_level_condition_scores': pairs}


def self_check():
    a = score({'a', 'b', 'c'}, {'a', 'b', 'd', 'e'})
    assert a['precision'] == .5 and a['recall'] == 2/3
    assert (a['tp'], a['fp'], a['fn']) == (2, 2, 1)
    assert score({'a'}, set())['precision'] == 0
    assert score({'a'}, {'a'})['exact_match']
    assert not score({'a'}, {'a', 'b'})['exact_match']
    # Regression evidence: the old syntactic parser mistakes SQLite DQS for a column;
    # catalog validation must reject that whole column-reference record, not score it.
    cov = normalized_coverage('SELECT Name FROM singer WHERE Citizenship != "France"', {}, True)
    actual = sets({'singer': ['Name', 'Citizenship']}, True)['column']
    assert ('singer', 'france') in set(map(tuple, cov['reference_columns'])) - actual
    # A PostgreSQL lateral function result is not a physical source column either.
    sql = "SELECT AVG((v->>'vibration_mmps')::numeric) FROM joint_condition, LATERAL jsonb_each(joint_health) AS j(k,v)"
    cov = normalized_coverage(sql, {}, False)
    actual = sets({'joint_condition': ['joint_health']}, False)['column']
    assert ('joint_condition', 'v') in set(map(tuple, cov['reference_columns'])) - actual


if __name__ == '__main__':
    HERE = analysis_directory(argparse.ArgumentParser(description=__doc__))
    self_check()
    result = {
        'version': 'schema-linking-reference-precision-recall-conservative-v1',
        'date': date.today().isoformat(),
        'definitions': {
            'reference': 'Unique physical table/column identifiers used in the saved reference SQL; conservative parser plus frozen-schema membership validation.',
            'prediction': 'Actual post-Schema-Linking schema passed downstream, not raw LLM mentions.',
            'precision': '|predicted intersection reference| / |predicted|; 0 for empty prediction with positive reference.',
            'recall': '|predicted intersection reference| / |reference|; zero-size reference excluded.',
            'micro': 'Sum intersection/reference/prediction counts across questions before dividing.',
            'macro': 'Compute each question metric then take its arithmetic mean.',
            'accuracy': 'Not reported as TN-based accuracy; exact-set match is separately counted.',
            'limitations': 'Gold-implementation consistency, not unique semantic necessity or downstream SQL accuracy; excludes unresolved lineage, wildcard, ambiguous names, noncatalog identifiers and zero-column extraction. No full-benchmark extrapolation.',
        },
        'groups': {},
    }
    for group in GROUPS:
        result['groups'][group] = analyze_group(group)
        print(group, json.dumps(result['groups'][group]['summary'], ensure_ascii=False), flush=True)
    result['verification'] = verify(result)
    destination = HERE / 'schema_linking_metrics.json'
    write_new(destination, result)
    reread = restore_jsonable(json.loads(destination.read_text()))
    assert verify(reread) == result['verification']
    print('Verified and saved:', destination, result['verification'], flush=True)
