"""Generation accuracy changes from saved candidate assessments only.

No inference, query re-execution, runtime edits, or source-record mutation.
Output is exclusively created; reruns never overwrite an existing analysis.
"""
from collections import Counter
from fractions import Fraction
import json

from .analyze import HERE, GROUPS, checked, read_db, write_new
from .paths import analysis_directory
import argparse
from datetime import date


def pool_stats(pool):
    assert pool, 'A zero-size pool requires a separately defined rate'
    counts = Counter('match' if p['bag_equal'] is True else
                     'mismatch' if p['bag_equal'] is False else 'unknown' for p in pool)
    n = len(pool)
    return {'candidates': n, **{k: counts[k] for k in ('match', 'mismatch', 'unknown')},
            'confirmed_match_rate': counts['match'] / n,
            'state': 'has_match' if counts['match'] else 'unknown' if counts['unknown'] else 'no_match'}


def direction(a, b):
    x, y = Fraction(a['match'], a['candidates']), Fraction(b['match'], b['candidates'])
    return 'increase' if y > x else 'decrease' if y < x else 'unchanged'


def summarize(rows):
    n = len(rows)
    out = {'questions': n, 'rate_directions': dict(Counter(r['rate_direction'] for r in rows)),
           'equal_candidate_count_questions': sum(r['native']['candidates'] == r['rc']['candidates'] for r in rows),
           'unknown_candidate_questions': sum(r['native']['unknown'] > 0 or r['rc']['unknown'] > 0 for r in rows)}
    for side in ('native', 'rc'):
        totals = {k: sum(r[side][k] for r in rows) for k in ('candidates', 'match', 'mismatch', 'unknown')}
        assert totals['candidates'] == totals['match'] + totals['mismatch'] + totals['unknown']
        out[side] = {**totals,
                     'confirmed_match_rate': totals['match'] / totals['candidates'] if totals['candidates'] else None,
                     'mean_question_match_rate': sum(r[side]['confirmed_match_rate'] for r in rows) / n if n else None,
                     'matching_questions': sum(r[side]['match'] > 0 for r in rows)}
    out['rate_change_pp'] = 100 * (out['rc']['confirmed_match_rate'] - out['native']['confirmed_match_rate']) if n else None
    transitions = Counter((r['native']['state'], r['rc']['state']) for r in rows)
    out['three_state_transitions'] = [{'native': a, 'rc': b, 'questions': v} for (a, b), v in sorted(transitions.items())]
    out['known_match_coverage'] = {
        'gain': sum(r['native']['match'] == 0 and r['rc']['match'] > 0 for r in rows),
        'loss': sum(r['native']['match'] > 0 and r['rc']['match'] == 0 for r in rows),
        'both_have_match': sum(r['native']['match'] > 0 and r['rc']['match'] > 0 for r in rows),
        'neither_has_confirmed_match': sum(r['native']['match'] == 0 and r['rc']['match'] == 0 for r in rows),
        'strict_no_match_to_match': transitions['no_match', 'has_match'],
        'unknown_to_match': transitions['unknown', 'has_match'],
        'strict_match_to_no_match': transitions['has_match', 'no_match'],
        'match_to_unknown': transitions['has_match', 'unknown'],
    }
    assert sum(out['rate_directions'].values()) == n
    c = out['known_match_coverage']
    assert c['gain'] - c['loss'] == out['rc']['matching_questions'] - out['native']['matching_questions']
    assert sum(c[k] for k in ('gain', 'loss', 'both_have_match', 'neither_has_confirmed_match')) == n
    return out


def analyze_group(group):
    rows = []
    with read_db(HERE / group / 'evaluation.sqlite3') as db:
        assert db.execute('pragma quick_check').fetchone()[0] == 'ok'
        for saved in db.execute('select payload_json,sha256 from items order by task_key'):
            item = checked(saved['payload_json'], saved['sha256'])
            assert item['reference_success'] is True
            a, b = (pool_stats(item['targets'][s + '/sql_generation']) for s in ('native', 'rc'))
            d = direction(a, b)
            # Independent integer cross-product check (no rounded percentages).
            cross = b['match'] * a['candidates'] - a['match'] * b['candidates']
            assert d == ('increase' if cross > 0 else 'decrease' if cross < 0 else 'unchanged')
            rows.append({'item_key': item['item_key'], 'source_payload_sha256': saved['sha256'],
                         'native': a, 'rc': b, 'rate_direction': d})
        assert len(rows) == len({r['item_key'] for r in rows}) == GROUPS[group]
        result = {'all': summarize(rows),
                  'both_12_candidates': summarize([r for r in rows if r['native']['candidates'] == r['rc']['candidates'] == 12]),
                  'all_candidates_comparable': summarize([r for r in rows if r['native']['unknown'] == r['rc']['unknown'] == 0]),
                  'items': rows}
        # Independently aggregate the persisted JSON via SQLite's JSON operations.
        for side in ('native', 'rc'):
            path = '$.targets."' + side + '/sql_generation"'
            check = db.execute('''select count(*) candidates,
                sum(json_extract(j.value, '$.bag_equal') = 1) matches,
                sum(json_extract(j.value, '$.bag_equal') = 0) mismatches,
                sum(json_extract(j.value, '$.bag_equal') is null) unknowns
                from items i, json_each(i.payload_json, ?) j''', (path,)).fetchone()
            assert list(check) == [result['all'][side][k] for k in ('candidates', 'match', 'mismatch', 'unknown')]
    old = json.loads((HERE / group / 'summary.json').read_text())['results']['sql_generation']['all']['bag_equal']
    for side in ('native', 'rc'):
        assert result['all'][side]['candidates'] == old[side]['slots']
        assert result['all'][side]['match'] == old[side]['matching_slots']
        assert result['all'][side]['matching_questions'] == old[side]['matching_questions']
    assert result['all']['known_match_coverage']['gain'] == len(old['paired']['new_matching_keys'])
    assert result['all']['known_match_coverage']['loss'] == len(old['paired']['lost_matching_keys'])
    return result


if __name__ == '__main__':
    HERE = analysis_directory(argparse.ArgumentParser(description=__doc__))
    assert direction({'match': 1, 'candidates': 3}, {'match': 2, 'candidates': 6}) == 'unchanged'
    assert direction({'match': 1, 'candidates': 4}, {'match': 1, 'candidates': 3}) == 'increase'
    result = {
        'date': date.today().isoformat(),
        'definition': 'Confirmed bag-equality matches divided by all actual candidate slots. Unknown comparisons remain in denominator but are not labeled proven mismatches. Repeated SQL counts once per sampling slot; missing samples are not invented.',
        'groups': {},
    }
    for group in GROUPS:
        result['groups'][group] = analyze_group(group)
        print(group, json.dumps(result['groups'][group]['all'], ensure_ascii=False), flush=True)
    rows = [r for group in result['groups'].values() for r in group['items']]
    result['pooled_all_five'] = summarize(rows)
    result['pooled_bird_interact'] = summarize([r for g in ('bird_interact_lite', 'bird_interact_full') for r in result['groups'][g]['items']])
    result['verification'] = {'questions_checked': len(rows), 'source_payload_checksums': len(rows),
                              'independent_sql_aggregate_checks': len(GROUPS) * 2,
                              'original_summary_agreement': True}
    destination = HERE / 'generation_metrics.json'
    write_new(destination, result)
    reread = json.loads(destination.read_text())
    assert reread == result
    for group in reread['groups'].values():
        assert summarize(group['items']) == group['all']
    print('Verified and saved:', destination, result['verification'], flush=True)
