"""Read-only Selection audit: fixed pools, model participation, and fair subsets."""
from collections import Counter, defaultdict
import json

from .analyze import HERE, GROUPS, checked, read_db, write_new, restore_jsonable, sha
from .summarize import state, signature, token_pair_summary
from .paths import analysis_directory
import argparse
from datetime import date
from scripts.rc_evaluation.deepeye.injection import manifest_prompt, render_rc_block


def summarize(rows):
    opportunity = [r for r in rows if r['input_has_match'] is True]
    out = {'n': len(rows), 'opportunity_n': len(opportunity)}
    for side in ('native', 'rc'):
        matched = sum(r[side] is True for r in rows)
        assert all(r['input_has_match'] is True for r in rows if r[side] is True)
        out[side] = {'matched': matched, 'unknown': sum(r[side] is None for r in rows),
                     'accuracy': matched / len(rows) if rows else None,
                     'opportunity_accuracy': matched / len(opportunity) if opportunity else None}
    out['gain'] = sum(r['native'] is False and r['rc'] is True for r in rows)
    out['loss'] = sum(r['native'] is True and r['rc'] is False for r in rows)
    out['both_correct'] = sum(r['native'] is True and r['rc'] is True for r in rows)
    out['both_incorrect'] = sum(r['native'] is False and r['rc'] is False for r in rows)
    return out


def failure_class(selected, pool, shortlist, llm):
    if selected is True:
        return 'selected_match'
    if selected is None:
        return 'selected_unresolved'
    if pool is None:
        return 'input_unresolved'
    if pool is False:
        return 'input_no_match'
    if shortlist is None:
        return 'shortlist_unresolved'
    if shortlist is False:
        return 'lost_before_final_choice'
    return 'model_miss' if llm else 'rule_miss'


def shortlist_profile(values):
    if not values:
        return 'empty'
    if any(v is None for v in values):
        return 'unresolved'
    if all(values):
        return 'all_match'
    if not any(values):
        return 'all_mismatch'
    return 'mixed'


def request_without_rc(kwargs, block):
    result = json.loads(json.dumps(kwargs))
    removed = 0
    for m in result.get('messages', []):
        content = m.get('content')
        if isinstance(content, str) and content.endswith('\n\n' + block):
            m['content'] = content[:-len('\n\n' + block)]
            removed += 1
    if removed != 1:
        raise ValueError('Expected one exact appended RC block')
    return result


def audit_sources(data):
    records = {r['attempt_id']: r for r in data['records'] if r['stage'] == 'sql_selection'}
    traces = defaultdict(lambda: {'api': 0, 'request_hashes': set(), 'scores': None,
                                 'shortlist_sqls': None, 'sample_tokens': Counter()})
    proof = Counter()
    for meta in data['manifests']:
        if meta['condition'] == 'rc' and meta['stage'] != 'sql_selection':
            continue
        with read_db(meta['run_dir'] + '/run.sqlite3') as db:
            m = db.execute('select payload_json,payload_checksum from manifest').fetchone()
            manifest = checked(m[0], m[1])
            assert m[1] == meta['manifest_sha256']
            rc = meta['condition'] == 'rc'
            blocks = {}
            if rc:
                assert manifest['rc_version'] == 3 and manifest['gold_corrected']
                prompt = manifest_prompt(manifest)
            query = """select e.attempt_id,e.kind,e.payload_json,e.payload_checksum
                from events e join attempts a using(attempt_id)
                where a.stage='sql_selection' and
                (e.kind in ('api_request','sample_result') or
                (e.kind='component_result' and json_extract(e.payload_json,'$.component')='selection.shortlist'))
                order by e.event_id"""
            for e in db.execute(query):
                assert e['attempt_id'] in records
                row = records[e['attempt_id']]
                p = restore_jsonable(checked(e['payload_json'], e['payload_checksum']))
                proof['source_event_checksums'] += 1
                t = traces[e['attempt_id']]
                if e['kind'] == 'component_result':
                    assert t['scores'] is None
                    t['shortlist_sqls'] = [r[0] for r in p['result']]
                    t['scores'] = [r[2] for r in p['result']]
                elif e['kind'] == 'sample_result':
                    if rc:
                        assert p['rc_applied'] is True
                    if p['succeeded']:
                        assert isinstance(p['usage'], dict)
                        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens'):
                            t['sample_tokens'][field] += p['usage'][field]
                else:
                    kwargs = p['kwargs']
                    if rc:
                        key = row['item_key']
                        if key not in blocks:
                            contract = manifest['contracts'][key]
                            assert contract['rc_version'] == 3 and contract['source_field'] == 'rc_round3'
                            blocks[key] = render_rc_block(contract, prompt_template=prompt)
                        kwargs = request_without_rc(kwargs, blocks[key])
                        proof['actual_rc3_requests_verified'] += 1
                    t['api'] += 1
                    t['request_hashes'].add(sha(json.dumps({'args': p.get('args', []), 'kwargs': kwargs},
                                                         sort_keys=True, ensure_ascii=False)))
    for aid, row in records.items():
        t = traces[aid]
        assert t['api'] == row['api'].get('api_request', 0)
        if t['api']:
            assert t['shortlist_sqls'] == row['selection_trace']['shortlist_sqls']
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            assert t['sample_tokens'][field] == row['effective']['known_tokens'][field]
        assert t['sample_tokens']['reasoning_tokens'] == row['effective']['known_reasoning_tokens']
        proof['token_aggregate_checks'] += 1
    return traces, dict(proof)


def analyze(group):
    raw = (HERE / group / 'offline.json').read_bytes()
    data = json.loads(raw)
    records = {(r['condition'], r['item_key']): r for r in data['records'] if r['stage'] == 'sql_selection'}
    traces, proof = audit_sources(data)
    rows = []
    with read_db(HERE / group / 'evaluation.sqlite3') as db:
        for saved in db.execute('select payload_json,sha256 from items order by task_key'):
            item = checked(saved['payload_json'], saved['sha256'])
            key, targets = item['item_key'], item['targets']
            pool = targets['native/sql_revision']
            pool_sql = {r['sql'] for r in pool}
            a, b = (records[s, key] for s in ('native', 'rc'))
            ta, tb = (traces[r['attempt_id']] for r in (a, b))
            active = tb['api'] > 0
            assert active == ((b['participation'] or {}).get('status') == 'participating')
            if active:
                assert ta['api'] > 0
            row = {'item_key': key, 'source_payload_sha256': saved['sha256'],
                   'active': active, 'input_has_match': state(pool),
                   'same_list_order': a['selection_trace']['shortlist_sqls'] == b['selection_trace']['shortlist_sqls'],
                   'same_non_rc_request': bool(ta['request_hashes']) and len(ta['request_hashes']) == len(tb['request_hashes']) == 1
                                          and ta['request_hashes'] == tb['request_hashes'],
                   'same_choice_weights': ta['scores'] is not None and ta['scores'] == tb['scores']}
            for side, record, trace in (('native', a, ta), ('rc', b, tb)):
                selected = targets[side + '/sql_selection']
                shortlist = targets[side + '/shortlist']
                assert len(selected) == 1
                assert selected[0]['sql'] in pool_sql
                assert all(s['sql'] in pool_sql for s in shortlist)
                assert [s['sql'] for s in shortlist] == record['selection_trace']['shortlist_sqls']
                row[side] = selected[0]['bag_equal']
                row[side + '_shortlist'] = {'profile': shortlist_profile([s['bag_equal'] for s in shortlist]),
                    'has_match': state(shortlist), 'length': len(shortlist),
                    'sql_hashes': [sha(s['sql']) for s in shortlist]}
                row[side + '_branch'] = record['selection_trace']['branch']
                llm = trace['api'] > 0 if side == 'native' or active else row[side + '_branch'] == 'pairwise_comparison'
                assert row[side + '_branch'] != 'unknown'
                row[side + '_failure_class'] = failure_class(row[side], row['input_has_match'], state(shortlist), llm)
                row[side + '_api_requests'] = trace['api']
                row[side + '_non_rc_request_hashes'] = sorted(trace['request_hashes'])
                row[side + '_choice_weights'] = trace['scores']
                row[side + '_attempt_id'] = record['attempt_id']
            assert row['native'] is not None and row['rc'] is not None and row['input_has_match'] is not None
            if not active:
                assert a['sqls'] == b['sqls']
            if row['same_list_order']:
                assert row['native_shortlist'] == row['rc_shortlist']
            rows.append(row)
    assert len(rows) == GROUPS[group]
    subsets = {
        'all': rows,
        'input_has_match': [r for r in rows if r['input_has_match']],
        'active': [r for r in rows if r['active']],
        'no_model': [r for r in rows if not r['active']],
        'same_list': [r for r in rows if r['active'] and r['same_list_order']],
        'changed_list': [r for r in rows if r['active'] and not r['same_list_order']],
        'same_list_mixed': [r for r in rows if r['active'] and r['same_list_order']
                            and r['native_shortlist']['profile'] == 'mixed'],
        'same_list_same_request_weights': [r for r in rows if r['active'] and r['same_list_order']
                                           and r['same_non_rc_request'] and r['same_choice_weights']],
        'same_list_mixed_same_request_weights': [r for r in rows if r['active'] and r['same_list_order']
                 and r['native_shortlist']['profile'] == 'mixed' and r['same_non_rc_request'] and r['same_choice_weights']],
    }
    result = {'group': group, 'source_offline_sha256': sha(raw),
              'subsets': {name: summarize(rs) for name, rs in subsets.items()},
              'failures': {s: dict(Counter(r[s + '_failure_class'] for r in rows)) for s in ('native', 'rc')},
              'active_profiles': {s: dict(Counter(r[s + '_shortlist']['profile'] for r in subsets['active'])) for s in ('native', 'rc')},
              'same_list_profiles': dict(Counter(r['native_shortlist']['profile'] for r in subsets['same_list'])),
              'tokens': {}, 'items': rows, 'verification': proof}
    for name in ('active', 'same_list', 'same_list_mixed', 'same_list_same_request_weights', 'same_list_mixed_same_request_weights'):
        pairs = [(records['native', r['item_key']], records['rc', r['item_key']]) for r in subsets[name]]
        complete = [(a, b) for a, b in pairs if all((x['sampling'] or {}).get('complete', True)
                    and x['effective']['usage_complete'] for x in (a, b)) and signature(a) == signature(b)]
        result['tokens'][name] = {'all_active': token_pair_summary(pairs),
                                 'complete_same_budget': token_pair_summary(complete)}
    old = json.loads((HERE / group / 'summary.json').read_text())
    assert result['subsets']['active']['n'] == old['selection']['active']
    assert result['subsets']['same_list']['n'] == old['selection']['same_shortlist']
    for s in ('native', 'rc'):
        assert result['subsets']['all'][s]['matched'] == old['results']['sql_selection']['all']['bag_equal'][s]['matching_questions']
    print(group, json.dumps({k: result[k] for k in ('subsets', 'failures', 'same_list_profiles', 'verification')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    HERE = analysis_directory(argparse.ArgumentParser(description=__doc__))
    result = {'date': date.today().isoformat(), 'groups': {}}
    for group in GROUPS:
        result['groups'][group] = analyze(group)
    destination = HERE / 'selection_metrics.json'
    write_new(destination, result)
    assert json.loads(destination.read_text()) == result
    print('Verified and saved:', destination, flush=True)
