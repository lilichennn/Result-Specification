"""Read-only Revision participation and quality audit. No model/SQL execution."""
from collections import Counter, defaultdict
import json

from .analyze import HERE, GROUPS, checked, read_db, write_new, restore_jsonable, sha
from .paths import analysis_directory
import argparse
from datetime import date
from scripts.rc_evaluation.deepeye.injection import manifest_prompt, render_rc_block


def normalize(sql):
    # Exact grouping convention in native SQLRevisionRunner._normalize_sql.
    return ' '.join(sql.split()).strip().lower() if sql else ''


def participation(data):
    records = {r['attempt_id']: r for r in data['records']
               if r['condition'] == 'rc' and r['stage'] == 'sql_revision'}
    by_item = defaultdict(dict)
    verified_events = verified_api = 0
    for meta in data['manifests']:
        if meta['condition'] != 'rc' or meta['stage'] != 'sql_revision':
            continue
        components, endings, groups, requests, samples = {}, {}, {}, [], []
        with read_db(meta['run_dir'] + '/run.sqlite3') as db:
            row = db.execute('select payload_json,payload_checksum from manifest').fetchone()
            manifest = checked(row[0], row[1])
            assert row[1] == meta['manifest_sha256']
            assert manifest['rc_version'] == 3 and manifest['gold_corrected']
            prompt = manifest_prompt(manifest)
            blocks = {}
            query = """select attempt_id,kind,payload_json,payload_checksum from events
                where kind in ('component_start','sampling_group_bound','sample_result','api_request')
                or (kind='component_result' and json_extract(payload_json,'$.component')='revision.candidate')
                order by event_id"""
            for e in db.execute(query):
                assert e['attempt_id'] in records
                p = restore_jsonable(checked(e['payload_json'], e['payload_checksum']))
                verified_events += 1
                key = records[e['attempt_id']]['item_key']
                kind = e['kind']
                if kind == 'component_start':
                    components[p['component_call_id']] = {
                        'item_key': key, 'component': p['component'],
                        'parent': p.get('parent_component_call_id'),
                        'input_sql': p.get('input_sql') if p['component'] == 'revision.candidate' else None,
                        'event_sha256': e['payload_checksum'], 'attempt_id': e['attempt_id']}
                elif kind == 'component_result':
                    endings[p['component_call_id']] = p['result'][0]
                elif kind == 'sampling_group_bound':
                    assert p['rc_applied'] is True
                    groups[p['group_id']] = p
                elif kind == 'sample_result':
                    assert p['rc_applied'] is True
                    samples.append({k: p[k] for k in ('component_call_id', 'group_id', 'succeeded')})
                elif kind == 'api_request':
                    if key not in blocks:
                        contract = manifest['contracts'][key]
                        assert contract['rc_version'] == 3 and contract['source_field'] == 'rc_round3'
                        blocks[key] = render_rc_block(contract, prompt_template=prompt)
                    assert any(blocks[key] in str(m.get('content', ''))
                               for m in p['kwargs']['messages'])
                    verified_api += 1
                    requests.append({k: p[k] for k in ('component_call_id', 'group_id')})
        def candidate(call_id):
            seen = set()
            while components[call_id]['component'] != 'revision.candidate':
                assert call_id not in seen
                seen.add(call_id)
                call_id = components[call_id]['parent']
            return call_id
        activity = defaultdict(lambda: {'api_requests': 0, 'successful_samples': 0,
                                        'failed_samples': 0, 'checkers': set(), 'groups': set()})
        for p in requests:
            assert p['group_id'] in groups
            cid = candidate(p['component_call_id'])
            a = activity[cid]
            a['api_requests'] += 1
            a['groups'].add(p['group_id'])
            a['checkers'].add(components[p['component_call_id']]['component'].removesuffix('.extraction'))
        for p in samples:
            cid = candidate(p['component_call_id'])
            activity[cid]['successful_samples' if p['succeeded'] else 'failed_samples'] += 1
        for cid, comp in components.items():
            if comp['component'] != 'revision.candidate':
                continue
            assert cid in endings
            key, norm = comp['item_key'], normalize(comp['input_sql'])
            assert norm not in by_item[key], (key, norm)
            a = activity[cid]
            if a['successful_samples'] or a['failed_samples']:
                assert a['api_requests'] > 0
            by_item[key][norm] = {
                **comp, 'output_sql': endings[cid], 'component_call_id': cid,
                'api_requests': a['api_requests'], 'successful_samples': a['successful_samples'],
                'failed_samples': a['failed_samples'], 'checkers': sorted(a['checkers']),
                'sampling_groups': len(a['groups']),
            }
    return dict(by_item), {'source_event_checksums': verified_events,
                          'actual_api_prompts_verified_rc3': verified_api}


def quality(rows):
    out = {'candidates': len(rows)}
    for field in ('generation', 'native_revision', 'rc_revision'):
        matches = sum(r[field] is True for r in rows)
        out[field] = {'match': matches, 'unknown': sum(r[field] is None for r in rows),
                      'rate': matches / len(rows) if rows else None}
    for before, after, key in (
        ('generation', 'native_revision', 'native_repair'),
        ('generation', 'rc_revision', 'rc_repair'),
        ('native_revision', 'rc_revision', 'rc_vs_native')):
        transitions = Counter((r[before], r[after]) for r in rows)
        out[key] = {
            'confirmed_gain': transitions[False, True], 'unknown_to_match': transitions[None, True],
            'confirmed_loss': transitions[True, False], 'match_to_unknown': transitions[True, None],
            'both_match': transitions[True, True],
            'transitions': [{'before': a, 'after': b, 'count': n} for (a, b), n in transitions.items()]}
        assert out[after]['match'] - out[before]['match'] == (
            out[key]['confirmed_gain'] + out[key]['unknown_to_match'] -
            out[key]['confirmed_loss'] - out[key]['match_to_unknown'])
    out['rc_sql_text_differs_from_native_revision'] = sum(r['rc_vs_native_text_changed'] for r in rows)
    out['rc_sql_text_differs_from_generation'] = sum(r['rc_vs_generation_text_changed'] for r in rows)
    return out


def question_stats(items):
    states = Counter()
    rates = Counter()
    for rows in items.values():
        a = sum(r['native_revision'] is True for r in rows)
        b = sum(r['rc_revision'] is True for r in rows)
        states[bool(a), bool(b)] += 1
        rates['increase' if b > a else 'decrease' if b < a else 'unchanged'] += 1
    return {'questions': len(items), 'native_has_match': sum(v for (a, b), v in states.items() if a),
            'rc_has_match': sum(v for (a, b), v in states.items() if b),
            'gain': states[False, True], 'loss': states[True, False], 'rate_directions': dict(rates)}


def analyze(group):
    raw = (HERE / group / 'offline.json').read_bytes()
    data = json.loads(raw)
    active, proof = participation(data)
    records = {r['item_key']: r for r in data['records'] if r['condition'] == 'rc' and r['stage'] == 'sql_revision'}
    items, uniques = {}, []
    with read_db(HERE / group / 'evaluation.sqlite3') as db:
        for row in db.execute('select payload_json,sha256 from items order by task_key'):
            item = checked(row['payload_json'], row['sha256'])
            key, target = item['item_key'], item['targets']
            gen, native, rc = (target[k] for k in ('native/sql_generation', 'native/sql_revision', 'rc/sql_revision'))
            assert len(gen) == len(native) == len(rc)
            assert all(r['bag_equal'] is not None for r in native + rc)
            seen, slots = set(), []
            for i, (a, b, c) in enumerate(zip(gen, native, rc)):
                norm = normalize(a['sql'])
                trace = active.get(key, {}).get(norm)
                if trace:
                    assert trace['output_sql'] == c['sql']
                participating = bool(trace and trace['api_requests'])
                slot = {'item_key': key, 'slot_index': i, 'rc_participated': participating,
                        'generation': a['bag_equal'], 'native_revision': b['bag_equal'], 'rc_revision': c['bag_equal'],
                        'rc_vs_native_text_changed': b['sql'] != c['sql'],
                        'rc_vs_generation_text_changed': a['sql'] != c['sql']}
                slots.append(slot)
                if norm not in seen and participating:
                    assert trace['input_sql'] == a['sql']
                    uniques.append({**slot, **{k: trace[k] for k in (
                        'component_call_id', 'attempt_id', 'api_requests', 'successful_samples',
                        'failed_samples', 'checkers', 'sampling_groups', 'event_sha256')},
                        'input_sql_sha256': sha(a['sql']), 'normalized_sql_sha256': sha(norm),
                        'covered_slots': [j for j, s in enumerate(gen) if normalize(s['sql']) == norm]})
                seen.add(norm)
            if key in active:
                assert set(active[key]) == seen
            participated = any(s['rc_participated'] for s in slots)
            assert participated == ((records[key]['participation'] or {}).get('status') == 'participating')
            items[key] = slots
    assert len(items) == GROUPS[group]
    all_slots = [r for rows in items.values() for r in rows]
    rc_slots = [r for r in all_slots if r['rc_participated']]
    inactive = [r for r in all_slots if not r['rc_participated']]
    actual_q = {k: rows for k, rows in items.items() if any(r['rc_participated'] for r in rows)}
    result = {
        'group': group, 'total_questions': len(items), 'source_offline_sha256': sha(raw),
        'rc_participation': {'questions': len(actual_q), 'unique_candidate_units': len(uniques),
            'original_candidate_slots': len(rc_slots),
            'units_with_successful_sample': sum(u['successful_samples'] > 0 for u in uniques),
            'units_with_no_successful_sample': sum(u['successful_samples'] == 0 for u in uniques),
            'api_requests': sum(u['api_requests'] for u in uniques),
            'checkers_candidate_units': dict(Counter(c for u in uniques for c in u['checkers']))},
        'all_slots': quality(all_slots), 'rc_involved_slots': quality(rc_slots),
        'rc_involved_unique_units': quality(uniques), 'rc_uninvolved_slots': quality(inactive),
        'all_questions': question_stats(items), 'participating_questions': question_stats(actual_q),
        'items': items, 'unique_units': uniques, 'verification': proof,
    }
    assert sum(len(u['covered_slots']) for u in uniques) == len(rc_slots)
    assert proof['actual_api_prompts_verified_rc3'] == result['rc_participation']['api_requests']
    old = json.loads((HERE / group / 'summary.json').read_text())
    assert len(actual_q) == old['usage']['sql_revision']['rc']['active']
    for side in ('native', 'rc'):
        q = old['results']['sql_revision']['all']['bag_equal'][side]
        assert result['all_slots'][side + '_revision']['match'] == q['matching_slots']
        assert result['all_questions'][side + '_has_match'] == q['matching_questions']
    print(group, json.dumps({k: result[k] for k in ('rc_participation', 'all_slots', 'rc_involved_unique_units', 'all_questions', 'verification')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    HERE = analysis_directory(argparse.ArgumentParser(description=__doc__))
    result = {'date': date.today().isoformat(), 'groups': {}}
    for group in GROUPS:
        result['groups'][group] = analyze(group)
    path = HERE / 'revision_metrics.json'
    write_new(path, result)
    assert json.loads(path.read_text()) == result
    print('Verified and saved:', path, flush=True)
