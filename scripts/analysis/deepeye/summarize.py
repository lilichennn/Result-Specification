"""Offline paired statistics, using only saved analysis products."""
from collections import Counter
import json
import math
import statistics
import numpy as np
from scipy.stats import binomtest
from .analyze import HERE, GROUPS, STAGES, checked, read_db, write_new, schema_coverage
from .paths import add_analysis_argument


def state(pool, field='bag_equal'):
    values = [r[field] for r in pool]
    return True if True in values else (None if None in values else False)


def pool_summary(pools, field='bag_equal'):
    return {'questions': len(pools), 'slots': sum(map(len, pools)),
            'matching_questions': sum(state(p, field) is True for p in pools),
            'unknown_questions': sum(state(p, field) is None for p in pools),
            'matching_slots': sum(r[field] is True for p in pools for r in p),
            'unknown_slots': sum(r[field] is None for p in pools for r in p),
            'executable_slots': sum(r['execution_success'] for p in pools for r in p),
            'unique_sql_sum': sum(len({r['sql'] for r in p}) for p in pools),
            'mean_question_matching_fraction': statistics.mean(sum(r[field] is True for r in p)/len(p) if p else 0 for p in pools) if pools else None}


def paired(items, stage, field='bag_equal'):
    pairs = [(r['item_key'], state(r['targets']['native/'+stage], field), state(r['targets']['rc/'+stage], field)) for r in items]
    counts = Counter((a, b) for _, a, b in pairs)
    gains, losses = counts[False, True], counts[True, False]
    comparable = sum(a is not None and b is not None for _, a, b in pairs)
    n = gains + losses
    return {'transitions': [{'native': a, 'rc': b, 'count': c} for (a, b), c in counts.items()],
            'comparable_pairs': comparable, 'comparable_gains': gains, 'comparable_losses': losses,
            'mcnemar_exact_p': binomtest(gains, n, 0.5).pvalue if n else 1.0,
            'new_matching_keys': [k for k, a, b in pairs if a is not True and b is True],
            'lost_matching_keys': [k for k, a, b in pairs if a is True and b is not True]}


def token_sum(rows):
    fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    out = {f: sum(r['effective']['known_tokens'][f] for r in rows) for f in fields}
    out['reasoning_tokens'] = sum(r['effective']['known_reasoning_tokens'] for r in rows)
    out['nonreasoning_completion_tokens'] = out['completion_tokens'] - out['reasoning_tokens']
    return out


def signature(row):
    return sorted(json.dumps(v, sort_keys=True) for v in row['budgets'])


def token_pair_summary(pairs):
    native, rc = token_sum([a for a, b in pairs]), token_sum([b for a, b in pairs])
    out = {'pairs': len(pairs), 'native': native, 'rc': rc,
           'relative_change_percent': {f: 100*(rc[f]/native[f]-1) if native[f] else None for f in native},
           'native_retained_samples': sum(a['effective']['retained_samples'] for a, b in pairs),
           'rc_retained_samples': sum(b['effective']['retained_samples'] for a, b in pairs)}
    if pairs:
        ratios = [b['effective']['known_tokens']['total_tokens']/a['effective']['known_tokens']['total_tokens']-1
                  for a, b in pairs if a['effective']['known_tokens']['total_tokens']]
        out['median_per_question_total_change_percent'] = 100*statistics.median(ratios) if ratios else None
        # Paired, question-level bootstrap of aggregate total-token ratio, not API samples.
        v = np.array([[a['effective']['known_tokens']['total_tokens'], b['effective']['known_tokens']['total_tokens']]
                      for a, b in pairs], dtype=np.float64)
        rng = np.random.default_rng(20260915)
        ratios_boot = []
        for _ in range(2000):
            s = v[rng.integers(0, len(v), len(v))].sum(axis=0)
            ratios_boot.append(100*(s[1]/s[0]-1) if s[0] else 0)
        out['total_ratio_paired_bootstrap_95_percent'] = np.quantile(ratios_boot, [.025, .975]).tolist()
    return out


def normalized_coverage(reference, linked, sqlite):
    if sqlite:
        import sqlglot
        from sqlglot import exp
        linked = {t.lower(): [c.lower() for c in cols] for t, cols in linked.items()}
        try:
            tree = sqlglot.parse_one(reference, read='sqlite')
            for identifier in tree.find_all(exp.Identifier):
                identifier.set('this', identifier.name.lower())
            reference = tree.sql(dialect='sqlite')
        except Exception:
            pass
    return schema_coverage(reference, linked, dialect='sqlite' if sqlite else 'postgres')


def summarize(group):
    folder = HERE / group
    data = json.loads((folder/'offline.json').read_text())
    with read_db(folder/'evaluation.sqlite3') as db:
        items = [checked(r['payload_json'], r['sha256']) for r in db.execute('select * from items')]
        queries = db.execute('select count(*) from queries').fetchone()[0]
    assert len(items) == GROUPS[group] == len({r['item_key'] for r in items})
    records = data['records']
    lookup = {(r['condition'], r['stage'], r['item_key']): r for r in records}
    item_lookup = {r['item_key']: r for r in items}
    result = {'group': group, 'n': len(items), 'queries': queries,
              'reference': {'executable': sum(r['reference_success'] for r in items),
                'failures': [{k: r[k] for k in ('item_key','reference_status','reference_type','reference_error')}
                             for r in items if not r['reference_success']],
                'empty': sum(r['reference_type']=='empty_result' for r in items),
                'all_null': sum(r['reference_type']=='all_null_result' for r in items)},
              'results': {}, 'usage': {}, 'tokens': {}, 'schema': {}, 'selection': {}, 'revision': {}}
    for stage in STAGES:
        active_keys = {r['item_key'] for r in records if r['stage']==stage and r['condition']=='rc'
                       and (r['participation'] or {}).get('status')=='participating'}
        pairs = [(lookup['native',stage,k], lookup['rc',stage,k]) for k in sorted(active_keys)]
        complete_pairs = [(a,b) for a,b in pairs if (a['sampling'] or {}).get('complete',True)
                          and (b['sampling'] or {}).get('complete',True)
                          and a['effective']['usage_complete'] and b['effective']['usage_complete']]
        same_budget = [(a,b) for a,b in complete_pairs if signature(a)==signature(b)]
        result['tokens'][stage] = {name: token_pair_summary(ps) for name,ps in
                                   [('active',pairs),('complete',complete_pairs),('same_budget',same_budget)]}
        result['usage'][stage] = {}
        for side in ('native','rc'):
            rows = [r for r in records if r['condition']==side and r['stage']==stage]
            result['usage'][stage][side] = {'tasks': len(rows),
                'active': sum(r['api'].get('api_request',0)>0 for r in rows),
                'api': dict(sum((Counter(r['api']) for r in rows),Counter())),
                'incomplete_tasks': sum(not (r['sampling'] or {}).get('complete',True) for r in rows),
                'failed_samples': sum(r['failed_samples'] for r in rows),
                'missing_usage': sum(r['effective']['samples_missing_usage'] for r in rows),
                'missing_reasoning': sum(r['effective']['samples_missing_reasoning'] for r in rows),
                'retained_samples': sum(r['effective']['retained_samples'] for r in rows),
                'tokens': token_sum(rows)}
        if stage=='schema_linking':
            continue
        result['results'][stage] = {}
        full_gen_keys = {k for k in item_lookup if all(len(lookup[s,'sql_generation',k]['sqls'])==12 for s in ('native','rc'))}
        subsets = [('all',items), ('active',[r for r in items if r['item_key'] in active_keys])]
        if stage=='sql_generation':
            subsets.append(('both_12_candidates',[r for r in items if r['item_key'] in full_gen_keys]))
        for subset, subset_rows in subsets:
            result['results'][stage][subset] = {field: {
                **{s:pool_summary([r['targets'][s+'/'+stage] for r in subset_rows],field) for s in ('native','rc')},
                'paired': paired(subset_rows,stage,field)} for field in ('bag_equal','ordered_equal','set_equal')}
    # Schema coverage uses actual kept tables/columns and a deliberately conservative parser.
    coverage = {}
    for side in ('native','rc'):
        rows = [r for r in records if r['condition']==side and r['stage']=='schema_linking']
        cov = {r['item_key']: normalized_coverage(data['references'][r['item_key']].get('sql'), r['linked'],
                 data['bindings'][0]['db_type']=='sqlite') for r in rows}
        coverage[side] = cov
        result['schema'][side] = {
            'mean_tables': statistics.mean(len(r['linked']) for r in rows),
            'mean_columns': statistics.mean(sum(map(len,r['linked'].values())) for r in rows)}
        for field in ('table_coverage','column_coverage'):
            vals = [r[field] for r in cov.values() if r[field] is not None]
            result['schema'][side][field] = {'available': len(vals), 'full': vals.count(1.0),
                                           'mean': statistics.mean(vals) if vals else None}
    result['schema']['coverage_pairs'] = {field: {
        'gains': [k for k in coverage['native'] if coverage['native'][k][field] is not None
                  and coverage['native'][k][field]<1 and coverage['rc'][k][field]==1],
        'losses': [k for k in coverage['native'] if coverage['native'][k][field]==1
                   and coverage['rc'][k][field] is not None and coverage['rc'][k][field]<1]}
        for field in ('table_coverage','column_coverage')}
    # Revision is compared with the very same native Generation input, not RC Generation.
    for side in ('native','rc'):
        transitions, slot_transitions = Counter(), Counter()
        for item in items:
            before=item['targets']['native/sql_generation']; after=item['targets'][side+'/sql_revision']
            transitions[state(before),state(after)]+=1
            assert len(before)==len(after)
            slot_transitions.update((a['bag_equal'],b['bag_equal']) for a,b in zip(before,after))
        result['revision'][side] = {
            'pool_transitions': [{'before':a,'after':b,'count':n} for (a,b),n in transitions.items()],
            'slot_transitions': [{'before':a,'after':b,'count':n} for (a,b),n in slot_transitions.items()]}
    # The Selection pre-shortlist can differ before RC is injected.
    stable_keys, changed_keys = [],[]
    active_selection = []
    branch_counts = {}
    for side in ('native','rc'):
        branch_counts[side] = {}
        for item in items:
            k=item['item_key']; row=lookup[side,'sql_selection',k]
            branch=row['selection_trace']['branch']
            stats=branch_counts[side].setdefault(branch,Counter())
            stats['tasks']+=1
            pool=state(item['targets']['native/sql_revision']) is True
            shortlist=state(item['targets'][side+'/shortlist']) is True
            selected=state(item['targets'][side+'/sql_selection']) is True
            stats['native_pool_matching']+=pool
            stats['shortlist_matching']+=shortlist
            stats['selected_matching']+=selected
            stats['pool_matching_but_not_selected']+=pool and not selected
    for item in items:
        k=item['item_key']; a=lookup['native','sql_selection',k]; b=lookup['rc','sql_selection',k]
        if (b['participation'] or {}).get('status')=='participating':
            active_selection.append(k)
            (stable_keys if a['selection_trace']['shortlist_sqls']==b['selection_trace']['shortlist_sqls']
             else changed_keys).append(k)
    result['selection']={'branches':branch_counts, 'active':len(active_selection),
        'same_shortlist':len(stable_keys),'changed_shortlist':len(changed_keys),
        'same_shortlist_paired':paired([item_lookup[k] for k in stable_keys],'sql_selection'),
        'changed_shortlist_paired':paired([item_lookup[k] for k in changed_keys],'sql_selection')}
    write_new(folder/'summary.json',result)
    # Complete change lists with relevant question/RC/SQL, not hand-picked only positive examples.
    changes = []
    for stage in STAGES[1:]:
        p = result['results'][stage]['all']['bag_equal']['paired']
        for direction, keys in [('gain',p['new_matching_keys']),('loss',p['lost_matching_keys'])]:
            for key in keys:
                item=item_lookup[key]
                contract=data['contracts'][key]
                changes.append({'group':group,'stage':stage,'direction':direction,'item_key':key,
                    'question':contract['question'],'evidence':contract['evidence'],'rc3':contract['final_rc'],
                    'reference':data['references'][key],
                    'native':item['targets']['native/'+stage], 'rc':item['targets']['rc/'+stage],
                    'native_shortlist':item['targets']['native/shortlist'] if stage=='sql_selection' else None,
                    'rc_shortlist':item['targets']['rc/shortlist'] if stage=='sql_selection' else None})
    write_new(folder/'changes.json',changes)
    print(group, json.dumps({'reference':result['reference'],
        'main':{stage:{s:result['results'][stage]['all']['bag_equal'][s]['matching_questions'] for s in ('native','rc')} for stage in STAGES[1:]},
        'tokens':{stage:result['tokens'][stage]['complete']['relative_change_percent']['total_tokens'] for stage in STAGES}},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    add_analysis_argument(ap)
    ap.add_argument('--group',choices=list(GROUPS));args=ap.parse_args()
    HERE = args.analysis_dir.resolve()
    results={}
    for g in [args.group] if args.group else GROUPS:
        p=HERE/g/'summary.json'
        results[g]=json.loads(p.read_text()) if p.exists() else summarize(g)
    if not args.group:
        tests=sorted((r['results'][s]['all']['bag_equal']['paired']['mcnemar_exact_p'],g,s)
                     for g,r in results.items() for s in STAGES[1:])
        adjusted=[];prev=0.0
        for i,(p,g,s) in enumerate(tests):
            prev=max(prev,min(1.0,p*(len(tests)-i)))
            adjusted.append({'group':g,'stage':s,'p':p,'holm_adjusted_p':prev})
        write_new(HERE/'summary_all.json',{'groups':results,'holm_15_sql_comparisons':adjusted})
