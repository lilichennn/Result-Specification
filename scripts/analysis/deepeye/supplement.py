"""Sensitivity analyses and result shape; never rerun SQL or call models."""
import argparse
from collections import Counter, defaultdict
import json
import zlib
import sqlglot
from .analyze import HERE, GROUPS, STAGES, checked, read_db, sha, write_new
from .summarize import pool_summary, paired, state, signature
from .paths import add_analysis_argument


def sampling_sensitivity(group, data=None, items=None):
    folder=HERE/group
    if data is None:
        data=json.loads((folder/'offline.json').read_text())
    if items is None:
        with read_db(folder/'evaluation.sqlite3') as db:
            items=[checked(r['payload_json'],r['sha256']) for r in db.execute('select * from items')]
    assert len(items)==GROUPS[group]
    lookup={(r['condition'],r['stage'],r['item_key']):r for r in data['records']}
    result={}
    for stage in STAGES[1:]:
        result[stage]={}
        complete=[];same=[]
        for item in items:
            key=item['item_key'];a=lookup['native',stage,key];b=lookup['rc',stage,key]
            if (b['participation'] or {}).get('status')!='participating':
                continue
            if not all((r['sampling'] or {}).get('complete',True) for r in (a,b)):
                continue
            complete.append(item)
            if signature(a)==signature(b):
                same.append(item)
        for label,subset in [('complete_active',complete),('same_budget_active',same)]:
            result[stage][label]={**{s:pool_summary([r['targets'][s+'/'+stage] for r in subset]) for s in ('native','rc')},
                                   'paired':paired(subset,stage)}
    write_new(folder/'sampling_sensitivity.json',result)
    print(group,'sampling sensitivity complete',flush=True)


def supplement(group):
    folder=HERE/group
    data=json.loads((folder/'offline.json').read_text())
    bindings={b['task_key']:b for b in data['bindings']}
    with read_db(folder/'evaluation.sqlite3') as db:
        items=[checked(r['payload_json'],r['sha256']) for r in db.execute('select * from items')]
        assert len(items)==GROUPS[group]
        shapes={}
        for row in db.execute('select cache_key,result from queries'):
            r=json.loads(zlib.decompress(row['result']))
            shapes[row['cache_key']]={'type':r['result_type'],'columns':len(r.get('result_cols') or []),
                                      'rows':len(r.get('result_rows') or [])}
    result={'group':group,'gold_order_requirements':Counter(),'order_aware':{},'nonempty_reference':{},
            'shape':{},'per_database':{},'generation_changes_by_shape':Counter()}
    for item in items:
        key=item['item_key'];binding=bindings[key]
        dialect='sqlite' if binding['db_type']=='sqlite' else 'postgres'
        try:
            tree=sqlglot.parse_one(data['references'][key]['sql'],read=dialect)
            ordered=bool(tree.args.get('order'))
        except Exception:
            ordered=None
        result['gold_order_requirements'][str(ordered)]+=1
        scope=key if binding['db_type']=='postgresql' else binding['database_path']
        for pool in item['targets'].values():
            for candidate in pool:
                candidate['gold_order_equal']=(candidate['ordered_equal'] if ordered else candidate['bag_equal']) if ordered is not None else None
                shape=shapes[sha(json.dumps([scope,candidate['sql']]))]
                candidate['column_count_equal']=(shape['columns']==item['reference_columns']) if candidate['execution_success'] and item['reference_success'] else None
                candidate['row_count_equal']=(shape['rows']==item['reference_rows']) if candidate['execution_success'] and item['reference_success'] else None
    for stage in STAGES[1:]:
        result['order_aware'][stage]={**{s:pool_summary([r['targets'][s+'/'+stage] for r in items],'gold_order_equal') for s in ('native','rc')},
                                     'paired':paired(items,stage,'gold_order_equal')}
        nonempty=[r for r in items if r['reference_success'] and r['reference_type']=='success']
        result['nonempty_reference'][stage]={s:pool_summary([r['targets'][s+'/'+stage] for r in nonempty]) for s in ('native','rc')}
        result['shape'][stage]={s:{f:sum(c[f] is True for r in items for c in r['targets'][s+'/'+stage])
                                   for f in ('column_count_equal','row_count_equal','bag_equal')}
                                for s in ('native','rc')}
        dbs={b['database_id'] for b in bindings.values()}
        result['per_database'][stage]={}
        for dbid in sorted(dbs):
            subset=[r for r in items if bindings[r['item_key']]['database_id']==dbid]
            values={s:sum(state(r['targets'][s+'/'+stage]) is True for r in subset) for s in ('native','rc')}
            result['per_database'][stage][dbid]={'n':len(subset),**values,'delta':values['rc']-values['native']}
    for item in items:
        a=item['targets']['native/sql_generation'];b=item['targets']['rc/sql_generation']
        if state(a) is not True and state(b) is True:
            result['generation_changes_by_shape']['gains']+=1
            result['generation_changes_by_shape']['gain_with_no_native_column_count_match']+=not any(c['column_count_equal'] is True for c in a)
        if state(a) is True and state(b) is not True:
            result['generation_changes_by_shape']['losses']+=1
            result['generation_changes_by_shape']['loss_with_no_rc_column_count_match']+=not any(c['column_count_equal'] is True for c in b)
    write_new(folder/'supplement.json',result)
    sampling_sensitivity(group,data,items)
    print(group,'supplement complete',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_analysis_argument(ap)
    ap.add_argument('--group',choices=list(GROUPS));args=ap.parse_args()
    HERE = args.analysis_dir.resolve()
    for g in [args.group] if args.group else GROUPS:
        supplement(g)
