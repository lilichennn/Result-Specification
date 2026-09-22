"""Recheck all saved comparisons and independently sum original sample usage."""
import argparse
from collections import defaultdict
import json
import sqlite3
import zlib
from .analyze import HERE, GROUPS, checked, read_db, sha, restore_jsonable, compare_results, _rows, write_new
from .paths import add_analysis_argument


def verify(group):
    folder=HERE/group
    data=json.loads((folder/'offline.json').read_text())
    records={r['attempt_id']:r for r in data['records']}
    bindings={b['task_key']:b for b in data['bindings']}
    sums=0
    for manifest in data['manifests']:
        with read_db(manifest['run_dir']+'/run.sqlite3') as db:
            query="""select attempt_id,count(*) n,
                sum(json_extract(payload_json,'$.usage.prompt_tokens')) p,
                sum(json_extract(payload_json,'$.usage.completion_tokens')) c,
                sum(json_extract(payload_json,'$.usage.total_tokens')) t,
                sum(json_extract(payload_json,'$.usage.reasoning_tokens')) r
                from events where kind='sample_result' and json_extract(payload_json,'$.succeeded')=1
                group by attempt_id"""
            for row in db.execute(query):
                if row['attempt_id'] not in records:
                    continue
                effective=records[row['attempt_id']]['effective']
                assert row['n']==effective['retained_samples']
                assert [row[f] for f in ('p','c','t')]==[effective['known_tokens'][f] for f in
                        ('prompt_tokens','completion_tokens','total_tokens')]
                assert row['r']==effective['reasoning_tokens']
                assert 0<=row['r']<=row['c'] and row['p']+row['c']==row['t']
                sums+=1
    with read_db(folder/'evaluation.sqlite3') as db:
        assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
        slots,checked_queries=0,set()
        def result(binding,sql):
            scope=binding['task_key'] if binding['db_type']=='postgresql' else binding['database_path']
            key=sha(json.dumps([scope,sql]))
            row=db.execute('select result,sha256 from queries where cache_key=?',(key,)).fetchone()
            raw=zlib.decompress(row['result']);assert sha(raw)==row['sha256'];checked_queries.add(key)
            return restore_jsonable(json.loads(raw))
        n=0
        for row in db.execute('select * from items'):
            item=checked(row['payload_json'],row['sha256']);k=item['item_key'];binding=bindings[k]
            reference=data['references'][k]
            if reference['status']!='available':
                raise ValueError('This verification run requires available references')
            gold=result(binding,reference['sql']);cache={}
            for pool in item['targets'].values():
                for candidate in pool:
                    sql=candidate['sql']
                    if sql not in cache:
                        pred=result(binding,sql)
                        c=compare_results(pred,gold)
                        try:
                            c['set_equal']=set(_rows(pred)[1])==set(_rows(gold)[1])
                        except (TypeError,ValueError,OverflowError,RecursionError):
                            c['set_equal']=None
                        cache[sql]=c
                    assert all(candidate[f]==cache[sql][f] for f in ('bag_equal','ordered_equal','set_equal','comparable'))
                    slots+=1
            n+=1
        assert n==GROUPS[group]
        total_queries=db.execute('select count(*) from queries').fetchone()[0]
        assert len(checked_queries)==total_queries
    proof={'group':group,'items':n,'comparisons_recomputed':slots,'query_checksums_verified':len(checked_queries),
           'original_successful_sample_token_aggregates_verified':sums,'sqlite_integrity':'ok'}
    write_new(folder/'verification.json',proof)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_analysis_argument(ap)
    ap.add_argument('--group',choices=list(GROUPS));args=ap.parse_args()
    HERE = args.analysis_dir.resolve()
    for g in [args.group] if args.group else GROUPS:
        verify(g)
