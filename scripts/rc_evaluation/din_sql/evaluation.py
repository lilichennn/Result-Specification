"""Strict paired evaluation, separate from immutable method executions."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import uuid

from scripts.baseline_adapters.dail_sql.execution import execute_sql
from scripts.rc_evaluation.deepeye.comparison import compare_results
from scripts.baseline_adapters.din_sql.inputs import DinSettings, OUTPUT_NODES, digest
from scripts.baseline_adapters.din_sql.records import load_prepared, write_json


def as_comparison_result(execution):
    return {'result_type':execution.get('status'), 'result_cols':execution.get('columns'),
            'result_rows':execution.get('rows')}


def sql_correct(predicted, gold):
    if gold.get('status') != 'success':
        return None, 'reference_execution_unavailable'
    error = predicted.get('error') or {}
    if predicted.get('status') == 'error':
        state = error.get('sqlstate') or ''
        # Only known deterministic query errors; storage/auth/connectivity unknown.
        if error.get('sqlite_errorcode') in (1, 19, 20, 23) or state[:2] in ('22','23','42'):
            return False, 'prediction_sql_error'
        return None, 'prediction_execution_unavailable'
    result = compare_results(as_comparison_result(predicted),as_comparison_result(gold))
    return result['bag_equal'],result['reason']


def evaluate_pair(base, rc, gold):
    b, br = sql_correct(base,gold)
    r, rr = sql_correct(rc,gold)
    return {'base_correct':b,'rc_correct':r,'base_reason':br,'rc_reason':rr,
            'comparable':b is not None and r is not None}


def normalize_usage(usage):
    usage = usage or {}
    number = lambda value: value if type(value) is int and value >= 0 else None
    inp, out, total = map(number,(usage.get('prompt_tokens'),usage.get('completion_tokens'),usage.get('total_tokens')))
    source = 'provider' if total is not None else 'unknown'
    if total is None and inp is not None and out is not None:
        total, source = inp+out, 'derived'
    reasoning = number((usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
    return {'input':inp,'output':out,'total':total,'total_source':source,'reasoning':reasoning}


def summarize_stage(rows):
    valid = [r for r in rows if r.get('base_correct') is not None and r.get('rc_correct') is not None]
    count = len(valid)
    valid_ids = {id(r) for r in valid}
    base = sum(r['base_correct'] for r in valid)
    rc = sum(r['rc_correct'] for r in valid)
    summary = {'finished_questions':len(rows),'quality_pairs':count,'base_correct':base,'rc_correct':rc,
               'base_pct':100*base/count if count else None,'rc_pct':100*rc/count if count else None,
               'delta_pp':100*(rc-base)/count if count else None,
               'wrong_to_right':sum(not r['base_correct'] and r['rc_correct'] for r in valid),
               'right_to_wrong':sum(r['base_correct'] and not r['rc_correct'] for r in valid),
               'excluded':dict(Counter(r.get('base_reason') or r.get('rc_reason') or 'missing_result'
                                      for r in rows if id(r) not in valid_ids))}
    for field in ('input','output','total'):
        pairs = [r for r in rows if r.get('base_'+field) is not None and r.get('rc_'+field) is not None]
        b = sum(r['base_'+field] for r in pairs)
        c = sum(r['rc_'+field] for r in pairs)
        summary[field+'_tokens'] = {'pairs':len(pairs),'base':b,'rc':c,
                                   'saving_pct':100*(1-c/b) if b else None}
    summary['token_pairs'] = summary['total_tokens']['pairs']
    summary['token_saving_pct'] = summary['total_tokens']['saving_pct']
    return summary


def summary_markdown(summary):
    number = lambda v: '未知' if v is None else f'{v:.2f}'
    lines = ['# DIN-SQL 原生与 RC3 对照','',
             '严格比较列数和列位置，忽略行顺序、保留重复行。质量和 Token 分别使用成对有效题；未结束题不进入主表。', '',
             '| 测试组 | 阶段 | 总题数 | 已有完整记录 | 质量有效题 | 原生正确率 % | RC3 正确率 % | 变化（百分点） |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group,stages in summary.items():
        for stage,row in stages.items():
            lines.append(f'| {group} | {stage.title()} | {row["total_questions"]} | {row["finished_questions"]} | '
                         f'{row["quality_pairs"]} | {number(row["base_pct"])} | {number(row["rc_pct"])} | {number(row["delta_pp"])} |')
    lines += ['', '| 测试组 | 阶段 | Token 类别 | 成对有效题 | 原生总量 | RC3 总量 | 节省比例 % |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for group,stages in summary.items():
        for stage,row in stages.items():
            for field in ('input','output','total'):
                value=row[field+'_tokens']
                lines.append(f'| {group} | {stage.title()} | {field} | {value["pairs"]} | {value["base"]} | '
                             f'{value["rc"]} | {number(value["saving_pct"])} |')
    return '\n'.join(lines)+'\n'


def evaluate(batch, *, groups, records):
    batch = Path(batch)
    prepared = load_prepared(batch)
    settings = DinSettings(**records.manifest.get('settings',{}))
    # Snapshot version choices ONCE. Later reruns cannot change this report.
    current = [r for r in records.current_rows() if r['group'] in groups]
    report_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    directory = batch/'reports'/report_id
    directory.mkdir(parents=True)
    write_json(directory/'snapshot.json', {'versions':current,'sql_timeout':settings.sql_timeout_seconds,
                                          'comparison':'exact columns/positions, duplicate-preserving row bag'})
    futures, bindings = {}, {}
    with ThreadPoolExecutor(max_workers=settings.sql_workers) as pool:
        def submit(database, sql):
            signature = digest([database,sql,settings.sql_timeout_seconds])
            if signature not in futures:
                futures[signature] = pool.submit(execute_sql,database,sql,timeout_seconds=settings.sql_timeout_seconds)
            return signature
        for row in current:
            version = row['attempt_id']
            task_key = f'{row["group"]}/{row["item_key"]}'
            reference = prepared.evaluation[task_key]
            nodes = {node:records.node(version,node) for node in OUTPUT_NODES}
            bindings[version] = {'row':row,'nodes':nodes,'gold':submit(reference['database'],reference['gold_sql']),
                                 'sqls':{n:submit(reference['database'],r['result']) for n,r in nodes.items()
                                         if r and r['status']=='succeeded'}}
        observations = {}
        for signature, future in futures.items():
            result = future.result()
            observations[signature] = result
            write_json(directory/'executions'/(signature+'.json'),result)
    details = []
    for version, binding in bindings.items():
        row, nodes = binding['row'],binding['nodes']
        for stage in ('generation','revision'):
            names = [stage+'_base',stage+'_rc3']
            results = [observations[binding['sqls'][n]] if n in binding['sqls'] else {'status':'no_sql'} for n in names]
            detail = {'group':row['group'],'question_id':row['item_key'],'version':version,'stage':stage,
                      **evaluate_pair(*results,observations[binding['gold']]),
                      'nodes':{n:{'status':nodes[n]['status'],'sql':nodes[n]['result'],'origin':nodes[n]['origin'],
                                  'ref':nodes[n]['ref'],'source_refs':nodes[n]['source_refs']} for n in names},
                      'execution_refs':{n:binding['sqls'].get(n) for n in names},'gold_ref':binding['gold']}
            for prefix,node in zip(('base','rc'),names):
                usage = normalize_usage(nodes[node]['usage'] if nodes[node]['status']=='succeeded' else None)
                detail.update({prefix+'_'+k:v for k,v in usage.items()})
            details.append(detail)
    summary = {}
    for group in groups:
        summary[group] = {stage:{'total_questions':len(records.manifest['groups'][group]['ids']),
                                **summarize_stage([r for r in details if r['group']==group and r['stage']==stage])}
                          for stage in ('generation','revision')}
    write_json(directory/'details.json',details)
    write_json(directory/'summary.json',summary)
    (directory/'tables.md').write_text(summary_markdown(summary),encoding='utf-8')
    return {'report':str(directory),'summary':summary}
