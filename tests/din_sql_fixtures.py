from scripts.baseline_adapters.din_sql.inputs import TaskKey, DinTask, NODES


def minimal_manifest(ids=('0','1')):
    return {'format':'din-sql-v1','batch_id':'fixture','groups':{'bird_dev':{'ids':list(ids)}}}


def make_task(label='EASY', group='bird_dev', question_id='0'):
    return DinTask(TaskKey(group,question_id), 'What is the maximum score?', '',
                   {'dialect':'sqlite','database_id':'scores'}, 'scores',
                   {k:'none' for k in ('population','row_grain','column_role','derivation','filter_policy','meta_review')}, label)


def terminal(node, status='succeeded'):
    return {'node':node,'status':status,'result':'SELECT 1' if status=='succeeded' else None,
            'reason':None,'usage':None,'origin':'test','input_fingerprint':'fixture',
            'parent_refs':{},'response_ref':None,'source_refs':{},'fallback_used':False}
