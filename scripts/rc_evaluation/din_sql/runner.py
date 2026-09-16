"""Fixed six-node DIN dependency graph, without stage-wide barriers."""
import asyncio
from pathlib import Path

from scripts.baseline_adapters.din_sql.inputs import NODES, OUTPUT_NODES, digest
from scripts.baseline_adapters.din_sql.prompts import PromptBuilder, parse_response, family


def dependencies(node, label):
    if node=='decomposition':
        return ('linking',) if label=='NESTED' else ()
    if node.startswith('generation'):
        return ('linking','decomposition')
    if node.startswith('revision'):
        return ('generation_base',)
    return ()


def outcome(node, *, status='succeeded', result=None, reason=None, origin='new_execution', **extra):
    return {'node':node,'status':status,'result':result,'reason':reason,'origin':origin,
            'usage':None,'input_fingerprint':None,'parent_refs':{},'response_ref':None,
            'source_refs':{},'fallback_used':False,**extra}


def copy_reusable(task, parent_version, new_version, records):
    """Preserve successful nodes only if all their real dependencies are reused."""
    reused = set()
    for node in NODES:
        old = records.node(parent_version,node)
        if old and old['status']=='succeeded' and all(p in reused for p in dependencies(node,task.label)):
            if records.node(new_version,node):
                reused.add(node)
                continue
            value = {k:v for k,v in old.items() if k!='ref'}
            value.update(origin='reused',source_refs={**old['source_refs'],'reused_node':old['ref']},
                         parent_refs={p:records.node(new_version,p)['ref'] for p in dependencies(node,task.label)})
            records.save_node(new_version,node,value)
            reused.add(node)
    return reused


async def run_question(task, version, records, execute_node, *, on_terminal=None):
    async def execute(node):
        existing = records.node(version,node)
        if existing:
            return existing
        parents = {parent:await pending[parent] for parent in dependencies(node,task.label)}
        failures = [p for p,r in parents.items() if r['status']!='succeeded']
        if failures:
            result = outcome(node,status='dependency_failed',reason={'dependencies':failures})
        elif node=='decomposition' and task.label!='NESTED':
            result = outcome(node,result={'label':task.label,'sub_questions':[]},origin='deterministic',
                             usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0})
        else:
            result = await execute_node(node,task,parents,version)
        result['parent_refs'] = {p:r['ref'] for p,r in parents.items()}
        if not result.get('input_fingerprint'):
            result['input_fingerprint'] = digest([str(task.key),node,result['parent_refs']])
        await asyncio.to_thread(records.save_node,version,node,result)
        saved = records.node(version,node)
        if on_terminal:
            on_terminal(task.key,node,saved['status'])
        return saved
    pending = {node:asyncio.create_task(execute(node)) for node in NODES}
    try:
        await asyncio.gather(*pending.values())
        await asyncio.to_thread(records.seal,version)
        return {node:records.node(version,node)['status'] for node in OUTPUT_NODES}
    except BaseException:
        for future in pending.values():
            future.cancel()
        await asyncio.gather(*pending.values(),return_exceptions=True)
        raise


class NodeExecutor:
    def __init__(self, prepared, requester, code_root=None):
        self.requester = requester
        self.builder = PromptBuilder(prepared,code_root or Path(__file__).resolve().parents[3])

    async def __call__(self, node, task, parents, version):
        messages = self.builder.build(node,task,parents)
        response = await self.requester.request(version,node,messages,family(task))
        if response['status']!='succeeded':
            return outcome(node,status='failed',reason=response.get('error'),response_ref=response.get('response_ref'))
        metadata = {'usage':response['usage'],'response_ref':response['response_ref'],
                    'source_refs':{'response_model':response.get('response_model')}}
        try:
            parsed = parse_response(node,family(task),response['content'],
                                    initial_sql=parents.get('generation_base',{}).get('result'))
        except (ValueError,TypeError) as exc:
            return outcome(node,status='failed',reason={'category':'business_parse','message':str(exc)},**metadata)
        return outcome(node,result=parsed['result'],reason=parsed['warnings'],
                       fallback_used=parsed['fallback_used'],**metadata)
