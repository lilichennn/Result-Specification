"""One-time method builders, preserving colleague templates and RC placement."""
from functools import lru_cache
import json
from pathlib import Path
import re

from .inputs import legacy_pure_functions


def family(task):
    return 'spider' if task.key.group.startswith('spider') else (
        'postgresql' if task.key.group.startswith('bird_interact') else 'bird')


class PromptBuilder:
    def __init__(self, prepared, code_root):
        self.prepared = prepared
        t = prepared.templates
        # Pure function bodies only. Inject frozen in-memory lookups instead of
        # their original per-question file readers; no legacy CLI is imported.
        constants = {**t['colleague'], 'ROOT':Path(code_root).parent}
        constants['literal_assignment'] = lambda path,name: t['spider' if path.name=='DIN-SQL.py' else 'bird'][name]
        constants['spider_schema'] = lambda db: self._spider(db)
        self.generation = legacy_pure_functions(Path(code_root),'sql_generation.py',
            ['schema_links_text','result_contract_text','spider_tables','build_spider_messages',
             'insert_before_answer','build_bird_messages'],constants)
        self.revision = legacy_pure_functions(Path(code_root),'self_correction.py',['build_messages'],
            {**constants, 'result_contract_text':self.generation['result_contract_text'],
             'insert_before_answer':self.generation['insert_before_answer']})
        self.linking = legacy_pure_functions(Path(code_root),'schema_linking.py',['build_prompt'],constants)
        self.decomposition = legacy_pure_functions(Path(code_root),'difficulty_decomposition.py',['build_prompt'],
            {**constants,'database_context':lambda context,meta:context})
        self.spider_schemas = {task.database['database_id']:prepared.schemas[task.schema_ref]['spider']
                               for task in prepared.tasks.values() if family(task)=='spider'}

    def _spider(self, db):
        if db=='college_2':
            return self.prepared.schemas['spider:college_2']
        return self.spider_schemas[db]

    def contract_block(self, rc3, *, revision=False):
        block = self.generation['result_contract_text']({'rc_round3':rc3})
        if revision:
            block = block.replace('Translate each field into SQL as follows:',
                                  'Check and correct the supplied SQL against each field as follows:')
            block = block.replace('supplied schema and schema links','supplied schema')
        return block

    def build(self, node, task, parents):
        dataset = family(task)
        bird = dataset!='spider'
        instance = {'question':task.question,'evidence':task.evidence,'db_id':task.database['database_id']}
        schema = self.prepared.schemas[task.schema_ref]
        context = schema['context']
        links = parents.get('linking',{}).get('result',[])
        sub = parents.get('decomposition',{}).get('result',{}).get('sub_questions',[])
        rc = {'rc_round3':task.rc3} if node.endswith('rc3') else None
        if node=='linking':
            examples = (self.prepared.templates['bird']['SYSTEM_SCHEMA_LINKING_TEMPLATE'].split('Few examples of this task are:\n###\n',1)[1]
                        if bird else self.prepared.templates['spider']['schema_linking_prompt']).strip()
            messages = [{'role':'user','content':self.linking['build_prompt'](examples,context,instance)}]
        elif node=='decomposition':
            messages = self.decomposition['build_prompt']('bird' if bird else 'spider',instance,links,context,None)
        elif node.startswith('generation'):
            messages = (self.generation['build_bird_messages'](instance,task.label,links,sub,context,rc)
                        if bird else self.generation['build_spider_messages'](instance,task.label,links,sub,rc))
        elif node.startswith('revision'):
            initial = parents['generation_base']['result']
            if bird:
                messages = self.revision['build_messages'](instance,initial,context,rc)
            else:
                prompt = (self.prepared.templates['spider']['debug_instruction']+schema['spider']+'\n'
                          +schema.get('primary','Primary_keys = []')+'#### Question: '+task.question
                          +'\n#### SQLite SQL QUERY\n'+initial)
                if rc:
                    prompt += self.contract_block(task.rc3,revision=True)
                messages = [{'role':'user','content':prompt+'\n#### SQLite FIXED SQL QUERY\nSELECT'}]
        else:
            raise ValueError(f'Unknown DIN node: {node}')
        if dataset=='postgresql':
            # Dialect adaptation is symmetric, including few-shot instructions.
            # Protect contract values: the RC must remain byte-for-byte intact.
            block = self.contract_block(task.rc3,revision=node.startswith('revision')) if rc else None
            converted = []
            for m in messages:
                chunks = m['content'].split(block) if block else [m['content']]
                converted.append({**m,'content':(block or '').join(
                    c.replace('SQLite','PostgreSQL').replace('sqlite','postgresql') for c in chunks)})
            messages = converted
        return messages


@lru_cache(maxsize=1)
def _parsers():
    root = Path(__file__).resolve().parents[3]
    generation = legacy_pure_functions(root,'sql_generation.py',['extract_initial_sql'])
    revision = legacy_pure_functions(root,'self_correction.py',['extract_corrected_sql'],generation)
    return {**generation,**revision,
            **legacy_pure_functions(root,'schema_linking.py',['extract_schema_links']),
            **legacy_pure_functions(root,'difficulty_decomposition.py',['parse_nested_output'])}


def parse_response(node, family, content, *, initial_sql=None):
    p = _parsers()
    warning, fallback = '', False
    if node=='linking':
        result = p['extract_schema_links'](content)
        warning = '' if result else 'Empty schema links retained under native behavior'
    elif node=='decomposition':
        result = {'label':'NESTED','sub_questions':p['parse_nested_output'](content)}
    elif node.startswith('revision') and family!='spider':
        result, warning = p['extract_corrected_sql'](content,initial_sql)
        fallback = bool(warning)
    else:
        if node.startswith('revision') and family=='spider':
            text = content.strip()
            fences = re.findall(r'```(?:sql)?\s*(.*?)```',text,re.I|re.S)
            if fences:
                text = fences[-1].strip()
            markers = list(re.finditer(r'\b(?:Revised_)?SQL\s*:\s*',text,re.I))
            if markers:
                text = text[markers[-1].end():].strip()
            # Official debuger ends in SELECT; some providers return the full query.
            content = text if re.match(r'^(?:SELECT|WITH)\b',text,re.I) else 'SELECT '+text
        result = p['extract_initial_sql'](content)
    return {'result':result,'fallback_used':fallback,'warnings':[warning] if warning else []}


def request_kwargs(node, family, messages, settings):
    kwargs = {'model':settings.model,'messages':messages,'temperature':settings.temperature,'n':1,'stream':False}
    if node=='linking':
        kwargs['max_tokens'] = 5000
    elif node=='decomposition':
        kwargs['max_tokens'] = 600 if family=='spider' else 2000
    if family=='spider':
        kwargs.update(top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,stop=['Q:'])
        if node.startswith('revision'):
            kwargs.update(max_tokens=350,stop=['#',';','\n\n'])
    return kwargs
