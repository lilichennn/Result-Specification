import unittest
from pathlib import Path
from scripts.baseline_adapters.din_sql.inputs import DinSettings, PreparedInputs, load_templates
from scripts.baseline_adapters.din_sql.prompts import PromptBuilder, parse_response, request_kwargs
from din_sql_fixtures import make_task


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.templates = load_templates(cls.root)

    def test_native_fallback_and_spider_continuation(self):
        parsed = parse_response('revision_base','bird','No changes needed.',initial_sql='SELECT 1')
        self.assertEqual(parsed['result'],'SELECT 1')
        self.assertTrue(parsed['fallback_used'])
        self.assertEqual(parse_response('revision_base','spider',' count(*) FROM t')['result'],'SELECT count(*) FROM t')
        self.assertEqual(parse_response('revision_base','spider','WITH c AS (SELECT 1) SELECT * FROM c')['result'],
                         'WITH c AS (SELECT 1) SELECT * FROM c')
        for continuation in ('name FROM a WHERE id IN (SELECT id FROM b)',
                             "name FROM a WHERE label = 'With bacon'", '```sql\ncount(*) FROM t\n```'):
            expected = 'SELECT '+continuation.replace('```sql\n','').replace('\n```','')
            self.assertEqual(parse_response('revision_base','spider',continuation)['result'],expected)

    def test_rc_only_difference_for_all_families_and_difficulties(self):
        for group in ('bird_dev','spider_dev','bird_interact_full'):
            for label in ('EASY','NON-NESTED','NESTED'):
                task = make_task(label,group)
                prepared = PreparedInputs({task.key:task},
                    {'scores':{'context':'Table scores, columns = [*,value]','spider':'Table scores, columns = [*,value]\nForeign_keys = []'},
                     'spider:college_2':'Table college, columns = [*,id]\nForeign_keys = []'},
                    {},self.templates,{}, {})
                builder = PromptBuilder(prepared,self.root)
                parents = {'linking':{'result':['scores.value']},
                           'decomposition':{'result':{'label':label,'sub_questions':['What is the score?']}},
                           'generation_base':{'result':'SELECT MAX(value) FROM scores'}}
                for stage in ('generation','revision'):
                    base = builder.build(stage+'_base',task,parents)
                    rc = builder.build(stage+'_rc3',task,parents)
                    block = builder.contract_block(task.rc3,revision=stage=='revision')
                    recovered = [{**m,'content':m['content'].replace(block,'')} for m in rc]
                    self.assertEqual(base,recovered,(group,label,stage))

    def test_request_parameters_and_no_forced_thinking(self):
        for family in ('bird','spider','postgresql'):
            kwargs = request_kwargs('generation_rc3',family,[],DinSettings())
            self.assertEqual(kwargs['n'],1)
            self.assertFalse(kwargs['stream'])
            for field in ('max_tokens','stream_options','extra_body','enable_thinking'):
                self.assertNotIn(field,kwargs)
        self.assertEqual(request_kwargs('revision_base','spider',[],DinSettings())['max_tokens'],350)
        self.assertEqual(request_kwargs('decomposition','spider',[],DinSettings())['max_tokens'],600)
