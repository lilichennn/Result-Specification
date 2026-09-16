import unittest
from scripts.rc_evaluation.din_sql.evaluation import evaluate_pair, summarize_stage, normalize_usage


class EvaluationTests(unittest.TestCase):
    def test_column_position_duplicates_and_empty_shape(self):
        gold = {'status':'success','columns':['a','b'],'rows':[(1,2),(1,2)]}
        for wrong in ({**gold,'rows':[(2,1),(2,1)]},{**gold,'rows':[(1,2)]},
                      {**gold,'columns':['a'],'rows':[(1,),(1,)]}):
            pair = evaluate_pair(wrong,gold,gold)
            self.assertFalse(pair['base_correct'])
            self.assertTrue(pair['rc_correct'])
        empty = {**gold,'rows':[]}
        self.assertFalse(evaluate_pair({**empty,'columns':['a']},empty,empty)['base_correct'])

    def test_sql_error_false_but_timeout_connection_and_bad_gold_unknown(self):
        good = {'status':'success','columns':['a'],'rows':[(1,)]}
        bad = {'status':'error','error':{'type':'OperationalError','sqlite_errorcode':1}}
        self.assertFalse(evaluate_pair(bad,good,good)['base_correct'])
        self.assertIsNone(evaluate_pair({'status':'timeout'},good,good)['base_correct'])
        self.assertIsNone(evaluate_pair({'status':'error','error':{'sqlstate':None,'type':'OperationalError'}},good,good)['base_correct'])
        self.assertIsNone(evaluate_pair(good,good,bad)['rc_correct'])

    def test_ratio_of_paired_means_not_mean_ratio(self):
        rows = [{'base_total':100,'rc_total':80},{'base_total':200,'rc_total':100},
                {'base_total':None,'rc_total':200}]
        result = summarize_stage(rows)
        self.assertEqual(result['token_saving_pct'],40.0)
        self.assertEqual(result['token_pairs'],2)

    def test_missing_usage_not_zero_and_reasoning_not_inferred(self):
        self.assertIsNone(normalize_usage(None)['total'])
        self.assertEqual(normalize_usage({'prompt_tokens':2,'completion_tokens':3})['total'],5)
        self.assertIsNone(normalize_usage({'prompt_tokens':2,'completion_tokens':3})['reasoning'])
        self.assertEqual(normalize_usage({'prompt_tokens':0,'completion_tokens':0,'total_tokens':0})['total'],0)
