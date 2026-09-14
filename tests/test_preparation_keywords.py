"""Instance-only checkpoints around the real native keyword extraction method."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))

from app.config.config import LLMConfig
from app.dataset.dataset import DataItem
from app.llm import LLM
from app.llm_extractor import LLMExtractor
from app.pipeline.value_retrieval.value_retrieval import ValueRetrievalRunner
from app.prompt import PromptFactory
from scripts.baseline_adapters.deepeye import preparation_store as preparation


class KeywordCheckpointTests(unittest.TestCase):
    def item(self, number=7):
        return DataItem(question_id=number, question='Find red stars', evidence='red means color',
            database_id='a', database_path='unused', database_schema={}, gold_sql='SECRET GOLD')

    def runner(self, content='<result>["red stars"]</result>'):
        runner = object.__new__(ValueRetrievalRunner)
        runner._llm = LLM(LLMConfig(model='offline', api_key='SECRET KEY',
            base_url='https://offline.invalid/v1', fix_end_token=False, temperature=.6))
        runner._llm.sample_max_attempts = 1
        runner._extractor_max_retry = 0
        runner._keyword_extractor = LLMExtractor(max_retry=0)
        runner._llm.request_once = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
        return runner

    def bind(self, runner, root, **kwargs):
        helper = getattr(preparation, 'bind_keyword_checkpoints', None)
        self.assertTrue(callable(helper), 'Keyword checkpoint binding is missing')
        return helper(runner, root, **kwargs)

    def test_restart_after_later_failure_reuses_native_result_and_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root, item = Path(temp), self.item()
            runner = self.runner()
            self.bind(runner, root)
            with patch('app.pipeline.value_retrieval.value_retrieval.embed_keywords',
                       side_effect=RuntimeError('embedding failed')):
                with self.assertRaisesRegex(RuntimeError, 'embedding failed'):
                    runner._retrieve_values_for_item(item)
            keywords, usage = item.question_keywords, item.value_retrieval_llm_cost
            self.assertCountEqual(keywords, ['red', 'stars', 'red stars'])
            self.assertEqual(usage['total_tokens'], 5)
            restarted = self.runner()
            restarted._llm.request_once.side_effect = AssertionError('must reuse native keywords')
            self.bind(restarted, root)
            self.assertEqual(restarted._extract_keywords(item), (keywords, usage))
            self.assertEqual(runner._llm.request_once.call_count, 1)

    def test_changed_public_inputs_configuration_and_rules_are_rejected(self):
        changes = {
            'question': lambda runner, item: setattr(item, 'question', 'changed'),
            'evidence': lambda runner, item: setattr(item, 'evidence', 'changed'),
            'db': lambda runner, item: setattr(item, 'database_id', 'changed'),
            'model': lambda runner, item: setattr(runner._llm.llm_config, 'model', 'changed'),
            'temperature': lambda runner, item: setattr(runner._llm.llm_config, 'temperature', .2),
            'fix_end_token': lambda runner, item: setattr(runner._llm.llm_config, 'fix_end_token', True),
            'max_retry': lambda runner, item: setattr(runner, '_extractor_max_retry', 2),
            'extractor_retry': lambda runner, item: setattr(runner._keyword_extractor, '_max_retry', 2),
            'sample_attempts': lambda runner, item: setattr(runner._llm, 'sample_max_attempts', 4),
        }
        for name, change in changes.items():
            with self.subTest(field=name), tempfile.TemporaryDirectory() as temp:
                runner, item = self.runner(), self.item()
                self.bind(runner, Path(temp))
                runner._extract_keywords(item)
                change(runner, item)
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    runner._extract_keywords(item)
                self.assertEqual(runner._llm.request_once.call_count, 1)

    def test_actual_native_prompt_changes_invalidate_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, item = self.runner(), self.item()
            self.bind(runner, Path(temp))
            runner._extract_keywords(item)
            with patch.object(PromptFactory, 'format_keywords_extraction_prompt', return_value='changed prompt'):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    runner._extract_keywords(item)
            self.assertEqual(runner._llm.request_once.call_count, 1)

    def test_gold_and_rotated_api_key_do_not_change_identity_or_enter_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, item = self.runner(), self.item()
            self.bind(runner, Path(temp))
            expected = runner._extract_keywords(item)
            item.gold_sql = 'DIFFERENT SECRET GOLD'
            runner._llm.llm_config.api_key = 'ROTATED SECRET KEY'
            self.assertEqual(runner._extract_keywords(item), expected)
            self.assertEqual(runner._llm.request_once.call_count, 1)
            stored = ''.join(path.read_text() for path in Path(temp).rglob('*.json'))
            for secret in ('SECRET GOLD', 'SECRET KEY', 'ROTATED'):
                self.assertNotIn(secret, stored)

    def test_native_fallback_and_empty_tokens_are_preserved_without_strict_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, item = self.runner('malformed output'), self.item()
            item.evidence = ''
            self.bind(runner, Path(temp))
            expected = runner._extract_keywords(item)
            self.assertIn('', expected[0])
            self.assertCountEqual(expected[0], ['Find', 'red', 'stars', ''])
            self.assertEqual(runner._extract_keywords(item), expected)
            self.assertEqual(runner._llm.request_once.call_count, 1)

    def test_instance_only_binding_and_return_values_do_not_alias_across_items(self):
        method = ValueRetrievalRunner._extract_keywords
        with tempfile.TemporaryDirectory() as temp:
            runner = self.runner()
            self.bind(runner, Path(temp))
            first = runner._extract_keywords(self.item(7))
            expected = deepcopy(first)
            first[0].append('mutated')
            first[1]['total_tokens'] = 999
            second = runner._extract_keywords(self.item(8))
            self.assertEqual(second, expected)
            self.assertEqual(runner._extract_keywords(self.item(7)), expected)
            self.assertIs(ValueRetrievalRunner._extract_keywords, method)
            other = self.runner()
            self.assertIs(other._extract_keywords.__func__, method)
            self.assertEqual(runner._llm.request_once.call_count, 2)

    def test_audit_context_has_original_id_and_keyword_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root, runner = Path(temp), self.runner()
            audit = preparation.PreparationAudit(root / 'audit.jsonl')
            original = runner._llm.request_once

            def request(*args, **kwargs):
                audit.emit({'kind': 'observed'})
                return original(*args, **kwargs)

            runner._llm.request_once = request
            self.bind(runner, root / 'checkpoints', audit=audit,
                label={'benchmark': 'bird', 'split': 'dev', 'item': 'wrong', 'step': 'wrong'})
            runner._extract_keywords(self.item())
            import json
            event = json.loads(audit.path.read_text())
            self.assertEqual(event['label'], {'benchmark': 'bird', 'split': 'dev', 'item': 7, 'step': 'keywords'})

    def test_concurrent_same_item_calls_share_checkpoint_without_shared_mutables(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, item = self.runner(), self.item()
            self.bind(runner, Path(temp))
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(runner._extract_keywords, [item] * 4))
            self.assertEqual(runner._llm.request_once.call_count, 1)
            self.assertTrue(all(result == results[0] for result in results))
            self.assertIsNot(results[0][0], results[1][0])
            self.assertIsNot(results[0][1], results[1][1])


if __name__ == '__main__':
    unittest.main()
