"""Exercise native Generation, RC formatting and persisted final SDK requests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import deepeye_bird_interact_run as entry
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
from scripts.baseline_adapters.deepeye.run_usage import observed_usage
from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
from scripts.rc_evaluation.deepeye.source import snapshot_source
from scripts.rc_evaluation.deepeye.runner import run_experiment
from .test_source import make_source


class NativeRCIntegrationTests(unittest.TestCase):
    def test_native_generation_records_exact_rc_requests_and_none_is_unmodified(self):
        # Break caught: injection outside trace/token formatting, lost context in
        # native inner pools, a skipped target, or a control receiving the RC.
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from openai.types.chat import ChatCompletion
        from scripts.deepeye_bird_interact_smoke import build_runtime_config
        from scripts.rc_evaluation.deepeye.injection import install_rc_prompts, render_rc_block

        environment = {'DASH_MODELS': 'fixture', 'DASH_BASE_URL': 'https://invalid.test/v1',
                       'DASH_API_KEY': 'fixture', 'EMBEDDING_MODEL': 'fixture',
                       'EMBEDDING_BASE_URL': 'https://invalid.test/v1', 'EMBEDDING_API_KEY': 'fixture'}
        rc = {'task_key': 'lite/a', 'round2': {
            'population': 'eligible entities', 'row_grain': 'one entity',
            'column_role': 'requested measure', 'derivation': 'requested computation',
            'filter_policy': 'requested relative selection',
            'meta_review': 'No relevant metadata refinement; Round-1 contract preserved.'}}
        all_prompts = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = make_source(root / 'source', stage='sql_generation')
            rc.update(db_id=tasks[0][1].database_id, question=tasks[0][1].question,
                      evidence=tasks[0][1].evidence)
            source = snapshot_source(root / 'source', tasks, 'sql_generation')
            for condition in ('none', 'rc'):
                with self.subTest(condition=condition):
                    sent = []
                    def create(**kwargs):
                        sent.append(deepcopy(kwargs))
                        return ChatCompletion(id='offline', created=0, model='fixture', object='chat.completion',
                            choices=[{'index': 0, 'finish_reason': 'stop', 'message': {
                                'role': 'assistant', 'content': '<result>SELECT x FROM t</result>'}}],
                            usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5})
                    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                             close=lambda: None)
                    manifest = {**source, 'format': 'deepeye-rc-evaluation-run-v1',
                        'target_stage': 'sql_generation', 'condition': condition, 'repeat_id': '1',
                        'continue_downstream': False, 'contracts': {'lite/a': rc} if condition == 'rc' else {},
                        'effective_config': source['source_manifest']['effective_config'],
                        'sources': source['source_manifest']['sources']}
                    path = root / condition
                    config = build_runtime_config(environment, 'lite', root, path / 'native')
                    config.run_config.parallelism = 4
                    factory = entry.bounded_runner_factory(config, 1200)
                    undo = install_postgres_support()
                    try:
                        with patch.object(LLM, '_create_client', return_value=client), \
                             patch.object(SchemaService, '_get_encoding',
                                          return_value=SimpleNamespace(encode=lambda text: list(text))), \
                             patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), \
                             RunStore.create(path, manifest) as store:
                            recorder = TraceRecorder(store)
                            with recorder.install(), install_rc_prompts():
                                result = run_experiment(store, factory, recorder, workers=1)
                            self.assertEqual(result['succeeded'], 1)
                            self.assertEqual(result['failed'], 0)
                            stage = next(row for row in store.attempts() if row['stage'] == 'sql_generation')
                            self.assertEqual(len(stage['payload']['artifact']['sql_candidates']), 3)
                            api_events = [event for event in store.events() if event['kind'] == 'api_request']
                            self.assertEqual(len(sent), 3)
                            self.assertEqual(len(api_events), 3)
                            wire = sorted(request['messages'][0]['content'] for request in sent)
                            disk = sorted(event['payload']['kwargs']['messages'][0]['content'] for event in api_events)
                            self.assertEqual(wire, disk)
                            all_prompts[condition] = wire
                            block = render_rc_block(rc)
                            self.assertEqual(sum(block in prompt for prompt in disk), 3 if condition == 'rc' else 0)
                            self.assertEqual(stage['payload']['rc_participation']['actual_request_count'],
                                             3 if condition == 'rc' else 0)
                            self.assertEqual(observed_usage(store)['reported_tokens']['total_tokens'], 15)
                            self.assertTrue(store.verify()['ok'])
                            before = store.attempts()
                            with recorder.install(), install_rc_prompts():
                                run_experiment(store, lambda *args: self.fail('completed run created runner'),
                                               recorder, workers=1)
                            self.assertEqual(store.attempts(), before)
                    finally:
                        undo()
            block = render_rc_block(rc)
            # Native LLM.ask strips the full prompt. Appending a block retains
            # the old template's trailing newline internally, not extra advice.
            self.assertEqual(all_prompts['none'], sorted(prompt.removesuffix('\n\n' + block).rstrip()
                                                        for prompt in all_prompts['rc']))


if __name__ == '__main__':
    unittest.main()
