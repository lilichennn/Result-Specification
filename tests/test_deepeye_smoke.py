from pathlib import Path
import importlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    smoke = importlib.import_module("scripts.deepeye_bird_interact_smoke")
except ModuleNotFoundError as exc:
    if exc.name != "scripts.deepeye_bird_interact_smoke":
        raise
    smoke = None


class SmokeEntrypointTest(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(smoke, "Smoke integration entry point is not implemented")
        return smoke

    def env(self):
        return {
            "DASH_BASE_URL": "https://chat.example.invalid/v1", "DASH_API_KEY": "sk-chat-secret",
            "DASH_MODELS": "chat-model", "EMBEDDING_BASE_URL": "https://embed.example.invalid/v1",
            "EMBEDDING_API_KEY": "sk-embed-secret", "EMBEDDING_MODEL": "qwen3.7-text-embedding-flash",
            "PG_HOST": "127.0.0.1", "PG_PORT": "5432", "PG_USER": "test_reader", "PG_PASSWORD": "pg-secret",
        }

    def test_reversed_model_and_key_are_rejected_without_leaking_values(self):
        module = self.require_module()
        values = self.env()
        values["EMBEDDING_MODEL"], values["EMBEDDING_API_KEY"] = values["EMBEDDING_API_KEY"], values["EMBEDDING_MODEL"]
        with self.assertRaises(ValueError) as context:
            module.validate_environment(values)
        self.assertNotIn("sk-embed-secret", str(context.exception))

    def test_redaction_hides_secrets_but_keeps_diagnostics(self):
        module = self.require_module()
        text = "failed: sk-chat-secret and pg-secret; HTTP 401"
        safe = module.redact(text, self.env())
        self.assertNotIn("sk-chat-secret", safe)
        self.assertNotIn("pg-secret", safe)
        self.assertIn("HTTP 401", safe)

    def test_output_creation_never_overwrites_prior_run(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "run"
            module.create_output_directory(output)
            marker = output / "result.json"
            marker.write_text("existing result")
            with self.assertRaises(FileExistsError):
                module.create_output_directory(output)
            self.assertEqual(marker.read_text(), "existing result")

    def test_config_keeps_all_generators_and_cpu_lookup_without_secret_snapshot(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as parent:
            cfg = module.build_runtime_config(self.env(), "lite", Path(parent), Path(parent) / "out")
            self.assertEqual(cfg.dataset_config.type, "bird_interact")
            self.assertEqual(cfg.run_config.embedding_batch_size, 20)
            self.assertEqual(cfg.value_retrieval_config.local_index_device, "cpu")
            for attr in ("dc_sampling_budget", "skeleton_sampling_budget", "icl_sampling_budget"):
                self.assertGreater(getattr(cfg.sql_generation_config, attr), 0)
            for secret in ("sk-chat-secret", "sk-embed-secret", "pg-secret"):
                self.assertNotIn(secret, str(cfg.dataset_config.model_dump()))

    def test_budget_rejection_is_recorded_even_if_caller_swallows_exception(self):
        recorder = self.require_module().CallRecorder(max_api_calls=0)
        with self.assertRaises(RuntimeError):
            recorder.wrap_api(lambda: None, "chat")()
        self.assertEqual(recorder.events[0]["kind"], "budget_exhausted")
        self.assertFalse(recorder.events[0]["success"])
        self.assertTrue(smoke.observation_failures(recorder.events))

    def test_selection_returning_no_votes_is_not_successful_coverage(self):
        recorder = self.require_module().CallRecorder()
        owner = SimpleNamespace(compare=lambda: ([], {"total_tokens": 0}))
        recorder.observe_method(owner, "compare", "selection.pairwise_comparison", count_output=True)
        owner.compare()
        self.assertEqual(recorder.events[0]["output_count"], 0)
        self.assertTrue(smoke.observation_failures(recorder.events))

    def test_successful_vote_and_shortcut_without_comparison_are_valid(self):
        recorder = self.require_module().CallRecorder()
        owner = SimpleNamespace(compare=lambda: (["TIE"], {"total_tokens": 1}))
        recorder.observe_method(owner, "compare", "selection.pairwise_comparison", count_output=True)
        owner.compare()
        self.assertEqual(smoke.observation_failures(recorder.events), [])
        self.assertEqual(smoke.observation_failures([]), [])

    def test_stage_timeout_is_configurable_without_changing_native_llm(self):
        module = self.require_module()
        from app.llm import LLM
        from tests.test_deepeye_sampling import llm_fixture, response
        native = LLM.request_once
        llm, calls = llm_fixture([response()])
        runner = SimpleNamespace(_llm=llm)
        module.configure_stage_calls(runner, "test", module.CallRecorder(), chat_timeout=300)
        llm.ask([])
        self.assertEqual(calls[0]['timeout'], 300)
        self.assertEqual(llm.sample_max_attempts, 4)
        self.assertIs(LLM.request_once, native)

    def test_thinking_budget_is_explicit_and_shared_by_all_stages(self):
        module = self.require_module()
        cfg = module.build_runtime_config(self.env(), "lite", Path("/tmp/input"), Path("/tmp/output"), thinking_budget=1024)
        for stage in module.STAGES:
            self.assertEqual(getattr(cfg, f"{stage}_config").llm.extra_body, {"thinking_budget": 1024})
        native = module.build_runtime_config(self.env(), "lite", Path("/tmp/input"), Path("/tmp/output"))
        self.assertEqual(native.sql_generation_config.llm.extra_body, {})

    def test_empty_extraction_is_observed_for_keywords_and_revision(self):
        module = self.require_module()
        from app.llm import LLM
        from app.llm_extractor import LLMExtractor
        for stage in ("value_retrieval", "sql_revision"):
            with self.subTest(stage=stage):
                recorder = module.CallRecorder()
                from tests.test_deepeye_sampling import llm_fixture, response
                llm, calls = llm_fixture([response('unparseable')] * 4)
                extractor = LLMExtractor(max_retry=2)
                checker = SimpleNamespace(_extractor=extractor, check_and_revise=lambda: None)
                runner = SimpleNamespace(_llm=llm, _keyword_extractor=extractor, _checkers=[checker],
                    _embedding_function=SimpleNamespace(client=SimpleNamespace(embeddings=SimpleNamespace(create=lambda: None))))
                module.configure_stage_calls(runner, stage, recorder)
                result, _ = extractor.extract_with_retry(llm=llm, messages=[], rule_parser=lambda _: None)
                self.assertEqual(len(calls), 4)
                self.assertEqual(result, [])
                self.assertTrue(module.observation_failures(recorder.events))

    def test_independent_example_reader_reuses_source_and_translation_without_aliasing(self):
        """Removing the reader snapshot/cache or copies must add I/O/transpiles or leak edits."""
        module = self.require_module()
        rows = [
            {'db_id': 'a', 'question': 'train a', 'evidence': 'ea', 'SQL': 'SELECT 1'},
            {'db_id': 'b', 'question': 'train b', 'evidence': 'eb', 'SQL': 'SELECT 2'},
            {'db_id': 'c', 'question': 'train c', 'evidence': 'ec', 'SQL': 'SELECT 3'},
        ]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'train.json'
            source.write_text(json.dumps(rows))
            read_count = 0
            native_read_bytes = Path.read_bytes

            def traced_read_bytes(path):
                nonlocal read_count
                if Path(path) == source:
                    read_count += 1
                return native_read_bytes(path)

            import sqlglot
            native_transpile = sqlglot.transpile
            transpile_count = 0

            def traced_transpile(*args, **kwargs):
                nonlocal transpile_count
                transpile_count += 1
                return native_transpile(*args, **kwargs)

            with patch.object(Path, 'read_bytes', traced_read_bytes), \
                 patch.object(sqlglot, 'transpile', traced_transpile):
                reader = module.IndependentExampleReader(source)
                first, provenance = reader.select(SimpleNamespace(
                    database_id='target-one', question='first target'), 3)
                second, second_provenance = reader.select(SimpleNamespace(
                    database_id='target-two', question='second target'), 3)

            self.assertEqual(read_count, 1)
            self.assertEqual(transpile_count, 3)
            self.assertEqual([row['db_id'] for row in provenance['examples']], ['a', 'b', 'c'])
            self.assertEqual([row['db_id'] for row in second_provenance['examples']], ['a', 'b', 'c'])
            self.assertIsNot(first, second)
            first[0]['question'] = 'private'
            provenance['examples'][0]['db_id'] = 'private'
            self.assertEqual(second[0]['question'], 'train a')
            self.assertEqual(second_provenance['examples'][0]['db_id'], 'a')

            original_hash = second_provenance['source_sha256']
            rows[0]['question'] = 'changed on disk'
            source.write_text(json.dumps(rows))
            changed, changed_provenance = module.IndependentExampleReader(source).select(
                SimpleNamespace(database_id='target-three', question='third target'), 3)
            self.assertNotEqual(changed_provenance['source_sha256'], original_hash)
            self.assertEqual(changed[0]['question'], 'changed on disk')

    def test_independent_examples_keep_source_order_and_target_exclusions(self):
        """Weakening target/domain exclusions or reordering candidates must change this result."""
        module = self.require_module()
        rows = [
            {'db_id': 'target', 'question': 'different', 'SQL': 'SELECT 0'},
            {'db_id': 'x', 'question': ' target question ', 'SQL': 'SELECT 0'},
            {'db_id': 'a', 'question': 'train a', 'SQL': 'SELECT 1'},
            {'db_id': 'a', 'question': 'duplicate domain', 'SQL': 'SELECT 9'},
            {'db_id': 'broken', 'question': 'broken', 'SQL': 'not valid sql ('},
            {'db_id': 'b', 'question': 'train b', 'SQL': 'SELECT 2'},
            {'db_id': 'c', 'question': 'train c', 'SQL': 'SELECT 3'},
        ]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'train.json'
            source.write_text(json.dumps(rows))
            examples, provenance = module.load_independent_examples(
                source, SimpleNamespace(database_id='target', question='target question'))
        self.assertEqual([row['question'] for row in examples], ['train a', 'train b', 'train c'])
        self.assertEqual([row['source_row'] for row in provenance['examples']], [2, 5, 6])


if __name__ == "__main__":
    unittest.main()
