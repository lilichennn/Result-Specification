from pathlib import Path
import importlib
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
        native = LLM.ask
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda: None)))
        llm = SimpleNamespace(_get_client=lambda: client)
        runner = SimpleNamespace(_llm=llm)
        with patch.object(LLM, "ask") as patched:
            module.configure_stage_calls(runner, "test", module.CallRecorder(), chat_timeout=300)
            llm.ask([])
            self.assertEqual(patched.retry_with.return_value.call_args.kwargs["timeout"], 300)
        self.assertIs(LLM.ask, native)

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
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda: None)))
                llm = SimpleNamespace(_get_client=lambda: client)
                extractor = LLMExtractor(max_retry=2)
                checker = SimpleNamespace(_extractor=extractor, check_and_revise=lambda: None)
                runner = SimpleNamespace(_llm=llm, _keyword_extractor=extractor, _checkers=[checker],
                    _embedding_function=SimpleNamespace(client=SimpleNamespace(embeddings=SimpleNamespace(create=lambda: None))))
                with patch.object(LLM, "ask") as patched:
                    patched.retry_with.return_value.return_value = ([SimpleNamespace(content="unparseable")],
                        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
                    module.configure_stage_calls(runner, stage, recorder)
                    result, _ = extractor.extract_with_retry(llm=llm, messages=[], rule_parser=lambda _: None)
                self.assertEqual(result, [])
                self.assertTrue(module.observation_failures(recorder.events))


if __name__ == "__main__":
    unittest.main()
