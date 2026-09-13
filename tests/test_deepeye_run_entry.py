from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


try:
    from scripts import deepeye_bird_interact_run as entry
except ImportError:
    entry = None


class DeepEyeRunEntryTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(entry, "DeepEye durable run CLI is missing")
        return entry

    def args(self, **overrides):
        values = {
            "workers": 4,
            "max_tokens": 6144,
            "thinking_budget": None,
            "chat_timeout": 300,
            "extractor_retries": 2,
            "direct_linking_budget": 1,
            "reversed_linking_budget": 1,
            "dc_generation_budget": 1,
            "skeleton_generation_budget": 1,
            "icl_generation_budget": 1,
            "revision_checker_budget": 1,
            "selection_evaluator_budget": 1,
            "pg_sslmode": "prefer",
            "pg_concurrency": 10,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_effective_config_binds_bounded_choices_but_not_rotatable_credentials(self):
        first = {
            "DASH_BASE_URL": "https://chat.example/v1",
            "DASH_MODELS": "chat-model",
            "DASH_API_KEY": "first-chat-secret",
            "EMBEDDING_BASE_URL": "https://embedding.example/v1",
            "EMBEDDING_MODEL": "embedding-model",
            "EMBEDDING_API_KEY": "first-embedding-secret",
            "PG_HOST": "db.example",
            "PG_PORT": "5432",
            "PG_USER": "reader",
            "PG_PASSWORD": "first-db-secret",
        }
        rotated = dict(first, DASH_API_KEY="rotated-chat", EMBEDDING_API_KEY="rotated-embedding",
                       PG_PASSWORD="rotated-db")

        module = self.module()
        actual = module.build_effective_config(first, self.args())

        self.assertEqual(actual, module.build_effective_config(rotated, self.args()))
        self.assertNotEqual(actual, module.build_effective_config({**rotated, "PG_USER": "other-role"}, self.args()))
        self.assertEqual(actual["profile"], "bounded-config")
        self.assertEqual(actual["scheduler"], {
            "mode": "pipeline_slots", "concurrency_unit": "question_pipeline",
        })
        self.assertNotIn("precomputed_embedding", actual)
        self.assertEqual(actual["stages"]["schema_linking"],
                         {"direct_linking_sampling_budget": 1,
                          "reversed_linking_sampling_budget": 1,
                          "value_distance_threshold": 0.05})
        self.assertEqual(actual["stages"]["sql_generation"],
                         {"dc_sampling_budget": 1, "skeleton_sampling_budget": 1,
                          "icl_sampling_budget": 1})
        self.assertEqual(actual["stages"]["sql_revision"]["checkers"], [
            "SyntaxChecker", "JoinChecker", "OrderByLimitChecker", "TimeChecker",
            "SelectChecker", "MaxMinChecker", "OrderByNullChecker", "ResultChecker",
        ])
        self.assertEqual(actual["stages"]["schema_linking"]["value_distance_threshold"], 0.05)
        self.assertEqual(actual["stages"]["sql_selection"]["filter_top_k_sql"], 2)
        self.assertEqual(actual["dataset"], {"sql_execution_timeout_seconds": 30,
                                             "max_value_example_length": 100})
        self.assertEqual(actual["postgres"]["principal"], "reader")
        self.assertEqual(actual["postgres"]["sslmode"], "prefer")
        # Omitting execution semantics would let adapter upgrades inherit guard-era results.
        self.assertEqual(actual["postgres"].get("execution_policy"), {
            "version": "postgres-original-sql-v1",
            "meta_fence": False,
            "sql_rewrite": False,
            "result_row_limit": None,
            "search_path": "pg_catalog,public",
            "single_statement": "extended_protocol",
            "read_only": True,
        })
        encoded = json.dumps(actual, sort_keys=True)
        for secret in ("first-chat-secret", "first-embedding-secret", "first-db-secret"):
            self.assertNotIn(secret, encoded)

    def test_manifest_changes_when_a_question_artifact_or_source_changes(self):
        module = self.module()
        config = module.build_effective_config({
            "DASH_BASE_URL": "https://chat.example/v1", "DASH_MODELS": "chat-model",
            "DASH_API_KEY": "secret", "EMBEDDING_BASE_URL": "https://embed.example/v1",
            "EMBEDDING_MODEL": "embed-model", "EMBEDDING_API_KEY": "secret",
            "PG_HOST": "db.example", "PG_PORT": "5432", "PG_USER": "reader",
            "PG_PASSWORD": "secret",
        }, self.args())
        sources = {"precompute_inputs_sha256": "a", "precompute_config_sha256": "b",
                   "few_shot_source_sha256": "c", "code_sha256": "d"}
        bindings = [{"task_key": "lite/a_1", "question_sha256": "q",
                     "schema_sha256": "s", "retrieval_sha256": "r1",
                     "keywords_sha256": "k", "few_shot_sha256": "f"}]

        first = module.build_manifest(config, sources, bindings)
        changed_bindings = [{**bindings[0], "retrieval_sha256": "r2"}]

        self.assertNotEqual(first, module.build_manifest(config, sources, changed_bindings))
        self.assertNotEqual(first, module.build_manifest(config, {**sources, "code_sha256": "new"}, bindings))
        self.assertEqual(first["item_count"], 1)
        self.assertFalse(first["accuracy_evaluated"])
        self.assertEqual(first["workflow"], ["schema_linking", "sql_generation",
                                              "sql_revision", "sql_selection"])

    def test_selection_runtime_and_manifest_use_the_same_template_threshold(self):
        # Catch an unoverridden native default or a manifest-only threshold change.
        environment = {
            "DASH_BASE_URL": "https://chat.example/v1", "DASH_MODELS": "chat-model",
            "DASH_API_KEY": "test-key", "EMBEDDING_BASE_URL": "https://embed.example/v1",
            "EMBEDDING_MODEL": "embed-model", "EMBEDDING_API_KEY": "test-key",
            "PG_HOST": "db.example", "PG_PORT": "5432", "PG_USER": "reader",
        }
        module = self.module()
        args = self.args(selection_evaluator_budget=5)
        recorded = module.build_effective_config(environment, args)["stages"]["sql_selection"]
        with tempfile.TemporaryDirectory() as temporary:
            runtime = module.build_runtime_config(environment, args, Path(temporary)).sql_selection_config
        self.assertEqual(runtime.shortcut_consistency_score_threshold, 0.6)
        self.assertEqual(recorded["shortcut_consistency_score_threshold"], 0.6)
        self.assertEqual(runtime.filter_top_k_sql, recorded["filter_top_k_sql"])
        self.assertEqual(runtime.evaluator_sampling_budget, 5)
        self.assertEqual(recorded["evaluator_sampling_budget"], 5)

    def test_task_filter_is_exact_and_rejects_unknown_instance(self):
        tasks = [("lite", SimpleNamespace(instance_id="same")),
                 ("full", SimpleNamespace(instance_id="same")),
                 ("full", SimpleNamespace(instance_id="other"))]

        module = self.module()
        selected = module.select_tasks(tasks, variants=["full"], item_keys=["full/same"])

        self.assertEqual([(variant, item.instance_id) for variant, item in selected], [("full", "same")])
        with self.assertRaises(ValueError):
            module.select_tasks(tasks, variants=None, item_keys=["same"])
        with self.assertRaises(ValueError):
            module.select_tasks(tasks, variants=None, item_keys=["lite/missing"])

    def test_probe_selection_covers_databases_is_order_independent_and_bounded(self):
        tasks = [(variant, SimpleNamespace(instance_id=f"{db}_{i}", database_id=db))
                 for variant in ("lite", "full") for db in ("a", "b", "c")
                 for i in range(4)]
        selected = self.module().select_probe_tasks(tasks, per_variant=5)
        reversed_selection = self.module().select_probe_tasks(list(reversed(tasks)), per_variant=5)
        keys = lambda rows: [(v, x.instance_id) for v, x in rows]
        self.assertEqual(keys(selected), keys(reversed_selection))
        self.assertEqual(len(selected), 10)
        for variant in ("lite", "full"):
            rows = [x for v, x in selected if v == variant]
            self.assertEqual(len(rows), 5)
            self.assertEqual({x.database_id for x in rows}, {"a", "b", "c"})
        with self.assertRaises(ValueError):
            self.module().select_probe_tasks(tasks, per_variant=2)
        with self.assertRaises(ValueError):
            self.module().select_probe_tasks(tasks, per_variant=13)

    def test_adaptive_arguments_bind_policy_and_reject_unbounded_settings(self):
        module = self.module()
        parser = module._build_parser()
        common = ["run", "--run-dir", "unused", "--precompute-dir", "unused",
                  "--adaptive-concurrency", "--workers", "50", "--inner-workers", "100",
                  "--probe-per-variant", "25"]
        args = parser.parse_args(common)
        module._validate_run_args(parser, args)
        policy = module.admission_settings(args)
        self.assertEqual(policy["pipeline"]["initial_limit"], 50)
        self.assertEqual(policy["pipeline"]["step"], 10)
        self.assertEqual(policy["pipeline"]["stable_window_s"], 60.0)
        self.assertTrue(policy["adaptive"])
        self.assertEqual(policy["postgres_limit"], 10)
        self.assertEqual(module.admission_settings(self.args()), {
            "enabled": True, "adaptive": False,
            "pipeline": {"fixed_limit": 4}, "postgres_limit": 10,
        })
        for extra in (["--concurrency-max", "40"], ["--concurrency-min", "0"],
                      ["--concurrency-window", "0"], ["--inner-workers", "0"],
                      ["--pg-concurrency", "0"], ["--item", "lite/x"]):
            with self.assertRaises(SystemExit):
                module._validate_run_args(parser, parser.parse_args(common + extra))

    def test_admission_context_limits_real_boundary_and_restores_after_failure(self):
        from scripts.baseline_adapters.deepeye import backend_hooks
        module = self.module()
        events = []
        recorder = SimpleNamespace(api_call=None,
                                   record_admission=lambda kind, payload: events.append((kind, payload)))
        args = module._build_parser().parse_args([
            "run", "--run-dir", "unused", "--precompute-dir", "unused", "--adaptive-concurrency"])
        original = backend_hooks.execute_postgres_sql
        pg_result = object()
        pg_calls = []
        def pg_boundary(*args, **kwargs):
            pg_calls.append((args, kwargs))
            return pg_result
        with self.assertRaisesRegex(RuntimeError, "pipeline fails"):
            with patch.object(backend_hooks, "execute_postgres_sql", pg_boundary):
                with module.admission_context(recorder, args) as controllers:
                    self.assertIsNot(backend_hooks.execute_postgres_sql, pg_boundary)
                    token = object()
                    self.assertIs(recorder.api_call(lambda: token, (), {}), token)
                    self.assertEqual(controllers["pipeline"].snapshot()["model"]["completed"], 1)
                    self.assertEqual(controllers["pipeline"].snapshot()["completed"], 0)
                    self.assertIs(backend_hooks.execute_postgres_sql("item", "SELECT 1", timeout=3), pg_result)
                    self.assertEqual(pg_calls, [(("item", "SELECT 1"), {"timeout": 3})])
                    self.assertEqual(controllers["postgres"].snapshot()["completed"], 1)
                    raise RuntimeError("pipeline fails")
        self.assertIs(backend_hooks.execute_postgres_sql, original)
        self.assertIsNone(recorder.api_call)
        self.assertEqual([kind for kind, _ in events],
                         ["api_admission", "api_completion", "postgres_admission", "postgres_completion"])

    def test_fixed_slots_also_limit_postgres_and_observe_model_requests(self):
        recorder = SimpleNamespace(api_call=None, record_admission=lambda *args: None)
        with self.module().admission_context(recorder, self.args(workers=2)) as controllers:
            slots = controllers["pipeline"]
            self.assertEqual(slots.snapshot()["current_limit"], 2)
            self.assertEqual(controllers["postgres"].snapshot()["current_limit"], 10)
            self.assertEqual(recorder.api_call(lambda: 42, (), {}), 42)
            self.assertEqual(slots.snapshot()["model"]["completed"], 1)
        self.assertIsNone(recorder.api_call)

    def test_inheritance_is_explicit_and_not_a_resume_manifest_bypass(self):
        parser = self.module()._build_parser()
        common = ["--run-dir", "new", "--precompute-dir", "frozen", "--inherit-from", "old"]
        for command in ("prepare", "run"):
            args = parser.parse_args([command, *common])
            self.assertEqual(args.inherit_from, Path("old"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["resume", *common])

    def test_prepare_inputs_loads_only_frozen_retrieval_and_three_independent_examples(self):
        module = self.module()
        self.assertTrue(hasattr(module, "prepare_inputs"), "Offline precomputed-input preparation is missing")
        from scripts.baseline_adapters.deepeye.precompute_pipeline import write_record

        original = SimpleNamespace(instance_id="a_1", database_id="a", question="question",
                                   evidence="evidence", database_schema={"tables": {}}, gold_sql="")
        loaded = SimpleNamespace(**vars(original))
        examples = [{"question": f"train {number}", "evidence": "", "sql": "SELECT 1"}
                    for number in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            precompute = root / "precomputed"
            question_dir = precompute / "questions" / "lite" / "a_1"
            input_record = write_record(precompute / "inputs.json", {"questions": ["fixture"]})
            config_record = write_record(precompute / "run_config.json", {"config": {
                "version": 2, "embedding": {"model": "frozen-embedding"},
            }})
            write_record(question_dir / "keywords.json", {"keywords": ["question"]})
            write_record(question_dir / "retrieval.json", {"retrieved_values": {}})
            (precompute / "verification.json").write_text(json.dumps({
                "complete": True, "question_counts": {"lite": 195, "full": 410},
                "expected_questions": 605, "database_count": 40, "errors": {},
                "verified_at": "first", "verification_seconds": 1.0,
            }))
            train = root / "train.json"
            train.write_text('[{"fixture": true}]')

            tasks, bindings, sources = module.prepare_inputs(
                precompute, train, variants=["lite"], item_keys=["lite/a_1"],
                inventory_loader=lambda: ([("lite", original)], {"a": original}),
                item_loader=lambda root, variant, instance_id, expected_item: loaded,
                example_loader=lambda source, item, count=3: (examples, {
                    "source_path": str(source), "source_sha256": "train",
                    "examples": [{"source_row": number, "db_id": f"db{number}",
                                  "original_sql": "SELECT 1",
                                  "dialect_conversion": "sqlglot sqlite -> postgres"}
                                 for number in range(3)],
                }),
                code_source_hasher=lambda: {"baseline_python_sha256": "code"},
            )

            verification = json.loads((precompute / "verification.json").read_text())
            verification.update(verified_at="second", verification_seconds=99.0)
            (precompute / "verification.json").write_text(json.dumps(verification))
            _, second_bindings, second_sources = module.prepare_inputs(
                precompute, train, variants=["lite"], item_keys=["lite/a_1"],
                inventory_loader=lambda: ([("lite", original)], {"a": original}),
                item_loader=lambda root, variant, instance_id, expected_item: loaded,
                example_loader=lambda source, item, count=3: (examples, {
                    "source_path": str(source), "source_sha256": "train",
                    "examples": [{"source_row": number, "db_id": f"db{number}",
                                  "original_sql": "SELECT 1",
                                  "dialect_conversion": "sqlglot sqlite -> postgres"}
                                 for number in range(3)],
                }),
                code_source_hasher=lambda: {"baseline_python_sha256": "code"},
            )

        self.assertEqual(tasks, [("lite", loaded)])
        self.assertEqual(loaded.few_shot_examples, examples)
        self.assertEqual(loaded.few_shot_preparation_metadata,
                         {"mode": "static_independent_bird_train", "num_examples": 3})
        self.assertEqual(bindings[0]["task_key"], "lite/a_1")
        self.assertEqual(bindings[0]["database_id"], "a")
        self.assertEqual(bindings[0]["few_shot_source_rows"], [
            {"source_row": number, "db_id": f"db{number}",
             "dialect_conversion": "sqlglot sqlite -> postgres"}
            for number in range(3)
        ])
        self.assertEqual(bindings, second_bindings)
        self.assertEqual(sources, second_sources)
        self.assertEqual(sources["few_shot_source_sha256"], hashlib.sha256(b'[{"fixture": true}]').hexdigest())
        self.assertEqual(sources["precompute_inputs_content_hash"], input_record["content_hash"])
        self.assertEqual(sources["precompute_config_content_hash"], config_record["content_hash"])
        self.assertEqual(sources["precompute_semantic_config"]["embedding"]["model"], "frozen-embedding")
        self.assertNotIn("precompute_verification_sha256", sources)
        self.assertEqual(sources["precompute_population"], {"lite": 195, "full": 410, "databases": 40})
        self.assertEqual(sources["code"], {"baseline_python_sha256": "code"})
        self.assertIn("locators", sources, "Replay source locators are not persisted")
        self.assertEqual(sources["locators"], {"precompute_dir": str(precompute.resolve()),
                                               "few_shot_source": str(train.resolve())})

    def test_inspect_and_export_need_neither_environment_nor_precompute_tree(self):
        module = self.module()
        from scripts.baseline_adapters.deepeye.run_store import RunStore

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with RunStore.create(root / "run", {"kind": "fixture"}) as store:
                attempt = store.begin_attempt("lite/a_1", "schema_linking", "fp")
                store.finish_attempt(attempt, "succeeded", {"ok": True})

            inspected = module.main(["inspect", "--run-dir", str(root / "run")])
            exported = module.main(["export", "--run-dir", str(root / "run"),
                                   "--export-dir", str(root / "export")])

            self.assertEqual(inspected, 0)
            self.assertEqual(exported, 0)
            self.assertTrue((root / "export" / "verification.json").is_file())
            self.assertTrue((root / "export" / "observed_usage.json").is_file(),
                            "Human-readable export omits run-level observed usage")
            usage = json.loads((root / "export" / "observed_usage.json").read_text())
            self.assertEqual(usage["requests"], 0)
            self.assertEqual(usage["semantics"], "reported_tokens_only_not_provider_bill")

    def test_prepare_is_offline_secret_free_and_resume_refuses_config_drift(self):
        module = self.module()
        fake_item = SimpleNamespace(instance_id="a_1", database_id="a")
        prepared = ([("lite", fake_item)], [{"task_key": "lite/a_1"}],
                    {"precompute_inputs_sha256": "frozen"})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = root / "runtime.env"
            environment.write_text("\n".join((
                "DASH_BASE_URL=https://chat.example/v1", "DASH_API_KEY=chat-secret",
                "DASH_MODELS=chat-model", "EMBEDDING_BASE_URL=https://embed.example/v1",
                "EMBEDDING_API_KEY=embedding-secret", "EMBEDDING_MODEL=embedding-model",
                "PG_HOST=db.example", "PG_PORT=5432", "PG_USER=reader",
                "PG_PASSWORD=database-secret",
            )))
            (root / "train.json").write_text("[]")
            run_dir = root / "new" / "parent" / "run"
            common = ["--run-dir", str(run_dir), "--precompute-dir", str(root / "precomputed"),
                      "--few-shot-source", str(root / "train.json"), "--env-file", str(environment),
                      "--adaptive-concurrency"]
            with patch.object(module, "prepare_inputs", return_value=prepared):
                self.assertEqual(module.main(["prepare", *common]), 0)

            from scripts.baseline_adapters.deepeye.run_store import RunStore
            with RunStore.open(run_dir, read_only=True) as store:
                encoded = json.dumps(store.manifest, sort_keys=True)
                self.assertEqual(store.manifest["effective_config"]["profile"], "bounded-config")
                self.assertEqual(store.summary()["attempts"], 0)
            for secret in ("chat-secret", "embedding-secret", "database-secret"):
                self.assertNotIn(secret, encoded)

            before = (run_dir / "run.sqlite3").read_bytes()
            with patch.object(module, "prepare_inputs", return_value=prepared):
                self.assertEqual(module.main(["resume", *common, "--direct-linking-budget", "2"]), 1)
            self.assertEqual((run_dir / "run.sqlite3").read_bytes(), before)
            with patch.object(module, "prepare_inputs", return_value=prepared):
                self.assertEqual(module.main(["resume", *common, "--concurrency-max", "60"]), 1)
            self.assertEqual((run_dir / "run.sqlite3").read_bytes(), before)

            environment.write_text(environment.read_text().replace("PG_USER=reader", "PG_USER=other-role"))
            with patch.object(module, "prepare_inputs", return_value=prepared):
                self.assertEqual(module.main(["resume", *common]), 1)
            self.assertEqual((run_dir / "run.sqlite3").read_bytes(), before)

    def test_run_returns_nonzero_when_any_item_failed_but_keeps_report(self):
        module = self.module()
        with patch.object(module, "_prepare_command", return_value={
            "succeeded": 4, "failed": 1, "items": {"full/q": {"status": "failed"}},
        }):
            result = module.main([
                "run", "--run-dir", "unused/run", "--precompute-dir", "unused/precompute",
            ])
        self.assertEqual(result, 1)

    def test_prepare_imports_from_read_only_source_before_any_execution(self):
        from scripts.baseline_adapters.deepeye import run_inheritance
        from scripts.baseline_adapters.deepeye.run_store import RunStore

        module = self.module()
        environment = {
            "DASH_BASE_URL": "https://chat.example/v1", "DASH_MODELS": "chat-model",
            "PG_HOST": "db.example", "PG_PORT": "5432", "PG_USER": "reader",
        }
        tasks = [("lite", SimpleNamespace(instance_id="q", database_id="db"))]
        prepared = (tasks, [{"task_key": "lite/q"}], {"fixture": True})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with RunStore.create(root / "source", {"fixture": "source"}):
                pass
            args = module._build_parser().parse_args([
                "prepare", "--run-dir", str(root / "destination"),
                "--precompute-dir", str(root / "frozen"),
                "--inherit-from", str(root / "source"),
            ])
            def import_boundary(source, destination, incoming):
                self.assertEqual(source.manifest, {"fixture": "source"})
                self.assertTrue(source._read_only)
                self.assertEqual(incoming, tasks)
                attempt = destination.begin_attempt("lite/q", "schema_linking", "fixture")
                destination.finish_attempt(attempt, "succeeded", {"fixture": "imported"})
                return {"imported": 1}

            with patch("scripts.deepeye_bird_interact_smoke.read_environment", return_value=environment), \
                    patch.object(module, "_resolve_few_shot_source", return_value=root / "train"), \
                    patch.object(module, "prepare_inputs", return_value=prepared), \
                    patch.object(run_inheritance, "inherit_checkpoints", side_effect=import_boundary) as inherited, \
                    patch.object(module, "_execute_pipeline") as execute:
                result = module._prepare_command(args, execute=False, resume=False)
            inherited.assert_called_once()
            execute.assert_not_called()
            self.assertEqual(result["inheritance"], {"imported": 1})
            self.assertEqual(result["store_summary"]["succeeded"], 1)
            self.assertTrue(result["verification"]["ok"])
            args.run_dir = root / "executed-destination"
            def execute_boundary(store, incoming, actual_environment, actual_args):
                self.assertEqual(store.summary()["succeeded"], 1,
                                 "Paid execution started before checkpoint import")
                self.assertEqual(store.completed("lite/q", "schema_linking", "fixture")["payload"],
                                 {"fixture": "imported"})
                return {"succeeded": 1, "failed": 0}
            with patch("scripts.deepeye_bird_interact_smoke.read_environment", return_value=environment), \
                    patch.object(module, "_resolve_few_shot_source", return_value=root / "train"), \
                    patch.object(module, "prepare_inputs", return_value=prepared), \
                    patch.object(run_inheritance, "inherit_checkpoints", side_effect=import_boundary), \
                    patch.object(module, "_execute_pipeline", side_effect=execute_boundary) as execute:
                result = module._prepare_command(args, execute=True, resume=False)
            execute.assert_called_once()
            self.assertEqual(result["inheritance"], {"imported": 1})
            with RunStore.open(root / "source", read_only=True) as source:
                self.assertEqual(source.summary()["attempts"], 0)

    def test_execute_installs_trace_and_postgres_only_for_the_pipeline_call(self):
        module = self.module()
        self.assertIn("pipeline_fn", inspect.signature(module._execute_pipeline).parameters,
                      "Pipeline wiring does not expose its external-call boundary")
        calls = []

        class Recorder:
            def __init__(self, store, secrets):
                self.api_call = None
                calls.append(("recorder", store, tuple(secrets)))

            def record_admission(self, kind, payload):
                calls.append((kind, payload))

            @contextmanager
            def install(self):
                calls.append(("trace-installed",))
                yield self
                calls.append(("trace-restored",))

        def install_support():
            calls.append(("postgres-installed",))
            return lambda: calls.append(("postgres-restored",))

        def pipeline(store, tasks, runner_factory, recorder, workers, slot_controller):
            self.assertEqual(os.environ["PG_HOST"], "new-host")
            self.assertEqual(os.environ["PG_PASSWORD"], "new-password")
            self.assertEqual(os.environ["PG_SSLMODE"], "verify-full")
            self.assertEqual(workers, 3)
            self.assertEqual(slot_controller.snapshot()["current_limit"], 3)
            self.assertIs(recorder.api_call, slot_controller)
            self.assertTrue(callable(runner_factory))
            calls.append(("pipeline", tuple(tasks)))
            return {"succeeded": 1, "failed": 0, "items": {}}

        environment = {
            "PG_HOST": "new-host", "PG_PORT": "5433", "PG_USER": "reader",
            "PG_PASSWORD": "new-password", "DASH_API_KEY": "chat-secret",
            "EMBEDDING_API_KEY": "embed-secret",
        }
        old_host = os.environ.get("PG_HOST")
        old_sslmode = os.environ.get("PG_SSLMODE")
        os.environ["PG_SSLMODE"] = "disable"
        try:
            result = module._execute_pipeline(
                "store", [("lite", "item")], environment,
                self.args(workers=3, pg_sslmode="verify-full"),
                pipeline_fn=pipeline, runner_factory_fn=lambda config: (lambda *_: config),
                recorder_type=Recorder, install_support=install_support,
                runtime_config_builder=lambda *_: "runtime-config",
            )
        finally:
            self.assertEqual(os.environ.get("PG_HOST"), old_host)
            self.assertEqual(os.environ.get("PG_SSLMODE"), "disable")
            if old_sslmode is None:
                os.environ.pop("PG_SSLMODE", None)
            else:
                os.environ["PG_SSLMODE"] = old_sslmode

        self.assertEqual(result["succeeded"], 1)
        self.assertEqual([call[0] for call in calls],
                         ["recorder", "postgres-installed", "trace-installed", "pipeline",
                          "trace-restored", "postgres-restored"])


if __name__ == "__main__":
    unittest.main()
