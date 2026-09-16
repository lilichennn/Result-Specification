"""DAIL frozen-input contracts; all fixtures are offline and non-sensitive."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.baseline_adapters.dail_sql.config import DailSettings, TaskKey, load_experiment_config
from scripts.baseline_adapters.dail_sql.inputs import build_manifest, runtime_task, validate_targets, write_manifest


class SettingsTests(unittest.TestCase):
    def test_default_request_is_one_thinking_enabled_single_choice_payload(self):
        messages = [{"role": "user", "content": "q"}]
        payload = DailSettings().request_kwargs("fixture-model", messages)
        self.assertEqual(payload, {"model": "fixture-model", "messages": messages,
                                   "n": 1, "temperature": 0.6,
                                   "extra_body": {"enable_thinking": True}})
        self.assertIs(payload["messages"], messages)

    def test_obsolete_multi_choice_settings_are_rejected_at_request_consumer(self):
        for n in (4, 5):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "frozen"):
                DailSettings(n=n).request_kwargs("fixture-model", [{"role": "user", "content": "q"}])

    def test_request_payload_emits_confirmed_sampling_and_no_token_cap(self):
        kwargs = DailSettings().request_kwargs("fixture-model", [{"role": "user", "content": "q"}])
        self.assertEqual(kwargs, {"model": "fixture-model", "messages": [{"role": "user", "content": "q"}],
                                  "n": 1, "temperature": 0.6,
                                  "extra_body": {"enable_thinking": True}})

    def test_config_consumer_preserves_five_sample_budget_and_per_sample_five_attempts(self):
        workspace = Path(__file__).resolve().parents[1]
        config = load_experiment_config(workspace / "config/dail_sql/experiment.json")
        self.assertEqual(config["settings"], {"k_shot": 9, "n": 1, "samples_per_round": 5,
                         "enable_thinking": True, "temperature": 0.6, "max_attempts": 5,
                         "request_timeout_seconds": 910, "max_tokens": None})
        settings = DailSettings(**config["settings"])
        payload = settings.request_kwargs("fixture-model", [{"role": "user", "content": "q"}])
        self.assertNotIn("samples_per_round", payload)
        self.assertNotIn("max_attempts", payload)
        self.assertNotIn("max_seq_len", payload)

    def test_config_consumer_rejects_changed_frozen_sampling_contract(self):
        workspace = Path(__file__).resolve().parents[1]
        base = json.loads((workspace / "config/dail_sql/experiment.json").read_text())
        for field, value in (("n", 4), ("n", 5), ("samples_per_round", 4),
                             ("enable_thinking", False), ("max_attempts", 4),
                             ("max_attempts", 6)):
            config = json.loads(json.dumps(base))
            config["settings"][field] = value
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "experiment.json"
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "frozen"):
                    load_experiment_config(path)

    def test_positive_token_cap_is_emitted_only_when_explicitly_configured(self):
        kwargs = DailSettings(max_tokens=512).request_kwargs("fixture-model", [{"role": "user", "content": "q"}])
        self.assertEqual(kwargs["max_tokens"], 512)
        for value in (True, 0, -1, 1.5, "512"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DailSettings(max_tokens=value).request_kwargs("fixture-model", [{"role": "user", "content": "q"}])

    def test_same_numeric_question_id_in_two_groups_has_distinct_identity(self):
        self.assertNotEqual(TaskKey("batch", "spider_dev", 1), TaskKey("batch", "bird_dev", "1"))
        self.assertEqual(TaskKey("batch", "spider_dev", 1).question_id, "1")
        for value in (None, "", "  "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TaskKey("batch", "spider_dev", value)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "meta" / "db").mkdir(parents=True)
        (self.root / "meta" / "db" / "public.csv").write_text(
            "column_name,column_type,sample_value,ref_key\nid,INT,[1],\n", encoding="utf-8")
        (self.root / "database").mkdir()
        (self.root / "database" / "db.sqlite").touch()
        self._json("questions.json", [{"index": 1, "db_id": "db", "question": "How many?", "evidence": ""}])
        self._json("rc.json", [{"index": 1, "db_id": "db", "question": "How many?", "evidence": "",
                                 "round3_status": "succeeded", "round3_error": None,
                                 "rc_round3": {"population": "All", "row_grain": "one",
                                               "column_role": "count", "derivation": "count",
                                               "filter_policy": "none", "meta_review": "none"}}])
        self._json("gold.json", [{"db_id": "db", "question": "How many?", "query": "SELECT count(*) FROM t"}])
        self._json("train.json", [{"db_id": "db", "question": "Training?", "query": "SELECT 1"}])
        self._json("train_tables.json", [{"db_id": "db", "table_names": ["t"]}])
        self._json("snapshot.json", {"version": 1, "format": "structured_dataset_snapshot", "num_items": 1,
                                     "snapshot_root": "snapshot.data"})
        (self.root / "snapshot.data").mkdir()
        (self.root / "snapshot.data" / "items.jsonl").write_text(json.dumps({"input": {
            "question_id": 1, "question": "How many?", "evidence": "", "database_id": "db",
            "database_path": str(self.root / "database" / "db.sqlite"),
            "database_schema": {"db_type": "sqlite", "db_id": "db"}}}) + "\n", encoding="utf-8")
        self.config = {"format": "dail-sql-inputs-v1", "settings": {"k_shot": 9, "n": 1,
                       "samples_per_round": 5, "enable_thinking": True,
                       "temperature": 0.6, "max_attempts": 5, "request_timeout_seconds": 910,
                       "max_tokens": None}, "training_pools": {"spider": {
                           "sources": ["train.json"], "schema_sources": ["train_tables.json"],
                           "database_roots": ["database"]}},
                       "groups": [{"name": "spider_dev", "questions": "questions.json", "rc": "rc.json",
                                   "snapshot": "snapshot.json", "meta": "meta", "evaluation": "gold.json",
                                   "training_pool": "spider", "compute_cv_link": True,
                                   "expected_count": 1}]}

    def _json(self, path, value):
        (self.root / path).write_text(json.dumps(value), encoding="utf-8")

    def test_manifest_separates_training_gold_from_target_gold_and_binds_public_meta(self):
        manifest = build_manifest(self.root, self.config)
        row = manifest["groups"]["spider_dev"]["rows"][0]
        task = runtime_task(row)
        self.assertEqual(task["question_id"], "1")
        self.assertEqual(task["database"], {"dialect": "sqlite", "database_id": "db",
                                            "path": str((self.root / "database" / "db.sqlite").resolve())})
        self.assertEqual(task["schema_ref"], str((self.root / "meta" / "db").resolve()))
        self.assertEqual(row["evaluation_binding"]["reference_sql"], "SELECT count(*) FROM t")
        self.assertNotIn("evaluation_binding", task)
        self.assertNotIn("SELECT count(*) FROM t", json.dumps(task))
        self.assertEqual(manifest["training_pools"]["spider"]["examples"][0]["sql"], "SELECT 1")
        self.assertEqual(manifest["training_pools"]["spider"]["examples"][0]["example_id"],
                         "spider/train.json/0")
        self.assertEqual(manifest["training_pools"]["spider"]["schema_sources"],
                         [str((self.root / "train_tables.json").resolve())])
        self.assertEqual(manifest["training_pools"]["spider"]["database_roots"],
                         [str((self.root / "database").resolve())])

    def test_missing_explicit_training_schema_or_database_root_is_rejected(self):
        for field, nonexistent in (("schema_sources", "missing_tables.json"),
                                   ("database_roots", "missing_database")):
            with self.subTest(field=field):
                config = json.loads(json.dumps(self.config))
                config["training_pools"]["spider"][field] = [nonexistent]
                with self.assertRaisesRegex(ValueError, field):
                    build_manifest(self.root, config)

    def test_target_validation_rejects_empty_unknown_and_conflicting_identity(self):
        manifest = build_manifest(self.root, self.config)
        self.assertEqual(validate_targets(manifest, "spider_dev", ["1"]), ["1"])
        for ids in ([], ["2"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_targets(manifest, "spider_dev", ids)
        with self.assertRaises(ValueError):
            validate_targets(manifest, "bird_dev", ["1"])
        self._json("questions.json", [{"index": 1, "db_id": "db", "question": "How many?", "evidence": ""},
                                      {"index": "1", "db_id": "db", "question": "How many?", "evidence": ""}])
        with self.assertRaises(ValueError):
            build_manifest(self.root, self.config)

    def test_selected_row_group_cannot_disagree_with_requested_group(self):
        manifest = build_manifest(self.root, self.config)
        manifest["groups"]["spider_dev"]["rows"][0]["group"] = "bird_dev"
        with self.assertRaises(ValueError):
            validate_targets(manifest, "spider_dev", ["1"])

    def test_selected_row_id_must_match_original_and_rc3_reference(self):
        for field, value in (("original_id", "2"), ("rc3_ref", "2")):
            with self.subTest(field=field):
                manifest = build_manifest(self.root, self.config)
                row = manifest["groups"]["spider_dev"]["rows"][0]
                if field == "rc3_ref":
                    row["rc3_ref"]["question_id"] = value
                else:
                    row[field] = value
                with self.assertRaises(ValueError):
                    validate_targets(manifest, "spider_dev", ["1"])
        manifest = build_manifest(self.root, self.config)
        manifest["groups"]["spider_dev"]["sources"].pop("rc")
        with self.assertRaises(ValueError):
            validate_targets(manifest, "spider_dev", ["1"])

    def test_selected_database_must_match_rc3_and_evaluation_binding(self):
        for mutate in (lambda row: row["database"].update(database_id="other"),
                       lambda row: row["database"].update(path="/wrong/db.sqlite"),
                       lambda row: row["database"].pop("database_id"),
                       lambda row: row["rc3_ref"].update(database_id="other")):
            with self.subTest(mutate=mutate):
                manifest = build_manifest(self.root, self.config)
                mutate(manifest["groups"]["spider_dev"]["rows"][0])
                with self.assertRaises(ValueError):
                    validate_targets(manifest, "spider_dev", ["1"])

    def test_rc3_identity_mismatch_fails_before_manifest_is_returned(self):
        rc = json.loads((self.root / "rc.json").read_text())
        rc[0]["question"] = "Different question"
        self._json("rc.json", rc)
        with self.assertRaises(ValueError):
            build_manifest(self.root, self.config)

    def test_manifest_write_is_versioned_and_never_overwrites_prior_binding(self):
        manifest = build_manifest(self.root, self.config)
        first = write_manifest(self.root, manifest)
        before = first.read_bytes()
        self.assertEqual(write_manifest(self.root, manifest), first)
        changed = json.loads(json.dumps(manifest))
        changed["groups"]["spider_dev"]["rows"][0]["rc3_ref"]["content"]["population"] = "Changed"
        second = write_manifest(self.root, changed)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)
        self.assertEqual(json.loads(second.read_text())["groups"]["spider_dev"]["rows"][0]
                         ["rc3_ref"]["content"]["population"], "Changed")

    def test_authoritative_config_change_creates_distinct_preparation_version(self):
        original = build_manifest(self.root, self.config)
        first = write_manifest(self.root, original)
        modified_config = json.loads(json.dumps(self.config))
        modified_config["groups"][0]["compute_cv_link"] = False
        modified = build_manifest(self.root, modified_config)
        second = write_manifest(self.root, modified)
        self.assertNotEqual(first, second)
        self.assertTrue(json.loads(first.read_text())["groups"]["spider_dev"]["compute_cv_link"])
        self.assertFalse(json.loads(second.read_text())["groups"]["spider_dev"]["compute_cv_link"])

    def test_publish_interruption_never_exposes_partial_canonical_manifest(self):
        manifest = build_manifest(self.root, self.config)
        with patch("os.link", side_effect=OSError("publish interrupted")):
            with self.assertRaisesRegex(OSError, "publish interrupted"):
                write_manifest(self.root, manifest)
        prepared = self.root / "baselines_reproduce" / "dail_sql" / "prepared"
        self.assertEqual(list(prepared.glob("*/inputs_manifest.json")), [])
        self.assertEqual(list(prepared.glob("*/.inputs-manifest-*")), [])
        published = write_manifest(self.root, manifest)
        self.assertEqual(json.loads(published.read_text())["format"], "dail-sql-inputs-manifest-v1")

    def test_existing_conflicting_canonical_bytes_are_never_altered(self):
        manifest = build_manifest(self.root, self.config)
        published = write_manifest(self.root, manifest)
        published.write_text("conflicting bytes", encoding="utf-8")
        with self.assertRaises(ValueError):
            write_manifest(self.root, manifest)
        self.assertEqual(published.read_text(encoding="utf-8"), "conflicting bytes")

    def test_source_fingerprints_track_public_meta_and_snapshot_item_content(self):
        first = build_manifest(self.root, self.config)["groups"]["spider_dev"]["sources"]
        public_csv = self.root / "meta" / "db" / "public.csv"
        public_csv.write_text(public_csv.read_text() + "name,TEXT,[],\n", encoding="utf-8")
        second = build_manifest(self.root, self.config)["groups"]["spider_dev"]["sources"]
        self.assertNotEqual(first["meta"]["sha256"], second["meta"]["sha256"])
        lines = (self.root / "snapshot.data" / "items.jsonl")
        record = json.loads(lines.read_text())
        record["input"]["database_schema"]["extra_physical_column"] = "hidden"
        lines.write_text(json.dumps(record) + "\n", encoding="utf-8")
        third = build_manifest(self.root, self.config)["groups"]["spider_dev"]["sources"]
        self.assertNotEqual(second["snapshot_items"]["sha256"], third["snapshot_items"]["sha256"])

    def test_actual_five_group_manifest_matches_deepeye_counts_and_ids(self):
        workspace = Path(__file__).resolve().parents[1]
        config = load_experiment_config(workspace / "config" / "dail_sql" / "experiment.json")
        manifest = build_manifest(workspace, config)
        self.assertEqual(manifest["group_order"], ["spider_dev", "bird_dev", "bird_interact_full",
                                                 "bird_interact_lite", "spider_test"])
        self.assertEqual([manifest["groups"][g]["count"] for g in manifest["group_order"]],
                         [1034, 1534, 410, 195, 2147])
        self.assertEqual(sum(manifest["groups"][g]["count"] for g in manifest["group_order"]), 5320)
        self.assertEqual([manifest["groups"][g]["compute_cv_link"] for g in manifest["group_order"]],
                         [True, False, False, False, True])
        self.assertEqual([manifest["groups"][g]["training_pool"] for g in manifest["group_order"]],
                         ["spider", "bird", "bird", "bird", "spider"])
        for g in manifest["group_order"]:
            self.assertEqual(set(manifest["groups"][g]["ids"]),
                             {row["question_id"] for row in manifest["groups"][g]["rows"]})
        self.assertEqual(len(manifest["training_pools"]["spider"]["examples"]), 8659)
        self.assertEqual(len(manifest["training_pools"]["bird"]["examples"]), 9428)


if __name__ == "__main__":
    unittest.main()
