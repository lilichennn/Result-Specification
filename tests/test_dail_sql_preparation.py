import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import numpy as np

from scripts.baseline_adapters.dail_sql.tokenizer import CoreNLPTokenizer, LocalCoreNLP
from scripts.baseline_adapters.dail_sql.native import mask_question, link_question, load_public_schema


ROOT = Path(__file__).resolve().parents[1]
STOPWORDS = ROOT / "baselines_reproduce/dail_sql/assets/nltk_data"


class WordTokenizer:
    def tokenize(self, text):
        return text.lower().split()

    def tokenize_for_copying(self, text):
        return self.tokenize(text), self.tokenize(text)


class PreparationTests(unittest.TestCase):
    def test_mask_schema_and_spider_cell_with_cv_disabled_no_connection(self):
        schema = {"table_names_original": ["employee"], "table_names": ["employee"],
                  "column_names_original": [[-1, "*"], [0, "name"]],
                  "column_names": [[-1, "*"], [0, "name"]], "column_types": ["text", "text"]}
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE employee(name TEXT)")
            db.execute("INSERT INTO employee VALUES ('sales')")
            enabled = link_question("list employee name in sales", schema, WordTokenizer(),
                                    compute_cv_link=True, connection=db, stopwords_path=STOPWORDS)
            self.assertEqual(mask_question(enabled), "list <mask> <mask> in <unk>")
        disabled = link_question("list employee name in sales", schema, WordTokenizer(),
                                 compute_cv_link=False, stopwords_path=STOPWORDS)
        self.assertEqual(mask_question(disabled), "list <mask> <mask> in sales")

    def test_actual_public_meta_layouts(self):
        spider = load_public_schema(ROOT / "scripts/spider_dev/preprocessed_data/meta/concert_singer")
        self.assertIn("Singer_ID", [col[1] for col in spider["column_names_original"]])
        pg = load_public_schema(ROOT / "scripts/bird_interact_full/preprocessed_data/meta/organ_transplant")
        self.assertIn("contrib_med_registry", [col[1] for col in pg["column_names_original"]])
        self.assertNotIn("sample_value", str(spider))
        self.assertTrue(spider["foreign_keys"])
        self.assertIn("INT", spider["column_types_original"])
        self.assertTrue(any(pg["column_descriptions"]))

    def test_client_rejects_non_loopback_endpoint(self):
        with self.assertRaises(ValueError):
            CoreNLPTokenizer("https://example.com")

    def test_windows_encoded_public_meta_and_explicit_keys(self):
        actual = load_public_schema(ROOT / "scripts/bird_dev/preprocessed_data/meta/formula_1")
        self.assertIn("qualifying", actual["table_names_original"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "people.csv").write_text('original_column_name,data_format,primary_key,foreign_keys\nid,integer,true,\nparent_id,integer,,"[[""people"", ""id""]]"\n')
            schema = load_public_schema(path)
            self.assertEqual(schema["primary_keys"], [1])
            self.assertEqual(schema["foreign_keys"], [[2, 1]])

    def test_public_meta_multiple_foreign_key_references(self):
        schema = load_public_schema(ROOT / "scripts/spider_dev/preprocessed_data/meta/dog_kennels")
        owners = schema["table_names_original"].index("Owners")
        owner_id = schema["column_names_original"].index([owners, "owner_id"])
        self.assertTrue(any(target == owner_id for _, target in schema["foreign_keys"]))
        self.assertEqual(len(schema["foreign_keys"]), len(set(map(tuple, schema["foreign_keys"]))))

    def test_dangling_public_foreign_key_is_retained_without_inference(self):
        schema = load_public_schema(ROOT / "scripts/spider_test/preprocessed_data/meta/book_1")
        invalid_id = schema["column_ref_keys"].index("Author.idAuthorA")
        self.assertIn({"column_id": invalid_id, "reference": "Author.idAuthorA"}, schema["unresolved_foreign_keys"])
        self.assertFalse(any(source == invalid_id for source, _ in schema["foreign_keys"]))
        self.assertTrue(schema["foreign_keys"])  # Valid Book reference remains.
        self.assertNotIn("idAuthorA", [name for _, name in schema["column_names_original"]])
        self.assertEqual(schema["column_types"][invalid_id], "number")


class CacheTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        meta = root / "meta"
        meta.mkdir()
        (meta / "employee.csv").write_text("column_name,column_type,sample_value,ref_key\nname,TEXT,,\n")
        schema = {"db_id": "work", "table_names_original": ["employee"], "table_names": ["employee"],
                  "column_names_original": [[-1, "*"], [0, "name"]],
                  "column_names": [[-1, "*"], [0, "name"]], "column_types": ["text", "text"],
                  "primary_keys": [], "foreign_keys": []}
        source = root / "tables.json"
        source.write_text(json.dumps([schema]))
        examples = [{"example_id": str(i), "database_id": "work", "question": f"list name {i}",
                     "sql": "SELECT name FROM employee", "evidence": ""} for i in range(3)]
        groups = {}
        for group in ("g1", "g2"):
            rows = [{"group": group, "question_id": str(i), "original_id": i, "question": f"list name {i}",
                     "evidence": "", "schema_ref": str(meta), "database": {"dialect": "sqlite", "database_id": "work", "path": None},
                     "rc3_ref": {}, "training_pool": "bird", "evaluation_binding": {"reference_sql": "SECRET_GOLD"}}
                    for i in range(2)]
            groups[group] = {"rows": rows, "ids": ["0", "1"], "count": 2,
                             "training_pool": "bird", "compute_cv_link": False, "sources": {}}
        manifest = {"format": "dail-sql-inputs-manifest-v1", "settings": {"k_shot": 2},
                    "group_order": ["g1", "g2"], "groups": groups,
                    "training_pools": {"bird": {"examples": examples, "sources": [], "schema_sources": [str(source)], "database_roots": []}}}
        self.encoded = []
        def encode(texts):
            self.encoded.extend(texts)
            return np.array([[len(text), int(text.rsplit(" ", 1)[-1])] for text in texts], dtype=np.float32)
        resources = {"tokenizer": WordTokenizer(), "encoder": encode,
                     "prompt_builder": lambda task, examples: json.dumps({"task": task, "examples": examples}),
                     "prompt_builder_version": "fixture-v1", "block_size": 1, "nltk_data": str(STOPWORDS),
                     "corenlp": {"version": "fixture"}, "mpnet": {"revision": "fixture"}}
        return manifest, resources

    def test_resume_reuses_pool_blocks_and_gold_never_enters_prepared_tasks(self):
        from scripts.baseline_adapters.dail_sql.preparation import prepare, status_preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            output = Path(directory) / "prepared"
            result = prepare(manifest, output, resources)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(len(self.encoded), 7)  # 3 train once + 4 targets
            self.assertEqual(prepare(manifest, output, resources), result)
            self.assertEqual(len(self.encoded), 7)
            self.assertEqual(status_preparation(Path(result["directory"]))["status"], "ready")
            for path in Path(result["directory"]).rglob("*.json"):
                if "evaluation" not in path.parts:
                    self.assertNotIn("SECRET_GOLD", path.read_text())
            self.assertIn("SECRET_GOLD", (Path(result["directory"]) / "evaluation/bindings.json").read_text())
            task = json.loads((Path(result["directory"]) / "groups/g1/blocks/000000/tasks.json").read_text())[0]
            order = np.load(Path(result["directory"]) / task["distance_order_ref"])
            self.assertEqual(order.dtype, np.int32)
            self.assertEqual(task["first_example_ids"], ["0", "1"])

    def test_interrupted_group_resumes_missing_block_and_completed_group_is_ready(self):
        from scripts.baseline_adapters.dail_sql.preparation import prepare, status_preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            output = Path(directory) / "prepared"
            original = resources["prompt_builder"]
            def fail(task, examples):
                if task["group"] == "g2":
                    raise RuntimeError("fixture interruption")
                return original(task, examples)
            resources["prompt_builder"] = fail
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                prepare(manifest, output, resources)
            directory = next(output.glob("preparation-*"))
            state = status_preparation(directory)
            self.assertEqual(state["groups"]["g1"]["status"], "ready")
            self.assertEqual(state["groups"]["g2"]["status"], "pending")
            resources["prompt_builder"] = original
            self.assertEqual(prepare(manifest, output, resources)["status"], "ready")
            # Failed prompt reuses already durable mask/vector block.
            self.assertEqual(len(self.encoded), 7)

    def test_missing_prompt_builder_fails_before_preparation_or_model_load(self):
        from scripts.baseline_adapters.dail_sql.preparation import prepare
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            del resources["prompt_builder"]
            with patch.dict("sys.modules", {"scripts.baseline_adapters.dail_sql.prompts": None}):
                with self.assertRaisesRegex(RuntimeError, "Task5 prompt builder"):
                    prepare(manifest, Path(directory) / "prepared", resources)
            self.assertEqual(self.encoded, [])

    def test_changed_meta_and_revision_create_distinct_cache_keys(self):
        from scripts.baseline_adapters.dail_sql.preparation import prepare
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            output = Path(directory) / "prepared"
            first = prepare(manifest, output, resources)
            (Path(directory) / "meta/employee.csv").write_text("column_name,column_type\nname,TEXT\nid,INT\n")
            second = prepare(manifest, output, resources)
            resources["mpnet"]["revision"] = "new-fixture"
            third = prepare(manifest, output, resources)
            self.assertEqual(len({r["directory"] for r in (first, second, third)}), 3)

    def test_cli_preparation_status_uses_completed_artifacts(self):
        from scripts.baseline_adapters.dail_sql.preparation import prepare
        from scripts.rc_evaluation.dail_sql.cli import main
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            result = prepare(manifest, Path(directory) / "prepared", resources)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["status", "--preparation", result["directory"]]), 0)
            self.assertEqual(json.loads(output.getvalue())["groups"]["g1"]["status"], "ready")

    def test_bulk_reader_decodes_blocks_once_and_schema_storage_is_shared(self):
        from scripts.baseline_adapters.dail_sql import preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            resources["block_size"] = 2
            group = manifest["groups"]["g1"]
            group["rows"] = [{**group["rows"][0], "question_id": str(i), "original_id": i} for i in range(20)]
            group["ids"] = [str(i) for i in range(20)]
            group["count"] = 20
            result = preparation.prepare(manifest, Path(directory) / "prepared", resources)
            root = Path(result["directory"])
            with patch.object(preparation, "_json", wraps=preparation._json) as read:
                tasks = preparation.load_prepared_group(root, "g1")
                for _ in range(3):
                    for i in range(20):
                        self.assertEqual(tasks[str(i)]["task"]["question_id"], str(i))
                        self.assertIn("schema", tasks[str(i)])
                block_reads = [call for call in read.call_args_list if Path(call.args[0]).name == "tasks.json"]
                self.assertEqual(len(block_reads), 10)
            raw = json.loads((root / "pools/bird/examples.json").read_text())
            self.assertNotIn("schema", raw[0])
            self.assertEqual(raw[0]["schema_ref"], raw[1]["schema_ref"])
            self.assertIn("schema", preparation.load_training_pool(root, "bird")[0])

    def test_prepare_selected_groups_reuses_the_same_canonical_cache(self):
        from scripts.baseline_adapters.dail_sql.preparation import prepare
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            resources["groups"] = ["g1"]
            first = prepare(manifest, Path(directory) / "prepared", resources)
            self.assertEqual(first["groups"]["g1"]["status"], "ready")
            self.assertEqual(first["groups"]["g2"]["status"], "pending")
            self.assertEqual(len(self.encoded), 5)
            resources["groups"] = ["g2"]
            second = prepare(manifest, Path(directory) / "prepared", resources)
            self.assertEqual(second["directory"], first["directory"])
            self.assertEqual(second["status"], "ready")
            self.assertEqual(len(self.encoded), 7)

    def test_cli_config_freezes_all_inputs_and_passes_selected_groups(self):
        from scripts.rc_evaluation.dail_sql.cli import main
        with tempfile.TemporaryDirectory() as directory:
            observed = {}
            def bounded_prepare(manifest_path, output, resources):
                observed.update(json.loads(Path(manifest_path).read_text()))
                observed["selected"] = resources["groups"]
                return {"status": "pending", "groups": {"spider_dev": {"status": "ready"}}}
            with patch("scripts.baseline_adapters.dail_sql.preparation.prepare", side_effect=bounded_prepare), redirect_stdout(io.StringIO()):
                result = main(["prepare", "--config", str(ROOT / "config/dail_sql/experiment.json"),
                               "--resources", str(ROOT / "baselines_reproduce/dail_sql/assets/resources_20260916.json"),
                               "--output", directory, "--groups", "spider_dev"])
            self.assertEqual(result, 0)
            self.assertEqual(observed["selected"], ["spider_dev"])
            self.assertEqual(len(observed["groups"]), 5)
            self.assertEqual(sum(group["count"] for group in observed["groups"].values()), 5320)

    def cv_fixture(self, directory):
        manifest, resources = self.fixture(directory)
        root = Path(directory)
        db_root = root / "databases"
        (db_root / "work").mkdir(parents=True)
        path = db_root / "work/work.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE employee(name TEXT)")
            connection.execute("INSERT INTO employee VALUES ('sales')")
        manifest["training_pools"]["bird"]["database_roots"] = [str(db_root)]
        for group in manifest["groups"].values():
            group["compute_cv_link"] = True
            for row in group["rows"]:
                row["database"]["path"] = str(path)
                row["question"] = "list employee name in sales"
        for row in manifest["training_pools"]["bird"]["examples"]:
            row["question"] = "list employee name in sales"
        def encode(texts):
            self.encoded.extend(texts)
            return np.array([[len(text), text.count("<unk>")] for text in texts], dtype=np.float32)
        resources["encoder"] = encode
        return manifest, resources, path

    def test_cv_database_cell_change_invalidates_mask_and_hashes_distinct_path_once(self):
        from scripts.baseline_adapters.dail_sql import preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources, database = self.cv_fixture(directory)
            output = Path(directory) / "prepared"
            with patch.object(preparation, "_file_hash", wraps=preparation._file_hash) as digest:
                first = preparation.prepare(manifest, output, resources)
                db_reads = [call for call in digest.call_args_list if Path(call.args[0]).resolve() == database.resolve()]
                self.assertEqual(len(db_reads), 1)  # Shared by 3 train and 4 target rows.
            before = preparation.load_prepared_task(Path(first["directory"]), "g1", "0")
            self.assertEqual(before["mask"], "list <mask> <mask> in <unk>")
            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE employee SET name = 'engineering'")
            second = preparation.prepare(manifest, output, resources)
            self.assertNotEqual(second["directory"], first["directory"])
            after = preparation.load_prepared_task(Path(second["directory"]), "g1", "0")
            self.assertEqual(after["mask"], "list <mask> <mask> in sales")

    def test_cv_off_ignores_database_changes_and_active_wal_is_rejected_for_cv(self):
        from scripts.baseline_adapters.dail_sql import preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources, database = self.cv_fixture(directory)
            for group in manifest["groups"].values():
                group["compute_cv_link"] = False
            output = Path(directory) / "prepared"
            first = preparation.prepare(manifest, output, resources)
            connection = sqlite3.connect(database)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("UPDATE employee SET name = 'engineering'")
                connection.commit()
                self.assertTrue(Path(str(database) + "-wal").is_file())
                with patch.object(preparation, "_file_hash", wraps=preparation._file_hash) as digest:
                    second = preparation.prepare(manifest, output, resources)
                    self.assertEqual(second["directory"], first["directory"])
                    self.assertFalse(any(Path(call.args[0]).resolve() == database.resolve() for call in digest.call_args_list))
                for group in manifest["groups"].values():
                    group["compute_cv_link"] = True
                with self.assertRaisesRegex(ValueError, "WAL"):
                    preparation.prepare(manifest, output, resources)
            finally:
                connection.close()

    def test_group_status_and_readers_require_pool_and_shared_schema_dependencies(self):
        from scripts.baseline_adapters.dail_sql import preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            result = preparation.prepare(manifest, Path(directory) / "prepared", resources)
            root = Path(result["directory"])
            task = preparation.load_prepared_task(root, "g1", "0")
            for path in (root / "pools/bird/vectors.npy", root / task["schema_ref"]):
                with self.subTest(dependency=path.name):
                    original = path.read_bytes()
                    path.write_bytes(b"corrupt fixture dependency")
                    try:
                        self.assertEqual(preparation.status_preparation(root)["groups"]["g1"]["status"], "pending")
                        with self.assertRaises(ValueError):
                            preparation.load_prepared_group(root, "g1")
                        with self.assertRaises(ValueError):
                            preparation.load_prepared_task(root, "g1", "0")
                    finally:
                        path.write_bytes(original)

    def test_unrelated_pending_pool_and_group_do_not_block_ready_group(self):
        from scripts.baseline_adapters.dail_sql import preparation
        with tempfile.TemporaryDirectory() as directory:
            manifest, resources = self.fixture(directory)
            manifest["training_pools"]["other"] = manifest["training_pools"]["bird"].copy()
            manifest["groups"]["g2"]["training_pool"] = "other"
            for row in manifest["groups"]["g2"]["rows"]:
                row["training_pool"] = "other"
            resources["groups"] = ["g1"]
            result = preparation.prepare(manifest, Path(directory) / "prepared", resources)
            self.assertEqual(result["training_pools"]["other"]["status"], "pending")
            self.assertEqual(result["groups"]["g2"]["status"], "pending")
            self.assertEqual(result["groups"]["g1"]["status"], "ready")
            root = Path(result["directory"])
            self.assertEqual(len(preparation.load_prepared_group(root, "g1")), 2)
            self.assertEqual(preparation.load_prepared_task(root, "g1", "0")["task"]["question_id"], "0")


class LocalTokenizerTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "baselines_reproduce/dail_sql/assets/resources_20260916.json").exists(), "local Java assets absent")
    def test_actual_pinned_java_transport_matches_native_probe(self):
        resources = json.loads((ROOT / "baselines_reproduce/dail_sql/assets/resources_20260916.json").read_text())
        probe = json.loads((ROOT / "baselines_reproduce/dail_sql/assets/corenlp_native_probe_20260916.json").read_text())
        with tempfile.TemporaryDirectory(dir=ROOT / "baselines_reproduce/dail_sql/assets") as directory:
            with LocalCoreNLP(resources, Path(directory), root=ROOT) as client:
                self.assertEqual(client.tokenize_for_copying(probe["text"]),
                                 (probe["lowercase_lemmas"], probe["lowercase_original_text"]))
                self.assertEqual(client.tokenize(""), [])
                self.assertTrue(client.url.startswith("http://127.0.0.1:"))


if __name__ == "__main__":
    unittest.main()
