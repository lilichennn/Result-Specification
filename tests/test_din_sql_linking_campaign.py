import asyncio
from collections import Counter
from dataclasses import asdict
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.baseline_adapters.din_sql.inputs import DinSettings, TaskKey, PreparedInputs, load_templates
from scripts.baseline_adapters.din_sql.records import DinRecords, write_json
from din_sql_fixtures import make_task


ROOT = Path(__file__).resolve().parents[1]


def make_source_batch(root: Path, settings: DinSettings | None = None):
    """Create one sealed source question for campaign-boundary tests."""

    code_root = root / "code"
    source = root / "source"
    meta = code_root / "scripts/bird_dev/preprocessed_data/meta/scores"
    meta.mkdir(parents=True)
    (meta / "scores.csv").write_text(
        "original_column_name,data_type,column_description,value_description,ref_key\n"
        "value,integer,score,,\n"
    )
    database = root / "scores.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("create table scores(value integer)")
    connection.close()
    task = make_task()
    object.__setattr__(task, "database", {
        "dialect": "sqlite", "database_id": "scores", "path": str(database)
    })
    templates = load_templates(ROOT)
    manifest = {
        "format": "din-sql-v1",
        "batch_id": "source",
        "settings": asdict(settings or DinSettings()),
        "groups": {"bird_dev": {"ids": ["0"]}},
    }
    source.mkdir()
    write_json(source / "manifest.json", manifest)
    write_json(source / "prepared/inputs.json", {
        "tasks": [asdict(task)],
        "schemas": {
            task.schema_ref: {
                "context": "Table scores, columns = [*,value]",
                "spider": None,
                "primary": "Primary_keys = []",
            }
        },
        "evaluation": {
            "bird_dev/0": {
                "gold_sql": "SELECT value FROM scores", "database": task.database,
            }
        },
        "templates": templates,
        "identities": {},
    })
    with DinRecords(source, manifest) as records:
        version = records.begin(task.key)
        records.save_node(version, "linking", terminal("linking", result=["scores.value"]))
        for node in ("generation_base", "generation_rc3", "revision_base", "revision_rc3"):
            records.save_node(version, node, terminal(node, result="SELECT value FROM scores"))
        records.seal(version)
    return code_root, source, task, meta


def campaign_api():
    try:
        from scripts.rc_evaluation.din_sql_linking.campaign import (
            implementation_hashes,
            schedule,
        )
        from scripts.rc_evaluation.din_sql_linking.runner import (
            copy_reusable,
            run_question,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("DIN Linking campaign is missing") from exc
    return implementation_hashes, schedule, copy_reusable, run_question


def terminal(node, status="succeeded", result=None, **extra):
    return {
        "node": node,
        "status": status,
        "result": result if result is not None else ({"filtered_metadata": []} if node == "schema_filter_rc3" else []),
        "reason": None,
        "usage": None,
        "origin": "test",
        "input_fingerprint": "fixture",
        "parent_refs": {},
        "response_ref": None,
        "source_refs": {},
        "fallback_used": False,
        **extra,
    }


class MemoryRecords:
    def __init__(self, manifest):
        self.manifest = manifest
        self.root = Path(tempfile.mkdtemp())
        self.versions = {}
        self.latest = {}
        self.pending = {}
        self.counter = 0

    def begin(self, key, parent_version=None):
        self.counter += 1
        version = f"v{self.counter}"
        self.versions[version] = {"key": key, "nodes": {}, "parent_version": parent_version}
        self.pending[key] = version
        return version

    def current(self, key):
        return self.latest.get(key)

    def unfinished(self, key):
        return self.pending.get(key)

    def node(self, version, node):
        return self.versions[version]["nodes"].get(node)

    def save_node(self, version, node, value):
        saved = {**value, "ref": {"attempt_id": version, "event_no": len(self.versions[version]["nodes"]) + 1}}
        self.versions[version]["nodes"][node] = saved
        return saved["ref"]

    def seal(self, version):
        key = self.versions[version]["key"]
        self.latest[key] = version
        self.pending.pop(key, None)

    def view(self, version):
        return self.versions[version]


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_after_filter_resumes_only_linking(self):
        _, _, _, run_question = campaign_api()
        task = make_task()
        records = MemoryRecords({"groups": {"bird_dev": {"ids": ["0"]}}})
        version = records.begin(task.key)
        records.save_node(version, "schema_filter_rc3", terminal("schema_filter_rc3"))
        calls = []

        async def execute(node, *args):
            calls.append(node)
            return terminal(node)

        await run_question(task, version, records, execute)
        self.assertEqual(calls, ["linking_rc3"])
        self.assertEqual(records.current(task.key), version)

    async def test_rerun_copies_successful_filter_but_not_failed_linking(self):
        _, _, copy_reusable, run_question = campaign_api()
        task = make_task()
        records = MemoryRecords({"groups": {"bird_dev": {"ids": ["0"]}}})
        old = records.begin(task.key)
        records.save_node(old, "schema_filter_rc3", terminal("schema_filter_rc3"))
        records.save_node(old, "linking_rc3", terminal("linking_rc3", "failed", result=None))
        records.seal(old)
        new = records.begin(task.key, parent_version=old)
        reused = copy_reusable(task, old, new, records)
        self.assertEqual(reused, {"schema_filter_rc3"})
        calls = []

        async def execute(node, *args):
            calls.append(node)
            return terminal(node)

        await run_question(task, new, records, execute)
        self.assertEqual(calls, ["linking_rc3"])
        self.assertEqual(records.node(new, "schema_filter_rc3")["origin"], "reused")


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    def test_node_specific_validators_explicitly_mark_model_output_value_errors(self):
        from scripts.baseline_adapters.din_sql_linking.transport import OutputValidationError
        from scripts.rc_evaluation.din_sql_linking.runner import (
            _validate_filter_model_output,
            _validate_linking_model_output,
        )

        task = make_task()
        with patch(
            "scripts.rc_evaluation.din_sql_linking.runner.parse_filter_content",
            side_effect=ValueError("unknown selected column"),
        ):
            with self.assertRaises(OutputValidationError):
                _validate_filter_model_output("bad filter", [])
        with patch(
            "scripts.rc_evaluation.din_sql_linking.runner.parse_response",
            side_effect=ValueError("malformed schema links"),
        ):
            with self.assertRaises(OutputValidationError):
                _validate_linking_model_output("bad links", task)

    def test_prepare_rejects_each_noncanonical_execution_profile_value(self):
        from scripts.rc_evaluation.din_sql_linking.campaign import prepare_batch

        invalid_values = {
            "request_limit": 7999,
            "start_rate": 49,
            "sql_workers": 19,
            "request_timeout_seconds": 909,
            "max_attempts": 4,
        }
        for name, invalid in invalid_values.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                values = asdict(DinSettings())
                values[name] = invalid
                code_root, source, _, _ = make_source_batch(
                    Path(tmp), DinSettings(**values)
                )
                with self.assertRaisesRegex(ValueError, "DIN Linking execution profile"):
                    prepare_batch(source, "focused", code_root=code_root)
                self.assertFalse(
                    (code_root / "baselines_reproduce/din_sql_linking/batches/focused").exists()
                )

    def test_paid_boundary_rejects_each_tampered_profile_without_new_version(self):
        from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
        from scripts.rc_evaluation.din_sql_linking.campaign import prepare_batch, run_batch

        invalid_values = {
            "request_limit": 7999,
            "start_rate": 49,
            "sql_workers": 19,
            "request_timeout_seconds": 909,
            "max_attempts": 4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root, source, _, _ = make_source_batch(root)
            batch = Path(prepare_batch(source, "focused", code_root=code_root)["batch"])
            manifest_path = batch / "manifest.json"
            original = json.loads(manifest_path.read_text())
            for name, invalid in invalid_values.items():
                with self.subTest(name=name):
                    changed = json.loads(json.dumps(original))
                    changed["settings"][name] = invalid
                    manifest_path.write_text(json.dumps(changed))
                    with patch(
                        "scripts.baseline_adapters.shared.transport.RequestDispatcher.make_client"
                    ) as paid:
                        with self.assertRaisesRegex(ValueError, "DIN Linking execution profile"):
                            run_batch(batch, root / "missing.env")
                    paid.assert_not_called()
                    manifest_path.write_text(json.dumps(original))
                    with LinkingRecords(batch, original, read_only=True) as records:
                        self.assertEqual(records.rows, {})

    def test_prepare_freezes_source_without_gold_and_status_verify_are_read_only(self):
        from scripts.rc_evaluation.din_sql_linking.campaign import (
            load_prepared, prepare_batch, status, verify_batch,
        )
        with tempfile.TemporaryDirectory() as tmp:
            code_root = Path(tmp) / "code"
            source = Path(tmp) / "source"
            meta = code_root / "scripts/bird_dev/preprocessed_data/meta/scores"
            meta.mkdir(parents=True)
            (meta / "scores.csv").write_text(
                "original_column_name,data_type,column_description,value_description,ref_key\n"
                "value,integer,score,,\n"
            )
            database = Path(tmp) / "scores.sqlite"
            connection = sqlite3.connect(database)
            connection.execute("create table scores(value integer)")
            connection.close()
            task = make_task()
            object.__setattr__(task, "database", {
                "dialect": "sqlite", "database_id": "scores", "path": str(database)
            })
            templates = load_templates(Path(__file__).resolve().parents[1])
            source_manifest = {
                "format": "din-sql-v1", "batch_id": "source",
                "settings": asdict(DinSettings()),
                "groups": {"bird_dev": {"ids": ["0"]}},
            }
            source.mkdir()
            write_json(source / "manifest.json", source_manifest)
            prepared = PreparedInputs(
                {task.key: task},
                {task.schema_ref: {"context": "Table scores, columns = [*,value]", "spider": None,
                                   "primary": "Primary_keys = []"}},
                {"bird_dev/0": {"gold_sql": "SELECT value FROM scores", "database": task.database}},
                templates, {}, {},
            )
            write_json(source / "prepared/inputs.json", {
                "tasks": [asdict(task)], "schemas": prepared.schemas,
                "evaluation": prepared.evaluation, "templates": templates, "identities": {},
            })
            with DinRecords(source, source_manifest) as records:
                version = records.begin(task.key)
                records.save_node(version, "linking", terminal("linking", result=["scores.value"]))
                for node in ("generation_base", "generation_rc3", "revision_base", "revision_rc3"):
                    records.save_node(version, node, terminal(node, result="SELECT value FROM scores"))
                records.seal(version)

            result = prepare_batch(source, "focused", code_root=code_root)
            batch = Path(result["batch"])
            frozen = load_prepared(batch)
            self.assertEqual(len(frozen.tasks), 1)
            self.assertFalse(hasattr(frozen, "evaluation"))
            self.assertFalse(hasattr(next(iter(frozen.tasks.values())), "label"))
            persisted = json.loads((batch / "prepared/inputs.json").read_text())
            self.assertNotIn("label", persisted["tasks"][0])
            self.assertNotIn("gold_sql", json.dumps(list(frozen.native_linking.values())))
            self.assertEqual(status(batch)["groups"]["bird_dev"]["total"], 1)
            self.assertTrue(verify_batch(batch)["ok"])
            (meta / "scores.csv").write_text(
                "original_column_name,data_type,column_description,value_description,ref_key\n"
                "value,integer,changed,,\n"
            )
            changed = verify_batch(batch)
            self.assertFalse(changed["ok"])
            self.assertEqual(changed["changed_inputs"], [str((meta / "scores.csv").resolve())])

    def test_new_implementation_hashes_are_isolated_from_old_din_hash_set(self):
        implementation_hashes, _, _, _ = campaign_api()
        root = Path(__file__).resolve().parents[1]
        new = implementation_hashes(root)
        self.assertTrue(any("din_sql_linking" in name for name in new))
        self.assertIn("result_contract/rc/rc_round1.py", new)
        self.assertIn("result_contract/rc/rc_round2.py", new)
        from scripts.rc_evaluation.din_sql.campaign import implementation_hashes as old_hashes
        self.assertFalse(any("din_sql_linking" in name for name in old_hashes(root)))

    async def test_all_five_groups_start_without_waiting_for_prior_group_terminal(self):
        _, schedule, _, _ = campaign_api()
        groups = (
            "spider_dev",
            "bird_dev",
            "bird_interact_full",
            "bird_interact_lite",
            "spider_test",
        )
        tasks = [make_task(group=group, question_id=str(index))
                 for index, group in enumerate(groups)]
        prepared = type("Prepared", (), {
            "tasks": {task.key: task for task in tasks},
            "groups": groups,
        })()
        records = MemoryRecords({
            "groups": {group: {"ids": [str(index)]}
                       for index, group in enumerate(groups)}
        })
        release = asyncio.Event()
        all_started = asyncio.Event()
        started = set()
        calls = Counter()

        async def execute(node, task, parent, version):
            calls[(task.key, node)] += 1
            if node == "schema_filter_rc3":
                started.add(task.key.group)
                if started == set(groups):
                    all_started.set()
                await release.wait()
            return terminal(node)

        pending = asyncio.create_task(schedule(prepared, records, execute))
        await asyncio.wait_for(all_started.wait(), 3)
        self.assertFalse(pending.done())
        self.assertTrue(all(records.current(task.key) is None for task in tasks))
        release.set()
        result = await pending
        self.assertEqual(result["started_groups"], list(groups))
        self.assertEqual(
            calls,
            Counter({(task.key, node): 1 for task in tasks for node in (
                "schema_filter_rc3", "linking_rc3"
            )}),
        )
        self.assertTrue(all(records.current(task.key) is not None for task in tasks))

    async def test_targeted_smoke_starts_selected_groups_in_parallel_once(self):
        _, schedule, _, _ = campaign_api()
        groups = ("spider_dev", "bird_dev", "bird_interact_full")
        tasks = [make_task(group=group, question_id=str(index))
                 for index, group in enumerate(groups)]
        prepared = type("Prepared", (), {
            "tasks": {task.key: task for task in tasks},
            "groups": groups,
        })()
        records = MemoryRecords({
            "groups": {group: {"ids": [str(index)]}
                       for index, group in enumerate(groups)}
        })
        targets = [task.key for task in tasks]
        release = asyncio.Event()
        all_started = asyncio.Event()
        started = set()
        calls = Counter()

        async def execute(node, task, parent, version):
            calls[(task.key, node)] += 1
            if node == "schema_filter_rc3":
                started.add(task.key)
                if started == set(targets):
                    all_started.set()
                await release.wait()
            return terminal(node)

        pending = asyncio.create_task(
            schedule(prepared, records, execute, targets=targets)
        )
        await asyncio.wait_for(all_started.wait(), 3)
        self.assertFalse(pending.done())
        release.set()
        result = await pending
        self.assertEqual(result["started_groups"], list(groups))
        self.assertTrue(all(count == 1 for count in calls.values()))
        self.assertEqual(len(calls), len(tasks) * 2)

    async def test_scoped_later_group_rerun_skips_empty_earlier_groups(self):
        _, schedule, _, _ = campaign_api()
        earlier = make_task(group="bird_dev", question_id="0")
        target = make_task(group="bird_interact_full", question_id="7")
        prepared = type("Prepared", (), {
            "tasks": {earlier.key: earlier, target.key: target},
            "groups": ("bird_dev", "bird_interact_full"),
        })()
        records = MemoryRecords({
            "groups": {
                "bird_dev": {"ids": ["0"]},
                "bird_interact_full": {"ids": ["7"]},
            }
        })

        async def execute(node, *args):
            return terminal(node)

        result = await schedule(prepared, records, execute, targets=[target.key])
        self.assertEqual(result["started_groups"], ["bird_interact_full"])
        self.assertIsNone(records.current(earlier.key))
        self.assertIsNotNone(records.current(target.key))


if __name__ == "__main__":
    unittest.main()
