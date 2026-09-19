import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.baseline_adapters.din_sql.inputs import DinSettings, PreparedInputs, load_templates
from scripts.baseline_adapters.din_sql.prompts import PromptBuilder
from din_sql_fixtures import make_task


ROOT = Path(__file__).resolve().parents[1]


def core_api():
    try:
        from scripts.baseline_adapters.din_sql_linking.core import (
            build_filter_payload,
            build_filtered_context,
            build_linking_payload,
            load_source_snapshot,
            parse_filter_content,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("DIN Linking adapter package is missing") from exc
    return {
        "build_filter_payload": build_filter_payload,
        "build_filtered_context": build_filtered_context,
        "build_linking_payload": build_linking_payload,
        "load_source_snapshot": load_source_snapshot,
        "parse_filter_content": parse_filter_content,
    }


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = load_templates(ROOT)

    def prepared(self, task, context):
        return PreparedInputs(
            {task.key: task},
            {
                task.schema_ref: {
                    "context": context,
                    "spider": context + "\nForeign_keys = []",
                    "primary": "Primary_keys = []",
                },
                "spider:college_2": {
                    "context": "Table college, columns = [*,id]",
                    "spider": "Table college, columns = [*,id]\nForeign_keys = []",
                    "primary": "Primary_keys = []",
                },
            },
            {},
            self.templates,
            {},
            {},
        )

    def test_full_context_linking_is_byte_identical_and_never_contains_rc(self):
        api = core_api()
        for group in ("bird_dev", "spider_dev", "bird_interact_full"):
            task = make_task(group=group)
            context = "Table scores, columns = [*,value]\nForeign_keys = []"
            prepared = self.prepared(task, context)
            builder = PromptBuilder(prepared, ROOT)
            expected = builder.build("linking", task, {})
            messages, kwargs = api["build_linking_payload"](
                task, context, builder, DinSettings()
            )
            self.assertEqual(messages, expected)
            self.assertEqual(kwargs["messages"], expected)
            self.assertEqual(kwargs["max_tokens"], 5000)
            prompt = json.dumps(messages, ensure_ascii=False)
            for value in task.rc3.values():
                self.assertNotIn(value, prompt)

    def test_filter_uses_question_hint_rc_and_rejects_bad_or_unknown_output(self):
        api = core_api()
        task = make_task()
        metadata = [{"table_name": "scores", "columns": [
            {"original_column_name": "value", "column_description": "score", "ref_key": ""}
        ]}]
        messages, kwargs = api["build_filter_payload"](task, metadata, DinSettings())
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn(task.question, serialized)
        self.assertIn("result_contract", serialized)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertNotIn("max_tokens", kwargs)
        parsed = api["parse_filter_content"](
            '{"tables":[{"name":"scores","columns":["value"]}]}', metadata
        )
        self.assertEqual(parsed["filtered_metadata"], metadata)
        for invalid in (
            "not json",
            '{"tables":[{"name":"missing","columns":[]}]}',
            '{"tables":[{"name":"scores","columns":["missing"]}]}',
        ):
            with self.assertRaises(ValueError):
                api["parse_filter_content"](invalid, metadata)

    def test_sqlite_filtered_context_preserves_selected_samples_and_fk_only(self):
        api = core_api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "db.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE parent(id INTEGER PRIMARY KEY, hidden TEXT);"
                "CREATE TABLE child(id INTEGER, parent_id INTEGER, note TEXT, "
                "FOREIGN KEY(parent_id) REFERENCES parent(id));"
                "INSERT INTO parent VALUES(1,'secret');"
                "INSERT INTO child VALUES(2,1,'omit');"
            )
            connection.close()
            meta = root / "meta"
            meta.mkdir()
            (meta / "parent.csv").write_text(
                "original_column_name,column_description,value_description\nid,parent id,\nhidden,hidden,\n"
            )
            (meta / "child.csv").write_text(
                "original_column_name,column_description,value_description\nid,child id,\nparent_id,parent,\nnote,note,\n"
            )
            task = make_task()
            object.__setattr__(task, "database", {
                "dialect": "sqlite", "database_id": "fixture", "path": str(database)
            })
            filtered = [
                {"table_name": "child", "columns": [
                    {"original_column_name": "parent_id", "column_description": "parent", "ref_key": "child.parent_id; parent.id"}
                ]},
                {"table_name": "parent", "columns": [
                    {"original_column_name": "id", "column_description": "parent id", "ref_key": "child.parent_id; parent.id"}
                ]},
            ]
            context = api["build_filtered_context"](
                task, filtered, meta_dir=meta, code_root=ROOT
            )
            self.assertIn("Table child, columns = [*,parent_id]", context)
            self.assertIn("Table parent, columns = [*,id]", context)
            self.assertIn("child.parent_id = parent.id", context)
            self.assertIn("[1]", context)
            self.assertNotIn("hidden TEXT", context)
            self.assertNotIn("note,note", context)

    def test_postgresql_filtered_context_projects_columns_and_accepts_empty_shapes(self):
        api = core_api()
        task = make_task(group="bird_interact_full")
        object.__setattr__(task, "database", {"dialect": "postgresql", "database_id": "pg"})
        calls = []

        def execute(database, sql, **kwargs):
            calls.append((database, sql, kwargs))
            return {"status": "success", "rows": [[1]], "columns": ["kept"]}

        metadata = [
            {"table_name": "has columns", "columns": [
                {"original_column_name": "kept", "data_type": "integer", "column_description": "shown", "value_description": ""}
            ]},
            {"table_name": "row only", "columns": []},
        ]
        context = api["build_filtered_context"](
            task, metadata, execute=execute, code_root=ROOT
        )
        self.assertEqual([item[1] for item in calls], ['SELECT "kept" FROM "has columns" LIMIT 3'])
        self.assertIn("Table has columns, columns = [*,kept]", context)
        self.assertIn("Table row only, columns = [*]", context)
        self.assertIn("Sample rows: []", context)
        self.assertEqual(
            api["build_filtered_context"](task, [], execute=execute, code_root=ROOT), ""
        )

    def test_real_source_snapshot_has_exactly_5320_unique_bound_tasks(self):
        api = core_api()
        source = ROOT / "baselines_reproduce/din_sql/batches/din_five_groups_prepared_20260916_v1"
        snapshot = api["load_source_snapshot"](source, ROOT)
        self.assertEqual(len(snapshot.tasks), 5320)
        self.assertEqual(len(snapshot.native_linking), 5320)
        self.assertEqual(set(snapshot.tasks), set(snapshot.native_linking))
        self.assertEqual(
            {group: sum(key.group == group for key in snapshot.tasks) for group in snapshot.groups},
            {
                "spider_dev": 1034,
                "bird_dev": 1534,
                "bird_interact_full": 410,
                "bird_interact_lite": 195,
                "spider_test": 2147,
            },
        )
        self.assertEqual(set(snapshot.metadata), {task.schema_ref for task in snapshot.tasks.values()})
        self.assertTrue(all(row["status"] == "succeeded" for row in snapshot.native_linking.values()))


if __name__ == "__main__":
    unittest.main()
