"""Tests for the read-only gold-SQL annotation source snapshot."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import fields
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rc_evaluation.schema_linking_gold.source import (
    _linked_schemas,
    canonical_schema,
    conservative_reference,
    feature_tags,
    load_offline_groups,
    select_pilot,
)


ROOT = Path(__file__).resolve().parents[4] / "docs" / "analysis_rc3_five_groups_20260915"
EXPECTED_GROUP_SIZES = {
    "bird_dev": 1534,
    "bird_interact_full": 410,
    "bird_interact_lite": 195,
    "spider_dev": 1034,
    "spider_test": 2147,
}


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_offline_groups(ROOT)

    def test_loads_every_verified_offline_binding_with_reference_sql(self):
        """Catches a loader that skips a group or binds SQL by source-row position."""
        self.assertEqual(len(self.tasks), 5320)
        self.assertEqual(Counter(task.group for task in self.tasks), EXPECTED_GROUP_SIZES)
        self.assertEqual(len({task.task_key for task in self.tasks}), 5320)
        for task in self.tasks:
            self.assertTrue(task.gold_sql.strip(), task.task_key)
            self.assertTrue(task.task_key.startswith(task.partition + "/"), task.task_key)
            self.assertIn(task.dialect, {"sqlite", "postgresql"})

        # BIRD-Interact uses a string identity: the reference is keyed by that
        # identity and must not be recovered from a source-row integer.
        bird = next(task for task in self.tasks if task.task_key == "bird_interact/lite/s:credit_11")
        self.assertEqual(bird.external_id, "credit_11")
        self.assertEqual(bird.gold_sql, bird.source_reference["sql"])

    def test_loader_uses_only_the_five_offline_json_artifacts(self):
        """Catches a loader that consults a sibling metrics artifact."""
        with tempfile.TemporaryDirectory() as directory:
            fixture_root = self._offline_root(Path(directory))
            (fixture_root / "schema_linking_metrics.json").write_text("not JSON", encoding="utf-8")
            tasks = load_offline_groups(fixture_root)
        self.assertEqual(len(tasks), 5320)
        self.assertTrue(any(field.name == "conservative_reference" for field in fields(tasks[0])))

    def test_conservative_reference_uses_frozen_schema_and_reports_unavailable_queries(self):
        """Catches missing parser calibration or invented elements for ambiguous SQL."""
        simple = next(task for task in self.tasks if task.task_key == "spider/dev/i:866")
        self.assertEqual(simple.conservative_reference, {
            "status": "available", "reason": None,
            "tables": ("highschooler",), "columns": (("highschooler", "grade"),),
        })
        ambiguous = conservative_reference(
            "SELECT * FROM singer", "sqlite",
            canonical_schema({"tables": {"singer": {"columns": {"Name": {"column_type": "TEXT"}}}}}),
        )
        self.assertEqual(ambiguous, {
            "status": "unavailable", "reason": "unresolved_or_wildcard_column",
            "tables": (), "columns": (),
        })

    def test_bird_interact_tasks_do_not_expose_full_binding_or_source_row(self):
        """Catches prohibited BIRD-Interact source-row provenance escaping the loader."""
        bird = next(task for task in self.tasks if task.task_key == "bird_interact/lite/s:credit_11")
        names = {field.name for field in fields(bird)}
        self.assertNotIn("source_binding", names)
        self.assertNotIn("source_row", names)
        self.assertNotIn("source_row", bird.source_reference)

    def test_loader_rejects_a_truncated_frozen_group(self):
        """Catches acceptance of internally consistent inputs with a wrong frozen count."""
        with tempfile.TemporaryDirectory() as directory:
            fixture_root = self._offline_root(Path(directory))
            group = "bird_interact_lite"
            payload = json.loads((ROOT / group / "offline.json").read_text(encoding="utf-8"))
            key = payload["bindings"].pop()["task_key"]
            payload["references"].pop(key)
            payload["inputs"].pop(key)
            payload["contracts"].pop(key)
            payload["records"] = [record for record in payload["records"] if record["item_key"] != key]
            (fixture_root / group).unlink()
            (fixture_root / group).mkdir()
            (fixture_root / group / "offline.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen group count"):
                load_offline_groups(fixture_root)

    def test_linked_schema_rejects_non_list_table_columns(self):
        """Catches malformed linked tables being silently discarded from a task."""
        records = [
            {"item_key": "demo", "stage": "schema_linking", "condition": "native",
             "status": "succeeded", "linked": {"orders": "id"}},
            {"item_key": "demo", "stage": "schema_linking", "condition": "rc",
             "status": "succeeded", "linked": {"orders": ["id"]}},
        ]
        with self.assertRaisesRegex(ValueError, "linked columns"):
            _linked_schemas(records, "demo")

    def test_freezes_stable_hashes_reuse_keys_and_linking_references(self):
        """Catches unstable hashing or an annotation cache shared across schemas."""
        repeated = load_offline_groups(ROOT)
        by_key = {task.task_key: task for task in self.tasks}
        for copy in repeated:
            task = by_key[copy.task_key]
            self.assertEqual(task.sql_sha256, copy.sql_sha256)
            self.assertEqual(task.schema_sha256, copy.schema_sha256)
            self.assertEqual(task.reuse_key, copy.reuse_key)
            self.assertEqual(len(task.sql_sha256), 64)
            self.assertEqual(len(task.schema_sha256), 64)
            self.assertEqual(task.native_linked_schema, copy.native_linked_schema)
            self.assertEqual(task.rc_linked_schema, copy.rc_linked_schema)

        task = by_key["spider/dev/i:169"]
        self.assertEqual(task.native_linked_schema, {"cars_data": ("Cylinders", "Id", "MPG", "Year")})
        self.assertEqual(task.rc_linked_schema, task.native_linked_schema)

        reuse = defaultdict(list)
        for task in self.tasks:
            reuse[task.reuse_key].append(task)
        duplicate_sets = [items for items in reuse.values() if len(items) > 1]
        self.assertTrue(duplicate_sets)
        for items in duplicate_sets:
            self.assertEqual({task.schema_sha256 for task in items}, {items[0].schema_sha256})
            self.assertEqual({task.sql_sha256 for task in items}, {items[0].sql_sha256})

    def test_canonical_schema_assigns_sort_stable_ids_and_reverse_mappings(self):
        """Catches catalog IDs changing with input dictionary insertion order."""
        schema = {
            "tables": {
                "zeta": {"columns": {"b": {"column_type": "INTEGER"}, "a": {"column_type": "TEXT"}}},
                "alpha": {"columns": {"name": {"column_type": "TEXT"}}},
            }
        }
        catalog = canonical_schema(schema)
        self.assertEqual(catalog["tables"], (
            {"id": "T1", "name": "alpha", "columns": ({"id": "C1", "name": "name", "type": "TEXT"},)},
            {"id": "T2", "name": "zeta", "columns": (
                {"id": "C2", "name": "a", "type": "TEXT"},
                {"id": "C3", "name": "b", "type": "INTEGER"},
            )},
        ))
        self.assertEqual(catalog["table_by_id"], {"T1": "alpha", "T2": "zeta"})
        self.assertEqual(catalog["column_by_id"], {
            "C1": {"table": "alpha", "column": "name"},
            "C2": {"table": "zeta", "column": "a"},
            "C3": {"table": "zeta", "column": "b"},
        })
        self.assertEqual(catalog["column_id_by_name"], {
            "alpha": {"name": "C1"}, "zeta": {"a": "C2", "b": "C3"},
        })

    def test_feature_tags_classify_pilot_strata_conservatively(self):
        """Catches a sampler that loses a documented complex-SQL stratum."""
        self.assertEqual(feature_tags("SELECT count(*) FROM singer", "sqlite"), ("ordinary",))
        self.assertEqual(feature_tags("WITH x AS (SELECT * FROM t) SELECT * FROM x", "sqlite"),
                         ("cte_or_subquery", "wildcard"))
        self.assertEqual(feature_tags("SELECT a FROM t UNION SELECT a FROM u", "postgresql"),
                         ("set_operation",))
        self.assertEqual(feature_tags("SELECT payload->>'name' FROM public.t", "postgresql"),
                         ("json", "schema_qualified"))
        self.assertEqual(feature_tags("SELECT * FROM t JOIN u USING (id)", "postgresql"),
                         ("wildcard", "using_or_natural_join"))
        self.assertEqual(feature_tags('SELECT Name FROM singer WHERE Citizenship != "France"', "sqlite"),
                         ("sqlite_double_quote_ambiguity",))
        self.assertEqual(feature_tags("SELECT * FROM t, LATERAL jsonb_each(payload)", "postgresql"),
                         ("wildcard", "json", "lateral_or_table_function"))

    def test_pilot_is_deterministic_and_covers_groups_and_present_features(self):
        """Catches nondeterministic or unstratified pilot selection."""
        pilot = select_pilot(self.tasks, size=200, seed=20260916)
        self.assertEqual([task.task_key for task in pilot],
                         [task.task_key for task in select_pilot(self.tasks, size=200, seed=20260916)])
        self.assertEqual(len(pilot), 200)
        self.assertEqual(len({task.task_key for task in pilot}), 200)
        self.assertEqual({task.group for task in pilot}, set(EXPECTED_GROUP_SIZES))
        all_features = {feature for task in self.tasks for feature in task.features}
        pilot_features = {feature for task in pilot for feature in task.features}
        self.assertTrue(all_features <= pilot_features)

    @staticmethod
    def _offline_root(destination: Path) -> Path:
        for group in EXPECTED_GROUP_SIZES:
            (destination / group).symlink_to(ROOT / group, target_is_directory=True)
        return destination


if __name__ == "__main__":
    unittest.main()
