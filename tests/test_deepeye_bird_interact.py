"""Offline BIRD-Interact dataset and PostgreSQL value-index checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "DeepEye-SQL"
sys.path.insert(0, str(BASELINE))


def _write_meta(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "original_column_name",
                "column_name",
                "column_description",
                "data_format",
                "value_description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_fixture(root: Path, rows: list[dict[str, object]] | None = None) -> Path:
    preprocessed = root / "preprocessed_data"
    preprocessed.mkdir(parents=True)
    (preprocessed / "bird_interact_lite.json").write_text(
        json.dumps(
            rows
            if rows is not None
            else [
                {
                    "index": "demo_007",
                    "db_id": "DemoDB",
                    "question": "Return the display value.",
                    "evidence": "Use the display label.",
                    "SQL": "SELECT physical_secret FROM items",
                    "follow_up": {"question": "This turn must not be loaded."},
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_meta(
        preprocessed / "meta" / "demodb" / "Items.csv",
        [
            {
                "original_column_name": "id",
                "column_name": "Identifier",
                "column_description": "Stable row identifier.",
                "data_format": "INTEGER",
                "value_description": "",
            },
            {
                "original_column_name": "label",
                "column_name": "Display Label",
                "column_description": "Human-readable label.",
                "data_format": "VARCHAR(80)",
                "value_description": "Shown to users.",
            },
            {
                "original_column_name": "payload",
                "column_name": "",
                "column_description": (
                    "Structured event payload.\n"
                    'Fields: {"payload_field":"A nested semantic field."}'
                ),
                "data_format": "jsonb",
                "value_description": "",
            },
        ],
    )
    return preprocessed


def _config(root: Path, **overrides: object):
    from scripts.baseline_adapters.deepeye.dataset import (
        BirdInteractDatasetConfig,
    )

    values = {"type": "bird_interact", "split": "lite", "root_path": str(root)}
    values.update(overrides)
    return BirdInteractDatasetConfig(**values)


class BirdInteractDatasetTest(unittest.TestCase):
    def test_loads_meta_allowlist_and_preserves_identity_through_snapshot(self) -> None:
        """Using gold/follow-up fields, flattening JSONB, or coercing IDs must fail."""
        from scripts.baseline_adapters.deepeye.dataset import (
            BirdInteractDataItem,
            BirdInteractDataset,
        )
        from app.dataset.utils import load_dataset, save_dataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            preprocessed = _make_fixture(root)
            dataset = BirdInteractDataset(_config(preprocessed))
            item = dataset[0]

            self.assertIsInstance(item, BirdInteractDataItem)
            self.assertEqual(item.instance_id, "demo_007")
            self.assertEqual(item.question_id, 0)
            self.assertEqual(item.db_type, "postgresql")
            self.assertEqual(item.database_id, "DemoDB")
            self.assertEqual(item.database_path, "DemoDB")
            self.assertEqual(item.gold_sql, "")
            self.assertEqual(item.question, "Return the display value.")
            self.assertEqual(item.evidence, "Use the display label.")
            self.assertEqual(
                list(item.database_schema["tables"]["Items"]["columns"]),
                ["id", "label", "payload"],
            )
            self.assertNotIn(
                "payload_field",
                item.database_schema["tables"]["Items"]["columns"],
            )
            payload = item.database_schema["tables"]["Items"]["columns"]["payload"]
            self.assertIn("Structured event payload.", payload["description"])
            self.assertIn('"payload_field"', payload["description"])
            self.assertFalse(payload["primary_key"])
            self.assertEqual(payload["foreign_keys"], [])

            snapshot = root / "dataset.snapshot"
            save_dataset(dataset, str(snapshot))
            snapshot_manifest = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(
                snapshot_manifest["dataset_class_module"],
                "scripts.baseline_adapters.deepeye.dataset",
            )
            self.assertEqual(
                snapshot_manifest["item_class_module"],
                "scripts.baseline_adapters.deepeye.dataset",
            )
            restored = load_dataset(str(snapshot))
            self.assertIsInstance(restored[0], BirdInteractDataItem)
            self.assertEqual(restored[0].instance_id, "demo_007")
            self.assertEqual(restored[0].db_type, "postgresql")

    def test_config_accepts_both_interact_splits_and_dataset_selects_requested_ids(self) -> None:
        """Adapter config or source-order-only filtering must not ignore explicit IDs."""
        from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            preprocessed = _make_fixture(
                root,
                [
                    {"index": "demo_1", "db_id": "DemoDB", "question": "One", "evidence": ""},
                    {"index": "demo_2", "db_id": "DemoDB", "question": "Two", "evidence": ""},
                ],
            )
            dataset = BirdInteractDataset(_config(preprocessed))
            self.assertEqual([item.instance_id for item in dataset], ["demo_1", "demo_2"])

            selected = BirdInteractDataset(
                _config(preprocessed), instance_ids=["demo_2", "demo_1"]
            )
            self.assertEqual([item.instance_id for item in selected], ["demo_2", "demo_1"])

            full_root = root / "full" / "preprocessed_data"
            full_root.mkdir(parents=True)
            (full_root / "bird_interact_full.json").write_text("[]", encoding="utf-8")
            (full_root / "meta").mkdir()
            full = BirdInteractDataset(_config(full_root, split="full"))
            self.assertEqual(len(full), 0)

    def test_rejects_duplicate_missing_unsafe_and_malformed_inputs(self) -> None:
        """Ambiguous IDs and incomplete Meta must fail closed instead of shrinking schema."""
        from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate_root = _make_fixture(
                root / "duplicate",
                [
                    {"index": "same", "db_id": "DemoDB", "question": "One", "evidence": ""},
                    {"index": "same", "db_id": "DemoDB", "question": "Two", "evidence": ""},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate.*same"):
                BirdInteractDataset(_config(duplicate_root))

            valid_root = _make_fixture(root / "valid")
            with self.assertRaisesRegex(ValueError, "missing.*not_there"):
                BirdInteractDataset(
                    _config(valid_root), instance_ids=["not_there"]
                )

            unsafe_root = _make_fixture(
                root / "unsafe",
                [{"index": "bad", "db_id": "../escape", "question": "Bad", "evidence": ""}],
            )
            with self.assertRaisesRegex(ValueError, "safe path component"):
                BirdInteractDataset(_config(unsafe_root))

            missing_meta_root = _make_fixture(root / "missing-meta")
            for path in (missing_meta_root / "meta" / "demodb").glob("*.csv"):
                path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Meta"):
                BirdInteractDataset(_config(missing_meta_root))

            malformed_root = _make_fixture(root / "malformed")
            _write_meta(
                malformed_root / "meta" / "demodb" / "Items.csv",
                [
                    {
                        "original_column_name": "",
                        "column_name": "",
                        "column_description": "Missing physical name.",
                        "data_format": "TEXT",
                        "value_description": "",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "column"):
                BirdInteractDataset(_config(malformed_root))


class PostgresValueIndexTest(unittest.TestCase):
    @staticmethod
    def _item() -> SimpleNamespace:
        return SimpleNamespace(
            db_type="postgresql",
            database_id="DemoDB",
            database_schema={
                "db_id": "DemoDB",
                "db_type": "postgresql",
                "tables": {
                    "Items": {
                        "columns": {
                            "id": {"column_type": "INTEGER"},
                            "label": {"column_type": "VARCHAR(80)"},
                            "code": {"column_type": "CHAR(36)"},
                            "payload": {"column_type": "JSONB"},
                        }
                    }
                },
            },
        )

    def test_builds_native_deterministic_capped_index_and_provenance(self) -> None:
        """Unbounded queries, JSONB flattening, or unstable value order must fail."""
        from app.db_utils.execution import SQLExecutionResult
        from app.vector_db.local_index import LocalValueIndex, get_local_index_path
        from scripts.baseline_adapters.deepeye.postgres_index import (
            build_postgres_value_index,
        )

        queries: list[str] = []

        def execute(_item, sql: str, timeout=None):
            queries.append(sql)
            if '"label"' in sql:
                return SQLExecutionResult(
                    result_type="success",
                    db_path="DemoDB",
                    sql=sql,
                    result_cols=["label"],
                    result_rows=[("Zulu",), ("Alpha",), ("Alpha",), ("x" * 101,)],
                )
            return SQLExecutionResult(
                result_type="success",
                db_path="DemoDB",
                sql=sql,
                result_cols=["code"],
                result_rows=[
                    ("123e4567-e89b-12d3-a456-426614174000",),
                    ("123e4567-e89b-12d3-a456-426614174001",),
                ],
            )

        batches: list[list[str]] = []

        def embed(values: list[str]) -> list[list[float]]:
            batches.append(list(values))
            return [[float(len(value)), 1.0] for value in values]

        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "scripts.baseline_adapters.deepeye.postgres_execution.execute_postgres_sql",
            side_effect=execute,
        ):
            store_root = Path(temporary_directory)
            first = build_postgres_value_index(
                self._item(), store_root, embed, max_values_per_column=2,
                embedding_batch_size=1,
            )
            index_path = get_local_index_path(store_root / "DemoDB")
            first_manifest = (index_path / "manifest.json").read_bytes()
            first_documents = next((index_path / "columns").glob("*.documents.json")).read_bytes()

            second = build_postgres_value_index(
                self._item(), store_root, embed, max_values_per_column=2,
                embedding_batch_size=1,
            )
            self.assertEqual(first, second)
            self.assertEqual((index_path / "manifest.json").read_bytes(), first_manifest)
            self.assertEqual(
                next((index_path / "columns").glob("*.documents.json")).read_bytes(),
                first_documents,
            )

            self.assertEqual(json.loads(first_documents), ["Alpha", "Zulu"])
            self.assertEqual(first["text_column_count"], 2)
            self.assertEqual(first["indexed_column_count"], 1)
            self.assertEqual(first["value_count"], 2)
            self.assertEqual(first["max_values_per_column"], 2)
            self.assertEqual(first["max_value_length"], 100)
            self.assertEqual(first["embedding_batch_size"], 1)
            self.assertEqual(batches[:2], [["Alpha"], ["Zulu"]])
            self.assertEqual(len(queries), 4)
            self.assertTrue(all("DISTINCT" in query for query in queries))
            self.assertTrue(all("ORDER BY" in query for query in queries))
            self.assertTrue(all("LIMIT 2" in query for query in queries))
            self.assertTrue(all('"payload"' not in query for query in queries))
            self.assertTrue(all('FROM "Items" ' in query for query in queries))

            manifest = json.loads(first_manifest)
            self.assertEqual(manifest["provenance"], first)
            self.assertEqual(len(manifest["columns"]), 1)
            loaded = LocalValueIndex(index_path, device="cpu")
            retrieved = loaded.retrieve_values_for_column(
                [[5.0, 1.0]], "Items", "label", 2, lower_meta_data=True
            )
            self.assertEqual(
                {entry["value"] for entry in retrieved["values"]},
                {"Alpha", "Zulu"},
            )

    def test_query_failure_is_not_recorded_as_an_empty_index(self) -> None:
        """A PostgreSQL error must abort without replacing a valid-looking manifest."""
        from app.db_utils.execution import SQLExecutionResult
        from scripts.baseline_adapters.deepeye.postgres_index import (
            build_postgres_value_index,
        )

        failed = SQLExecutionResult(
            result_type="execution_error",
            db_path="DemoDB",
            sql="SELECT",
            error_message="sanitized PostgreSQL failure",
        )
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "scripts.baseline_adapters.deepeye.postgres_execution.execute_postgres_sql",
            return_value=failed,
        ):
            store_root = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "Items.label"):
                build_postgres_value_index(self._item(), store_root, lambda values: [])
            self.assertFalse(
                (store_root / "DemoDB" / "local_index" / "manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
