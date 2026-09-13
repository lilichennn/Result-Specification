"""Offline adapter relocation and immutable historical-record compatibility."""

import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "baselines/DeepEye-SQL"))
OLD_DATASET = "result_contract.baseline_adapters.deepeye.dataset"
NEW_PACKAGE = "scripts.baseline_adapters.deepeye"


@dataclasses.dataclass
class UnrelatedRecord:
    value: str


class AdapterLayoutTests(unittest.TestCase):
    def migrated_module(self, suffix):
        try:
            return importlib.import_module(f"{NEW_PACKAGE}.{suffix}")
        except ModuleNotFoundError as error:
            self.fail(f"Relocated adapter must be importable: {error}")

    def legacy_config(self):
        return {
            "__run_store_type__": "pydantic",
            "module": OLD_DATASET,
            "qualname": "BirdInteractDatasetConfig",
            "fields": {"split": "lite", "root_path": "/offline/data",
                       "save_path": "/offline/frozen.snapshot"},
        }

    def test_old_type_reference_restores_as_the_relocated_config_class(self):
        """Moving modules must not strand old Pydantic type references."""
        store_module = self.migrated_module("run_store")
        dataset = self.migrated_module("dataset")
        restored = store_module.restore_jsonable(self.legacy_config())
        self.assertIs(type(restored), dataset.BirdInteractDatasetConfig)
        self.assertEqual(restored.split, "lite")
        self.assertEqual(restored.root_path, "/offline/data")
        self.assertEqual(restored.save_path, "/offline/frozen.snapshot")

    def test_historical_manifest_reads_and_verifies_without_rewriting_bytes(self):
        """Compatibility belongs in type resolution, not a rewrite of hashed records."""
        store_module = self.migrated_module("run_store")
        dataset = self.migrated_module("dataset")
        config = dataset.BirdInteractDatasetConfig(**self.legacy_config()["fields"])
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            # Emit a real old type label at creation; immutable rows are never updated.
            with patch.object(dataset.BirdInteractDatasetConfig, "__module__", OLD_DATASET):
                with store_module.RunStore.create(run_dir, {"config": config}):
                    pass
            database = run_dir / "run.sqlite3"
            with sqlite3.connect(database) as connection:
                raw, checksum = connection.execute(
                    "SELECT payload_json, payload_checksum FROM manifest"
                ).fetchone()
            self.assertEqual(json.loads(raw)["config"]["module"], OLD_DATASET)
            self.assertEqual(checksum, hashlib.sha256(raw.encode("utf-8")).hexdigest())
            self.assertEqual(dataset.BirdInteractDatasetConfig.__module__,
                             f"{NEW_PACKAGE}.dataset")
            with store_module.RunStore.open(run_dir, read_only=True) as store:
                self.assertIs(type(store.manifest["config"]),
                              dataset.BirdInteractDatasetConfig)
                self.assertTrue(store.verify()["ok"])
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute(
                    "SELECT payload_json, payload_checksum FROM manifest"
                ).fetchone(), (raw, checksum))

    def test_legacy_mapping_does_not_rewrite_similar_or_embedded_prefixes(self):
        """Substring replacement would break unrelated, otherwise importable types."""
        store_module = self.migrated_module("run_store")
        for name in (
            "result_contract.baseline_adapters_extra.dataset",
            "vendor.result_contract.baseline_adapters.deepeye.dataset",
        ):
            module = ModuleType(name)
            module.UnrelatedRecord = UnrelatedRecord
            # Real importlib resolution from an isolated module-cache fixture.
            with self.subTest(module=name), patch.dict(sys.modules, {name: module}):
                restored = store_module.restore_jsonable({
                    "__run_store_type__": "dataclass", "module": name,
                    "qualname": "UnrelatedRecord", "fields": {"value": "kept"},
                })
                self.assertEqual(restored, UnrelatedRecord("kept"))


if __name__ == "__main__":
    unittest.main()
