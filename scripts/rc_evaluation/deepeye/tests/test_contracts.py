from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import tempfile

from scripts.rc_evaluation.deepeye import contracts as contracts_module
from scripts.rc_evaluation.deepeye.contracts import load_contracts


ROUND1 = {
    "population": "population one",
    "row_grain": "row one",
    "column_role": "columns one",
    "derivation": "derivation one",
    "filter_policy": "filter one",
}
ROUND2 = {**ROUND1, "meta_review": "review one"}


def _record(index: str = "item_1", **changes: object) -> dict:
    record = {
        "index": index,
        "db_id": "database_one",
        "question": "Question one?",
        "evidence": "Evidence one.",
        "round1_status": "succeeded",
        "round1_error": None,
        "rc_round1": ROUND1,
        "round2_status": "succeeded",
        "round2_error": None,
        "rc_round2": ROUND2,
    }
    record.update(changes)
    return record


def _item(instance_id: str = "item_1", **changes: object) -> SimpleNamespace:
    values = {
        "instance_id": instance_id,
        "database_id": "database_one",
        "question": "Question one?",
        "evidence": "Evidence one.",
    }
    values.update(changes)
    return SimpleNamespace(**values)


class LoadContractsTest(TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def _write(self, records: list[dict], filename: str = "contracts.json") -> Path:
        path = self.root / filename
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_contract_module_exposes_render_function_for_controller(self) -> None:
        self.assertTrue(callable(getattr(contracts_module, "render_rc_block", None)))

    def test_loads_exact_selected_record_with_normalized_rcs_and_hashes(self) -> None:
        selected = _record()
        path = self._write([selected, _record("not_selected")])
        canonical = json.dumps(
            selected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        with (
            patch("result_contract.rc.generate_round1", side_effect=AssertionError("must not generate")),
            patch("result_contract.rc.generate_round2", side_effect=AssertionError("must not generate")),
        ):
            contracts = load_contracts({"lite": path}, [("lite", _item())])

        self.assertEqual(set(contracts), {"lite/item_1"})
        contract = contracts["lite/item_1"]
        self.assertEqual(
            contract,
            {
                "task_key": "lite/item_1",
                "db_id": "database_one",
                "question": "Question one?",
                "evidence": "Evidence one.",
                "round1": ROUND1,
                "round2": ROUND2,
                "source_file": str(path.resolve()),
                "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "record_sha256": hashlib.sha256(canonical).hexdigest(),
            },
        )

    def test_rejects_duplicate_indices_in_a_source_file(self) -> None:
        path = self._write([_record(), _record()])

        with self.assertRaisesRegex(ValueError, "duplicate.*item_1"):
            load_contracts({"lite": path}, [("lite", _item())])

    def test_rejects_missing_selected_record(self) -> None:
        path = self._write([_record("some_other_item")])

        with self.assertRaisesRegex(ValueError, "missing.*lite/item_1"):
            load_contracts({"lite": path}, [("lite", _item())])

    def test_rejects_duplicate_selected_tasks(self) -> None:
        path = self._write([_record()])

        with self.assertRaisesRegex(ValueError, "duplicate task.*lite/item_1"):
            load_contracts({"lite": path}, [("lite", _item()), ("lite", _item())])

    def test_rejects_database_question_or_evidence_misalignment(self) -> None:
        cases = (
            ("database", _record(db_id="other_database")),
            ("question", _record(question="Different question?")),
            ("evidence", _record(evidence="Different evidence.")),
        )
        for label, record in cases:
            with self.subTest(label=label):
                path = self._write([record], f"{label}.json")
                with self.assertRaisesRegex(ValueError, f"{label}.*lite/item_1"):
                    load_contracts({"lite": path}, [("lite", _item())])

    def test_rejects_failed_or_incomplete_rounds(self) -> None:
        cases = (
            ("Round 1.*failed", _record(round1_status="failed", round1_error="bad")),
            ("Round 2.*failed", _record(round2_status="failed", round2_error="bad")),
            (
                "Round-2 RC fields",
                _record(rc_round2={key: value for key, value in ROUND2.items() if key != "meta_review"}),
            ),
        )
        for position, (message, record) in enumerate(cases):
            with self.subTest(message=message):
                path = self._write([record], f"failed_{position}.json")
                with self.assertRaisesRegex(ValueError, message):
                    load_contracts({"lite": path}, [("lite", _item())])

    def test_rejects_missing_variant_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "source.*full"):
            load_contracts({}, [("full", _item())])
