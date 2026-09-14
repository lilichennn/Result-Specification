from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import generate_rc as generator


FIELDS = ("population", "row_grain", "column_role", "derivation", "filter_policy")
ANSWER = {**{key: "Corrected " + key for key in FIELDS}, "meta_review": "Reviewed"}


def instance(index: str) -> dict:
    return {"index": index, "db_id": "demo", "question": "Question " + index, "evidence": ""}


def contract(index: str) -> dict:
    return {
        **instance(index),
        "round1_status": "succeeded", "round1_error": None,
        "round2_status": "succeeded", "round2_error": None,
        "rc_round1": {key: "Original " + key for key in FIELDS},
        "rc_round2": {**{key: "Original " + key for key in FIELDS}, "meta_review": "Original review"},
        "extra_metadata": {"keep": index},
    }


def gold(index: str) -> dict:
    row = instance(index)
    return {key: row[key] for key in ("index", "db_id", "question")} | {"gold_sql": "SELECT '" + index + "'"}


class PartialRound3Test(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "questions.json"
        self.gold = self.root / "gold.json"
        self.output = self.root / "rc.json"
        self.calls = []

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")

    def setup_rows(self, inputs=("a", "b", "c"), references=("a",), records=None):
        self.write(self.input, [instance(x) for x in inputs])
        self.write(self.gold, [gold(x) for x in references])
        self.original = records if records is not None else [contract(x) for x in inputs]
        self.write(self.output, self.original)

    def model(self, messages):
        self.calls.append(messages)
        return json.dumps(ANSWER)

    def run_generation(self, partial=True):
        try:
            return generator.generate_round3_file(
                self.input, self.gold, self.output, model_call=self.model,
                concurrency=1, allow_partial_gold=partial,
            )
        except TypeError as exc:
            if "allow_partial_gold" in str(exc):
                self.fail("Round3 cannot yet safely accept an explicitly partial gold input")
            raise

    def test_partial_updates_only_selected_record_and_preserves_order(self):
        self.setup_rows(records=[contract("b"), contract("a"), contract("c")])
        result = self.run_generation()
        self.assertEqual([row["index"] for row in result], ["b", "a", "c"])
        self.assertEqual(result[0], self.original[0])
        self.assertEqual(result[2], self.original[2])
        self.assertEqual(result[1]["rc_round1"], self.original[1]["rc_round1"])
        self.assertEqual(result[1]["rc_round2"], self.original[1]["rc_round2"])
        self.assertEqual(result[1]["rc_round3"], ANSWER)
        self.assertEqual(result[1]["round3_status"], "succeeded")
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("SELECT 'a'", self.calls[0][1]["content"])

    def test_partial_mode_is_not_enabled_by_default(self):
        self.setup_rows()
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "ID sets"):
            generator.generate_round3_file(self.input, self.gold, self.output, model_call=self.model)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.output.read_bytes(), before)

    def test_unknown_reference_id_fails_even_in_partial_mode(self):
        self.setup_rows(references=("a", "outside"))
        before = self.output.read_bytes()
        with self.assertRaises(ValueError):
            self.run_generation()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.output.read_bytes(), before)

    def test_duplicate_or_mismatched_reference_fails_before_calls(self):
        for rows in ([gold("a"), gold("a")], [gold("a") | {"db_id": "wrong"}],
                     [gold("a") | {"question": "wrong"}], [gold("a") | {"gold_sql": " "}]):
            with self.subTest(rows=rows):
                self.setup_rows()
                self.write(self.gold, rows)
                before = self.output.read_bytes()
                with self.assertRaises(ValueError):
                    self.run_generation()
                self.assertEqual(self.calls, [])
                self.assertEqual(self.output.read_bytes(), before)

    def test_unselected_legacy_and_extra_records_are_untouched(self):
        legacy = contract("b")
        legacy.pop("round1_status")
        legacy["round2_status"] = "failed"
        records = [contract("a"), legacy, contract("outside")]
        self.setup_rows(inputs=("a", "b"), records=records)
        result = self.run_generation()
        self.assertEqual(result[1:], records[1:])
        self.assertEqual(len(self.calls), 1)

    def test_selected_legacy_round1_status_can_still_be_inferred(self):
        selected = contract("a")
        selected.pop("round1_status")
        self.setup_rows(inputs=("a",), records=[selected])
        result = self.run_generation()
        self.assertEqual(result[0]["rc_round3"], ANSWER)
        self.assertNotIn("round1_status", result[0])

    def test_selected_missing_round2_fails_without_changing_output(self):
        selected = contract("a")
        selected["rc_round2"] = None
        self.setup_rows(records=[selected, contract("b"), contract("c")])
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "Round-2"):
            self.run_generation()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.output.read_bytes(), before)

    def test_existing_success_is_reused_without_new_calls(self):
        selected = contract("a") | {"rc_round3": ANSWER, "round3_status": "succeeded", "round3_error": None}
        self.setup_rows(records=[selected, contract("b"), contract("c")])
        before = self.output.read_bytes()
        self.assertEqual(self.run_generation(), self.original)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.output.read_bytes(), before)

    def test_empty_explicit_subset_is_a_noop(self):
        self.setup_rows(references=())
        before = self.output.read_bytes()
        self.assertEqual(self.run_generation(), self.original)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.calls, [])

    def test_interruption_keeps_whole_file_and_resume_only_finishes_pending(self):
        self.setup_rows(references=("a", "b"))
        original_write = generator._write_json_atomic
        def commit_then_interrupt(path, value):
            original_write(path, value)
            self.assertEqual([r["index"] for r in json.loads(path.read_text())], ["a", "b", "c"])
            raise RuntimeError("simulated interruption after first committed result")
        with patch.object(generator, "_write_json_atomic", side_effect=commit_then_interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_generation()
        interrupted = json.loads(self.output.read_text())
        self.assertEqual(interrupted[0]["rc_round3"], ANSWER)
        self.assertEqual(interrupted[1:], self.original[1:])
        self.calls.clear()
        result = self.run_generation()
        self.assertEqual(len(self.calls), 1)
        self.assertIn("SELECT 'b'", self.calls[0][1]["content"])
        self.assertEqual(result[0], interrupted[0])
        self.assertEqual(result[2], self.original[2])

    def test_complete_gold_input_remains_supported(self):
        self.setup_rows(inputs=("a", "b"), references=("a", "b"))
        result = self.run_generation(partial=False)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(row["round3_status"] == "succeeded" for row in result))

    def test_cli_reads_custom_gold_file_and_preserves_unselected_rows(self):
        split_root = self.root / "spider2_lite"
        self.input = split_root / "preprocessed_data/spider2_lite.json"
        self.output = split_root / "rc.json"
        self.setup_rows()
        args = ["generate_rc.py", "--dataset_split", "spider2_lite", "--llm", "qwen38",
                "--round3", "--gold-file", str(self.gold), "--allow-partial-gold", "--concurrency", "1"]
        with patch.object(generator, "SCRIPT_DIR", self.root), patch("sys.argv", args), \
             patch.object(generator, "configure_logging"), \
             patch.object(generator, "call_model", side_effect=lambda messages, llm: self.model(messages)):
            generator.main()
        result = json.loads(self.output.read_text())
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["rc_round3"], ANSWER)
        self.assertEqual(result[1:], self.original[1:])

    def test_partial_flags_require_round3(self):
        args = ["generate_rc.py", "--dataset_split", "demo", "--llm", "qwen38", "--allow-partial-gold"]
        with patch("sys.argv", args), self.assertRaises(SystemExit) as error:
            generator.parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
