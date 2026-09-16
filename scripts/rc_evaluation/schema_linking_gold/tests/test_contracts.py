"""Behavioral tests for blind gold-SQL annotation model contracts."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from scripts.rc_evaluation.schema_linking_gold.contracts import (
    normalize_annotation,
    parse_response,
)
from scripts.rc_evaluation.schema_linking_gold.prompt import render_batch_prompt
from scripts.rc_evaluation.schema_linking_gold.source import canonical_schema


def _task(key: str = "spider/dev/i:0") -> SimpleNamespace:
    return SimpleNamespace(
        task_key=key,
        dialect="postgresql",
        gold_sql="SELECT COUNT(*) FROM orders",
        schema=canonical_schema({"tables": {
            "orders": {"columns": {
                "id": {"column_type": "INTEGER"},
                "payload": {"column_type": "JSONB"},
            }},
            "users": {"columns": {"id": {"column_type": "INTEGER"}}},
        }}),
        question="This must never reach the model",
        evidence="This must never reach the model",
        native_linked_schema={"leak_native": ("id",)},
        rc_linked_schema={"leak_rc": ("id",)},
    )


def _raw(task: SimpleNamespace, **changes: object) -> dict:
    value = {
        "task_key": task.task_key,
        "status": "resolved",
        "required_table_ids": ["T1"],
        "required_column_ids": [],
        "evidence": [],
        "json_paths": [],
        "review_reasons": [],
    }
    value.update(changes)
    return value


class GoldAnnotationContractTests(unittest.TestCase):
    def test_count_star_keeps_its_physical_table_without_columns(self):
        """Catches a normalizer that invents a column for COUNT(*)."""
        task = _task()

        annotation = normalize_annotation(_raw(task), task)

        self.assertEqual(annotation["required_table_ids"], ("T1",))
        self.assertEqual(annotation["required_column_ids"], ())
        self.assertEqual(annotation["required_tables"], ("orders",))
        self.assertEqual(annotation["required_columns"], ())

    def test_projection_wildcard_expands_each_catalog_column(self):
        """Catches a response contract that treats table.* as one pseudo-column."""
        task = _task()
        raw = _raw(task, required_column_ids=["C1", "C2"])

        annotation = normalize_annotation(raw, task)

        self.assertEqual(annotation["required_table_ids"], ("T1",))
        self.assertEqual(annotation["required_columns"], (("orders", "id"), ("orders", "payload")))

    def test_json_path_keeps_only_the_physical_carrier_column(self):
        """Catches treating a JSON path as a catalog column."""
        task = _task()
        raw = _raw(task, required_table_ids=[], required_column_ids=["C2"],
                   evidence=["orders.payload supplies the JSON value"],
                   json_paths=[{"column_id": "C2", "path": "$.customer.id"}])

        annotation = normalize_annotation(raw, task)

        self.assertEqual(annotation["required_table_ids"], ("T1",))
        self.assertEqual(annotation["required_columns"], (("orders", "payload"),))
        self.assertEqual(annotation["json_paths"], (("C2", "$.customer.id"),))

    def test_cte_alias_never_becomes_a_physical_table(self):
        """Catches accepting CTE/output aliases as table catalog members."""
        task = _task()
        raw = _raw(task, required_table_ids=[], required_column_ids=["C1"],
                   evidence=["cte alias traces to orders.id"])

        annotation = normalize_annotation(raw, task)

        self.assertEqual(annotation["required_tables"], ("orders",))
        self.assertNotIn("cte_orders", annotation["required_tables"])

    def test_parse_response_rejects_unknown_catalog_ids(self):
        """Catches silently accepting or correcting a model-invented ID."""
        task = _task()
        payload = json.dumps([_raw(task, required_table_ids=["T999"])])

        with self.assertRaisesRegex(ValueError, "unknown.*T999"):
            parse_response(payload, [task])

    def test_parse_response_rejects_duplicate_catalog_ids(self):
        """Catches deduplicating a malformed model response instead of rejecting it."""
        task = _task()
        payload = json.dumps([_raw(task, required_column_ids=["C1", "C1"])])

        with self.assertRaisesRegex(ValueError, "duplicate.*C1"):
            parse_response(payload, [task])

    def test_parse_response_rejects_missing_extra_and_wrong_task_identity(self):
        """Catches a batch parser that does not bind each response to its selected task."""
        first, second = _task("spider/dev/i:0"), _task("spider/dev/i:1")
        cases = {
            "missing": ([_raw(first)], "missing.*spider/dev/i:1"),
            "extra": ([_raw(first), _raw(second), _raw(_task("spider/dev/i:2"))], "extra.*spider/dev/i:2"),
            "wrong": ([_raw(_task("not/a/selected/task"))], "extra.*not/a/selected/task"),
        }
        for label, (entries, message) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                parse_response(json.dumps(entries), [first, second])

    def test_parse_response_accepts_a_fenced_json_array(self):
        """Catches a parser that needlessly rejects JSON fenced by a chat model."""
        task = _task()
        response = "```json\n" + json.dumps([_raw(task)]) + "\n```"

        annotations = parse_response(response, [task])

        self.assertEqual(annotations[task.task_key]["required_tables"], ("orders",))

    def test_needs_review_and_invalid_sql_require_a_reason(self):
        """Catches terminal non-resolved states that cannot be audited later."""
        task = _task()
        for status in ("needs_review", "invalid_sql"):
            with self.subTest(status=status):
                annotation = normalize_annotation(_raw(
                    task, status=status, required_table_ids=[],
                    review_reasons=["unresolvable identifier"],
                ), task)
                self.assertEqual(annotation["status"], status)
                self.assertEqual(annotation["review_reasons"], ("unresolvable identifier",))
        for status in ("needs_review", "invalid_sql"):
            with self.subTest(status=status, missing_reason=True), self.assertRaisesRegex(ValueError, "review reason"):
                normalize_annotation(_raw(task, status=status, required_table_ids=[]), task)

    def test_parser_rejects_malformed_status_and_unexpected_fields(self):
        """Catches accepting a malformed status or an uncontracted response shape."""
        task = _task()
        for raw, message in (
            (_raw(task, status="RESOLVED"), "status"),
            ({key: value for key, value in _raw(task).items() if key != "evidence"}, "missing.*evidence"),
            (_raw(task, unexpected=True), "unexpected"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                normalize_annotation(raw, task)

    def test_prompt_is_blind_and_supplies_a_closed_catalog_and_physical_rules(self):
        """Catches prompt leakage or omission of rules needed for physical dependencies."""
        task = _task()

        prompt = render_batch_prompt([task])

        self.assertIn(task.task_key, prompt)
        self.assertIn(task.gold_sql, prompt)
        self.assertIn('"id":"T1"', prompt)
        for forbidden in (task.question, task.evidence, "leak_native", "leak_rc"):
            self.assertNotIn(forbidden, prompt)
        for required_rule in (
            "COUNT(*)", "table.*", "JSON", "CTE", "USING", "NATURAL JOIN",
            "Function arguments", "SQLite", "PostgreSQL", "needs_review", "invalid_sql",
        ):
            self.assertIn(required_rule, prompt)

    def test_same_schema_batch_serializes_the_catalog_only_once(self):
        """Catches multiplying an identical schema catalog by every SQL in the batch."""
        first, second = _task("spider/dev/i:0"), _task("spider/dev/i:1")

        prompt = render_batch_prompt([first, second])
        payload = json.loads(prompt.split("\n\nINPUTS:\n", 1)[1])

        self.assertEqual(payload["dialect"], "postgresql")
        self.assertEqual(len(payload["schema_catalog"]["tables"]), 2)
        self.assertEqual(
            payload["tasks"],
            [
                {"task_key": first.task_key, "gold_sql": first.gold_sql},
                {"task_key": second.task_key, "gold_sql": second.gold_sql},
            ],
        )


if __name__ == "__main__":
    unittest.main()
