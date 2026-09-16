"""Hand-calculated tests for the one-off gold-annotation report bundle."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rc_evaluation.schema_linking_gold.reporting import export_annotations
from scripts.rc_evaluation.schema_linking_gold.source import AnnotationTask
from scripts.rc_evaluation.schema_linking_gold.store import AnnotationStore


_ENDPOINT_HASH = "f" * 64


def _task(
    key: str,
    *,
    group: str,
    feature: str,
    native: dict[str, tuple[str, ...]],
    rc3: dict[str, tuple[str, ...]],
    parser: dict,
    number: int,
) -> AnnotationTask:
    table_names = {"orders", "customers", "users", "noise"}
    schema = {
        "tables": tuple(
            {
                "id": f"T{index}",
                "name": table,
                "columns": ({"id": f"C{index}", "name": "id", "type": "INTEGER"},),
            }
            for index, table in enumerate(sorted(table_names), start=1)
        ),
        "table_by_id": {},
        "table_id_by_name": {},
        "column_by_id": {},
        "column_id_by_name": {},
    }
    return AnnotationTask(
        task_key=key,
        group=group,
        partition="dev",
        external_id=number,
        database_id=f"db{number}",
        dialect="sqlite",
        gold_sql="SELECT 1",
        sql_sha256=f"{number:x}" * 64,
        schema=schema,
        schema_sha256=f"{number + 5:x}" * 64,
        source_schema_sha256=f"{number + 10:x}" * 64,
        source_hash=f"{number + 1:x}" * 64,
        reuse_key=("sqlite", f"{number + 5:x}" * 64, f"{number:x}" * 64),
        features=(feature,),
        native_linked_schema=native,
        rc_linked_schema=rc3,
        conservative_reference=parser,
        source_reference={},
    )


def _annotation(
    key: str,
    status: str,
    tables: list[str],
    columns: list[list[str]],
    *,
    reason: str | None = None,
) -> dict:
    return {
        "task_key": key,
        "status": status,
        "required_table_ids": [],
        "required_column_ids": [],
        "required_tables": tables,
        "required_columns": columns,
        "evidence": [],
        "json_paths": [],
        "review_reasons": [] if reason is None else [reason],
    }


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store_path = self.root / "pilot.sqlite3"
        available = lambda tables, columns: {
            "status": "available", "reason": None, "tables": tables, "columns": columns,
        }
        unavailable = {"status": "unavailable", "reason": "parser rejected", "tables": (), "columns": ()}
        self.tasks = [
            _task(
                "g1/t1", group="g1", feature="wildcard", number=1,
                native={"orders": ()}, rc3={"orders": (), "customers": ("id",)},
                parser=available(("orders",), ()),
            ),
            _task(
                "g1/t2", group="g1", feature="ordinary", number=2,
                native={"orders": (), "noise": ()}, rc3={"orders": ("id",)},
                parser=available(("orders",), (("orders", "other"),)),
            ),
            _task(
                "g2/t3", group="g2", feature="json", number=3,
                native={"users": ("id",)}, rc3={"users": ("id",)}, parser=unavailable,
            ),
            _task(
                "g1/t4", group="g1", feature="ordinary", number=4,
                native={"orders": ("id",)}, rc3={"orders": ("id",)},
                parser=available(("orders",), (("orders", "id"),)),
            ),
            _task(
                "g2/t5", group="g2", feature="set_operation", number=5,
                native={"users": ("id",)}, rc3={"users": ("id",)},
                parser=available(("users",), (("users", "id"),)),
            ),
        ]
        self.pilot_keys = [task.task_key for task in self.tasks]
        self.tasks.append(replace(self.tasks[0], task_key="g1/t6", external_id=6))
        manifest = {
            "kind": "gold_sql_schema_linking",
            "model": "test-model",
            "prompt_version": "gold-sql-schema-linking-v1",
            "selection": "pilot",
            "selection_size": len(self.tasks),
            "pilot_size": len(self.pilot_keys),
            "pilot_seed": 17,
            "pilot_task_keys": self.pilot_keys,
            "tasks": [
                {
                    "task_key": task.task_key,
                    "group": task.group,
                    "dialect": task.dialect,
                    "schema_sha256": task.schema_sha256,
                    "sql_sha256": task.sql_sha256,
                    "source_hash": task.source_hash,
                    "source_schema_sha256": task.source_schema_sha256,
                    "features": list(task.features),
                }
                for task in self.tasks
            ],
        }
        annotations = {
            "g1/t1": _annotation("g1/t1", "resolved", ["orders"], []),
            "g1/t2": _annotation("g1/t2", "resolved", ["orders"], [["orders", "id"]]),
            "g2/t3": _annotation("g2/t3", "resolved", ["users"], [["users", "id"]]),
            "g1/t4": _annotation("g1/t4", "needs_review", [], [], reason="ambiguous"),
            "g2/t5": _annotation("g2/t5", "invalid_sql", [], [], reason="invalid reference"),
        }
        keys = sorted(annotations)
        with AnnotationStore.create(self.store_path, manifest) as store:
            failed = store.start_attempt(keys)
            store.finish_attempt(
                failed["attempt_id"], "failed", raw_response="not json",
                usage={"available": False},
                latency_seconds=1.5, endpoint_hash=_ENDPOINT_HASH,
                error={"kind": "parse_failure", "type": "ValueError"},
            )
            succeeded = store.start_attempt(keys)
            store.finish_attempt(
                succeeded["attempt_id"], "succeeded", raw_response="[]",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                latency_seconds=2.5, endpoint_hash=_ENDPOINT_HASH,
            )
            store.accept_annotations(succeeded["attempt_id"], annotations)

    def tearDown(self):
        self.temporary.cleanup()

    def _export(self, name: str) -> tuple[dict, Path]:
        output = self.root / name
        result = export_annotations(self.store_path, output, tasks=self.tasks)
        report = json.loads((output / "pilot_report.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "success")
        return report, output

    def test_empty_gold_micro_macro_exact_and_full_recall_metrics_are_hand_calculated(self):
        """Catches treating an empty gold-column set as one recalled column or as macro recall zero."""
        report, _ = self._export("metrics")

        native_tables = report["metrics"]["native"]["tables"]
        self.assertEqual({key: native_tables[key] for key in ("questions", "tp", "fp", "fn")},
                         {"questions": 3, "tp": 3, "fp": 1, "fn": 0})
        self.assertEqual(native_tables["micro"], {
            "precision": 0.75, "recall": 1.0, "f1": 6 / 7,
        })
        self.assertEqual(native_tables["macro"], {
            "precision": 5 / 6, "precision_questions": 3,
            "recall": 1.0, "recall_questions": 3,
            "f1": 8 / 9, "f1_questions": 3,
        })
        self.assertEqual(native_tables["exact_set"], {"count": 2, "ratio": 2 / 3})
        self.assertEqual(native_tables["full_recall"], {"count": 3, "eligible": 3, "ratio": 1.0})

        native_columns = report["metrics"]["native"]["columns"]
        self.assertEqual({key: native_columns[key] for key in ("tp", "fp", "fn")}, {"tp": 1, "fp": 0, "fn": 1})
        self.assertEqual(native_columns["micro"], {"precision": 1.0, "recall": 0.5, "f1": 2 / 3})
        self.assertEqual(native_columns["macro"], {
            "precision": 0.5, "precision_questions": 2,
            "recall": 0.5, "recall_questions": 2,
            "f1": 0.5, "f1_questions": 2,
        })
        self.assertEqual(native_columns["exact_set"], {"count": 2, "ratio": 2 / 3})
        self.assertEqual(native_columns["full_recall"], {"count": 1, "eligible": 2, "ratio": 0.5})

        rc3_columns = report["metrics"]["rc3"]["columns"]
        self.assertEqual({key: rc3_columns[key] for key in ("tp", "fp", "fn")}, {"tp": 2, "fp": 1, "fn": 0})
        self.assertEqual(rc3_columns["micro"], {"precision": 2 / 3, "recall": 1.0, "f1": 0.8})
        self.assertEqual(rc3_columns["macro"], {
            "precision": 2 / 3, "precision_questions": 3,
            "recall": 1.0, "recall_questions": 2,
            "f1": 2 / 3, "f1_questions": 3,
        })

    def test_paired_directions_parser_agreement_and_unresolved_exclusions_are_explicit(self):
        """Catches dropping wrong-to-less-wrong pairs, parser disagreements, or unresolved labels."""
        report, _ = self._export("quality")

        direction = {"basis": "symmetric_difference_errors", "questions": 3,
                     "improvements": 1, "regressions": 1, "unchanged": 1}
        self.assertEqual(report["paired_native_to_rc3"]["tables"], direction)
        self.assertEqual(report["paired_native_to_rc3"]["columns"], direction)
        agreement = report["parser_agreement"]
        self.assertEqual(agreement["eligible"], 2)
        self.assertEqual(agreement["table_exact"], {"count": 2, "ratio": 1.0})
        self.assertEqual(agreement["column_exact"], {"count": 1, "ratio": 0.5})
        self.assertEqual(agreement["both_exact"], {"count": 1, "ratio": 0.5})
        self.assertEqual([row["task_key"] for row in agreement["disagreements"]], ["g1/t2"])
        self.assertEqual(agreement["excluded"], {"annotation_unresolved": 2, "parser_unavailable": 1})
        self.assertEqual(
            [(row["task_key"], row["status"]) for row in report["excluded_annotations"]],
            [("g1/t4", "needs_review"), ("g2/t5", "invalid_sql")],
        )
        self.assertEqual(report["status_counts"]["overall"], {
            "resolved": 3, "needs_review": 1, "invalid_sql": 1, "pending": 0,
        })
        self.assertEqual(report["status_counts"]["by_feature"]["ordinary"], {
            "resolved": 1, "needs_review": 1, "invalid_sql": 0, "pending": 0,
        })

    def test_request_accounting_and_exports_are_deterministic_and_exclusive_new(self):
        """Catches unstable row order, overwritten reports, or omission of failed model output and usage."""
        first_report, first = self._export("first")
        second_report, second = self._export("second")

        self.assertEqual(first_report, second_report)
        for name in ("annotations.jsonl", "pilot_report.json", "pilot_report.md"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        rows = [json.loads(line) for line in (first / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["task_key"] for row in rows], sorted(self.pilot_keys))
        self.assertEqual(first_report["coverage"], {
            "total_tasks": 5, "accepted_tasks": 5, "resolved_tasks": 3,
            "needs_review_tasks": 1, "invalid_sql_tasks": 1, "pending_tasks": 0,
            "accepted_ratio": 1.0, "estimated_full_run_coverage": 0.6,
            "cached_unique_input_count": 5, "accepted_unique_input_count": 5,
            "manifest_total_tasks": 6, "store_accepted_tasks": 6,
        })
        self.assertEqual(first_report["request_accounting"], {
            "attempts": 2, "finished": 2, "unfinished": 0, "succeeded": 1,
            "failed": 1, "retries": 1,
            "latency_seconds": {"total": 4.0, "mean": 2.0, "maximum": 2.5},
            "tokens": {
                "prompt": 10, "completion": 5, "total": 15,
                "known_total_attempts": 1, "unknown_total_attempts": 1,
            },
        })
        self.assertEqual(len(first_report["rejected_outputs"]), 1)
        self.assertEqual(first_report["rejected_outputs"][0]["raw_response"], "not json")
        markdown = (first / "pilot_report.md").read_text(encoding="utf-8")
        self.assertIn("g1/t2", markdown)
        self.assertIn("not json", markdown)
        self.assertIn("unknown-token attempts: 1", markdown)

        with self.assertRaises(FileExistsError):
            export_annotations(self.store_path, first, tasks=self.tasks)

    def test_scope_selects_exact_pilot_keys_or_all_manifest_tasks_and_names_full_outputs(self):
        """Catches cache-rebound non-pilot peers entering pilot quality metrics or colliding with full exports."""
        pilot_report, pilot = self._export("scoped-pilot")
        full = self.root / "scoped-full"

        result = export_annotations(self.store_path, full, tasks=self.tasks, scope="full")
        full_report = json.loads((full / "full_report.json").read_text(encoding="utf-8"))
        full_rows = [
            json.loads(line)
            for line in (full / "annotations.full.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        self.assertEqual(pilot_report["scope"], "pilot")
        self.assertEqual(pilot_report["coverage"]["total_tasks"], 5)
        self.assertEqual(
            [json.loads(line)["task_key"] for line in
             (pilot / "annotations.jsonl").read_text(encoding="utf-8").splitlines()],
            sorted(self.pilot_keys),
        )
        self.assertEqual(result["files"], ["annotations.full.jsonl", "full_report.json", "full_report.md"])
        self.assertEqual(full_report["scope"], "full")
        self.assertEqual(full_report["coverage"]["total_tasks"], 6)
        self.assertEqual([row["task_key"] for row in full_rows], sorted(task.task_key for task in self.tasks))

    def test_export_can_add_new_artifacts_to_an_existing_pilot_directory(self):
        """Catches treating the output directory itself as the immutable artifact path."""
        output = self.root / "existing-pilot"
        output.mkdir()
        (output / "pilot.sqlite3").write_bytes(b"separate existing store placeholder")

        result = export_annotations(self.store_path, output, tasks=self.tasks)

        self.assertEqual(result["status"], "success")
        self.assertTrue((output / "annotations.jsonl").is_file())
        self.assertEqual((output / "pilot.sqlite3").read_bytes(), b"separate existing store placeholder")
        with self.assertRaises(FileExistsError):
            export_annotations(self.store_path, output, tasks=self.tasks)


if __name__ == "__main__":
    unittest.main()
