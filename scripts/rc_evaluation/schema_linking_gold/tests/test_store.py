"""Behavioral tests for the durable gold-SQL annotation store."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.rc_evaluation.schema_linking_gold.store import AnnotationStore


def _manifest(*, second_schema: bool = False) -> dict:
    """A hand-checked frozen pilot with one reusable input and one distinct one."""
    shared_schema = "a" * 64
    return {
        "format": "schema-linking-gold-annotation-v1",
        "model": "test-model",
        "prompt_version": "v1",
        "sampling_seed": 17,
        "tasks": [
            {"task_key": "spider/dev/i:0", "group": "spider_dev", "dialect": "sqlite",
             "schema_sha256": shared_schema, "sql_sha256": "b" * 64},
            {"task_key": "spider/test/i:1", "group": "spider_test", "dialect": "sqlite",
             "schema_sha256": ("c" * 64 if second_schema else shared_schema), "sql_sha256": "b" * 64},
            {"task_key": "bird/dev/i:2", "group": "bird_dev", "dialect": "postgresql",
             "schema_sha256": "d" * 64, "sql_sha256": "e" * 64},
        ],
    }


def _annotation(task_key: str, status: str = "resolved") -> dict:
    return {
        "task_key": task_key,
        "status": status,
        "required_table_ids": ["T1"],
        "required_column_ids": ["C1"],
        "required_tables": ["orders"],
        "required_columns": [["orders", "id"]],
        "evidence": ["orders.id is selected"],
        "json_paths": [],
        "review_reasons": [],
    }


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "pilot.sqlite3"
        self.manifest = _manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def open(self):
        return AnnotationStore.create(self.path, self.manifest)

    def test_manifest_is_frozen_and_rejects_a_different_resume(self):
        """Catches a resume path that silently changes the selected pilot or model."""
        with self.open() as store:
            self.assertEqual(store.manifest, self.manifest)
        altered = _manifest()
        altered["model"] = "other-model"
        with self.assertRaisesRegex(ValueError, "manifest"):
            AnnotationStore.create(self.path, altered)
        connection = sqlite3.connect(self.path)
        with self.assertRaisesRegex(sqlite3.DatabaseError, "immutable"):
            connection.execute("UPDATE manifest SET payload_json = '{}' WHERE singleton = 1")
        connection.close()

    def test_wal_mode_persists_a_started_attempt_before_any_request(self):
        """Catches an attempt recorder that is not durable before the remote send."""
        with self.open() as store:
            started = store.start_attempt(["spider/dev/i:0"])
            self.assertEqual(started["attempt_no"], 1)
        connection = sqlite3.connect(self.path)
        self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        self.assertEqual(connection.execute("SELECT attempt_id FROM attempts").fetchone()[0], started["attempt_id"])
        self.assertIsNone(connection.execute("SELECT attempt_id FROM outcomes").fetchone())
        connection.close()

    def test_results_require_a_finished_successful_attempt(self):
        """Catches accepting a label that has no durable successful request outcome."""
        with self.open() as store:
            started = store.start_attempt(["spider/dev/i:0"])
            with self.assertRaisesRegex(ValueError, "finished.*successful"):
                store.accept_annotations(started["attempt_id"], {"spider/dev/i:0": _annotation("spider/dev/i:0")})
            store.finish_attempt(started["attempt_id"], "succeeded", raw_response="[]",
                                 usage={"total_tokens": 9}, latency_seconds=0.4,
                                 endpoint_hash="endpoint-digest")
            store.accept_annotations(started["attempt_id"], {"spider/dev/i:0": _annotation("spider/dev/i:0")})
            self.assertEqual(store.status()["accepted_tasks"], 2)

    def test_resume_retries_an_unfinished_attempt_without_losing_it(self):
        """Catches a crash/resume path that drops an in-flight attempt or blocks its retry."""
        with self.open() as store:
            first = store.start_attempt(["bird/dev/i:2"])
        with AnnotationStore.open(self.path, self.manifest) as resumed:
            self.assertEqual([task["task_key"] for task in resumed.pending_tasks()], ["bird/dev/i:2", "spider/dev/i:0"])
            second = resumed.start_attempt(["bird/dev/i:2"])
            self.assertEqual(second["attempt_no"], first["attempt_no"] + 1)
            self.assertEqual(resumed.status()["unfinished_attempts"], 2)

    def test_accepted_label_cannot_be_replaced_in_place(self):
        """Catches a later model call overwriting the frozen accepted annotation."""
        with self.open() as store:
            first = store.start_attempt(["spider/dev/i:0"])
            store.finish_attempt(first["attempt_id"], "succeeded", raw_response="[]")
            store.accept_annotations(first["attempt_id"], {"spider/dev/i:0": _annotation("spider/dev/i:0")})
            self.assertEqual(store.pending_tasks()[0]["task_key"], "bird/dev/i:2")
            with self.assertRaisesRegex(ValueError, "accepted"):
                store.start_attempt(["spider/test/i:1"])
            self.assertEqual(store.annotation_for_task("spider/test/i:1")["required_table_ids"], ["T1"])

    def test_cache_key_includes_schema_and_reuses_only_identical_input(self):
        """Catches reusing a label across schemas or re-requesting an exact duplicate."""
        with self.open() as store:
            first = store.start_attempt(["spider/dev/i:0"])
            store.finish_attempt(first["attempt_id"], "succeeded", raw_response="[]")
            store.accept_annotations(first["attempt_id"], {"spider/dev/i:0": _annotation("spider/dev/i:0")})
            self.assertEqual(store.annotation_for_task("spider/test/i:1")["task_key"], "spider/test/i:1")
            self.assertEqual([task["task_key"] for task in store.pending_tasks()], ["bird/dev/i:2"])
        other_path = Path(self.temporary.name) / "other.sqlite3"
        other_manifest = _manifest(second_schema=True)
        with AnnotationStore.create(other_path, other_manifest) as other:
            self.assertEqual(
                [task["task_key"] for task in other.pending_tasks()],
                ["bird/dev/i:2", "spider/dev/i:0", "spider/test/i:1"],
            )

    def test_verify_reports_checksum_corruption(self):
        """Catches a verifier that trusts a tampered immutable history row."""
        with self.open() as store:
            started = store.start_attempt(["bird/dev/i:2"])
            store.finish_attempt(started["attempt_id"], "failed", error={"kind": "timeout"})
            self.assertTrue(store.verify()["ok"])
        connection = sqlite3.connect(self.path)
        connection.execute("DROP TRIGGER outcomes_no_update")
        connection.execute("UPDATE outcomes SET raw_response = 'tampered' WHERE attempt_id = ?", (started["attempt_id"],))
        connection.commit()
        connection.close()
        with AnnotationStore.open(self.path, self.manifest) as reopened:
            report = reopened.verify()
        self.assertFalse(report["ok"])
        self.assertIn("checksum mismatch", "\n".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
