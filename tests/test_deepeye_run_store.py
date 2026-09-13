from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid
from decimal import Decimal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/DeepEye-SQL"))

from scripts.baseline_adapters.deepeye.run_store import (
    RunStore,
    restore_jsonable,
    to_jsonable,
)
from scripts.baseline_adapters.deepeye.run_usage import observed_usage


class Colour(enum.Enum):
    RED = "red"


@dataclasses.dataclass(frozen=True)
class ExampleRecord:
    path: Path
    amount: Decimal


class ExampleModel(BaseModel):
    name: str
    count: int


class RunStoreTests(unittest.TestCase):
    def test_create_records_manifest_and_refuses_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            manifest = {"model": "offline", "seed": 7}
            with RunStore.create(run_dir, manifest) as store:
                self.assertEqual(store.run_dir, run_dir)
                self.assertEqual(store.manifest, manifest)
            with self.assertRaises(FileExistsError):
                RunStore.create(run_dir, manifest)

    def test_invalid_manifest_is_rejected_before_creating_a_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with self.assertRaises(TypeError):
                RunStore.create(run_dir, ["not", "a", "mapping"])  # type: ignore[arg-type]
            self.assertFalse(run_dir.exists())

    def test_attempts_are_append_only_and_completion_is_fingerprint_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {"seed": 1}) as store:
                first = store.begin_attempt("db/q1", "generate", "input-v1")
                self.assertEqual(store.append_event(first, "prompt", {"tokens": 9}), 1)
                store.finish_attempt(first, "failed", {"error": "bad output"})
                second = store.begin_attempt("db/q1", "generate", "input-v1")
                store.finish_attempt(second, "succeeded", {"sql": "SELECT 1", "acc": 1})
                self.assertEqual(
                    store.completed("db/q1", "generate", "input-v1"),
                    {
                        "attempt_id": second,
                        "attempt_no": 2,
                        "payload": {"sql": "SELECT 1", "acc": 1},
                    },
                )
                third = store.begin_attempt("db/q1", "generate", "input-v2")
                store.finish_attempt(third, "failed", {"error": "different input"})

                with self.assertRaises(ValueError):
                    store.completed("db/q1", "generate", "unseen")
                self.assertEqual([row["attempt_no"] for row in store.attempts()], [1, 2, 3])
                self.assertEqual(store.events(first)[0]["payload"], {"tokens": 9})

    def test_completed_returns_none_when_no_successful_lineage_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                attempt = store.begin_attempt("q", "stage", "fp-old")
                store.finish_attempt(attempt, "failed", {})
                self.assertIsNone(store.completed("q", "stage", "fp-new"))

    def test_duplicate_finish_and_writes_after_terminal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                attempt_id = store.begin_attempt("q", "stage", "fp")
                store.finish_attempt(attempt_id, "succeeded", {"ok": True})
                with self.assertRaises(ValueError):
                    store.finish_attempt(attempt_id, "failed", {})
                with self.assertRaises(ValueError):
                    store.append_event(attempt_id, "late", {})
                with self.assertRaises(ValueError):
                    store.finish_attempt("missing", "failed", {})
                with self.assertRaises(ValueError):
                    store.finish_attempt(attempt_id, "unknown", {})

    def test_concurrent_event_appends_are_committed_and_numbered(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                attempt_id = store.begin_attempt("q", "stage", "fp")
                errors: list[BaseException] = []

                def append(number: int) -> None:
                    try:
                        store.append_event(attempt_id, "progress", {"number": number})
                    except BaseException as exc:  # pragma: no cover - asserted below
                        errors.append(exc)

                threads = [threading.Thread(target=append, args=(number,)) for number in range(64)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.assertEqual(errors, [])
                events = store.events(attempt_id)
                self.assertEqual([event["event_no"] for event in events], list(range(1, 65)))
                self.assertEqual({event["payload"]["number"] for event in events}, set(range(64)))

    def test_iter_events_streams_verified_rows_with_attempt_and_kind_filters(self):
        with tempfile.TemporaryDirectory() as temp:
            with RunStore.create(Path(temp) / "run", {}) as store:
                first = store.begin_attempt("one", "stage", "fp")
                second = store.begin_attempt("two", "stage", "fp")
                store.append_event(first, "api_request", {"call_id": "a"})
                store.append_event(first, "sql_execute", {"sql": "SELECT 1"})
                store.append_event(second, "api_response", {"call_id": "b"})

                stream = store.iter_events(first, kinds={"api_request", "api_response"})
                self.assertFalse(isinstance(stream, list))
                self.assertEqual(
                    [(event["attempt_id"], event["kind"]) for event in stream],
                    [(first, "api_request")],
                )
                self.assertEqual(store.events(second)[0]["kind"], "api_response")

    def test_manifest_mismatch_is_refused_without_changing_database(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {"seed": 1}):
                pass
            database = run_dir / "run.sqlite3"
            before = database.read_bytes()
            with self.assertRaises(ValueError):
                RunStore.open(run_dir, expected_manifest={"seed": 2})
            self.assertEqual(database.read_bytes(), before)

    def test_only_one_writer_opens_but_read_only_diagnostics_remain_available(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {"seed": 1}) as writer:
                attempt_id = writer.begin_attempt("q", "stage", "fp")
                writer.append_event(attempt_id, "started", {})
                command = (
                    "from pathlib import Path; "
                    "from scripts.baseline_adapters.deepeye.run_store import RunStore; "
                    f"RunStore.open(Path({str(run_dir)!r}))"
                )
                started = time.monotonic()
                result = subprocess.run(
                    [sys.executable, "-c", command],
                    cwd=Path(__file__).resolve().parents[1],
                    env=self._subprocess_env(),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertLess(time.monotonic() - started, 4)
                with RunStore.open(run_dir, read_only=True) as reader:
                    self.assertEqual(len(reader.events()), 1)
                    with self.assertRaises(PermissionError):
                        reader.begin_attempt("other", "stage", "fp")

    def test_reopen_reports_unfinished_attempt_as_interrupted_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {}) as store:
                attempt_id = store.begin_attempt("q", "stage", "fp")
            before = (run_dir / "run.sqlite3").read_bytes()
            with RunStore.open(run_dir, read_only=True) as store:
                attempt = store.attempts()[0]
                self.assertEqual(attempt["attempt_id"], attempt_id)
                self.assertEqual(attempt["status"], "interrupted")
                self.assertIsNone(attempt["finished_at"])
            self.assertEqual((run_dir / "run.sqlite3").read_bytes(), before)

    def test_subprocess_crash_preserves_commits_and_discards_open_transaction(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {}):
                pass
            script = r'''from pathlib import Path
from scripts.baseline_adapters.deepeye.run_store import RunStore
import sys
store = RunStore.open(Path(sys.argv[1]))
done = store.begin_attempt("done", "stage", "fp")
store.finish_attempt(done, "succeeded", {"value": 1})
store.begin_attempt("open", "stage", "fp")
store._connection.execute("BEGIN IMMEDIATE")
store._connection.execute("INSERT INTO events(attempt_id, event_no, kind, payload_json, payload_checksum, created_at, record_checksum) VALUES (?, ?, ?, ?, ?, ?, ?)", (done, 99, "uncommitted", "{}", "bad", "now", "bad"))
print("ready", flush=True)
sys.stdin.read()
'''
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(run_dir)],
                cwd=Path(__file__).resolve().parents[1],
                env=self._subprocess_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(process.stdout.readline().strip(), "ready")
            process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

            with RunStore.open(run_dir) as store:
                self.assertEqual(store.completed("done", "stage", "fp")["payload"], {"value": 1})
                self.assertEqual(store.events(), [])
                open_attempt = next(row for row in store.attempts() if row["item_key"] == "open")
                self.assertEqual(open_attempt["status"], "interrupted")
                retry = store.begin_attempt("open", "stage", "fp")
                retry_row = next(row for row in store.attempts() if row["attempt_id"] == retry)
                self.assertEqual(retry_row["attempt_no"], 2)

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        environment = os.environ.copy()
        baseline = Path(__file__).resolve().parents[1] / "baselines/DeepEye-SQL"
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(baseline) + (os.pathsep + existing if existing else "")
        return environment

    def test_checksums_detect_tampering_and_prevent_success_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {}) as store:
                attempt_id = store.begin_attempt("q", "stage", "fp")
                store.finish_attempt(attempt_id, "succeeded", {"answer": 1})
            with sqlite3.connect(run_dir / "run.sqlite3") as connection:
                connection.execute("DROP TRIGGER finishes_no_update")
                connection.execute(
                    "UPDATE finishes SET payload_json = ? WHERE attempt_id = ?",
                    ('{"answer":2}', attempt_id),
                )
            with RunStore.open(run_dir, read_only=True) as store:
                with self.assertRaises(ValueError):
                    store.completed("q", "stage", "fp")
                report = store.verify()
                self.assertFalse(report["ok"])
                self.assertGreaterEqual(report["checksum_errors"], 1)

    def test_sqlite_records_cannot_be_updated_or_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {}) as store:
                store.begin_attempt("q", "stage", "fp")
            with sqlite3.connect(run_dir / "run.sqlite3") as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("UPDATE attempts SET stage = 'other'")
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("DELETE FROM attempts")

    def test_export_is_fresh_human_readable_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with RunStore.create(root / "run", {"seed": 3}) as store:
                attempt_id = store.begin_attempt("q", "stage", "fp")
                store.append_event(attempt_id, "metric", {"accuracy": Decimal("0.5")})
                store.finish_attempt(attempt_id, "succeeded", {"sql": "SELECT 1"})
                export_dir = store.export(root / "export")
                self.assertEqual(export_dir, root / "export")
                self.assertEqual(json.loads((export_dir / "manifest.json").read_text())["seed"], 3)
                attempts = [json.loads(line) for line in (export_dir / "attempts.jsonl").read_text().splitlines()]
                self.assertEqual(attempts[0]["status"], "succeeded")
                self.assertTrue(json.loads((export_dir / "verification.json").read_text())["ok"])
                self.assertTrue(json.loads((export_dir / "COMPLETE.json").read_text())["complete"])
                with self.assertRaises(FileExistsError):
                    store.export(export_dir)

    def test_read_only_export_uses_one_snapshot_while_writer_appends(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            with RunStore.create(run_dir, {}) as writer:
                first = writer.begin_attempt("one", "stage", "fp")
                writer.append_event(first, "api_request", {"call_id": "first"})
                with RunStore.open(run_dir, read_only=True) as reader:
                    paused = threading.Event()
                    resume = threading.Event()
                    errors: list[BaseException] = []
                    original_write = RunStore._write_json

                    def pause_after_snapshot(path: Path, value: object) -> None:
                        original_write(path, value)
                        if path.name == "manifest.json":
                            paused.set()
                            if not resume.wait(5):
                                raise TimeoutError("writer did not resume export")

                    def export() -> None:
                        try:
                            reader.export(
                                root / "export",
                                extra_reports={"observed_usage.json": observed_usage},
                            )
                        except BaseException as exc:  # pragma: no cover - asserted below
                            errors.append(exc)

                    with patch.object(RunStore, "_write_json", side_effect=pause_after_snapshot):
                        thread = threading.Thread(target=export)
                        thread.start()
                        self.assertTrue(paused.wait(5))
                        second = writer.begin_attempt("two", "stage", "fp")
                        writer.append_event(second, "api_request", {"call_id": "second"})
                        resume.set()
                        thread.join(5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])

            attempts = [json.loads(line) for line in (root / "export/attempts.jsonl").read_text().splitlines()]
            events = [json.loads(line) for line in (root / "export/events.jsonl").read_text().splitlines()]
            self.assertEqual([row["attempt_id"] for row in attempts], [first])
            self.assertEqual([row["attempt_id"] for row in events], [first])
            usage = json.loads((root / "export/observed_usage.json").read_text())
            self.assertEqual(usage["requests"], 1)
            self.assertEqual(usage["unanswered_requests"], 1)
            self.assertTrue((root / "export/COMPLETE.json").is_file())

    def test_failed_export_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with RunStore.create(root / "run", {}) as store:
                original_write = RunStore._write_json

                def fail_summary(path: Path, value: object) -> None:
                    if path.name == "summary.json":
                        raise OSError("disk full")
                    original_write(path, value)

                with patch.object(RunStore, "_write_json", side_effect=fail_summary):
                    with self.assertRaises(OSError):
                        store.export(root / "export")
            self.assertFalse((root / "export/COMPLETE.json").exists())

    def test_extra_report_failure_leaves_export_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with RunStore.create(root / "run", {}) as store:
                def fail_report(_: RunStore) -> object:
                    raise OSError("usage calculation failed")

                with self.assertRaises(OSError):
                    store.export(
                        root / "export",
                        extra_reports={"observed_usage.json": fail_report},
                    )
            self.assertFalse((root / "export/COMPLETE.json").exists())

    def test_extra_report_names_are_validated_before_export_directory_creation(self):
        invalid_names = (
            "manifest.json",
            "COMPLETE.json",
            "../escape.json",
            "nested/report.json",
            "nested\\report.json",
            "report.txt",
            "",
        )
        for name in invalid_names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with RunStore.create(root / "run", {}) as store:
                    with self.assertRaises((TypeError, ValueError)):
                        store.export(
                            root / "export",
                            extra_reports={name: lambda _: {}},
                        )
                self.assertFalse((root / "export").exists())

    def test_summary_counts_attempt_statuses_and_events(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            with RunStore.create(run_dir, {}) as store:
                success = store.begin_attempt("one", "stage", "a")
                store.append_event(success, "metric", {})
                store.finish_attempt(success, "succeeded", {})
                failed = store.begin_attempt("two", "stage", "b")
                store.finish_attempt(failed, "failed", {})
                store.begin_attempt("three", "stage", "c")
            with RunStore.open(run_dir, read_only=True) as store:
                self.assertEqual(
                    store.summary(),
                    {"attempts": 3, "succeeded": 1, "failed": 1, "interrupted": 1, "events": 1},
                )


class JsonableTests(unittest.TestCase):
    def test_supported_nested_values_round_trip_losslessly(self):
        value = {
            "model": ExampleModel(name="test", count=2),
            "record": ExampleRecord(Path("data/file.json"), Decimal("1.2300")),
            "tuple": (dt.date(2026, 9, 11), dt.time(3, 4, 5, 6)),
            "set": {uuid.UUID("12345678-1234-5678-1234-567812345678"), Colour.RED},
            "datetime": dt.datetime(2026, 9, 11, 3, 4, tzinfo=dt.timezone.utc),
            "bytes": b"\x00\xff",
            "decimal_nan": Decimal("NaN"),
            "non_string_dict": {1: Path("one")},
        }
        restored = restore_jsonable(to_jsonable(value))
        self.assertEqual(restored["model"], value["model"])
        self.assertEqual(restored["record"], value["record"])
        self.assertEqual(restored["tuple"], value["tuple"])
        self.assertEqual(restored["set"], value["set"])
        self.assertEqual(restored["datetime"], value["datetime"])
        self.assertEqual(restored["bytes"], value["bytes"])
        self.assertTrue(restored["decimal_nan"].is_nan())
        self.assertEqual(restored["non_string_dict"], value["non_string_dict"])

    def test_non_finite_floats_use_valid_tagged_json_and_round_trip(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                encoded = to_jsonable(value)
                json.dumps(encoded, allow_nan=False)
                restored = restore_jsonable(encoded)
                if math.isnan(value):
                    self.assertTrue(math.isnan(restored))
                else:
                    self.assertEqual(restored, value)

    def test_unknown_objects_are_rejected(self):
        with self.assertRaises(TypeError):
            to_jsonable(object())


if __name__ == "__main__":
    unittest.main()
