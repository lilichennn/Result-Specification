import contextlib
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zlib

from scripts.baseline_adapters.deepeye.run_store import restore_jsonable
from scripts.baseline_adapters.din_sql.inputs import DinSettings, NODES, PreparedInputs, TaskKey
from scripts.baseline_adapters.din_sql.records import DinRecords, hydrate_batch
from tests.din_sql_fixtures import make_task, terminal


def lines(path):
    return [restore_jsonable(json.loads(line)) for line in Path(path).read_text().splitlines()]


class DinCompactReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_batch(self, groups=("bird_dev",), ids=("0", "1")):
        batch = self.root / "batch"
        tasks = {}
        evaluation = {}
        databases = {}
        for group in groups:
            database = self.root / f"{group}.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE numbers(value INTEGER)")
                connection.execute("INSERT INTO numbers VALUES (1)")
            databases[group] = database
            for question_id in ids:
                task = make_task(group=group, question_id=question_id)
                task = type(task)(task.key, task.question, task.evidence,
                                  {"dialect": "sqlite", "database_id": group, "path": str(database)},
                                  task.schema_ref, task.rc3, task.label, task.source_refs)
                tasks[task.key] = task
                evaluation[f"{group}/{question_id}"] = {
                    "database": task.database,
                    "gold_sql": "SELECT value FROM numbers",
                }
        settings = asdict(DinSettings(sql_workers=20, sql_timeout_seconds=180))
        manifest = {"format": "din-sql-v1", "batch_id": "fixture", "settings": settings,
                    "groups": {group: {"ids": list(ids)} for group in groups}}
        prepared = PreparedInputs(tasks, {}, evaluation, {}, {}, {})
        with DinRecords(batch, manifest) as records:
            hydrate_batch(batch, prepared, records)
        return batch, manifest, databases

    def finish(self, batch, manifest, key, *, status="succeeded", sqls=None, requests=False):
        sqls = sqls or {}
        with DinRecords(batch, manifest) as records:
            version = records.unfinished(key) or records.begin(key)
            if requests:
                records.append(version, "node_input", {"node": "generation_base",
                    "input_fingerprint": "prompt", "kwargs": {"messages": ["SECRET-PROMPT" * 2000]}})
                identity = {"node": "generation_base", "request_id": "request-1", "attempt_no": 1,
                            "batch_id": "fixture", "group": key.group, "question_id": key.question_id,
                            "round_execution_id": version + ":generation_base", "sample_position": 0}
                records.append(version, "attempt_queued", identity)
                records.append(version, "request_attempt", identity)
                records.append(version, "request_dispatch", {**identity, "telemetry": {"queue": 0.2}})
                response_ref = records.append(version, "request_result", {**identity,
                    "body": {"provider": "SECRET-PROVIDER" * 2000,
                             "choices": [{"message": {"content": "SECRET-CONTENT"}}]}})
                records.append(version, "request_outcome", {"node": "generation_base",
                    "status": "failed", "response_ref": response_ref, "usage": {"total_tokens": 7},
                    "request_id": "request-1", "attempt_no": 1,
                    "error": {"category": "http", "type": "fixture"}})
                records.append(version, "attempt_queued", {**identity, "request_id": "queued-only", "attempt_no": 2})
            for node in NODES:
                value = terminal(node, status if node.startswith(("generation", "revision")) else "succeeded")
                if value["status"] == "succeeded" and node in sqls:
                    value["result"] = sqls[node]
                value["usage"] = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
                records.save_node(version, node, value)
            records.seal(version)
            return version

    def test_record_export_freezes_latest_failed_current_and_omits_large_request_bodies(self):
        from scripts.rc_evaluation.din_sql.compact_reporting import export_records

        batch, manifest, _ = self.make_batch(ids=("0", "1", "missing"))
        key = TaskKey("bird_dev", "0")
        old = self.finish(batch, manifest, key)
        current = self.finish(batch, manifest, key, status="failed", requests=True)
        other = self.finish(batch, manifest, TaskKey("bird_dev", "1"))
        output = self.root / "records"
        result = export_records(batch, output)

        versions = {row["question_id"]: row for row in lines(output / "versions.jsonl")}
        self.assertEqual(versions["0"]["version_id"], current)
        self.assertNotEqual(versions["0"]["version_id"], old)
        self.assertEqual(versions["1"]["version_id"], other)
        # Hydration creates an unfinished attempt for every prepared question;
        # it must remain explicit and must not displace a sealed current one.
        self.assertEqual(versions["missing"]["state"], "pending")
        self.assertIsNotNone(versions["missing"]["pending_version_id"])
        nodes = [row for row in lines(output / "nodes.jsonl") if row["question_id"] == "0"]
        self.assertEqual({row["node"] for row in nodes}, set(NODES))
        self.assertTrue(all({"result", "status", "origin", "usage", "reason", "fallback_used", "refs"}
                            <= set(row) for row in nodes))
        request_text = (output / "requests.jsonl").read_text()
        self.assertNotIn("SECRET-PROMPT", request_text)
        self.assertNotIn("SECRET-PROVIDER", request_text)
        self.assertNotIn("SECRET-CONTENT", request_text)
        self.assertEqual({row["kind"] for row in lines(output / "requests.jsonl")},
                         {"attempt_queued", "request_attempt", "request_dispatch", "request_outcome"})
        self.assertIn("queued-only", request_text)
        self.assertEqual(lines(output / "failed_questions.jsonl"),
                         [{"group": "bird_dev", "question_id": "0", "version_id": current,
                           "state": "failed"}])
        verification = json.loads((output / "verification.json").read_text())
        self.assertTrue(verification["ok"])
        self.assertEqual(verification["manifest_questions"], 3)
        self.assertEqual(verification["exported_question_set_sha256"],
                         verification["manifest_question_set_sha256"])
        for name, expected in verification["files"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected["sha256"])
        self.assertEqual(Path(result), output)

    def test_compact_evaluation_compresses_typed_results_and_preserves_din_comparison(self):
        from scripts.rc_evaluation.din_sql import compact_reporting

        batch, manifest, _ = self.make_batch(ids=("0",))
        sqls = {"generation_base": "SELECT missing FROM numbers",
                "generation_rc3": "SELECT value FROM numbers",
                "revision_base": "SELECT SLOW",
                "revision_rc3": "SELECT value FROM numbers"}
        version = self.finish(batch, manifest, TaskKey("bird_dev", "0"), sqls=sqls)

        def execute(database, sql, *, timeout_seconds):
            common = {"dialect": "sqlite", "database_id": database["database_id"], "sql": sql,
                      "timeout_seconds": timeout_seconds, "rows": [], "columns": [], "error": None}
            if "missing" in sql:
                return {**common, "status": "error",
                        "error": {"type": "OperationalError", "sqlite_errorcode": 1}}
            if "SLOW" in sql:
                return {**common, "status": "timeout",
                        "error": {"type": "OperationalError", "sqlite_errorcode": 9}}
            return {**common, "status": "success", "columns": ["value"], "rows": [(1,)]}

        output = self.root / "evaluation"
        with patch.object(compact_reporting, "execute_sql", side_effect=execute):
            result = compact_reporting.evaluate_compact(batch, output)
        with sqlite3.connect(output / "evaluation.sqlite3") as connection:
            blob, checksum, raw_bytes, compressed_bytes, status = connection.execute(
                "SELECT result,raw_sha256,raw_bytes,compressed_bytes,status FROM queries "
                "WHERE sql='SELECT value FROM numbers'").fetchone()
            raw = zlib.decompress(blob)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), checksum)
            self.assertEqual(len(raw), raw_bytes)
            self.assertEqual(len(blob), compressed_bytes)
            self.assertEqual(restore_jsonable(json.loads(raw))["rows"], [(1,)])
            self.assertEqual(status, "success")
            payload = restore_jsonable(json.loads(connection.execute(
                "SELECT payload_json FROM items WHERE grp='bird_dev' AND question='0'").fetchone()[0]))
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        self.assertEqual(payload["version_id"], version)
        self.assertFalse(payload["stages"]["generation"]["base_correct"])
        self.assertTrue(payload["stages"]["generation"]["rc_correct"])
        self.assertIsNone(payload["stages"]["revision"]["base_correct"])
        self.assertEqual(quick, "ok")
        versions = json.loads((output / "versions.json").read_text())
        self.assertEqual(versions["settings"], {"sql_workers": 20, "sql_timeout_seconds": 180})
        self.assertEqual(Path(result), output)

    def test_compact_evaluation_runs_in_parallel_and_resumes_without_retrying_timeout(self):
        from scripts.rc_evaluation.din_sql import compact_reporting

        batch, manifest, _ = self.make_batch(groups=("bird_dev", "spider_dev"), ids=("0", "1"))
        for group in manifest["groups"]:
            for question_id in manifest["groups"][group]["ids"]:
                self.finish(batch, manifest, TaskKey(group, question_id), sqls={
                    "generation_base": f"SELECT {1 + int(question_id)}",
                    "generation_rc3": "SELECT TIMEOUT",
                    "revision_base": f"SELECT {3 + int(question_id)}",
                    "revision_rc3": "SELECT 4"})
        active = peak = calls = 0
        lock = threading.Lock()

        def execute(database, sql, *, timeout_seconds):
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
            time.sleep(0.03)
            with lock:
                active -= 1
            base = {"dialect": "sqlite", "database_id": database["database_id"], "sql": sql,
                    "timeout_seconds": timeout_seconds, "rows": [], "columns": [], "error": None}
            return ({**base, "status": "timeout", "error": {"type": "timeout"}}
                    if sql == "SELECT TIMEOUT" else
                    {**base, "status": "success", "columns": ["v"], "rows": [(1,)]})

        output = self.root / "evaluation-resume"
        result_sizes = []
        original_item = compact_reporting._evaluation_item
        def observed_item(key, current, references, results):
            result_sizes.append(len(results))
            return original_item(key, current, references, results)
        with patch.object(compact_reporting, "execute_sql", side_effect=execute), \
             patch.object(compact_reporting, "_evaluation_item", side_effect=observed_item):
            compact_reporting.evaluate_compact(batch, output)
        self.assertGreater(peak, 1)
        self.assertTrue(result_sizes)
        self.assertLessEqual(max(result_sizes), 5, "item construction must not receive the global result cache")
        first_calls = calls
        with patch.object(compact_reporting, "execute_sql", side_effect=AssertionError("must reuse cache/items")):
            compact_reporting.evaluate_compact(batch, output)
        self.assertEqual(calls, first_calls)
        with sqlite3.connect(output / "evaluation.sqlite3") as connection:
            self.assertGreater(connection.execute(
                "SELECT count(*) FROM queries WHERE status='timeout'").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM items").fetchone()[0], 4)

    def test_handoff_is_resumable_allowlisted_and_uses_independent_sqlite_snapshots(self):
        from scripts.rc_evaluation.din_sql import compact_reporting
        export_handoff = compact_reporting.export_handoff

        batch, manifest, _ = self.make_batch(ids=("0", "1"))
        self.finish(batch, manifest, TaskKey("bird_dev", "0"))
        monitoring = batch / "monitoring"
        monitoring.mkdir()
        for name in ("implementation.json", "endpoint.json", "live.json"):
            (monitoring / name).write_text(json.dumps({"name": name}))
        (monitoring / "controller.stdout.log").write_text("SECRET LOG")
        (monitoring / "launch.json").write_text('{"env":"SECRET"}')
        (batch / "reports").mkdir()
        (batch / "reports" / "large.json").write_text("REDUNDANT")
        guide = self.root / "guide.md"
        guide.write_text("handoff guide")
        output = self.root / "handoff"
        with patch("scripts.rc_evaluation.din_sql.compact_reporting.execute_sql") as execute:
            execute.return_value = {"status": "success", "columns": ["value"], "rows": [(1,)], "error": None}
            export_handoff(batch, output, guide=guide)
            self.assertFalse((output / "COMPLETE.json").exists())
            self.assertFalse(json.loads((output / "index.json").read_text())["complete"])
            self.finish(batch, manifest, TaskKey("bird_dev", "1"))
            original_export = compact_reporting.export_records
            def invalid_records(*args, **kwargs):
                destination = original_export(*args, **kwargs)
                verification = json.loads((destination / "verification.json").read_text())
                verification["ok"] = False
                (destination / "verification.json").write_text(json.dumps(verification))
                return destination
            with patch.object(compact_reporting, "export_records", side_effect=invalid_records):
                export_handoff(batch, output, guide=guide)
            invalid_index = json.loads((output / "index.json").read_text())
            self.assertFalse(invalid_index["complete"])
            self.assertFalse(invalid_index["acceptance"]["record_verification_ok"])
            self.assertFalse((output / "COMPLETE.json").exists())
            export_handoff(batch, output, guide=guide)

        raw = output / "raw_records"
        expected = {"manifest.json", "prepared/inputs.json", "prepared/import_report.json",
                    "monitoring/implementation.json", "monitoring/endpoint.json", "monitoring/live.json",
                    "group-bird_dev/run.sqlite3"}
        actual = {str(path.relative_to(raw)) for path in raw.rglob("*") if path.is_file()}
        self.assertEqual(actual, expected)
        self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))
        self.assertNotEqual((raw / "group-bird_dev/run.sqlite3").stat().st_ino,
                            (batch / "group-bird_dev/run.sqlite3").stat().st_ino)
        with sqlite3.connect(raw / "group-bird_dev/run.sqlite3") as connection:
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
        index = json.loads((output / "index.json").read_text())
        self.assertTrue(index["complete"])
        self.assertEqual(index["raw_records_directory"], "raw_records")
        self.assertEqual(index["record_directory"], "records")
        self.assertEqual(index["evaluation_directory"], "evaluation")
        self.assertEqual(index["guide"], "guide.md")
        self.assertEqual(index["record_summary"], "records/summary.json")
        self.assertEqual(index["record_verification"], "records/verification.json")
        self.assertEqual(index["evaluation_summary"], "evaluation/summary.json")
        self.assertEqual(index["evaluation_versions"], "evaluation/versions.json")
        self.assertEqual(index["evaluation_progress"], "evaluation/progress.json")
        self.assertEqual(index["evaluation_tables"], "evaluation/tables.md")
        self.assertEqual(index["evaluation_database"], "evaluation/evaluation.sqlite3")
        self.assertTrue(all(index["acceptance"].values()))
        self.assertTrue(index["source_runstore_verification"]["bird_dev"]["ok"])
        self.assertTrue(all(not Path(row["path"]).is_absolute() for row in index["files"]))
        self.assertTrue((output / "COMPLETE.json").is_file())
        self.assertEqual((output / "guide.md").read_text(), "handoff guide")
        with patch.object(compact_reporting, "export_records", side_effect=RuntimeError("injected crash")):
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                export_handoff(batch, output, guide=guide)
        self.assertFalse((output / "COMPLETE.json").exists(),
                         "a resumed export must revoke stale completion before doing work")

    def test_cli_accepts_export_commands_without_changing_existing_entrypoints(self):
        from scripts.rc_evaluation.din_sql import compact_reporting
        from scripts.rc_evaluation.din_sql.cli import main

        batch, _, _ = self.make_batch(ids=("0",))
        output = self.root / "handoff-cli"
        guide = self.root / "guide.md"
        guide.write_text("guide")
        with patch.object(compact_reporting, "export_handoff", return_value=output) as export, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            main(["export-handoff", "--batch", str(batch), "--output", str(output),
                  "--guide", str(guide)])
        export.assert_called_once_with(batch, output, guide=guide)
        self.assertEqual(json.loads(stdout.getvalue())["output"], str(output))


if __name__ == "__main__":
    unittest.main()
