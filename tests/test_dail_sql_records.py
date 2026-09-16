import copy
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.dail_sql.config import MODES, TaskKey
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from scripts.baseline_adapters.deepeye.run_store import RunStore


def manifest():
    return {"batch_id": "batch", "groups": {"spider_dev": {"ids": ["0", "1", "2"]},
                                             "bird_dev": {"ids": ["1"]}}}


_DEFAULT = object()


def complete_round(records, version, rid, round_no=1, parent=None, rc=False, usage=_DEFAULT,
                   aggregate=_DEFAULT, persist=True, content="SELECT 1", rows=None):
    samples, candidates = [], []
    usage = {"total_tokens": 3} if usage is _DEFAULT else usage
    for pos in range(5):
        attempt = records.append(version, "request_attempt", {
            "round_execution_id": rid, "sample_position": pos, "attempt_no": 1})
        choice = {"index": 0, "message": {"content": content}}
        result = records.append(version, "request_result", {
            "request_attempt_id": attempt, "status": "success", "choice": choice,
            "usage": usage})
        samples.append({"sample_position": pos, "status": "success", "request_attempt_ids": [attempt],
                        "successful_request_id": result, "success_usage": usage,
                        "choice": choice, "error": None})
        execution = records.append(version, "vote_execution", {
            "status": "success", "rows": [[Decimal("1")]] if rows is None else rows})
        candidate = {"candidate_id": rid + "-" + str(pos), "choice_position": pos,
                     "source_request_id": result,
                     "provider_choice_index": 0, "raw_text": content or "", "candidate_sql": "SELECT 1",
                     "vote_sql": "SELECT 1", "vote_execution_ref": execution}
        records.append(version, "candidate", candidate)
        candidates.append(candidate)
    payload = {"round_execution_id": rid, "round_no": round_no, "status": "success", "rc_injected": rc,
               "actual_parent_round_id": parent, "example_ids": [str(i) for i in range(9)], "samples": samples,
               "request_attempt_ids": [s["request_attempt_ids"][0] for s in samples],
               "successful_request_ids": [s["successful_request_id"] for s in samples],
               "success_usage": {"total_tokens": 15} if aggregate is _DEFAULT else aggregate, "candidates": candidates,
               "selection": {"candidate_id": candidates[0]["candidate_id"]},
               "next_example_ids": [str(i) for i in range(9)] if round_no == 1 else [], "error": None}
    if persist:
        records.append(version, "round_result", payload)
    return payload


def complete_modes(records, version):
    modes = {}
    for mode in MODES:
        first = complete_round(records, version, mode + "-r1", rc=mode in ("rc_first", "rc_both"))
        second = complete_round(records, version, mode + "-r2", 2, first["round_execution_id"],
                                mode in ("rc_second", "rc_both"))
        modes[mode] = records.append(version, "mode_result", {
            "mode": mode, "status": "succeeded", "first_round_id": first["round_execution_id"],
            "second_round_id": second["round_execution_id"],
            "final_candidate_id": second["selection"]["candidate_id"], "failure_origin": None})
    return modes


class DailRecordsTests(unittest.TestCase):
    def test_reopen_rejects_corrupt_missing_and_foreign_sources_without_partial_trust(self):
        for damage in ("checksum", "missing", "foreign", "wrong_kind"):
            with self.subTest(damage=damage):
                root = self.root / damage
                with DailRecords(root, manifest()) as writer:
                    version = writer.begin_version(self.key)
                    result = complete_round(writer, version, "a", persist=False)
                    if damage == "foreign":
                        other = writer.begin_version(TaskKey("batch", "spider_dev", "2"))
                        foreign = complete_round(writer, other, "other")
                        result["candidates"] = foreign["candidates"]
                    elif damage == "wrong_kind":
                        result["candidates"][0]["vote_execution_ref"] = result["successful_request_ids"][0]
                    # Bypass DailRecords to emulate invalid legacy/imported consumer
                    # with a valid RunStore checksum. Hydration must revalidate it.
                    store, attempt, _ = writer._version(version)
                    store.append_event(attempt["attempt_id"], "round_result", result)
                db = root / ("group-" + self.key.group.encode().hex()) / "run.sqlite3"
                with sqlite3.connect(db) as connection:
                    if damage == "checksum":
                        connection.execute("DROP TRIGGER events_no_update")
                        connection.execute("UPDATE events SET payload_json='{}' WHERE kind='vote_execution'")
                    elif damage == "missing":
                        connection.execute("DROP TRIGGER events_no_delete")
                        connection.execute("DELETE FROM events WHERE kind='vote_execution'")
                with DailRecords(root, manifest()) as reopened:
                    for _ in range(2):
                        with self.assertRaises(ValueError):
                            reopened.find_source(version, "round_result", "a")

    def test_round_reads_each_durable_source_at_most_once(self):
        version = self.records.begin_version(self.key)
        result = complete_round(self.records, version, "counted", persist=False)
        reads = Counter()
        original = RunStore._event_dict
        def decode(row):
            reads[(row["kind"], row["event_no"])] += 1
            return original(row)
        with patch.object(RunStore, "_event_dict", side_effect=decode):
            self.records.append(version, "round_result", result)
        self.assertTrue(all(count <= 1 for count in reads.values()), reads)

    def test_mode_and_seal_reuse_verified_sources_and_reopen_hydrates_once(self):
        version = self.records.begin_version(self.key)
        for rid, number, parent, rc in (("a", 1, None, False), ("b", 1, None, True),
                                         ("c", 2, "a", False), ("d", 2, "a", True)):
            complete_round(self.records, version, rid, number, parent, rc)
        self.records.close()
        reads = Counter()
        original = RunStore._event_dict
        def decode(row):
            reads[(row["kind"], row["event_no"])] += 1
            return original(row)
        with DailRecords(self.root, manifest()) as reopened, patch.object(RunStore, "_event_dict", side_effect=decode):
            modes = {}
            for mode in MODES:
                first = "b" if mode in ("rc_first", "rc_both") else "a"
                second = "d" if mode in ("rc_second", "rc_both") else "c"
                modes[mode] = reopened.append(version, "mode_result", {
                    "mode": mode, "status": "succeeded", "first_round_id": first,
                    "second_round_id": second, "final_candidate_id": second + "-0", "failure_origin": None})
            reopened.seal(version, modes)
        heavy = {source: count for source, count in reads.items()
                 if source[0] in ("vote_execution", "request_result", "candidate")}
        self.assertEqual(len(heavy), 60)
        self.assertTrue(all(count == 1 for count in heavy.values()), heavy)

    def test_read_only_history_and_sealed_version_coexist_with_writer(self):
        self.assertIn('read_only', __import__('inspect').signature(DailRecords).parameters)
        version = self.records.begin_version(self.key)
        self.records.seal(version, complete_modes(self.records, version))
        interrupted = self.records.begin_version(self.key)
        with DailRecords(self.root, manifest(), read_only=True) as reader:
            self.assertEqual([v['version_id'] for v in reader.iter_versions()], [version, interrupted])
            frozen = reader.get_version(version)
            self.assertEqual(frozen['task_key']['question_id'], '1')
            self.assertEqual(frozen['modes']['native']['status'], 'succeeded')
            self.assertEqual(len(list(reader.iter_events(version, 'request_attempt'))), 40)
            for operation in (lambda: reader.begin_version(self.key),
                              lambda: reader.append(interrupted, 'diagnostic', {}),
                              lambda: reader.seal(version, {})):
                with self.assertRaises(PermissionError):
                    operation()
        self.assertFalse((self.root / ('group-' + 'bird_dev'.encode().hex())).exists())
        missing = self.root / 'missing'
        with self.assertRaises(FileNotFoundError):
            DailRecords(missing, manifest(), read_only=True)
        self.assertFalse(missing.exists())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "records"
        self.records = DailRecords(self.root, manifest())
        self.addCleanup(self.records.close)
        self.key = TaskKey("batch", "spider_dev", "1")

    def test_membership_and_frozen_manifest(self):
        for key in (TaskKey("wrong", "spider_dev", "1"), TaskKey("batch", "unknown", "1"),
                    TaskKey("batch", "spider_dev", "99")):
            with self.assertRaises(ValueError):
                self.records.begin_version(key)
        self.records.close()
        changed = manifest()
        changed["model"] = "changed"
        with self.assertRaises(ValueError):
            DailRecords(self.root, changed)

    def test_requester_history_is_round_scoped_compact_and_defensive(self):
        version = self.records.begin_version(self.key)
        complete_round(self.records, version, "old", persist=False)
        new = self.records.append(version, "request_attempt", {
            "round_execution_id": "new", "sample_position": 2, "attempt_no": 1})
        self.assertTrue(hasattr(self.records, "request_history"), "compact requester accessor missing")
        with patch.object(RunStore, "iter_events", side_effect=AssertionError("repeated scan")):
            self.assertEqual(self.records.version_key(version), self.key)
            history = self.records.request_history(version, "new")
            self.assertEqual(history, [{"sample_position": 0, "attempts": []},
                {"sample_position": 1, "attempts": []},
                {"sample_position": 2, "attempts": [{"request_attempt_id": new,
                    "request_result_id": None, "status": None}]},
                {"sample_position": 3, "attempts": []}, {"sample_position": 4, "attempts": []}])
            history[2]["attempts"].clear()
            self.assertEqual(len(self.records.request_history(version, "new")[2]["attempts"]), 1)

    def test_twelve_interleaved_versions_do_not_rescan_after_resume_hydration(self):
        versions = [self.records.begin_version(self.key) for _ in range(12)]
        sources = [self.records.append(v, "diagnostic", {"body": "reasoning" * 100}) for v in versions]
        self.records.close()
        with DailRecords(self.root, manifest()) as reopened:
            for v in versions:
                reopened.append(v, "diagnostic", {"checkpoint": "hydrated"})
            with patch.object(RunStore, "iter_events", side_effect=AssertionError("repeated history scan")):
                for _ in range(2):
                    for v, source in zip(versions, sources):
                        reopened.append(v, "diagnostic", {"progress": 1})
                        self.assertEqual(reopened.get_event(v, source)["body"], "reasoning" * 100)

    def test_success_without_usage_completes_with_unknown_aggregate(self):
        version = self.records.begin_version(self.key)
        complete_round(self.records, version, "unknown-usage", usage=None, aggregate=None)
        result = self.records.get_round(version, "unknown-usage")
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["success_usage"])
        self.assertTrue(all(s["success_usage"] is None for s in result["samples"]))

    def test_public_usage_aggregation_preserves_absence_and_missing_fields(self):
        from scripts.baseline_adapters.dail_sql.records import aggregate_observed_usage
        self.assertIsNone(aggregate_observed_usage([{"total_tokens": 3}, None]))
        self.assertEqual(aggregate_observed_usage([{"total_tokens": 3}, {}]), {"total_tokens": None})
        self.assertIsNone(aggregate_observed_usage([]))

    def test_round_cannot_hide_started_attempt_or_downgrade_real_success(self):
        version = self.records.begin_version(self.key)
        result = complete_round(self.records, version, "ledger", persist=False)
        self.records.close()
        with DailRecords(self.root, manifest()) as reopened:
            bad = copy.deepcopy(result)
            bad.update(status="failed", error="cancelled", success_usage=None, candidates=[], selection=None)
            for sample in bad["samples"]:
                sample.update(status="failed", successful_request_id=None, success_usage=None, error="cancelled")
                sample.pop("choice")
            bad["successful_request_ids"] = []
            with self.assertRaises(ValueError):
                reopened.append(version, "round_result", bad)
            started = reopened.append(version, "request_attempt", {
                "round_execution_id": "interrupted", "sample_position": 0, "attempt_no": 1})
            bad["round_execution_id"] = "interrupted"
            for sample in bad["samples"]:
                sample["request_attempt_ids"] = []
            bad["request_attempt_ids"] = []
            with self.assertRaises(ValueError):
                reopened.append(version, "round_result", bad)
            bad["samples"][0]["request_attempt_ids"] = [started]
            bad["request_attempt_ids"] = [started]
            reopened.append(version, "round_result", bad)

    def test_round_rejects_foreign_request_candidates_even_with_same_positions(self):
        version = self.records.begin_version(self.key)
        old = complete_round(self.records, version, "old", content="SELECT 1")
        new = complete_round(self.records, version, "new", content="SELECT 2", persist=False)
        new["candidates"] = old["candidates"]
        new["selection"] = old["selection"]
        with self.assertRaises(ValueError):
            self.records.append(version, "round_result", new)

    def test_candidate_raw_text_matches_request_and_null_content_maps_to_empty(self):
        version = self.records.begin_version(self.key)
        result = complete_round(self.records, version, "empty", content=None, persist=False)
        bad = copy.deepcopy(result["candidates"][0])
        bad.update(candidate_id="wrong-text", raw_text="invented")
        self.records.append(version, "candidate", bad)
        result["candidates"][0] = bad
        result["selection"] = {"candidate_id": "wrong-text"}
        with self.assertRaises(ValueError):
            self.records.append(version, "round_result", result)
        result["candidates"][0] = self.records.get_event(version, self.records.events(version, "candidate")[0]["event_id"])
        result["selection"] = {"candidate_id": "empty-0"}
        self.records.append(version, "round_result", result)

    def test_seal_is_durable_point_read_and_late_append_rejected(self):
        version = self.records.begin_version(self.key)
        modes = complete_modes(self.records, version)
        self.assertEqual(self.records.seal(version, modes), version)
        self.assertTrue(self.records.is_sealed(self.key, version))
        with self.assertRaises(ValueError):
            self.records.append(version, "late", {})
        self.records.close()
        with DailRecords(self.root, manifest()) as reopened:
            with patch("scripts.baseline_adapters.deepeye.run_store.RunStore.iter_events", side_effect=AssertionError("history scan")):
                self.assertTrue(reopened.is_sealed(self.key, version))
            self.assertFalse(reopened.is_sealed(TaskKey("batch", "bird_dev", "1"), version))
            self.assertEqual(reopened.get_round(version, "native-r1")["success_usage"], {"total_tokens": 15})

    def test_missing_mode_and_cross_version_references_rejected(self):
        version = self.records.begin_version(self.key)
        modes = complete_modes(self.records, version)
        with self.assertRaises(ValueError):
            self.records.seal(version, {k: v for k, v in modes.items() if k != "native"})
        other = self.records.begin_version(TaskKey("batch", "spider_dev", "2"))
        with self.assertRaises(ValueError):
            self.records.seal(other, modes)

    def test_invented_round_success_sources_rejected(self):
        version = self.records.begin_version(self.key)
        round_ = complete_round(self.records, version, "real")
        bad = copy.deepcopy(round_)
        bad["round_execution_id"] = "invented"
        bad["samples"][0]["successful_request_id"] = "not-persisted"
        with self.assertRaises(ValueError):
            self.records.append(version, "round_result", bad)

    def test_request_results_require_unique_real_attempt_source(self):
        version = self.records.begin_version(self.key)
        with self.assertRaises(ValueError):
            self.records.append(version, "request_result", {"request_attempt_id": "missing", "status": "failed"})
        attempt = self.records.append(version, "request_attempt", {"round_execution_id": "r", "sample_position": 0, "attempt_no": 1})
        self.records.append(version, "request_result", {"request_attempt_id": attempt, "status": "failed", "usage": None})
        with self.assertRaises(ValueError):
            self.records.append(version, "request_result", {"request_attempt_id": attempt, "status": "success"})

    def test_sampling_attempt_budget_and_terminal_success_are_not_reset(self):
        version = self.records.begin_version(self.key)
        for number in range(1, 6):
            self.records.append(version, "request_attempt", {"round_execution_id": "r", "sample_position": 0, "attempt_no": number})
        with self.assertRaises(ValueError):
            self.records.append(version, "request_attempt", {"round_execution_id": "r", "sample_position": 0, "attempt_no": 6})
        with self.assertRaises(ValueError):
            self.records.append(version, "request_attempt", {"round_execution_id": "r", "sample_position": 0, "attempt_no": 1})
        complete_round(self.records, version, "finished")
        with self.assertRaises(ValueError):
            self.records.append(version, "request_attempt", {"round_execution_id": "finished", "sample_position": 0, "attempt_no": 2})

    def test_nested_and_unknown_usage_are_preserved(self):
        version = self.records.begin_version(self.key)
        r = complete_round(self.records, version, "nested", usage={"total_tokens": 3, "details": {"reasoning": None}},
                           aggregate={"total_tokens": 15, "details": {"reasoning": None}})
        self.assertIsNone(self.records.get_round(version, "nested")["success_usage"]["details"]["reasoning"])

    def test_failed_round_does_not_accept_unresolved_candidate_refs(self):
        version = self.records.begin_version(self.key)
        bad = {"round_execution_id": "failed", "round_no": 1, "status": "failed", "rc_injected": False,
               "actual_parent_round_id": None, "example_ids": [],
               "samples": [{"sample_position": p, "status": "failed", "request_attempt_ids": [],
                            "successful_request_id": None, "success_usage": None, "error": "cancelled"} for p in range(5)],
               "request_attempt_ids": [], "successful_request_ids": [], "success_usage": None,
               "candidates": [{"candidate_id": "missing"}], "selection": None, "next_example_ids": [], "error": "cancelled"}
        with self.assertRaises(ValueError):
            self.records.append(version, "round_result", bad)

    def test_result_inline_serialization_and_version_recovery_events(self):
        version = self.records.begin_version(self.key)
        event = self.records.append(version, "vote_execution", {"rows": [[Decimal("1.20"), None]]})
        self.records.close()
        with DailRecords(self.root, manifest()) as reopened:
            self.assertEqual(reopened.get_event(version, event)["rows"], [[Decimal("1.20"), None]])
            self.assertEqual(reopened.events(version, "vote_execution")[0]["event_id"], event)

    def test_failed_terminal_modes_can_seal_without_fabricated_success(self):
        version = self.records.begin_version(self.key)
        modes = {}
        for mode in MODES:
            modes[mode] = self.records.append(version, "mode_result", {
                "mode": mode, "status": "failed", "first_round_id": None, "second_round_id": None,
                "final_candidate_id": None, "failure_origin": {"stage": "preparation", "error": "unavailable"}})
        self.records.seal(version, modes)
        self.assertTrue(self.records.is_sealed(self.key, version))

    def test_shared_second_round_preserves_actual_parent_with_equal_examples(self):
        version = self.records.begin_version(self.key)
        complete_round(self.records, version, "plain-first")
        complete_round(self.records, version, "rc-first", rc=True)
        second = complete_round(self.records, version, "shared-second", 2, "plain-first")
        event = self.records.append(version, "mode_result", {
            "mode": "rc_first", "status": "succeeded", "first_round_id": "rc-first",
            "second_round_id": "shared-second", "final_candidate_id": second["selection"]["candidate_id"],
            "failure_origin": None})
        self.assertEqual(self.records.get_event(version, event)["first_round_id"], "rc-first")
        self.assertEqual(self.records.get_round(version, "shared-second")["actual_parent_round_id"], "plain-first")

    def test_crash_boundaries_preserve_old_or_new_whole_version(self):
        with CurrentIndex(Path(self.tmp.name) / "index.sqlite3") as index:
            old = self.records.begin_version(self.key)
            self.records.seal(old, complete_modes(self.records, old))
            index.publish(self.key, old, index.claim(self.key), self.records.is_sealed)
            new = self.records.begin_version(self.key)
            lease = index.claim(self.key)
            self.records.append(new, "diagnostic", {"checkpoint": "half"})
            with self.assertRaises(ValueError):
                index.publish(self.key, new, lease, self.records.is_sealed)
            self.assertEqual(index.current(self.key), old)
            self.records.seal(new, complete_modes(self.records, new))
            self.assertEqual(index.current(self.key), old)
            index.publish(self.key, new, lease, self.records.is_sealed)
        with CurrentIndex(Path(self.tmp.name) / "index.sqlite3") as index:
            self.assertEqual(index.current(self.key), new)
        self.assertTrue(self.records.is_sealed(self.key, old))

    def test_standalone_run_store_import_and_lazy_original_exports(self):
        root = Path(__file__).resolve().parents[1]
        standalone = subprocess.run([sys.executable, "-E", "-B", "-c",
            "from scripts.baseline_adapters.deepeye.run_store import RunStore; import sys; assert 'app' not in sys.modules"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(standalone.returncode, 0, standalone.stderr)
        legacy = subprocess.run([sys.executable, "-E", "-B", "-c",
            "import sys; sys.path.insert(0, 'baselines/DeepEye-SQL'); "
            "import scripts.baseline_adapters.deepeye as p; "
            "assert all(getattr(p, name) is not None for name in p.__all__)"], cwd=root, capture_output=True, text=True)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
