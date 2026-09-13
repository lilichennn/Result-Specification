"""Offline safety/characterization tests; never load credentials or call a remote API."""
import tempfile
import unittest
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from scripts.efficiency_probe.core import ProbeBudget, run_requests, summarize, write_json
from scripts.efficiency_probe.local import LocalServer, thread_probe
from scripts.efficiency_probe.legacy import characterize


def work():
    return [{"stage": "sql_generation", "messages": [{"role": "user", "content": "fixture"}],
             "source_event_id": 1, "prompt_sha256": "fixture", "parser": "sql"}]


class ProbeTests(unittest.TestCase):
    def test_legacy_characterization_is_pinned_and_does_not_call_current_llm(self):
        from app.llm import LLM
        with patch.object(LLM, 'ask', side_effect=AssertionError('current runtime used')):
            result = characterize()
        self.assertEqual(result['source_revision'], 'be573b1f2ce07d2fd91cd25d2c37da5917c2f56b')
        self.assertEqual(result['connection_then_success']['requests'], 9)

    def test_group_probe_keeps_individual_results_and_obeys_request_limit(self):
        from scripts.efficiency_probe.local import group_probe
        with tempfile.TemporaryDirectory() as td:
            s = group_probe(Path(td)/'groups', workers=8, concurrency=4,
                            coordinators=6, groups=12, samples=5, delay=.003)
            self.assertEqual(s['requests'], 60)
            self.assertEqual(s['simulated_failures'], 1)
            self.assertEqual(s['complete_groups'], 11)
            self.assertLessEqual(s['peak_model_requests'], 4)
            self.assertGreater(s['peak_model_requests'], 1)
            self.assertTrue(s['integrity']['ok'])
            self.assertEqual(s['outstanding'], 0)
            self.assertEqual(s['retained_successful_samples'], 59)

    def test_group_probe_rejects_unbounded_configuration(self):
        from scripts.efficiency_probe.local import group_probe
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                group_probe(Path(td)/'bad', workers=8, concurrency=0, coordinators=6,
                            groups=12, samples=5, delay=.01)
            self.assertFalse((Path(td)/'bad').exists())

    def test_paced_admission_has_no_catch_up_burst(self):
        from scripts.efficiency_probe.core import AdmissionPacer
        p = AdmissionPacer(4)
        self.assertEqual(p.delay(10), 0)
        p.admitted(10)
        self.assertAlmostEqual(p.delay(10.1), .15)
        self.assertEqual(p.delay(20), 0)
        p.admitted(20)
        self.assertAlmostEqual(p.delay(20), .25)
        for invalid in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                AdmissionPacer(invalid)

    def test_pacing_and_worker_limit_are_both_effective(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, delay=.18) as server:
            out = Path(td)/'paced'
            s = run_requests(out, work(), ProbeBudget(4, 8, 20, 100000, 2),
                             server.url, 'local-only', 'fixture', workers=2, requests_per_second=10)
            self.assertEqual(s['parsed'], 8)
            self.assertLessEqual(s['peak_sdk_inflight'], 2)
            self.assertLessEqual(server.peak, 2)
            self.assertEqual(s['manifest']['worker_threads'], 2)
            c = sqlite3.connect(out/'run.sqlite3')
            submitted = sorted(datetime.fromisoformat(json.loads(r[0])['submitted_at']).timestamp()
                               for r in c.execute("select payload_json from events where kind='request'"))
            c.close()
            self.assertGreaterEqual(min(b-a for a,b in zip(submitted, submitted[1:])), .095)
            self.assertGreaterEqual(submitted[-1]-submitted[0], .69)
            self.assertEqual(s['outstanding'], 0)

    def test_paced_admission_deadline_does_not_send_waiting_work(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1) as server:
            s = run_requests(Path(td)/'run', work(), ProbeBudget(4, 8, .25, 10000, 2),
                             server.url, 'local-only', 'fixture', workers=4, requests_per_second=2)
            self.assertEqual(server.received, 1)
            self.assertEqual(s['requests'], 1)
            self.assertEqual(s['stop_reason'], 'admission_deadline')

    def test_invalid_worker_configuration_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                run_requests(Path(td)/'bad', work(), ProbeBudget(1, 2, 1, 1000, 1),
                             'http://127.0.0.1:1/v1', 'local', 'fixture', workers=0)
            self.assertFalse((Path(td)/'bad').exists())

    def test_remote_requires_explicit_authorization_and_all_finite_budgets(self):
        for override in ({"authorized": False}, {"max_requests": 0}, {"admission_seconds": 0},
                         {"token_stop": 0}, {"admission_seconds": float("inf")},
                         {"max_requests": 7}, {"concurrency": 0}):
            args = dict(concurrency=3, max_requests=6, admission_seconds=30, token_stop=1000,
                        timeout=1, authorized=True)
            args.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                ProbeBudget(**args).validate(remote=True)

    def test_unknown_tokens_are_not_zero_and_reasoning_is_not_added_twice(self):
        rows = [{"elapsed_seconds": 2, "http_success": True, "nonempty": True, "parsed": True,
                 "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                           "completion_tokens_details": {"reasoning_tokens": 15}}},
                {"elapsed_seconds": 3, "http_success": False, "nonempty": False, "parsed": False,
                 "usage": None, "error_type": "APIConnectionError"}]
        s = summarize(rows, elapsed=4)
        self.assertEqual(s["reported_tokens"]["total_tokens"], 30)
        self.assertEqual(s["reasoning_tokens_reported"], 15)
        self.assertEqual(s["usage_unknown_requests"], 1)
        self.assertEqual(s["parsed_per_minute"], 15)

    def test_cli_has_no_implicit_paid_mode(self):
        from scripts.deepeye_efficiency_probe import main
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                main(["remote", "--output", str(Path(td)/"not_created")])
            self.assertFalse((Path(td)/"not_created").exists())

    def test_admission_deadline_can_prevent_any_sdk_call(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1) as server:
            s = run_requests(Path(td)/"run", work(), ProbeBudget(1, 2, .0000001, 10000, 1),
                             server.url, "local-only", "fixture")
            self.assertEqual(server.received, 0)
            self.assertEqual(s["requests"], 0)
            self.assertEqual(s["stop_reason"], "admission_deadline")

    def test_parse_acceptance_is_not_sql_correctness(self):
        from scripts.efficiency_probe.workload import parse_response
        self.assertIsNone(parse_response({"parser": "sql"}, "SELECT 1"))
        self.assertEqual(parse_response({"parser": "sql"}, "<result>SELECT non_existing_column</result>"),
                         "SELECT non_existing_column")
        self.assertEqual(parse_response({"parser": "selection"}, "<result>tie</result>"), "TIE")
        self.assertIsNone(parse_response({"parser": "selection"}, "<result>C</result>"))

    def test_output_artifacts_cannot_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "summary.json"
            write_json(p, {"old": 1})
            with self.assertRaises(FileExistsError):
                write_json(p, {"new": 2})
            self.assertIn('"old"', p.read_text())

    def test_real_local_http_overlaps_and_incremental_records_verify(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=4) as server:
            s = run_requests(Path(td)/"run", work(), ProbeBudget(4, 8, 20, 100000, 2),
                             server.url, "local-only", "fixture")
            self.assertEqual(s["requests"], 8)
            self.assertEqual(s["parsed"], 8)
            self.assertGreaterEqual(server.peak, 4)
            self.assertEqual(s["peak_sdk_inflight"], 4)
            self.assertGreaterEqual(s["peak_wire_inflight"], 4)
            self.assertTrue(s["integrity"]["ok"])

    def test_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=2, status=500) as server:
            s = run_requests(Path(td)/"run", work(), ProbeBudget(2, 4, 20, 100000, 2),
                             server.url, "local-only", "fixture")
            self.assertEqual(s["requests"], 4)
            self.assertEqual(server.received, 4)
            self.assertEqual(s["http_success"], 0)

    def test_auth_failure_stops_issuing_new_requests_and_drains(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, status=401) as server:
            s = run_requests(Path(td)/"run", work(), ProbeBudget(1, 2, 20, 100000, 2),
                             server.url, "local-only", "fixture")
            self.assertEqual(s["requests"], 1)
            self.assertEqual(s["stop_reason"], "authentication_error")
            self.assertTrue(s["integrity"]["ok"])

    def test_token_threshold_stops_new_requests(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1) as server:
            s = run_requests(Path(td)/"run", work(), ProbeBudget(1, 2, 20, 1, 2),
                             server.url, "local-only", "fixture")
            self.assertEqual(s["requests"], 1)
            self.assertEqual(s["stop_reason"], "reported_token_threshold")

    def test_timeout_is_not_hidden_by_sdk_retry(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, delay=.3) as server:
            s = run_requests(Path(td)/"run", work(), ProbeBudget(1, 2, 20, 10000, .05),
                             server.url, "local-only", "fixture")
            self.assertEqual(server.received, 2)
            self.assertEqual(s["errors_by_type"], {"APITimeoutError": 2})

    def test_barrier_measures_real_thread_overlap(self):
        s = thread_probe(24, hold_seconds=.03)
        self.assertEqual(s["peak_waiting"], 24)
        self.assertEqual(s["joined"], 24)
        self.assertIsNone(s["error"])

    def test_thread_creation_failure_releases_existing_workers(self):
        import threading
        original = threading.Thread.start
        calls = 0
        def start(t):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("fixture capacity reached")
            original(t)
        with patch.object(threading.Thread, "start", start):
            s = thread_probe(10, hold_seconds=.01)
        self.assertEqual(s["joined"], 3)
        self.assertIn("RuntimeError", s["error"])

    def test_actual_old_ask_loses_three_successes_on_group_retry(self):
        c = characterize()["connection_then_success"]
        self.assertEqual(c["requests"], 9)
        self.assertEqual(c["reported_total_tokens"], 240)
        self.assertEqual(c["native_total_tokens"], 150)
        self.assertEqual(c["retained_response_ids"], ["r4", "r5", "r6", "r7", "r8"])

    def test_actual_old_extractor_preserves_valid_results_on_parser_retry(self):
        c = characterize()["parse_then_success"]
        self.assertEqual(c["requests"], 6)
        self.assertEqual(c["native_total_tokens"], 180)
        self.assertEqual(c["retained_response_ids"], ["r0", "r1", "r2", "r4", "r5"])

    def test_actual_old_nested_retries_can_report_zero_tokens_after_many_responses(self):
        c = characterize()["exhausted"]
        self.assertEqual(c["requests"], 16)
        self.assertEqual(c["reported_total_tokens"], 360)
        self.assertEqual(c["native_total_tokens"], 0)
        self.assertEqual(c["retained_response_ids"], [])


if __name__ == "__main__":
    unittest.main()
