"""Independent checks for rate/occupancy arithmetic used in capacity reports."""
import unittest


class ReportTests(unittest.TestCase):
    def test_stale_exported_integrity_does_not_hide_current_checksum_corruption(self):
        import scripts.deepeye_efficiency_report as report
        from scripts.efficiency_probe.core import run_requests, ProbeBudget
        from scripts.efficiency_probe.local import LocalServer
        from pathlib import Path
        import tempfile
        import sqlite3
        workload = [{'stage': 'sql_generation', 'parser': 'sql', 'source_event_id': 1,
                     'messages': [{'role': 'user', 'content': 'fixture'}], 'prompt_sha256': 'fixture',
                     'item_key': 'local/test', 'branch_path': 'fixture'}]
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, delay=.01) as server:
            base, run = Path(td), Path(td)/'remote_a'
            run_requests(run, workload, ProbeBudget(2, 2, 5, 10000, 2),
                         server.url, 'local-only', 'fixture')
            with sqlite3.connect(run/'run.sqlite3') as db:
                db.execute('drop trigger events_no_update')
                db.execute("update events set record_checksum='corrupt' where kind='response'")
                self.assertEqual(db.execute('pragma integrity_check').fetchone()[0], 'ok')
            with self.assertRaisesRegex(ValueError, 'integrity|checksum'):
                report.analyze(run)
            with self.assertRaisesRegex(ValueError, 'integrity|checksum'):
                report.analyze_campaign(base)

    def test_campaign_counts_unique_requests_and_checks_changing_total_limit(self):
        import scripts.deepeye_efficiency_report as report
        from scripts.efficiency_probe.core import run_requests, ProbeBudget
        from scripts.efficiency_probe.local import LocalServer
        from pathlib import Path
        import tempfile
        self.assertTrue(hasattr(report, 'analyze_campaign'), 'missing campaign timeline accounting')
        workload = [{'stage': 'sql_generation', 'parser': 'sql', 'source_event_id': 1,
                     'messages': [{'role': 'user', 'content': 'fixture'}], 'prompt_sha256': 'fixture',
                     'item_key': 'local/test', 'branch_path': 'fixture'}]
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, delay=.01) as server:
            base = Path(td)
            run_requests(base/'remote_a', workload, ProbeBudget(2, 2, 5, 10000, 2),
                         server.url, 'local-only', 'fixture')
            run_requests(base/'remote_b', workload, ProbeBudget(3, 2, 5, 10000, 2),
                         server.url, 'local-only', 'fixture', draining_run=base/'remote_a')
            result = report.analyze_campaign(base)
            self.assertEqual(result['requests'], 4)
            self.assertEqual(result['http_success'], 4)
            self.assertEqual(result['reported_tokens']['total_tokens'], 120)
            self.assertEqual([p['limit'] for p in result['phases']], [2, 3])
            self.assertTrue(result['all_phase_limits_respected'])
            self.assertEqual(sum(p['new_headers'] for p in result['phases']), 4)
            self.assertEqual(sum(p['successful_terminals_all_origins'] for p in result['phases']), 4)

    def test_handoff_analysis_collects_all_ancestors_and_requires_their_terminals(self):
        import scripts.deepeye_efficiency_report as report
        from tests.test_deepeye_efficiency_handoff import fixture_source
        from pathlib import Path
        import tempfile
        self.assertTrue(hasattr(report, 'completed_handoff_runs'), 'missing full-chain analysis')
        with tempfile.TemporaryDirectory() as td:
            a_path, b_path = Path(td)/'a', Path(td)/'b'
            a_store, a = fixture_source(a_path, count=1)
            b_store, b = fixture_source(b_path, count=1, parent=a_path)
            try:
                b_store.append_event(b, 'response', {'request_no': 0})
                with self.assertRaises(ValueError):
                    report.completed_handoff_runs(b_path)
                a_store.append_event(a, 'error', {'request_no': 0})
                self.assertEqual(report.completed_handoff_runs(b_path), [b_path.resolve(), a_path.resolve()])
            finally:
                a_store.close()
                b_store.close()

    def test_handoff_peak_uses_overlap_not_sum_of_separate_peaks(self):
        import scripts.deepeye_efficiency_report as report
        self.assertTrue(hasattr(report, 'handoff_statistics'), 'missing combined handoff accounting')
        s = report.handoff_statistics([(0, 5), (0, 2)], [(3, 8), (6, 10)], 3)
        self.assertEqual(s['peak_inflight'], 2)
        self.assertEqual(s['source_pending_at_first_new_wire'], 1)
        self.assertEqual(s['duration_seconds'], 7)
        self.assertAlmostEqual(s['mean_inflight'], 11/7)

    def test_handoff_has_no_overlap_when_source_is_already_drained(self):
        import scripts.deepeye_efficiency_report as report
        self.assertTrue(hasattr(report, 'handoff_statistics'), 'missing combined handoff accounting')
        s = report.handoff_statistics([(0, 1)], [(2, 4)], 3)
        self.assertEqual(s['peak_inflight'], 1)
        self.assertEqual(s['source_pending_at_first_new_wire'], 0)

    def test_rolling_window_is_half_open_and_unsorted_input_is_supported(self):
        from scripts.deepeye_efficiency_report import rolling_peak
        self.assertEqual(rolling_peak([2, 0, 1], 1), 1)
        self.assertEqual(rolling_peak([.1, .2, .3, 1.2], 1), 3)
        self.assertEqual(rolling_peak([], 1), 0)
        with self.assertRaises(ValueError):
            rolling_peak([0], 0)

    def test_occupancy_integrates_waiting_time_not_just_configured_limit(self):
        from scripts.deepeye_efficiency_report import interval_statistics
        intervals = [(0, 4), (1, 3), (2, 5)]
        s = interval_statistics(intervals, 0, 5, 3)
        self.assertEqual(s['peak_inflight'], 3)
        self.assertAlmostEqual(s['mean_inflight'], 1.8)
        self.assertAlmostEqual(s['fraction_at_90pct_capacity'], .2)
        clipped = interval_statistics(intervals, 2, 4, 3)
        self.assertAlmostEqual(clipped['mean_inflight'], 2.5)
        self.assertAlmostEqual(clipped['fraction_at_90pct_capacity'], .5)

    def test_finishing_and_starting_at_same_time_does_not_double_count(self):
        from scripts.deepeye_efficiency_report import interval_statistics
        s = interval_statistics([(0, 1), (1, 2)], 0, 2, 2)
        self.assertEqual(s['peak_inflight'], 1)
        self.assertEqual(s['mean_inflight'], 1)
        self.assertEqual(s['fraction_at_90pct_capacity'], 0)
