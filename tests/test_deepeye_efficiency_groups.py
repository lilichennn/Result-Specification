"""Grouped SDK capacity tests: local HTTP only, no credentials or SQL execution."""
import inspect
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import threading

from scripts.efficiency_probe.core import ProbeBudget, run_requests
from scripts.efficiency_probe.local import LocalServer


def workload():
    return [dict(stage='generation' if n == 4 else 'revision',
                 messages=[dict(role='user', content=f'fixture source {n}')],
                 source_event_id=n, prompt_sha256=f'fixture-{n}', parser='sql')
            for n in (1, 4, 5)]


def events(out, kind):
    with sqlite3.connect(out/'run.sqlite3') as c:
        return [json.loads(r[0]) for r in c.execute(
            'select payload_json from events where kind=? order by event_id', (kind,))]


class GroupedSDKTests(unittest.TestCase):
    def test_native_node_profiles_do_not_pad_every_group_to_five(self):
        from scripts.efficiency_probe import groups
        self.assertTrue(hasattr(groups, 'build_plan'), 'CLI must derive native node sample counts')
        sources = [dict(stage='sql_generation', branch_path=['generation.dc']),
                   dict(stage='sql_revision', branch_path=['revision.SyntaxChecker.extraction']),
                   dict(stage='sql_revision', branch_path=['revision.OrderByNullChecker.extraction']),
                   dict(stage='sql_revision', branch_path=['revision.ResultChecker.extraction'])]
        self.assertEqual(groups.build_plan(sources, 'generation', 2),
                         [dict(workload_index=0, samples=4)]*2)
        self.assertEqual(groups.build_plan(sources, 'revision', 4),
                         [dict(workload_index=1, samples=5), dict(workload_index=2, samples=1),
                          dict(workload_index=3, samples=5), dict(workload_index=1, samples=5)])

    def test_thread_creation_failure_unwinds_without_sending_requests(self):
        real_start = threading.Thread.start
        created = []
        def start(thread):
            if thread.name.startswith('probe-group'):
                if len(created) == 1:
                    raise RuntimeError('fixture OS thread exhaustion')
                created.append(thread)
            return real_start(thread)
        with tempfile.TemporaryDirectory() as td, LocalServer(1) as server:
            with patch.object(threading.Thread, 'start', start), self.assertRaises(RuntimeError):
                self.call(Path(td)/'thread_error', server,
                          [dict(workload_index=2, samples=5)]*5)
            self.assertEqual(server.received, 0)
            self.assertTrue(all(not t.is_alive() for t in created))

    def test_group_recording_failure_stops_new_paid_work(self):
        from scripts.efficiency_probe.core import RunStore
        real_append = RunStore.append_event
        def append(store, attempt, kind, payload, *args, **kwargs):
            if kind == 'group_result':
                raise OSError('fixture group record failure')
            return real_append(store, attempt, kind, payload, *args, **kwargs)
        with tempfile.TemporaryDirectory() as td, LocalServer(1) as server:
            with patch.object(RunStore, 'append_event', append), self.assertRaises(OSError):
                self.call(Path(td)/'record_error', server,
                          [dict(workload_index=0, samples=1)]*20,
                          coordinators=1, requests=20, rate=10)
            self.assertLess(server.received, 20)
            self.assertEqual(server.active, 0)

    def test_local_group_cli_uses_real_sdk_path(self):
        from scripts.deepeye_efficiency_probe import main
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'cli'
            try:
                main(['local-group-http', '--output', str(out), '--concurrency', '4',
                      '--workers', '4', '--coordinators', '3', '--groups', '6',
                      '--group-profile', 'revision', '--requests', '8'])
            except SystemExit:
                self.fail('CLI is missing the grouped real HTTP mode')
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['requests'], 8)
            self.assertEqual(summary['peak_wire_inflight'], 4)
            self.assertTrue(summary['integrity']['ok'])

    def call(self, out, server, plan, *, requests=20, coordinators=3, deadline=20,
             rate=None, target_successes=None):
        self.assertIn('group_plan', inspect.signature(run_requests).parameters,
                      'The real SDK probe must exercise actual group coordinators')
        return run_requests(out, workload(), ProbeBudget(4, requests, deadline, 100000, 3),
            server.url, 'local-only', 'fixture', workers=4, group_plan=plan,
            coordinators=coordinators, requests_per_second=rate,
            stop_after_target_successes=target_successes)

    def test_mixed_groups_keep_source_sample_ids_and_exact_sizes(self):
        plan = [dict(workload_index=i, samples=n) for i, n in enumerate((1, 4, 5))]
        with tempfile.TemporaryDirectory() as td, LocalServer(4, delay=.03) as server:
            out = Path(td)/'mixed'
            s = self.call(out, server, plan)
            self.assertEqual(s['requests'], 10)
            self.assertEqual(s['group_summary']['complete_groups'], 3)
            self.assertEqual(s['group_summary']['peak_coordinators'], 3)
            self.assertEqual(s['peak_wire_inflight'], 4)
            self.assertEqual(server.received, 10)
            self.assertTrue(s['integrity']['ok'])
            rows = events(out, 'response')
            for group_no, n in enumerate((1, 4, 5)):
                group = [r for r in rows if r['group_no'] == group_no]
                self.assertEqual(sorted(r['sample_no'] for r in group), list(range(n)))
                self.assertEqual({r['source_event_id'] for r in group}, {n})
            self.assertEqual(sorted(r['request_no'] for r in rows), list(range(10)))
            self.assertEqual(s['outstanding'], 0)
            from scripts.deepeye_efficiency_report import verified_summary
            verified_summary(out)  # JSON export must exactly match authoritative finish.
            from scripts.deepeye_efficiency_report import analyze
            self.assertTrue(analyze(out)['pairing_ok'])

    def test_stop_releases_unsent_samples_without_deadlocking_coordinators(self):
        plan = [dict(workload_index=2, samples=5) for _ in range(12)]
        with tempfile.TemporaryDirectory() as td, LocalServer(1) as server:
            s = self.call(Path(td)/'stop', server, plan, requests=7, coordinators=2)
            self.assertEqual(s['requests'], 7)
            self.assertEqual(server.received, 7)
            self.assertEqual(s['group_summary']['complete_groups'], 1)
            self.assertLessEqual(s['group_summary']['peak_coordinators'], 2)
            self.assertEqual(s['group_summary']['active_at_finish'], 0)
            self.assertEqual(s['outstanding'], 0)
            self.assertTrue(s['integrity']['ok'])

    def test_group_deadline_does_not_send_unadmitted_samples(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(1) as server:
            s = self.call(Path(td)/'deadline', server,
                          [dict(workload_index=2, samples=5)]*5,
                          deadline=.3, rate=2)
            self.assertLessEqual(server.received, 1)
            self.assertEqual(s['stop_reason'], 'admission_deadline')
            self.assertEqual(s['outstanding'], 0)
            self.assertEqual(s['group_summary']['active_at_finish'], 0)

    def test_budget_stop_reason_is_not_relabelled_during_slow_drain(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(1, delay=.6) as server:
            s = self.call(Path(td)/'slow_drain', server,
                          [dict(workload_index=0, samples=1)]*3,
                          requests=1, deadline=.15)
            self.assertEqual(server.received, 1)
            self.assertEqual(s['requests'], 1)
            self.assertEqual(s['stop_reason'], 'request_budget_completed')
            self.assertEqual(s['outstanding'], 0)

    def test_failed_samples_are_not_retried_or_counted_as_complete(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(1, status=500) as server:
            s = self.call(Path(td)/'errors', server, [dict(workload_index=1, samples=4)])
            self.assertEqual(server.received, 4)
            self.assertEqual(s['group_summary']['complete_groups'], 0)
            self.assertEqual(s['group_summary']['failed_groups'], 1)
            self.assertEqual(s['group_summary']['unfinished_groups'], 0)

    def test_short_remote_policy_requires_target_then_new_successful_returns(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(4, delay=.05) as server:
            s = self.call(Path(td)/'target', server,
                          [dict(workload_index=2, samples=5)]*20,
                          requests=100, target_successes=4)
            self.assertEqual(s['stop_reason'], 'target_reached_and_returns_confirmed')
            self.assertTrue(s['target_confirmation']['reached'])
            self.assertGreaterEqual(s['target_confirmation']['successful_returns_after_target'], 4)
            self.assertLess(s['requests'], 100)
            self.assertEqual(s['outstanding'], 0)

    def test_invalid_group_plan_fails_before_creating_any_output(self):
        with tempfile.TemporaryDirectory() as td, LocalServer(1) as server:
            for plan in ([], [dict(workload_index=9, samples=4)],
                         [dict(workload_index=0, samples=0)]):
                out = Path(td)/'bad'
                with self.assertRaises(ValueError):
                    self.call(out, server, plan)
                self.assertFalse(out.exists())
            self.assertEqual(server.received, 0)


if __name__ == '__main__':
    unittest.main()
