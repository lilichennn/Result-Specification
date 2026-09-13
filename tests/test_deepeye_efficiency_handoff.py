"""Handoff admission must count old calls and fail closed; no remote requests."""
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.efficiency_probe import core
from scripts.efficiency_probe.local import LocalServer


def fixture_source(path, count=4, issued=None, parent=None):
    manifest = {'kind': 'prechange_sdk_capacity_probe', 'budget': {'max_requests': count},
                'model': 'fixture', 'endpoint': 'local', 'workload_sha256': 'fixture'}
    if parent:
        manifest['handoff'] = {'source_run': str(parent)}
    store = core.RunStore.create(path, manifest)
    attempt = store.begin_attempt('capacity', 'probe', 'fixture')
    for i in range(count if issued is None else issued):
        store.append_event(attempt, 'request', {'request_no': i})
    return store, attempt


class HandoffTests(unittest.TestCase):
    def test_corrupt_manifest_or_consumed_event_is_rejected(self):
        for target in ('manifest', 'payload', 'record'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as td:
                path = Path(td)/'old'
                store, _ = fixture_source(path, count=1)
                store.close()
                with sqlite3.connect(path/'run.sqlite3') as db:
                    # Test-only corruption, bypassing the normal writer guard.
                    db.execute('drop trigger manifest_no_update' if target == 'manifest'
                               else 'drop trigger events_no_update')
                    if target == 'manifest':
                        db.execute("update manifest set payload_checksum='corrupt'")
                    else:
                        column = 'payload_json' if target == 'payload' else 'record_checksum'
                        value = '{"request_no": 0, "corrupt": true}' if target == 'payload' else 'corrupt'
                        db.execute(f'update events set {column}=?', (value,))
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    self.reader(path)

    def test_corrupt_new_terminal_cannot_release_a_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'old'
            store, a = fixture_source(path, count=1)
            try:
                with self.reader(path) as reader:
                    store.append_event(a, 'response', {'request_no': 0})
                    with sqlite3.connect(path/'run.sqlite3') as db:
                        db.execute('drop trigger events_no_update')
                        db.execute("update events set record_checksum='corrupt' where kind='response'")
                    with self.assertRaisesRegex(ValueError, 'checksum'):
                        reader.pending()
            finally:
                store.close()

    def test_slow_handoff_logging_cannot_create_a_catch_up_submission_burst(self):
        from datetime import datetime
        import sqlite3
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, delay=.001) as server:
            old, new = Path(td)/'old', Path(td)/'new'
            store, a = fixture_source(old, count=1)
            store.append_event(a, 'response', {'request_no': 0})
            original = core.RunStore.append_event

            def slow_log(writer, attempt, kind, payload, *args, **kwargs):
                if kind == 'handoff_admission' and payload['request_no'] == 0:
                    time.sleep(.12)
                return original(writer, attempt, kind, payload, *args, **kwargs)

            try:
                with patch.object(core.RunStore, 'append_event', slow_log):
                    s = core.run_requests(new, [{'stage': 'sql_generation', 'parser': 'sql',
                        'messages': [{'role': 'user', 'content': 'fixture'}], 'prompt_sha256': 'fixture',
                        'source_event_id': 1}], core.ProbeBudget(4, 8, 10, 100000, 2),
                        server.url, 'local-only', 'fixture', requests_per_second=50, draining_run=old)
                self.assertEqual(s['http_success'], 8)
                with sqlite3.connect(new/'run.sqlite3') as db:
                    stamps = sorted(datetime.fromisoformat(json.loads(r[0])['submitted_at']).timestamp()
                        for r in db.execute("select payload_json from events where kind='request'"))
                self.assertGreaterEqual(min(b-a for a,b in zip(stamps, stamps[1:])), .016)
            finally:
                store.close()

    def reader(self, path):
        self.assertTrue(hasattr(core, 'DrainingProbe'), 'missing draining-source admission reader')
        return core.DrainingProbe(path)

    def test_completed_source_requests_release_capacity_once(self):
        # Ignoring error terminals or double-releasing a terminal must break this test.
        with tempfile.TemporaryDirectory() as td:
            store, a = fixture_source(Path(td)/'old')
            try:
                with self.reader(Path(td)/'old') as reader:
                    self.assertEqual(reader.pending(), 4)
                    store.append_event(a, 'response', {'request_no': 0})
                    self.assertEqual(reader.pending(), 3)
                    store.append_event(a, 'error', {'request_no': 1})
                    self.assertEqual(reader.pending(), 2)
                    self.assertEqual(reader.pending(), 2)
                    store.append_event(a, 'error', {'request_no': 1})
                    with self.assertRaises(ValueError):
                        reader.pending()
            finally:
                store.close()

    def test_every_unfinished_ancestor_counts_against_the_shared_limit(self):
        # Counting only the direct predecessor would permit excess total calls.
        with tempfile.TemporaryDirectory() as td:
            first = Path(td)/'first'
            second = Path(td)/'second'
            a_store, a = fixture_source(first, count=3)
            b_store, b = fixture_source(second, count=4, parent=first)
            try:
                a_store.append_event(a, 'response', {'request_no': 0})
                b_store.append_event(b, 'response', {'request_no': 0})
                with self.reader(second) as reader:
                    self.assertEqual(reader.pending(), 5)
                    a_store.append_event(a, 'error', {'request_no': 1})
                    self.assertEqual(reader.pending(), 4)
                    b_store.append_event(b, 'response', {'request_no': 1})
                    self.assertEqual(reader.pending(), 3)
            finally:
                a_store.close()
                b_store.close()

    def test_cyclic_handoff_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            first, second = Path(td)/'first', Path(td)/'second'
            a_store, _ = fixture_source(first, parent=second)
            b_store, _ = fixture_source(second, parent=first)
            try:
                with self.assertRaises(ValueError):
                    self.reader(second)
            finally:
                a_store.close()
                b_store.close()

    def test_source_must_have_exhausted_its_admission_budget(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = fixture_source(Path(td)/'old', issued=3)
            try:
                with self.assertRaises(ValueError):
                    self.reader(Path(td)/'old')
            finally:
                store.close()

    def test_unexpected_new_source_request_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            store, a = fixture_source(Path(td)/'old')
            try:
                with self.reader(Path(td)/'old') as reader:
                    store.append_event(a, 'request', {'request_no': 4})
                    with self.assertRaises(ValueError):
                        reader.pending()
            finally:
                store.close()

    def test_real_sdk_waits_for_old_capacity_and_records_total_reservations(self):
        # Removing subtraction of the old pending count permits calls before release.
        self.assertTrue(hasattr(core, 'DrainingProbe'), 'missing handoff implementation')
        with tempfile.TemporaryDirectory() as td, LocalServer(target=1, delay=.3) as server:
            old = Path(td)/'old'
            store, a = fixture_source(old)
            errors = []

            def release():
                try:
                    deadline = time.monotonic()+5
                    while not (Path(td)/'new'/'manifest.json').exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    if not (Path(td)/'new'/'manifest.json').exists():
                        errors.append('new probe failed to finish admission initialization')
                    time.sleep(.2)
                    if server.received:
                        errors.append('new HTTP sent while old calls used all four slots')
                    store.append_event(a, 'response', {'request_no': 0})
                    deadline = time.monotonic()+3
                    while server.received == 0 and time.monotonic() < deadline:
                        time.sleep(.01)
                    if server.received != 1:
                        errors.append('expected exactly one freed slot to be used')
                    time.sleep(.08)
                    if server.received != 1:
                        errors.append('more than one new call while three old calls pending')
                finally:
                    for i in range(1, 4):
                        store.append_event(a, 'response', {'request_no': i})

            t = threading.Thread(target=release)
            t.start()
            try:
                s = core.run_requests(Path(td)/'new', [{'stage': 'sql_generation', 'parser': 'sql',
                    'messages': [{'role': 'user', 'content': 'fixture'}], 'prompt_sha256': 'fixture',
                    'source_event_id': 1}], core.ProbeBudget(4, 8, 10, 100000, 2), server.url,
                    'local-fixture', 'fixture', workers=4, requests_per_second=50, draining_run=old)
                self.assertEqual(s['http_success'], 8)
                self.assertEqual(s['handoff']['source_pending_at_start'], 4)
                self.assertLessEqual(s['handoff']['peak_reserved_total'], 4)
                self.assertTrue(s['integrity']['ok'])
            finally:
                t.join(5)
                store.close()
            self.assertFalse(t.is_alive())
            self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
