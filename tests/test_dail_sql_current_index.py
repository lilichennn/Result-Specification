import concurrent.futures
from pathlib import Path
import tempfile
import sqlite3
import unittest

from scripts.baseline_adapters.dail_sql.config import TaskKey
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex


class CurrentIndexTests(unittest.TestCase):
    def test_prepared_group_binding_is_immutable_and_readonly_safe(self):
        self.assertTrue(hasattr(self.index, 'bind_prepared_group'), 'group binding missing')
        self.index.bind_prepared_group('batch', 'a', 'first')
        self.index.bind_prepared_group('batch', 'a', 'first')
        with self.assertRaises(ValueError):
            self.index.bind_prepared_group('batch', 'a', 'changed')
        with CurrentIndex(self.path, read_only=True) as reader:
            self.assertEqual(reader.prepared_group('batch', 'a'), 'first')
            self.assertIsNone(reader.prepared_group('batch', 'b'))
            with self.assertRaises(PermissionError):
                reader.bind_prepared_group('batch', 'b', 'second')

    def test_assignment_claim_and_finish_rollback_as_one_transaction(self):
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER deny_assignment BEFORE INSERT ON campaign_assignments BEGIN SELECT RAISE(ABORT, 'crash'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.index.claim_assignment(self.key, 'v1')
        self.assertIsNone(self.index.assignment(self.key))
        self.assertEqual(self.index.snapshot(), {})
        with sqlite3.connect(self.path) as db:
            db.execute('DROP TRIGGER deny_assignment')
        lease = self.index.claim_assignment(self.key, 'v1')
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER deny_finish BEFORE UPDATE ON campaign_assignments BEGIN SELECT RAISE(ABORT, 'crash'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.index.finish_assignment(self.key, 'v1', lease, lambda k, v: True)
        self.assertIsNone(self.index.current(self.key))
        self.assertEqual(self.index.assignment(self.key)['lease'], lease)
        self.assertEqual(self.index.assignment(self.key)['state'], 'active')

    def test_assignment_recovery_is_durable_and_first_version_survives_rerun(self):
        self.assertTrue(hasattr(self.index, 'claim_assignment'), 'campaign assignment missing')
        lease = self.index.claim_assignment(self.key, 'first')
        with CurrentIndex(self.path) as reopened:
            fresh = reopened.recover_assignment(self.key)
            self.assertNotEqual(lease['token'], fresh['token'])
        self.assertEqual(self.index.assignment(self.key)['lease'], fresh)
        self.index.record_campaign_mode(self.key, 'first', 'native', 'event', fresh, 'failed')
        with self.assertRaises(ValueError):
            self.index.finish_assignment(self.key, 'first', lease, lambda k, v: True)
        with self.assertRaises(ValueError):
            self.index.finish_assignment(self.key, 'first', fresh, lambda k, v: False)
        self.index.finish_assignment(self.key, 'first', fresh, lambda k, v: True)
        newer = self.index.claim_assignment(self.key, 'second')
        self.assertEqual(self.index.current(self.key), 'first')
        self.assertEqual(self.index.assignment(self.key)['first_version'], 'first')
        self.index.record_campaign_mode(self.key, 'second', 'native', 'new-event', newer, 'succeeded')
        self.assertEqual(self.index.campaign_counts('batch', 'spider_dev')['terminal']['native'], 1)
        self.assertEqual(self.index.campaign_counts('batch', 'spider_dev')['failed']['native'], 1)
        with CurrentIndex(self.path, read_only=True) as reader:
            self.assertEqual(reader.assignment(self.key)['version'], 'second')
            with self.assertRaises(PermissionError):
                reader.recover_assignment(self.key)

    def test_read_only_snapshot_does_not_initialize_or_mutate(self):
        self.assertIn('read_only', __import__('inspect').signature(CurrentIndex).parameters)
        self.publish(self.key, 'old')
        with CurrentIndex(self.path, read_only=True) as reader:
            self.assertEqual(reader.snapshot(), {self.key: 'old'})
            with self.assertRaises(PermissionError):
                reader.claim(self.key)
            with self.assertRaises(PermissionError):
                reader.mark_group_started('batch', 'bird_dev')
            def never_verify(key, version):
                self.fail('read-only publication invoked a writer callback')
            with self.assertRaises(PermissionError):
                reader.publish(self.key, 'other', {}, never_verify)
            self.publish(self.key, 'new')
            self.assertEqual(reader.snapshot(), {self.key: 'new'})
        missing = self.path.parent / 'missing' / 'current.sqlite3'
        with self.assertRaises(FileNotFoundError):
            CurrentIndex(missing, read_only=True)
        self.assertFalse(missing.parent.exists())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "current.sqlite3"
        self.index = CurrentIndex(self.path)
        self.addCleanup(self.index.close)
        self.key = TaskKey("batch", "spider_dev", "1")

    def publish(self, key, version):
        self.index.publish(key, version, self.index.claim(key), lambda k, v: True)

    def test_replace_one_question_preserves_neighbors(self):
        keys = [TaskKey("batch", "spider_dev", str(i)) for i in range(3)]
        for key in keys:
            self.publish(key, "old-" + key.question_id)
        before = self.index.snapshot()
        self.publish(keys[1], "new-1")
        self.assertEqual(self.index.current(keys[1]), "new-1")
        self.assertEqual(self.index.current(keys[0]), before[keys[0]])
        self.assertEqual(self.index.current(keys[2]), before[keys[2]])

    def test_empty_and_cross_group_and_batch_keys(self):
        self.assertEqual(self.index.snapshot(), {})
        for key in (self.key, TaskKey("batch", "bird_dev", "1"), TaskKey("other", "spider_dev", "1")):
            self.publish(key, key.batch_id + key.group)
        self.assertEqual(len(self.index.snapshot()), 3)
        self.assertEqual(len(self.index.snapshot("bird_dev")), 1)

    def test_claims_are_exclusive_and_reopen_does_not_steal(self):
        lease = self.index.claim(self.key)
        with CurrentIndex(self.path) as other:
            with self.assertRaises(ValueError):
                other.claim(self.key)
            fresh = other.recover(self.key, lease)
            with self.assertRaises(ValueError):
                self.index.publish(self.key, "late", lease, lambda k, v: True)
            with self.assertRaises(ValueError):
                self.index.release(self.key, lease)
            other.publish(self.key, "new", fresh, lambda k, v: True)
        self.assertEqual(self.index.current(self.key), "new")

    def test_unsealed_and_stale_epoch_or_version_cannot_publish(self):
        lease = self.index.claim(self.key)
        self.assertEqual(self.index.snapshot(), {self.key: None})
        with self.assertRaises(ValueError):
            self.index.publish(self.key, "bad", lease, lambda k, v: False)
        for field, value in (("expected_epoch", 8), ("expected_version", "wrong"), ("token", "old")):
            with self.assertRaises(ValueError):
                self.index.publish(self.key, "bad", {**lease, field: value}, lambda k, v: True)
        self.index.release(self.key, lease)
        self.publish(self.key, "good")
        with self.assertRaises(ValueError):
            self.index.publish(self.key, "late", lease, lambda k, v: True)

    def test_different_question_parallel_publish_preserves_both(self):
        keys = [TaskKey("batch", "spider_dev", str(i)) for i in range(12)]
        def worker(key):
            with CurrentIndex(self.path) as index:
                index.publish(key, key.question_id, index.claim(key), lambda k, v: True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, keys))
        self.assertEqual(self.index.snapshot(), {key: key.question_id for key in keys})

    def test_failure_between_update_and_commit_rolls_back_pointer_and_lease(self):
        self.publish(self.key, "old")
        lease = self.index.claim(self.key)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER inject_failure BEFORE DELETE ON leases BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.index.publish(self.key, "new", lease, lambda k, v: True)
        self.assertEqual(self.index.current(self.key), "old")
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TRIGGER inject_failure")
        self.index.publish(self.key, "new", lease, lambda k, v: True)
        self.assertEqual(self.index.current(self.key), "new")

    def test_state_pointers_are_idempotent_fenced_and_durable(self):
        lease = self.index.claim(self.key)
        self.index.record_mode(self.key, "v1", "native", "e1", lease)
        self.index.record_mode(self.key, "v1", "native", "e1", lease)
        with self.assertRaises(ValueError):
            self.index.record_mode(self.key, "v1", "native", "different", lease)
        self.index.mark_group_started("batch", "spider_dev")
        self.index.mark_group_started("batch", "spider_dev")
        with CurrentIndex(self.path) as other:
            self.assertEqual(other.mode_states(self.key, "v1"), {"native": "e1"})
            self.assertTrue(other.group_started("batch", "spider_dev"))
        self.index.release(self.key, lease)
        with self.assertRaises(ValueError):
            self.index.record_mode(self.key, "v1", "rc_first", "e2", lease)
