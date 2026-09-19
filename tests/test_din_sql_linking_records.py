import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.baseline_adapters.din_sql.inputs import TaskKey


NODES = ("schema_filter_rc3", "linking_rc3")


def records_api():
    try:
        from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
    except ModuleNotFoundError as exc:
        raise AssertionError("DIN Linking durable records are missing") from exc
    return LinkingRecords


def manifest(ids=("0", "1")):
    return {
        "format": "din-sql-linking-v1",
        "batch_id": "fixture",
        "groups": {"bird_dev": {"ids": list(ids)}},
    }


def terminal(status="succeeded", **extra):
    value = {
        "status": status,
        "result": {"ok": True} if status == "succeeded" else None,
        "error": None if status == "succeeded" else {"category": "fixture"},
        "input_fingerprint": "fixture",
        "response_ref": None,
        "parent_refs": {},
    }
    value.update(extra)
    return value


class LinkingRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "batch"
        self.records = records_api()(self.root, manifest())

    def tearDown(self):
        self.records.close()
        self.tmp.cleanup()

    def test_two_terminal_nodes_seal_and_survive_reopen(self):
        key = TaskKey("bird_dev", "0")
        version = self.records.begin(key)
        first = self.records.save_node(version, NODES[0], terminal())
        self.records.save_node(
            version,
            NODES[1],
            terminal(parent_refs={NODES[0]: first}),
        )
        self.assertIsNone(self.records.current(key))
        self.records.seal(version)
        self.assertEqual(self.records.current(key), version)

        self.records.close()
        self.records = records_api()(self.root, manifest())
        self.assertEqual(self.records.current(key), version)
        self.assertEqual(self.records.node(version, NODES[1])["status"], "succeeded")
        self.assertEqual(len(self.records.current_rows()), 1)

    def test_group_run_store_uses_the_frozen_batch_format(self):
        self.assertEqual(
            self.records.stores["bird_dev"].manifest["format"],
            "din-sql-linking-v1",
        )

    def test_filter_failure_still_requires_dependency_failed_linking(self):
        version = self.records.begin(TaskKey("bird_dev", "0"))
        filter_ref = self.records.save_node(version, NODES[0], terminal("failed"))
        self.records.save_node(
            version,
            NODES[1],
            terminal(
                "dependency_failed",
                error={"category": "dependency_failed"},
                parent_refs={NODES[0]: filter_ref},
            ),
        )
        self.records.seal(version)
        self.assertEqual(self.records.rows[version]["status"], "failed")

    def test_request_history_indexes_input_attempt_and_outcome_refs(self):
        version = self.records.begin(TaskKey("bird_dev", "0"))
        input_ref = self.records.append(
            version,
            "node_input",
            {"node": NODES[0], "kwargs": {"messages": ["large"]}, "input_fingerprint": "abc"},
        )
        self.records.append(version, "request_attempt", {"node": NODES[0], "attempt_no": 1})
        self.records.append(
            version,
            "request_outcome",
            {"node": NODES[0], "attempt_no": 1, "status": "failed"},
        )
        history = self.records.request_history(version)
        self.assertEqual(history["inputs"][NODES[0]]["ref"], input_ref)
        self.assertEqual(history["attempts"][NODES[0]][0]["attempt_no"], 1)
        self.assertEqual(history["outcomes"][NODES[0]][0]["status"], "failed")
        self.assertEqual(self.records.read_ref(input_ref)["input_fingerprint"], "abc")

    def test_rejects_unknown_duplicate_or_incomplete_terminal_state(self):
        version = self.records.begin(TaskKey("bird_dev", "0"))
        with self.assertRaises(ValueError):
            self.records.save_node(version, "linking", terminal())
        self.records.save_node(version, NODES[0], terminal())
        with self.assertRaises(ValueError):
            self.records.save_node(version, NODES[0], terminal("failed"))
        with self.assertRaises(ValueError):
            self.records.seal(version)

    def test_latest_version_changes_only_after_new_version_is_sealed(self):
        key = TaskKey("bird_dev", "0")
        old = self.records.begin(key)
        old_filter = self.records.save_node(old, NODES[0], terminal())
        self.records.save_node(
            old,
            NODES[1],
            terminal(parent_refs={NODES[0]: old_filter}),
        )
        self.records.seal(old)
        new = self.records.begin(key, parent_version=old)
        self.assertEqual(self.records.current(key), old)
        self.assertEqual(self.records.unfinished(key), new)
        new_filter = self.records.save_node(new, NODES[0], terminal("failed"))
        self.records.save_node(
            new,
            NODES[1],
            terminal(
                "dependency_failed",
                parent_refs={NODES[0]: new_filter},
            ),
        )
        self.records.seal(new)
        self.assertEqual(self.records.current(key), new)

    def test_public_views_are_copies_and_seal_rebuilds_from_durable_events(self):
        """Removing defensive copies or trusting the cache must break this test."""

        version = self.records.begin(TaskKey("bird_dev", "0"))
        filter_ref = self.records.save_node(version, NODES[0], terminal())
        self.records.save_node(
            version,
            NODES[1],
            terminal(parent_refs={NODES[0]: filter_ref}),
        )

        self.records.node(version, NODES[0])["status"] = "failed"
        leaked = self.records.view(version)
        leaked["nodes"][NODES[1]]["status"] = "failed"
        self.records.seal(version)

        self.assertEqual(self.records.rows[version]["status"], "succeeded")
        self.assertEqual(self.records.node(version, NODES[0])["status"], "succeeded")
        self.assertEqual(self.records.node(version, NODES[1])["status"], "succeeded")

    def test_request_history_is_a_copy(self):
        """A caller must not be able to rewrite cached request provenance."""

        version = self.records.begin(TaskKey("bird_dev", "0"))
        self.records.append(
            version,
            "node_input",
            {"node": NODES[0], "kwargs": {"model": "fixture"}, "input_fingerprint": "abc"},
        )
        history = self.records.request_history(version)
        history["inputs"][NODES[0]]["input_fingerprint"] = "changed"
        self.assertEqual(
            self.records.request_history(version)["inputs"][NODES[0]]["input_fingerprint"],
            "abc",
        )

    def test_seal_rejects_impossible_dependency_topology(self):
        """A successful Linking node may not follow a failed schema filter."""

        version = self.records.begin(TaskKey("bird_dev", "0"))
        filter_ref = self.records.save_node(version, NODES[0], terminal("failed"))
        self.records.save_node(
            version,
            NODES[1],
            terminal(parent_refs={NODES[0]: filter_ref}),
        )
        with self.assertRaisesRegex(ValueError, "dependency"):
            self.records.seal(version)

    def test_seal_rejects_missing_or_wrong_parent_ref(self):
        """Changing the Linking parent ref must make the durable chain invalid."""

        version = self.records.begin(TaskKey("bird_dev", "0"))
        self.records.save_node(version, NODES[0], terminal())
        self.records.save_node(version, NODES[1], terminal(parent_refs={}))
        with self.assertRaisesRegex(ValueError, "parent"):
            self.records.seal(version)

    def test_begin_is_atomic_for_one_question(self):
        """Two threads may not both create an unfinished version for one task."""

        key = TaskKey("bird_dev", "0")
        store = self.records.stores[key.group]
        original = store.begin_attempt
        first_entered = threading.Event()
        release_first = threading.Event()
        call_count = 0
        call_lock = threading.Lock()

        def delayed_begin(*args, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                position = call_count
            if position == 1:
                first_entered.set()
                release_first.wait(2)
            return original(*args, **kwargs)

        results = []
        errors = []

        def begin():
            try:
                results.append(self.records.begin(key))
            except Exception as exc:  # the second caller must be rejected
                errors.append(exc)

        with patch.object(store, "begin_attempt", side_effect=delayed_begin):
            first = threading.Thread(target=begin)
            second = threading.Thread(target=begin)
            first.start()
            self.assertTrue(first_entered.wait(1))
            second.start()
            time.sleep(0.05)
            release_first.set()
            first.join(2)
            second.join(2)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(self.records.unfinished(key), results[0])


if __name__ == "__main__":
    unittest.main()
