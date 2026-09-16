"""Four-mode orchestration and offline production round-chain contracts."""
import asyncio
import copy
import importlib
import importlib.util
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import openai  # Keep SDK import outside the short-deadline async fixtures.

from scripts.baseline_adapters.dail_sql import retrieval  # Likewise load sklearn before scheduling.
from scripts.baseline_adapters.dail_sql.config import DailSettings, TaskKey
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.dail_sql.transport import GroupRequester
from scripts.baseline_adapters.shared.transport import RequestStopped
from tests.test_dail_sql_prompts import SCHEMA


class MemoryRecords:
    """Orchestration double; production integration below uses DailRecords."""
    def __init__(self):
        self.events = []
        self.sealed = False

    def find_source(self, version, kind, identity):
        field = {"round_result": "round_execution_id", "mode_result": "mode"}[kind]
        return next((copy.deepcopy(e) for e in self.events
                     if e["kind"] == kind and e["payload"][field] == identity), None)

    def append(self, version, kind, payload):
        event_id = f"{version}#{len(self.events)}"
        self.events.append({"event_id": event_id, "kind": kind, "payload": copy.deepcopy(payload)})
        return event_id

    def seal(self, version, modes):
        assert len(modes) == 4
        self.sealed = True


class CompositeTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_save_and_seal_leave_loop_responsive_and_drain_cancellation(self):
        for operation in ("append", "seal"):
            with self.subTest(operation=operation):
                self.records = MemoryRecords()
                entered, release = threading.Event(), threading.Event()
                original = getattr(self.records, operation)
                main_thread = threading.get_ident()
                def slow(*args):
                    if operation == "seal" or args[1] == "mode_result":
                        self.assertNotEqual(threading.get_ident(), main_thread)
                        entered.set()
                        release.wait(2)
                    return original(*args)
                for gate in self.first_release + self.second_release:
                    gate.set()
                with patch.object(self.records, operation, side_effect=slow):
                    pending = self.start()
                    try:
                        for _ in range(200):
                            if entered.is_set() or pending.done():
                                break
                            await asyncio.sleep(.001)
                        self.assertTrue(entered.is_set())
                        pending.cancel()
                        await asyncio.sleep(.01)
                        self.assertFalse(pending.done(), "Owned persistence must drain before cancellation")
                    finally:
                        release.set()
                        await asyncio.gather(pending, return_exceptions=True)
                self.assertTrue(any(e["kind"] == "mode_result" for e in self.records.events))
                self.assertEqual(self.records.sealed, operation == "seal")

    def setUp(self):
        name = "scripts.baseline_adapters.dail_sql.composite"
        self.assertIsNotNone(importlib.util.find_spec(name), "Task6 composite is missing")
        self.api = importlib.import_module(name)
        self.records = MemoryRecords()
        self.calls = []
        self.first_ids = [f"initial{i}" for i in range(9)]
        self.next_ids = [f"next{i}" for i in range(9)]
        self.first_release = [asyncio.Event(), asyncio.Event()]
        self.second_release = [asyncio.Event(), asyncio.Event()]
        self.first_status = ["success", "success"]
        self.different = False
        self.started = asyncio.Queue()

    async def round(self, number, rc, ids, *, parent_round_id=None):
        label = ("B" if rc else "A") if number == 1 else (
            ("F" if rc else "E") if ids != self.next_ids else ("D" if rc else "C"))
        self.calls.append((label, list(ids), parent_round_id))
        self.started.put_nowait(label)
        await (self.first_release if number == 1 else self.second_release)[int(rc)].wait()
        status = self.first_status[int(rc)] if number == 1 else "success"
        result = {"round_execution_id": label, "round_no": number, "rc_injected": rc,
                  "actual_parent_round_id": parent_round_id, "example_ids": list(ids),
                  "status": status, "selection": {"candidate_id": label + "0"},
                  "next_example_ids": (list(reversed(self.next_ids)) if rc and self.different else self.next_ids) if number == 1 else [],
                  "error": None if status == "success" else {"category": "samples_failed"}}
        self.records.append("v", "round_result", result)
        return result

    def start(self, callback=None):
        return asyncio.create_task(self.api.run_composite({"first_example_ids": self.first_ids},
            version_id="v", run_round=self.round, records=self.records, on_mode_result=callback))

    async def take(self, count):
        return {await asyncio.wait_for(self.started.get(), 2) for _ in range(count)}

    async def test_order_and_rc_condition_are_part_of_reuse_key(self):
        key = self.api.second_round_key
        self.assertEqual(key(["a", "b"], False), (("a", "b"), False))
        self.assertNotEqual(key(["a", "b"], False), key(["b", "a"], False))
        self.assertNotEqual(key(["a", "b"], False), key(["a", "b"], True))

    async def test_b_first_waits_for_a_then_four_actual_rounds_keep_a_parent(self):
        pending = self.start()
        self.assertEqual(await self.take(2), {"A", "B"})
        self.first_release[1].set()
        await asyncio.sleep(.01)
        self.assertEqual(len(self.calls), 2)
        self.first_release[0].set()
        self.assertEqual(await self.take(2), {"C", "D"})
        for gate in self.second_release:
            gate.set()
        result = await pending
        self.assertEqual(len(self.calls), 4)
        self.assertEqual({m: (r["first_round_id"], r["second_round_id"]) for m, r in result["modes"].items()},
                         {"native": ("A", "C"), "rc_first": ("B", "C"), "rc_second": ("A", "D"), "rc_both": ("B", "D")})
        self.assertEqual([c[2] for c in self.calls[2:]], ["A", "A"])
        self.assertTrue(self.records.sealed)

    async def test_inflight_reuse_and_incremental_modes_before_other_rc_round_finishes(self):
        notifications = []
        async def notified(mode, event_id):
            self.assertEqual(self.records.find_source("v", "mode_result", mode)["event_id"], event_id)
            notifications.append(mode)
        pending = self.start(notified)
        await self.take(2)
        self.first_release[0].set()
        self.assertEqual(await self.take(2), {"C", "D"})
        self.first_release[1].set()
        self.second_release[0].set()
        for _ in range(100):
            if len(notifications) == 2:
                break
            await asyncio.sleep(.001)
        self.assertEqual(set(notifications), {"native", "rc_first"})
        self.assertFalse(pending.done())
        self.assertEqual(len(self.calls), 4)
        self.second_release[1].set()
        await pending

    async def test_different_order_requires_six_rounds_and_b_parents(self):
        self.different = True
        for gate in self.first_release + self.second_release:
            gate.set()
        result = await self.start()
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(result["modes"]["rc_first"]["second_round_id"], "E")
        self.assertEqual(result["modes"]["rc_both"]["second_round_id"], "F")
        self.assertEqual({label: parent for label, _, parent in self.calls},
                         {"A": None, "B": None, "C": "A", "D": "A", "E": "B", "F": "B"})

    async def test_a_failure_leaves_b_independent_and_both_failures_end_without_seconds(self):
        for statuses, expected_calls in ((["failed", "success"], 4), (["failed", "failed"], 2)):
            with self.subTest(statuses=statuses):
                self.first_status = statuses
                self.calls.clear()
                self.records = MemoryRecords()
                for gate in self.first_release + self.second_release:
                    gate.set()
                result = await self.start()
                self.assertEqual(len(self.calls), expected_calls)
                self.assertEqual(result["modes"]["native"]["status"], "dependency_failed")
                if statuses[1] == "success":
                    self.assertEqual(result["modes"]["rc_first"]["status"], "succeeded")
                    self.assertEqual([c[2] for c in self.calls[2:]], ["B", "B"])
                else:
                    self.assertTrue(all(m["status"] == "dependency_failed" for m in result["modes"].values()))

    async def test_cancelled_composite_drains_owned_work_and_does_not_seal_failures(self):
        pending = self.start()
        await self.take(2)
        self.first_release[0].set()
        await self.take(2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(self.records.sealed)
        self.assertFalse(any(e["kind"] == "mode_result" for e in self.records.events))

    async def test_cancelled_mode_waiter_leaves_other_shared_round_consumers_running(self):
        pending = self.start()
        await self.take(2)
        self.first_release[0].set()
        await self.take(2)
        self.first_release[1].set()
        await asyncio.sleep(.01)
        # The production API deliberately does not expose per-mode task handles.
        # Locate this one internal waiter solely to inject cancellation in flight.
        waiter = next(t for t in asyncio.all_tasks() if t.get_coro().cr_frame is not None
                      and t.get_coro().__name__ == "finish_mode"
                      and t.get_coro().cr_frame.f_locals.get("mode") == "native")
        waiter.cancel()
        for gate in self.second_release:
            gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        modes = {e["payload"]["mode"]: e["payload"] for e in self.records.events if e["kind"] == "mode_result"}
        self.assertEqual(set(modes), {"rc_first", "rc_second", "rc_both"})
        self.assertTrue(all(mode["status"] == "succeeded" for mode in modes.values()))
        self.assertEqual(modes["rc_first"]["second_round_id"], "C")
        self.assertEqual(len(self.calls), 4)
        self.assertFalse(self.records.sealed)


class OfflineDispatcher:
    """Replace only HTTP; retain real five-slot requester and all persistence."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.calls = []
        self.failed_positions = set()
        self.pause = False
        self.raw = "SELECT DISTINCT id FROM t"
        self.different = False

    def stop(self, **kwargs):
        self.stop_event.set()

    async def call_chat(self, client, *, identity, sdk_kwargs, on_started, on_finished):
        on_started()
        self.calls.append((dict(identity), copy.deepcopy(sdk_kwargs)))
        await asyncio.sleep((4 - identity["sample_position"]) * .001)
        content = sdk_kwargs["messages"][-1]["content"]
        second = "CACHED_FIRST" not in content
        rc = "<result_contract>" in content
        if self.pause and second and not rc and identity["sample_position"] == 2:
            self.stop_event.set()
            raise RequestStopped("offline batch pause")
        if second and not rc and identity["sample_position"] in self.failed_positions:
            raise TimeoutError("offline timeout")
        raw = "SELECT COUNT(*) FROM t" if self.different and not second and rc else self.raw
        return {"choices": [{"index": 7, "message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


class ProductionRoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_sql_pool_does_not_block_lookup_prompt_or_request_dispatch(self):
        release = threading.Event()
        blockers = [self.executor.submit(release.wait, 3) for _ in range(2)]
        pending = asyncio.create_task(self.runner()(1, False, self.ids[:9]))
        try:
            for _ in range(300):
                if len(self.dispatcher.calls) == 5:
                    break
                await asyncio.sleep(.001)
            self.assertEqual(len(self.dispatcher.calls), 5)
            self.assertFalse(pending.done(), "SQL selection should still wait for its own pool")
        finally:
            release.set()
            await pending
            for blocker in blockers:
                blocker.result()

    async def test_fifth_interruption_resumes_to_four_terminal_modes_without_requests(self):
        from scripts.rc_evaluation.dail_sql.campaign import _interrupt_unanswered
        for persisted_cancel in (False, True):
            with self.subTest(persisted_cancel=persisted_cancel):
                self.version = self.records.begin_version(self.key)
                for rc in (False, True):
                    identity = json.dumps([self.version, 1, self.ids[:9], rc], separators=(',', ':'))
                    rid = 'r1-' + hashlib.sha256(identity.encode()).hexdigest()
                    for position in range(5):
                        for number in range(1, 6):
                            attempt = self.records.append(self.version, 'request_attempt', {
                                'round_execution_id': rid, 'sample_position': position, 'attempt_no': number})
                            if number < 5 or persisted_cancel:
                                self.records.append(self.version, 'request_result', {
                                    'request_attempt_id': attempt, 'status': 'failed', 'usage': None,
                                    'error': {'category': 'interrupted' if number == 5 else 'timeout',
                                              'retryable': number != 5, 'pause': False}})
                _interrupt_unanswered(self.records, self.version)
                history = self.events('request_result')
                result = await self.run_all()
                self.assertTrue(self.records.is_sealed(self.key, self.version))
                self.assertEqual({m['status'] for m in result['modes'].values()}, {'dependency_failed'})
                self.assertEqual(self.dispatcher.calls, [])
                self.assertEqual(len(self.events('request_attempt')), 50)
                self.assertEqual(self.events('request_result'), history)
                self.assertEqual(self.events('vote_execution'), [])
                for round_ in self.events('round_result'):
                    self.assertEqual([s['error']['category'] for s in round_['payload']['samples']],
                                     ['attempts_exhausted'] * 5)

    def setUp(self):
        self.api = importlib.import_module("scripts.baseline_adapters.dail_sql.composite")
        self.assertTrue(hasattr(self.api, "make_round_runner"), "Task6 production round factory is missing")
        from scripts.baseline_adapters.dail_sql import native, prompts
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "records"
        self.manifest = {"batch_id": "b", "groups": {"spider_dev": {"ids": ["1"]}}}
        self.records = DailRecords(self.root, self.manifest)
        self.addCleanup(lambda: self.records.close())
        self.key = TaskKey("b", "spider_dev", "1")
        self.version = self.records.begin_version(self.key)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="composite-sql")
        self.addCleanup(self.executor.shutdown)
        db = Path(self.tmp.name) / "t.sqlite"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE t(id INTEGER)")
            conn.executemany("INSERT INTO t VALUES (?)", [(1,), (1,), (2,)])
        task = {"group": "spider_dev", "question_id": "1", "question": "List IDs", "evidence": "",
                "database": {"dialect": "sqlite", "database_id": "t", "path": str(db)}}
        self.ids = [str(i) for i in range(18)]
        self.examples = {str(i): {"example_id": str(i), "question": f"Example {i}", "evidence": "",
            "sql": "SELECT DISTINCT id FROM t" if i < 9 else "SELECT COUNT(*) FROM t"} for i in range(18)}
        self.skeletons = {key: native.sql_skeleton(row["sql"], SCHEMA, "sqlite") for key, row in self.examples.items()}
        self.prepared = {"task": task, "schema": SCHEMA, "first_example_ids": self.ids[:9]}
        self.messages = prompts.build_prompt({**task, "schema": SCHEMA}, list(self.examples.values())[:9])
        self.messages[0]["content"] = "CACHED_FIRST\n" + self.messages[0]["content"]
        from scripts.rc_evaluation.dail_sql.contracts import RC_FIELDS
        self.rc3 = {field: "bound " + field for field in RC_FIELDS}
        self.dispatcher = OfflineDispatcher()

    def runner(self):
        requester = GroupRequester(self.dispatcher, object(), DailSettings(), self.records)
        return self.api.make_round_runner(self.prepared, version_id=self.version, requester=requester,
            records=self.records, model="offline", examples_by_id=self.examples, skeletons_by_id=self.skeletons,
            distance_ids=self.ids, first_messages=self.messages, rc3=self.rc3, sql_executor=self.executor,
            sql_timeout_seconds=1)

    async def run_all(self, on_mode_result=None):
        return await self.api.run_composite(self.prepared, version_id=self.version, run_round=self.runner(),
            records=self.records, on_mode_result=on_mode_result)

    def events(self, kind):
        return self.records.events(self.version, kind)

    async def test_real_sqlite_twenty_requests_twenty_votes_four_selections_and_exact_sources(self):
        result = await self.run_all()
        self.assertEqual(len(self.dispatcher.calls), 20)
        self.assertEqual(len(self.events("vote_execution")), 20)
        self.assertEqual(len(self.events("selection")), 4)
        self.assertEqual(len(result["round_ids"]), 4)
        self.assertTrue(self.records.is_sealed(self.key, self.version))
        for event in self.events("round_result"):
            r = event["payload"]
            self.assertEqual(r["success_usage"]["total_tokens"], 60)
            self.assertEqual(r["selection"]["candidate_sql"], "SELECT DISTINCT id FROM t")
            for pos, candidate in enumerate(r["candidates"]):
                self.assertEqual(candidate["source_request_id"], r["samples"][pos]["successful_request_id"])
                source = self.records.get_event(self.version, candidate["source_request_id"])
                self.assertEqual(source["choice"]["message"]["content"], candidate["raw_text"])
                self.assertEqual(candidate["provider_choice_index"], 7)
        for _, kwargs in self.dispatcher.calls:
            self.assertEqual((kwargs["n"], kwargs["temperature"], kwargs["extra_body"]), (1, .6, {"enable_thinking": True}))
            self.assertNotIn("max_tokens", kwargs)
        native, rc_first = result["modes"]["native"], result["modes"]["rc_first"]
        self.assertEqual(native["second_round_id"], rc_first["second_round_id"])
        source = self.records.get_round(self.version, native["second_round_id"])
        self.assertEqual(source["actual_parent_round_id"], native["first_round_id"])

    async def test_distinct_retrieval_lists_use_thirty_real_requests_and_six_selections(self):
        self.dispatcher.different = True
        result = await self.run_all()
        self.assertEqual((len(self.dispatcher.calls), len(self.events("vote_execution")), len(self.events("selection"))), (30, 30, 6))
        firsts = [self.records.get_round(self.version, result["modes"][m]["first_round_id"]) for m in ("native", "rc_first")]
        self.assertEqual([r["next_example_ids"] for r in firsts], [self.ids[:9], self.ids[9:]])

    async def test_postgres_cte_first_round_reaches_second_round_without_sql_rewrite(self):
        from scripts.baseline_adapters.dail_sql.execution import execute_sql
        from scripts.baseline_adapters.dail_sql.prompts import build_prompt
        # Only replace the external PostgreSQL service with the local fixture;
        # this CTE executes unchanged in both dialects. Retrieval stays real.
        self.prepared["task"]["database"]["dialect"] = "postgresql"
        self.dispatcher.raw = "WITH c AS (SELECT id FROM t) SELECT id FROM c"
        expected = "with c as ( select _ from _ ) select _ from c"
        self.skeletons = {i: ("select _ from _" if int(i) < 9 else expected) for i in self.ids}
        self.messages = build_prompt({**self.prepared["task"], "schema": SCHEMA},
                                     [self.examples[i] for i in self.ids[:9]])
        def local_execute(database, sql, *, timeout_seconds):
            self.assertEqual(sql, self.dispatcher.raw)
            return execute_sql({**database, "dialect": "sqlite"}, sql, timeout_seconds=timeout_seconds)
        runner = self.api.make_round_runner(self.prepared, version_id=self.version,
            requester=GroupRequester(self.dispatcher, object(), DailSettings(), self.records),
            records=self.records, model="offline", examples_by_id=self.examples,
            skeletons_by_id=self.skeletons, distance_ids=self.ids, first_messages=self.messages,
            rc3=self.rc3, sql_executor=self.executor, sql_timeout_seconds=1, execute=local_execute)
        result = await self.api.run_composite(self.prepared, version_id=self.version,
                                             run_round=runner, records=self.records)
        self.assertEqual({m["status"] for m in result["modes"].values()}, {"succeeded"})
        self.assertEqual(len(self.dispatcher.calls), 20)
        for event in self.events("round_result"):
            round_ = event["payload"]
            self.assertEqual(round_["next_example_ids"] if round_["round_no"] == 1
                             else round_["example_ids"], self.ids[9:])
            self.assertEqual(round_["selection"]["candidate_sql"], self.dispatcher.raw)
            for candidate in round_["candidates"]:
                vote = self.records.get_event(self.version, candidate["vote_execution_ref"])
                self.assertEqual(vote["rows"], [(1,), (1,), (2,)])

    async def test_shared_exhaustion_keeps_partial_success_and_never_retries_successful_slots(self):
        self.dispatcher.failed_positions = {2, 4}
        result = await self.run_all()
        failed = [e["payload"] for e in self.events("round_result") if e["payload"]["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        r = failed[0]
        self.assertEqual([len(s["request_attempt_ids"]) for s in r["samples"]], [1, 1, 5, 1, 5])
        self.assertEqual(len(r["request_attempt_ids"]), 13)
        self.assertEqual(len(r["successful_request_ids"]), 3)
        self.assertIsNone(r["success_usage"])
        self.assertEqual(r["candidates"], [])
        self.assertEqual(len(self.dispatcher.calls), 28)
        self.assertEqual(len(self.events("selection")), 3)
        self.assertEqual([result["modes"][m]["status"] for m in ("native", "rc_first", "rc_second", "rc_both")],
                         ["failed", "failed", "succeeded", "succeeded"])

    async def test_all_failed_slots_share_one_twenty_five_attempt_budget(self):
        self.dispatcher.failed_positions = set(range(5))
        await self.run_all()
        r = next(e["payload"] for e in self.events("round_result") if e["payload"]["status"] == "failed")
        self.assertEqual(len(r["request_attempt_ids"]), 25)
        self.assertEqual(len(self.dispatcher.calls), 40)

    async def test_stopped_transport_pauses_and_same_version_resume_preserves_sample_budget(self):
        self.dispatcher.pause = True
        with self.assertRaises(self.api.CompositePaused):
            await self.run_all()
        self.assertFalse(self.records.is_sealed(self.key, self.version))
        self.assertFalse(any(e["payload"]["status"] == "failed" for e in self.events("round_result")))
        before = self.events("request_result")
        successful_attempts = {e["payload"]["request_attempt_id"] for e in before if e["payload"]["status"] == "success"}
        counts = len(self.dispatcher.calls)
        self.dispatcher.pause = False
        self.dispatcher.stop_event.clear()
        result = await self.run_all()
        self.assertTrue(result["sealed"])
        self.assertEqual(sum(e["payload"]["status"] == "success" for e in self.events("request_result")), 20)
        self.assertEqual(len(self.dispatcher.calls) - counts, 20 - len(successful_attempts))

    async def test_callback_crash_reconciles_durable_modes_without_appends_or_sql_reexecution(self):
        async def crash(mode, event_id):
            raise OSError("offline current-index interruption")
        with self.assertRaises(OSError):
            await self.run_all(crash)
        self.assertFalse(self.records.is_sealed(self.key, self.version))
        old_modes = {e["payload"]["mode"]: e["event_id"] for e in self.events("mode_result")}
        self.assertEqual(len(old_modes), 4)
        old_count = len(self.events("selection"))
        self.records.close()
        self.records = DailRecords(self.root, self.manifest)
        # Trigger the one allowed compact hydration, then forbid history scans.
        self.records.find_source(self.version, "mode_result", "native")
        from scripts.baseline_adapters.deepeye.run_store import RunStore
        notified = {}
        with patch.object(RunStore, "iter_events", side_effect=AssertionError("No hot-path history scans")):
            result = await self.run_all(lambda mode, event: notified.update({mode: event}))
        self.assertEqual(notified, old_modes)
        self.assertEqual(result["mode_event_ids"], old_modes)
        self.assertEqual((len(self.dispatcher.calls), len(self.events("selection"))), (20, old_count))

    async def test_all_sql_errors_keep_native_fallback_and_parse_failure_is_retrieval_failure(self):
        self.dispatcher.raw = "SELECT id FROM nonexistent"
        result = await self.run_all()
        self.assertEqual(len(self.dispatcher.calls), 20)
        self.assertTrue(all(e["payload"]["fallback"] for e in self.events("selection")))
        self.assertTrue(all(m["status"] == "succeeded" for m in result["modes"].values()))
        self.version = self.records.begin_version(self.key)
        self.dispatcher.calls.clear()
        self.dispatcher.raw = "I cannot answer this question."
        result = await self.run_all()
        self.assertEqual(len(self.dispatcher.calls), 10)
        for event in self.events("round_result"):
            r = event["payload"]
            self.assertEqual(r["error"]["category"], "retrieval_parse")
            self.assertEqual(len(r["successful_request_ids"]), 5)
            self.assertEqual(len(r["candidates"]), 5)
            self.assertTrue(r["selection"]["fallback"])
            self.assertIsNone(r["success_usage"])

    async def test_cpu_retrieval_and_prompt_work_use_shared_executor_and_frozen_inputs(self):
        from scripts.baseline_adapters.dail_sql import native, retrieval, prompts
        original_skeleton, original_qualified, original_prompt = native.sql_skeleton, retrieval.qualified_examples, prompts.build_prompt
        threads, pool_refs, sqls, scan_seconds = [], [], [], []
        # Real BIRD-size skeleton scan; both first rounds share this one map.
        self.skeletons.update({f"extra{i}": self.skeletons["0"] for i in range(9410)})
        def skeleton(sql, *args):
            threads.append(threading.current_thread().name)
            sqls.append(sql)
            return original_skeleton(sql, *args)
        def qualified(pool, sql):
            threads.append(threading.current_thread().name)
            pool_refs.append(pool)
            time.sleep(.04)
            started = time.perf_counter()
            result = original_qualified(pool, sql)
            scan_seconds.append(time.perf_counter() - started)
            return result
        def prompt(*args):
            threads.append(threading.current_thread().name)
            return original_prompt(*args)
        run_round = self.runner()
        self.messages[0]["content"] = "MUTATED"
        self.rc3["population"] = "MUTATED"
        self.prepared["task"]["question"] = "MUTATED"
        with patch.object(native, "sql_skeleton", side_effect=skeleton), patch.object(retrieval, "qualified_examples", side_effect=qualified), patch.object(prompts, "build_prompt", side_effect=prompt):
            pending = asyncio.create_task(self.api.run_composite(self.prepared, version_id=self.version, run_round=run_round, records=self.records))
            ticks = 0
            while not pending.done():
                ticks += 1
                await asyncio.sleep(.002)
            await pending
        self.assertGreater(ticks, 10)
        self.assertEqual(len(pool_refs), 2)
        self.assertTrue(all(pool is self.skeletons for pool in pool_refs))
        self.assertEqual(sqls, ["SELECT DISTINCT id FROM t"] * 2)
        self.assertEqual(len(threads), 6)  # Two first skeletons, two filters, two second prompts.
        self.assertTrue(all(not name.startswith("composite-sql") and name != threading.current_thread().name
                            for name in threads), threads)
        self.assertFalse(any("MUTATED" in kwargs["messages"][0]["content"] for _, kwargs in self.dispatcher.calls))
        self.cpu_evidence = {"pool_size": len(self.skeletons), "full_scans": len(pool_refs),
                             "native_scan_seconds": scan_seconds, "loop_ticks_2ms": ticks,
                             "worker_threads": sorted(set(threads))}

    async def test_recovered_round_keeps_original_parent_and_no_request_or_selection_repeats(self):
        result = await self.run_all()
        first_a = result["modes"]["native"]["first_round_id"]
        first_b = result["modes"]["rc_first"]["first_round_id"]
        second_id = result["modes"]["native"]["second_round_id"]
        original = self.records.get_round(self.version, second_id)
        self.records.close()
        self.records = DailRecords(self.root, self.manifest)
        restored = await self.runner()(2, False, original["example_ids"], parent_round_id=first_b)
        self.assertEqual(restored, original)
        self.assertEqual(restored["actual_parent_round_id"], first_a)
        self.assertEqual((len(self.dispatcher.calls), len(self.events("selection"))), (20, 4))

    async def test_round_cancellation_checkpoints_sql_before_same_version_resume(self):
        from scripts.baseline_adapters.dail_sql.execution import execute_sql
        committed = threading.Event()
        release = threading.Event()
        sql_calls = []
        def execute(database, sql, *, timeout_seconds):
            sql_calls.append(sql)
            if len(sql_calls) == 2:
                committed.set()
                release.wait(2)
            return execute_sql(database, sql, timeout_seconds=timeout_seconds)
        run_round = self.api.make_round_runner(self.prepared, version_id=self.version,
            requester=GroupRequester(self.dispatcher, object(), DailSettings(), self.records), records=self.records,
            model="offline", examples_by_id=self.examples, skeletons_by_id=self.skeletons, distance_ids=self.ids,
            first_messages=self.messages, rc3=self.rc3, sql_executor=self.executor, sql_timeout_seconds=1, execute=execute)
        pending = asyncio.create_task(run_round(1, False, self.ids[:9]))
        while not committed.is_set():
            await asyncio.sleep(.001)
        pending.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(pending.done(), "Cancellation must wait for SQL/record ownership")
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(len(self.events("candidate")), 2)
        self.assertEqual(len(self.events("round_result")), 0)
        self.records.close()
        self.records = DailRecords(self.root, self.manifest)
        restored = await self.runner()(1, False, self.ids[:9])
        self.assertEqual(restored["status"], "success")
        self.assertEqual(len(self.dispatcher.calls), 5)
        self.assertEqual(len(self.events("vote_execution")), 5)
        self.assertEqual(len(self.events("candidate")), 5)
