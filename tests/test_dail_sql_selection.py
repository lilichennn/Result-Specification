import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
from pathlib import Path
import random
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.dail_sql import native
from scripts.baseline_adapters.dail_sql.config import TaskKey
from scripts.baseline_adapters.dail_sql.records import DailRecords, aggregate_observed_usage
from scripts.baseline_adapters.deepeye.run_store import RunStore


def successful(rows):
    return {"status": "success", "rows": rows}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        name = "scripts.baseline_adapters.dail_sql.selection"
        self.assertIsNotNone(importlib.util.find_spec(name), "Task5 selection missing")
        self.api = importlib.import_module(name)

    def test_five_votes_distinct_repeated_candidates_and_first_member(self):
        candidates = [{"candidate_id": str(i), "candidate_sql": sql} for i, sql in enumerate([
            "SELECT DISTINCT id FROM t", "SELECT id FROM t", "SELECT 2", "invalid", "SELECT 3"])]
        result = self.api.cluster_and_choose(candidates, [successful([(1,), (1,)]), successful([(1,), (1,)]),
            successful([(2,)]), {"status": "error", "rows": []}, successful([(3,)])], comparison_seed=123)
        self.assertEqual(result["clusters"], [["0", "1"], ["2"], ["4"]])
        self.assertEqual(result["winner_candidate_id"], "0")
        self.assertFalse(result["tie"])
        self.assertFalse(result["fallback"])
        self.assertEqual(result["comparison_seed"], 123)
        self.assertEqual([c["equal"] for c in result["comparisons"]], [True, False, False, False])

    def test_ties_empty_results_and_all_error_fallback(self):
        candidates = [{"candidate_id": str(i)} for i in range(5)]
        cases = [([successful([(1,)]), successful([(2,)]), successful([(2,)]), successful([(1,)]), successful([(3,)])], [["0", "3"], ["1", "2"], ["4"]], True, False),
                 ([successful([]), successful([]), {"status": "error"}, {"status": "timeout"}, successful([(1,)])], [["0", "1"], ["4"]], False, False),
                 ([{"status": "error"}] * 5, [], False, True)]
        for executions, clusters, tie, fallback in cases:
            result = self.api.cluster_and_choose(candidates, executions)
            self.assertEqual((result["clusters"], result["winner_candidate_id"], result["tie"], result["fallback"]), (clusters, "0", tie, fallback))

    def test_native_equivalence_column_permutation_bags_and_wide_local_rng(self):
        cases = [([(1, 2), (3, 4)], [(4, 3), (2, 1)], True),
                 ([(1,), (1,), (2,)], [(1,), (2,), (2,)], False),
                 ([], [], True), ([(None, 1)], [(1, None)], True)]
        state = random.getstate()
        for left, right, expected in cases:
            self.assertEqual(native.result_eq(left, right, False, rng=random.Random(10)), expected)
        left = [tuple(range(i, i + 8)) for i in range(30)]
        right = [tuple(reversed(row)) for row in reversed(left)]
        self.assertTrue(native.result_eq(left, right, False, rng=random.Random(11)))
        self.assertEqual(random.getstate(), state)

    def test_sqlite_vote_transformations_and_explicit_pg_boundary(self):
        sql = "SELECT DISTINCT id FROM t WHERE id > = YEAR(CURDATE())"
        result = self.api.prepare_vote_sql(sql, "sqlite")
        self.assertEqual(result["vote_sql"], "SELECT  id FROM t WHERE id >= 2020")
        self.assertEqual([s["name"] for s in result["transformations"]], ["native.postprocess", "native.remove_distinct", "native.replace_cur_year"])
        pg = self.api.prepare_vote_sql("SELECT DISTINCT ON (id) id, '{\"x\":\"! =\"}'::jsonb FROM t", "postgresql")
        self.assertEqual(pg["vote_sql"], "SELECT DISTINCT ON (id) id, '{\"x\":\"! =\"}'::jsonb FROM t")
        self.assertTrue(pg["compatibility"])
        # Nested containers compare losslessly while the stored results retain their values.
        rows = [({"x": [1, None]}, [1, 2])]
        output = self.api.cluster_and_choose([{"candidate_id": "a"}, {"candidate_id": "b"}],
            [{**successful(rows), "dialect": "postgresql"}, {**successful([([1, 2], {"x": [1, None]})]), "dialect": "postgresql"}])
        self.assertEqual(output["clusters"], [["a", "b"]])
        self.assertEqual(rows, [({"x": [1, None]}, [1, 2])])

    def test_pg_multistatement_preserved_for_protocol_rejection(self):
        result = self.api.prepare_vote_sql("SELECT DISTINCT 1; SELECT 2", "postgresql")
        self.assertEqual(result["vote_sql"], "SELECT DISTINCT 1; SELECT 2")
        self.assertTrue(any("single" in item["reason"].lower() for item in result["compatibility"]))

    def test_pg_container_comparison_keeps_boolean_numeric_types_and_dict_order(self):
        executions = [{**successful([({"a": 1, "b": [None, False]},)]), "dialect": "postgresql"},
                      {**successful([({"b": [None, False], "a": 1},)]), "dialect": "postgresql"},
                      {**successful([({"a": True, "b": [None, False]},)]), "dialect": "postgresql"}]
        output = self.api.cluster_and_choose([{"candidate_id": str(i)} for i in range(3)], executions)
        self.assertEqual(output["clusters"], [["0", "1"], ["2"]])


class RoundTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        name = "scripts.baseline_adapters.dail_sql.selection"
        self.assertIsNotNone(importlib.util.find_spec(name), "Task5 selection missing")
        self.api = importlib.import_module(name)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.records = DailRecords(Path(self.tmp.name) / "records", {"batch_id": "b", "groups": {"spider_dev": {"ids": ["1"]}}})
        self.addCleanup(self.records.close)
        self.version = self.records.begin_version(TaskKey("b", "spider_dev", "1"))
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sql-test")
        self.addCleanup(self.pool.shutdown)

    def generation(self, contents):
        samples = []
        for i, content in enumerate(contents):
            attempt = self.records.append(self.version, "request_attempt", {"round_execution_id": "r", "sample_position": i, "attempt_no": 1})
            choice = {"index": 7 + i, "message": {"content": content}}
            response = self.records.append(self.version, "request_result", {"request_attempt_id": attempt, "status": "success", "choice": choice, "usage": {"total_tokens": 3}})
            samples.append({"sample_position": i, "status": "success", "request_attempt_ids": [attempt],
                "successful_request_id": response, "success_usage": {"total_tokens": 3}, "choice": choice, "error": None})
        return {"status": "success", "samples": samples, "choices": [s["choice"] for s in samples],
            "request_attempt_ids": [s["request_attempt_ids"][0] for s in samples],
            "successful_request_ids": [s["successful_request_id"] for s in samples],
            "success_usage": aggregate_observed_usage([s["success_usage"] for s in samples]), "error": None}

    async def test_round_real_sqlite_provenance_raw_candidate_vote_and_record_validation(self):
        from scripts.baseline_adapters.dail_sql.execution import execute_sql
        db = Path(self.tmp.name) / "t.sqlite"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE t(id INTEGER)")
            conn.executemany("INSERT INTO t VALUES (?)", [(1,), (1,)])
        generation = self.generation(["SELECT DISTINCT id FROM t", "SELECT id FROM t", "SELECT 2", None, "SELECT 3"])
        task = {"database": {"dialect": "sqlite", "path": str(db), "database_id": "t"}}
        output = await self.api.run_round(generation, task, execute=execute_sql, records=self.records,
            round_execution_id="r", version_id=self.version, timeout_seconds=1, executor=self.pool)
        candidates = output["candidates"]
        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0]["candidate_sql"], "SELECT DISTINCT id FROM t")
        self.assertEqual(candidates[0]["vote_sql"], "SELECT  id FROM t")
        self.assertEqual(candidates[3]["raw_text"], "")
        self.assertIsNone(candidates[3]["candidate_sql"])
        self.assertEqual(output["selection"]["candidate_id"], candidates[0]["candidate_id"])
        self.assertEqual(output["selection"]["candidate_sql"], "SELECT DISTINCT id FROM t")
        for i, candidate in enumerate(candidates):
            self.assertEqual(candidate["source_request_id"], generation["samples"][i]["successful_request_id"])
            self.assertEqual(candidate["provider_choice_index"], 7 + i)
            event = self.records.get_event(self.version, candidate["vote_execution_ref"])
            self.assertEqual(event["candidate_id"], candidate["candidate_id"])
            self.assertEqual(event["source_request_id"], candidate["source_request_id"])
        result = {**generation, **output, "round_no": 1, "rc_injected": False, "actual_parent_round_id": None,
            "example_ids": list(map(str, range(9))), "next_example_ids": list(map(str, range(9)))}
        self.records.append(self.version, "round_result", result)
        self.assertEqual(self.records.get_round(self.version, "r")["candidates"], candidates)

    async def test_pool_keeps_loop_responsive_and_duplicates_keep_five_votes(self):
        generation = self.generation(["SELECT 1"] * 5)
        calls = []
        def execute(database, sql, *, timeout_seconds):
            calls.append(threading.current_thread().name)
            time.sleep(.06)
            return successful([(1,)])
        pending = asyncio.create_task(self.api.run_round(generation, {"database": {"dialect": "sqlite"}}, execute=execute,
            records=self.records, round_execution_id="r", version_id=self.version, timeout_seconds=1, executor=self.pool))
        ticks = 0
        while not pending.done():
            await asyncio.sleep(.005)
            ticks += 1
        output = await pending
        self.assertGreater(ticks, 3)
        self.assertTrue(all(name.startswith("sql-test") for name in calls))
        self.assertEqual(len(output["selection"]["clusters"][0]), 5)
        self.assertEqual(len(set(c["candidate_id"] for c in output["candidates"])), 5)

    async def test_failed_or_forged_generation_has_no_sql_or_candidate_side_effects(self):
        generation = self.generation(["SELECT 1"] * 5)
        async def check(value):
            def execute(*args, **kwargs):
                raise AssertionError("SQL should not run")
            with self.assertRaises(ValueError):
                await self.api.run_round(value, {"database": {"dialect": "sqlite"}}, execute=execute, records=self.records,
                    round_execution_id="r", version_id=self.version, timeout_seconds=1, executor=self.pool)
        await check({**generation, "status": "failed"})
        generation["samples"][0]["choice"]["message"]["content"] = "SELECT forged"
        await check(generation)
        self.assertEqual(self.records.events(self.version, "candidate"), [])

    async def test_partial_candidate_resume_uses_verified_point_sources(self):
        generation = self.generation(["SELECT 1"] * 5)
        calls = []
        def interrupted(database, sql, *, timeout_seconds):
            calls.append(sql)
            if len(calls) == 3:
                raise RuntimeError("process interrupted")
            return successful([(1,)])
        args = {"records": self.records, "round_execution_id": "r", "version_id": self.version,
                "timeout_seconds": 1, "executor": self.pool}
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            await self.api.run_round(generation, {"database": {"dialect": "sqlite"}}, execute=interrupted, **args)
        self.assertEqual(len(self.records.events(self.version, "candidate")), 2)
        self.assertIsNone(self.records.find_source(self.version, "candidate", "missing"))
        with self.assertRaises(ValueError):
            self.records.find_source(self.version, "vote_execution", "missing")
        source = self.records.find_source(self.version, "candidate", "r:candidate:0")
        original_reference = source["payload"]["vote_execution_ref"]
        source["payload"]["raw_text"] = "mutated"
        self.assertEqual(self.records.find_source(self.version, "candidate", "r:candidate:0")["payload"]["raw_text"], "SELECT 1")
        self.assertEqual(source["kind"], "candidate")
        self.assertEqual(self.records.get_event(self.version, source["event_id"])["candidate_id"], "r:candidate:0")
        self.records.close()
        self.records = DailRecords(Path(self.tmp.name) / "records", {"batch_id": "b", "groups": {"spider_dev": {"ids": ["1"]}}})
        self.addCleanup(self.records.close)
        self.records.request_history(self.version, "r")  # One recovery hydration allowed.
        args["records"] = self.records
        replay_calls = []
        def resumed(database, sql, *, timeout_seconds):
            replay_calls.append(sql)
            return successful([(1,)])
        with patch.object(RunStore, "iter_events", side_effect=AssertionError("No ordinary-round history scan")):
            output = await self.api.run_round(generation, {"database": {"dialect": "sqlite"}}, execute=resumed, **args)
        self.assertEqual(len(replay_calls), 3)
        self.assertEqual(output["candidates"][0]["vote_execution_ref"], original_reference)
        self.assertEqual(len(self.records.events(self.version, "candidate")), 5)
        self.assertEqual(len(output["selection"]["clusters"][0]), 5)

    async def test_cancellation_waits_for_sql_worker_before_return_and_stops_next_candidate(self):
        generation = self.generation(["SELECT 1"] * 5)
        started, finished = threading.Event(), threading.Event()
        calls = []
        def execute(database, sql, *, timeout_seconds):
            calls.append(sql)
            started.set()
            time.sleep(.08)
            finished.set()
            return successful([(1,)])
        pending = asyncio.create_task(self.api.run_round(generation, {"database": {"dialect": "sqlite"}}, execute=execute,
            records=self.records, round_execution_id="r", version_id=self.version, timeout_seconds=1, executor=self.pool))
        while not started.is_set():
            await asyncio.sleep(.001)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(finished.is_set(), "run_round returned while SQL still owns record resources")
        self.assertEqual(len(calls), 1)
