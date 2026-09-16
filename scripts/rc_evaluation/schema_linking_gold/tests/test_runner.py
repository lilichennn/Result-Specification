"""Behavioral tests for bounded gold-SQL annotation orchestration."""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest

from scripts.rc_evaluation.schema_linking_gold.cli import build_parser, main
from scripts.rc_evaluation.schema_linking_gold.runner import (
    EXPECTED_MODEL,
    PILOT_LIMITS,
    RunnerSettings,
    build_batches,
    load_settings,
    prepare_pilot,
    run_annotations,
)
from scripts.rc_evaluation.schema_linking_gold.source import canonical_schema
from scripts.rc_evaluation.schema_linking_gold.store import AnnotationStore


def _task(key: str, *, schema_hash: str = "a" * 64, sql_hash: str | None = None):
    return SimpleNamespace(
        task_key=key,
        group=key.split("/", 1)[0],
        dialect="sqlite",
        gold_sql="SELECT COUNT(*) FROM orders",
        schema_sha256=schema_hash,
        sql_sha256=sql_hash or hashlib.sha256(key.encode()).hexdigest(),
        schema=canonical_schema({"tables": {
            "orders": {"columns": {"id": {"column_type": "INTEGER"}}},
        }}),
    )


def _manifest(tasks, model=EXPECTED_MODEL, pilot_keys=None):
    representatives = {}
    for task in tasks:
        cache_key = (task.dialect, task.schema_sha256, task.sql_sha256)
        if cache_key not in representatives or task.task_key < representatives[cache_key].task_key:
            representatives[cache_key] = task
    representatives = list(representatives.values())
    pilot_keys = set(pilot_keys if pilot_keys is not None else (task.task_key for task in representatives))
    pilot = [task for task in representatives if task.task_key in pilot_keys]
    remaining = [task for task in representatives if task.task_key not in pilot_keys]
    return {
        "model": model,
        "prompt_version": "gold-sql-schema-linking-v1",
        "pilot_size": len(pilot_keys),
        "pilot_task_keys": sorted(pilot_keys),
        "batch_plan": [
            {"phase": phase, "task_keys": [task.task_key for task in batch]}
            for phase, selected in (("pilot", pilot), ("rest", remaining))
            for batch in build_batches(selected)
        ],
        "tasks": [
            {
                "task_key": task.task_key,
                "group": task.group,
                "dialect": task.dialect,
                "schema_sha256": task.schema_sha256,
                "sql_sha256": task.sql_sha256,
            }
            for task in tasks
        ],
    }


def _response_for(prompt: str) -> str:
    inputs = json.loads(prompt.split("\n\nINPUTS:\n", 1)[1])["tasks"]
    return json.dumps([
        {
            "task_key": item["task_key"],
            "status": "resolved",
            "required_table_ids": ["T1"],
            "required_column_ids": [],
            "evidence": [],
            "json_paths": [],
            "review_reasons": [],
        }
        for item in inputs
    ])


class _LocalClient:
    """Fake only the remote model; requests still cross the real dispatcher."""

    def __init__(self, dispatcher, handler):
        self.dispatcher = dispatcher
        self.handler = handler
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)

        async def operation():
            content = await self.handler(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 7, "completion_tokens": 5}),
            )

        return self.dispatcher.call(operation)

    def close(self):
        return None


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = RunnerSettings(
            model=EXPECTED_MODEL,
            base_url="https://model.invalid/v1",
            api_key="super-secret-token",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, tasks, handler, **kwargs):
        path = self.root / "annotations.sqlite3"
        clients = []

        def factory(dispatcher, settings):
            client = _LocalClient(dispatcher, handler)
            clients.append(client)
            return client

        with AnnotationStore.create(path, _manifest(tasks)) as store:
            result = run_annotations(
                store,
                tasks,
                self.settings,
                client_factory=factory,
                limits=kwargs.pop("limits", replace(
                    PILOT_LIMITS, start_rate=10_000.0, request_timeout=2.0,
                )),
                **kwargs,
            )
            status = store.status()
        return path, result, status, clients

    def test_env_file_loads_required_values_without_exposing_the_secret(self):
        """Catches ignoring config/.env or retaining a printable credential."""
        env_file = self.root / ".env"
        env_file.write_text(
            "DASH_MODELS=file-model\nDASH_BASE_URL=https://file.invalid/v1\nDASH_API_KEY=file-secret\n",
            encoding="utf-8",
        )

        settings = load_settings(env_file, {"DASH_MODELS": "process-model"})

        self.assertEqual(settings.model, "process-model")
        self.assertEqual(settings.base_url, "https://file.invalid/v1")
        self.assertEqual(settings.api_key, "file-secret")
        self.assertNotIn("file-secret", repr(settings))
        self.assertNotIn("file-secret", json.dumps(settings.public_summary()))

    def test_batches_contain_at_most_twelve_tasks_from_one_schema(self):
        """Catches cross-schema requests or the wrong model batch size."""
        tasks = [_task(f"g/a/{number}", schema_hash="a" * 64) for number in range(13)]
        tasks += [_task(f"g/b/{number}", schema_hash="b" * 64) for number in range(2)]

        batches = build_batches(tasks)

        self.assertEqual([len(batch) for batch in batches], [12, 1, 2])
        for batch in batches:
            self.assertEqual(len({(task.dialect, task.schema_sha256) for task in batch}), 1)

    def test_request_is_deterministic_and_cache_reuse_avoids_a_second_call(self):
        """Catches nonzero temperature, model drift, or requesting duplicate cache inputs."""
        first = _task("spider/dev/i:0", sql_hash="1" * 64)
        duplicate = _task("spider/test/i:1", sql_hash="1" * 64)

        async def handler(kwargs):
            return _response_for(kwargs["messages"][0]["content"])

        path, result, status, clients = self._run([first, duplicate], handler)

        self.assertEqual(result["status"], "success")
        self.assertEqual(status["accepted_tasks"], 2)
        self.assertEqual(len(clients[0].requests), 1)
        request = clients[0].requests[0]
        self.assertEqual(request["model"], EXPECTED_MODEL)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["n"], 1)
        self.assertEqual([message["role"] for message in request["messages"]], ["user"])
        with sqlite3.connect(path) as connection:
            endpoint_hash, usage_json, latency = connection.execute(
                "SELECT endpoint_hash, usage_json, latency_seconds FROM outcomes"
            ).fetchone()
        self.assertEqual(endpoint_hash, hashlib.sha256(self.settings.base_url.encode()).hexdigest())
        self.assertEqual(json.loads(usage_json), {"completion_tokens": 5, "prompt_tokens": 7})
        self.assertGreaterEqual(latency, 0)
        self.assertNotIn(self.settings.api_key, path.read_bytes().decode("utf-8", errors="ignore"))
        self.assertNotIn(self.settings.base_url, path.read_bytes().decode("utf-8", errors="ignore"))

    def test_retryable_request_failure_is_recorded_before_success(self):
        """Catches skipping durable failures or failing to retry a transient request."""
        calls = 0

        async def handler(kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("secret-bearing upstream failure")
            return _response_for(kwargs["messages"][0]["content"])

        path, result, status, _ = self._run([_task("spider/dev/i:0")], handler)

        self.assertEqual(result["status"], "success")
        self.assertEqual(status["attempts"], 2)
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT status, error_json, endpoint_hash, usage_json, latency_seconds FROM outcomes ORDER BY rowid"
            ).fetchall()
        self.assertEqual([row[0] for row in rows], ["failed", "succeeded"])
        self.assertNotIn("secret-bearing", rows[0][1])
        self.assertTrue(all(row[2] and row[3] is not None and row[4] >= 0 for row in rows))
        self.assertEqual(json.loads(rows[0][3]), {"available": False})

    def test_parse_failure_records_raw_response_then_retries(self):
        """Catches treating malformed JSON as success or losing diagnostic raw output."""
        calls = 0

        async def handler(kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "not-json"
            return _response_for(kwargs["messages"][0]["content"])

        path, result, status, _ = self._run([_task("spider/dev/i:0")], handler)

        self.assertEqual(result["status"], "success")
        self.assertEqual(status["attempts"], 2)
        with sqlite3.connect(path) as connection:
            failed = connection.execute(
                "SELECT status, raw_response, error_json FROM outcomes ORDER BY rowid LIMIT 1"
            ).fetchone()
        self.assertEqual(failed[0:2], ("failed", "not-json"))
        self.assertEqual(json.loads(failed[2])["kind"], "parse_failure")

    def test_four_failed_attempts_are_exhausted_across_resume(self):
        """Catches a fifth request after the persisted four-attempt budget is exhausted."""
        calls = 0

        async def handler(kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("unavailable")

        task = _task("spider/dev/i:0")
        path, result, status, _ = self._run([task], handler)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["exhausted_batches"], 1)
        self.assertEqual((calls, status["attempts"]), (4, 4))

        clients = []

        def factory(dispatcher, settings):
            client = _LocalClient(dispatcher, handler)
            clients.append(client)
            return client

        with AnnotationStore.open(path, _manifest([task])) as store:
            resumed = run_annotations(
                store, [task], self.settings, client_factory=factory,
                limits=replace(PILOT_LIMITS, start_rate=10_000.0, request_timeout=2.0),
            )
        self.assertEqual(resumed["exhausted_batches"], 1)
        self.assertEqual(calls, 4)
        self.assertEqual(clients, [])

    def test_resume_retries_unfinished_attempt_and_keeps_it_auditable(self):
        """Catches resume losing or overwriting a crash-interrupted attempt."""
        task = _task("spider/dev/i:0")
        path = self.root / "annotations.sqlite3"
        with AnnotationStore.create(path, _manifest([task])) as store:
            store.start_attempt([task.task_key])

        async def handler(kwargs):
            return _response_for(kwargs["messages"][0]["content"])

        clients = []

        def factory(dispatcher, settings):
            client = _LocalClient(dispatcher, handler)
            clients.append(client)
            return client

        with AnnotationStore.open(path, _manifest([task])) as store:
            result = run_annotations(
                store, [task], self.settings, client_factory=factory,
                limits=replace(PILOT_LIMITS, start_rate=10_000.0, request_timeout=2.0),
            )
            status = store.status()
        self.assertEqual(result["status"], "success")
        self.assertEqual(status["attempts"], 2)
        self.assertEqual(status["unfinished_attempts"], 1)

    def test_shared_transport_and_worker_pool_bound_request_concurrency(self):
        """Catches unbounded model calls despite the configured in-flight cap."""
        lock = threading.Lock()
        active = peak = 0

        async def handler(kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.03)
            with lock:
                active -= 1
            return _response_for(kwargs["messages"][0]["content"])

        tasks = [_task(f"group/item/{number}", schema_hash=f"{number + 1:064x}") for number in range(6)]
        limits = replace(
            PILOT_LIMITS,
            request_limit=2,
            request_workers=2,
            http_connections=2,
            start_rate=10_000.0,
            request_timeout=2.0,
        )

        _, result, _, _ = self._run(tasks, handler, limits=limits)

        self.assertEqual(result["status"], "success")
        self.assertEqual(peak, 2)

    def test_bounded_smoke_succeeds_without_claiming_the_store_is_complete(self):
        """Catches treating an intentionally bounded whole-batch run as a failed command."""
        tasks = [_task(f"group/item/{number}", schema_hash=f"{number + 1:064x}") for number in range(2)]

        async def handler(kwargs):
            return _response_for(kwargs["messages"][0]["content"])

        _, result, status, _ = self._run(tasks, handler, batch_limit=1)

        self.assertEqual(result["status"], "success")
        self.assertEqual(status["accepted_inputs"], 1)
        self.assertEqual(status["pending_tasks"], 1)

    def test_model_mismatch_fails_before_client_creation(self):
        """Catches sending requests under a model different from the frozen manifest."""
        task = _task("spider/dev/i:0")
        path = self.root / "annotations.sqlite3"
        with AnnotationStore.create(path, _manifest([task])) as store:
            with self.assertRaisesRegex(ValueError, "configured model"):
                run_annotations(store, [task], replace(self.settings, model="other"),
                                client_factory=lambda *_: self.fail("client must not be created"))

    def test_preparation_rejects_any_model_except_the_expected_one_before_loading_sources(self):
        """Catches permanently freezing a typo or unintended production model."""
        with self.assertRaisesRegex(ValueError, EXPECTED_MODEL):
            prepare_pilot(
                self.root / "missing-source",
                self.root / "never-created.sqlite3",
                replace(self.settings, model="wrong-model"),
                size=5320,
            )

    def test_permanent_4xx_stops_new_admission_without_retrying_or_leaking_details(self):
        """Catches four retries and continued fan-out after an authentication/configuration error."""
        calls = 0

        class AuthenticationFailure(RuntimeError):
            status_code = 401

        async def handler(kwargs):
            nonlocal calls
            calls += 1
            raise AuthenticationFailure("api-key=super-secret-token")

        tasks = [_task(f"group/item/{number}", schema_hash=f"{number + 1:064x}") for number in range(3)]
        limits = replace(
            PILOT_LIMITS,
            request_limit=1,
            request_workers=1,
            http_connections=1,
            start_rate=10_000.0,
            request_timeout=2.0,
        )

        path, result, status, _ = self._run(tasks, handler, limits=limits)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["permanent_failures"], 1)
        self.assertEqual((calls, status["attempts"]), (1, 1))
        with sqlite3.connect(path) as connection:
            error_json = connection.execute("SELECT error_json FROM outcomes").fetchone()[0]
        self.assertEqual(json.loads(error_json), {
            "kind": "permanent_request_failure",
            "retryable": False,
            "type": "AuthenticationFailure",
        })
        self.assertNotIn("super-secret-token", error_json)

    def test_phase_runs_frozen_pilot_batches_before_remaining_batches(self):
        """Catches lexicographic pending-task slicing instead of the frozen pilot phase."""
        tasks = [_task(f"group/item/{number}", schema_hash=f"{number + 1:064x}") for number in range(4)]
        manifest = _manifest(tasks, pilot_keys={tasks[1].task_key, tasks[3].task_key})
        path = self.root / "phases.sqlite3"
        requested = []

        async def handler(kwargs):
            payload = json.loads(kwargs["messages"][0]["content"].split("\n\nINPUTS:\n", 1)[1])
            requested.append([item["task_key"] for item in payload["tasks"]])
            return _response_for(kwargs["messages"][0]["content"])

        def factory(dispatcher, settings):
            return _LocalClient(dispatcher, handler)

        limits = replace(PILOT_LIMITS, start_rate=10_000.0, request_timeout=2.0)
        with AnnotationStore.create(path, manifest) as store:
            pilot = run_annotations(
                store, tasks, self.settings, phase="pilot",
                client_factory=factory, limits=limits,
            )
            self.assertEqual(store.status()["accepted_inputs"], 2)
            remaining = run_annotations(
                store, tasks, self.settings, phase="rest",
                client_factory=factory, limits=limits,
            )
        self.assertEqual((pilot["status"], remaining["status"]), ("success", "success"))
        self.assertEqual(requested, [[tasks[1].task_key], [tasks[3].task_key],
                                     [tasks[0].task_key], [tasks[2].task_key]])

    def test_all_phase_releases_pilot_and_rest_batches_in_one_run(self):
        """Catches forcing the remaining phase to wait for every pilot batch to finish."""
        tasks = [_task(f"group/item/{number}", schema_hash=f"{number + 1:064x}") for number in range(4)]
        manifest = _manifest(tasks, pilot_keys={tasks[1].task_key, tasks[3].task_key})
        path = self.root / "all-phase.sqlite3"

        async def handler(kwargs):
            return _response_for(kwargs["messages"][0]["content"])

        def factory(dispatcher, settings):
            return _LocalClient(dispatcher, handler)

        with AnnotationStore.create(path, manifest) as store:
            result = run_annotations(
                store, tasks, self.settings, phase="all", client_factory=factory,
                limits=replace(PILOT_LIMITS, start_rate=10_000.0, request_timeout=2.0),
            )
            status = store.status()

        self.assertEqual(result["status"], "success")
        self.assertEqual(status["accepted_inputs"], 4)

    def test_bounded_failure_and_full_resume_share_one_frozen_batch_retry_budget(self):
        """Catches rebuilding a new batch key that gives the same inputs eight attempts."""
        calls = 0

        async def handler(kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("transient")

        tasks = [_task(f"group/item/{number}") for number in range(12)]
        path, first, status, _ = self._run(tasks, handler, batch_limit=1)
        self.assertEqual((first["exhausted_batches"], calls, status["attempts"]), (1, 4, 4))

        clients = []

        def factory(dispatcher, settings):
            client = _LocalClient(dispatcher, handler)
            clients.append(client)
            return client

        with AnnotationStore.open(path, _manifest(tasks)) as store:
            resumed = run_annotations(
                store, tasks, self.settings, client_factory=factory,
                limits=replace(PILOT_LIMITS, start_rate=10_000.0, request_timeout=2.0),
            )
        self.assertEqual((resumed["exhausted_batches"], calls), (1, 4))
        self.assertEqual(clients, [])

    def test_task_limit_must_end_on_a_frozen_batch_boundary(self):
        """Catches task-level slicing that changes persisted batch identity."""
        tasks = [_task(f"group/item/{number}") for number in range(12)]
        path = self.root / "boundary.sqlite3"
        with AnnotationStore.create(path, _manifest(tasks)) as store:
            with self.assertRaisesRegex(ValueError, "batch boundary"):
                run_annotations(store, tasks, self.settings, task_limit=3,
                                client_factory=lambda *_: self.fail("must fail before client creation"))

    def test_cli_registers_all_required_commands_without_importing_reporting(self):
        """Catches a missing command or eager dependency on unfinished reporting code."""
        parser = build_parser()
        actions = [action for action in parser._actions if action.dest == "command"]
        self.assertEqual(set(actions[0].choices), {"prepare-pilot", "run", "status", "verify", "export"})

        full_export = parser.parse_args(["export", "--output", str(self.root / "export"), "--scope", "full"])
        self.assertEqual(full_export.scope, "full")

    def test_verify_cli_has_distinct_command_status_and_store_progress(self):
        """Catches store verification progress overwriting the CLI success/failure status."""
        task = _task("spider/dev/i:0")
        path = self.root / "verify.sqlite3"
        with AnnotationStore.create(path, _manifest([task])) as store:
            attempt = store.start_attempt([task.task_key])
            store.finish_attempt(
                attempt["attempt_id"], "failed", usage={"available": False},
                latency_seconds=0.0, endpoint_hash="f" * 64,
                error={"kind": "fixture"},
            )
        output = io.StringIO()
        with redirect_stdout(output):
            healthy_code = main(["verify", "--store", str(path)])
        healthy = json.loads(output.getvalue())
        self.assertEqual(healthy_code, 0)
        self.assertEqual(healthy["status"], "success")
        self.assertTrue(healthy["ok"])
        self.assertIsInstance(healthy["progress"], dict)

        with sqlite3.connect(path) as connection:
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'outcomes_no_update'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER outcomes_no_update")
            connection.execute("UPDATE outcomes SET record_checksum = ?", ("0" * 64,))
            connection.execute(trigger_sql)
        output = io.StringIO()
        with redirect_stdout(output):
            corrupt_code = main(["verify", "--store", str(path)])
        corrupt = json.loads(output.getvalue())
        self.assertEqual(corrupt_code, 1)
        self.assertEqual(corrupt["status"], "failed")
        self.assertFalse(corrupt["ok"])

    def test_cli_can_freeze_the_full_5320_task_store_then_bound_the_same_run(self):
        """Catches forcing pilot labels into a separate store that full annotation cannot resume."""
        source_root = Path(__file__).resolve().parents[4] / "docs" / "analysis_rc3_five_groups_20260915"
        store_path = self.root / "full.sqlite3"

        result = prepare_pilot(source_root, store_path, self.settings, size=5320, seed=20260916)
        run_args = build_parser().parse_args([
            "run", "--store", str(store_path), "--phase", "pilot", "--batch-limit", "1",
        ])

        self.assertEqual(result["progress"]["total_tasks"], 5320)
        with AnnotationStore.open(store_path) as store:
            manifest = store.manifest
            self.assertEqual(manifest["selection"], "full")
            self.assertEqual(manifest["selection_size"], 5320)
            self.assertEqual(len(manifest["pilot_task_keys"]), 200)
            pilot_tasks = [task for task in manifest["tasks"] if task["task_key"] in manifest["pilot_task_keys"]]
            self.assertEqual({task["group"] for task in pilot_tasks}, {
                "bird_dev", "bird_interact_full", "bird_interact_lite", "spider_dev", "spider_test",
            })
            self.assertEqual({task["group"]: sum(item["group"] == task["group"] for item in pilot_tasks)
                              for task in pilot_tasks}, {
                "bird_dev": 40, "bird_interact_full": 40, "bird_interact_lite": 40,
                "spider_dev": 40, "spider_test": 40,
            })
            plan_keys = [key for batch in manifest["batch_plan"] for key in batch["task_keys"]]
            self.assertEqual(len(plan_keys), store.status()["unique_inputs"])
            self.assertEqual(len(plan_keys), len(set(plan_keys)))
            phases = [batch["phase"] for batch in manifest["batch_plan"]]
            self.assertEqual(phases, sorted(phases, key={"pilot": 0, "rest": 1}.get))
        self.assertEqual((run_args.phase, run_args.batch_limit), ("pilot", 1))


if __name__ == "__main__":
    unittest.main()
