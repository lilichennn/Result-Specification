"""Five-group focused DIN RC3 schema-filtering and Linking campaign."""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any

from dotenv import load_dotenv
from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits
from scripts.baseline_adapters.din_sql.inputs import (
    DinSettings,
    TaskKey,
    digest,
    file_hash,
    validate_settings,
)
from scripts.baseline_adapters.din_sql_linking.core import (
    LinkingPrepared,
    load_source_snapshot,
    prepared_payload,
    restore_prepared,
)
from .runner import NODES, NodeExecutor, copy_reusable, run_question


CODE_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_EXECUTION_PROFILE = {
    "request_limit": 8000,
    "start_rate": 50,
    "sql_workers": 20,
    "request_timeout_seconds": 910,
    "max_attempts": 5,
}


def validate_execution_profile(settings: DinSettings) -> DinSettings:
    """Reject configuration drift before preparation or any paid boundary."""

    validate_settings(settings)
    changed = {
        name: {"expected": expected, "actual": getattr(settings, name)}
        for name, expected in REQUIRED_EXECUTION_PROFILE.items()
        if getattr(settings, name) != expected
    }
    if changed:
        raise ValueError(f"DIN Linking execution profile differs: {changed}")
    return settings


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def implementation_hashes(code_root: str | Path) -> dict[str, str]:
    """Freeze this sibling implementation without changing old DIN hashes."""

    code_root = Path(code_root)
    paths: list[Path] = []
    for directory in (
        "scripts/baseline_adapters/din_sql_linking",
        "scripts/rc_evaluation/din_sql_linking",
    ):
        paths.extend(sorted((code_root / directory).glob("*.py")))
    paths.extend(code_root / path for path in (
        "scripts/baseline_adapters/din_sql/inputs.py",
        "scripts/baseline_adapters/din_sql/prompts.py",
        "scripts/baseline_adapters/din_sql/records.py",
        "scripts/baseline_adapters/shared/transport.py",
        "scripts/baseline_adapters/deepeye/run_store.py",
        "scripts/baseline_adapters/dail_sql/execution.py",
        "scripts/baseline_adapters/dail_sql/transport.py",
        "result_contract/rc/filter.py",
        "result_contract/rc/rc_round1.py",
        "result_contract/rc/rc_round2.py",
        "baselines_reproduce/DIN-SQL/schema_linking.py",
        "uv.lock",
    ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DIN Linking implementation dependency missing: {missing}")
    return {
        str(path.relative_to(code_root)): file_hash(path)
        for path in paths
    }


def save_prepared(root: str | Path, prepared: LinkingPrepared) -> None:
    path = Path(root) / "prepared/inputs.json"
    payload = prepared_payload(prepared)
    if path.exists() and digest(_read_json(path)) != digest(payload):
        raise ValueError("Prepared DIN Linking inputs differ; create a new batch")
    if not path.exists():
        _write_json(path, payload)


def load_prepared(root: str | Path) -> LinkingPrepared:
    return restore_prepared(_read_json(Path(root) / "prepared/inputs.json"))


def prepare_batch(
    source_batch: str | Path,
    batch_id: str,
    *,
    code_root: str | Path = CODE_ROOT,
    groups: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Create a no-model-call batch from the sealed six-node DIN source."""

    if not batch_id or Path(batch_id).name != batch_id or batch_id in (".", ".."):
        raise ValueError("batch-id must be one directory name")
    code_root = Path(code_root).resolve()
    root = code_root / "baselines_reproduce/din_sql_linking/batches" / batch_id
    source_manifest = _read_json(Path(source_batch) / "manifest.json")
    settings = validate_execution_profile(
        DinSettings(**source_manifest.get("settings", {}))
    )
    prepared = load_source_snapshot(source_batch, code_root, groups=groups)
    manifest = {
        "format": "din-sql-linking-v1",
        "batch_id": batch_id,
        "source": prepared.source,
        "settings": asdict(settings),
        "inputs_fingerprint": digest({
            "source": prepared.source,
            "identities": prepared.identities,
            "groups": {
                group: [key.question_id for key in prepared.tasks if key.group == group]
                for group in prepared.groups
            },
        }),
        "groups": {
            group: {"ids": [key.question_id for key in prepared.tasks if key.group == group]}
            for group in prepared.groups
        },
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and _read_json(manifest_path) != manifest:
        raise ValueError("Existing DIN Linking batch manifest differs")
    from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
    with LinkingRecords(root, manifest):
        save_prepared(root, prepared)
    return {"batch": str(root), "tasks": len(prepared.tasks), "groups": list(prepared.groups)}


def resolve_scope(operation, previous, targets, *, all_pending=False):
    if operation == "resume":
        if previous is None:
            raise ValueError("Batch has not been launched; use run")
        if all_pending:
            return None
        return ([TaskKey(**key) for key in previous["targets"]]
                if previous.get("targets") is not None else None)
    return targets


def status(batch: str | Path) -> dict[str, Any]:
    batch = Path(batch)
    manifest = _read_json(batch / "manifest.json")
    from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
    groups = {}
    with LinkingRecords(batch, manifest, read_only=True) as records:
        for group, definition in manifest["groups"].items():
            nodes = {node: Counter() for node in NODES}
            versions = 0
            for question_id in definition["ids"]:
                key = TaskKey(group, question_id)
                version = records.unfinished(key) or records.current(key)
                versions += int(version is not None)
                if version:
                    for node in NODES:
                        value = records.node(version, node)
                        if value:
                            nodes[node][value["status"]] += 1
            groups[group] = {
                "total": len(definition["ids"]),
                "versions": versions,
                "nodes": {node: dict(counts) for node, counts in nodes.items()},
            }
    live = batch / "monitoring/live.json"
    return {"groups": groups, "last_controller_snapshot": _read_json(live) if live.exists() else None}


def verify_batch(batch: str | Path) -> dict[str, Any]:
    batch = Path(batch)
    manifest = _read_json(batch / "manifest.json")
    prepared = load_prepared(batch)
    expected = {
        TaskKey(group, question_id)
        for group, definition in manifest["groups"].items()
        for question_id in definition["ids"]
    }
    reports = {}
    from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
    with LinkingRecords(batch, manifest, read_only=True) as records:
        for group, store in records.stores.items():
            reports[group] = store.verify()
    identity_ok = set(prepared.tasks) == expected == set(prepared.native_linking)
    changed_inputs = []
    for path, expected_hash in prepared.identities.items():
        try:
            current_hash = file_hash(path)
        except (OSError, ValueError):
            current_hash = None
        if current_hash != expected_hash:
            changed_inputs.append(path)
    return {
        "ok": identity_ok and not changed_inputs and all(report["ok"] for report in reports.values()),
        "identity_ok": identity_ok,
        "changed_inputs": sorted(changed_inputs),
        "task_count": len(prepared.tasks),
        "stores": reports,
    }


async def schedule(prepared, records, execute_node, *, targets=None, dispatcher=None):
    groups = list(prepared.groups)
    selected = set(targets) if targets is not None else None
    tasks_by_group = {group: [] for group in groups}
    counts = {group: dict.fromkeys(NODES, 0) for group in groups}
    for key, task in prepared.tasks.items():
        if selected is not None and key not in selected:
            continue
        tasks_by_group[key.group].append(task)
        version = records.unfinished(key) or records.current(key)
        if version:
            parent = records.view(version).get("parent_version")
            if parent and records.unfinished(key) == version:
                await asyncio.to_thread(copy_reusable, task, parent, version, records)
            for node in NODES:
                counts[key.group][node] += int(records.node(version, node) is not None)
    # A targeted rerun may select only a later benchmark group.  Empty groups
    # are outside that paid scope and must not create a synthetic 80% barrier.
    groups = [group for group in groups if tasks_by_group[group]]

    wake = asyncio.Event()
    running: set[asyncio.Task] = set()
    started: list[str] = []
    last_checkpoint = 0.0

    def terminal(key, node, state):
        counts[key.group][node] += 1
        wake.set()

    def finished(_future):
        wake.set()

    async def launch(group):
        started.append(group)
        for task in tasks_by_group[group]:
            version = records.unfinished(task.key)
            if version is None and records.current(task.key) is not None:
                continue
            if version is None:
                version = await asyncio.to_thread(records.begin, task.key)
            future = asyncio.create_task(
                run_question(task, version, records, execute_node, on_terminal=terminal)
            )
            future.add_done_callback(finished)
            running.add(future)

    async def checkpoint(state, *, force=False):
        nonlocal last_checkpoint
        now = time.monotonic()
        if not force and state == "running" and now - last_checkpoint < 2:
            return
        payload = {
            "state": state,
            "pid": os.getpid(),
            "started_groups": list(started),
            "terminal": counts,
            "active_questions": sum(not future.done() for future in running),
            "dispatcher": dispatcher.snapshot() if dispatcher else None,
        }
        await asyncio.to_thread(_write_json, records.root / "monitoring/live.json", payload)
        last_checkpoint = now

    try:
        # All selected benchmark groups enter the controller together.  The
        # only dependency is local to a question: run_question still executes
        # schema_filter_rc3 before linking_rc3.  RequestDispatcher owns the
        # global 8,000-in-flight and 50-starts/second bounds.
        for group in groups:
            await launch(group)

        while running:
            wake.clear()
            for future in list(running):
                if future.done():
                    future.result()
                    running.remove(future)
            await checkpoint("running")
            if not running:
                break
            try:
                await asyncio.wait_for(wake.wait(), timeout=10)
            except TimeoutError:
                pass
        await checkpoint("completed", force=True)
        return {"state": "completed", "started_groups": started, "terminal": counts}
    except BaseException:
        if dispatcher:
            dispatcher.stop(cancel_active=True)
        for future in running:
            future.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await checkpoint("paused", force=True)
        raise


def run_batch(
    batch: str | Path,
    env_file: str | Path,
    *,
    operation: str = "run",
    targets: list[TaskKey] | None = None,
    all_pending: bool = False,
) -> dict[str, Any]:
    batch = Path(batch)
    manifest = _read_json(batch / "manifest.json")
    prepared = load_prepared(batch)
    load_dotenv(env_file, override=True)
    settings = validate_execution_profile(DinSettings(**manifest["settings"]))
    model = os.environ.get("DASH_MODELS", "").strip()
    if model != settings.model:
        raise ValueError("DASH_MODELS differs from the frozen DIN Linking model")
    key, url = os.environ.get("DASH_API_KEY"), os.environ.get("DASH_BASE_URL")
    if not key or not url:
        raise ValueError("DASH_API_KEY and DASH_BASE_URL required")
    # All large input validation occurs exactly once per controller invocation.
    for path, expected in prepared.identities.items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen DIN Linking input changed: {path}")

    from scripts.baseline_adapters.din_sql_linking.records import LinkingRecords
    from scripts.baseline_adapters.din_sql_linking.transport import LinkingRequester
    with LinkingRecords(batch, manifest) as records:
        implementation = implementation_hashes(CODE_ROOT)
        implementation_path = batch / "monitoring/implementation.json"
        if implementation_path.exists() and _read_json(implementation_path) != implementation:
            raise ValueError("DIN Linking implementation changed since launch; create a new batch")
        endpoint = {"model": model, "base_url": url}
        endpoint_path = batch / "monitoring/endpoint.json"
        if endpoint_path.exists() and _read_json(endpoint_path) != endpoint:
            raise ValueError("Frozen endpoint differs")
        launch_path = batch / "monitoring/launch.json"
        previous = _read_json(launch_path) if launch_path.exists() else None
        if operation == "run" and previous:
            raise ValueError("Batch was already launched; use resume or rerun")
        targets = resolve_scope(operation, previous, targets, all_pending=all_pending)
        parents = previous.get("rerun_parents", {}) if operation == "resume" and not all_pending else {}
        if operation == "rerun":
            if not targets or any(target not in prepared.tasks for target in targets):
                raise ValueError("Explicit valid rerun targets required")
            if any(records.unfinished(target) for target in targets):
                raise ValueError("Target already has an unfinished version; resume it")
            if any(records.current(target) is None for target in targets):
                raise ValueError("Rerun requires finished previous versions")
            parents = {
                f"{target.group}/{target.question_id}": records.current(target)
                for target in targets
            }
        _write_json(launch_path, {
            "operation": operation,
            "pid": os.getpid(),
            "targets": [asdict(target) for target in targets] if targets is not None else None,
            "rerun_parents": parents,
        })
        for target in dict.fromkeys(targets or []):
            parent = parents.get(f"{target.group}/{target.question_id}")
            if not parent:
                continue
            version = records.unfinished(target)
            if version is None and records.current(target) == parent:
                version = records.begin(target, parent_version=parent)
            if version is not None:
                copy_reusable(prepared.tasks[target], parent, version, records)

        _write_json(endpoint_path, endpoint)
        _write_json(implementation_path, implementation)
        limits = RequestLimits(
            request_limit=settings.request_limit,
            request_workers=settings.request_limit,
            http_connections=settings.request_limit,
            start_rate=settings.start_rate,
            request_timeout=settings.request_timeout_seconds,
        )
        dispatcher = RequestDispatcher(limits)
        try:
            client = dispatcher.make_client(api_key=key, base_url=url)
            requester = LinkingRequester(dispatcher, client, settings, records)
            executor = NodeExecutor(prepared, requester, settings, code_root=CODE_ROOT)

            async def main():
                loop = asyncio.get_running_loop()
                current = asyncio.current_task()
                installed = []
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, current.cancel)
                        installed.append(sig)
                    except NotImplementedError:
                        pass
                try:
                    return await schedule(
                        prepared, records, executor, targets=targets, dispatcher=dispatcher
                    )
                finally:
                    for sig in installed:
                        loop.remove_signal_handler(sig)

            return asyncio.run(main())
        finally:
            dispatcher.close()
