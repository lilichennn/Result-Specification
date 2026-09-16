"""Purpose-built, durable model runner for frozen gold-SQL annotations."""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits

from .contracts import parse_response
from .prompt import render_batch_prompt
from .source import AnnotationTask, load_offline_groups, select_pilot
from .store import AnnotationStore


PROMPT_VERSION = "gold-sql-schema-linking-v1"
PILOT_SIZE = 200
PILOT_SEED = 20260916
MODEL_BATCH_SIZE = 12
MAX_ATTEMPTS = 4
PILOT_LIMITS = RequestLimits(
    request_limit=20,
    request_workers=20,
    coordinator_workers=20,
    http_connections=20,
    start_rate=20.0,
    request_timeout=910.0,
)


@dataclass(frozen=True)
class RunnerSettings:
    """The three process-local model settings; only safe metadata is printable."""

    model: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)

    def public_summary(self) -> dict[str, str]:
        return {"model": self.model, "endpoint_hash": _endpoint_hash(self.base_url)}


def load_settings(env_file: str | Path, environ: Mapping[str, str] | None = None) -> RunnerSettings:
    """Load the requested dotenv file with process environment taking precedence."""
    from dotenv import dotenv_values

    path = Path(env_file)
    if not path.is_file():
        raise ValueError("Requested model environment file is unavailable")
    file_values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    process_values = dict(os.environ if environ is None else environ)
    values = {**file_values, **process_values}
    required = ("DASH_MODELS", "DASH_BASE_URL", "DASH_API_KEY")
    if any(not isinstance(values.get(name), str) or not values[name].strip() for name in required):
        raise ValueError("Missing required model environment configuration")
    model = values["DASH_MODELS"].strip()
    if "," in model or model.startswith("["):
        raise ValueError("Gold annotation requires exactly one configured model")
    return RunnerSettings(
        model=model,
        base_url=values["DASH_BASE_URL"].strip(),
        api_key=values["DASH_API_KEY"].strip(),
    )


def build_batches(tasks: Iterable[AnnotationTask], batch_size: int = MODEL_BATCH_SIZE) -> list[list[AnnotationTask]]:
    """Create deterministic batches that never mix physical schema catalogs."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    grouped: dict[tuple[str, str], list[AnnotationTask]] = defaultdict(list)
    seen: set[str] = set()
    for task in tasks:
        task_key = getattr(task, "task_key", None)
        dialect = getattr(task, "dialect", None)
        schema_hash = getattr(task, "schema_sha256", None)
        if not all(isinstance(value, str) and value for value in (task_key, dialect, schema_hash)):
            raise ValueError("every task requires task_key, dialect, and schema_sha256")
        if task_key in seen:
            raise ValueError(f"duplicate task: {task_key}")
        seen.add(task_key)
        grouped[(dialect, schema_hash)].append(task)
    batches = []
    for group_key in sorted(grouped):
        group = sorted(grouped[group_key], key=lambda task: task.task_key)
        batches.extend(group[offset:offset + batch_size] for offset in range(0, len(group), batch_size))
    return batches


def prepare_pilot(
    source_root: str | Path,
    store_path: str | Path,
    settings: RunnerSettings,
    *,
    size: int = PILOT_SIZE,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    """Freeze the deterministic pilot manifest without making a network request."""
    all_tasks = load_offline_groups(source_root)
    if size == len(all_tasks):
        tasks = sorted(all_tasks, key=lambda task: task.task_key)
        selection = "full"
    else:
        tasks = select_pilot(all_tasks, size=size, seed=seed)
        selection = "pilot"
    source_hashes = {task.group: task.source_hash for task in all_tasks}
    if any(len({task.source_hash for task in all_tasks if task.group == group}) != 1 for group in source_hashes):
        raise ValueError("a source group has inconsistent frozen hashes")
    manifest = {
        "kind": "gold_sql_schema_linking",
        "model": settings.model,
        "prompt_version": PROMPT_VERSION,
        "selection": selection,
        "selection_size": size,
        "pilot_seed": seed,
        "source_hashes": source_hashes,
        "tasks": [_manifest_task(task) for task in tasks],
    }
    with AnnotationStore.create(store_path, manifest) as store:
        return {"status": "success", "store": str(store.path), "progress": store.status()}


def run_store(
    store_path: str | Path,
    source_root: str | Path,
    settings: RunnerSettings,
    *,
    task_limit: int | None = None,
    batch_limit: int | None = None,
) -> dict[str, Any]:
    """Open one frozen store, revalidate its source tasks, and resume requests."""
    tasks = load_offline_groups(source_root)
    with AnnotationStore.open(store_path) as store:
        return run_annotations(
            store,
            tasks,
            settings,
            task_limit=task_limit,
            batch_limit=batch_limit,
        )


def run_annotations(
    store: AnnotationStore,
    tasks: Sequence[AnnotationTask],
    settings: RunnerSettings,
    *,
    client_factory: Callable[[RequestDispatcher, RunnerSettings], Any] | None = None,
    limits: RequestLimits = PILOT_LIMITS,
    task_limit: int | None = None,
    batch_limit: int | None = None,
) -> dict[str, Any]:
    """Run cache-miss batches with bounded requests and four persisted attempts."""
    if not isinstance(settings, RunnerSettings):
        raise TypeError("settings must be RunnerSettings")
    manifest = store.manifest
    frozen_model = manifest.get("model", manifest.get("model_name"))
    if settings.model != frozen_model:
        raise ValueError("configured model differs from the frozen model")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("store uses an unsupported frozen prompt version")
    _positive_optional_limit(task_limit, "task_limit")
    _positive_optional_limit(batch_limit, "batch_limit")

    task_by_key = _validated_task_map(tasks, manifest)
    pending = store.pending_tasks()
    pending_tasks = [task_by_key[item["task_key"]] for item in pending]
    if task_limit is not None:
        pending_tasks = pending_tasks[:task_limit]
    batches = build_batches(pending_tasks)
    if batch_limit is not None:
        batches = batches[:batch_limit]

    runnable: deque[list[AnnotationTask]] = deque()
    exhausted = 0
    for batch in batches:
        if _attempt_count(store.path, batch) >= MAX_ATTEMPTS:
            exhausted += 1
        else:
            runnable.append(batch)
    if not runnable:
        progress = store.status()
        return {
            "status": "success" if exhausted == 0 else "incomplete",
            "completed_batches": 0,
            "exhausted_batches": exhausted,
            "progress": progress,
        }

    dispatcher = RequestDispatcher(limits)
    client = None
    completed = 0
    endpoint_hash = _endpoint_hash(settings.base_url)
    try:
        client = (
            client_factory(dispatcher, settings)
            if client_factory is not None
            else dispatcher.make_client(api_key=settings.api_key, base_url=settings.base_url)
        )
        with ThreadPoolExecutor(max_workers=limits.request_workers, thread_name_prefix="gold-schema-link") as executor:
            in_flight: dict[Future[Any], tuple[list[AnnotationTask], dict[str, Any], float]] = {}
            while runnable or in_flight:
                while runnable and len(in_flight) < limits.request_workers:
                    batch = runnable.popleft()
                    attempt = store.start_attempt(batch)
                    request = _request_fields(batch, settings.model)
                    started_at = time.monotonic()
                    future = executor.submit(client.chat.completions.create, **request)
                    in_flight[future] = (batch, attempt, started_at)
                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    batch, attempt, started_at = in_flight.pop(future)
                    latency = max(0.0, time.monotonic() - started_at)
                    raw_response = None
                    usage: Any = {"available": False}
                    try:
                        response = future.result()
                        usage = _response_usage(response)
                        raw_response = _response_content(response)
                        annotations = parse_response(raw_response, list(batch))
                    except Exception as error:
                        store.finish_attempt(
                            attempt["attempt_id"],
                            "failed",
                            raw_response=raw_response,
                            usage=usage,
                            latency_seconds=latency,
                            endpoint_hash=endpoint_hash,
                            error=_safe_error(error, parse_failure=raw_response is not None),
                        )
                        if attempt["attempt_no"] < MAX_ATTEMPTS:
                            runnable.append(batch)
                        else:
                            exhausted += 1
                        continue
                    store.finish_attempt(
                        attempt["attempt_id"],
                        "succeeded",
                        raw_response=raw_response,
                        usage=usage,
                        latency_seconds=latency,
                        endpoint_hash=endpoint_hash,
                    )
                    store.accept_annotations(attempt["attempt_id"], annotations)
                    completed += 1
    finally:
        if client is not None and callable(getattr(client, "close", None)):
            client.close()
        dispatcher.close()

    progress = store.status()
    return {
        "status": "success" if exhausted == 0 else "incomplete",
        "completed_batches": completed,
        "exhausted_batches": exhausted,
        "progress": progress,
    }


def _manifest_task(task: AnnotationTask) -> dict[str, Any]:
    return {
        "task_key": task.task_key,
        "group": task.group,
        "dialect": task.dialect,
        "schema_sha256": task.schema_sha256,
        "sql_sha256": task.sql_sha256,
        "source_hash": task.source_hash,
        "source_schema_sha256": task.source_schema_sha256,
        "features": list(task.features),
    }


def _validated_task_map(tasks: Sequence[AnnotationTask], manifest: Mapping[str, Any]) -> dict[str, AnnotationTask]:
    provided: dict[str, AnnotationTask] = {}
    for task in tasks:
        key = getattr(task, "task_key", None)
        if not isinstance(key, str) or not key or key in provided:
            raise ValueError("source tasks must have unique non-empty task keys")
        provided[key] = task
    frozen_tasks = manifest.get("tasks")
    if not isinstance(frozen_tasks, list):
        raise ValueError("frozen manifest has no task list")
    selected: dict[str, AnnotationTask] = {}
    for frozen in frozen_tasks:
        if not isinstance(frozen, Mapping) or not isinstance(frozen.get("task_key"), str):
            raise ValueError("frozen manifest has a malformed task")
        key = frozen["task_key"]
        task = provided.get(key)
        if task is None:
            raise ValueError(f"frozen task is unavailable from source: {key}")
        for name in ("group", "dialect", "schema_sha256", "sql_sha256"):
            if frozen.get(name) != getattr(task, name, None):
                raise ValueError(f"frozen task provenance mismatch for {key}: {name}")
        if "source_hash" in frozen and frozen["source_hash"] != getattr(task, "source_hash", None):
            raise ValueError(f"frozen task provenance mismatch for {key}: source_hash")
        if "source_schema_sha256" in frozen and frozen["source_schema_sha256"] != getattr(task, "source_schema_sha256", None):
            raise ValueError(f"frozen task provenance mismatch for {key}: source_schema_sha256")
        selected[key] = task
    return selected


def _request_fields(batch: list[AnnotationTask], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": render_batch_prompt(batch)}],
        "temperature": 0,
        "n": 1,
    }


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("model response has no first message content") from exc
    if not isinstance(content, str):
        raise ValueError("model response content must be text")
    return content


def _response_usage(response: Any) -> Any:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"available": False}
    if isinstance(usage, Mapping):
        value = dict(usage)
    elif callable(getattr(usage, "model_dump", None)):
        value = usage.model_dump()
    else:
        value = vars(usage)
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return normalized if isinstance(normalized, dict) and normalized else {"available": False}
    except (TypeError, ValueError):
        return {"available": False}


def _safe_error(error: Exception, *, parse_failure: bool) -> dict[str, str]:
    return {
        "kind": "parse_failure" if parse_failure else "request_failure",
        "type": type(error).__name__,
    }


def _endpoint_hash(base_url: str) -> str:
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()


def _batch_key(batch: Sequence[AnnotationTask]) -> str:
    task_keys = tuple(sorted(task.task_key for task in batch))
    payload = json.dumps(task_keys, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _attempt_count(store_path: Path, batch: Sequence[AnnotationTask]) -> int:
    connection = sqlite3.connect(store_path)
    try:
        row = connection.execute(
            "SELECT COALESCE(MAX(attempt_no), 0) FROM attempts WHERE batch_key = ?",
            (_batch_key(batch),),
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def _positive_optional_limit(value: int | None, label: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{label} must be a positive integer")
