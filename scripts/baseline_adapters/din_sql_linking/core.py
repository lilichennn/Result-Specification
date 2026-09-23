"""Frozen inputs and prompt/context construction for DIN RC3 Linking.

RC3 is consumed only by the schema filter.  The downstream Linking prompt is
the original DIN prompt with the full database context replaced by the cropped
context produced for that question.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

from result_contract.rc.filter import apply_filter, build_filter_messages, parse_filter_response
from scripts.baseline_adapters.dail_sql.execution import execute_sql
from scripts.baseline_adapters.din_sql.inputs import (
    DinSettings,
    TaskKey,
    file_hash,
    legacy_pure_functions,
)
from scripts.baseline_adapters.din_sql.prompts import PromptBuilder, family, request_kwargs
from scripts.baseline_adapters.din_sql.records import DinRecords, load_prepared, read_json


@dataclass
class LinkingPrepared:
    """Self-contained inputs for the focused two-node campaign."""

    tasks: dict[TaskKey, "LinkingTask"]
    schemas: dict[str, dict[str, Any]]
    metadata: dict[str, list[dict[str, Any]]]
    templates: dict[str, Any]
    native_linking: dict[TaskKey, dict[str, Any]]
    identities: dict[str, str]
    groups: tuple[str, ...]
    source: dict[str, Any]


@dataclass(frozen=True)
class LinkingTask:
    """Public-input-only task used by the focused Linking experiment.

    The source DIN task also carries a difficulty label derived from gold SQL.
    Keeping a separate type makes that field impossible to serialize into this
    blind comparison batch by accident.
    """

    key: TaskKey
    question: str
    evidence: str
    database: dict[str, Any]
    schema_ref: str
    rc3: dict[str, Any]
    source_refs: dict[str, Any] = field(default_factory=dict)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                raise ValueError(f"Metadata CSV has no header: {path}")
            return [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _load_database_metadata(meta_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")
    paths = sorted(meta_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No metadata CSV files found: {meta_dir}")
    metadata = [
        {"table_name": path.stem, "columns": _read_csv(path)}
        for path in paths
    ]
    return metadata, {str(path.resolve()): file_hash(path) for path in paths}


def _source_file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_meta_dir(task: LinkingTask, code_root: str | Path) -> Path:
    """Use the prepared Meta binding; older task records retain their default."""
    if task.source_refs.get("meta_dir"):
        return Path(task.source_refs["meta_dir"])
    return Path(code_root) / "data" / task.key.group / "meta" / task.database["database_id"]


def load_source_snapshot(
    source_batch: str | Path,
    code_root: str | Path,
    *,
    groups: list[str] | tuple[str, ...] | None = None,
) -> LinkingPrepared:
    """Freeze tasks, metadata, templates, contexts and native Linking refs.

    No gold annotation is loaded here.  The formal comparison batch therefore
    remains blind to the Task-2 evaluation reference.
    """

    source_batch = Path(source_batch).resolve(strict=True)
    code_root = Path(code_root).resolve(strict=True)
    manifest_path = source_batch / "manifest.json"
    inputs_path = source_batch / "prepared/inputs.json"
    manifest = read_json(manifest_path)
    source_prepared = load_prepared(source_batch)
    all_groups = tuple(manifest["groups"])
    selected_groups = tuple(groups) if groups is not None else all_groups
    if not selected_groups or len(set(selected_groups)) != len(selected_groups):
        raise ValueError("Selected groups must be non-empty and unique")
    unknown = set(selected_groups) - set(all_groups)
    if unknown:
        raise ValueError(f"Unknown source groups: {sorted(unknown)}")

    tasks = {
        key: LinkingTask(
            key=task.key,
            question=task.question,
            evidence=task.evidence,
            database=task.database,
            schema_ref=task.schema_ref,
            rc3=task.rc3,
            source_refs=task.source_refs,
        )
        for key, task in source_prepared.tasks.items()
        if key.group in selected_groups
    }
    expected_keys = {
        TaskKey(group, question_id)
        for group in selected_groups
        for question_id in manifest["groups"][group]["ids"]
    }
    if set(tasks) != expected_keys:
        raise ValueError("Source manifest and prepared task identities differ")

    identities = {
        str(manifest_path): _source_file_digest(manifest_path),
        str(inputs_path): _source_file_digest(inputs_path),
    }
    metadata: dict[str, list[dict[str, Any]]] = {}
    for task in tasks.values():
        if task.schema_ref in metadata:
            continue
        meta_dir = _task_meta_dir(task, code_root)
        raw, hashes = _load_database_metadata(meta_dir)
        metadata[task.schema_ref] = raw
        identities.update(hashes)

    native_linking: dict[TaskKey, dict[str, Any]] = {}
    with DinRecords(source_batch, manifest, read_only=True) as records:
        for key in tasks:
            version = records.current(key)
            if version is None:
                raise ValueError(f"Source question has no finished version: {key}")
            node = records.node(version, "linking")
            if node is None or node.get("status") != "succeeded":
                raise ValueError(f"Source question has no successful Linking node: {key}")
            native_linking[key] = {
                **{name: value for name, value in node.items() if name != "ref"},
                "ref": node["ref"],
                "source_version": version,
            }

    schemas = {
        ref: source_prepared.schemas[ref]
        for ref in {task.schema_ref for task in tasks.values()}
    }
    # PromptBuilder has a fixed Spider example fallback.
    if "spider:college_2" in source_prepared.schemas:
        schemas["spider:college_2"] = source_prepared.schemas["spider:college_2"]
    source = {
        "batch_id": manifest["batch_id"],
        "batch_path": str(source_batch),
        "manifest_sha256": identities[str(manifest_path)],
        "prepared_sha256": identities[str(inputs_path)],
    }
    return LinkingPrepared(
        tasks=tasks,
        schemas=schemas,
        metadata=metadata,
        templates=source_prepared.templates,
        native_linking=native_linking,
        identities=identities,
        groups=selected_groups,
        source=source,
    )


def prepared_payload(prepared: LinkingPrepared) -> dict[str, Any]:
    """Return a JSON-safe, deterministic snapshot without gold annotations."""

    return {
        "tasks": [asdict(prepared.tasks[key]) for key in prepared.tasks],
        "schemas": prepared.schemas,
        "metadata": prepared.metadata,
        "templates": prepared.templates,
        "native_linking": [
            {"key": asdict(key), "node": prepared.native_linking[key]}
            for key in prepared.tasks
        ],
        "identities": prepared.identities,
        "groups": list(prepared.groups),
        "source": prepared.source,
    }


def restore_prepared(payload: dict[str, Any]) -> LinkingPrepared:
    """Restore a snapshot produced by :func:`prepared_payload`."""

    tasks: dict[TaskKey, LinkingTask] = {}
    for row in payload["tasks"]:
        row = dict(row)
        row["key"] = TaskKey(**row["key"])
        task = LinkingTask(**row)
        if task.key in tasks:
            raise ValueError(f"Duplicate prepared task: {task.key}")
        tasks[task.key] = task
    native_linking = {}
    for row in payload["native_linking"]:
        key = TaskKey(**row["key"])
        if key in native_linking:
            raise ValueError(f"Duplicate native Linking snapshot: {key}")
        native_linking[key] = row["node"]
    if set(tasks) != set(native_linking):
        raise ValueError("Prepared tasks and native Linking snapshots differ")
    return LinkingPrepared(
        tasks=tasks,
        schemas=payload["schemas"],
        metadata=payload["metadata"],
        templates=payload["templates"],
        native_linking=native_linking,
        identities=payload["identities"],
        groups=tuple(payload["groups"]),
        source=payload["source"],
    )


def build_filter_payload(
    task: LinkingTask,
    metadata: list[dict[str, Any]],
    settings: DinSettings,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build one Q + Hint + RC3 schema-filter request."""

    messages = build_filter_messages(
        metadata,
        question=task.question,
        evidence=task.evidence,
        rc=task.rc3,
    )
    kwargs = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "n": 1,
        "stream": False,
    }
    return messages, kwargs


def parse_filter_content(content: str, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a model selection against the canonical metadata and crop it."""

    selection = parse_filter_response(content)
    return {
        "selection": selection,
        "filtered_metadata": apply_filter(metadata, selection),
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def postgresql_filtered_context(
    database: dict[str, Any],
    metadata: list[dict[str, Any]],
    *,
    execute: Callable[..., dict[str, Any]] = execute_sql,
    timeout_seconds: float = 180,
) -> str:
    """Render the existing DIN PostgreSQL context shape for a cropped schema."""

    parts: list[str] = []
    for table in metadata:
        table_name = table["table_name"]
        columns = table.get("columns")
        if not isinstance(table_name, str) or not table_name or not isinstance(columns, list):
            raise ValueError("Filtered PostgreSQL metadata has an invalid table")
        names = [row.get("original_column_name") or row.get("column_name") for row in columns]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"Filtered PostgreSQL metadata has an invalid column: {table_name}")
        parts.append(f'Table {table_name}, columns = [*,{",".join(names)}]' if names
                     else f'Table {table_name}, columns = [*]')
        for row, name in zip(columns, names):
            parts.append(
                f'Column {name}: type -> {row.get("data_format", row.get("data_type", ""))}, '
                f'column description -> {row.get("column_description", "")}, '
                f'value description -> {row.get("value_description", "")}'
            )
        if names:
            query = (
                f'SELECT {", ".join(_quote_identifier(name) for name in names)} '
                f'FROM {_quote_identifier(table_name)} LIMIT 3'
            )
            sampled = execute(database, query, timeout_seconds=timeout_seconds)
            if sampled.get("status") != "success":
                raise ValueError(
                    f'Cannot sample {database.get("database_id")}.{table_name}: '
                    f'{sampled.get("error")}'
                )
            rows = sampled.get("rows", [])
        else:
            rows = []
        parts.append("Sample rows: " + json.dumps(rows, ensure_ascii=False, default=str))
    return "\n".join(parts)


def build_filtered_context(
    task: LinkingTask,
    filtered_metadata: list[dict[str, Any]],
    *,
    meta_dir: str | Path | None = None,
    code_root: str | Path,
    execute: Callable[..., dict[str, Any]] = execute_sql,
    timeout_seconds: float = 180,
) -> str:
    """Create a context from the cropped canonical metadata."""

    if task.database["dialect"] == "postgresql":
        return postgresql_filtered_context(
            task.database,
            filtered_metadata,
            execute=execute,
            timeout_seconds=timeout_seconds,
        )
    if task.database["dialect"] != "sqlite":
        raise ValueError(f'Unsupported database dialect: {task.database["dialect"]}')
    if meta_dir is None:
        meta_dir = _task_meta_dir(task, code_root)
    database_context = _sqlite_context_builder(str(Path(code_root).resolve()))
    return database_context(
        Path(task.database["path"]),
        Path(meta_dir),
        filtered=filtered_metadata,
    )


@lru_cache(maxsize=None)
def _sqlite_context_builder(code_root: str) -> Callable[..., str]:
    """Load the legacy pure formatter once per process, never once per task."""

    return legacy_pure_functions(
        Path(code_root),
        "schema_linking.py",
        ["quote_identifier", "column_descriptions", "database_context"],
    )["database_context"]


def build_linking_payload(
    task: LinkingTask,
    context: str,
    builder: PromptBuilder,
    settings: DinSettings,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the native DIN Linking request with only its context substituted."""

    dataset = family(task)
    bird = dataset != "spider"
    instance = {
        "question": task.question,
        "evidence": task.evidence,
        "db_id": task.database["database_id"],
    }
    examples = (
        builder.prepared.templates["bird"]["SYSTEM_SCHEMA_LINKING_TEMPLATE"]
        .split("Few examples of this task are:\n###\n", 1)[1]
        if bird
        else builder.prepared.templates["spider"]["schema_linking_prompt"]
    ).strip()
    messages = [{
        "role": "user",
        "content": builder.linking["build_prompt"](examples, context, instance),
    }]
    if dataset == "postgresql":
        messages = [{
            **message,
            "content": message["content"].replace("SQLite", "PostgreSQL").replace("sqlite", "postgresql"),
        } for message in messages]
    return messages, request_kwargs("linking", dataset, messages, settings)
