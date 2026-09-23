"""Two-node per-question dependency chain for focused DIN Linking."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from scripts.baseline_adapters.din_sql.inputs import digest
from scripts.baseline_adapters.din_sql.prompts import PromptBuilder, family, parse_response
from scripts.baseline_adapters.din_sql_linking.core import (
    build_filter_payload,
    build_filtered_context,
    build_linking_payload,
    parse_filter_content,
)
from scripts.baseline_adapters.din_sql_linking.transport import OutputValidationError


NODES = ("schema_filter_rc3", "linking_rc3")


def _validate_filter_model_output(content, metadata):
    """Mark only filter parser/selection ValueErrors as model-output failures."""

    try:
        return parse_filter_content(content, metadata)
    except ValueError as exc:
        raise OutputValidationError("Invalid schema-filter model output") from exc


def _validate_linking_model_output(content, task):
    """Mark only native Linking parser ValueErrors as model-output failures."""

    try:
        return parse_response("linking", family(task), content)
    except ValueError as exc:
        raise OutputValidationError("Invalid Linking model output") from exc


def outcome(
    node: str,
    *,
    status: str = "succeeded",
    result: Any = None,
    reason: Any = None,
    origin: str = "new_execution",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "node": node,
        "status": status,
        "result": result,
        "reason": reason,
        "origin": origin,
        "usage": None,
        "input_fingerprint": None,
        "parent_refs": {},
        "response_ref": None,
        "source_refs": {},
        "fallback_used": False,
        **extra,
    }


def copy_reusable(task, parent_version: str, new_version: str, records) -> set[str]:
    """A successful filter is independent of the failed downstream Linking."""

    old = records.node(parent_version, "schema_filter_rc3")
    if not old or old.get("status") != "succeeded":
        return set()
    existing = records.node(new_version, "schema_filter_rc3")
    if existing:
        return {"schema_filter_rc3"}
    value = {name: item for name, item in old.items() if name != "ref"}
    value.update(
        origin="reused",
        source_refs={**old.get("source_refs", {}), "reused_node": old["ref"]},
        parent_refs={},
    )
    records.save_node(new_version, "schema_filter_rc3", value)
    return {"schema_filter_rc3"}


async def run_question(task, version: str, records, execute_node, *, on_terminal=None):
    """Resume the first missing node and seal after both nodes are terminal."""

    filtered = records.node(version, "schema_filter_rc3")
    if filtered is None:
        filtered = await execute_node("schema_filter_rc3", task, None, version)
        await asyncio.to_thread(records.save_node, version, "schema_filter_rc3", filtered)
        filtered = records.node(version, "schema_filter_rc3")
        if on_terminal:
            on_terminal(task.key, "schema_filter_rc3", filtered["status"])

    linked = records.node(version, "linking_rc3")
    if linked is None:
        if filtered["status"] != "succeeded":
            linked = outcome(
                "linking_rc3",
                status="dependency_failed",
                reason={"dependencies": ["schema_filter_rc3"]},
                input_fingerprint=digest([
                    str(task.key), "linking_rc3", filtered["ref"], "dependency_failed"
                ]),
                parent_refs={"schema_filter_rc3": filtered["ref"]},
            )
        else:
            linked = await execute_node("linking_rc3", task, filtered, version)
            linked["parent_refs"] = {"schema_filter_rc3": filtered["ref"]}
        await asyncio.to_thread(records.save_node, version, "linking_rc3", linked)
        linked = records.node(version, "linking_rc3")
        if on_terminal:
            on_terminal(task.key, "linking_rc3", linked["status"])

    await asyncio.to_thread(records.seal, version)
    return {node: records.node(version, node)["status"] for node in NODES}


class NodeExecutor:
    """Build each request once and delegate durable retries to LinkingRequester."""

    def __init__(
        self,
        prepared,
        requester,
        settings,
        *,
        code_root: str | Path | None = None,
        context_workers: int | None = None,
        execute=None,
    ):
        self.prepared = prepared
        self.requester = requester
        self.settings = settings
        self.code_root = Path(code_root or Path(__file__).resolve().parents[3])
        self.builder = PromptBuilder(prepared, self.code_root)
        self.context_slots = asyncio.Semaphore(context_workers or settings.sql_workers)
        self.execute = execute

    async def _context(self, task, filtered_metadata):
        kwargs = {
            "code_root": self.code_root,
            "timeout_seconds": self.settings.sql_timeout_seconds,
        }
        if self.execute is not None:
            kwargs["execute"] = self.execute
        async with self.context_slots:
            return await asyncio.to_thread(
                build_filtered_context,
                task,
                filtered_metadata,
                **kwargs,
            )

    async def __call__(self, node, task, parent, version):
        metadata = self.prepared.metadata[task.schema_ref]
        if node == "schema_filter_rc3":
            _, kwargs = build_filter_payload(task, metadata, self.settings)
            validator = lambda content: _validate_filter_model_output(content, metadata)
        elif node == "linking_rc3":
            context = await self._context(task, parent["result"]["filtered_metadata"])
            _, kwargs = build_linking_payload(task, context, self.builder, self.settings)

            def validator(content):
                return _validate_linking_model_output(content, task)
        else:
            raise ValueError(f"Unknown DIN Linking node: {node}")

        response = await self.requester.request(version, node, kwargs, validator)
        if response["status"] != "succeeded":
            return outcome(
                node,
                status="failed",
                reason=response.get("error"),
                response_ref=response.get("response_ref"),
                input_fingerprint=digest(kwargs),
            )
        parsed = response["parsed"]
        if node == "schema_filter_rc3":
            result, warnings, fallback = parsed, [], False
        else:
            result = parsed["result"]
            warnings = parsed.get("warnings", [])
            fallback = parsed.get("fallback_used", False)
        return outcome(
            node,
            result=result,
            reason=warnings,
            usage=response.get("usage"),
            input_fingerprint=digest(kwargs),
            response_ref=response.get("response_ref"),
            source_refs={"response_model": response.get("response_model")},
            fallback_used=fallback,
        )
