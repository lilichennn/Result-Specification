"""One question/version: four modes sharing complete second-round executions."""
import asyncio
import copy
from functools import lru_cache, partial
import hashlib
import inspect
import json
import math
from pathlib import Path

from .config import MODES
from .execution import execute_sql


class CompositePaused(RuntimeError):
    """Transport/configuration interruption requires resuming the same version."""


@lru_cache(maxsize=1)
def _rc_definition():
    from scripts.rc_evaluation.dail_sql import contracts
    return Path(contracts.__file__).with_name("rc_prompt.txt").read_text(encoding="utf-8")


async def _offload(executor, function, *args):
    """Drain owned CPU/persistence work before exposing cancellation to caller."""
    pending = asyncio.get_running_loop().run_in_executor(executor, partial(function, *args))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not pending.cancelled():
            pending.exception()
        raise


def make_round_runner(prepared_task: dict, *, version_id: str, requester, records,
                      model: str, examples_by_id, skeletons_by_id, distance_ids,
                      first_messages: list[dict], rc3: dict, sql_executor,
                      sql_timeout_seconds: float, execute=execute_sql,
                      rc_definition: str | None = None):
    """Build the production callback for one Task4 prepared task/version.

    Required prepared keys: task (question/evidence/database), schema, and
    first_example_ids. Caller loads first_messages and ordered distance_ids once
    from that preparation, and supplies shared immutable example/skeleton maps
    once per training pool. This factory never opens a manifest/cache or rebuilds
    either pool lookup. Caller owns the requester, records and shared executor.
    """
    from . import native, prompts, retrieval, selection
    from scripts.rc_evaluation.dail_sql.contracts import inject_rc3

    if sql_executor is None:
        raise ValueError("A caller-owned SQL executor is required")
    if (not isinstance(sql_timeout_seconds, (int, float))
            or not math.isfinite(sql_timeout_seconds) or sql_timeout_seconds <= 0):
        raise ValueError("SQL timeout must be positive and finite")
    runtime = prepared_task["task"]
    task = copy.deepcopy({"question": runtime["question"], "evidence": runtime.get("evidence", ""),
                          "database": runtime["database"], "schema": prepared_task["schema"]})
    initial_ids = tuple(prepared_task["first_example_ids"])
    order = tuple(distance_ids)
    if len(initial_ids) != 9 or len(order) < 9:
        raise ValueError("Nine prepared examples are required")
    base_messages, contract = copy.deepcopy(first_messages), copy.deepcopy(rc3)
    definition = _rc_definition() if rc_definition is None else rc_definition

    def messages_for(round_no, injected, ids):
        messages = (copy.deepcopy(base_messages) if round_no == 1 else
                    prompts.build_prompt(task, [examples_by_id[identity] for identity in ids]))
        return inject_rc3(messages, contract, definition) if injected else messages

    def next_examples(sql):
        skeleton = native.sql_skeleton(sql, task["schema"], task["database"]["dialect"])
        qualified = retrieval.qualified_examples(skeletons_by_id, skeleton)
        return retrieval.choose_examples(order, k=9, qualified_ids=qualified)

    async def run_round(round_no, rc_injected, example_ids, *, parent_round_id=None):
        ids = list(example_ids)
        if (round_no not in (1, 2) or type(rc_injected) is not bool or len(ids) != 9
                or (round_no == 1 and (parent_round_id is not None or tuple(ids) != initial_ids))
                or (round_no == 2 and not parent_round_id)):
            raise ValueError("Invalid round inputs")
        # Parent is provenance, not identity. Repeated resume uses the same five
        # sample ledgers, including attempts spent before round_result existed.
        identity = json.dumps([version_id, round_no, ids, rc_injected], ensure_ascii=False, separators=(",", ":"))
        round_id = f"r{round_no}-" + hashlib.sha256(identity.encode()).hexdigest()
        saved = await _offload(None, records.find_source, version_id, "round_result", round_id)
        if saved is not None:
            return saved["payload"]  # Retain the original actual parent verbatim.
        messages = await _offload(None, messages_for, round_no, rc_injected, ids)
        generation = await requester.generate(version_id=version_id, round_execution_id=round_id,
                                               messages=messages, model=model)
        if generation["status"] != "success" and any(
                (sample.get("error") or {}).get("pause") or
                (sample.get("error") or {}).get("category") == "interrupted"
                for sample in generation["samples"]):
            raise CompositePaused("Request group paused; preserve this version for resume")
        result = {"round_execution_id": round_id, "round_no": round_no,
                  "rc_injected": rc_injected, "actual_parent_round_id": parent_round_id,
                  "example_ids": ids, "candidates": [], "selection": None, "next_example_ids": [],
                  **{key: generation[key] for key in ("status", "samples", "request_attempt_ids",
                     "successful_request_ids", "success_usage", "error")}}
        if generation["status"] == "success":
            detail = await selection.run_round(generation, task, execute=execute, records=records,
                round_execution_id=round_id, version_id=version_id, timeout_seconds=sql_timeout_seconds,
                executor=sql_executor)
            result.update(detail)
            if round_no == 1:
                try:
                    result["next_example_ids"] = await _offload(None, next_examples,
                                                               detail["selection"]["candidate_sql"])
                except native.SkeletonError as exc:
                    result.update(status="failed", success_usage=None,
                                  error={"category": "retrieval_parse", "type": type(exc).__name__,
                                         "message": str(exc)})
        await _offload(None, records.append, version_id, "round_result", result)
        return result

    return run_round


def second_round_key(example_ids: list[str], rc_injected: bool) -> tuple:
    return (tuple(example_ids), rc_injected)


def shared_second(tasks, example_ids, rc_injected, parent_round_id, run_round):
    """Register without yielding; awaiters must shield this owned shared task."""
    key = second_round_key(example_ids, rc_injected)
    if key not in tasks:
        tasks[key] = asyncio.create_task(run_round(2, rc_injected, list(example_ids),
                                                   parent_round_id=parent_round_id))
    return tasks[key]


async def run_composite(task: dict, *, version_id: str, run_round, records,
                        on_mode_result=None) -> dict:
    """Consume Task4's prepared_task (requires first_example_ids).

    run_round must persist its round_result before returning. Notifications are
    delivered after durable mode events, including replay on recovery; consumers
    reconcile idempotently by event ID. Raised exceptions/cancellation pause this
    version. Only explicit failed round results become terminal mode failures.
    """
    first_ids = list(task["first_example_ids"])
    second_round_tasks, owned, modes, mode_events = {}, [], {}, {}

    async def notify(mode, event_id):
        if on_mode_result is not None:
            value = on_mode_result(mode, event_id)
            if inspect.isawaitable(value):
                await value

    for mode in MODES:
        saved = await _offload(None, records.find_source, version_id, "mode_result", mode)
        if saved is not None:
            modes[mode], mode_events[mode] = saved["payload"], saved["event_id"]
            await notify(mode, saved["event_id"])

    async def first_source(rc):
        first = await run_round(1, rc, list(first_ids), parent_round_id=None)
        if first["status"] == "success":
            if rc:
                # A registers C/D before B may reuse them, even if B won HTTP.
                await asyncio.shield(first_a)
            for second_rc in (False, True):
                shared_second(second_round_tasks, first["next_example_ids"], second_rc,
                              first["round_execution_id"], run_round)
        return first

    async def finish_mode(mode):
        first_rc = mode in ("rc_first", "rc_both")
        second_rc = mode in ("rc_second", "rc_both")
        first = await asyncio.shield(first_b if first_rc else first_a)
        result = {"mode": mode, "first_round_id": first["round_execution_id"],
                  "second_round_id": None, "final_candidate_id": None, "failure_origin": None}
        if first["status"] != "success":
            result.update(status="dependency_failed", failure_origin={
                "round_execution_id": first["round_execution_id"], "error": first["error"]})
        else:
            second = await asyncio.shield(second_round_tasks[second_round_key(first["next_example_ids"], second_rc)])
            result["second_round_id"] = second["round_execution_id"]
            if second["status"] == "success":
                result.update(status="succeeded", final_candidate_id=second["selection"]["candidate_id"])
            else:
                result.update(status="failed", failure_origin={
                    "round_execution_id": second["round_execution_id"], "error": second["error"]})
        event_id = await _offload(None, records.append, version_id, "mode_result", result)
        modes[mode], mode_events[mode] = result, event_id
        await notify(mode, event_id)

    try:
        if len(modes) != len(MODES):
            first_a = asyncio.create_task(first_source(False))
            first_b = asyncio.create_task(first_source(True))
            owned.extend((first_a, first_b))
            waiters = [asyncio.create_task(finish_mode(mode)) for mode in MODES if mode not in modes]
            owned.extend(waiters)
            # A cancelled individual waiter cannot cancel a shared execution or
            # another mode. Drain independent outcomes before propagating pause.
            outcomes = await asyncio.gather(*waiters, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        await _offload(None, records.seal, version_id, mode_events)
        return {"version_id": version_id, "modes": modes, "mode_event_ids": mode_events,
                "round_ids": list(dict.fromkeys(rid for mode in MODES for rid in (
                    modes[mode]["first_round_id"], modes[mode]["second_round_id"]) if rid is not None)),
                "sealed": True}
    finally:
        all_tasks = owned + list(second_round_tasks.values())
        for pending in all_tasks:
            if not pending.done():
                pending.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
