"""Native ordered execution voting, with durable raw/candidate/vote evidence."""
import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
import hashlib
import math
import random
import threading

import sqlparse

from . import native
from .prompts import extract_sql
from .records import aggregate_observed_usage


def _remove_pg_distinct(sql):
    """Native removal except DISTINCT belonging to a null-safe comparison.

    Token offsets preserve literals, comments and original formatting. DISTINCT
    ON is handled by the caller, exactly as before this compatibility repair.
    """
    tokens = list(sqlparse.lexer.tokenize(sql))
    significant = [i for i, (kind, _) in enumerate(tokens)
                   if kind not in sqlparse.tokens.Whitespace and kind not in sqlparse.tokens.Comment]
    words = [tokens[i][1].upper() for i in significant]
    protected = {significant[i] for i, word in enumerate(words)
                 if word == "DISTINCT" and words[i + 1:i + 2] == ["FROM"]
                 and (words[max(0, i - 1):i] == ["IS"]
                      or words[max(0, i - 2):i] == ["IS", "NOT"])}
    return "".join(value for i, (_, value) in enumerate(tokens)
                   if value.lower() != "distinct" or i in protected)


def prepare_vote_sql(candidate_sql: str | None, dialect: str) -> dict:
    """SQLite calls unchanged DAIL helpers; PG exceptions are explicit evidence."""
    if dialect not in ("sqlite", "postgresql"):
        raise ValueError("Unsupported SQL dialect")
    result = {"vote_sql": candidate_sql, "transformations": [], "compatibility": [], "error": None}
    if candidate_sql is None:
        result["error"] = "No extracted SQL"
        return result

    def transform(name, function):
        before = result["vote_sql"]
        after = function(before)
        if before != after:
            result["transformations"].append({"name": name, "before": before, "after": after})
        result["vote_sql"] = after

    try:
        if dialect == "sqlite":
            transform("native.postprocess", native.postprocess)
            transform("native.remove_distinct", native.remove_distinct)
            transform("native.replace_cur_year", native.replace_cur_year)
        else:
            # Native text replacement can alter JSON/strings, and YEAR(CURDATE)
            # is a MySQL-to-SQLite repair, not a PostgreSQL transformation.
            result["compatibility"].extend([
                {"operation": "native.postprocess", "status": "skipped", "reason": "Preserve PostgreSQL literal and operator text"},
                {"operation": "native.replace_cur_year", "status": "skipped", "reason": "SQLite-only native YEAR(CURDATE()) compatibility"}])
            tokens = [value.upper() for token, value in sqlparse.lexer.tokenize(candidate_sql)
                      if token not in sqlparse.tokens.Whitespace and token not in sqlparse.tokens.Comment]
            if len(sqlparse.parse(candidate_sql)) != 1:
                result["compatibility"].append({"operation": "native.remove_distinct", "status": "skipped", "reason": "Preserve original text for single-statement protocol rejection"})
            elif any(tokens[i:i + 2] == ["DISTINCT", "ON"] for i in range(len(tokens) - 1)):
                result["compatibility"].append({"operation": "native.remove_distinct", "status": "skipped", "reason": "PostgreSQL DISTINCT ON cannot lose DISTINCT alone"})
            else:
                transform("postgresql.remove_distinct", _remove_pg_distinct)
    except Exception as exc:
        result["error"] = f"Vote preprocessing failed ({type(exc).__name__})"
    return result


@dataclass(frozen=True)
class _PGContainer:
    """Comparison-only typed container; never substitutes for stored raw cells."""
    kind: str
    items: tuple


def _pg_hashable(value, *, nested=False):
    if isinstance(value, dict):
        items = [(_pg_hashable(key, nested=True), _pg_hashable(item, nested=True)) for key, item in value.items()]
        items.sort(key=lambda pair: (str(type(pair[0])), repr(pair[0])))
        return _PGContainer("dict", tuple(items))
    if isinstance(value, list):
        return _PGContainer("list", tuple(_pg_hashable(item, nested=True) for item in value))
    if isinstance(value, tuple):
        return _PGContainer("tuple", tuple(_pg_hashable(item, nested=True) for item in value))
    if nested:
        return _PGContainer(f"{type(value).__module__}.{type(value).__qualname__}", (value,))
    return value


def cluster_and_choose(ordered_candidates: list[dict], executions: list[dict], *, comparison_seed: int = 0) -> dict:
    """Original-order first-equal clusters; max preserves the earliest tied group."""
    if len(ordered_candidates) != len(executions) or not ordered_candidates:
        raise ValueError("Candidates and executions must be nonempty and aligned")
    ids = [candidate["candidate_id"] for candidate in ordered_candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate ID (duplicate SQL is allowed)")
    rng = random.Random(comparison_seed)
    clusters, comparisons, results = [], [], {}
    for candidate, execution in zip(ordered_candidates, executions):
        if execution["status"] != "success":
            continue
        cid = candidate["candidate_id"]
        rows = [tuple(row) for row in execution["rows"]]
        if execution.get("dialect") == "postgresql":
            rows = [tuple(_pg_hashable(value) for value in row) for row in rows]
        results[cid] = rows
        for cluster in clusters:
            center = cluster[0]
            error = None
            try:
                equal = native.result_eq(results[center], rows, False, rng=rng)
            except Exception as exc:
                equal, error = False, f"Native comparison failed ({type(exc).__name__})"
            comparisons.append({"center_candidate_id": center, "candidate_id": cid, "equal": equal, "error": error})
            if equal:
                cluster.append(cid)
                break
        else:
            clusters.append([cid])
    winner = max(clusters, key=len)[0] if clusters else ids[0]
    maximum = max(map(len, clusters), default=0)
    return {"clusters": clusters, "winner_candidate_id": winner,
            "tie": sum(len(cluster) == maximum for cluster in clusters) > 1,
            "fallback": not clusters, "comparisons": comparisons, "comparison_seed": comparison_seed,
            "comparator": "DAIL-SQL utils/post_process.py:result_eq(order_matters=False)",
            "comparison_compatibility": "PG containers use typed hashable comparison copies; scalar values and stored rows are unchanged"}


def _validate_generation(generation, records, version_id, round_execution_id):
    samples = generation.get("samples")
    if generation.get("status") != "success" or not isinstance(samples, list) or len(samples) != 5:
        raise ValueError("Selection requires five successful samples")
    history = records.request_history(version_id, round_execution_id)
    for pos, sample in enumerate(samples):
        if sample.get("sample_position") != pos or sample.get("status") != "success":
            raise ValueError("Selection requires five ordered successful samples")
        attempts = history[pos]["attempts"]
        source = records.get_event(version_id, sample["successful_request_id"])
        if (sample["request_attempt_ids"] != [entry["request_attempt_id"] for entry in attempts]
                or not attempts or attempts[-1]["request_result_id"] != sample["successful_request_id"]
                or attempts[-1]["status"] != "success" or source.get("status") != "success"
                or source.get("choice") != sample.get("choice") or source.get("usage") != sample.get("success_usage")):
            raise ValueError("Sample differs from its durable successful request")
        content = sample["choice"]["message"].get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("SQL choice content must be text or null")
    if (generation.get("request_attempt_ids") != [rid for sample in samples for rid in sample["request_attempt_ids"]]
            or generation.get("successful_request_ids") != [sample["successful_request_id"] for sample in samples]
            or generation.get("success_usage") != aggregate_observed_usage([sample["success_usage"] for sample in samples])):
        raise ValueError("Generation request provenance does not match samples")


def _run_round(generation, task, *, execute, records, round_execution_id, version_id, timeout_seconds, cancelled):
    _validate_generation(generation, records, version_id, round_execution_id)
    database = {key: task["database"][key] for key in ("dialect", "database_id", "path") if key in task["database"]}
    candidates, executions = [], []
    for sample in generation["samples"]:
        if cancelled.is_set():
            raise InterruptedError("Selection cancelled between SQL executions")
        pos = sample["sample_position"]
        cid = f"{round_execution_id}:candidate:{pos}"
        raw = sample["choice"]["message"].get("content") or ""
        source_id = sample["successful_request_id"]
        saved = records.find_source(version_id, "candidate", cid)
        if saved is not None:
            candidate = saved["payload"]
            execution = records.get_event(version_id, candidate["vote_execution_ref"])
            if (candidate["source_request_id"] != source_id or candidate["raw_text"] != raw
                    or candidate["choice_position"] != pos or candidate["provider_choice_index"] != sample["choice"].get("index")
                    or execution["database"] != database or execution["timeout_seconds"] != timeout_seconds
                    or execution["candidate_id"] != cid or execution["source_request_id"] != source_id
                    or execution["sql"] != candidate["vote_sql"]):
                raise ValueError("Stored candidate does not match this execution input")
        else:
            extraction = extract_sql(raw, database["dialect"])
            vote = prepare_vote_sql(extraction["candidate_sql"], database["dialect"])
            if extraction["extraction_error"] or vote["error"]:
                execution = {"status": "error", "rows": [], "columns": [], "column_types": [], "value_types": [],
                             "column_count": 0, "elapsed_seconds": 0.0,
                             "error": {"type": "extraction" if extraction["extraction_error"] else "vote_preprocessing",
                                       "message": extraction["extraction_error"] or vote["error"]}}
            else:
                execution = execute(database, vote["vote_sql"], timeout_seconds=timeout_seconds)
            execution = {**execution, "candidate_id": cid, "source_request_id": source_id,
                         "round_execution_id": round_execution_id, "database": database, "dialect": database["dialect"],
                         "sql": vote["vote_sql"], "timeout_seconds": timeout_seconds,
                         "vote_transformations": vote["transformations"], "compatibility": vote["compatibility"]}
            reference = records.append(version_id, "vote_execution", execution)
            candidate = {"candidate_id": cid, "choice_position": pos, "provider_choice_index": sample["choice"].get("index"),
                         "source_request_id": source_id, **extraction, "vote_sql": vote["vote_sql"], "vote_execution_ref": reference}
            records.append(version_id, "candidate", candidate)
        candidates.append(candidate)
        executions.append(execution)
    seed = int.from_bytes(hashlib.sha256(f"{version_id}\0{round_execution_id}".encode()).digest()[:8], "big")
    if cancelled.is_set():
        raise InterruptedError("Selection cancelled before result clustering")
    selection = cluster_and_choose(candidates, executions, comparison_seed=seed)
    winner = next(candidate for candidate in candidates if candidate["candidate_id"] == selection["winner_candidate_id"])
    selection.update(candidate_id=winner["candidate_id"], candidate_sql=winner["candidate_sql"],
                     round_execution_id=round_execution_id,
                     candidate_ids=[candidate["candidate_id"] for candidate in candidates],
                     vote_execution_refs=[candidate["vote_execution_ref"] for candidate in candidates],
                     native_source=native.source_fingerprints()[native.VOTE_SOURCE])
    reference = records.append(version_id, "selection", selection)
    return {"round_execution_id": round_execution_id, "status": "success", "candidates": candidates,
            "selection": {**selection, "selection_ref": reference}, "error": None}


async def run_round(generation: dict, task: dict, *, execute, records, round_execution_id: str,
                    version_id: str, timeout_seconds: float, executor: Executor) -> dict:
    """Process one successful generation on a caller-owned shared SQL pool.

    SQL execution, result comparison and durable payload serialization all run
    in that pool. The caller assembles and appends round_result with examples,
    parent and requester fields. Interrupted uncommitted SQL may repeat; durable
    candidates reuse their original vote executions without repeating requests.
    The caller must not concurrently process the same version/round twice.
    """
    if executor is None:
        raise ValueError("A dedicated SQL executor is required")
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("SQL timeout must be positive and finite")
    cancelled = threading.Event()
    operation = partial(_run_round, generation, task, execute=execute, records=records, round_execution_id=round_execution_id,
                        version_id=version_id, timeout_seconds=timeout_seconds, cancelled=cancelled)
    pending = asyncio.get_running_loop().run_in_executor(executor, operation)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # A Python thread cannot be force-cancelled. Let its current bounded SQL
        # finish, checkpoint that candidate, and stop before the next one. Drain
        # before returning so the caller can close records or resume safely.
        cancelled.set()
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if pending.done() and not pending.cancelled():
            pending.exception()  # Retrieve worker exceptions during cancellation.
        raise
