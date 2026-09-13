"""Bounded, no-retry SDK probe with append-only evidence and resource telemetry."""
from __future__ import annotations

import concurrent.futures as cf
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from urllib.parse import urlsplit
from .handoff import DrainingProbe

import psutil

BASELINE = Path(__file__).resolve().parents[2]/"baselines/DeepEye-SQL"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))
from scripts.baseline_adapters.deepeye.run_store import RunStore


def utc():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    # Canonical evidence is in transactional RunStore; exports are never overwritten.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resources():
    p = psutil.Process()
    vm, swap = psutil.virtual_memory(), psutil.swap_memory()
    cpu = p.cpu_times()
    return {"at": utc(), "threads": p.num_threads(), "rss_bytes": p.memory_info().rss,
            "fds": p.num_fds(), "system_memory_percent": vm.percent,
            "available_memory_bytes": vm.available, "swap_used_bytes": swap.used,
            "swap_out_bytes": swap.sout, "disk_free_bytes": psutil.disk_usage(Path.cwd()).free,
            "cpu_user_seconds": cpu.user, "cpu_system_seconds": cpu.system}


@dataclass(frozen=True)
class ProbeBudget:
    concurrency: int
    max_requests: int
    admission_seconds: float
    token_stop: int
    timeout: float = 660
    authorized: bool = False

    def validate(self, remote=False):
        for name in ("concurrency", "max_requests", "token_stop"):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("admission_seconds", "timeout"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if remote and (not self.authorized or self.max_requests > 2*self.concurrency):
            raise ValueError("Remote exploration needs explicit authorization and at most 2C requests")


def distribution(values):
    a = sorted(values)
    if not a:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {"n": len(a), **{f"p{q}": a[max(0, math.ceil(q/100*len(a))-1)]
                            for q in (50, 95, 99)}, "max": a[-1]}


class AdmissionPacer:
    """Single-scheduler spacing, without saved credits or catch-up bursts."""
    def __init__(self, requests_per_second=None):
        if requests_per_second is not None and (
                not math.isfinite(requests_per_second) or requests_per_second <= 0):
            raise ValueError('requests_per_second must be finite and positive')
        self.interval = 1 / requests_per_second if requests_per_second else 0
        self.next_at = 0

    def delay(self, now):
        return max(0, self.next_at-now)

    def admitted(self, now):
        self.next_at = now+self.interval


def summarize(rows, elapsed):
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {k: sum((r.get("usage") or {}).get(k) or 0 for r in rows) for k in token_fields}
    known = [r for r in rows if all(isinstance((r.get("usage") or {}).get(k), int)
                                    for k in token_fields)]
    reasoning = [(r.get("usage") or {}).get("completion_tokens_details") or {} for r in rows]
    reasoning = [r["reasoning_tokens"] for r in reasoning if isinstance(r.get("reasoning_tokens"), int)]
    good = sum(bool(r.get("parsed")) for r in rows)
    return {"requests": len(rows), "http_success": sum(bool(r.get("http_success")) for r in rows),
            "nonempty": sum(bool(r.get("nonempty")) for r in rows), "parsed": good,
            "errors_by_type": dict(Counter(r["error_type"] for r in rows if r.get("error_type"))),
            "reported_tokens": totals, "usage_unknown_requests": len(rows)-len(known),
            "reasoning_tokens_reported": sum(reasoning), "reasoning_usage_known_requests": len(reasoning),
            "elapsed_seconds": elapsed, "parsed_per_minute": good*60/max(elapsed, .001),
            "http_success_per_minute": sum(bool(r.get("http_success")) for r in rows)*60/max(elapsed,.001),
            "latency_seconds": distribution([r["elapsed_seconds"] for r in rows]),
            "success_latency_seconds": distribution([r["elapsed_seconds"] for r in rows if r.get("http_success")]),
            "sdk_to_first_transport_event_seconds": distribution([r["transport_start_delay"] for r in rows
                                                                    if r.get("transport_start_delay") is not None]),
            "input_composition": dict(Counter(r.get("stratum", "fixture") for r in rows))}


def run_requests(run_dir, workload, budget, base_url, api_key, model, *, provider_tpm=None,
                 workers=None, requests_per_second=None, draining_run=None,
                 group_plan=None, coordinators=None, stop_after_target_successes=None):
    """Only one SDK create per task. No retries, SQL execution, or production patching.

    SDK timeout is per transport operation, NOT a claimed hard total deadline.
    SIGINT/SIGTERM or STOP file stops admission, drains calls and commits a summary.
    """
    from openai import OpenAI, DefaultHttpxClient
    from openai._base_client import httpx2 as httpx
    from .workload import parse_response

    remote = urlsplit(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}
    budget.validate(remote=remote)
    workers = budget.concurrency if workers is None else workers
    if type(workers) is not int or workers <= 0:
        raise ValueError('workers must be a positive integer')
    pacer = AdmissionPacer(requests_per_second)
    effective_limit = min(workers, budget.concurrency)
    if not workload:
        raise ValueError("workload cannot be empty")
    if group_plan is not None:
        from .groups import GroupQueue, validate_plan, warm_pool
        validate_plan(group_plan, workload, coordinators)
        if draining_run:
            raise ValueError('grouped probes do not support draining-run handoff')
    if stop_after_target_successes is not None and (
            type(stop_after_target_successes) is not int or stop_after_target_successes <= 0):
        raise ValueError('stop_after_target_successes must be a positive integer')
    source = DrainingProbe(draining_run) if draining_run else None
    source_pending = source.pending() if source else 0
    if source and remote and any(source.manifest.get(k) != v for k, v in
            {'model': model, 'endpoint': base_url, 'workload_sha256': fingerprint(workload)}.items()):
        source.close()
        raise ValueError('handoff source must use the same endpoint, model and workload')
    if source_pending > budget.concurrency:
        source.close()
        raise ValueError('existing source calls already exceed the combined target')
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "prechange_sdk_capacity_probe", "created_at": utc(), "budget": asdict(budget),
                "model": model, "endpoint": base_url, "provider_tpm_user_reported": provider_tpm,
                "sdk_retries": 0, "outer_retries": 0, "n": 1, "max_tokens": 16384, "temperature": .6,
                "http_max_connections": budget.concurrency, "http_max_keepalive_connections": budget.concurrency,
                "worker_threads": workers, "effective_submission_limit": effective_limit,
                "requests_per_second": requests_per_second,
                "pacing_semantics": "submission spacing, burst=1, no catch-up; actual wire timestamps recorded",
                "workload_sha256": fingerprint(workload), "remote": remote,
                "timeout_semantics": "SDK per-operation; not yet validated as total deadline",
                "measurement": "wire in-flight = request headers sent until SDK terminates; not server GPU concurrency",
                "connection_wait": "first transport event minus SDK start is a proxy incl. serialization/DNS; not exact pool wait",
                "token_threshold": "reported tokens stop NEW calls; in-flight/unknown usage can exceed threshold",
                "stage_sampling": "stratified by stage and prompt length, repeated fixed workload; not natural pipeline proportions"}
    if group_plan is not None:
        manifest['grouping'] = dict(coordinators=coordinators, groups=len(group_plan),
            plan_sha256=fingerprint(group_plan), prewarm_both_pools=True,
            semantics='one blocking coordinator per active node; same prompt per sample; shared SDK admission; no retry')
    manifest['stop_after_target_successes'] = stop_after_target_successes
    if source:
        manifest['handoff'] = {'source_run': str(source.path), 'source_manifest_sha256': fingerprint(source.manifest),
            'source_runs': source.source_runs(),
            'source_pending_at_start': source_pending, 'combined_concurrency': budget.concurrency,
            'semantics': 'one new sender; all old ancestors fully issued and only draining; conservative terminal reservations'}
    probe_sources = list(Path(__file__).parent.glob("*.py"))+[Path(__file__).parents[1]/"deepeye_efficiency_probe.py"]
    manifest["probe_source_sha256"] = {str(p.relative_to(Path(__file__).parents[2])):
                                        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(probe_sources)}
    store = RunStore.create(run_dir, manifest)
    attempt = store.begin_attempt("capacity", "probe", fingerprint(manifest))
    write_json(run_dir/"manifest.json", manifest)
    write_json(run_dir/"workload.json", workload)
    if group_plan is not None:
        write_json(run_dir/'group_plan.json', group_plan)
    lock, local = threading.Lock(), threading.local()
    stop, monitor_stop = threading.Event(), threading.Event()
    state = {"sdk": 0, "wire": 0, "peak_sdk": 0, "peak_wire": 0, "reason": None,
             "target_reached_at": None, "target_reached_clock": None, "successes_after_target": 0}
    groups = None
    handoff_peak = 0
    rows, resource_rows, commits = [], [], []
    start = time.perf_counter()

    def request_stop(reason):
        with lock:
            if state["reason"] is None:
                state["reason"] = reason
        stop.set()

    def trace(name, info):
        row = getattr(local, "row", None)
        if row is None:
            return
        if row["transport_start_delay"] is None:
            row["transport_start_delay"] = time.perf_counter()-local.started
        if name.endswith("send_request_headers.complete") and not row.get("wire_started"):
            row["wire_started"] = True
            row["wire_started_at"] = utc()
            with lock:
                state["wire"] += 1
                state["peak_wire"] = max(state["peak_wire"], state["wire"])
                if state['wire'] >= budget.concurrency and state['target_reached_at'] is None:
                    state['target_reached_at'] = utc()
                    state['target_reached_clock'] = time.perf_counter()

    def request_hook(request):
        request.extensions["trace"] = trace

    def monitor():
        while not monitor_stop.is_set():
            try:
                r = resources()
                with lock:
                    r.update(sdk_inflight=state["sdk"], wire_inflight=state["wire"])
                if groups:
                    with groups.lock:
                        r.update(active_coordinators=groups.active, queued_samples=groups.queue.qsize())
                resource_rows.append(r)
                store.append_event(attempt, "resource", r)
                if r["system_memory_percent"] >= 85 or r["disk_free_bytes"] < 2*1024**3:
                    request_stop("local_resource_pressure")
                if (run_dir/"STOP").exists():
                    request_stop("operator_stop_file")
            except Exception as e:
                request_stop("resource_recording_failure:"+type(e).__name__)
                return
            monitor_stop.wait(2)

    def one(i, submitted_at, submitted_clock, item_index=None, group_metadata=None):
        item = workload[i % len(workload) if item_index is None else item_index]
        row = {"request_no": i, "stratum": item.get("stratum", item["stage"]),
               "source_event_id": item.get("source_event_id"), "prompt_sha256": item["prompt_sha256"],
               "submitted_at": submitted_at, "executor_wait_seconds": time.perf_counter()-submitted_clock,
               "started_at": utc(), "http_success": False, "nonempty": False, "parsed": False,
               "usage": None, "transport_start_delay": None}
        row.update(group_metadata or {})
        store.append_event(attempt, "request", dict(row))
        t = time.perf_counter()
        local.row, local.started = row, t
        with lock:
            state["sdk"] += 1
            state["peak_sdk"] = max(state["peak_sdk"], state["sdk"])
        response = None
        try:
            response = client.chat.completions.create(model=model, messages=item["messages"],
                        n=1, max_tokens=16384, temperature=.6, timeout=budget.timeout)
            row["http_success"] = True
            row["response_id"] = response.id
            row["usage"] = response.usage.model_dump() if response.usage else None
            content = response.choices[0].message.content if response.choices else None
            row["nonempty"] = bool(content and content.strip())
            try:
                row["parsed"] = bool(parse_response(item, content or ""))
            except Exception as e:
                row["parser_error_type"] = type(e).__name__
        except Exception as e:
            row.update(error_type=type(e).__name__, status_code=getattr(e, "status_code", None))
            row["error_message"] = str(e).replace(api_key, "[redacted]")[:1000]
            # Provider error body is useful for RPM/TPM discrimination; redact credential.
            body = getattr(e, "body", None)
            row["error_body"] = json.dumps(body, ensure_ascii=False, default=str)[:4000].replace(api_key, "[redacted]") if body else None
            row["request_id"] = getattr(e, "request_id", None)
            if row["status_code"] in {401, 403}:
                request_stop("authentication_error")
        finally:
            row["elapsed_seconds"] = time.perf_counter()-t
            row["finished_at"] = utc()
            with lock:
                state["sdk"] -= 1
                if row.get("wire_started"):
                    state["wire"] -= 1
                if (state['target_reached_clock'] is not None and row['http_success'] and row['parsed']):
                    state['successes_after_target'] += 1
                    row['successful_return_after_target'] = True
            local.row = None
        payload = {**row, "response": response.model_dump() if response else None}
        committed = time.perf_counter()
        store.append_event(attempt, "response" if response else "error", payload)
        row["record_commit_seconds"] = time.perf_counter()-committed
        return row

    http = DefaultHttpxClient(limits=httpx.Limits(max_connections=budget.concurrency,
                       max_keepalive_connections=budget.concurrency),
                       event_hooks={"request": [request_hook]}, trust_env=False)
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=budget.timeout, http_client=http)
    monitor_thread = threading.Thread(target=monitor, name="probe-resource", daemon=True)
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda number, frame: request_stop("operator_signal"))
    monitor_thread.start()
    issued, tokens, recent = 0, 0, deque(maxlen=50)
    last_progress = start
    group_waiters = {}
    try:
        with cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="probe-sdk") as pool:
            if group_plan is not None:
                warm_pool(pool, workers)
                groups = GroupQueue(group_plan, coordinators,
                    lambda kind, data: store.append_event(attempt, kind, data), stop)
                groups.start(coordinators)
                store.append_event(attempt, 'pools_ready', resources())
            pending = set()
            while pending or issued < budget.max_requests:
                if groups and groups.done() and not pending:
                    break
                with lock:
                    confirmed = (stop_after_target_successes is not None and
                                 state['successes_after_target'] >= stop_after_target_successes)
                if confirmed:
                    request_stop('target_reached_and_returns_confirmed')
                if groups and (stop.is_set() or issued >= budget.max_requests):
                    groups.cancel_unsent()
                if source and not stop.is_set():
                    try:
                        source_pending = source.pending()
                    except Exception as e:
                        request_stop('handoff_source_failure:'+type(e).__name__)
                allowed = min(effective_limit, max(0, budget.concurrency-source_pending))
                if time.perf_counter()-start >= budget.admission_seconds:
                    request_stop("admission_deadline")
                while not stop.is_set() and len(pending) < allowed and issued < budget.max_requests:
                    if time.perf_counter()-start >= budget.admission_seconds:
                        request_stop("admission_deadline")
                        break
                    if pacer.delay(time.perf_counter()) > 0:
                        break
                    job = groups.take() if groups else None
                    if groups and job is None:
                        break
                    try:
                        if source:
                            reserved = source_pending+len(pending)+1
                            handoff_peak = max(handoff_peak, reserved)
                            store.append_event(attempt, 'handoff_admission', {'at': utc(), 'request_no': issued,
                                'source_pending': source_pending, 'new_reserved': len(pending)+1,
                                'reserved_total': reserved, 'combined_limit': budget.concurrency})
                        submitted = time.perf_counter()
                        future = pool.submit(one, issued, utc(), submitted,
                            job[0] if job else None, job[1] if job else None)
                        pending.add(future)
                        if job:
                            group_waiters[future] = job[2]
                        # Logging or submit itself may be slow. Never convert that
                        # elapsed time into credits for a catch-up admission burst.
                        pacer.admitted(time.perf_counter())
                        issued += 1
                    except RuntimeError:
                        if job:
                            job[2].cancel()
                        request_stop("thread_creation_failure")
                        break
                if issued >= budget.max_requests:
                    # Freeze why admission ended before later drain-only guards.
                    request_stop("request_budget_completed")
                if not pending:
                    if stop.is_set() or issued >= budget.max_requests:
                        break
                    stop.wait(.01 if groups else .1 if allowed == 0 else min(.25, pacer.delay(time.perf_counter())))
                    continue
                timeout = .25
                if not stop.is_set() and issued < budget.max_requests and len(pending) < allowed:
                    timeout = min(timeout, max(.001, pacer.delay(time.perf_counter())))
                done, pending = cf.wait(pending, timeout=timeout, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    try:
                        row = future.result()
                    except Exception:
                        if future in group_waiters:
                            group_waiters.pop(future).set_exception(RuntimeError('request worker failed'))
                        request_stop("recording_or_worker_failure")
                        raise
                    if future in group_waiters:
                        group_waiters.pop(future).set_result(row)
                    rows.append(row)
                    commits.append(row["record_commit_seconds"])
                    recent.append(row)
                    tokens += (row.get("usage") or {}).get("total_tokens") or 0
                    if tokens >= budget.token_stop:
                        request_stop("reported_token_threshold")
                # Avoid counting just one isolated transient error as overload.
                if len(recent) >= 20 and sum(r.get("status_code") == 429 for r in recent) >= max(5, len(recent)*.2):
                    request_stop("frequent_rate_limits")
                if len(recent) >= 50 and sum(not r["http_success"] for r in recent) >= len(recent)*.2:
                    request_stop("frequent_request_failures")
                if done and (time.perf_counter()-last_progress >= 10 or not pending):
                    print(json.dumps({"issued": issued, "completed": len(rows), "inflight": len(pending),
                                      "reported_tokens": tokens, "stop": state["reason"]}), flush=True)
                    last_progress = time.perf_counter()
        elapsed = time.perf_counter()-start
    finally:
        # SDK executor has drained before reaching here, even on an exception.
        # Resolve pending group waiters so shutdown can never wait on unsent work.
        for future, waiter in group_waiters.items():
            try:
                waiter.set_result(future.result())
            except Exception as e:
                waiter.set_exception(e)
        try:
            if groups:
                groups.close()
        finally:
            monitor_stop.set()
            monitor_thread.join()
            client.close()
            if source:
                source.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if sys.exc_info()[0] is not None:
                store.close()  # Preserve incomplete evidence, release the run lock on failure.
    summary = summarize(rows, elapsed)
    summary.update(target_concurrency=budget.concurrency, peak_sdk_inflight=state["peak_sdk"],
                   peak_wire_inflight=state["peak_wire"], stop_reason=state["reason"] or "request_budget_completed",
                   issued=issued, outstanding=issued-len(rows),
                   commit_seconds=distribution(commits), resources_peak={
                       k: max((r[k] for r in resource_rows), default=None)
                       for k in ("threads", "rss_bytes", "fds", "system_memory_percent", "swap_used_bytes")},
                   resources_before=resource_rows[0] if resource_rows else None,
                   resources_after=resources(), manifest=manifest)
    summary['executor_wait_seconds'] = distribution([r['executor_wait_seconds'] for r in rows])
    summary['target_confirmation'] = dict(reached=state['target_reached_at'] is not None,
        reached_at=state['target_reached_at'], successful_returns_after_target=state['successes_after_target'],
        required_successful_returns=stop_after_target_successes)
    if groups:
        summary['group_summary'] = groups.summary()
    if source:
        summary['handoff'] = {**manifest['handoff'], 'peak_reserved_total': handoff_peak,
                              'note': 'reservation peak is conservative; reconstruct actual combined wire peak from both runs'}
    store.finish_attempt(attempt, "succeeded" if len(rows)==issued else "failed", summary)
    summary["integrity"] = store.verify()
    store.close()
    write_json(run_dir/"summary.json", summary)
    return summary
