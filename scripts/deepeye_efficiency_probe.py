"""Independent B0-B4 pre-change tests. Run with python -m scripts.deepeye_efficiency_probe.

No production imports run experiments. Remote mode requires explicit finite budgets.
All outputs use new run directories; this tool cannot resume/overwrite an old run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import time

from scripts.efficiency_probe.core import ProbeBudget, fingerprint, resources, run_requests, utc, write_json
from scripts.efficiency_probe.local import LocalServer, thread_probe, group_probe
from scripts.efficiency_probe.legacy import characterize
from scripts.efficiency_probe.workload import prepare_workload
from scripts.efficiency_probe.groups import build_plan, validate_plan

CODE = Path(__file__).resolve().parents[1]
BASE = CODE/"baselines_reproduce/deepeye_bird_interact"
OLD = BASE/"runs/deepeye_bird_interact_formal_budget_20260913_native/run.sqlite3"


def baseline(out):
    from scripts.deepeye_bird_interact_run import code_source_hashes
    import psutil
    out.mkdir(parents=True, exist_ok=False)
    def command(args):
        return subprocess.check_output(args, cwd=CODE, text=True).strip()
    c = sqlite3.connect(OLD.resolve().as_uri()+"?mode=ro", uri=True)
    old_manifest = json.loads(c.execute("select payload_json from manifest").fetchone()[0])
    snapshot = {"at": utc(), "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_branch": command(["git", "branch", "--show-current"]),
        "git_status_at_snapshot": command(["git", "status", "--porcelain"]),
        "git_tracked_diff": command(["git", "diff", "HEAD", "--stat"]),
        "production_hashes": code_source_hashes(), "resources": resources(),
        "rlimit_nofile": resource.getrlimit(resource.RLIMIT_NOFILE),
        "rlimit_nproc": resource.getrlimit(resource.RLIMIT_NPROC),
        "kernel_limits": command(["sysctl", "kern.num_threads", "kern.num_taskthreads"]),
        "dependencies": {k: importlib.metadata.version(k) for k in ("openai", "httpx2", "aiohttp", "psutil")},
        "old_run": str(OLD), "old_manifest": old_manifest,
        "old_max_event": c.execute("select max(event_id) from events").fetchone()[0],
        "old_pipeline_outcomes": dict(c.execute("select f.status,count(*) from finishes f join attempts a using(attempt_id) where a.stage='pipeline' group by f.status")),
        "production_processes": [{"pid": p.pid, "name": p.info["name"]} for p in psutil.process_iter(["name", "cmdline"])
            if p.info["cmdline"] and any("deepeye_bird_interact_run.py" in x or "rc_evaluation.deepeye.cli" in x
                                            for x in p.info["cmdline"][1:])],
        "scope": "B0-B4 only; production unchanged; no old run resume, retry1, RC or SQL execution",
        "remote_authorization": {"levels": [300,600,1200,2000], "max_requests_per_level": "2C",
              "provider_tpm_user_reported": 5000000, "authorized_by": "user reply in current task"}}
    c.close()
    plan = BASE/"rc_evaluation/deepeye_bird_interact_formal_budget_20260913/campaign_plan.json"
    snapshot["campaign_plan"] = json.loads(plan.read_text())
    automation = Path(os.environ.get("CODEX_HOME", str(Path.home()/".codex")))/"automations/deepeye-50/automation.toml"
    if automation.is_file():
        import tomllib
        a = tomllib.loads(automation.read_text())
        snapshot["old_automation"] = {k: a.get(k) for k in ("id", "name", "status", "updated_at")}
    write_json(out/"baseline.json", snapshot)
    return snapshot


def regression(out):
    out.mkdir(parents=True, exist_ok=False)
    results = []
    commands = [[sys.executable, "-E", "-B", "-m", "unittest", "discover", "-s", "tests", "-q"],
                [sys.executable, "-E", "-B", "-m", "unittest", "discover", "-s",
                 "scripts/rc_evaluation/deepeye/tests", "-t", ".", "-q"]]
    for i, cmd in enumerate(commands):
        start = time.monotonic()
        p = subprocess.run(cmd, cwd=CODE, env={**os.environ, "DEEPEYE_TEST_PG": "0"},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (out/f"suite_{i}.log").open("x") as f:
            f.write(p.stdout)
        results.append({"command": cmd, "exit_code": p.returncode,
                        "elapsed_seconds": time.monotonic()-start, "log": f"suite_{i}.log"})
    write_json(out/"summary.json", results)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["baseline", "regression", "legacy", "threads", "local-http", "local-groups", "local-group-http", "workload", "remote"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, default=OLD)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--workers", type=int, help="SDK worker pool size; default equals concurrency")
    p.add_argument("--requests-per-second", type=float, help="Smooth request admission, burst=1")
    p.add_argument("--draining-run", type=Path, help="Fully-issued old probe; subtract its outstanding calls from the shared concurrency cap")
    p.add_argument("--coordinators", type=int, default=256)
    p.add_argument("--groups", type=int, default=1200)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument('--group-profile', choices=['generation', 'revision'],
                   help='Use blocking coordinators with native node sample counts (4, or mixed 1/5)')
    p.add_argument('--stop-after-target-successes', type=int,
                   help='Stop new calls after the real wire target is reached and this many valid returns arrive')
    p.add_argument('--barrier-timeout', type=float, default=120,
                   help='Local HTTP first-wave concurrency barrier timeout')
    p.add_argument("--fixture-delay", type=float, default=.05)
    p.add_argument("--requests", type=int)
    p.add_argument("--admission-seconds", type=float)
    p.add_argument("--token-stop", type=int)
    p.add_argument("--timeout", type=float, default=660)
    p.add_argument("--workload", type=Path)
    p.add_argument("--authorize-remote", action="store_true")
    p.add_argument("--provider-tpm", type=int)
    p.add_argument("--env-file", type=Path, default=CODE/"config/.env")
    args = p.parse_args(argv)
    if args.mode == "baseline":
        result = baseline(args.output)
    elif args.mode == "regression":
        result = regression(args.output)
    elif args.mode in {"legacy", "threads", "workload"}:
        args.output.mkdir(parents=True, exist_ok=False)
        result = characterize() if args.mode == "legacy" else thread_probe(args.concurrency) if args.mode == "threads" else prepare_workload(args.source)
        write_json(args.output/"summary.json", result)
    elif args.mode == "local-groups":
        result = group_probe(args.output, workers=args.workers if args.workers is not None else args.concurrency,
                             concurrency=args.concurrency, coordinators=args.coordinators,
                             groups=args.groups, samples=args.samples, delay=args.fixture_delay)
    elif args.mode == 'local-group-http':
        if not args.group_profile:
            p.error('local-group-http requires --group-profile')
        fixture = [dict(stage=stage, branch_path=[node], parser='sql', source_event_id=i,
                        prompt_sha256=f'local-{i}', messages=[dict(role='user', content=f'local fixture {i}')])
                   for i, (stage, node) in enumerate([
                       ('sql_generation', 'generation.dc'), ('sql_revision', 'revision.SyntaxChecker'),
                       ('sql_revision', 'revision.OrderByNullChecker'), ('sql_revision', 'revision.ResultChecker')])]
        if args.workload:
            data = json.loads(args.workload.read_text())
            fixture = data if isinstance(data, list) else data['workload']
        plan = build_plan(fixture, args.group_profile, args.groups)
        validate_plan(plan, fixture, args.coordinators)
        requests = args.requests or sum(s['samples'] for s in plan)
        budget = ProbeBudget(args.concurrency, requests, args.admission_seconds or 300, 10**9,
                             max(180, args.barrier_timeout+60))
        with LocalServer(min(args.concurrency, args.workers or args.concurrency, requests),
                         delay=args.fixture_delay, barrier_timeout=args.barrier_timeout) as server:
            result = run_requests(args.output, fixture, budget, server.url, 'local-fixture', 'fixture',
                workers=args.workers, requests_per_second=args.requests_per_second,
                group_plan=plan, coordinators=args.coordinators,
                stop_after_target_successes=args.stop_after_target_successes)
            write_json(args.output/'server.json', dict(peak=server.peak, received=server.received,
                barrier_timed_out=server.barrier_timed_out, active_at_client_finish=server.active))
    elif args.mode == "local-http":
        budget = ProbeBudget(args.concurrency, args.requests or args.concurrency*2, 180, 10**9, 120)
        fixture = [{"messages": [{"role": "user", "content": "local fixture"}], "prompt_sha256": "local",
                    "source_event_id": None, "stage": "local_fixture", "parser": "sql"}]
        with LocalServer(min(args.concurrency, args.workers if args.workers is not None else args.concurrency), barrier_timeout=90) as server:
            result = run_requests(args.output, fixture, budget, server.url, "local-fixture", "fixture",
                                  workers=args.workers, requests_per_second=args.requests_per_second)
            write_json(args.output/"server.json", {"peak": server.peak, "received": server.received,
                "barrier_timed_out": server.barrier_timed_out, "active_at_client_finish": server.active})
    else:
        if not all((args.requests, args.admission_seconds, args.token_stop, args.workload)):
            p.error("remote needs --requests, --admission-seconds, --token-stop and --workload")
        budget = ProbeBudget(args.concurrency, args.requests, args.admission_seconds, args.token_stop,
                             args.timeout, args.authorize_remote)
        budget.validate(remote=True)  # Fail closed BEFORE reading credentials or opening client.
        data = json.loads(args.workload.read_text())
        workload = data if isinstance(data, list) else data['workload']
        plan = build_plan(workload, args.group_profile, args.groups) if args.group_profile else None
        if plan is not None:
            validate_plan(plan, workload, args.coordinators)
        from scripts.deepeye_bird_interact_smoke import read_environment
        env = read_environment(args.env_file)
        result = run_requests(args.output, workload, budget, env["DASH_BASE_URL"], env["DASH_API_KEY"],
                              env["DASH_MODELS"], provider_tpm=args.provider_tpm,
                              workers=args.workers, requests_per_second=args.requests_per_second,
                              draining_run=args.draining_run, group_plan=plan,
                              coordinators=args.coordinators,
                              stop_after_target_successes=args.stop_after_target_successes)
    # Workload and baseline contain long prompts. Do not flood terminal with them.
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "summary": {k:v for k,v in result.items() if k not in {"workload", "old_manifest", "campaign_plan", "manifest"}}}
                     if isinstance(result, dict) else result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
