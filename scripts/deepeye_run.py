"""Crash-safe, gold-free DeepEye SQL runs from explicitly bound native workloads.

The four native stages share one run-owned sampling and HTTP runtime. All
selected questions are eligible; their stage dependencies remain sequential.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
import signal
from pathlib import Path
import sys
import threading


CODE_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = CODE_ROOT / "baselines/DeepEye-SQL"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))

STAGES = ("schema_linking", "sql_generation", "sql_revision", "sql_selection")
SELECTION_SHORTCUT_THRESHOLD = 0.6  # Official Qwen3.6 BIRD template.


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda candidate: candidate.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def code_source_hashes() -> dict[str, str]:
    """Hash executable Python and dependency locks without binding output paths."""

    baseline_files = list((BASELINE_ROOT / "app").rglob("*.py"))
    baseline_files.append(BASELINE_ROOT / "runner/create_vector_db_parallel.py")
    adapter_files = list((CODE_ROOT / "scripts/baseline_adapters/deepeye").glob("*.py"))
    support_files = [CODE_ROOT / "scripts/deepeye_bird_interact_precompute.py",
                     Path(__file__).resolve()]
    lock_files = [CODE_ROOT / "pyproject.toml", CODE_ROOT / "uv.lock"]
    for path in baseline_files + adapter_files + support_files + lock_files:
        if not path.is_file():
            raise FileNotFoundError(f"Run source file is missing: {path}")
    return {
        "baseline_python_sha256": _tree_sha256(baseline_files, BASELINE_ROOT),
        "adapter_python_sha256": _tree_sha256(adapter_files, CODE_ROOT),
        "entrypoints_sha256": _tree_sha256(support_files, CODE_ROOT),
        "dependency_locks_sha256": _tree_sha256(lock_files, CODE_ROOT),
    }


def build_effective_config(environment: dict[str, str], args) -> dict:
    """Return every result-affecting operational choice, excluding credentials."""

    if getattr(args, 'workload', None):
        return _native_effective_config(environment, args)
    return {
        "profile": "shared-sampling-v1",
        "chat": {
            "model": environment["DASH_MODELS"],
            "endpoint": environment["DASH_BASE_URL"].rstrip("/"),
            "max_tokens": args.max_tokens,
            "temperature": 0.6,
            "thinking_budget": args.thinking_budget,
            "timeout_seconds": args.chat_timeout,
            "sdk_max_retries": 0,
            "sample_max_attempts": 4,
            "extractor_max_retries": args.extractor_retries,
            "n_call_strategy": "split",
            "max_request_n": 1,
        },
        "postgres": {
            "host": environment["PG_HOST"],
            "port": int(environment["PG_PORT"]),
            "principal": environment["PG_USER"],
            "sslmode": args.pg_sslmode,
            "read_only": True,
            "statement_timeout_seconds": 30,
            "execution_policy": {
                "version": "postgres-original-sql-v1",
                "meta_fence": False,
                "sql_rewrite": False,
                "result_row_limit": None,
                "search_path": "pg_catalog,public",
                "single_statement": "extended_protocol",
                "read_only": True,
            },
        },
        "dataset": {
            "sql_execution_timeout_seconds": 30,
            "max_value_example_length": 100,
        },
        "few_shot": {"mode": "static_independent_bird_train", "examples_per_item": 3},
        "scheduler": {"mode": "all_questions", "concurrency_unit": "model_request"},
        "runtime": {"version": "shared-sampling-runtime-v1", **runtime_limits(args)},
        "admission": admission_settings(args),
        "stages": {
            "schema_linking": {
                "direct_linking_sampling_budget": args.direct_linking_budget,
                "reversed_linking_sampling_budget": args.reversed_linking_budget,
                "value_distance_threshold": 0.05,
            },
            "sql_generation": {
                "dc_sampling_budget": args.dc_generation_budget,
                "skeleton_sampling_budget": args.skeleton_generation_budget,
                "icl_sampling_budget": args.icl_generation_budget,
            },
            "sql_revision": {
                "checker_sampling_budget": args.revision_checker_budget,
                "checkers": [
                    "SyntaxChecker", "JoinChecker", "OrderByLimitChecker", "TimeChecker",
                    "SelectChecker", "MaxMinChecker", "OrderByNullChecker", "ResultChecker",
                ],
            },
            "sql_selection": {
                "evaluator_sampling_budget": args.selection_evaluator_budget,
                "filter_top_k_sql": 2,
                "shortcut_consistency_score_threshold": SELECTION_SHORTCUT_THRESHOLD,
                "timing_refine_repeat": 2,
            },
        },
    }


def build_manifest(effective_config: dict, sources: dict, item_bindings: list[dict]) -> dict:
    """Build the immutable, secret-free run identity consumed by ``RunStore``."""

    ordered = sorted((dict(binding) for binding in item_bindings), key=lambda row: row["task_key"])
    keys = [binding["task_key"] for binding in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("Run manifest contains duplicate task keys")
    return {
        "format": "deepeye-run-v2",
        "fingerprint_algorithm": "manifest-digest-v2",
        "scope": "gold-free four-stage native workflow from frozen prepared inputs",
        "workflow": list(STAGES),
        "accuracy_evaluated": False,
        "dynamic_few_shot_retrieval": False,
        "effective_config": effective_config,
        "sources": dict(sources),
        "item_count": len(ordered),
        "items": ordered,
    }


def select_tasks(tasks, *, variants=None, item_keys=None):
    """Select only explicit ``variant/instance_id`` identities without ambiguity."""

    variants = set(variants or ("lite", "full"))
    invalid_variants = variants.difference(("lite", "full"))
    if invalid_variants:
        raise ValueError(f"Unknown variants: {sorted(invalid_variants)}")
    by_key = {f"{variant}/{item.instance_id}": (variant, item) for variant, item in tasks}
    if len(by_key) != len(tasks):
        raise ValueError("Input task identities are not unique")
    if item_keys:
        requested = list(item_keys)
        if any(key.count("/") != 1 for key in requested):
            raise ValueError("Item filters must use variant/instance_id")
        unknown = set(requested).difference(by_key)
        if unknown:
            raise ValueError(f"Unknown item keys: {sorted(unknown)}")
        requested_set = set(requested)
    else:
        requested_set = set(by_key)
    return [pair for key, pair in by_key.items()
            if pair[0] in variants and key in requested_set]


def select_probe_tasks(tasks, *, per_variant: int):
    """Deterministic, gold-free coverage sample: one per DB, then fill each variant."""
    if per_variant < 1:
        raise ValueError("Probe size must be positive")
    selected = []
    for variant in sorted({variant for variant, _ in tasks}):
        rows = [pair for pair in tasks if pair[0] == variant]
        rows.sort(key=lambda pair: hashlib.sha256(
            f"deepeye-probe-v1/{variant}/{pair[1].instance_id}".encode()).hexdigest())
        by_db = {}
        for pair in rows:
            by_db.setdefault(pair[1].database_id, pair)
        if not len(by_db) <= per_variant <= len(rows):
            raise ValueError(f"Probe size for {variant} must cover its {len(by_db)} databases "
                             f"and not exceed its {len(rows)} questions")
        chosen = {pair[1].instance_id for pair in by_db.values()}
        for pair in rows:
            if len(chosen) >= per_variant:
                break
            chosen.add(pair[1].instance_id)
        selected.extend(pair for pair in rows if pair[1].instance_id in chosen)
    return selected


def admission_settings(args) -> dict:
    if getattr(args, 'adaptive_concurrency', False) or any(
            getattr(args, 'concurrency_' + name, None) is not None
            for name in ('initial', 'step', 'min', 'max', 'window')):
        raise ValueError('Legacy adaptive pipeline flags are unsupported; configure explicit request/runtime limits')
    return {"enabled": True, "adaptive": False, "pipeline": {"mode": "all_questions"},
            "postgres_limit": args.pg_concurrency}


def runtime_limits(args):
    from scripts.baseline_adapters.deepeye.request_dispatch import RequestLimits
    defaults = asdict(RequestLimits())
    for name in defaults:
        value = getattr(args, 'chat_timeout' if name == 'request_timeout' else name, None)
        if value is not None:
            defaults[name] = value
    return asdict(RequestLimits(**defaults))


@contextmanager
def sampling_runtime(recorder, args):
    """Own the one runtime until all traced work and resource cleanup finishes."""
    from scripts.baseline_adapters.deepeye.run_resources import SamplingRuntime
    runtime = SamplingRuntime(stop_event=recorder.stop_event, emit=recorder.record_admission,
                              **runtime_limits(args))
    previous = {}
    signals = 0
    def stop(signum, frame):
        nonlocal signals
        signals += 1
        runtime.stop(cancel_active=signals > 1)
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, stop)
        yield runtime
        if runtime.dispatch.fatal_error is not None:
            raise runtime.dispatch.fatal_error
    finally:
        try:
            runtime.close()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


@contextmanager
def admission_context(recorder, args, *, population=1):
    """Bookkeep all questions and independently limit PostgreSQL execution."""
    settings = admission_settings(args)
    from scripts.baseline_adapters.deepeye import backend_hooks
    from scripts.baseline_adapters.deepeye.run_admission import (
        FixedAdmission,
    )
    from scripts.baseline_adapters.deepeye.run_slots import WorkflowSlots

    slots = WorkflowSlots(population)
    workload = _workload(args)
    if workload is not None and workload['benchmark'] != 'bird_interact':
        yield {'pipeline': slots}
        return
    postgres = FixedAdmission(settings["postgres_limit"], emit=recorder.record_admission)
    original_pg = backend_hooks.execute_postgres_sql
    backend_hooks.execute_postgres_sql = lambda *args, **kwargs: postgres(original_pg, args, kwargs)
    try:
        yield {"pipeline": slots, "postgres": postgres}
    finally:
        backend_hooks.execute_postgres_sql = original_pg


def _prepare_interact_inputs(precompute_dir: Path, few_shot_source: Path, *, variants=None,
                   item_keys=None, inventory_loader=None, item_loader=None,
                   example_loader=None, code_source_hasher=None, probe_per_variant=None):
    """Load gold-free precomputed items and attach three static cross-domain examples."""

    from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
    from scripts.baseline_adapters.deepeye.precompute_pipeline import (
        PrecomputedInputReader, question_row, read_record,
    )
    from scripts.deepeye_bird_interact_precompute import load_inputs
    from scripts.baseline_adapters.deepeye.runtime_config import IndependentExampleReader

    inventory_loader = inventory_loader or load_inputs
    code_source_hasher = code_source_hasher or code_source_hashes
    precompute_dir = Path(precompute_dir).resolve()
    few_shot_source = Path(few_shot_source).resolve()
    input_reader = PrecomputedInputReader(precompute_dir)
    example_reader = IndependentExampleReader(few_shot_source)

    verification_path = precompute_dir / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    expected_counts = {"lite": 195, "full": 410}
    if (verification.get("complete") is not True
            or verification.get("question_counts") != expected_counts
            or verification.get("expected_questions") != 605
            or verification.get("database_count") != 40
            or verification.get("errors") != {}):
        raise ValueError("Precomputation must be verified complete for all 605 questions and 40 databases")

    all_tasks, _ = inventory_loader()
    selected = select_tasks(all_tasks, variants=variants, item_keys=item_keys)
    if probe_per_variant is not None:
        selected = select_probe_tasks(selected, per_variant=probe_per_variant)
    if not selected:
        raise ValueError("Task filters selected no questions")
    prepared, bindings = [], []
    for variant, expected_item in selected:
        if item_loader is None:
            item = input_reader.load(
                variant, expected_item.instance_id, expected_item=expected_item)
        else:
            item = item_loader(precompute_dir, variant, expected_item.instance_id,
                               expected_item=expected_item)
        if getattr(item, "gold_sql", ""):
            raise ValueError("Target gold SQL must not enter the run")
        if example_loader is None:
            examples, provenance = example_reader.select(item, count=3)
        else:
            examples, provenance = example_loader(few_shot_source, item, count=3)
        if len(examples) != 3:
            raise ValueError("Exactly three independent static training examples are required")
        item.few_shot_examples = examples
        item.few_shot_preparation_metadata = {
            "mode": "static_independent_bird_train", "num_examples": 3,
        }
        keywords, retrieval = input_reader.records(variant, item.instance_id)
        provenance_rows = provenance.get("examples") if isinstance(provenance, dict) else None
        if (not isinstance(provenance_rows, list) or len(provenance_rows) != 3
                or any(not isinstance(row, dict) or type(row.get("source_row")) is not int
                       for row in provenance_rows)):
            raise ValueError("Few-shot provenance must identify exactly three source rows")
        bindings.append({
            "task_key": f"{variant}/{item.instance_id}",
            "database_id": item.database_id,
            "question_sha256": fingerprint(question_row(item)),
            "schema_sha256": fingerprint(item.database_schema),
            "keywords_sha256": keywords["content_hash"],
            "retrieval_sha256": retrieval["content_hash"],
            "few_shot_sha256": fingerprint(examples),
            "few_shot_source_rows": [{key: row[key] for key in
                                      ("source_row", "db_id", "dialect_conversion")}
                                     for row in provenance_rows],
        })
        prepared.append((variant, item))

    inputs_path = precompute_dir / "inputs.json"
    config_path = precompute_dir / "run_config.json"
    # Parsing sealed records rejects corrupted population/config artifacts as well.
    inputs_record = read_record(inputs_path)
    config_record = read_record(config_path)
    sources = {
        "precompute_inputs_content_hash": inputs_record["content_hash"],
        "precompute_config_content_hash": config_record["content_hash"],
        "precompute_semantic_config": config_record["config"],
        "few_shot_source_sha256": example_reader.source_sha256,
        "precompute_population": {"lite": 195, "full": 410, "databases": 40},
        "locators": {"precompute_dir": str(precompute_dir),
                     "few_shot_source": str(few_shot_source)},
        "code": code_source_hasher(),
    }
    return prepared, bindings, sources


def prepare_inputs(precompute_dir=None, few_shot_source=None, *, workload=None,
                   variants=None, item_keys=None, **testing_overrides):
    from scripts.baseline_adapters.deepeye.workloads import load_items, external_id, task_key
    if workload is None:
        tasks, bindings, sources = _prepare_interact_inputs(
            precompute_dir, few_shot_source, variants=variants, item_keys=item_keys,
            **testing_overrides)
        for (partition, item), binding in zip(tasks, bindings):
            binding.update(partition=partition, external_id=external_id(item),
                           benchmark='bird_interact', split=partition,
                           task_key=task_key(partition, item))
        return tasks, bindings, sources
    tasks, bindings, sources = load_items(workload)
    available = {binding['task_key'] for binding in bindings}
    if item_keys and set(item_keys).difference(available):
        raise ValueError(f'Unknown item keys: {sorted(set(item_keys).difference(available))}')
    chosen = set(item_keys or available)
    selected = [(pair, binding) for pair, binding in zip(tasks, bindings)
                if binding['task_key'] in chosen and
                (not variants or pair[0] in variants or binding['split'] in variants)]
    if not selected:
        raise ValueError('Task filters selected no questions')
    if testing_overrides.get('probe_per_variant') is not None:
        count = testing_overrides['probe_per_variant']
        # General probe uses typed identity without changing native fields.
        selected.sort(key=lambda pair: hashlib.sha256(pair[1]['task_key'].encode()).hexdigest())
        by_db = {}
        for pair in selected:
            by_db.setdefault(pair[1]['database_id'], pair)
        if not len(by_db) <= count <= len(selected):
            raise ValueError('Probe size must cover all databases and fit the selected workload')
        keys = {pair[1]['task_key'] for pair in by_db.values()}
        for pair in selected:
            if len(keys) == count:
                break
            keys.add(pair[1]['task_key'])
        selected = [pair for pair in selected if pair[1]['task_key'] in keys]
    sources['code'] = testing_overrides.get('code_source_hasher', code_source_hashes)()
    return [pair[0] for pair in selected], [pair[1] for pair in selected], sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from scripts.baseline_adapters.deepeye.run_operations import add_sample_commands
    add_sample_commands(commands)
    for name in ("prepare", "run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == 'resume':
            command.add_argument('--unfinished-only', action='store_true',
                                 help='Validate full manifest; execute only nonterminal native items')
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--precompute-dir", type=Path)
        source.add_argument("--workload", type=Path)
        command.add_argument("--env-file", type=Path, default=CODE_ROOT / "config/.env")
        command.add_argument("--few-shot-source", type=Path)
        command.add_argument("--variant", action="append", choices=("lite", "full"))
        command.add_argument("--item", dest="item_keys", action="append",
                             help="Exact variant/instance_id; repeat to select several")
        command.add_argument("--workers", type=int,
                             help="Legacy pipeline throttle; rejected in shared runtime mode")
        command.add_argument("--inner-workers", type=int,
                             help="Legacy per-runner pool width; use --coordinator-workers")
        command.add_argument('--request-limit', type=int, default=8000)
        command.add_argument('--request-workers', type=int, default=8000)
        command.add_argument('--coordinator-workers', type=int, default=6000,
                             help='One shared pool across all questions and native branches')
        command.add_argument('--http-connections', type=int, default=8000)
        command.add_argument('--request-start-rate', dest='start_rate', type=float, default=50.0)
        command.add_argument('--retry-delay', type=float, default=0.0)
        command.add_argument("--probe-per-variant", type=int,
                             help="Gold-free coverage sample per selected variant; includes every database")
        command.add_argument("--adaptive-concurrency", action="store_true",
                             help="Legacy adaptive pipeline throttle; rejected in shared runtime mode")
        command.add_argument("--concurrency-initial", type=int)
        command.add_argument("--concurrency-step", type=int)
        command.add_argument("--concurrency-min", type=int)
        command.add_argument("--concurrency-max", type=int)
        command.add_argument("--concurrency-window", type=float)
        command.add_argument("--pg-concurrency", type=int, default=10,
                             help="Independent PostgreSQL cap; never expanded to model-worker count")
        command.add_argument("--max-tokens", type=int, default=16384)
        command.add_argument("--thinking-budget", type=int, help='Legacy thinking toggle; rejected in this version')
        command.add_argument("--chat-timeout", type=int, default=660,
                             help='Deadline on the actual HTTP operation, including response reads')
        command.add_argument(
            "--pg-sslmode",
            choices=("disable", "allow", "prefer", "require", "verify-ca", "verify-full"),
            default="prefer",
            help="Explicit libpq SSL mode, frozen in the run manifest (default: prefer)",
        )
        command.add_argument("--extractor-retries", type=int, default=2,
                             help='Native constructor compatibility value; per-sample attempts remain four')
        command.add_argument("--direct-linking-budget", type=int, default=4)
        command.add_argument("--reversed-linking-budget", type=int, default=4)
        command.add_argument("--dc-generation-budget", type=int, default=4)
        command.add_argument("--skeleton-generation-budget", type=int, default=4)
        command.add_argument("--icl-generation-budget", type=int, default=4)
        command.add_argument("--revision-checker-budget", type=int, default=5)
        command.add_argument("--selection-evaluator-budget", type=int, default=5)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--run-dir", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--export-dir", type=Path, required=True)
    native = commands.add_parser('prepare-native', help='Run native value retrieval and few-shot preparation')
    native.add_argument('--workload', type=Path, action='append', required=True)
    native.add_argument('--output', type=Path)
    native.add_argument('--output-dir', type=Path)
    native.add_argument('--env-file', type=Path, default=CODE_ROOT / 'config/.env')
    native.add_argument('--preparation-workers', type=int, default=200)
    native.add_argument('--embedding-config', type=Path, help='JSON overrides for shared EmbeddingLimits')
    return parser


def _diagnostics(run_dir: Path) -> dict:
    from scripts.baseline_adapters.deepeye.run_store import RunStore
    from scripts.baseline_adapters.deepeye.run_usage import observed_usage

    with RunStore.open(run_dir.resolve(), read_only=True) as store:
        with store._read_snapshot():
            verification = store.verify()
            return {"manifest": store.manifest, "summary": store.summary(verification=verification),
                    "observed_usage": observed_usage(store), "verification": verification}


def _validate_run_args(parser: argparse.ArgumentParser, args) -> None:
    positive = (
        "max_tokens", "chat_timeout", "extractor_retries",
        "direct_linking_budget", "reversed_linking_budget", "dc_generation_budget",
        "skeleton_generation_budget", "icl_generation_budget",
        "revision_checker_budget", "selection_evaluator_budget",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("Token/time limits, retries, and all stage budgets must be positive")
    if args.thinking_budget is not None:
        parser.error('thinking-budget is unsupported in this experimental version')
    if args.workers is not None or args.inner_workers is not None:
        parser.error('Legacy workers/inner-workers are unsupported; use --coordinator-workers and --request-workers')
    if args.probe_per_variant is not None and (args.probe_per_variant < 1 or args.item_keys):
        parser.error("probe-per-variant must be positive and cannot be combined with --item")
    if args.pg_concurrency < 1:
        parser.error("pg-concurrency must be positive")
    try:
        admission_settings(args)
        runtime_limits(args)
    except ValueError as error:
        parser.error(str(error))


def _resolve_few_shot_source(requested: Path | None) -> Path:
    if requested is not None:
        source = requested.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"BIRD training source not found: {source}")
        return source
    candidates = (CODE_ROOT.parent / "BIRD/data/train/train.json",
                  CODE_ROOT.parent / "BIRD1.0/data/train/train.json")
    source = next((path.resolve() for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError("BIRD train.json not found; provide --few-shot-source")
    return source


def _prepare_command(args, *, execute: bool, resume: bool) -> dict:
    from scripts.baseline_adapters.deepeye.run_store import RunStore
    from scripts.baseline_adapters.deepeye.run_usage import observed_usage
    environment = read_environment(args.env_file.resolve(), args=args)
    source = None if getattr(args, 'workload', None) else _resolve_few_shot_source(args.few_shot_source)
    tasks, bindings, sources = prepare_inputs(
        args.precompute_dir, source, variants=args.variant, item_keys=args.item_keys,
        workload=getattr(args, 'workload', None), probe_per_variant=args.probe_per_variant,
    )
    effective = build_effective_config(environment, args)
    manifest = build_manifest(effective, sources, bindings)
    if resume:
        store_context = RunStore.open(args.run_dir.resolve(), expected_manifest=manifest)
    else:
        args.run_dir.resolve().parent.mkdir(parents=True, exist_ok=True)
        store_context = RunStore.create(args.run_dir.resolve(), manifest)
    with store_context as store:
        if execute:
            result = (_execute_pipeline(store, tasks, environment, args) if tasks else
                      {'succeeded': 0, 'failed': 0, 'executed': False})
        else:
            result = {"prepared": len(tasks), "executed": False}
        verification = store.verify()
        if not verification["ok"]:
            raise RuntimeError("RunStore verification failed")
        return {**result, "run_dir": str(args.run_dir.resolve()),
                "store_summary": store.summary(verification=verification), "observed_usage": observed_usage(store),
                "verification": verification}


def build_runtime_config(environment, args, run_dir: Path):
    """Create the formal budgets; constructor pools are replaced before work."""

    if getattr(args, 'workload', None):
        return _native_runtime_config(environment, args, run_dir)
    from scripts.baseline_adapters.deepeye.runtime_config import build_runtime_config as smoke_config

    native_output = Path(run_dir) / ".native_non_authoritative"
    preprocessed = CODE_ROOT / "data/bird_interact_lite"
    config = smoke_config(environment, "lite", preprocessed, native_output,
                          max_tokens=args.max_tokens, thinking_budget=args.thinking_budget)
    config.run_config.parallelism = runtime_limits(args)['coordinator_workers']
    config.llm_extractor_config.max_retry = args.extractor_retries
    config.schema_linking_config.direct_linking_sampling_budget = args.direct_linking_budget
    config.schema_linking_config.reversed_linking_sampling_budget = args.reversed_linking_budget
    config.sql_generation_config.dc_sampling_budget = args.dc_generation_budget
    config.sql_generation_config.skeleton_sampling_budget = args.skeleton_generation_budget
    config.sql_generation_config.icl_sampling_budget = args.icl_generation_budget
    config.sql_revision_config.checker_sampling_budget = args.revision_checker_budget
    # An empty list selects the native default eight checkers in SQLRevisionRunner.
    config.sql_revision_config.checkers = []
    config.sql_selection_config.evaluator_sampling_budget = args.selection_evaluator_budget
    config.sql_selection_config.shortcut_consistency_score_threshold = SELECTION_SHORTCUT_THRESHOLD
    return config


def _workload(args):
    from scripts.baseline_adapters.deepeye.workloads import load_workload
    return load_workload(args.workload) if getattr(args, 'workload', None) else None


def read_environment(path, *, args):
    """Require only credentials used by this workload; never construct clients."""
    import shlex
    values = {}
    path = Path(path)
    if path.is_file():
        for number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ValueError(f'Invalid environment entry at line {number}')
            key, raw_value = line.split('=', 1)
            parts = shlex.split(raw_value, comments=True)
            if len(parts) != 1 or key.strip() in values:
                raise ValueError(f'Malformed or duplicate environment field at line {number}')
            values[key.strip()] = parts[0]
    for key in ('DASH_MODELS', 'DASH_BASE_URL', 'DASH_API_KEY', 'EMBEDDING_MODEL',
                'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'PG_HOST', 'PG_PORT', 'PG_USER', 'PG_PASSWORD'):
        if key not in values and os.environ.get(key):
            values[key] = os.environ[key]
    required = ['DASH_MODELS', 'DASH_BASE_URL', 'DASH_API_KEY']
    workload = _workload(args)
    if workload is None or workload['benchmark'] == 'bird_interact':
        required += ['PG_HOST', 'PG_PORT', 'PG_USER', 'PG_PASSWORD']
    for key in required:
        if not values.get(key):
            raise ValueError(f'Missing environment field: {key}')
    return values


def _native_runtime_config(environment, args, run_dir):
    from types import SimpleNamespace
    import tomllib
    from app.config.config import (DatasetConfig, LLMConfig, VectorDatabaseConfig,
        ValueRetrievalConfig, FewShotIndexConfig, SchemaLinkingConfig, SQLGenerationConfig,
        SQLRevisionConfig, SQLSelectionConfig, LLMExtractorConfig, RunConfig,
        _resolve_llm_config, _resolve_embedding_config)
    from scripts.baseline_adapters.deepeye.dataset import BirdInteractDatasetConfig
    workload = _workload(args)
    raw = {}
    if workload.get('native_config'):
        path = Path(workload['native_config'])
        raw = (tomllib.loads(path.read_text()) if path.suffix == '.toml' else json.loads(path.read_text()))
        def resolve_paths(section):
            for name, value in section.items():
                if isinstance(value, dict):
                    resolve_paths(value)
                elif (isinstance(value, str) and name not in ('embedding_model_name_or_path', 'model_name_or_path')
                      and (name.endswith('_path') or name.endswith('_dir') or name.endswith('_root'))):
                    section[name] = str((path.parent / value).resolve())
        resolve_paths(raw)
    output = Path(run_dir) / '.native_non_authoritative'
    llm = LLMConfig(model=environment['DASH_MODELS'], base_url=environment['DASH_BASE_URL'],
                    api_key=environment.get('DASH_API_KEY', ''), max_tokens=args.max_tokens,
                    temperature=.6, n_call_strategy='split', max_request_n=1)
    dataset_cls = BirdInteractDatasetConfig if workload['benchmark'] == 'bird_interact' else DatasetConfig
    dataset = dict(raw.get('dataset', {}))
    dataset.update(type=workload['benchmark'], split=workload['split'], root_path=workload['resource_root'],
                   save_path=str(output / 'input.snapshot'))
    dataset.setdefault('sql_execution_timeout', 30 if workload['benchmark'] == 'bird_interact' else 600)
    dataset.setdefault('max_value_example_length', 100 if workload['benchmark'] == 'bird_interact' else 50)
    dynamic = workload.get('few_shot_strategy') == 'native_dynamic'
    vector = {**raw.get('embedding', {}), **raw.get('vector_database', {})}
    vector.setdefault('store_root_path', str(output / 'value_index'))
    vector.setdefault('build_backend', raw.get('value_retrieval', {}).get('backend', 'local_index'))
    vector.setdefault('embedding_device', 'cpu')
    if environment.get('EMBEDDING_MODEL'):
        vector.update(api_type='openai', embedding_model_name_or_path=environment['EMBEDDING_MODEL'],
                      base_url=environment.get('EMBEDDING_BASE_URL'), api_key=environment.get('EMBEDDING_API_KEY'))
    few_shot = dict(raw.get('few_shot_index', {}))
    few_shot.setdefault('save_path', str(output / 'few_shot_index'))
    few_shot.setdefault('prepared_save_path', str(output / 'few_shot.snapshot'))
    def preparation_llm(section, name):
        return _resolve_llm_config(raw.get('llm'), section.get('llm'), name, required=False,
            llm_profiles=raw.get('llm_profiles'), default_profile=raw.get('run', {}).get('default_llm_profile'),
            section_profile=section.get('llm_profile'))
    few_shot['llm'] = preparation_llm(few_shot, 'few_shot_index')
    few_shot['embedding'] = _resolve_embedding_config(raw.get('embedding'), few_shot.get('embedding'),
                                                     'few_shot_index', required=False)
    if dynamic:
        from app.config.config import EmbeddingConfig
        few_shot.setdefault('num_examples', 7)
        few_shot.setdefault('question_weight', .6)
        few_shot.setdefault('sql_weight', .4)
        few_shot['llm'] = few_shot['llm'] or llm
        if environment.get('EMBEDDING_MODEL'):
            few_shot['embedding'] = EmbeddingConfig(api_type='openai',
                embedding_model_name_or_path=environment['EMBEDDING_MODEL'],
                base_url=environment.get('EMBEDDING_BASE_URL'), api_key=environment.get('EMBEDDING_API_KEY'),
                embedding_device='cpu')
    preliminary = dict(few_shot.get('preliminary_sql', {}))
    preliminary['llm'] = preparation_llm(preliminary, 'few_shot_index.preliminary_sql')
    if dynamic:
        preliminary.setdefault('enabled', True)
        preliminary.setdefault('dc_sampling_budget', 4)
        preliminary.setdefault('skeleton_sampling_budget', 4)
        preliminary['llm'] = preliminary['llm'] or llm
    few_shot['preliminary_sql'] = preliminary
    run = {**raw.get('run', {}), 'parallelism': runtime_limits(args)['coordinator_workers']}
    if dynamic:
        run.setdefault('embedding_batch_size', 20)
        run.setdefault('llm_timeout', args.chat_timeout)
    if dynamic and workload.get('preparation_root'):
        from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
        shared = Path(workload['preparation_root'])
        train_type = workload.get('few_shot_dataset', 'bird' if workload['benchmark'] == 'bird_interact' else workload['benchmark'])
        def public(value):
            return {k: v for k, v in value.model_dump(mode='json').items() if k != 'api_key'} if value else None
        key = fingerprint({'dataset': train_type, 'source': workload.get('few_shot_source'),
                           'embedding': public(few_shot['embedding']), 'llm': public(few_shot['llm'])})[:16]
        if 'save_path' not in raw.get('few_shot_index', {}):
            few_shot['save_path'] = str(shared / 'few_shot' / (train_type + '-' + key))
        if 'store_root_path' not in raw.get('vector_database', {}):
            vector['store_root_path'] = str(shared / 'values' / (workload['benchmark'] + '-' +
                fingerprint({'resource_root': workload['resource_root'], 'model': vector.get('embedding_model_name_or_path')})[:16]))
    config = SimpleNamespace(dataset_config=dataset_cls(**dataset),
        run_config=RunConfig(**run),
        vector_database_config=VectorDatabaseConfig(**vector),
        few_shot_index_config=FewShotIndexConfig(**few_shot),
        llm_extractor_config=LLMExtractorConfig(max_retry=args.extractor_retries))
    classes = {'value_retrieval': ValueRetrievalConfig, 'schema_linking': SchemaLinkingConfig,
               'sql_generation': SQLGenerationConfig, 'sql_revision': SQLRevisionConfig,
               'sql_selection': SQLSelectionConfig}
    for stage, cls in classes.items():
        values = dict(raw.get(stage, {}))
        values.update(llm=llm, save_path=str(output / f'{stage}.snapshot'))
        if stage == 'value_retrieval':
            values.setdefault('backend', 'chroma' if vector['build_backend'] == 'both' else vector['build_backend'])
            if vector['build_backend'] not in ('both', values['backend']):
                raise ValueError('Vector build_backend does not provide the value_retrieval backend')
        setattr(config, stage + '_config', cls(**values))
    config.schema_linking_config.direct_linking_sampling_budget = args.direct_linking_budget
    config.schema_linking_config.reversed_linking_sampling_budget = args.reversed_linking_budget
    config.sql_generation_config.dc_sampling_budget = args.dc_generation_budget
    config.sql_generation_config.skeleton_sampling_budget = args.skeleton_generation_budget
    config.sql_generation_config.icl_sampling_budget = args.icl_generation_budget
    config.sql_revision_config.checker_sampling_budget = args.revision_checker_budget
    if 'checkers' not in raw.get('sql_revision', {}):
        config.sql_revision_config.checkers = ['SyntaxChecker', 'JoinChecker', 'OrderByLimitChecker',
            'TimeChecker', 'SelectChecker', 'MaxMinChecker', 'OrderByNullChecker', 'ResultChecker']
    config.sql_selection_config.evaluator_sampling_budget = args.selection_evaluator_budget
    config.sql_selection_config.filter_top_k_sql = 2
    config.sql_selection_config.shortcut_consistency_score_threshold = SELECTION_SHORTCUT_THRESHOLD
    return config


def _native_effective_config(environment, args):
    config = _native_runtime_config(environment, args, Path('/RUN'))
    def freeze(value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json')
        if isinstance(value, dict):
            return {key: freeze(item) for key, item in value.items()
                    if key not in ('api_key', 'password')}
        if isinstance(value, list):
            return [freeze(item) for item in value]
        return value
    result = {'profile': 'shared-sampling-v1', 'native': freeze(vars(config)),
              'workload': _workload(args),
              'dataset': {'sql_execution_timeout_seconds': config.dataset_config.sql_execution_timeout,
                          'max_value_example_length': config.dataset_config.max_value_example_length},
              'runtime': {'version': 'shared-sampling-runtime-v1', **runtime_limits(args)},
              'admission': admission_settings(args),
              'scheduler': {'mode': 'all_questions', 'concurrency_unit': 'model_request'},
              'chat': {'model': environment['DASH_MODELS'], 'endpoint': environment['DASH_BASE_URL'].rstrip('/'),
                       'timeout_seconds': args.chat_timeout, 'sample_max_attempts': 4, 'sdk_max_retries': 0,
                       'max_tokens': args.max_tokens, 'temperature': .6, 'thinking_budget': args.thinking_budget,
                       'extractor_max_retries': args.extractor_retries, 'n_call_strategy': 'split', 'max_request_n': 1},
              'stages': {stage: freeze(getattr(config, stage + '_config')) for stage in STAGES}}
    if config.dataset_config.type == 'bird_interact':
        result['postgres'] = {'host': environment['PG_HOST'], 'port': int(environment['PG_PORT']),
                              'principal': environment['PG_USER'], 'sslmode': args.pg_sslmode,
                              'read_only': True, 'statement_timeout_seconds': 30}
    return result


@contextmanager
def backend_context(environment, args, *, install_support=None):
    workload = _workload(args)
    if workload is not None and workload['benchmark'] != 'bird_interact':
        yield
        return
    from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
    install_support = install_support or install_postgres_support
    fields = ('PG_HOST', 'PG_PORT', 'PG_USER', 'PG_PASSWORD', 'PG_SSLMODE')
    previous = {key: os.environ.get(key) for key in fields}
    undo = None
    try:
        os.environ.update({key: environment[key] for key in fields[:-1]})
        os.environ['PG_SSLMODE'] = args.pg_sslmode
        undo = install_support()
        yield
    finally:
        if undo is not None:
            undo()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bounded_runner_factory(config, chat_timeout: int, native_factory_builder=None, *, runtime=None):
    """Keep native stage logic while applying the smoke-tested transport bounds."""

    from scripts.baseline_adapters.deepeye.run_pipeline import native_runner_factory

    native_factory_builder = native_factory_builder or native_runner_factory
    base_factory = native_factory_builder(config)

    def factory(stage, items):
        runner = base_factory(stage, items)
        llm = runner._llm
        if runtime is None:
            client = llm._get_client()
            client.max_retries = 0
        llm.sample_max_attempts = 4
        request_once = llm.request_once

        def bounded_request(*args, **kwargs):
            kwargs["timeout"] = min(kwargs.get("timeout", chat_timeout), chat_timeout)
            return request_once(*args, **kwargs)

        llm.request_once = bounded_request
        return runner

    return factory


def _execute_pipeline(store, tasks, environment, args, *, pipeline_fn=None,
                      runner_factory_fn=None, recorder_type=None,
                      install_support=None, runtime_config_builder=None):
    """Execute the four native stages; imported lazily so preparation stays offline."""

    from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
    from scripts.baseline_adapters.deepeye.run_pipeline import (
        native_runner_factory, run_pipeline, prepare_run, select_unfinished,
    )
    from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder

    pipeline_fn = pipeline_fn or run_pipeline
    runner_factory_fn = runner_factory_fn or native_runner_factory
    recorder_type = recorder_type or TraceRecorder
    install_support = install_support or install_postgres_support
    runtime_config_builder = runtime_config_builder or build_runtime_config
    run_dir = getattr(store, "run_dir", Path("."))
    secret_values = [environment.get(key) for key in
                     ("DASH_API_KEY", "EMBEDDING_API_KEY", "PG_PASSWORD")]
    recorder = recorder_type(store, secrets=secret_values, stop_event=threading.Event())
    prepared, recovery = None, None
    if pipeline_fn is run_pipeline:
        prepared = prepare_run(store, tasks, checkpoints=recorder.sampling_checkpoints)
        if getattr(args, 'unfinished_only', False):
            tasks, recovery = select_unfinished(store, tasks, prepared=prepared)
        if not tasks:
            return {'succeeded': 0, 'failed': 0, 'executed': False, 'recovery': recovery}
    config = runtime_config_builder(environment, args, run_dir)
    with backend_context(environment, args, install_support=install_support):
        with recorder.install(), admission_context(recorder, args, population=len(tasks)) as controllers, \
                sampling_runtime(recorder, args) as runtime:
            runner_factory = (bounded_runner_factory(config, args.chat_timeout,
                              native_factory_builder=runner_factory_fn, runtime=runtime)
                              if runner_factory_fn is native_runner_factory else runner_factory_fn(config))
            result = pipeline_fn(store, tasks, runner_factory, recorder, runtime=runtime,
                                 slot_controller=controllers["pipeline"],
                                 **({'prepared': prepared} if prepared is not None else {}))
            if recovery is not None:
                result['recovery'] = recovery
            result['runtime'] = runtime.snapshot()
            if controllers:
                result["admission"] = {name: gate.snapshot() for name, gate in controllers.items()}
            return result


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare-native':
            from scripts.baseline_adapters.deepeye.workload_preparation import prepare_many
            report = prepare_many(args.workload, args.output, args.output_dir,
                args.env_file, workers=args.preparation_workers, embedding_config=args.embedding_config)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command in ('samples', 'renew-samples'):
            from scripts.baseline_adapters.deepeye.run_operations import sample_command
            print(json.dumps(sample_command(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "inspect":
            report = _diagnostics(args.run_dir)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0 if report["verification"]["ok"] else 1
        if args.command == "export":
            from scripts.baseline_adapters.deepeye.run_store import RunStore
            from scripts.baseline_adapters.deepeye.run_usage import observed_usage

            with RunStore.open(args.run_dir.resolve(), read_only=True) as store:
                destination = store.export(
                    args.export_dir.resolve(),
                    extra_reports={"observed_usage.json": observed_usage},
                )
            print(json.dumps({"export_dir": str(destination)}, ensure_ascii=False))
            return 0
        _validate_run_args(parser, args)
        report = _prepare_command(args, execute=args.command != "prepare",
                                  resume=args.command == "resume")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 1 if report.get("failed", 0) else 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
