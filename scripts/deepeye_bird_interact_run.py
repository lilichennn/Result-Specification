"""Crash-safe, gold-free DeepEye SQL runs from frozen BIRD-Interact precomputation.

This entry point runs a deliberately bounded four-stage configuration.  It is an
integration workflow, not an exact reproduction of the original paper defaults.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys


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
    adapter_files = list((CODE_ROOT / "scripts/baseline_adapters/deepeye").glob("*.py"))
    support_files = [CODE_ROOT / "scripts/deepeye_bird_interact_smoke.py",
                     CODE_ROOT / "scripts/deepeye_bird_interact_precompute.py",
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

    return {
        "profile": "bounded-config",
        "chat": {
            "model": environment["DASH_MODELS"],
            "endpoint": environment["DASH_BASE_URL"].rstrip("/"),
            "max_tokens": args.max_tokens,
            "temperature": 0.6,
            "thinking_budget": args.thinking_budget,
            "timeout_seconds": args.chat_timeout,
            "sdk_max_retries": 0,
            "native_ask_attempts": 2,
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
        "workers": args.workers,
        "inner_workers": getattr(args, "inner_workers", None) or args.workers,
        "scheduler": {"mode": "pipeline_slots", "concurrency_unit": "question_pipeline"},
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
        "format": "deepeye-bird-interact-run-v1",
        "scope": "gold-free bounded four-stage PostgreSQL workflow from frozen value retrieval",
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
    if not getattr(args, "adaptive_concurrency", False):
        return {"enabled": True, "adaptive": False,
                "pipeline": {"fixed_limit": args.workers},
                "postgres_limit": args.pg_concurrency}
    from scripts.baseline_adapters.deepeye.run_admission import AdaptivePolicy
    policy = AdaptivePolicy(
        initial_limit=args.concurrency_initial, step=args.concurrency_step,
        min_limit=args.concurrency_min, max_limit=args.concurrency_max,
        stable_window_s=args.concurrency_window,
    )
    return {"enabled": True, "adaptive": True, "pipeline": asdict(policy),
            "postgres_limit": args.pg_concurrency}


@contextmanager
def admission_context(recorder, args):
    """Limit question pipelines and PostgreSQL; observe model calls without gating."""
    settings = admission_settings(args)
    from scripts.baseline_adapters.deepeye import backend_hooks
    from scripts.baseline_adapters.deepeye.run_admission import (
        AdaptivePolicy, FixedAdmission,
    )
    from scripts.baseline_adapters.deepeye.run_slots import PipelineSlots

    slots = PipelineSlots(
        fixed_limit=args.workers,
        policy=AdaptivePolicy(**settings["pipeline"]) if settings["adaptive"] else None,
        emit=recorder.record_admission,
    )
    postgres = FixedAdmission(settings["postgres_limit"], emit=recorder.record_admission)
    original_api = recorder.api_call
    original_pg = backend_hooks.execute_postgres_sql
    recorder.api_call = slots
    backend_hooks.execute_postgres_sql = lambda *args, **kwargs: postgres(original_pg, args, kwargs)
    try:
        yield {"pipeline": slots, "postgres": postgres}
    finally:
        backend_hooks.execute_postgres_sql = original_pg
        recorder.api_call = original_api


def prepare_inputs(precompute_dir: Path, few_shot_source: Path, *, variants=None,
                   item_keys=None, inventory_loader=None, item_loader=None,
                   example_loader=None, code_source_hasher=None, probe_per_variant=None):
    """Load gold-free precomputed items and attach three static cross-domain examples."""

    from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
    from scripts.baseline_adapters.deepeye.precompute_pipeline import (
        item_directory, load_precomputed_item, question_row, read_record,
    )
    from scripts.deepeye_bird_interact_precompute import load_inputs
    from scripts.deepeye_bird_interact_smoke import load_independent_examples

    inventory_loader = inventory_loader or load_inputs
    item_loader = item_loader or load_precomputed_item
    example_loader = example_loader or load_independent_examples
    code_source_hasher = code_source_hasher or code_source_hashes
    precompute_dir = Path(precompute_dir).resolve()
    few_shot_source = Path(few_shot_source).resolve()

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
        item = item_loader(precompute_dir, variant, expected_item.instance_id,
                           expected_item=expected_item)
        if getattr(item, "gold_sql", ""):
            raise ValueError("Target gold SQL must not enter the run")
        examples, provenance = example_loader(few_shot_source, item, count=3)
        if len(examples) != 3:
            raise ValueError("Exactly three independent static training examples are required")
        item.few_shot_examples = examples
        item.few_shot_preparation_metadata = {
            "mode": "static_independent_bird_train", "num_examples": 3,
        }
        directory = item_directory(precompute_dir, variant, item.instance_id)
        keywords = read_record(directory / "keywords.json")
        retrieval = read_record(directory / "retrieval.json")
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
        "few_shot_source_sha256": _file_sha256(few_shot_source),
        "precompute_population": {"lite": 195, "full": 410, "databases": 40},
        "locators": {"precompute_dir": str(precompute_dir),
                     "few_shot_source": str(few_shot_source)},
        "code": code_source_hasher(),
    }
    return prepared, bindings, sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name != "resume":
            command.add_argument("--inherit-from", type=Path,
                                 help="Explicitly import compatible successful stage prefixes into a new run")
        command.add_argument("--precompute-dir", type=Path, required=True)
        command.add_argument("--env-file", type=Path, default=CODE_ROOT / "config/.env")
        command.add_argument("--few-shot-source", type=Path)
        command.add_argument("--variant", action="append", choices=("lite", "full"))
        command.add_argument("--item", dest="item_keys", action="append",
                             help="Exact variant/instance_id; repeat to select several")
        command.add_argument("--workers", type=int, default=4,
                             help="Fixed number of concurrent question pipelines (when not adaptive)")
        command.add_argument("--inner-workers", type=int,
                             help="Native inner pool width; defaults to --workers")
        command.add_argument("--probe-per-variant", type=int,
                             help="Gold-free coverage sample per selected variant; includes every database")
        command.add_argument("--adaptive-concurrency", action="store_true",
                             help="Dynamically adjust question pipeline slots, not model request slots")
        command.add_argument("--concurrency-initial", type=int, default=50)
        command.add_argument("--concurrency-step", type=int, default=10)
        command.add_argument("--concurrency-min", type=int, default=10)
        command.add_argument("--concurrency-max", type=int, default=100)
        command.add_argument("--concurrency-window", type=float, default=60.0)
        command.add_argument("--pg-concurrency", type=int, default=10,
                             help="Independent PostgreSQL cap in both fixed and adaptive modes")
        command.add_argument("--max-tokens", type=int, default=6144)
        command.add_argument("--thinking-budget", type=int)
        command.add_argument("--chat-timeout", type=int, default=300)
        command.add_argument(
            "--pg-sslmode",
            choices=("disable", "allow", "prefer", "require", "verify-ca", "verify-full"),
            default="prefer",
            help="Explicit libpq SSL mode, frozen in the run manifest (default: prefer)",
        )
        command.add_argument("--extractor-retries", type=int, default=2)
        command.add_argument("--direct-linking-budget", type=int, default=1)
        command.add_argument("--reversed-linking-budget", type=int, default=1)
        command.add_argument("--dc-generation-budget", type=int, default=1)
        command.add_argument("--skeleton-generation-budget", type=int, default=1)
        command.add_argument("--icl-generation-budget", type=int, default=1)
        command.add_argument("--revision-checker-budget", type=int, default=1)
        command.add_argument("--selection-evaluator-budget", type=int, default=1)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--run-dir", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--export-dir", type=Path, required=True)
    return parser


def _diagnostics(run_dir: Path) -> dict:
    from scripts.baseline_adapters.deepeye.run_store import RunStore
    from scripts.baseline_adapters.deepeye.run_usage import observed_usage

    with RunStore.open(run_dir.resolve(), read_only=True) as store:
        return {"manifest": store.manifest, "summary": store.summary(),
                "observed_usage": observed_usage(store), "verification": store.verify()}


def _validate_run_args(parser: argparse.ArgumentParser, args) -> None:
    positive = (
        "workers", "max_tokens", "chat_timeout", "extractor_retries",
        "direct_linking_budget", "reversed_linking_budget", "dc_generation_budget",
        "skeleton_generation_budget", "icl_generation_budget",
        "revision_checker_budget", "selection_evaluator_budget",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("Workers, token/time limits, retries, and all stage budgets must be positive")
    if args.thinking_budget is not None and args.thinking_budget < 1:
        parser.error("thinking-budget must be positive when supplied")
    if args.inner_workers is not None and args.inner_workers < 1:
        parser.error("inner-workers must be positive")
    if args.probe_per_variant is not None and (args.probe_per_variant < 1 or args.item_keys):
        parser.error("probe-per-variant must be positive and cannot be combined with --item")
    if args.pg_concurrency < 1:
        parser.error("pg-concurrency must be positive")
    try:
        admission_settings(args)
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
    from scripts.deepeye_bird_interact_smoke import read_environment

    environment = read_environment(args.env_file.resolve())
    source = _resolve_few_shot_source(args.few_shot_source)
    tasks, bindings, sources = prepare_inputs(
        args.precompute_dir, source, variants=args.variant, item_keys=args.item_keys,
        probe_per_variant=args.probe_per_variant,
    )
    effective = build_effective_config(environment, args)
    manifest = build_manifest(effective, sources, bindings)
    if resume:
        store_context = RunStore.open(args.run_dir.resolve(), expected_manifest=manifest)
    else:
        args.run_dir.resolve().parent.mkdir(parents=True, exist_ok=True)
        store_context = RunStore.create(args.run_dir.resolve(), manifest)
    with store_context as store:
        inheritance = None
        if getattr(args, "inherit_from", None) is not None:
            if resume:
                raise ValueError("Checkpoint inheritance is only allowed for a new run")
            from scripts.baseline_adapters.deepeye.run_inheritance import inherit_checkpoints

            with RunStore.open(args.inherit_from.resolve(), read_only=True) as source_store:
                inheritance = inherit_checkpoints(source_store, store, tasks)
        if execute:
            result = _execute_pipeline(store, tasks, environment, args)
        else:
            result = {"prepared": len(tasks), "executed": False}
        if inheritance is not None:
            result["inheritance"] = inheritance
        verification = store.verify()
        if not verification["ok"]:
            raise RuntimeError("RunStore verification failed")
        return {**result, "run_dir": str(args.run_dir.resolve()),
                "store_summary": store.summary(), "observed_usage": observed_usage(store),
                "verification": verification}


def build_runtime_config(environment, args, run_dir: Path):
    """Create native runner config with explicit bounded, non-paper-default budgets."""

    from scripts.deepeye_bird_interact_smoke import build_runtime_config as smoke_config

    native_output = Path(run_dir) / ".native_non_authoritative"
    preprocessed = CODE_ROOT / "scripts/bird_interact_lite/preprocessed_data"
    config = smoke_config(environment, "lite", preprocessed, native_output,
                          max_tokens=args.max_tokens, thinking_budget=args.thinking_budget)
    config.run_config.parallelism = getattr(args, "inner_workers", None) or args.workers
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


def bounded_runner_factory(config, chat_timeout: int, native_factory_builder=None):
    """Keep native stage logic while applying the smoke-tested transport bounds."""

    from app.llm import LLM
    from tenacity import stop_after_attempt, wait_fixed
    from scripts.baseline_adapters.deepeye.run_pipeline import native_runner_factory

    native_factory_builder = native_factory_builder or native_runner_factory
    base_factory = native_factory_builder(config)
    bounded_ask = LLM.ask.retry_with(stop=stop_after_attempt(2), wait=wait_fixed(1))

    def factory(stage, items):
        runner = base_factory(stage, items)
        llm = runner._llm
        client = llm._get_client()
        client.max_retries = 0

        def ask(*args, **kwargs):
            kwargs["timeout"] = min(kwargs.get("timeout", chat_timeout), chat_timeout)
            return bounded_ask(llm, *args, **kwargs)

        llm.ask = ask
        return runner

    return factory


def _execute_pipeline(store, tasks, environment, args, *, pipeline_fn=None,
                      runner_factory_fn=None, recorder_type=None,
                      install_support=None, runtime_config_builder=None):
    """Execute the four native stages; imported lazily so preparation stays offline."""

    from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
    from scripts.baseline_adapters.deepeye.run_pipeline import (
        native_runner_factory, run_pipeline,
    )
    from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder

    pipeline_fn = pipeline_fn or run_pipeline
    runner_factory_fn = runner_factory_fn or native_runner_factory
    recorder_type = recorder_type or TraceRecorder
    install_support = install_support or install_postgres_support
    runtime_config_builder = runtime_config_builder or build_runtime_config
    run_dir = getattr(store, "run_dir", Path("."))
    config = runtime_config_builder(environment, args, run_dir)
    if runner_factory_fn is native_runner_factory:
        runner_factory = bounded_runner_factory(config, args.chat_timeout,
                                                native_factory_builder=runner_factory_fn)
    else:
        runner_factory = runner_factory_fn(config)
    secret_values = [environment.get(key) for key in
                     ("DASH_API_KEY", "EMBEDDING_API_KEY", "PG_PASSWORD")]
    recorder = recorder_type(store, secrets=secret_values)
    pg_fields = ("PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_SSLMODE")
    previous = {key: os.environ.get(key) for key in pg_fields}
    os.environ.update({key: environment[key] for key in pg_fields[:-1]})
    os.environ["PG_SSLMODE"] = args.pg_sslmode
    undo_support = None
    try:
        undo_support = install_support()
        with recorder.install(), admission_context(recorder, args) as controllers:
            result = pipeline_fn(store, tasks, runner_factory, recorder, workers=args.workers,
                                 slot_controller=controllers["pipeline"])
            if controllers:
                result["admission"] = {name: gate.snapshot() for name, gate in controllers.items()}
            return result
    finally:
        try:
            if undo_support is not None:
                undo_support()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
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
