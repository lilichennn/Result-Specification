"""Bounded PostgreSQL integration smoke using the original DeepEye runners.

Run with uv run python. This is not a gold-SQL accuracy evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

CODE_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = CODE_ROOT / "baselines/DeepEye-SQL"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))

ENV_FIELDS = (
    "DASH_BASE_URL", "DASH_API_KEY", "DASH_MODELS", "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD",
)
STAGES = ("value_retrieval", "schema_linking", "sql_generation", "sql_revision", "sql_selection")


def read_environment(path: Path) -> dict[str, str]:
    values = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ENV_FIELDS:
            continue
        if key in values:
            raise ValueError(f"Duplicate environment field: {key}")
        parts = shlex.split(value, comments=True)
        if len(parts) != 1:
            raise ValueError(f"Missing or malformed environment field: {key}")
        values[key] = parts[0]
    validate_environment(values)
    return values


def validate_environment(values: dict[str, str]) -> None:
    for key in ENV_FIELDS:
        if not values.get(key):
            raise ValueError(f"Missing environment field: {key}")
    for key in ("DASH_MODELS", "EMBEDDING_MODEL"):
        if values[key].startswith("sk-") or not re.fullmatch(r"[\w./:-]+", values[key]):
            raise ValueError(f"Invalid model name in {key}; check model/key field positions")
    for key in ("DASH_BASE_URL", "EMBEDDING_BASE_URL"):
        url = urlsplit(values[key])
        if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError(f"Invalid API root in {key}")
        if url.path.rstrip("/").endswith(("/embeddings", "/chat/completions")):
            raise ValueError(f"{key} must be the API root, not a request endpoint")
    if not values["PG_PORT"].isdigit() or not 1 <= int(values["PG_PORT"]) <= 65535:
        raise ValueError("Invalid PG_PORT")


def redact(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        if value and ("KEY" in key or "PASSWORD" in key or value.startswith("sk-")):
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)


def create_output_directory(path: Path) -> None:
    # Even an empty existing directory may belong to another run.
    path.mkdir(parents=True, exist_ok=False)


def build_runtime_config(values, variant, preprocessed_dir, output_dir, *, max_tokens=6144, thinking_budget=None):
    from app.config.config import (
        LLMConfig, VectorDatabaseConfig, ValueRetrievalConfig,
        SchemaLinkingConfig, SQLGenerationConfig, SQLRevisionConfig, SQLSelectionConfig,
        FewShotIndexConfig, LLMExtractorConfig,
    )
    from scripts.baseline_adapters.deepeye.dataset import BirdInteractDatasetConfig
    llm = LLMConfig(model=values["DASH_MODELS"], base_url=values["DASH_BASE_URL"],
                    api_key=values["DASH_API_KEY"], max_tokens=max_tokens,
                    temperature=0.6, n_call_strategy="split", max_request_n=1,
                    extra_body={} if thinking_budget is None else {"thinking_budget": thinking_budget})
    output_dir = Path(output_dir)
    paths = {stage: str(output_dir / f"{stage}.snapshot") for stage in STAGES}
    return SimpleNamespace(
        dataset_config=BirdInteractDatasetConfig(type="bird_interact", split=variant,
            root_path=str(preprocessed_dir), save_path=str(output_dir / "input.snapshot"), sql_execution_timeout=30),
        run_config=SimpleNamespace(parallelism=2, embedding_batch_size=20, progress_log_interval=1, checkpoint_interval=1),
        llm_extractor_config=LLMExtractorConfig(max_retry=2),
        vector_database_config=VectorDatabaseConfig(api_type="openai", embedding_model_name_or_path=values["EMBEDDING_MODEL"],
            base_url=values["EMBEDDING_BASE_URL"], api_key=values["EMBEDDING_API_KEY"],
            store_root_path=str(output_dir / "value_index"), build_backend="local_index", embedding_device="cpu"),
        few_shot_index_config=FewShotIndexConfig(prepared_save_path=str(output_dir / "unused_dynamic_few_shot.snapshot")),
        value_retrieval_config=ValueRetrievalConfig(llm=llm, backend="local_index", local_index_device="cpu", save_path=paths["value_retrieval"]),
        schema_linking_config=SchemaLinkingConfig(llm=llm, save_path=paths["schema_linking"], direct_linking_sampling_budget=1, reversed_linking_sampling_budget=1),
        sql_generation_config=SQLGenerationConfig(llm=llm, save_path=paths["sql_generation"], dc_sampling_budget=1, skeleton_sampling_budget=1, icl_sampling_budget=1),
        sql_revision_config=SQLRevisionConfig(llm=llm, save_path=paths["sql_revision"], checker_sampling_budget=1),
        sql_selection_config=SQLSelectionConfig(llm=llm, save_path=paths["sql_selection"], evaluator_sampling_budget=1),
    )


class IndependentExampleReader:
    """One immutable source snapshot with per-row dialect conversion reuse."""

    def __init__(self, source: Path):
        self.source = Path(source)
        payload = self.source.read_bytes()
        self.source_sha256 = hashlib.sha256(payload).hexdigest()
        self._rows = json.loads(payload)
        self._translated = {}

    def _translate(self, index, row):
        import sqlglot

        if index not in self._translated:
            translated = None
            if row.get("question") and row.get("SQL"):
                try:
                    queries = sqlglot.transpile(
                        row["SQL"], read="sqlite", write="postgres",
                        unsupported_level=sqlglot.ErrorLevel.RAISE)
                except sqlglot.errors.SqlglotError:
                    queries = []
                if len(queries) == 1:
                    translated = (
                        {"question": row["question"], "evidence": row.get("evidence", ""),
                         "sql": queries[0]},
                        {"source_row": index, "db_id": row.get("db_id"),
                         "original_sql": row["SQL"],
                         "dialect_conversion": "sqlglot sqlite -> postgres"},
                    )
            self._translated[index] = translated
        return self._translated[index]

    def select(self, target, count=3):
        examples, provenance, domains = [], [], set()
        for index, row in enumerate(self._rows):
            db_id = row.get("db_id")
            if (db_id == target.database_id or db_id in domains
                    or row.get("question", "").strip() == target.question.strip()):
                continue
            translated = self._translate(index, row)
            if translated is None:
                continue
            example, source_row = translated
            examples.append(deepcopy(example))
            provenance.append(deepcopy(source_row))
            domains.add(db_id)
            if len(examples) == count:
                return examples, {
                    "source_path": str(self.source),
                    "source_sha256": self.source_sha256,
                    "examples": provenance,
                }
        raise ValueError(
            "Insufficient independent training examples; no target gold or synthetic fallback will be used")


def load_independent_examples(source: Path, target, count=3):
    """Select examples through a fresh reader so source changes stay observable."""
    return IndependentExampleReader(source).select(target, count)


class CallRecorder:
    """Observe original methods without changing their outputs or ranking rules."""
    def __init__(self, max_api_calls=80):
        self.lock = threading.Lock()
        self.events = []
        self.api_calls = 0
        self.max_api_calls = max_api_calls

    def record(self, event):
        with self.lock:
            self.events.append(event)

    def wrap_api(self, original, service):
        def call(*args, **kwargs):
            with self.lock:
                if self.api_calls >= self.max_api_calls:
                    self.events.append({"kind": "budget_exhausted", "service": service, "success": False})
                    raise RuntimeError("Smoke API request budget exhausted")
                self.api_calls += 1
            started = time.monotonic()
            event = {"kind": "api", "service": service}
            try:
                response = original(*args, **kwargs)
                if getattr(response, "usage", None):
                    event["usage"] = response.usage.model_dump()
                event["success"] = True
                return response
            except Exception as exc:
                event.update(success=False, error_type=type(exc).__name__)
                raise
            finally:
                event["seconds"] = round(time.monotonic() - started, 3)
                self.record(event)
        return call

    def observe_method(self, owner, name, label, *, count_output=False):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            event = {"kind": "branch", "branch": label}
            try:
                result = original(*args, **kwargs)
                event["success"] = True
                if count_output:
                    event["output_count"] = len(result[0]) if result[0] is not None else 0
                if label.startswith("schema_linking."):
                    event["has_result"] = result[0] is not None
                return result
            except Exception as exc:
                event.update(success=False, error_type=type(exc).__name__)
                raise
            finally:
                self.record(event)
        setattr(owner, name, call)


def observation_failures(events):
    """Reject silent native fallbacks after empty votes or failed model calls.

For this smoke, even a recovered API error is recorded as an imperfect run.
Native results are preserved; only the external acceptance result changes.
"""
    failures = []
    for event in events:
        label = event.get("branch", event.get("service", event["kind"]))
        if event.get("success") is False or event.get("has_result") is False:
            failures.append(f"{label}: {event.get('error_type', event['kind'])} failed")
        elif event.get("kind") == "branch" and event.get("output_count") == 0:
            failures.append(f"{label}: no valid output")
    return failures


def configure_stage_calls(runner, stage, recorder, *, chat_timeout=300):
    llm = runner._llm
    client = llm._get_client()
    client.max_retries = 0
    client.chat.completions.create = recorder.wrap_api(client.chat.completions.create, "chat")
    llm.sample_max_attempts = 4
    native_request = llm.request_once
    def bounded_request(*args, **kwargs):
        kwargs["timeout"] = min(kwargs.get("timeout", chat_timeout), chat_timeout)
        return native_request(*args, **kwargs)
    llm.request_once = bounded_request
    clients = [client]
    if stage == "value_retrieval":
        recorder.observe_method(runner._keyword_extractor, "extract_with_retry", "value_retrieval.keyword_extraction", count_output=True)
        embedding_client = runner._embedding_function.client
        embedding_client.timeout = 60
        embedding_client.max_retries = 0
        embedding_client.embeddings.create = recorder.wrap_api(embedding_client.embeddings.create, "embedding")
        clients.append(embedding_client)
    elif stage == "schema_linking":
        for name in ("direct", "reversed", "value"):
            recorder.observe_method(getattr(runner, f"_{name}_linker"), "link", f"schema_linking.{name}", count_output=name != "value")
    elif stage == "sql_generation":
        for name in ("dc", "skeleton", "icl"):
            recorder.observe_method(getattr(runner, f"_{name}_generator"), "generate", f"generation.{name}", count_output=True)
    elif stage == "sql_revision":
        for checker in runner._checkers:
            recorder.observe_method(checker, "check_and_revise", f"revision.{type(checker).__name__}")
            recorder.observe_method(checker._extractor, "extract_with_retry", f"revision.{type(checker).__name__}.extraction", count_output=True)
    elif stage == "sql_selection":
        recorder.observe_method(runner, "_compare_sqls", "selection.pairwise_comparison", count_output=True)
    return clients


def json_write(path, value, env):
    path.write_text(redact(json.dumps(value, ensure_ascii=False, indent=2, default=str), env) + "\n", encoding="utf-8")


def run_smoke(args, env):
    from app.dataset import save_dataset, load_dataset
    from scripts.baseline_adapters.deepeye.dataset import BirdInteractDataset
    from scripts.baseline_adapters.deepeye.postgres_index import build_postgres_value_index
    from scripts.baseline_adapters.deepeye.hooks import install_postgres_support
    from app.vector_db import get_embedding_function
    from scripts.baseline_adapters.deepeye.postgres_execution import execute_postgres_sql
    from app.pipeline import ValueRetrievalRunner, SchemaLinkingRunner, SQLGenerationRunner, SQLRevisionRunner, SQLSelectionRunner
    from app.logger import logger

    output = args.output_dir.resolve()
    create_output_directory(output)
    logger.remove()
    def safe_sink(message):
        text = redact(str(message), env)
        sys.stderr.write(text)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(text)
    logger.add(safe_sink, level="INFO", format="{time:HH:mm:ss} | {level} | {message}", diagnose=False, backtrace=False)

    preprocessed = args.preprocessed_dir or CODE_ROOT / "data" / f"bird_interact_{args.variant}"
    cfg = build_runtime_config(env, args.variant, preprocessed, output, max_tokens=args.max_tokens, thinking_budget=args.thinking_budget)
    instance_id = args.instance_id or {"lite": "alien_1", "full": "exchange_traded_funds_1"}[args.variant]
    recorder = CallRecorder(args.max_api_calls)
    summary = {"success": False, "variant": args.variant, "instance_id": instance_id,
        "chat_model": env["DASH_MODELS"], "embedding_model": env["EMBEDDING_MODEL"],
        "runtime_python": sys.executable, "stages": {}, "accuracy_evaluated": False,
        "scope": "bounded PostgreSQL integration smoke; no RC injection",
        "limits": {"values_per_column": args.max_values_per_column, "embedding_batch_size": 20,
                   "sampling_per_generator": 1, "sampling_per_checker": 1, "max_tokens": args.max_tokens,
                   "sql_timeout_seconds": 30, "chat_timeout_seconds": args.chat_timeout, "max_api_calls": args.max_api_calls,
                   "thinking_budget": args.thinking_budget},
        "dynamic_few_shot_retrieval_tested": False,
        "dialect_notes": ["Meta has no FK declarations; infer joins from question/evidence/descriptions",
                          "SQLite-only TimeChecker is a PostgreSQL no-op; other seven checkers unchanged"]}
    old_pg = {k: os.environ.get(k) for k in ENV_FIELDS if k.startswith("PG_")}
    os.environ.update({k: env[k] for k in old_pg})
    started = time.monotonic()
    undo_hooks = None
    try:
        undo_hooks = install_postgres_support()
        dataset = BirdInteractDataset(cfg.dataset_config, instance_ids=[instance_id])
        item = dataset[0]
        summary["db_id"] = item.database_id
        summary["schema_sha256"] = hashlib.sha256(json.dumps(item.database_schema, sort_keys=True).encode()).hexdigest()
        source = args.few_shot_source
        if source is None:
            source = next((p for p in (CODE_ROOT.parent / "BIRD/data/train/train.json", CODE_ROOT.parent / "BIRD1.0/data/train/train.json") if p.is_file()), None)
        if source is None:
            raise ValueError("BIRD train.json not found; provide --few-shot-source")
        examples, provenance = load_independent_examples(source.resolve(), item)
        item.few_shot_examples = examples
        item.few_shot_preparation_metadata = {"mode": "static_independent_bird_train", "num_examples": len(examples)}
        json_write(output / "few_shot_provenance.json", {**provenance, "rendered_examples": examples}, env)
        save_dataset(dataset, cfg.dataset_config.save_path)
        embedding = get_embedding_function(model_name_or_path=env["EMBEDDING_MODEL"], api_type="openai",
            base_url=env["EMBEDDING_BASE_URL"], api_key=env["EMBEDDING_API_KEY"])
        embedding.client.timeout = 60
        embedding.client.max_retries = 0
        embedding.client.embeddings.create = recorder.wrap_api(embedding.client.embeddings.create, "embedding")
        try:
            summary["value_index"] = build_postgres_value_index(item, Path(cfg.vector_database_config.store_root_path), embedding,
                max_values_per_column=args.max_values_per_column, embedding_batch_size=20)
        finally:
            embedding.client.close()
        runners = (ValueRetrievalRunner, SchemaLinkingRunner, SQLGenerationRunner, SQLRevisionRunner, SQLSelectionRunner)
        for stage, runner_type in zip(STAGES, runners):
            stage_started = time.monotonic()
            runner = runner_type.from_config(cfg)
            clients = configure_stage_calls(runner, stage, recorder, chat_timeout=args.chat_timeout)
            try:
                runner.run()
            finally:
                runner._clean_up()
                for client in clients:
                    client.close()
            snapshot = getattr(cfg, f"{stage}_config").save_path
            stage_dataset = load_dataset(snapshot)
            current = stage_dataset[0]
            # Gold-free data has no measured recall; the external hook keeps it None.
            summary["stages"][stage] = {"complete": current.is_stage_complete(stage), "seconds": round(time.monotonic() - stage_started, 3), "snapshot": str(snapshot)}
            if stage == "sql_generation":
                summary["sql_candidates"] = current.sql_candidates
            elif stage == "sql_revision":
                summary["sql_candidates_after_revision"] = current.sql_candidates_after_revision
            json_write(output / "summary.json", summary, env)
            json_write(output / "calls.json", recorder.events, env)
            if not current.is_stage_complete(stage) or observation_failures(recorder.events):
                raise RuntimeError(f"{stage} did not pass smoke coverage checks; see calls.json")
        final_item = stage_dataset[0]
        final = execute_postgres_sql(final_item, final_item.final_selected_sql, timeout=30)
        summary.update(final_sql=final_item.final_selected_sql, final_execution={
            "result_type": final.result_type, "columns": final.result_cols,
            "row_count": len(final.result_rows) if final.result_rows is not None else None,
            "error_message": final.error_message, "seconds": final.execution_time},
            native_pipeline_token_usage=final_item.total_llm_cost,
            schema_linking_recall=None)
        generated = {event["branch"] for event in recorder.events if event.get("branch", "").startswith("generation.") and event.get("output_count", 0) > 0}
        summary["all_three_generators_produced_candidates"] = len(generated) == 3
        summary["selection_pairwise_comparison_used"] = any(e.get("branch") == "selection.pairwise_comparison" for e in recorder.events)
        summary["success"] = all(v["complete"] for v in summary["stages"].values()) and len(generated) == 3 and final.result_rows is not None and not observation_failures(recorder.events)
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": redact(str(exc), env)}
        logger.error("Smoke failed: {}", summary["error"])
    finally:
        if undo_hooks is not None:
            undo_hooks()
        for key, value in old_pg.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        summary["wall_seconds"] = round(time.monotonic() - started, 3)
        summary["api_request_count"] = recorder.api_calls
        summary["observation_failures"] = observation_failures(recorder.events)
        json_write(output / "calls.json", recorder.events, env)
        json_write(output / "summary.json", summary, env)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("lite", "full"), required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--env-file", type=Path, default=CODE_ROOT / "config/.env")
    parser.add_argument("--preprocessed-dir", type=Path)
    parser.add_argument("--few-shot-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-values-per-column", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--max-api-calls", type=int, default=80)
    parser.add_argument("--chat-timeout", type=int, default=300,
                        help="Per-request timeout in seconds (native default: 300)")
    parser.add_argument("--thinking-budget", type=int,
                        help="Optional provider-specific thinking token budget; omitted preserves provider defaults")
    args = parser.parse_args()
    if not 1 <= args.max_values_per_column <= 1000 or not 128 <= args.max_tokens <= 16384 or not 1 <= args.max_api_calls <= 200 or not 30 <= args.chat_timeout <= 600:
        parser.error("Smoke limits: values 1..1000, max-tokens 128..16384, API calls 1..200, chat timeout 30..600")
    if args.thinking_budget is not None and not 1 <= args.thinking_budget <= 32768:
        parser.error("thinking-budget must be 1..32768 when supplied")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = CODE_ROOT / "outputs/deepeye_bird_interact" / f"{stamp}_{args.variant}"
    env = {}
    try:
        env = read_environment(args.env_file)
        summary = run_smoke(args, env)
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}", env), file=sys.stderr)
        return 1
    print(json.dumps({"success": summary["success"], "output_dir": str(args.output_dir), "error": summary.get("error")}, ensure_ascii=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
