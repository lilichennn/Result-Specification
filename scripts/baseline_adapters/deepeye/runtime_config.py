"""Shared BIRD-Interact runtime configuration and independent example selection."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shlex
from types import SimpleNamespace
from urllib.parse import urlsplit

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
