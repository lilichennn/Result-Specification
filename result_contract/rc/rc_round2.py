from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rc_round1 import (
    DEFAULT_MAX_ATTEMPTS,
    ModelCall,
    Round1RC,
    call_model,
)


ROUND2_FIELDS = (
    "population",
    "row_grain",
    "column_role",
    "derivation",
    "filter_policy",
    "meta_review",
)
PROMPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Round2RC:
    population: str
    row_grain: str
    column_role: str
    derivation: str
    filter_policy: str
    meta_review: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "Round2RC":
        if not isinstance(value, Mapping):
            raise ValueError("Round-2 RC must be a JSON object")
        if set(value) != set(ROUND2_FIELDS):
            missing = sorted(set(ROUND2_FIELDS) - set(value))
            unexpected = sorted(set(value) - set(ROUND2_FIELDS))
            raise ValueError(
                f"Round-2 RC fields do not match: missing={missing}, "
                f"unexpected={unexpected}"
            )

        normalized: dict[str, str] = {}
        for field in ROUND2_FIELDS:
            field_value = value[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"Round-2 RC field must be non-empty text: {field}")
            normalized[field] = field_value.strip()
        return cls(**normalized)


def generate_round2(
    question: str,
    evidence: str,
    round1_rc: Round1RC | Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
    model_call: ModelCall | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    metadata_complete: bool = True,
) -> Round2RC:
    """Conservatively refine a Round-1 RC using database metadata."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    messages = build_round2_messages(
        question=question,
        evidence=evidence,
        round1_rc=round1_rc,
        metadata=metadata,
        metadata_complete=metadata_complete,
    )
    invoke = model_call or call_model
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            LOGGER.info("Round-2 model request attempt %d/%d", attempt, max_attempts)
            result = parse_round2_response(invoke(messages))
            LOGGER.info("Round-2 model request succeeded on attempt %d/%d", attempt, max_attempts)
            return result
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Round-2 model request failed on attempt %d/%d: %s: %s",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(attempt)
    assert last_error is not None
    raise RuntimeError(f"Round-2 RC generation failed after {max_attempts} attempts") from last_error


def build_round2_messages(
    question: str,
    evidence: str,
    round1_rc: Round1RC | Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
    metadata_complete: bool = True,
) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(evidence, str):
        raise ValueError("evidence must be text")
    normalized_round1 = (
        round1_rc if isinstance(round1_rc, Round1RC) else Round1RC.from_value(round1_rc)
    )
    normalized_metadata = _normalize_metadata(metadata)

    user_prompt = _load_prompt("rc_round2_user.txt")
    replacements = {
        "<<QUESTION>>": question.strip(),
        "<<EVIDENCE>>": evidence.strip() or "(empty)",
        "<<ROUND1_RC>>": json.dumps(
            normalized_round1.to_dict(), ensure_ascii=False, indent=2
        ),
        "<<METADATA_SCOPE>>": (
            "complete database metadata catalog"
            if metadata_complete
            else "selected metadata subset"
        ),
        "<<METADATA>>": json.dumps(normalized_metadata, ensure_ascii=False, indent=2),
    }
    for marker, replacement in replacements.items():
        user_prompt = user_prompt.replace(marker, replacement)

    return [
        {"role": "system", "content": _load_prompt("rc_round2_system.txt")},
        {"role": "user", "content": user_prompt},
    ]


def parse_round2_response(raw_response: str) -> Round2RC:
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("model response must be non-empty text")
    content = raw_response.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model response is not valid JSON: {exc}") from exc
    return Round2RC.from_value(value)


def _normalize_metadata(
    metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(metadata, (str, bytes)) or not isinstance(metadata, Sequence):
        raise ValueError("metadata must be a sequence of table metadata objects")
    normalized: list[dict[str, Any]] = []
    for position, table in enumerate(metadata):
        if not isinstance(table, Mapping):
            raise ValueError(f"metadata table at position {position} must be an object")
        table_name = table.get("table_name")
        columns = table.get("columns")
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError(f"metadata table at position {position} has no table_name")
        if not isinstance(columns, list):
            raise ValueError(f"metadata table {table_name!r} must contain a columns list")
        normalized_columns: list[dict[str, Any]] = []
        for column_position, column in enumerate(columns):
            if not isinstance(column, Mapping):
                raise ValueError(
                    f"metadata column {column_position} in table {table_name!r} "
                    "must be an object"
                )
            normalized_columns.append(dict(column))
        normalized.append(
            {"table_name": table_name.strip(), "columns": normalized_columns}
        )
    if not normalized:
        raise ValueError("metadata must contain at least one table")
    return normalized


@lru_cache(maxsize=None)
def _load_prompt(filename: str) -> str:
    prompt_path = PROMPT_DIR / filename
    if not prompt_path.is_file():
        raise FileNotFoundError(f"RC prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()
