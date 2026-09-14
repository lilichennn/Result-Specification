from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rc_round1 import DEFAULT_MAX_ATTEMPTS, ModelCall, call_model
from .rc_round2 import ROUND2_FIELDS, Round2RC


PROMPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Round3RC:
    population: str
    row_grain: str
    column_role: str
    derivation: str
    filter_policy: str
    meta_review: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "Round3RC":
        if not isinstance(value, Mapping):
            raise ValueError("Round-3 RC must be a JSON object")
        if set(value) != set(ROUND2_FIELDS):
            missing = sorted(set(ROUND2_FIELDS) - set(value))
            unexpected = sorted(set(value) - set(ROUND2_FIELDS))
            raise ValueError(
                f"Round-3 RC fields do not match: missing={missing}, "
                f"unexpected={unexpected}"
            )

        normalized: dict[str, str] = {}
        for field in ROUND2_FIELDS:
            field_value = value[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"Round-3 RC field must be non-empty text: {field}")
            normalized[field] = field_value.strip()
        return cls(**normalized)


def generate_round3(
    question: str,
    evidence: str,
    round2_rc: Round2RC | Mapping[str, Any],
    gold_sql: str,
    model_call: ModelCall | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Round3RC:
    """Minimally correct a Round-2 RC using the benchmark gold SQL."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    messages = build_round3_messages(
        question=question,
        evidence=evidence,
        round2_rc=round2_rc,
        gold_sql=gold_sql,
    )
    invoke = model_call or call_model
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            LOGGER.debug("Round-3 model request attempt %d/%d", attempt, max_attempts)
            result = parse_round3_response(invoke(messages))
            LOGGER.debug(
                "Round-3 model request succeeded on attempt %d/%d",
                attempt,
                max_attempts,
            )
            return result
        except Exception as exc:
            last_error = exc
            LOGGER.debug(
                "Round-3 model request failed on attempt %d/%d: %s: %s",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(attempt)
    assert last_error is not None
    raise RuntimeError(
        f"Round-3 RC generation failed after {max_attempts} attempts"
    ) from last_error


def build_round3_messages(
    question: str,
    evidence: str,
    round2_rc: Round2RC | Mapping[str, Any],
    gold_sql: str,
) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(evidence, str):
        raise ValueError("evidence must be text")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError("gold_sql must be non-empty text")
    normalized_round2 = (
        round2_rc if isinstance(round2_rc, Round2RC) else Round2RC.from_value(round2_rc)
    )

    user_prompt = _load_prompt("rc_round3_user.txt")
    replacements = {
        "<<QUESTION>>": question.strip(),
        "<<EVIDENCE>>": evidence.strip() or "(empty)",
        "<<ROUND2_RC>>": json.dumps(
            normalized_round2.to_dict(), ensure_ascii=False, indent=2
        ),
        "<<GOLD_SQL>>": gold_sql.strip(),
    }
    for marker, replacement in replacements.items():
        user_prompt = user_prompt.replace(marker, replacement)

    return [
        {"role": "system", "content": _load_prompt("rc_round3_system.txt")},
        {"role": "user", "content": user_prompt},
    ]


def parse_round3_response(raw_response: str) -> Round3RC:
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
    return Round3RC.from_value(value)


@lru_cache(maxsize=None)
def _load_prompt(filename: str) -> str:
    prompt_path = PROMPT_DIR / filename
    if not prompt_path.is_file():
        raise FileNotFoundError(f"RC prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()
