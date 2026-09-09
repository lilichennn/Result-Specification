from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.request import Request, urlopen


ROUND1_FIELDS = (
    "population",
    "row_grain",
    "column_role",
    "derivation",
    "filter_policy",
)
PROMPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 600
LOGGER = logging.getLogger(__name__)


class ModelCall(Protocol):
    """Model adapter boundary used by RC generation."""

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str: ...


@dataclass(frozen=True)
class Round1RC:
    population: str
    row_grain: str
    column_role: str
    derivation: str
    filter_policy: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "Round1RC":
        if not isinstance(value, Mapping):
            raise ValueError("Round-1 RC must be a JSON object")
        if set(value) != set(ROUND1_FIELDS):
            missing = sorted(set(ROUND1_FIELDS) - set(value))
            unexpected = sorted(set(value) - set(ROUND1_FIELDS))
            raise ValueError(
                f"Round-1 RC fields do not match: missing={missing}, unexpected={unexpected}"
            )

        normalized: dict[str, str] = {}
        for field in ROUND1_FIELDS:
            field_value = value[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"Round-1 RC field must be non-empty text: {field}")
            normalized[field] = field_value.strip()
        return cls(**normalized)


def generate_round1(
    question: str,
    evidence: str,
    model_call: ModelCall | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Round1RC:
    """Generate a Round-1 RC from question and evidence."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    messages = build_round1_messages(question=question, evidence=evidence)
    invoke = model_call or call_model
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            LOGGER.debug("Round-1 model request attempt %d/%d", attempt, max_attempts)
            result = parse_round1_response(invoke(messages))
            LOGGER.debug("Round-1 model request succeeded on attempt %d/%d", attempt, max_attempts)
            return result
        except Exception as exc:
            last_error = exc
            LOGGER.debug(
                "Round-1 model request failed on attempt %d/%d: %s: %s",
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(attempt)
    assert last_error is not None
    raise RuntimeError(f"Round-1 RC generation failed after {max_attempts} attempts") from last_error


def call_model(
    messages: Sequence[Mapping[str, str]],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Call the OpenAI-compatible endpoint configured in code/config/.env once."""
    _load_env(Path(config_path))
    base_url = os.environ["DASH_BASE_URL"].rstrip("/")
    api_key = os.environ["DASH_API_KEY"]
    model = os.environ["DASH_MODELS"]
    payload = json.dumps(
        {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model response does not contain choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty content")
    return content


def build_round1_messages(question: str, evidence: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(evidence, str):
        raise ValueError("evidence must be text")

    user_prompt = _load_prompt("rc_round1_user.txt")
    user_prompt = user_prompt.replace("<<QUESTION>>", question.strip())
    user_prompt = user_prompt.replace("<<EVIDENCE>>", evidence.strip() or "(empty)")
    return [
        {"role": "system", "content": _load_prompt("rc_round1_system.txt")},
        {"role": "user", "content": user_prompt},
    ]


def parse_round1_response(raw_response: str) -> Round1RC:
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("model response must be non-empty text")
    content = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model response is not valid JSON: {exc}") from exc
    return Round1RC.from_value(value)


@lru_cache(maxsize=None)
def _load_prompt(filename: str) -> str:
    prompt_path = PROMPT_DIR / filename
    if not prompt_path.is_file():
        raise FileNotFoundError(f"RC prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _load_env(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"LLM config not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
