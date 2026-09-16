"""Frozen DAIL-SQL experiment settings and task identity."""

from dataclasses import dataclass
import json
from pathlib import Path


MODES = ("native", "rc_first", "rc_second", "rc_both")


@dataclass(frozen=True)
class TaskKey:
    batch_id: str
    group: str
    question_id: str

    def __post_init__(self):
        for field in ("batch_id", "group", "question_id"):
            original = getattr(self, field)
            if original is None:
                raise ValueError(f"empty {field}")
            value = str(original)
            if not value or value.isspace():
                raise ValueError(f"empty {field}")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class DailSettings:
    k_shot: int = 9
    n: int = 1
    samples_per_round: int = 5
    enable_thinking: bool = True
    temperature: float = 0.6
    max_attempts: int = 5  # Per sample: initial request plus four retries.
    request_timeout_seconds: float = 910.0
    max_tokens: int | None = None

    def validate(self):
        if ((self.k_shot, self.n, self.samples_per_round, self.temperature,
             self.max_attempts, self.request_timeout_seconds) != (9, 1, 5, 0.6, 5, 910)
                or self.enable_thinking is not True):
            raise ValueError("DAIL-SQL confirmed experiment settings must remain frozen")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer or None")
        return self

    def request_kwargs(self, model: str, messages: list[dict]) -> dict:
        self.validate()
        if not model or not messages:
            raise ValueError("model and messages are required")
        kwargs = {"model": model, "messages": messages, "n": self.n,
                  "temperature": self.temperature,
                  "extra_body": {"enable_thinking": self.enable_thinking}}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs


def load_experiment_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("format") != "dail-sql-inputs-v1":
        raise ValueError("unsupported experiment config version")
    DailSettings(**config["settings"]).validate()
    return config
