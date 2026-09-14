"""Fixed Result Contract prompt injection for DeepEye's current PromptFactory."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import hashlib
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping

from result_contract.rc import Round2RC


_PROMPT_PATH = Path(__file__).with_name("rc_prompt.txt")
_FINAL_MARKER = "<<FINAL_RC>>"
_STAGES = frozenset(
    {"schema_linking", "sql_generation", "sql_revision", "sql_selection"}
)
_FORMAT_METHODS = tuple(
    f"format_{stem}_prompt"
    for stem in (
        "direct_linking",
        "skeleton_sql_generation",
        "dc_sql_generation",
        "icl_sql_generation",
        "execution_checker",
        "common_checker",
        "br_pair_selection",
    )
)
_RC_BLOCK: ContextVar[str | None] = ContextVar("deepeye_rc_block", default=None)
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


def _prompt_template(*, legacy=False) -> str:
    path = _PROMPT_PATH.with_name('rc_prompt_legacy.txt') if legacy else _PROMPT_PATH
    try:
        template = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"RC prompt definition cannot be read: {_PROMPT_PATH}") from error
    marker = '<<ROUND2_RC>>' if legacy else _FINAL_MARKER
    if template.count(marker) != 1:
        raise RuntimeError("RC prompt definition must contain exactly one final RC marker")
    return template


def freeze_prompt() -> dict:
    text = _prompt_template()
    return {'text': text, 'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}


def manifest_prompt(manifest: Mapping) -> str | None:
    frozen = manifest.get('rc_prompt')
    if frozen is None:
        return None
    text = frozen['text']
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != frozen['sha256']:
        raise ValueError('Frozen RC prompt hash mismatch')
    return text


def rc_labels(manifest: Mapping) -> dict:
    version = manifest.get('rc_version', 2) if manifest.get('condition') == 'rc' else None
    return {'rc_version': version, 'gold_corrected': version == 3}


def render_rc_block(contract: dict, *, prompt_template: str | None = None) -> str:
    """Render only six selected fields; legacy contracts use archived prose."""

    if not isinstance(contract, Mapping):
        raise TypeError("contract must be a dictionary")
    try:
        versioned = 'rc_version' in contract
        if versioned and (type(contract['rc_version']) is not int or contract['rc_version'] not in (2, 3)):
            raise ValueError('Unsupported RC version')
        final_rc = Round2RC.from_value(contract.get('final_rc') if versioned else contract.get('round2')).to_dict()
    except ValueError as error:
        raise ValueError(f"contract has no valid final RC: {error}") from error
    serialized = json.dumps(final_rc, ensure_ascii=False, indent=2)
    template = prompt_template if prompt_template is not None else _prompt_template(legacy=not versioned)
    marker = _FINAL_MARKER if _FINAL_MARKER in template else '<<ROUND2_RC>>'
    if template.count(marker) != 1:
        raise ValueError('RC prompt template must have exactly one final RC marker')
    return template.replace(marker, serialized)


@contextmanager
def rc_context(
    stage: str, task_key: str, contract: dict | None, *, prompt_template: str | None = None
) -> Iterator[None]:
    """Bind one task's RC block to the current execution context."""

    if stage not in _STAGES:
        raise ValueError(f"Unknown DeepEye stage: {stage!r}")
    if not isinstance(task_key, str) or not task_key:
        raise ValueError("task_key must be non-empty text")
    if contract is None:
        block = None
    else:
        if not isinstance(contract, Mapping):
            raise TypeError("contract must be a dictionary or None")
        if contract.get("task_key") != task_key:
            raise ValueError(
                f"RC task key mismatch: context={task_key!r}, contract={contract.get('task_key')!r}"
            )
        block = render_rc_block(dict(contract), prompt_template=prompt_template)
    token = _RC_BLOCK.set(block)
    try:
        yield
    finally:
        _RC_BLOCK.reset(token)


def _with_rc(original):
    @wraps(original)
    def format_prompt(*args, **kwargs):
        prompt = original(*args, **kwargs)
        block = _RC_BLOCK.get()
        if block is None:
            return prompt
        if not isinstance(prompt, str):
            raise TypeError("DeepEye prompt formatter returned a non-string value")
        return f"{prompt}\n\n{block}"

    return format_prompt


@contextmanager
def install_rc_prompts() -> Iterator[None]:
    """Reversibly wrap the seven PromptFactory methods currently installed."""

    global _INSTALLED
    from app.prompt.factory import PromptFactory

    with _INSTALL_LOCK:
        if _INSTALLED:
            raise RuntimeError("DeepEye RC prompts are already installed")
        _INSTALLED = True

    originals: list[tuple[str, Any]] = []
    try:
        for name in _FORMAT_METHODS:
            descriptor = vars(PromptFactory).get(name)
            if descriptor is None:
                raise RuntimeError(f"DeepEye PromptFactory method is missing: {name}")
            originals.append((name, descriptor))
            setattr(PromptFactory, name, staticmethod(_with_rc(getattr(PromptFactory, name))))
        yield
    finally:
        for name, descriptor in reversed(originals):
            setattr(PromptFactory, name, descriptor)
        with _INSTALL_LOCK:
            _INSTALLED = False


def _contains_complete_block(value: Any, block: str) -> bool:
    if isinstance(value, str):
        return block in value
    if isinstance(value, Mapping):
        return any(_contains_complete_block(item, block) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_complete_block(item, block) for item in value)
    return False


def count_rc_requests(events: list[dict], block: str) -> int:
    """Count API request events whose actual request data contains the full block."""

    if not isinstance(block, str) or not block:
        raise ValueError("block must be non-empty text")
    count = 0
    for event in events:
        if not isinstance(event, Mapping) or event.get("kind") != "api_request":
            continue
        payload = event.get("payload")
        kwargs = payload.get("kwargs") if isinstance(payload, Mapping) else None
        messages = kwargs.get("messages") if isinstance(kwargs, Mapping) else None
        if _contains_complete_block(messages, block):
            count += 1
    return count
