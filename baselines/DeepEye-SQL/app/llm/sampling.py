"""Per-sample retry execution; dependency-neutral observation hooks."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

MAX_SAMPLE_ATTEMPTS = 4
TOKEN_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
_OBSERVER = ContextVar('deepeye_sampling_observer', default=None)
_IDENTITY = ContextVar('deepeye_sampling_identity', default=None)


@contextmanager
def observe_sampling(observer):
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def sampling_identity():
    return dict(_IDENTITY.get() or {})


def _emit(kind, payload):
    observer = _OBSERVER.get()
    if observer is not None:
        observer(kind, payload)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def response_usage(response):
    usage = _get(response, 'usage')
    if usage is None:
        return None
    result = {}
    for name in TOKEN_FIELDS:
        value = _get(usage, name)
        result[name] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    reasoning = _get(_get(usage, 'completion_tokens_details'), 'reasoning_tokens', _get(usage, 'reasoning_tokens'))
    result['reasoning_tokens'] = reasoning if isinstance(reasoning, int) and not isinstance(reasoning, bool) and reasoning >= 0 else None
    return result


def error_usage(error):
    """Capture usage attached to parsed or HTTP error responses."""
    for candidate in (getattr(error, 'body', None), getattr(error, 'response', None)):
        usage = response_usage(candidate)
        if usage is not None:
            return usage
        json_body = getattr(candidate, 'json', None)
        if callable(json_body):
            try:
                usage = response_usage(json_body())
            except (ValueError, TypeError):
                continue
            if usage is not None:
                return usage
    return None


@dataclass
class SampleAttempt:
    attempt_number: int
    status: str
    usage: dict | None
    error: str | None = None


@dataclass
class SampleOutcome:
    group_id: str
    sample_index: int
    result: Any = None
    usage: dict | None = None
    attempts: list[SampleAttempt] = field(default_factory=list)
    succeeded: bool = False
    fatal: bool = False


@dataclass
class GroupOutcome:
    group_id: str
    target_n: int
    samples: list[SampleOutcome]

    @property
    def results(self):
        return [sample.result for sample in self.samples if sample.succeeded]

    @property
    def complete(self):
        return len(self.results) == self.target_n

    @property
    def effective_usage(self):
        # Native consumers need numeric three-field costs. This is a known
        # subtotal; nullable sample usage and completeness remain in the ledger.
        return {key: sum((sample.usage or {}).get(key) or 0 for sample in self.samples
                         if sample.succeeded) for key in TOKEN_FIELDS}


class SamplingIncompleteError(RuntimeError):
    def __init__(self, outcome):
        self.outcome = outcome
        super().__init__(f'Incomplete sampling group: {len(outcome.results)}/{outcome.target_n}')


def execute_sample(request: Callable, parser: Callable, *, group_id: str,
                   sample_index: int, max_attempts: int = MAX_SAMPLE_ATTEMPTS):
    """One fixed sample; schedulers may reuse this function unchanged."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= MAX_SAMPLE_ATTEMPTS:
        raise ValueError('sample attempts must be between 1 and 4, including first')
    outcome = SampleOutcome(group_id, sample_index)
    for number in range(1, max_attempts + 1):
        identity = {'group_id': group_id, 'sample_index': sample_index, 'sample_attempt': number}
        token = _IDENTITY.set(identity)
        error_text, usage = None, None
        try:
            try:
                response = request()
            except Exception as error:
                usage = error_usage(error)
                status, error_text = 'api_error', f'{type(error).__name__}: {error}'
                outcome.fatal = not isinstance(error, (RateLimitError, APITimeoutError,
                                                      APIConnectionError, InternalServerError))
            else:
                usage = response_usage(response)
                choices = _get(response, 'choices') or []
                message = _get(choices[0], 'message') if choices else None
                content = _get(message, 'content')
                if not isinstance(content, str) or not content.strip():
                    status = 'empty'
                else:
                    try:
                        result = parser(message)
                        valid = result is not None and not (
                            isinstance(result, (dict, list, str, tuple, set)) and not result)
                        status = 'succeeded' if valid else 'parse_rejected'
                    except Exception as error:
                        status, error_text = 'parse_rejected', f'{type(error).__name__}: {error}'
                    if status == 'succeeded':
                        outcome.result, outcome.usage, outcome.succeeded = result, usage, True
        finally:
            _IDENTITY.reset(token)
        outcome.attempts.append(SampleAttempt(number, status, usage, error_text))
        _emit('sample_attempt', {**identity, 'status': status, 'usage': usage, 'error': error_text})
        if outcome.succeeded or outcome.fatal:
            break
    _emit('sample_result', {'group_id': group_id, 'sample_index': sample_index,
        'succeeded': outcome.succeeded, 'fatal': outcome.fatal, 'result': outcome.result,
        'usage': outcome.usage, 'attempt_count': len(outcome.attempts)})
    return outcome


def execute_group(request: Callable, parser: Callable, *, n: int,
                  max_attempts: int = MAX_SAMPLE_ATTEMPTS):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('sampling target must be a positive integer')
    group = GroupOutcome(uuid4().hex, n, [])
    _emit('sampling_group_start', {'group_id': group.group_id, 'target_n': n})
    for index in range(n):
        sample = execute_sample(request, parser, group_id=group.group_id,
                                sample_index=index, max_attempts=max_attempts)
        group.samples.append(sample)
        if sample.fatal:
            break
    _emit('sampling_group_result', {'group_id': group.group_id, 'target_n': n,
        'success_count': len(group.results), 'complete': group.complete})
    return group
