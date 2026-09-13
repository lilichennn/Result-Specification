"""Per-sample retry execution; dependency-neutral observation hooks."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable
from uuid import uuid4
import hashlib
import marshal
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

MAX_SAMPLE_ATTEMPTS = 4
TOKEN_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
_OBSERVER = ContextVar('deepeye_sampling_observer', default=None)
_IDENTITY = ContextVar('deepeye_sampling_identity', default=None)
_CHECKPOINTS = ContextVar('deepeye_sampling_checkpoints', default=None)


class SamplingPaused(BaseException):
    """Cooperative stop; native Exception fallbacks must not swallow it."""


class SamplingIdentityError(BaseException):
    """Invalid recovery identity must not become an optional native fallback."""


@contextmanager
def sampling_checkpoints(checkpoints):
    token = _CHECKPOINTS.set(checkpoints)
    try:
        yield
    finally:
        _CHECKPOINTS.reset(token)


def check_sampling_stop():
    checkpoints = _CHECKPOINTS.get()
    if checkpoints is not None:
        checkpoints.check_stop()


def _parser_owner(value, names, seen):
    """Capture the attributes a parser references, not executor/client state."""
    if value is None or isinstance(value, (bool, int, float, str, dict, list, tuple, set)):
        return value
    if callable(value):
        return parser_identity(value, seen)
    kind = {'module': type(value).__module__, 'name': type(value).__qualname__}
    if id(value) in seen:
        return kind
    seen = (*seen, id(value))
    return {**kind, 'attributes': {name: _parser_owner(getattr(value, name), (), seen)
            for name in names if hasattr(value, name)}}


def _parser_state(value, seen=()):
    """Only lossless, explicit state is eligible for wrapped-callable reuse."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if id(value) in seen:
        raise TypeError('unsupported parser callable: cyclic state')
    seen = (*seen, id(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('unsupported parser callable: non-string state keys')
        return {key: _parser_state(item, seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {'container': type(value).__name__, 'items': [_parser_state(item, seen) for item in value]}
    raise TypeError(f'unsupported parser callable state: {type(value).__qualname__}')


def parser_identity(parser, seen=()):
    try:
        return _parser_identity(parser, seen)
    except Exception as error:
        raise SamplingIdentityError(str(error)) from error


def _parser_identity(parser, seen=()):
    if type(parser) is partial:
        return {'kind': 'partial', 'function': parser_identity(parser.func, seen),
                'args': _parser_state(parser.args), 'keywords': _parser_state(parser.keywords)}
    # Callable instances may depend on mutable class attributes, descriptors,
    # nested helpers or opaque state. Do not infer a complete identity for them.
    if not isinstance(parser, (FunctionType, MethodType)):
        raise TypeError(f'unsupported parser callable: {type(parser).__qualname__}')
    code = parser.__code__
    closure = getattr(parser, '__closure__', None) or ()
    names = code.co_names if code else ()
    return {'module': getattr(parser, '__module__', type(parser).__module__),
            'name': getattr(parser, '__qualname__', type(parser).__qualname__),
            'code': hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
            'defaults': getattr(parser, '__defaults__', None),
            'keyword_defaults': getattr(parser, '__kwdefaults__', None),
            'owner': _parser_owner(getattr(parser, '__self__', None), names, seen),
            'closure': [_parser_owner(cell.cell_contents, names, seen) for cell in closure]}


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
    response_id: str | None = None
    restored_from_event: int | None = None
    rc_applied: bool = False


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
    checkpoints = _CHECKPOINTS.get()
    check_sampling_stop()
    outcome = checkpoints.restore(group_id, sample_index) if checkpoints else SampleOutcome(group_id, sample_index)
    limit = checkpoints.attempt_limit(group_id, sample_index) if checkpoints else max_attempts
    for number in range(len(outcome.attempts) + 1, limit + 1):
        if outcome.succeeded or outcome.fatal:
            break
        check_sampling_stop()
        identity = {'group_id': group_id, 'sample_index': sample_index, 'sample_attempt': number}
        if checkpoints:
            checkpoints.start_attempt(identity)
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
                        outcome.response_id = _get(response, 'id')
        finally:
            _IDENTITY.reset(token)
        outcome.attempts.append(SampleAttempt(number, status, usage, error_text))
        if checkpoints:
            checkpoints.finish_sample_attempt(identity, outcome)
        _emit('sample_attempt', {**identity, 'status': status, 'usage': usage, 'error': error_text})
        if outcome.succeeded or outcome.fatal:
            break
    _emit('sample_result', {'group_id': group_id, 'sample_index': sample_index,
        'succeeded': outcome.succeeded, 'fatal': outcome.fatal, 'result': outcome.result,
        'usage': outcome.usage, 'attempt_count': len(outcome.attempts),
        'response_id': outcome.response_id, 'restored_from_event': outcome.restored_from_event,
        'rc_applied': outcome.rc_applied})
    return outcome


def execute_group(request: Callable, parser: Callable, *, n: int,
                  max_attempts: int = MAX_SAMPLE_ATTEMPTS, recovery_identity=None):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('sampling target must be a positive integer')
    check_sampling_stop()
    checkpoints = _CHECKPOINTS.get()
    group_id = checkpoints.group(recovery_identity, n, max_attempts) if checkpoints else uuid4().hex
    group = GroupOutcome(group_id, n, [])
    _emit('sampling_group_start', {'group_id': group.group_id, 'target_n': n})
    for index in range(n):
        sample = execute_sample(request, parser, group_id=group.group_id,
                                sample_index=index, max_attempts=max_attempts)
        group.samples.append(sample)
        if sample.fatal:
            break
    _emit('sampling_group_result', {'group_id': group.group_id, 'target_n': n,
        'success_count': len(group.results), 'complete': group.complete})
    check_sampling_stop()
    return group
