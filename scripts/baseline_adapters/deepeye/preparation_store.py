"""Small atomic checkpoints for expensive preparation steps, not a second run DB."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import json
import threading
import time
import uuid
import fcntl

from .precompute_cache import fingerprint
from .precompute_pipeline import read_record, write_record, utc_now

_LABEL = ContextVar('deepeye_preparation_label', default=None)


@contextmanager
def preparation_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Preparation output is already in use') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PreparationStore:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = fingerprint(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'identity.json'
        if path.exists():
            if read_record(path)['identity'] != self.identity:
                raise ValueError('Preparation checkpoint identity changed; select a new output directory')
        else:
            write_record(path, {'identity': self.identity})
        self._lock = threading.RLock()

    def step(self, name, function):
        if not name or not name.replace('_', '').isalnum():
            raise ValueError('Unsafe preparation step name')
        with self._lock:
            path = self.root / (name + '.json')
            if path.exists():
                record = read_record(path)
                if record.get('identity') != self.identity:
                    raise ValueError('Preparation step identity changed')
                return record['result']
            result = function()
            write_record(path, {'identity': self.identity, 'result': result, 'completed_at': utc_now()})
            return result


def bind_keyword_checkpoints(runner, root, *, audit=None, label=None):
    """Checkpoint this native runner's keyword result before later retrieval work.

    Preserve native fallback/order/usage exactly. The root is workload-scoped;
    credentials and gold answers never participate in a checkpoint identity.
    """
    from contextlib import nullcontext
    from copy import deepcopy
    import inspect
    from app.prompt import PromptFactory
    from app.llm_extractor import LLMExtractor
    from app.pipeline.value_retrieval import utils
    from .workloads import external_id

    original = runner._extract_keywords
    root, base_label = Path(root), dict(label or {})
    locks, locks_guard = {}, threading.Lock()
    rules = fingerprint([inspect.getsource(function) for function in (
        original, utils.extract_keywords, utils._parse_keywords_response,
        utils._post_process_keywords, LLMExtractor.extract_with_retry)])

    def extract(data_item):
        item_id = external_id(data_item)
        key = fingerprint({'item': item_id})
        config = runner._llm.llm_config
        identity = {'format': 'native-keywords-v1', 'item': item_id,
            'question': data_item.question, 'evidence': data_item.evidence,
            'database_id': data_item.database_id,
            'prompt': PromptFactory.format_keywords_extraction_prompt(data_item.question, data_item.evidence),
            'llm': config.model_dump(mode='json', exclude={'api_key'}),
            'extractor_max_retry': runner._extractor_max_retry,
            'actual_extractor_max_retry': (runner._keyword_extractor.max_retry
                                          if runner._keyword_extractor is not None else None),
            'sample_max_attempts': getattr(runner._llm, 'sample_max_attempts', None),
            'fix_end_token': config.fix_end_token, 'end_token': '</result>', 'n': 1,
            'native_rules': rules}
        with locks_guard:
            lock = locks.setdefault(key, threading.Lock())

        def run_native():
            keywords, usage = original(data_item)
            return {'keywords': keywords, 'usage': usage}

        context = audit.label({**base_label, 'item': item_id, 'step': 'keywords'}) if audit else nullcontext()
        with context, lock:
            # Reopen by identity on every call so changed mutable input/config
            # is rejected even within one runner; per-item locks deduplicate races.
            store = PreparationStore(root / key, identity)
            result = store.step('keywords', run_native)
            return deepcopy(result['keywords']), deepcopy(result['usage'])

    runner._extract_keywords = extract
    return runner


class PreparationAudit:
    """Append request attempts and usage without depending on a four-stage run."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event, payload=None):
        from .run_store import to_jsonable
        from .run_trace import _prepare_value
        if payload is not None:
            event = {'kind': event, **payload}
        record = to_jsonable(_prepare_value({'at': utc_now(), 'label': _LABEL.get(), **event}))
        with self._lock, self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()

    @contextmanager
    def label(self, value):
        token = _LABEL.set(value)
        try:
            yield
        finally:
            _LABEL.reset(token)

    def bind_llm(self, llm, runtime, *, default_label=None, bind_client=True):
        from types import SimpleNamespace
        if bind_client:
            runtime.bind_runner(SimpleNamespace(_llm=llm))
        llm.sample_max_attempts = 4
        original = llm.request_once
        def request(*args, **kwargs):
            started, call_id = time.monotonic(), uuid.uuid4().hex
            kwargs['timeout'] = min(kwargs.get('timeout') or runtime.limits.request_timeout, runtime.limits.request_timeout)
            label = _LABEL.get() or default_label
            self.emit({'kind': 'chat_request', 'call_id': call_id, 'label': label,
                       'model': llm.llm_config.model, 'messages': args[0] if args else kwargs.get('messages'),
                       'system_message': kwargs.get('system_message')})
            try:
                response = original(*args, **kwargs)
            except BaseException as error:
                self.emit({'kind': 'chat_error', 'call_id': call_id, 'label': label, 'error_type': type(error).__name__,
                           'status_code': getattr(error, 'status_code', None), 'seconds': time.monotonic()-started})
                raise
            usage = getattr(response, 'usage', None)
            self.emit({'kind': 'chat_response', 'call_id': call_id, 'label': label, 'seconds': time.monotonic()-started,
                       'usage': usage, 'usage_unknown': usage is None,
                       'choices': getattr(response, 'choices', None)})
            return response
        llm.request_once = request
