"""Observe native preliminary generation/selection without changing its policy.

Attach once before submitting questions through the preparation runtime's
context-propagating executors. No SQL or model request is repeated for logging.
"""
from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
import threading
import time

from .precompute_cache import fingerprint
from .run_store import to_jsonable
from .run_trace import _prepare_value


_CURRENT = ContextVar('deepeye_preliminary_diagnostics', default=None)


def _short(value, limit=512):
    if value is None or isinstance(value, (bool, int, float)):
        return to_jsonable(value)
    text = str(value)
    return text if len(text) <= limit else text[:limit] + '…'


class _ExecutionObserver:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def execute(self, item, sql, *args, **kwargs):
        state = _CURRENT.get()
        if state is None:
            return self.original.execute(item, sql, *args, **kwargs)
        record = {'sql': sql, 'result_hash': None}
        state['diagnostics']['executions'].append(record)
        started = time.monotonic()
        try:
            result = self.original.execute(item, sql, *args, **kwargs)
        except BaseException as error:
            record.update(error_type=type(error).__name__, error_message=_short(error))
            raise
        else:
            rows, columns = result.result_rows, result.result_cols
            record.update(result_type=result.result_type, execution_time=result.execution_time,
                error_message=_short(result.error_message), row_count=len(rows) if rows is not None else None,
                column_count=len(columns) if columns is not None else None,
                columns_preview=[_short(column) for column in (columns or [])[:20]],
                rows_preview=[[_short(cell) for cell in row[:20]] for row in (rows or [])[:20]],
                preview_truncated=bool(rows and (len(rows) > 20 or any(len(row) > 20 for row in rows[:20]))))
            if rows is not None:
                state['rows'][id(rows)] = record
            return result
        finally:
            record['observed_seconds'] = time.monotonic() - started

    def hash_result(self, item, rows, *args, **kwargs):
        # Preserve exactly the native equivalence key and number of calls.
        result = self.original.hash_result(item, rows, *args, **kwargs)
        state = _CURRENT.get()
        if state is not None and id(rows) in state['rows']:
            # Native keys can contain every row. Store a digest, never that key.
            state['rows'][id(rows)]['result_hash'] = fingerprint(to_jsonable(_prepare_value(result)))
        return result


def attach_preliminary_diagnostics(generator, audit=None):
    """Return a question callable producing native fields plus diagnostics.

    Exceptions retain the native propagation/swallowing behavior. The optional
    audit also receives the partial diagnostic record when native execution
    raises before a successful per-question checkpoint can be written.
    """
    existing = getattr(generator, '_preparation_diagnostic_call', None)
    if existing is not None:
        return existing

    for name in ('dc', 'skeleton'):
        branch = getattr(generator, '_' + name + '_generator')
        original = branch.generate

        def observe(function, branch_name):
            @wraps(function)
            def wrapped(*args, **kwargs):
                state = _CURRENT.get()
                if state is None:
                    return function(*args, **kwargs)
                started = time.monotonic()
                record = {'status': 'running'}
                with state['lock']:
                    state['diagnostics']['branches'][branch_name] = record
                try:
                    result = function(*args, **kwargs)
                except BaseException as error:
                    record.update(status='failed', error_type=type(error).__name__, error_message=_short(error))
                    raise
                else:
                    candidates, usage = result
                    record.update(status='succeeded', candidates=list(candidates or []), token_usage=usage)
                    return result
                finally:
                    record['seconds'] = time.monotonic() - started
            return wrapped

        branch.generate = observe(original, name)

    generator._execution_service = _ExecutionObserver(generator._execution_service)

    def run(item):
        diagnostics = {'branches': {}, 'executions': [], 'fallback_reason': None}
        state = {'diagnostics': diagnostics, 'rows': {}, 'lock': threading.Lock()}
        token = _CURRENT.set(state)
        try:
            result = asdict(generator.generate(item))
            if result['sql'] is None:
                if any(row['status'] == 'failed' for row in diagnostics['branches'].values()):
                    diagnostics['fallback_reason'] = 'generation_error'
                elif not result['candidates']:
                    diagnostics['fallback_reason'] = 'no_candidates'
                else:
                    diagnostics['fallback_reason'] = 'no_executable_candidates'
            return dict(result, diagnostics=diagnostics)
        except BaseException as error:
            diagnostics['error_type'] = type(error).__name__
            raise
        finally:
            _CURRENT.reset(token)
            if audit is not None:
                audit.emit({'kind': 'preliminary_diagnostics', 'diagnostics': diagnostics})

    generator._preparation_diagnostic_call = run
    return run
