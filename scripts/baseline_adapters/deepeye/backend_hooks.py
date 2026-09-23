"""Runtime wrapper factories; importing this module does not patch DeepEye.

The caller installs these wrappers after loading the unmodified baseline.
Each factory receives the original native callable and preserves its behavior
for SQLite/cloud items. This makes installation and restoration explicit.
"""
from functools import wraps
import hashlib
import json

from .postgres_execution import execute_postgres_sql


def make_execute_sql_for_data_item(original):
    """Wrap app.db_utils.execution.execute_sql_for_data_item and bound imports."""
    @wraps(original)
    def execute(data_item, sql, timeout=None, **kwargs):
        if getattr(data_item, "db_type", None) == "postgresql":
            return execute_postgres_sql(data_item, sql, timeout=timeout)
        return original(data_item, sql, timeout=timeout, **kwargs)
    return execute


def make_measure_execution_time_for_data_item(original):
    """Wrap native timing; submit original SQL to read-only PostgreSQL."""
    @wraps(original)
    def measure(data_item, sql, timeout=None, repeat=10, initial_execution_time=None):
        if getattr(data_item, "db_type", None) != "postgresql":
            return original(data_item, sql, timeout=timeout, repeat=repeat,
                            initial_execution_time=initial_execution_time)
        elapsed = []
        for _ in range(max(1, repeat)):
            result = execute_postgres_sql(data_item, sql, timeout)
            if result.result_rows is None or result.execution_time is None:
                return float("inf")
            elapsed.append(result.execution_time)
        return sum(elapsed) / len(elapsed)
    return measure


def _postgres_result_key(data_item, sql, timeout):
    schema_fingerprint = hashlib.sha256(json.dumps(
        data_item.database_schema, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()
    return ("postgresql", data_item.database_id, schema_fingerprint, sql, timeout)


def make_result_cache_key(original):
    """Return a wrapper to install as ExecutionService._build_result_key staticmethod."""
    @wraps(original)
    def key(data_item, sql, timeout):
        if getattr(data_item, "db_type", None) == "postgresql":
            return _postgres_result_key(data_item, sql, timeout)
        return original(data_item, sql, timeout)
    return key


def make_time_cache_key(original):
    """Return a wrapper to install as ExecutionService._build_time_key staticmethod."""
    @wraps(original)
    def key(data_item, sql, timeout, repeat):
        if getattr(data_item, "db_type", None) == "postgresql":
            return (*_postgres_result_key(data_item, sql, timeout), repeat)
        return original(data_item, sql, timeout, repeat)
    return key


def make_execution_result_hash(original):
    """Wrap pipeline.utils.get_execution_result_hash and its bound imports."""
    @wraps(original)
    def result_hash(data_item, result_rows):
        if getattr(data_item, "db_type", None) == "postgresql":
            if result_rows is None:
                return None
            from app.pipeline.utils import make_hashable

            return frozenset(make_hashable(result_rows))
        return original(data_item, result_rows)
    return result_hash
