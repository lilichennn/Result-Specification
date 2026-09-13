"""Independent strict result equality; never use native frozenset hashes."""
from collections import Counter
import datetime as dt
from decimal import Decimal
from fractions import Fraction
import math
from uuid import UUID

SUCCESS_TYPES = frozenset({'success', 'empty_result', 'all_null_result'})


def _cell(value):
    if value is None:
        return ('null',)
    if type(value) is bool:
        return ('bool', value)
    if type(value) is str:
        return ('str', value)
    if type(value) is int:
        return ('number', Fraction(value))
    if type(value) is float and math.isfinite(value):
        return ('number', Fraction.from_float(value))
    if type(value) is Decimal and value.is_finite():
        return ('number', Fraction(value))
    if type(value) in (list, tuple):
        return ('array', tuple(_cell(item) for item in value))
    if type(value) is dict and all(type(key) is str for key in value):
        return ('object', tuple((key, _cell(value[key])) for key in sorted(value)))
    if type(value) is bytes:
        return ('bytes', value)
    if type(value) is UUID:
        return ('uuid', value.int)
    if type(value) is dt.date:
        return ('date', value.toordinal())
    if type(value) in (dt.datetime, dt.time):
        micros = ((value.hour * 60 + value.minute) * 60 + value.second) * 1_000_000 + value.microsecond
        if type(value) is dt.datetime:
            micros += value.toordinal() * 86_400_000_000
        offset = value.utcoffset()
        if offset is not None:
            micros -= (offset.days * 86400 + offset.seconds) * 1_000_000 + offset.microseconds
        return (type(value).__name__, 'aware' if offset is not None else 'naive', micros)
    raise ValueError('nonfinite_or_unsupported_cell_type:' + type(value).__name__)


def _rows(result):
    if not isinstance(result, dict) or result.get('result_type') not in SUCCESS_TYPES:
        raise ValueError('execution_unavailable')
    cols, rows = result.get('result_cols'), result.get('result_rows')
    if not isinstance(cols, (list, tuple)) or not isinstance(rows, (list, tuple)):
        raise ValueError('missing_result_shape')
    normalized = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(cols):
            raise ValueError('invalid_row_shape')
        normalized.append(tuple(_cell(value) for value in row))
    return len(cols), normalized


def compare_results(predicted: dict, reference: dict) -> dict:
    """Bag and ordered equality with exact finite numeric equivalence.

    Floats retain their exact binary value (0.1 != Decimal('0.1')). Bool,
    text and NULL have distinct domains. Column aliases do not participate.
    JSON objects are key-order independent; arrays retain order. Native results
    lack PG type OIDs, so JSON arrays and PG arrays share structural sequence
    semantics. Aware timestamps compare exact UTC instants; naive timestamps
    stay distinct. Aware times use offset-adjusted microseconds (no day wrap).
    Unsupported PG types are unavailable, even when both sides look alike.
    """
    try:
        pc, pr = _rows(predicted)
        rc, rr = _rows(reference)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        return {'comparable': False, 'bag_equal': None, 'ordered_equal': None, 'reason': str(error)}
    return {'comparable': True, 'bag_equal': pc == rc and Counter(pr) == Counter(rr),
            'ordered_equal': pc == rc and pr == rr, 'reason': None}
