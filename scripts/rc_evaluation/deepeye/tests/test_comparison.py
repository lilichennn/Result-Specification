import unittest
import datetime as dt
from decimal import Decimal
from uuid import UUID

from scripts.rc_evaluation.deepeye.comparison import compare_results


def result(rows, cols=None, kind='success'):
    return {'result_type': kind, 'result_rows': rows,
            'result_cols': ['x'] if cols is None else cols}


class ComparisonTests(unittest.TestCase):
    def test_duplicates_and_order_are_preserved(self):
        self.assertFalse(compare_results(result([[1], [1]]), result([[1]]))['bag_equal'])
        got = compare_results(result([[2], [1], [1]]), result([[1], [2], [1]]))
        self.assertTrue(got['bag_equal'])
        self.assertFalse(got['ordered_equal'])

    def test_null_empty_column_positions_and_aliases(self):
        self.assertTrue(compare_results(result([[None]], kind='all_null_result'), result([[None]]))['bag_equal'])
        self.assertFalse(compare_results(result([[None]]), result([[0]]))['bag_equal'])
        self.assertTrue(compare_results(result([], ['a'], 'empty_result'), result([], ['b'], 'empty_result'))['ordered_equal'])
        self.assertFalse(compare_results(result([], ['a']), result([], ['a', 'b']))['bag_equal'])
        self.assertFalse(compare_results(result([[1, 2]], ['a', 'b']), result([[2, 1]], ['b', 'a']))['bag_equal'])

    def test_exact_numeric_equivalence_without_bool_or_text_coercion(self):
        for value in [1, 1.0, Decimal('1.000')]:
            self.assertTrue(compare_results(result([[value]]), result([[1]]))['bag_equal'])
        for value in [True, '1', Decimal('1.00000000000000000001')]:
            self.assertFalse(compare_results(result([[value]]), result([[1]]))['bag_equal'])
        self.assertFalse(compare_results(result([[0.1]]), result([[Decimal('0.1')]]))['bag_equal'])
        self.assertTrue(compare_results(result([[0.5]]), result([[Decimal('0.5')]]))['bag_equal'])

    def test_unknown_types_nonfinite_and_bad_shape_are_unavailable(self):
        for value in [float('nan'), float('inf'), Decimal('NaN'), {1: 'non-json-key'}, {1}, object()]:
            got = compare_results(result([[value]]), result([[value]]))
            self.assertFalse(got['comparable'])
            self.assertIsNone(got['bag_equal'])
        for predicted in [result([[1, 2]]), result(None), result([], kind='timeout'), {}]:
            self.assertFalse(compare_results(predicted, result([]))['comparable'])

    def test_json_objects_and_array_cells_keep_structure_and_order(self):
        left = {'a': [1, None, {'b': True}], 'label': 'x'}
        right = {'label': 'x', 'a': [Decimal('1'), None, {'b': True}]}
        self.assertTrue(compare_results(result([[left]]), result([[right]]))['bag_equal'])
        self.assertFalse(compare_results(result([[[1, 2]]]), result([[[2, 1]]]))['bag_equal'])
        self.assertTrue(compare_results(result([[[1, 2]]]), result([[(1, 2)]]))['bag_equal'])
        self.assertFalse(compare_results(result([[{'a': None}]]), result([[{}]]))['bag_equal'])
        self.assertFalse(compare_results(result([[{'a': True}]]), result([[{'a': 1}]]))['bag_equal'])
        self.assertFalse(compare_results(result([[{'a': float('nan')}]]), result([[{}]]))['comparable'])

    def test_temporal_binary_uuid_cells_are_typed_and_exact(self):
        utc = dt.datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=dt.timezone.utc)
        same = dt.datetime(2026, 1, 2, 11, 4, 5, 6, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        self.assertTrue(compare_results(result([[utc]]), result([[same]]))['bag_equal'])
        self.assertFalse(compare_results(result([[utc]]), result([[utc.replace(tzinfo=None)]]))['bag_equal'])
        self.assertFalse(compare_results(result([[utc.date()]]), result([[utc.date().isoformat()]]))['bag_equal'])
        for value in [dt.date(2026, 1, 1), dt.time(3, 4, 5, 6), b'\x00\xff', UUID(int=42)]:
            self.assertTrue(compare_results(result([[value]]), result([[value]]))['bag_equal'])
            self.assertFalse(compare_results(result([[value]]), result([[str(value)]]))['bag_equal'])
        self.assertTrue(compare_results(result([[dt.time(11, tzinfo=dt.timezone(dt.timedelta(hours=8)))]]),
                                        result([[dt.time(3, tzinfo=dt.timezone.utc)]]))['bag_equal'])


if __name__ == '__main__':
    unittest.main()
