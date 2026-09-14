import importlib
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))

try:
    cache_module = importlib.import_module('scripts.baseline_adapters.deepeye.precompute_cache')
except ModuleNotFoundError as exc:
    if exc.name != 'scripts.baseline_adapters.deepeye.precompute_cache':
        raise
    cache_module = None


class VectorCacheTests(unittest.TestCase):
    def test_bulk_reads_and_writes_preserve_existing_format_and_validate_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.sqlite'
            with self.cache(path) as cache:
                cache.embed(['old'], lambda _: [[1., 2.]])
                cache.put_many(['a', 'b'], [[3., 4.], [5., 6.]])
                found = cache.get_many(['b', 'old', 'missing', 'b'])
                np.testing.assert_equal(found['b'], [5., 6.])
                np.testing.assert_equal(found['old'], [1., 2.])
                self.assertIsNone(found['missing'])
                with self.assertRaises(ValueError):
                    cache.put_many(['c', 'd'], [[7., 8.], [0., 0.]])
                self.assertIsNone(cache.get('c'))
            with self.cache(path) as cache:
                np.testing.assert_equal(cache.read(['a', 'old']), [[3., 4.], [1., 2.]])

    def cache(self, path, namespace=None):
        self.assertIsNotNone(cache_module, 'Persistent vector cache is missing')
        return cache_module.VectorCache(path, namespace or {'model': 'test'})

    def test_duplicates_preserve_order_and_reopening_needs_no_api(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.sqlite'
            calls = []
            def embed(texts):
                calls.append(texts)
                return [[1., 2.] if text == 'a' else [3., 4.] for text in texts]
            with self.cache(path) as cache:
                actual = cache.embed(['b', 'a', 'b'], embed)
                np.testing.assert_equal(actual, [[3., 4.], [1., 2.], [3., 4.]])
            with self.cache(path) as cache:
                actual = cache.embed(['a', 'b'], lambda _: self.fail('Unexpected API call'))
                np.testing.assert_equal(actual, [[1., 2.], [3., 4.]])
            self.assertEqual(calls, [['b', 'a']])

    def test_successful_batches_survive_a_later_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.cache(Path(temp) / 'vectors.sqlite') as cache:
                def flaky(texts):
                    if texts == ['c']:
                        raise RuntimeError('connection unavailable')
                    return [[1., 2.], [3., 4.]]
                with self.assertRaises(RuntimeError):
                    cache.embed(['a', 'b', 'c'], flaky, batch_size=2)
                seen = []
                actual = cache.embed(['a', 'b', 'c'], lambda texts: seen.append(texts) or [[5., 6.]])
                np.testing.assert_equal(actual, [[1., 2.], [3., 4.], [5., 6.]])
                self.assertEqual(seen, [['c']])

    def test_models_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.sqlite'
            with self.cache(path, {'model': 'first'}) as cache:
                cache.embed(['a'], lambda _: [[1., 2.]])
            with self.cache(path, {'model': 'second'}) as cache:
                np.testing.assert_equal(cache.embed(['a'], lambda _: [[3., 4.]]), [[3., 4.]])

    def test_invalid_embeddings_are_not_persisted(self):
        for malformed in ([], [[1., float('nan')]], [[0., 0.]], [[1., 2.], [3., 4.]]):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as temp:
                with self.cache(Path(temp) / 'vectors.sqlite') as cache:
                    with self.assertRaises(ValueError):
                        cache.embed(['a'], lambda _: malformed)
                    np.testing.assert_equal(cache.embed(['a'], lambda _: [[1., 2.]]), [[1., 2.]])

    def test_dimension_change_is_rejected_and_empty_input_skips_api(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.cache(Path(temp) / 'vectors.sqlite') as cache:
                cache.embed(['a'], lambda _: [[1., 2.]])
                with self.assertRaises(ValueError):
                    cache.embed(['b'], lambda _: [[1., 2., 3.]])
                self.assertEqual(cache.embed([], lambda _: self.fail('Unexpected API')).shape, (0, 2))

    def test_corrupted_cached_bytes_are_not_accepted(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.sqlite'
            with self.cache(path) as cache:
                cache.embed(['a'], lambda _: [[1., 2.]])
            with sqlite3.connect(path) as conn:
                conn.execute("UPDATE vectors SET vector = ?", (b'bad',))
            with self.cache(path) as cache:
                with self.assertRaises(ValueError):
                    cache.embed(['a'], lambda _: self.fail('Corruption must not silently trigger API'))


if __name__ == '__main__':
    unittest.main()
