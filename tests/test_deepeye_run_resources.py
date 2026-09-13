"""Resource safety checks use native caches and real executor work, offline."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib
import importlib.util
from pathlib import Path
import sys
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))


class ResourcesTest(unittest.TestCase):
    def module(self):
        name = 'scripts.baseline_adapters.deepeye.run_resources'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Run-scoped resource management is missing')
        return importlib.import_module(name)

    def test_every_inner_pool_drains_before_any_cleanup_even_if_cleanup_raises(self):
        # Cleaning the first runner prematurely could reset services used by another.
        resources = self.module()
        release, entered, cleaned = threading.Event(), threading.Event(), threading.Event()
        second_pool = ThreadPoolExecutor(max_workers=1)
        future = second_pool.submit(lambda: (entered.set(), release.wait(5)))
        self.assertTrue(entered.wait(1))
        order, errors = [], []
        def clean_first():
            cleaned.set()
            self.assertTrue(future.done())
            order.append('first')
            raise ValueError('cleanup fixture')
        first = SimpleNamespace(_clean_up=clean_first)
        second = SimpleNamespace(_inner_thread_pool_executor=second_pool,
            _clean_up=lambda: order.append('second'))
        def cleanup():
            try:
                resources.close_runners([(first, lambda: order.append('trace'), None),
                    (second, None, SimpleNamespace(close=lambda: order.append('client')))])
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=cleanup)
        thread.start()
        try:
            self.assertFalse(cleaned.wait(0.05), 'No global service reset while inner work lives')
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(order, ['first', 'second', 'trace', 'client'])
        self.assertEqual(str(errors[0]), 'cleanup fixture')

    def test_profile_guard_ignores_stale_identity_cache_without_resetting_other_caches(self):
        resources = self.module()
        from app.services.schema_service import get_schema_service, reset_schema_service
        from tests.test_deepeye_run_pipeline import item
        service = get_schema_service()
        schema = item().database_schema
        key = (id(schema), 0, True, True, True, True, True)
        service._schema_profile_cache.set(key, 'WRONG SCHEMA FROM REUSED ID')
        service._value_examples_cache.set(('keep', 't', 'x'), ['kept'])
        original_cache = service._schema_profile_cache
        try:
            with resources.protect_schema_profiles():
                profile = service.build_schema_profile(schema)
                self.assertIn('x', profile)
                self.assertNotIn('WRONG SCHEMA', profile)
                self.assertEqual(len(service._schema_profile_cache), 1)
                self.assertEqual(service._value_examples_cache.get(('keep', 't', 'x')), ['kept'])
            self.assertIs(service._schema_profile_cache, original_cache)
            self.assertEqual(service._value_examples_cache.get(('keep', 't', 'x')), ['kept'])
            self.assertNotIn('WRONG SCHEMA', service.build_schema_profile(schema))
        finally:
            reset_schema_service()

    def test_profile_cache_reuses_equal_content_but_not_mutations_or_render_options(self):
        from app.services import schema_service
        from tests.test_deepeye_run_pipeline import item
        service = schema_service.get_schema_service()
        schema = item().database_schema
        native_render = schema_service.get_database_schema_profile
        try:
            with patch.object(schema_service, 'get_database_schema_profile', wraps=native_render) as render:
                with self.module().protect_schema_profiles():
                    first = service.build_schema_profile(schema)
                    same = service.build_schema_profile(copy.deepcopy(schema))
                    self.assertEqual(first, same)
                    self.assertEqual(render.call_count, 1)
                    schema['tables']['t']['columns']['y'] = copy.deepcopy(schema['tables']['t']['columns']['x'])
                    changed = service.build_schema_profile(schema)
                    self.assertNotEqual(first, changed)
                    service.build_schema_profile(schema, include_description=False)
                    self.assertEqual(render.call_count, 3)
        finally:
            schema_service.reset_schema_service()

    def test_profile_cache_is_bounded_and_preserves_column_order(self):
        from app.services import schema_service
        from app.services._bounded_cache import BoundedCache
        from tests.test_deepeye_run_pipeline import item
        service = schema_service.get_schema_service()
        service._schema_profile_cache = BoundedCache(2)
        original_method = service.build_schema_profile
        schema = item().database_schema
        schema['tables']['t']['columns']['y'] = copy.deepcopy(schema['tables']['t']['columns']['x'])
        reverse = copy.deepcopy(schema)
        reverse['tables']['t']['columns'] = dict(reversed(list(reverse['tables']['t']['columns'].items())))
        try:
            with self.module().protect_schema_profiles():
                first = service.build_schema_profile(schema)
                second = service.build_schema_profile(reverse)
                self.assertNotEqual(first, second)
                service.build_schema_profile(schema, include_description=False)
                self.assertEqual(len(service._schema_profile_cache), 2)
            self.assertEqual(service.build_schema_profile, original_method)
            self.assertNotIn('build_schema_profile', vars(service))
        finally:
            schema_service.reset_schema_service()

    def test_shutdown_interrupt_is_deferred_until_live_native_work_drains(self):
        resources = self.module()
        release, entered, interrupted, cleaned = (threading.Event() for _ in range(4))
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(lambda: (entered.set(), release.wait(5)))
        self.assertTrue(entered.wait(1))
        original_shutdown = pool.shutdown
        interrupt = KeyboardInterrupt('controlled shutdown interruption')
        calls, errors, order = [], [], []
        def shutdown(*args, **kwargs):
            calls.append('shutdown')
            if len(calls) == 1:
                interrupted.set()
                raise interrupt
            return original_shutdown(*args, **kwargs)
        pool.shutdown = shutdown
        def clean():
            cleaned.set()
            self.assertTrue(future.done(), 'Global services cannot reset while native work is live')
            order.append('runner')
        runner = SimpleNamespace(_inner_thread_pool_executor=pool, _clean_up=clean)
        def close():
            try:
                resources.close_runners([(runner, lambda: order.append('trace'),
                    SimpleNamespace(close=lambda: order.append('client')))])
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=close)
        thread.start()
        try:
            self.assertTrue(interrupted.wait(1))
            self.assertFalse(cleaned.wait(0.1), 'Shutdown interruption must not authorize cleanup')
            self.assertTrue(thread.is_alive(), 'Interrupt must wait for the native future')
        finally:
            release.set()
            thread.join(3)
            original_shutdown(wait=True)
        self.assertFalse(thread.is_alive())
        self.assertEqual(order, ['runner', 'trace', 'client'])
        self.assertEqual(errors, [interrupt])

    def test_real_sigint_cannot_make_shutdown_finish_before_tracked_future(self):
        # Python can mark a joined thread stopped after SIGINT while its Future
        # is still running. Exercise an actual signal only in an isolated child.
        script = '''
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import sys
import threading
from types import SimpleNamespace
sys.path.insert(0, 'baselines/DeepEye-SQL')
from scripts.baseline_adapters.deepeye.run_resources import close_runners, instrument_native_pools
release = threading.Event()
pool = ThreadPoolExecutor(max_workers=1)
cleaned = []
runner = SimpleNamespace(_inner_thread_pool_executor=pool, _clean_up=lambda: cleaned.append(future.done()))
undo = instrument_native_pools(runner)
future = pool.submit(lambda: release.wait(5))
signal_timer = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGINT))
release_timer = threading.Timer(0.4, release.set)
signal_timer.start()
release_timer.start()
try:
    try:
        close_runners([(runner, undo, None)])
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('The original SIGINT must propagate')
    assert future.done(), 'Shutdown returned while its tracked native future remained live'
    assert cleaned == [True], 'Native cleanup ran before actual work completion'
finally:
    release.set()
    signal_timer.join()
    release_timer.join()
'''
        result = subprocess.run([sys.executable, '-E', '-B', '-c', script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
