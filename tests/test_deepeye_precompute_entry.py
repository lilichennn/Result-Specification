from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))
from scripts import deepeye_bird_interact_precompute as entry


class PrecomputeEntryTests(unittest.TestCase):
    def test_inventory_cannot_succeed_with_missing_question_or_duplicate_id(self):
        tasks, databases = [], {}
        for variant, count, offset, db_count in [('lite', 195, 0, 18), ('full', 410, 18, 22)]:
            for i in range(count):
                item = SimpleNamespace(instance_id=f'item_{i}', database_id=f'db_{offset+i%db_count}')
                tasks.append((variant, item))
                databases[item.database_id] = item
        self.assertTrue(hasattr(entry, 'validate_inventory'), 'Fixed input population validation is missing')
        entry.validate_inventory(tasks, databases)
        with self.assertRaises(ValueError):
            entry.validate_inventory(tasks[:-1], databases)
        duplicated = tasks[:-1] + [tasks[-2]]
        with self.assertRaises(ValueError):
            entry.validate_inventory(duplicated, databases)

    def test_frozen_input_list_is_compared_not_overwritten(self):
        self.assertTrue(hasattr(entry, 'freeze_inventory'), 'Immutable question inventory is missing')
        item = SimpleNamespace(instance_id='a_1', database_id='a', question='first', evidence='', database_schema={'tables': {}})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry.freeze_inventory(root, [('lite', item)], {'a': item})
            previous = (root / 'inputs.json').read_bytes()
            item.question = 'changed'
            with self.assertRaises(ValueError):
                entry.freeze_inventory(root, [('lite', item)], {'a': item})
            self.assertEqual((root / 'inputs.json').read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
