"""Exact Spider2 SQLite Meta scope, using real local schema introspection."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/DeepEye-SQL'))

from scripts.baseline_adapters.deepeye import workloads


class SQLiteMetaTests(unittest.TestCase):
    def fixture(self, root, meta, *, autoincrement=True, benchmark='spider2'):
        db = root / ('different-physical.sqlite' if benchmark == 'spider2' else 'DB.sqlite')
        with sqlite3.connect(db) as connection:
            connection.executescript('CREATE TABLE Customers(id INTEGER PRIMARY KEY ' +
                ('AUTOINCREMENT' if autoincrement else '') + ', name TEXT, private_text TEXT, '
                'name_upper TEXT GENERATED ALWAYS AS (upper(name)) VIRTUAL, '
                'name_stored TEXT GENERATED ALWAYS AS (lower(name)) STORED); '
                'CREATE TABLE private_table(x TEXT); '
                'CREATE VIEW private_view AS SELECT id, name FROM Customers;')
        connection.close()
        (root / 'database_description').mkdir()
        directory = root / 'meta' / 'DB'
        directory.mkdir(parents=True)
        bird = benchmark in ('bird', 'bird_interact')
        for table, columns in meta.items():
            header = ('original_column_name,data_format' if bird else 'column_name,column_type')
            (directory / (table + '.csv')).write_text(header + '\n' +
                ''.join(column + ',TEXT\n' for column in columns))
        (root / 'questions.json').write_text(json.dumps([{'index': 'local001' if benchmark == 'spider2' else 7,
            'db_id': 'DB', 'question': 'Find customer', 'evidence': '', 'SQL': 'UNUSED GOLD'}]))
        (root / 'rc.json').write_text('[]')
        return {'benchmark': benchmark, 'split': 'lite' if benchmark == 'spider2' else 'dev',
            'questions': str(root / 'questions.json'), 'meta': str(root / 'meta'),
            'resource_root': str(root), 'rc': str(root / 'rc.json'),
            'database_paths': {'DB': str(db)}}

    def load(self, workload):
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
            try:
                return workloads.load_items(workload, require_prepared=False)
            except ValueError as error:
                self.fail(f'Expected exact SQLite Meta support, got: {error}')

    def test_exact_namespace_preserves_native_names_types_meta_and_visibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = self.fixture(root, {'dB.Db.CUSTOMERS': ['ID', 'name']})
            meta_bytes = next((root / 'meta/DB').glob('*.csv')).read_bytes()
            tasks, bindings, sources = self.load(workload)
            item = tasks[0][1]
            self.assertEqual(set(item.database_schema['tables']), {'Customers'})
            columns = item.database_schema['tables']['Customers']['columns']
            self.assertEqual(set(columns), {'id', 'name'})
            self.assertEqual(columns['id']['column_type'], 'INTEGER')
            self.assertTrue(columns['id']['primary_key'])
            self.assertEqual(item.instance_id, 'local001')
            self.assertEqual(bindings[0]['database_id'], 'DB')
            self.assertEqual(next((root / 'meta/DB').glob('*.csv')).read_bytes(), meta_bytes)
            self.assertEqual(item.gold_sql, '')

    def test_bare_meta_names_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            tasks, _, _ = self.load(self.fixture(Path(temp), {'Customers': ['id']}))
            self.assertEqual(set(tasks[0][1].database_schema['tables']['Customers']['columns']), {'id'})

    def test_wrong_namespaces_and_alias_collisions_are_rejected(self):
        cases = [({'OTHER.DB.Customers': ['id']}, 'namespace'),
                 ({'DB.OTHER.Customers': ['id']}, 'namespace'),
                 ({'DB.Customers': ['id']}, 'namespace'),
                 ({'DB.DB.Customers.more': ['id']}, 'namespace'),
                 ({'DB.DB.Customers': ['id', 'ID']}, 'collision'),
                 ({'DB.DB.Customers': ['id'], 'customers': ['id']}, 'collision')]
        for meta, reason in cases:
            with self.subTest(meta=meta), tempfile.TemporaryDirectory() as temp:
                workload = self.fixture(Path(temp), meta)
                with self.assertRaisesRegex(ValueError, reason):
                    workloads.load_items(workload, require_prepared=False)

    def test_generated_columns_and_explicit_sequence_use_real_readonly_schema(self):
        from app.db_utils.schema import load_database_schema_dict
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = self.fixture(root, {'DB.DB.Customers': ['id', 'name_upper', 'name_stored'],
                                           'DB.DB.SQLITE_SEQUENCE': ['name', 'seq']})
            db = Path(workload['database_paths']['DB'])
            original_schema = load_database_schema_dict(str(db))
            before_schema, before_bytes = deepcopy(original_schema), db.read_bytes()
            connections = []
            native_connect = sqlite3.connect

            def connect(path, *args, **kwargs):
                connections.append((path, kwargs))
                return native_connect(path, *args, **kwargs)

            with patch.object(sqlite3, 'connect', side_effect=connect):
                tasks, _, _ = self.load(workload)
            tables = tasks[0][1].database_schema['tables']
            self.assertEqual(set(tables), {'Customers', 'sqlite_sequence'})
            self.assertEqual(set(tables['sqlite_sequence']['columns']), {'name', 'seq'})
            self.assertEqual(tables['sqlite_sequence']['columns']['seq']['column_type'], '')
            self.assertEqual(tables['Customers']['columns']['name_upper']['column_type'], 'TEXT')
            self.assertEqual(tables['Customers']['columns']['name_stored']['column_type'], 'TEXT')
            self.assertIsNone(tables['Customers']['columns']['name_upper']['value_examples'])
            self.assertEqual(original_schema, before_schema)
            self.assertEqual(db.read_bytes(), before_bytes)
            self.assertTrue(any(options.get('uri') and 'mode=ro' in str(path)
                                for path, options in connections))

    def test_unrequested_internal_tables_and_generated_columns_stay_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            tasks, _, _ = self.load(self.fixture(Path(temp), {'DB.DB.Customers': ['id']}))
            self.assertEqual(set(tasks[0][1].database_schema['tables']), {'Customers'})
            self.assertEqual(set(tasks[0][1].database_schema['tables']['Customers']['columns']), {'id'})

    def test_other_native_excluded_tables_and_nonexistent_columns_are_not_fabricated(self):
        cases = [({'DB.DB.private_view': ['id']}, True, 'table absent'),
                 ({'DB.DB.SQLITE_SEQUENCE': ['name', 'seq']}, False, 'table absent'),
                 ({'DB.DB.Customers': ['not_real']}, True, 'column absent')]
        for meta, auto, error in cases:
            with self.subTest(meta=meta), tempfile.TemporaryDirectory() as temp:
                workload = self.fixture(Path(temp), meta, autoincrement=auto)
                with self.assertRaisesRegex(ValueError, error):
                    workloads.load_items(workload, require_prepared=False)

    def test_existing_regular_column_omitted_by_native_schema_is_not_supplemented(self):
        from app.db_utils.schema import load_database_schema_dict
        with tempfile.TemporaryDirectory() as temp:
            workload = self.fixture(Path(temp), {'DB.DB.Customers': ['private_text']})
            schema = deepcopy(load_database_schema_dict(workload['database_paths']['DB']))
            del schema['tables']['Customers']['columns']['private_text']
            with patch('app.db_utils.schema.load_database_schema_dict', return_value=schema):
                with self.assertRaisesRegex(ValueError, 'column absent'):
                    workloads.load_items(workload, require_prepared=False)

    def test_bigquery_names_are_not_normalized_or_supplemented(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = self.fixture(root, {'PROJECT.DATASET.Customers': ['id']})
            questions = root / 'questions.json'
            questions.write_text(questions.read_text().replace('local001', 'bq001'))
            schema = {'db_id': 'DB', 'db_path': 'DB', 'db_type': 'bigquery', 'tables': {
                'PROJECT.DATASET.Customers': {'table_name': 'Customers', 'columns': {
                    'id': {'column_name': 'id', 'column_type': 'INT64', 'foreign_keys': []}}}}}
            with patch('app.db_utils.cloud_schema.load_cloud_database_schema_dict', return_value=schema), \
                    patch.object(sqlite3, 'connect', side_effect=AssertionError('not SQLite')):
                tasks, _, _ = self.load(workload)
            self.assertEqual(tasks[0][1].database_schema, schema)

    def test_complete_prepared_snapshot_validates_without_database_introspection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = self.fixture(root, {'DB.DB.Customers': ['id', 'name_upper'],
                                           'DB.DB.SQLITE_SEQUENCE': ['name', 'seq']})
            tasks, _, _ = self.load(workload)
            self.save(root, workload, tasks[0][1])
            with patch.object(sqlite3, 'connect', side_effect=AssertionError('prepared snapshot must not be augmented')):
                loaded, _, _ = workloads.load_items(workload)
            self.assertEqual(loaded[0][1].database_schema, tasks[0][1].database_schema)

    def test_incomplete_prepared_snapshots_are_rejected_not_augmented(self):
        for stage, omission in [('base', 'column'), ('base', 'table'), ('retrieved', 'column')]:
            with self.subTest(stage=stage, omission=omission), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                workload = self.fixture(root, {'DB.DB.Customers': ['id', 'name_upper'],
                                               'DB.DB.SQLITE_SEQUENCE': ['name', 'seq']})
                tasks, _, _ = self.load(workload)
                item = tasks[0][1]
                schema = item.database_schema if stage == 'base' else item.database_schema_after_value_retrieval
                if omission == 'column':
                    del schema['tables']['Customers']['columns']['name_upper']
                else:
                    del schema['tables']['sqlite_sequence']
                self.save(root, workload, item)
                with patch.object(sqlite3, 'connect', side_effect=AssertionError('must reject without reopening DB')):
                    with self.assertRaisesRegex(ValueError, 'Prepared schema'):
                        workloads.load_items(workload)

    def test_empty_prepared_snapshot_rejects_identity_without_introspection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = self.fixture(root, {'DB.DB.Customers': ['id', 'name_upper'],
                                           'DB.DB.SQLITE_SEQUENCE': ['name', 'seq']})
            self.save(root, workload, None)
            manifest_path = Path(workload['prepared_dataset'])
            manifest = json.loads(manifest_path.read_text())
            # The native empty writer cannot infer the item type from a first row.
            manifest.update(item_class_module='app.dataset.spider2_dataset',
                            item_class_name='Spider2DataItem')
            manifest_path.write_text(json.dumps(manifest))
            parsed, _ = workloads._prepared_items(workload['prepared_dataset'], 'spider2')
            self.assertEqual(parsed, {})  # Valid structured snapshot, but no selected identity.
            with patch('app.db_utils.schema.load_database_schema_dict',
                       side_effect=AssertionError('empty prepared snapshot must not load raw schema')), \
                    patch.object(sqlite3, 'connect', side_effect=AssertionError('no introspection')):
                with self.assertRaisesRegex(ValueError, 'Prepared dataset lacks original identity.*local001'):
                    workloads.load_items(workload, require_prepared=True)

    def test_other_benchmark_sqlite_scopes_do_not_gain_prefix_aliases(self):
        for benchmark in ('bird', 'spider'):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as temp:
                workload = self.fixture(Path(temp), {'DB.DB.Customers': ['id']}, benchmark=benchmark)
                with self.assertRaisesRegex(ValueError, 'Meta table absent'):
                    workloads.load_items(workload, require_prepared=False)

    def save(self, root, workload, item):
        from app.dataset.spider2_dataset import Spider2LiteDataset
        from app.dataset.utils import save_dataset
        dataset = object.__new__(Spider2LiteDataset)
        dataset._data = [] if item is None else [item]
        dataset._config = SimpleNamespace(type='spider2', split='lite', root_path=str(root))
        output = root / 'prepared.snapshot'
        save_dataset(dataset, str(output))
        workload['prepared_dataset'] = str(output)


if __name__ == '__main__':
    unittest.main()
