import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.rc_evaluation.dail_sql.linking_package import export_package


def _raw(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw(value) + b'\n')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _info(path):
    return {'bytes': path.stat().st_size, 'sha256': _sha(path)}


class LinkingPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.destination = self.root / 'published'
        self.store = self.root / 'run.sqlite3'
        self.evaluation = self.source / 'evaluation' / 'frozen'
        self.versions = [{'task_key': {'batch_id': 'batch', 'group': 'bird_dev',
                                      'question_id': str(i)}, 'version_id': f'v{i}'}
                         for i in range(2)]
        records = self.source / 'record_export'
        records.mkdir(parents=True)
        (records / 'versions.jsonl').write_bytes(b''.join(_raw(row) + b'\n' for row in self.versions))
        for name, rows in [('modes', 8), ('rounds', 4), ('candidates', 4), ('failed_questions', 0)]:
            (records / f'{name}.jsonl').write_bytes(b'{ "preserved" : true }\n' * rows)
        _write(records / 'verification.json', {
            'record_export_complete': True, 'counts': {'versions': 2, 'modes': 8,
            'rounds': 4, 'candidates': 4, 'failed_questions': 0},
            'files': {p.name: _info(p) for p in records.glob('*.jsonl')}})
        _write(self.evaluation / 'summary.json', {'generation_metric': 'retained'})
        _write(self.evaluation / 'versions.json', {'versions': self.versions, 'source': 'retained'})
        _write(self.source / 'evaluation' / 'latest.json', {'directory': 'frozen'})
        with sqlite3.connect(self.evaluation / 'evaluation.sqlite3') as connection:
            connection.execute('CREATE TABLE items (payload_json TEXT)')
            connection.execute('INSERT INTO items VALUES (?)', ('{"historical_sql":true}',))
        _write(self.source / 'inputs_manifest.json', {'batch_id': 'batch'})
        _write(self.source / 'index.json', {
            'format': 'dail-offline-handoff-v2', 'batch_id': 'batch',
            'record_export_complete': True, 'reference_sql_evaluation_complete': True,
            'record_directory': 'source/record_export', 'raw_batch_directory': 'batch',
            'evaluation_directory': 'source/evaluation/frozen',
            'input_manifest_sha256': _sha(self.source / 'inputs_manifest.json')})
        _write(self.source / 'verification.json', {
            'record_export_complete': True, 'reference_sql_evaluation_complete': True,
            'counts': {'versions': 2},
            'files': {'evaluation/frozen/evaluation.sqlite3': _info(self.evaluation / 'evaluation.sqlite3')}})
        (self.source / '.DS_Store').write_bytes(b'ignored')
        with sqlite3.connect(self.store) as connection:
            connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
            connection.execute('INSERT INTO metadata VALUES (?, ?)', ('fixture', '{"complete":true}'))
            connection.execute('CREATE TABLE items (grp TEXT, question TEXT, payload_json TEXT, '
                               'sha256 TEXT, PRIMARY KEY(grp, question))')
            for i, version in enumerate(self.versions):
                success = i == 0
                prediction = {'status': 'succeeded', 'tables': ['t'], 'columns': [['t', 'c']],
                              'table_recall': 1.0, 'column_recall': 1.0}
                rc3 = prediction if success else {'status': 'dependency_failed', 'tables': [],
                                                  'columns': [], 'table_recall': 0., 'column_recall': 0.}
                usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                filter_usage = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
                detail = {'group': 'bird_dev', 'question_id': str(i), 'version_id': version['version_id'],
                          'annotation_status': 'resolved', 'gold': {'tables': ['t'], 'columns': [['t', 'c']]},
                          'base': prediction, 'rc3': rc3,
                          'filter': {**rc3, 'status': 'succeeded' if success else 'failed',
                                     'table_reduction': 0. if success else None,
                                     'column_reduction': 0. if success else None},
                          'tokens': {'base_linking': usage, 'linking_rc3': usage,
                                     'schema_filter_rc3': filter_usage, 'rc3_combined': filter_usage}}
                payload = {**version, 'linking': {'base': prediction, 'rc3': rc3,
                                                'filter': {'status': 'succeeded' if success else 'failed'},
                                                'source': {'filter_reused': True}}, 'evaluation': detail}
                raw = _raw(payload)
                connection.execute('INSERT INTO items VALUES (?, ?, ?, ?)',
                                   ('bird_dev', str(i), raw.decode(), hashlib.sha256(raw).hexdigest()))

    def test_self_contained_sealed_export_preserves_generation_and_failed_filter(self):
        before = {str(p.relative_to(self.source)): _sha(p) for p in self.source.rglob('*') if p.is_file()}
        result = export_package(self.source, self.destination, self.store)
        self.assertEqual(result, self.destination)
        self.assertEqual(before, {str(p.relative_to(self.source)): _sha(p)
                                 for p in self.source.rglob('*') if p.is_file()})
        self.assertFalse((result / '.DS_Store').exists())
        for name in ('modes', 'rounds', 'candidates', 'failed_questions'):
            self.assertEqual((result / f'record_export/{name}.jsonl').read_bytes(),
                             (self.source / f'record_export/{name}.jsonl').read_bytes())
        index = json.loads((result / 'index.json').read_text())
        self.assertEqual(index['format'], 'dail-offline-handoff-v3')
        self.assertEqual(index['record_directory'], 'record_export')
        self.assertEqual(index['evaluation_directory'], 'evaluation/frozen')
        self.assertEqual(index['raw_batch_directory'], '../batch')
        self.assertEqual(json.loads((result / 'COMPLETE.json').read_text())['index_sha256'], _sha(result / 'index.json'))
        actual_files = {str(p.relative_to(result)) for p in result.rglob('*') if p.is_file()}
        self.assertEqual(actual_files - {'index.json', 'COMPLETE.json'}, {row['path'] for row in index['files']})
        for row in index['files']:
            self.assertEqual(_info(result / row['path']), {k: row[k] for k in ('bytes', 'sha256')})
        linking = [json.loads(line) for line in (result / 'record_export/linking.jsonl').read_text().splitlines()]
        versions = [json.loads(line) for line in (result / 'record_export/versions.jsonl').read_text().splitlines()]
        for row, expected, link in zip(versions, self.versions, linking):
            self.assertEqual({k: row[k] for k in expected}, expected)
            self.assertEqual(row['linking_ref']['task_key'], row['task_key'])
            self.assertEqual(row['linking_ref']['sha256'], hashlib.sha256(_raw(link)).hexdigest())
        self.assertEqual(linking[1]['rc3']['status'], 'dependency_failed')
        summary = json.loads((result / 'evaluation/frozen/summary.json').read_text())
        self.assertEqual(summary['generation_metric'], 'retained')
        self.assertEqual(summary['linking']['overall']['rc3']['columns']['macro']['recall'], .5)
        self.assertEqual(summary['linking']['overall']['tokens']['rc3_combined']['total_tokens']['sum'], 24)
        self.assertEqual(summary['linking']['local_llm_tokens'], 0)
        with sqlite3.connect(result / 'evaluation/frozen/evaluation.sqlite3') as connection:
            self.assertEqual(connection.execute('SELECT payload_json FROM items').fetchone()[0], '{"historical_sql":true}')
        with sqlite3.connect(result / 'raw_records/linking.sqlite3') as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM items').fetchone()[0], 2)

    def test_existing_destination_is_never_replaced(self):
        self.destination.mkdir()
        marker = self.destination / 'user.txt'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            export_package(self.source, self.destination, self.store)
        self.assertEqual(marker.read_text(), 'keep')

    def test_source_checksum_failure_rejects_publication(self):
        (self.source / 'record_export/modes.jsonl').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'hash|checksum'):
            export_package(self.source, self.destination, self.store)
        self.assertFalse(self.destination.exists())

    def test_store_checksum_failure_rejects_publication(self):
        with sqlite3.connect(self.store) as connection:
            connection.execute("UPDATE items SET sha256 = 'corrupted' WHERE question = '1'")
        with self.assertRaisesRegex(ValueError, 'hash|checksum'):
            export_package(self.source, self.destination, self.store)
        self.assertFalse(self.destination.exists())

    def test_missing_question_rejects_publication(self):
        with sqlite3.connect(self.store) as connection:
            connection.execute("DELETE FROM items WHERE question = '1'")
        with self.assertRaisesRegex(ValueError, 'question|coverage'):
            export_package(self.source, self.destination, self.store)
        self.assertFalse(self.destination.exists())

    def test_different_evaluation_order_keeps_original_versions(self):
        _write(self.evaluation / 'versions.json', {'versions': self.versions[::-1]})
        result = export_package(self.source, self.destination, self.store)
        saved = json.loads((result / 'evaluation/frozen/versions.json').read_text())
        self.assertEqual(saved['versions'], self.versions[::-1])

    def test_committed_wal_is_in_standalone_snapshot(self):
        connection = sqlite3.connect(self.store)
        try:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('INSERT INTO metadata VALUES (?, ?)', ('wal_record', '"included"'))
            connection.commit()
            self.assertTrue(Path(str(self.store) + '-wal').exists())
            result = export_package(self.source, self.destination, self.store)
            with sqlite3.connect(result / 'raw_records/linking.sqlite3') as saved:
                self.assertEqual(saved.execute("SELECT value FROM metadata WHERE key='wal_record'").fetchone()[0], '"included"')
            self.assertFalse(list(result.rglob('*-wal')))
            self.assertFalse(list(result.rglob('*-shm')))
        finally:
            connection.close()

    def test_corrupted_schema_rejects_publication(self):
        with sqlite3.connect(self.store) as connection:
            connection.execute('CREATE TABLE schemas(schema_id TEXT PRIMARY KEY, payload_json TEXT, sha256 TEXT)')
            connection.execute('INSERT INTO schemas VALUES (?, ?, ?)', ('bad', '{}', 'bad'))
        with self.assertRaisesRegex(ValueError, 'schema checksum'):
            export_package(self.source, self.destination, self.store)
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
