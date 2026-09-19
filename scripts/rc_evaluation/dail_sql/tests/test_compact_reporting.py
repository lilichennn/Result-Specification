import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.dail_sql.config import TaskKey
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.deepeye.run_store import to_jsonable
from scripts.rc_evaluation.dail_sql.tests.test_reporting import four_rounds
from tests.test_dail_sql_records import manifest


class CompactReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        database = self.root / 'query.sqlite'
        with sqlite3.connect(database) as db:
            db.execute('CREATE TABLE t(a)')
        binding = {'database': {'dialect':'sqlite', 'database_id':'query', 'path':str(database)},
                   'reference_sql':'SELECT 1'}
        evaluation = self.root / 'preparation/evaluation'
        evaluation.mkdir(parents=True)
        bindings = {group:{qid:binding for qid in spec['ids']} for group,spec in manifest()['groups'].items()}
        raw = json.dumps(bindings).encode()
        (evaluation/'bindings.json').write_bytes(raw)
        (evaluation/'source.json').write_text(json.dumps({'bindings':'evaluation/bindings.json',
                                                          'sha256':hashlib.sha256(raw).hexdigest()}))
        self.manifest = {**manifest(), 'evaluation_source':str(evaluation/'source.json'),
                         'sql_timeout_seconds':60}
        self.batch = self.root/'batch'
        self.records = DailRecords(self.batch, self.manifest)
        self.index = CurrentIndex(self.batch/'current.sqlite3')
        self.addCleanup(self.records.close)
        self.addCleanup(self.index.close)

    def publish(self, key):
        version, _ = four_rounds(self.records, key)
        self.index.publish(key, version, self.index.claim(key), self.records.is_sealed)
        return version

    def test_compact_export_cross_question_deduplicates_and_avoids_result_jsonl(self):
        from scripts.rc_evaluation.dail_sql.compact_reporting import export_compact_current
        self.publish(TaskKey('batch','spider_dev','0'))
        self.publish(TaskKey('batch','spider_dev','1'))
        output = export_compact_current(self.batch, evaluation_profile='deepeye')
        self.assertFalse((output/'executions.jsonl').exists())
        self.assertFalse((output/'candidates.jsonl').exists())
        with sqlite3.connect(output/'evaluation.sqlite3') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM items').fetchone()[0], 4)
            self.assertEqual(db.execute('SELECT count(*) FROM queries').fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM items WHERE version_id IS NOT NULL").fetchone()[0], 2)
            payload = json.loads(db.execute("SELECT payload_json FROM items WHERE grp='spider_dev' AND question='0'").fetchone()[0])
        self.assertEqual(len(payload['candidates']), 20)
        self.assertTrue(all(c['bag_equal'] is True for c in payload['candidates']))
        versions = json.loads((output/'versions.json').read_text())
        self.assertEqual(versions['evaluation_policy']['groups']['spider_dev']['timeout_seconds'], 600)
        self.assertEqual(versions['format'], 'dail-compact-evaluation-v1')

    def test_interrupted_work_resumes_completed_items_and_cached_queries(self):
        from scripts.rc_evaluation.dail_sql import compact_reporting
        self.publish(TaskKey('batch','spider_dev','0'))
        self.publish(TaskKey('batch','spider_dev','1'))
        # Different SQL per task is unnecessary: fail after one completed item
        # through the explicit question-completion hook.
        completed = 0
        def interrupt_once(_key):
            nonlocal completed
            completed += 1
            if completed == 2:
                raise RuntimeError('injected interruption')
        with patch.object(compact_reporting, '_after_item', side_effect=interrupt_once):
            with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                compact_reporting.export_compact_current(self.batch, evaluation_profile='deepeye')
        work = next((self.batch/'exports/compact').glob('.working-deepeye-*/evaluation.sqlite3'))
        with sqlite3.connect(work) as db:
            completed = db.execute('SELECT count(*) FROM items').fetchone()[0]
            cached = db.execute('SELECT count(*) FROM queries').fetchone()[0]
        self.assertGreaterEqual(completed, 1)
        self.assertEqual(cached, 1)
        with patch.object(compact_reporting, 'execute_sql', side_effect=AssertionError('cache should satisfy all SQL')):
            output = compact_reporting.export_compact_current(self.batch, evaluation_profile='deepeye')
        with sqlite3.connect(output/'evaluation.sqlite3') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM items').fetchone()[0], 4)

    def test_seed_import_compresses_results_and_is_reused(self):
        from scripts.rc_evaluation.dail_sql import compact_reporting
        self.publish(TaskKey('batch','spider_dev','0'))
        seed = self.root/'seed'; seed.mkdir()
        bindings = json.loads((self.root/'preparation/evaluation/bindings.json').read_text())
        policy = compact_reporting.evaluation_policy(
            [TaskKey('batch', group, qid) for group, spec in self.manifest['groups'].items()
             for qid in spec['ids']], bindings, 60, 'deepeye')
        (seed/'progress.json').write_text(json.dumps({'evaluation_policy':policy}))
        row = {'execution_id':'old', 'task_key':{'batch_id':'batch','group':'spider_dev','question_id':'0'},
               'version_id':'ignored', 'database':json.loads((self.root/'preparation/evaluation/bindings.json').read_text())['spider_dev']['0']['database'],
               'database_version':None, 'sql':'SELECT 1',
               'result':{'status':'success','dialect':'sqlite','database_id':'query','sql':'SELECT 1',
                         'timeout_seconds':600,'rows':[(1,)],'columns':['1'],'column_count':1,
                         'column_types':[None],'value_types':[['builtins.int']], 'elapsed_seconds':0.1,'error':None}}
        # Legacy JSONL was already converted through RunStore's typed codec.
        (seed/'executions.jsonl').write_text(json.dumps(to_jsonable(row))+'\n')
        with patch.object(compact_reporting, 'execute_sql', side_effect=AssertionError('seed should satisfy SQL')):
            output = compact_reporting.export_compact_current(self.batch, evaluation_profile='deepeye', seed=seed)
        with sqlite3.connect(output/'evaluation.sqlite3') as db:
            stored, raw_bytes, compressed_bytes = db.execute(
                'SELECT count(*),sum(raw_bytes),sum(length(result)) FROM queries').fetchone()
            payload = json.loads(db.execute(
                "SELECT payload_json FROM items WHERE grp='spider_dev' AND question='0'").fetchone()[0])
        self.assertEqual(stored, 1)
        self.assertLess(compressed_bytes, raw_bytes)
        self.assertTrue(all(candidate['bag_equal'] is True for candidate in payload['candidates']))

    def test_compact_seed_is_normalized_without_rerunning_sql(self):
        from scripts.rc_evaluation.dail_sql import compact_reporting
        self.publish(TaskKey('batch','spider_dev','0'))
        bindings = json.loads((self.root/'preparation/evaluation/bindings.json').read_text())
        keys = [TaskKey('batch', group, qid) for group, spec in self.manifest['groups'].items()
                for qid in spec['ids']]
        policy = compact_reporting.evaluation_policy(keys, bindings, 60, 'deepeye')
        seed = self.root/'compact-seed'; seed.mkdir()
        (seed/'versions.json').write_text(json.dumps({'evaluation_policy':policy}))
        store = compact_reporting.EvaluationStore(seed/'evaluation.sqlite3', {'seed':'legacy'})
        key = TaskKey('batch','spider_dev','0')
        binding = bindings['spider_dev']['0']
        cache_key, scope = compact_reporting._query_identity(key,binding['database'],None,'SELECT 1')
        result = {'status':'success','dialect':'sqlite','database_id':'query','sql':'SELECT 1',
                  'timeout_seconds':600,'rows':[(1,)],'columns':['1'],'column_count':1,
                  'column_types':[None],'value_types':[['builtins.int']], 'elapsed_seconds':0.1,'error':None}
        store._put_query(cache_key,scope,'SELECT 1',to_jsonable(result))
        store.close()
        destination = self.root/'normalized'
        with patch.object(compact_reporting, 'execute_sql', side_effect=AssertionError('compact seed should satisfy SQL')):
            output = compact_reporting.export_compact_current(
                self.batch,evaluation_profile='deepeye',seed=seed,output_root=destination)
        with sqlite3.connect(output/'evaluation.sqlite3') as db:
            payload = json.loads(db.execute(
                "SELECT payload_json FROM items WHERE grp='spider_dev' AND question='0'").fetchone()[0])
        self.assertTrue(all(candidate['bag_equal'] is True for candidate in payload['candidates']))

    def test_explicit_output_root_is_self_contained_and_current(self):
        from scripts.rc_evaluation.dail_sql import compact_reporting
        self.publish(TaskKey('batch','spider_dev','0'))
        destination = self.root/'analysis/evaluation'
        output = compact_reporting.export_compact_current(
            self.batch, evaluation_profile='deepeye', output_root=destination)
        self.assertEqual(output.parent, destination)
        self.assertEqual(compact_reporting.compact_current_export(self.batch, destination), output)
        self.assertTrue((output/'evaluation.sqlite3').is_file())
        self.assertFalse((self.batch/'exports/compact').exists())


if __name__ == '__main__':
    unittest.main()
