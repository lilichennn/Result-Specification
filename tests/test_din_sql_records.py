import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.baseline_adapters.din_sql.inputs import TaskKey, OUTPUT_NODES
from scripts.baseline_adapters.din_sql.records import DinRecords
from din_sql_fixtures import minimal_manifest, terminal


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'batch'
        self.manifest = minimal_manifest()
        self.records = DinRecords(self.root, self.manifest)

    def tearDown(self):
        self.records.close()
        self.tmp.cleanup()

    def complete(self, key, status='succeeded'):
        version = self.records.begin(key)
        for node in OUTPUT_NODES:
            self.records.save_node(version,node,terminal(node,status))
        self.records.seal(version)
        return version

    def test_new_failed_version_replaces_old_success_only_after_seal(self):
        key = TaskKey('bird_dev','0')
        old = self.complete(key)
        other = self.complete(TaskKey('bird_dev','1'))
        new = self.records.begin(key,parent_version=old)
        self.assertEqual(self.records.current(key),old)
        for node in OUTPUT_NODES:
            self.records.save_node(new,node,terminal(node,'failed'))
        self.records.seal(new)
        self.assertEqual(self.records.current(key),new)
        self.assertEqual(self.records.current(TaskKey('bird_dev','1')),other)
        self.records.close()
        self.records = DinRecords(self.root,self.manifest)
        self.assertEqual(self.records.current(key),new)

    def test_second_writer_rejected_and_readonly_allowed(self):
        with self.assertRaises(RuntimeError):
            DinRecords(self.root,self.manifest)
        with DinRecords(self.root,self.manifest,read_only=True) as reader:
            self.assertEqual(reader.current_rows(),[])

    def test_resume_does_not_read_response_bodies_and_seal_does_not_scan(self):
        version = self.records.begin(TaskKey('bird_dev','0'))
        self.records.append(version,'request_result',{'node':'linking','body':{'huge':'x'*10000}})
        self.records.save_node(version,'generation_base',terminal('generation_base'))
        self.records.close()
        self.records = DinRecords(self.root,self.manifest)
        self.assertEqual(self.records.unfinished(TaskKey('bird_dev','0')),version)
        store = self.records.stores['bird_dev']
        with patch.object(store,'attempts',side_effect=AssertionError('whole scan')):
            for node in OUTPUT_NODES[1:]:
                self.records.save_node(version,node,terminal(node))
            self.records.seal(version)
        self.assertEqual(self.records.current(TaskKey('bird_dev','0')),version)

    def test_incomplete_seal_and_duplicate_nodes_rejected(self):
        v = self.records.begin(TaskKey('bird_dev','0'))
        self.records.save_node(v,'generation_base',terminal('generation_base'))
        with self.assertRaises(ValueError):
            self.records.seal(v)
        with self.assertRaises(ValueError):
            self.records.save_node(v,'generation_base',terminal('generation_base','failed'))
