import copy
import json
from pathlib import Path
import unittest

from scripts.rc_evaluation.dail_sql import contracts


class InjectionTests(unittest.TestCase):
    def test_rc_is_reversible_and_exact_without_mutating_shared_messages(self):
        self.assertTrue(hasattr(contracts, "inject_rc3"), "Task6 RC injection is missing")
        messages = [{"role": "system", "content": "SQL only", "metadata": {"a": [1]}},
                    {"role": "user", "content": "Question\nSELECT "}]
        original = copy.deepcopy(messages)
        rc = {field: f"unique {field}: 中文 \"quoted\"\nnext" for field in contracts.RC_FIELDS}
        definition = "Fixed meanings"
        injected = contracts.inject_rc3(messages, rc, definition)
        self.assertEqual(messages, original)
        self.assertEqual(injected[-1]["content"].split("</result_contract>\n\n", 1)[1], "Question\nSELECT ")
        block = injected[-1]["content"].split("<rc_round3>\n", 1)[1].split("\n</rc_round3>", 1)[0]
        self.assertEqual(json.loads(block), rc)
        restored = copy.deepcopy(injected)
        restored[-1]["content"] = restored[-1]["content"].split("</result_contract>\n\n", 1)[1]
        self.assertEqual(restored, original)
        injected[0]["metadata"]["a"].append(2)
        self.assertEqual(messages, original)

    def test_rejects_incomplete_rc_and_never_injects_round_history(self):
        self.assertTrue(hasattr(contracts, "inject_rc3"), "Task6 RC injection is missing")
        messages = [{"role": "user", "content": "Q\nSELECT "}]
        with self.assertRaises(ValueError):
            contracts.inject_rc3(messages, {"population": "old"}, "definition")
        rc = {field: field for field in contracts.RC_FIELDS}
        rc.update(round1="SECRET1", round2="SECRET2", gold_sql="SECRET3")
        definition = Path(contracts.__file__).with_name("rc_prompt.txt").read_text()
        value = contracts.inject_rc3(messages, rc, definition)[0]["content"]
        self.assertFalse(any(secret in value for secret in ("SECRET1", "SECRET2", "SECRET3")))
        self.assertTrue(value.endswith("Q\nSELECT "))
