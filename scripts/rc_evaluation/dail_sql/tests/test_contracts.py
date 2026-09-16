import unittest

from scripts.rc_evaluation.dail_sql.contracts import select_rc3


class Rc3ContractsTests(unittest.TestCase):
    def test_successful_rc3_is_selected_with_source_identity(self):
        record = {"index": 1, "db_id": "db", "question": "Q", "evidence": "",
                  "round3_status": "succeeded", "round3_error": None,
                  "rc_round3": {"population": "all", "row_grain": "one", "column_role": "count",
                                "derivation": "count", "filter_policy": "none", "meta_review": "none"}}
        self.assertEqual(select_rc3(record)["rc_round3"]["population"], "all")
        self.assertEqual(select_rc3(record)["question_id"], "1")

    def test_rc2_only_or_failed_rc3_never_falls_back(self):
        base = {"index": 1, "db_id": "db", "question": "Q", "rc_round2": {"population": "old"}}
        for record in (base, {**base, "round3_status": "failed", "rc_round3": {"population": "new"}}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                select_rc3(record)

    def test_duplicate_or_incomplete_rc3_binding_is_rejected(self):
        base = {"index": 1, "db_id": "db", "question": "Q", "round3_status": "succeeded",
                "rc_round3": {"population": "all", "row_grain": "one", "column_role": "count",
                              "derivation": "count", "filter_policy": "none", "meta_review": "none"}}
        for record in ({**base, "rc_round3": [base["rc_round3"], base["rc_round3"]]},
                       {**base, "rc_round3": {"population": "all"}},
                       {**base, "round3_error": "bad"}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                select_rc3(record)


if __name__ == "__main__":
    unittest.main()
