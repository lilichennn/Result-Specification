import unittest
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.baseline_adapters.dail_sql.retrieval import (
    choose_examples, distance_order, qualified_examples,
)
from scripts.baseline_adapters.dail_sql.native import sql_skeleton, SkeletonError, skeleton_similarity, mask_question


class RetrievalTests(unittest.TestCase):
    def test_second_round_prefers_structure_then_fills_by_distance(self):
        order = ["A", "B", "C", "D"]
        self.assertEqual(choose_examples(order, k=3, qualified_ids=None), ["A", "B", "C"])
        self.assertEqual(choose_examples(order, k=3, qualified_ids={"B", "D"}), ["B", "D", "A"])

    def test_ties_keep_train_order_and_threshold_includes_equality(self):
        self.assertEqual(distance_order(np.array([[1, 0], [-1, 0], [0, 0]]), [0, 0]), [2, 0, 1])
        # Intersection=17, union=20: exactly .85; repeated tokens matter.
        self.assertEqual(qualified_examples({"yes": " ".join(["x"] * 20), "no": "x y"},
                                           " ".join(["x"] * 17)), {"yes"})

    def test_full_pool_filter_can_select_beyond_first_nine(self):
        ids = [str(i) for i in range(15)]
        self.assertEqual(choose_examples(ids, k=2, qualified_ids={"13", "14"}), ["13", "14"])

    def test_bad_vectors_do_not_publish_nondeterministic_order(self):
        with self.assertRaises(ValueError):
            distance_order([[float("nan")]], [0])

    def test_native_sqlite_skeleton_literals_aliases_and_join(self):
        schema = {"table_names_original": ["employee", "dept"],
                  "column_names_original": [[-1, "*"], [0, "name"], [0, "id"], [1, "id"]]}
        self.assertEqual(sql_skeleton("SELECT e.name FROM employee AS e JOIN dept AS d ON e.id = d.id WHERE e.id > 12", schema, "sqlite"),
                         "select _ from _ where _")
        self.assertEqual(sql_skeleton("SELECT name FROM employee ORDER BY name", schema, "sqlite"),
                         "select _ from _ order by _ asc")

    def test_postgres_quoted_identifier_cast_and_invalid_sql(self):
        schema = {"table_names_original": ["employee"], "column_names_original": [[-1, "*"], [0, "name"]]}
        self.assertEqual(sql_skeleton('SELECT "name" FROM "employee" WHERE "name" ILIKE \'A%\'', schema, "postgresql"),
                         "select _ from _ where _ ilike _")
        with self.assertRaises(SkeletonError):
            sql_skeleton("SELECT ( FROM", schema, "postgresql")

    def test_postgres_cte_as_delimiters_survive_native_alias_removal(self):
        schema = {"table_names_original": ["employee"],
                  "column_names_original": [[-1, "*"], [0, "id"]]}
        cases = [
            ("WITH c AS (SELECT e.id AS id FROM employee AS e) SELECT id FROM c",
             "with c as ( select _ from _ ) select _ from c"),
            ("WITH c AS (SELECT id FROM employee), d AS (SELECT id FROM c) SELECT id FROM d",
             "with c as ( select _ from _ ) , d as ( select _ from c ) select _ from d"),
            ("WITH c(id) AS (SELECT id FROM employee) SELECT id FROM c",
             "with c ( _ ) as ( select _ from _ ) select _ from c"),
        ]
        for sql, expected in cases:
            with self.subTest(sql=sql):
                try:
                    actual = sql_skeleton(sql, schema, "postgresql")
                except SkeletonError as exc:
                    self.fail(f"Valid PostgreSQL CTE must reach example retrieval: {exc.__cause__}")
                self.assertEqual(actual, expected)

    def test_postgres_recorded_smoke_ctes_reach_retrieval(self):
        import json
        cases = json.loads((Path(__file__).parent / "fixtures/dail_sql_postgres_cte.json").read_text())
        for case in cases:
            with self.subTest(question=case["question_id"]):
                try:
                    skeleton = sql_skeleton(case["sql"], case["schema"], "postgresql")
                except SkeletonError as exc:
                    self.fail(f"Executable smoke SQL must yield a skeleton: {exc.__cause__}")
                self.assertTrue(skeleton.startswith(case["prefix"]))
                self.assertIn(case["structure"], skeleton)
                self.assertEqual(skeleton.count("("), skeleton.count(")"))

    def test_postgres_cte_keeps_invalid_queries_rejected(self):
        schema = {"table_names_original": ["employee"],
                  "column_names_original": [[-1, "*"], [0, "id"]]}
        for sql in ("WITH c AS (SELECT id FROM employee SELECT id FROM c",
                    "WITH c AS (SELECT 1) SELECT * FROM c; SELECT 2"):
            with self.subTest(sql=sql), self.assertRaises(SkeletonError):
                sql_skeleton(sql, schema, "postgresql")

    def test_actual_native_mask_selectors_match_stable_full_pool_selection(self):
        source = Path(__file__).resolve().parents[1] / "baselines/DAIL-SQL/prompt/ExampleSelectorTemplate.py"
        tree = ast.parse(source.read_text())
        namespace = {"np": np, "jaccard_similarity": skeleton_similarity,
                     "mask_question_with_schema_linking": lambda rows, **kw: [mask_question(row) for row in rows]}
        linked = {"question_for_copying": ["q"], "sc_link": {"q_col_match": {}, "q_tab_match": {}},
                  "cv_link": {"num_date_match": {}, "cell_match": {}}}
        train = [{**linked, "db_id": "same_database", "id": str(i), "pre_skeleton": " ".join(["x"] * 20) if i >= 9 else "y"}
                 for i in range(11)]
        target = {**linked, "db_id": "same_database", "pre_skeleton": " ".join(["x"] * 17)}
        vectors = np.array([[i // 2, 0] for i in range(11)], dtype=np.float32)
        instance = SimpleNamespace(train_json=train, train_embeddings=vectors, threshold=.85,
                                   mask_token="<mask>", value_token="<unk>",
                                   bert_model=SimpleNamespace(encode=lambda _: np.array([[0, 0]], dtype=np.float32)))
        for name, qualified in [("EuclideanDistanceQuestionMaskSelector", None),
                                ("EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector", {"9", "10"})]:
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_examples")
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
            actual = [row["id"] for row in namespace["get_examples"](instance, target, 3)]
            expected = choose_examples([str(i) for i in distance_order(vectors, [0, 0])], k=3, qualified_ids=qualified)
            self.assertEqual(actual, expected)
            self.assertEqual(actual, ["0", "1", "2"] if qualified is None else ["9", "10", "0"])

    def test_postgres_public_qualified_names_and_spaces_are_masked(self):
        schema = {"table_names_original": ["work team"], "column_names_original": [[-1, "*"], [0, "person name"]]}
        self.assertEqual(sql_skeleton('SELECT p."person name" FROM public."work team" AS p', schema, "postgresql"),
                         "select _ from _")


if __name__ == "__main__":
    unittest.main()
