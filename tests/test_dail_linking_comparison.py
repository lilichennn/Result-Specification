"""Offline contracts for projecting native DAIL linking into physical names."""

from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
import unittest


def comparison_api():
    try:
        from scripts.baseline_adapters.dail_sql.linking_comparison import (
            crop_schema,
            normalized_linking,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("DAIL linking comparison core is missing") from exc
    return crop_schema, normalized_linking


def schema_fixture():
    return {
        "db_id": "fixture",
        "table_names_original": ["ignored", "Parent ", "Child"],
        "table_names": ["unused label", "parent label", "child label"],
        "table_descriptions": ["omit", "parent description", "child description"],
        "column_names_original": [
            [-1, "*"], [0, "junk"], [1, "id "], [1, "hidden"],
            [2, "id"], [2, "parent_id"], [2, "note"],
        ],
        "column_names": [
            [-1, "*"], [0, "junk label"], [1, "parent identifier"], [1, "hidden label"],
            [2, "child identifier"], [2, "parent reference"], [2, "note label"],
        ],
        "column_types": ["text", "text", "number", "text", "number", "number", "text"],
        "column_types_original": ["text", "TEXT", "INTEGER", "TEXT", "INT", "BIGINT", "VARCHAR"],
        "column_descriptions": ["", "junk", "parent PK", "private", "child PK", "FK", "note"],
        "column_ref_keys": ["", "", "", "", "", "Parent .id ", "missing.id"],
        "primary_keys": [2, 4],
        "foreign_keys": [[5, 2], [4, 1]],
        "unresolved_foreign_keys": [
            {"column_id": 3, "reference": "missing.hidden"},
            {"column_id": 6, "reference": "missing.id", "reason": "not in public meta"},
        ],
        "extra": {"provenance": ["original"]},
    }


def selected_metadata():
    # Selection order must never replace the native schema order.
    return [
        {"table_name": "Child", "columns": [
            {"original_column_name": "note", "column_description": "do not replace native"},
            {"original_column_name": "parent_id"},
        ]},
        {"table_name": "Parent", "columns": [{"original_column_name": "id"}]},
    ]


class CropSchemaTests(unittest.TestCase):
    def test_nonidentity_projection_preserves_native_labels_details_and_remaps_keys(self):
        crop_schema, _ = comparison_api()
        source = schema_fixture()
        before = deepcopy(source)
        result = crop_schema(source, selected_metadata())
        self.assertEqual(result, {
            "table_local_to_full": [1, 2],
            "column_local_to_full": [0, 2, 5, 6],
            "schema": {
                "db_id": "fixture",
                "table_names_original": ["Parent ", "Child"],
                "table_names": ["parent label", "child label"],
                "table_descriptions": ["parent description", "child description"],
                "column_names_original": [[-1, "*"], [0, "id "], [1, "parent_id"], [1, "note"]],
                "column_names": [[-1, "*"], [0, "parent identifier"], [1, "parent reference"], [1, "note label"]],
                "column_types": ["text", "number", "number", "text"],
                "column_types_original": ["text", "INTEGER", "BIGINT", "VARCHAR"],
                "column_descriptions": ["", "parent PK", "FK", "note"],
                "column_ref_keys": ["", "", "Parent .id ", "missing.id"],
                "primary_keys": [1], "foreign_keys": [[2, 1]],
                "unresolved_foreign_keys": [
                    {"column_id": 3, "reference": "missing.id", "reason": "not in public meta"},
                ],
                "extra": {"provenance": ["original"]},
            },
        })
        result["schema"]["extra"]["provenance"].append("changed")
        self.assertEqual(source, before)

    def test_empty_selection_keeps_only_wildcard_and_table_without_columns_is_valid(self):
        crop_schema, _ = comparison_api()
        empty = crop_schema(schema_fixture(), [])
        self.assertEqual(empty["table_local_to_full"], [])
        self.assertEqual(empty["column_local_to_full"], [0])
        self.assertEqual(empty["schema"]["column_names_original"], [[-1, "*"]])
        self.assertEqual(empty["schema"]["column_types_original"], ["text"])
        self.assertEqual(empty["schema"]["primary_keys"], [])
        self.assertEqual(empty["schema"]["foreign_keys"], [])
        self.assertEqual(empty["schema"]["unresolved_foreign_keys"], [])
        table_only = crop_schema(schema_fixture(), [{"table_name": "Child", "columns": []}])
        self.assertEqual(table_only["table_local_to_full"], [2])
        self.assertEqual(table_only["column_local_to_full"], [0])

    def test_rejects_unknown_or_duplicate_physical_selections(self):
        crop_schema, _ = comparison_api()
        cases = [
            [{"table_name": "missing", "columns": []}],
            [{"table_name": "Child", "columns": [{"original_column_name": "missing"}]}],
            [{"table_name": "Child", "columns": [{"original_column_name": "*"}]}],
            [{"table_name": "Parent", "columns": []}, {"table_name": "Parent ", "columns": []}],
            [{"table_name": "Parent", "columns": [
                {"original_column_name": "id"}, {"original_column_name": "id "},
            ]}],
        ]
        for selected in cases:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                crop_schema(schema_fixture(), selected)

    def test_trim_fallback_requires_unique_name_but_exact_physical_match_wins(self):
        crop_schema, _ = comparison_api()
        schema = schema_fixture()
        schema["table_names_original"][0] = " Parent"
        with self.assertRaisesRegex(ValueError, "[Aa]mbiguous"):
            crop_schema(schema, [{"table_name": "Parent", "columns": []}])
        exact = crop_schema(schema, [{"table_name": "Parent ", "columns": []}])
        self.assertEqual(exact["table_local_to_full"], [1])
        schema = schema_fixture()
        schema["column_names_original"][3][1] = " id"
        with self.assertRaisesRegex(ValueError, "[Aa]mbiguous"):
            crop_schema(schema, [{"table_name": "Parent", "columns": [{"original_column_name": "id"}]}])

    def test_column_name_fallback_selects_original_column_and_full_selection_is_identity(self):
        crop_schema, _ = comparison_api()
        schema = schema_fixture()
        selected = [{"table_name": name, "columns": [
            {"column_name": column} for table, column in schema["column_names_original"] if table == index
        ]} for index, name in enumerate(schema["table_names_original"])]
        result = crop_schema(schema, list(reversed(selected)))
        self.assertEqual(result["schema"], schema)
        self.assertEqual(result["table_local_to_full"], [0, 1, 2])
        self.assertEqual(result["column_local_to_full"], [0, 1, 2, 3, 4, 5, 6])

    def test_rejects_invalid_native_indices_and_misaligned_details(self):
        crop_schema, _ = comparison_api()
        for field, value in (
            ("primary_keys", [7]), ("foreign_keys", [[2, -1]]),
            ("unresolved_foreign_keys", [{"column_id": 99, "reference": "missing.id"}]),
            ("column_types", ["text"]),
            ("column_names_original", [[-1, "*"]]),
        ):
            schema = schema_fixture()
            schema[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                crop_schema(schema, [])


class NormalizeLinkingTests(unittest.TestCase):
    def test_unions_all_native_channels_and_includes_column_parents(self):
        _, normalized_linking = comparison_api()
        linking = {
            "question": ["child", "3", "a", "name"],
            "sc_link": {"q_tab_match": {"0,0": "TPM"}, "q_col_match": {"3,2": "CEM", "3,0": "CEM"}},
            "cv_link": {"num_date_match": {"1,5": "NUMBER"}, "cell_match": {"2,6": "PARTIALMATCH", "3,2": "EXACTMATCH"}},
        }
        self.assertEqual(normalized_linking(linking, schema_fixture()), {
            "tables": ["Child", "Parent", "ignored"],
            "columns": [["Child", "note"], ["Child", "parent_id"], ["Parent", "id"]],
        })

    def test_cropped_local_ids_are_resolved_by_cropped_schema(self):
        crop_schema, normalized_linking = comparison_api()
        cropped = crop_schema(schema_fixture(), selected_metadata())["schema"]
        linking = {"question": ["value"], "sc_link": {"q_col_match": {"0,2": "CPM"}},
                   "cv_link": {"cell_match": {"0,3": "EXACTMATCH"}}}
        self.assertEqual(normalized_linking(linking, cropped), {
            "tables": ["Child"], "columns": [["Child", "note"], ["Child", "parent_id"]],
        })

    def test_missing_channels_and_wildcard_only_yield_empty_sets(self):
        _, normalized_linking = comparison_api()
        self.assertEqual(normalized_linking({}, schema_fixture()), {"tables": [], "columns": []})
        self.assertEqual(normalized_linking({"sc_link": {"q_col_match": {"0,0": "CEM"}}}, schema_fixture()),
                         {"tables": [], "columns": []})

    def test_rejects_malformed_negative_or_out_of_range_match_indices(self):
        _, normalized_linking = comparison_api()
        for channel, match_name, key in (
            ("sc_link", "q_col_match", "0,7"), ("sc_link", "q_tab_match", "0,3"),
            ("cv_link", "num_date_match", "0,-1"), ("cv_link", "cell_match", "0,7"),
            ("sc_link", "q_col_match", "-1,2"), ("sc_link", "q_col_match", "1,2"),
            ("sc_link", "q_col_match", "no,2"), ("sc_link", "q_col_match", "0,2,3"),
            ("sc_link", "q_col_match", 2),
        ):
            linking = {"question": ["word"], channel: {match_name: {key: "MATCH"}}}
            with self.subTest(channel=channel, match_name=match_name, key=key), self.assertRaises(ValueError):
                normalized_linking(linking, schema_fixture())

    def test_rejects_ambiguous_names_after_evaluator_whitespace_normalization(self):
        _, normalized_linking = comparison_api()
        for field, index, value in (
            ("table_names_original", 0, "Parent"),
            ("column_names_original", 3, [1, "id"]),
        ):
            schema = schema_fixture()
            schema[field][index] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "[Aa]mbiguous"):
                normalized_linking({}, schema)


class NativeEquivalenceTests(unittest.TestCase):
    def test_full_crop_retains_native_schema_and_value_linking(self):
        crop_schema, normalized_linking = comparison_api()
        from scripts.baseline_adapters.dail_sql.native import link_question

        class Words:
            def tokenize(self, text):
                return text.lower().split()

            def tokenize_for_copying(self, text):
                return self.tokenize(text), text.split()

        schema = {"table_names_original": ["unused", "person"], "table_names": ["unused", "person"],
                  "column_names_original": [[-1, "*"], [0, "unused"], [1, "name"], [1, "age"]],
                  "column_names": [[-1, "*"], [0, "unused"], [1, "name"], [1, "age"]],
                  "column_types": ["text", "text", "text", "number"], "primary_keys": [], "foreign_keys": []}
        metadata = [{"table_name": "person", "columns": [{"column_name": "age"}, {"column_name": "name"}]},
                    {"table_name": "unused", "columns": [{"column_name": "unused"}]}]
        with tempfile.TemporaryDirectory() as directory, sqlite3.connect(":memory:") as database:
            corpus = Path(directory) / "corpora/stopwords"
            corpus.mkdir(parents=True)
            (corpus / "english").write_text("the\na\n", encoding="utf-8")
            database.executescript("CREATE TABLE person(name TEXT, age INTEGER); INSERT INTO person VALUES ('alice', 30);")
            kwargs = {"compute_cv_link": True, "stopwords_path": Path(directory), "connection": database}
            original = link_question("person alice 30", schema, Words(), **kwargs)
            full = link_question("person alice 30", crop_schema(schema, metadata)["schema"], Words(), **kwargs)
            self.assertEqual(full, original)
            self.assertEqual(normalized_linking(full, schema), {
                "tables": ["person"], "columns": [["person", "age"], ["person", "name"]],
            })
            cropped = crop_schema(schema, metadata[:1])["schema"]
            filtered = link_question("person alice 30", cropped, Words(), **kwargs)
            self.assertEqual(normalized_linking(filtered, cropped), normalized_linking(full, schema))
            self.assertEqual(filtered["cv_link"]["cell_match"], {"1,1": "EXACTMATCH"})
            self.assertEqual(filtered["cv_link"]["num_date_match"], {"2,2": "NUMBER"})


if __name__ == "__main__":
    unittest.main()
