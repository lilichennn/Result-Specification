from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_variant_fixture(root: Path, variant: str) -> Path:
    variant_root = root / f"bird-interact-{variant}"
    database_root = variant_root / "demo"
    database_root.mkdir(parents=True)
    _write_jsonl(
        variant_root / "bird_interact_data.jsonl",
        [
            {
                "instance_id": "demo_1",
                "selected_database": "demo",
                "category": "Query",
                "query": "Return the demo value.",
                "amb_user_query": "Return it.",
                "external_knowledge": [],
                "follow_up": {
                    "category": "Query",
                    "query": "Now return another value.",
                },
            },
            {
                "instance_id": "demo_M_1",
                "selected_database": "demo",
                "category": "Management",
                "query": "Create a demo table.",
                "amb_user_query": "Create it.",
                "external_knowledge": [],
            },
        ],
    )
    _write_jsonl(database_root / "demo_kb.jsonl", [])
    (database_root / "demo_schema.txt").write_text(
        'CREATE TABLE "items" (\n'
        '    "id" integer NOT NULL,\n'
        '    "value" text NULL,\n'
        '    PRIMARY KEY ("id")\n'
        ');\n\n'
        'First 3 rows:\n'
        'id  value\n'
        '1   secret sample\n'
        '...\n',
        encoding="utf-8",
    )
    (database_root / "demo_column_meaning_base.json").write_text(
        json.dumps(
            {
                "demo|items|id": "INTEGER. Unique item identifier.",
                "demo|items|value": "TEXT. Stored demo value.",
            }
        ),
        encoding="utf-8",
    )
    return variant_root


class BirdInteractPublicApiTest(unittest.TestCase):
    def test_module_exposes_preprocess_function(self) -> None:
        """Deleting the adapter module must break its public API contract."""
        spec = importlib.util.find_spec(
            "result_contract.data_preprocess.bird_interact"
        )
        self.assertIsNotNone(spec)
        if spec is None:
            return

        module = importlib.import_module(
            "result_contract.data_preprocess.bird_interact"
        )
        self.assertTrue(callable(module.preprocess_bird_interact))


class BirdInteractPreprocessTest(unittest.TestCase):
    def test_lite_writes_only_top_level_queries_with_source_ids(self) -> None:
        """Including Management or follow-up turns must break the primary-query corpus."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_variant_fixture(root, "lite")
            output_dir = root / "output"

            summary = preprocess_bird_interact(
                interact_root=root,
                variant="lite",
                output_dir=output_dir,
            )

            rows = json.loads(
                (output_dir / "bird_interact_lite.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["instance_count"], 1)
            self.assertEqual(
                rows,
                [
                    {
                        "index": "demo_1",
                        "db_id": "demo",
                        "question": "Return the demo value.",
                        "evidence": "",
                    }
                ],
            )

    def test_evidence_expands_dependencies_and_audits_missing_ids(self) -> None:
        """Evidence must include KB dependencies without hiding broken references."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = _make_variant_fixture(root, "lite")
            data_path = variant_root / "bird_interact_data.jsonl"
            source_rows = [json.loads(line) for line in data_path.read_text().splitlines()]
            source_rows[0]["external_knowledge"] = [2, 99]
            _write_jsonl(data_path, source_rows)
            _write_jsonl(
                variant_root / "demo" / "demo_kb.jsonl",
                [
                    {
                        "id": 1,
                        "knowledge": "Leaf Metric",
                        "description": "This entry is only a dependency.",
                        "definition": "leaf = value",
                        "children_knowledge": -1,
                    },
                    {
                        "id": 2,
                        "knowledge": "Root Rule",
                        "description": "The directly referenced rule.",
                        "definition": "root = 2 * value",
                        "children_knowledge": [1],
                    },
                ],
            )
            output_dir = root / "output"

            summary = preprocess_bird_interact(
                interact_root=root,
                variant="lite",
                output_dir=output_dir,
            )

            rows = json.loads(
                (output_dir / "bird_interact_lite.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (output_dir / "preprocess_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                rows[0]["evidence"],
                "[1] Leaf Metric\n"
                "Description: This entry is only a dependency.\n"
                "Definition: leaf = value\n\n"
                "[2] Root Rule\n"
                "Description: The directly referenced rule.\n"
                "Definition: root = 2 * value",
            )
            self.assertEqual(summary["missing_knowledge_reference_count"], 1)
            self.assertEqual(
                report["missing_knowledge_references"],
                [
                    {
                        "instance_id": "demo_1",
                        "db_id": "demo",
                        "knowledge_id": 99,
                    }
                ],
            )

    def test_full_joins_question_by_instance_id_from_livesqlbench(self) -> None:
        """Full questions must come from the ID join, not row order or amb_user_query."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = _make_variant_fixture(root, "full")
            data_path = variant_root / "bird_interact_data.jsonl"
            interact_rows = [json.loads(line) for line in data_path.read_text().splitlines()]
            for row in interact_rows:
                row.pop("query", None)
            _write_jsonl(data_path, interact_rows)

            livesqlbench_root = root / "livesqlbench"
            livesqlbench_root.mkdir()
            livesqlbench_database_root = livesqlbench_root / "demo"
            livesqlbench_database_root.mkdir()
            for suffix in ("schema.txt", "column_meaning_base.json"):
                shutil.copy2(
                    variant_root / "demo" / f"demo_{suffix}",
                    livesqlbench_database_root / f"demo_{suffix}",
                )
            _write_jsonl(
                livesqlbench_root / "livesqlbench_data.jsonl",
                [
                    {
                        "instance_id": "demo_M_1",
                        "selected_database": "demo",
                        "query": "This joined row must still be filtered out.",
                        "category": "Management",
                    },
                    {
                        "instance_id": "demo_1",
                        "selected_database": "demo",
                        "query": "Question recovered from LiveSQLBench.",
                        "category": "Query",
                    },
                ],
            )
            output_dir = root / "output"

            summary = preprocess_bird_interact(
                interact_root=root,
                variant="full",
                output_dir=output_dir,
                livesqlbench_root=livesqlbench_root,
            )

            rows = json.loads(
                (output_dir / "bird_interact_full.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (output_dir / "preprocess_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["question"], "Question recovered from LiveSQLBench.")
            self.assertEqual(summary["query_join_count"], 1)
            self.assertEqual(report["question_source"], "livesqlbench_data.jsonl")
            self.assertEqual(report["query_join_count"], 1)

    def test_full_requires_livesqlbench_root_without_creating_output(self) -> None:
        """Omitting Full's auxiliary source must fail clearly before writing output."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_variant_fixture(root, "full")
            output_dir = root / "output"

            with self.assertRaisesRegex(ValueError, "livesqlbench_root is required"):
                preprocess_bird_interact(
                    interact_root=root,
                    variant="full",
                    output_dir=output_dir,
                )

            self.assertFalse(output_dir.exists())

    def test_full_rejects_an_incomplete_query_join(self) -> None:
        """Silently falling back to the ambiguous Full prompt would corrupt the corpus."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_variant_fixture(root, "full")
            livesqlbench_root = root / "livesqlbench"
            livesqlbench_root.mkdir()
            _write_jsonl(livesqlbench_root / "livesqlbench_data.jsonl", [])

            with self.assertRaisesRegex(ValueError, "Missing LiveSQLBench query"):
                preprocess_bird_interact(
                    interact_root=root,
                    variant="full",
                    output_dir=root / "output",
                    livesqlbench_root=livesqlbench_root,
                )

    def test_meta_is_derived_from_ddl_without_copying_sample_values(self) -> None:
        """The generated Meta must describe physical columns but remain value-free."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_variant_fixture(root, "lite")
            output_dir = root / "output"

            summary = preprocess_bird_interact(
                interact_root=root,
                variant="lite",
                output_dir=output_dir,
            )

            meta_path = output_dir / "meta" / "demo" / "items.csv"
            with meta_path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                list(rows[0]),
                [
                    "original_column_name",
                    "column_name",
                    "column_description",
                    "data_format",
                    "value_description",
                ],
            )
            self.assertEqual(
                rows,
                [
                    {
                        "original_column_name": "id",
                        "column_name": "",
                        "column_description": "INTEGER. Unique item identifier.",
                        "data_format": "integer",
                        "value_description": "",
                    },
                    {
                        "original_column_name": "value",
                        "column_name": "",
                        "column_description": "TEXT. Stored demo value.",
                        "data_format": "text",
                        "value_description": "",
                    },
                ],
            )
            self.assertNotIn("secret sample", meta_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(summary["database_count"], 1)
            self.assertEqual(summary["meta_file_count"], 1)
            self.assertEqual(summary["column_count"], 2)

    def test_meta_parser_handles_quoted_create_jsonb_and_multiword_types(self) -> None:
        """Full-style quoted DDL and nested meanings must not create phantom columns."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = _make_variant_fixture(root, "lite")
            database_root = variant_root / "demo"
            (database_root / "demo_schema.txt").write_text(
                '"CREATE" TABLE "DataFlow" (\n'
                '    "event_id" bigint NOT NULL,\n'
                '    "payload" jsonb DEFAULT \'{}\'::jsonb,\n'
                '    "recorded_at" timestamp without time zone DEFAULT now(),\n'
                '    "PRIMARY" KEY ("event_id")\n'
                ');\n\n'
                '"First" 3 rows:\n'
                'event_id  payload\n'
                '1         {"payload_field": "secret sample"}\n'
                '...\n',
                encoding="utf-8",
            )
            (database_root / "demo_column_meaning_base.json").write_text(
                json.dumps(
                    {
                        "different_prefix|DataFlow|event_id": "Event identifier.",
                        "different_prefix|DataFlow|payload": {
                            "column_meaning": "Structured event payload.",
                            "fields_meaning": {
                                "payload_field": "A nested semantic field."
                            },
                        },
                        "different_prefix|DataFlow|recorded_at": "Event timestamp.",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "output"

            summary = preprocess_bird_interact(
                interact_root=root,
                variant="lite",
                output_dir=output_dir,
            )

            meta_path = output_dir / "meta" / "demo" / "DataFlow.csv"
            with meta_path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["original_column_name"] for row in rows], [
                "event_id",
                "payload",
                "recorded_at",
            ])
            self.assertEqual(
                [row["data_format"] for row in rows],
                ["bigint", "jsonb", "timestamp without time zone"],
            )
            self.assertIn("Structured event payload.", rows[1]["column_description"])
            self.assertIn("payload_field", rows[1]["column_description"])
            self.assertNotIn("secret sample", meta_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(summary["meta_file_count"], 1)
            self.assertEqual(summary["column_count"], 3)

    def test_create_table_text_in_sample_rows_cannot_create_meta(self) -> None:
        """Only DDL before each sample block may define physical tables."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = _make_variant_fixture(root, "lite")
            schema_path = variant_root / "demo" / "demo_schema.txt"
            schema_text = schema_path.read_text(encoding="utf-8")
            schema_path.write_text(
                schema_text.replace(
                    "1   secret sample\n...",
                    'CREATE TABLE "phantom" ("leaked_value" text);\n...',
                ),
                encoding="utf-8",
            )

            summary = preprocess_bird_interact(
                interact_root=root,
                variant="lite",
                output_dir=root / "output",
            )

            self.assertEqual(summary["meta_file_count"], 1)
            self.assertFalse(
                (root / "output" / "meta" / "demo" / "phantom.csv").exists()
            )

    def test_rerun_replaces_the_owned_output_tree(self) -> None:
        """A changed schema must not leave obsolete Meta CSVs behind."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = _make_variant_fixture(root, "lite")
            database_root = variant_root / "demo"
            output_dir = root / "output"
            preprocess_bird_interact(root, "lite", output_dir)

            (database_root / "demo_schema.txt").write_text(
                'CREATE TABLE "replacement" (\n'
                '    "replacement_id" integer NOT NULL\n'
                ');\n\n'
                'First 3 rows:\n'
                'replacement_id\n'
                '1\n'
                '...\n',
                encoding="utf-8",
            )
            (database_root / "demo_column_meaning_base.json").write_text(
                json.dumps(
                    {"demo|replacement|replacement_id": "Replacement identifier."}
                ),
                encoding="utf-8",
            )

            preprocess_bird_interact(root, "lite", output_dir)

            self.assertFalse((output_dir / "meta" / "demo" / "items.csv").exists())
            self.assertTrue(
                (output_dir / "meta" / "demo" / "replacement.csv").is_file()
            )

    def test_failed_rerun_preserves_the_previous_output_tree(self) -> None:
        """A generation failure must not mix partially new Meta with old JSON."""
        import result_contract.data_preprocess.bird_interact as adapter

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_variant_fixture(root, "lite")
            output_dir = root / "output"
            adapter.preprocess_bird_interact(root, "lite", output_dir)
            previous_meta = (output_dir / "meta" / "demo" / "items.csv").read_bytes()
            original_writer = adapter._write_meta_csv

            def write_then_fail(path: Path, rows: list[dict[str, str]]) -> None:
                changed_rows = [dict(row) for row in rows]
                changed_rows[0]["column_description"] = "PARTIAL NEW CONTENT"
                original_writer(path, changed_rows)
                raise RuntimeError("injected failure")

            with mock.patch.object(adapter, "_write_meta_csv", write_then_fail):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    adapter.preprocess_bird_interact(root, "lite", output_dir)

            self.assertEqual(
                (output_dir / "meta" / "demo" / "items.csv").read_bytes(),
                previous_meta,
            )

    def test_database_id_must_be_one_safe_path_component(self) -> None:
        """Dataset-controlled database IDs must not escape source or output roots."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = _make_variant_fixture(root, "lite")
            data_path = variant_root / "bird_interact_data.jsonl"
            rows = [json.loads(line) for line in data_path.read_text().splitlines()]
            rows[0]["selected_database"] = "../escape"
            _write_jsonl(data_path, rows)

            with self.assertRaisesRegex(ValueError, "safe path component"):
                preprocess_bird_interact(root, "lite", root / "output")


class BirdInteractRealDataAcceptanceTest(unittest.TestCase):
    def test_current_sources_match_the_frozen_preprocess_profile(self) -> None:
        """Publishing truncated or partially parsed benchmark artifacts must fail CI."""
        from result_contract.data_preprocess.bird_interact import (
            preprocess_bird_interact,
        )

        project_root = Path(__file__).resolve().parents[2]
        interact_root = project_root / "BIRD-Interact" / "BIRD-Interact-ADK"
        livesqlbench_root = project_root / "livesqlbench-base-full-v1"
        if not (
            (interact_root / "bird-interact-lite" / "bird_interact_data.jsonl").is_file()
            and (
                interact_root / "bird-interact-full" / "bird_interact_data.jsonl"
            ).is_file()
            and (livesqlbench_root / "livesqlbench_data.jsonl").is_file()
        ):
            self.skipTest("BIRD-Interact or LiveSQLBench source checkout is unavailable")

        expected = {
            "lite": {
                "instance_count": 195,
                "query_join_count": 0,
                "missing_knowledge_reference_count": 0,
                "database_count": 18,
                "meta_file_count": 175,
                "column_count": 2286,
            },
            "full": {
                "instance_count": 410,
                "query_join_count": 410,
                "missing_knowledge_reference_count": 3,
                "database_count": 22,
                "meta_file_count": 244,
                "column_count": 1942,
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            for variant in ("lite", "full"):
                summary = preprocess_bird_interact(
                    interact_root=interact_root,
                    variant=variant,
                    output_dir=output_root / variant,
                    livesqlbench_root=(
                        livesqlbench_root if variant == "full" else None
                    ),
                )
                for key, value in expected[variant].items():
                    self.assertEqual(summary[key], value, f"{variant}.{key}")

            full_rows = json.loads(
                (output_root / "full" / "bird_interact_full.json").read_text()
            )
            livesqlbench_rows = {
                row["instance_id"]: row
                for row in map(
                    json.loads,
                    (livesqlbench_root / "livesqlbench_data.jsonl")
                    .read_text()
                    .splitlines(),
                )
            }
            self.assertTrue(
                all(
                    row["question"] == livesqlbench_rows[row["index"]]["query"]
                    for row in full_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
