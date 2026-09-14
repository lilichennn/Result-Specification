from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_rc_gold.py"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _instance(index: str, db_id: str, question: str) -> dict[str, str]:
    return {
        "index": index,
        "db_id": db_id,
        "question": question,
        "evidence": "Fixture evidence.",
    }


def _bird_row(
    index: str,
    db_id: str,
    question: str | None,
    sol_sql: object,
    *,
    category: str = "Query",
    preprocess_sql: object = None,
    clean_up_sqls: object = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": index,
        "selected_database": db_id,
        "category": category,
        "sol_sql": sol_sql,
        "preprocess_sql": [] if preprocess_sql is None else preprocess_sql,
        "clean_up_sqls": [] if clean_up_sqls is None else clean_up_sqls,
    }
    if question is not None:
        row["query"] = question
    return row


class PrepareRcGoldTest(unittest.TestCase):
    def test_lite_matches_reference_sql_and_reports_explicit_exclusion(self) -> None:
        """Changing ID joins, SQL preservation, or exclusion handling must fail."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "source"
            variant_root = dataset_root / "bird-interact-lite"
            _write_jsonl(
                variant_root / "bird_interact_data.jsonl",
                [
                    _bird_row("keep_1", "keep", "Keep question?", "SELECT 1;"),
                    _bird_row(
                        "skip_2", "skip", "Skip question?", ["SELECT\n  2;"]
                    ),
                    _bird_row(
                        "management_3",
                        "keep",
                        "Do not include me",
                        ["UPDATE x SET y = 1"],
                        category="Management",
                    ),
                ],
            )
            input_path = root / "input.json"
            _write_json(
                input_path,
                [
                    _instance("keep_1", "keep", "Keep question?"),
                    _instance("skip_2", "skip", "Skip question?"),
                ],
            )
            output_dir = root / "out"

            result = prepare_rc_gold(
                "bird_interact_lite",
                dataset_root=dataset_root,
                input_path=input_path,
                output_dir=output_dir,
                exclusions=["skip_2=Reference SQL is intentionally unusable"],
                cwd=root,
            )

            gold = json.loads((output_dir / "gold_sql.json").read_text())
            report = json.loads(
                (output_dir / "gold_sql_preparation.json").read_text()
            )
            self.assertEqual(
                gold,
                [
                    {
                        "index": "keep_1",
                        "db_id": "keep",
                        "question": "Keep question?",
                        "gold_sql": "SELECT 1;",
                    }
                ],
            )
            self.assertEqual(result["counts"], report["counts"])
            self.assertEqual(result["gold_sql"], report["gold_sql"])
            self.assertEqual(
                report["gold_sql"],
                {
                    "path": "gold_sql.json",
                    "sha256": hashlib.sha256(
                        (output_dir / "gold_sql.json").read_bytes()
                    ).hexdigest(),
                },
            )
            self.assertEqual(
                report["counts"],
                {"total": 2, "ready": 1, "missing_gold_sql": 0, "excluded": 1},
            )
            self.assertEqual(
                [record["status"] for record in report["records"]],
                ["ready", "excluded"],
            )
            self.assertEqual(
                report["records"][1]["reason"],
                "Reference SQL is intentionally unusable",
            )
            self.assertEqual(
                set(gold[0]), {"index", "db_id", "question", "gold_sql"}
            )
            self.assertEqual(report["input"]["sha256"], hashlib.sha256(input_path.read_bytes()).hexdigest())
            serialized_report = json.dumps(report)
            self.assertNotIn(str(root), serialized_report)
            self.assertEqual(report["source_roots"]["dataset_root"], "source")

    def test_full_uses_livesqlbench_question_but_bird_merged_sql(self) -> None:
        """Using LiveSQLBench sol_sql or the ambiguous merged prompt must fail."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = root / "bird-interact-full"
            _write_jsonl(
                variant_root / "bird_interact_data.jsonl",
                [_bird_row("full_1", "demo", None, ["SELECT 'merged';"])],
            )
            livesqlbench_root = root / "live"
            _write_jsonl(
                livesqlbench_root / "livesqlbench_data.jsonl",
                [
                    {
                        "instance_id": "full_1",
                        "selected_database": "demo",
                        "category": "Query",
                        "query": "Joined full question?",
                        "sol_sql": ["SELECT 'wrong source';"],
                    }
                ],
            )
            input_path = root / "input.json"
            _write_json(input_path, [_instance("full_1", "demo", "Joined full question?")])

            prepare_rc_gold(
                "bird_interact_full",
                dataset_root=variant_root,
                livesqlbench_root=livesqlbench_root,
                input_path=input_path,
                output_dir=root / "out",
                cwd=root,
            )

            gold = json.loads((root / "out" / "gold_sql.json").read_text())
            report = json.loads(
                (root / "out" / "gold_sql_preparation.json").read_text()
            )
            self.assertEqual(gold[0]["gold_sql"], "SELECT 'merged';")
            self.assertEqual(len(report["sources"]), 2)
            self.assertEqual(
                {source["root"] for source in report["sources"]},
                {"dataset_root", "livesqlbench_root"},
            )

    def test_spider_matches_selected_source_rows_and_records_missing_sql(self) -> None:
        """Inventing SQL or using a fallback answer artifact must fail."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spider_root = root / "Spider2.0" / "spider2-lite"
            _write_jsonl(
                spider_root / "spider2-lite.jsonl",
                [
                    {"instance_id": "ga001", "db": "ga-demo", "question": "One?"},
                    {"instance_id": "sf002", "db": "sf", "question": "Two?"},
                    {"instance_id": "unused", "db": "other", "question": "Other?"},
                ],
            )
            sql_path = spider_root / "evaluation_suite" / "gold" / "sql" / "ga001.sql"
            sql_path.parent.mkdir(parents=True)
            sql_path.write_text("-- keep this formatting\nSELECT 1;\n", encoding="utf-8")
            # This tempting fallback must not make sf002 ready.
            (sql_path.parent.parent / "sf_sf002.csv").write_text("answer\n", encoding="utf-8")
            input_path = root / "input.json"
            _write_json(
                input_path,
                [
                    _instance("ga001", "GA_DEMO", "One?"),
                    _instance("sf002", "SF", "Two?"),
                ],
            )

            prepare_rc_gold(
                "spider2_lite",
                dataset_root=root / "Spider2.0",
                input_path=input_path,
                output_dir=root / "out",
                cwd=root,
            )

            gold = json.loads((root / "out" / "gold_sql.json").read_text())
            report = json.loads(
                (root / "out" / "gold_sql_preparation.json").read_text()
            )
            self.assertEqual(gold[0]["gold_sql"], "-- keep this formatting\nSELECT 1;\n")
            self.assertEqual([row["index"] for row in gold], ["ga001"])
            self.assertEqual(
                report["counts"],
                {"total": 2, "ready": 1, "missing_gold_sql": 1, "excluded": 0},
            )
            self.assertEqual(report["records"][1]["status"], "missing_gold_sql")
            self.assertEqual(
                report["records"][1]["reference"]["path"],
                "spider2-lite/evaluation_suite/gold/sql/sf002.sql",
            )

    def test_duplicate_unknown_and_malformed_exclusions_are_rejected(self) -> None:
        """An exclusion typo or conflicting rationale must never silently drop data."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = root / "bird-interact-lite"
            _write_jsonl(
                variant_root / "bird_interact_data.jsonl",
                [_bird_row("one", "db", "Question?", ["SELECT 1"])],
            )
            input_path = root / "input.json"
            _write_json(input_path, [_instance("one", "db", "Question?")])
            base = dict(
                dataset_root=variant_root,
                input_path=input_path,
                output_dir=root / "out",
                cwd=root,
            )
            cases = [
                (["missing=reason"], "Unknown exclusion id"),
                (["one=first", "one=second"], "Duplicate exclusion id"),
                (["one"], "INDEX=REASON"),
                (["one=   "], "non-empty reason"),
            ]
            for exclusions, message in cases:
                with self.subTest(exclusions=exclusions):
                    with self.assertRaisesRegex(ValueError, message):
                        prepare_rc_gold(
                            "bird_interact_lite", exclusions=exclusions, **base
                        )
            self.assertFalse((root / "out").exists())

    def test_strict_mismatch_preserves_existing_outputs(self) -> None:
        """Question, database, duplicate, or missing source joins must fail before publish."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        mutations = [
            ("question", [_instance("one", "db", "Wrong?")], "Question mismatch"),
            ("database", [_instance("one", "other", "Question?")], "Database mismatch"),
            (
                "duplicate input",
                [_instance("one", "db", "Question?"), _instance("one", "db", "Question?")],
                "Duplicate preprocessed index",
            ),
            ("missing source", [_instance("absent", "db", "Question?")], "No BIRD-Interact Query"),
        ]
        for label, instances, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                variant_root = root / "bird-interact-lite"
                _write_jsonl(
                    variant_root / "bird_interact_data.jsonl",
                    [_bird_row("one", "db", "Question?", ["SELECT 1"])],
                )
                input_path = root / "input.json"
                _write_json(input_path, instances)
                output_dir = root / "out"
                output_dir.mkdir()
                old_gold = b"old gold\n"
                old_report = b"old report\n"
                (output_dir / "gold_sql.json").write_bytes(old_gold)
                (output_dir / "gold_sql_preparation.json").write_bytes(old_report)

                with self.assertRaisesRegex(ValueError, message):
                    prepare_rc_gold(
                        "bird_interact_lite",
                        dataset_root=variant_root,
                        input_path=input_path,
                        output_dir=output_dir,
                        cwd=root,
                    )

                self.assertEqual((output_dir / "gold_sql.json").read_bytes(), old_gold)
                self.assertEqual(
                    (output_dir / "gold_sql_preparation.json").read_bytes(), old_report
                )

    def test_invalid_sql_forms_and_required_setup_are_fatal(self) -> None:
        """Ambiguous answers, malformed SQL, and stateful tasks must not enter gold."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        cases = [
            (["SELECT 1", "SELECT 2"], [], [], "ambiguous sol_sql"),
            (42, [], [], "sol_sql"),
            (["-- comment only\n/* still comment */"], [], [], "non-comment SQL"),
            (["SELECT 1"], ["CREATE TABLE x(a int)"], [], "preprocess_sql"),
            (["SELECT 1"], [], ["DROP TABLE x"], "clean_up_sqls"),
        ]
        for sol_sql, setup, cleanup, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                variant_root = root / "bird-interact-lite"
                _write_jsonl(
                    variant_root / "bird_interact_data.jsonl",
                    [
                        _bird_row(
                            "one",
                            "db",
                            "Question?",
                            sol_sql,
                            preprocess_sql=setup,
                            clean_up_sqls=cleanup,
                        )
                    ],
                )
                input_path = root / "input.json"
                _write_json(input_path, [_instance("one", "db", "Question?")])
                with self.assertRaisesRegex(ValueError, message):
                    prepare_rc_gold(
                        "bird_interact_lite",
                        dataset_root=variant_root,
                        input_path=input_path,
                        output_dir=root / "out",
                        cwd=root,
                    )
                self.assertFalse((root / "out").exists())

    def test_spider_rejects_path_traversal_and_comment_only_gold(self) -> None:
        """An instance ID must never escape the exact gold/sql directory."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        cases = [
            ("../escape", "SELECT 1", "safe filename"),
            ("safe", "-- no query here\n/* nothing */", "non-comment SQL"),
        ]
        for index, sql, message in cases:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                spider_root = root / "spider2-lite"
                _write_jsonl(
                    spider_root / "spider2-lite.jsonl",
                    [{"instance_id": index, "db": "db", "question": "Question?"}],
                )
                sql_path = spider_root / "evaluation_suite" / "gold" / "sql" / "safe.sql"
                sql_path.parent.mkdir(parents=True)
                sql_path.write_text(sql, encoding="utf-8")
                input_path = root / "input.json"
                _write_json(input_path, [_instance(index, "DB", "Question?")])
                with self.assertRaisesRegex(ValueError, message):
                    prepare_rc_gold(
                        "spider2_lite",
                        dataset_root=spider_root,
                        input_path=input_path,
                        output_dir=root / "out",
                        cwd=root,
                    )

    def test_empty_evidence_is_valid_preprocessed_input(self) -> None:
        """Requiring evidence content would reject valid existing RC inputs."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = root / "bird-interact-lite"
            _write_jsonl(
                variant_root / "bird_interact_data.jsonl",
                [_bird_row("one", "db", "Q", ["SELECT 1"])],
            )
            input_path = root / "input.json"
            _write_json(
                input_path,
                [{**_instance("one", "db", "Q"), "evidence": ""}],
            )

            report = prepare_rc_gold(
                "bird_interact_lite",
                dataset_root=variant_root,
                input_path=input_path,
                output_dir=root / "out",
                cwd=root,
            )

            self.assertEqual(report["counts"]["ready"], 1)
            self.assertEqual(
                json.loads((root / "out" / "gold_sql.json").read_text()),
                [
                    {
                        "index": "one",
                        "db_id": "db",
                        "question": "Q",
                        "gold_sql": "SELECT 1",
                    }
                ],
            )

    def test_input_validation_and_output_collision(self) -> None:
        """Malformed model inputs and overwriting the preprocessed corpus must fail."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        bad_instances = [
            [{"index": 1, "db_id": "db", "question": "Q", "evidence": "E"}],
            [_instance("one", "", "Q")],
            [_instance("one", "db", "")],
            [{"index": "one", "db_id": "db", "question": "Q"}],
            [{**_instance("one", "db", "Q"), "evidence": None}],
            [{**_instance("one", "db", "Q"), "evidence": 42}],
        ]
        for instances in bad_instances:
            with self.subTest(instances=instances), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                variant_root = root / "bird-interact-lite"
                _write_jsonl(
                    variant_root / "bird_interact_data.jsonl",
                    [_bird_row("one", "db", "Q", ["SELECT 1"])],
                )
                input_path = root / "input.json"
                _write_json(input_path, instances)
                with self.assertRaises(ValueError):
                    prepare_rc_gold(
                        "bird_interact_lite",
                        dataset_root=variant_root,
                        input_path=input_path,
                        output_dir=root / "out",
                        cwd=root,
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "gold_sql.json"
            _write_json(input_path, [_instance("one", "db", "Q")])
            with self.assertRaisesRegex(ValueError, "collides with an output"):
                prepare_rc_gold(
                    "bird_interact_lite",
                    dataset_root=root,
                    input_path=input_path,
                    output_dir=root,
                    cwd=root,
                )
            self.assertEqual(json.loads(input_path.read_text())[0]["index"], "one")

    def test_report_replace_failure_leaves_detectable_mismatch_and_recovers(self) -> None:
        """Treating the first replace as complete would hide a mixed output generation."""
        from scripts.prepare_rc_gold import prepare_rc_gold

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            variant_root = root / "bird-interact-lite"
            source_path = variant_root / "bird_interact_data.jsonl"
            input_path = root / "input.json"
            output_dir = root / "out"
            _write_jsonl(
                source_path,
                [_bird_row("one", "db", "Q", ["SELECT 'old'"])],
            )
            _write_json(input_path, [_instance("one", "db", "Q")])
            prepare_rc_gold(
                "bird_interact_lite",
                dataset_root=variant_root,
                input_path=input_path,
                output_dir=output_dir,
                cwd=root,
            )
            old_report_bytes = (
                output_dir / "gold_sql_preparation.json"
            ).read_bytes()
            old_report = json.loads(old_report_bytes)
            self.assertEqual(
                old_report["gold_sql"]["sha256"],
                hashlib.sha256((output_dir / "gold_sql.json").read_bytes()).hexdigest(),
            )

            _write_jsonl(
                source_path,
                [_bird_row("one", "db", "Q", ["SELECT 'new'"])],
            )
            real_replace = os.replace

            def fail_report_replace(source: object, destination: object) -> None:
                if Path(destination).name == "gold_sql_preparation.json":
                    raise OSError("injected report replace failure")
                real_replace(source, destination)

            with mock.patch(
                "scripts.prepare_rc_gold.os.replace",
                side_effect=fail_report_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected report replace failure"):
                    prepare_rc_gold(
                        "bird_interact_lite",
                        dataset_root=variant_root,
                        input_path=input_path,
                        output_dir=output_dir,
                        cwd=root,
                    )

            current_gold_hash = hashlib.sha256(
                (output_dir / "gold_sql.json").read_bytes()
            ).hexdigest()
            self.assertEqual(
                (output_dir / "gold_sql_preparation.json").read_bytes(),
                old_report_bytes,
            )
            self.assertNotEqual(old_report["gold_sql"]["sha256"], current_gold_hash)
            self.assertEqual(list(output_dir.glob(".*.tmp")), [])

            recovered_report = prepare_rc_gold(
                "bird_interact_lite",
                dataset_root=variant_root,
                input_path=input_path,
                output_dir=output_dir,
                cwd=root,
            )
            recovered_hash = hashlib.sha256(
                (output_dir / "gold_sql.json").read_bytes()
            ).hexdigest()
            self.assertEqual(recovered_report["gold_sql"]["sha256"], recovered_hash)
            self.assertEqual(
                json.loads(
                    (output_dir / "gold_sql_preparation.json").read_text()
                )["gold_sql"]["sha256"],
                recovered_hash,
            )

    def test_cli_help_and_cwd_defaults(self) -> None:
        """Breaking the documented invocation or resolving defaults from install path must fail."""
        help_run = subprocess.run(
            [str(PYTHON), "-E", "-B", str(SCRIPT_PATH), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        for flag in (
            "--dataset_split",
            "--dataset-root",
            "--livesqlbench-root",
            "--input-path",
            "--output-dir",
            "--exclude",
        ):
            self.assertIn(flag, help_run.stdout)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jsonl(
                root
                / "BIRD-Interact"
                / "BIRD-Interact-ADK"
                / "bird-interact-lite"
                / "bird_interact_data.jsonl",
                [_bird_row("one", "db", "Question?", ["SELECT 1"])],
            )
            input_path = root / "input.json"
            output_dir = root / "out"
            _write_json(input_path, [_instance("one", "db", "Question?")])
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            run = subprocess.run(
                [
                    str(PYTHON),
                    "-E",
                    "-B",
                    str(SCRIPT_PATH),
                    "--dataset_split",
                    "bird_interact_lite",
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            gold_path = output_dir / "gold_sql.json"
            self.assertEqual(json.loads(gold_path.read_text())[0]["index"], "one")


if __name__ == "__main__":
    unittest.main()
