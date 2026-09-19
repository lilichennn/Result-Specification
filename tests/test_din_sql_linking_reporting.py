import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.baseline_adapters.din_sql.inputs import NODES, digest


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _reseal_source(root, *, exclude=()):
    """Rebuild the synthetic source seal after an intentional fixture change."""
    excluded = {str(Path(value)) for value in exclude}
    index_path = root / "index.json"
    index = json.loads(index_path.read_text())
    files = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_file() and relative not in {"index.json", "COMPLETE.json"} | excluded:
            files.append({"path": relative, "bytes": path.stat().st_size,
                          "sha256": _sha(path)})
    index["files"] = files
    _write_json(index_path, index)
    _write_json(root / "COMPLETE.json", {
        "complete": True, "index_sha256": _sha(index_path),
    })


class DinLinkingUnifiedReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _make_source_handoff(self):
        root = self.root / "source"
        group = "spider_dev"
        ids = ["0", "1", "2", "3"]
        versions = {question_id: f"base-v{question_id}" for question_id in ids}
        linking = {
            "0": ("succeeded", ["t.*"]),
            "1": ("succeeded", ["t.*"]),
            "2": ("succeeded", []),
            "3": ("failed", None),
        }
        questions = [{
            "group": group, "question_id": question_id,
            "version_id": versions[question_id], "schema_ref": "spider_dev:fixture",
            "question": f"question {question_id}", "evidence": "", "rc3": {},
            "database": {"dialect": "sqlite", "database_id": "fixture", "path": "/fixture.sqlite"},
            "source_refs": {"input": "synthetic"}, "label": "EASY",
        } for question_id in ids]
        nodes = []
        version_rows = []
        for question_id in ids:
            node_refs = {}
            for event_no, node in enumerate(NODES, 1):
                status, result = (linking[question_id] if node == "linking"
                                  else ("succeeded", f"{node}-{question_id}"))
                ref = {"group": group, "attempt_id": versions[question_id], "event_no": event_no}
                node_refs[node] = ref
                nodes.append({
                    "group": group, "question_id": question_id,
                    "version_id": versions[question_id], "node": node,
                    "status": status, "result": result, "origin": "fixture",
                    "usage": ({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
                              if status == "succeeded" else None),
                    "reason": None, "fallback_used": False,
                    "refs": {"event": ref, "parent_refs": {}, "response_ref": None, "source_refs": {}},
                    "input_fingerprint": f"base-input-{question_id}-{node}",
                    "event_checksum": f"base-event-{question_id}-{event_no}",
                })
            version_rows.append({
                "group": group, "question_id": question_id,
                "version_id": versions[question_id], "state": "succeeded",
                "attempt_no": 1, "started_at": "start", "finished_at": "finish",
                "parent_version_id": None,
                "attempt_checksum": f"base-attempt-{question_id}",
                "finish_checksum": f"base-finish-{question_id}",
                "node_refs": node_refs, "request_events": 0,
            })
        records = root / "records"
        _write_jsonl(records / "questions.jsonl", questions)
        _write_jsonl(records / "versions.jsonl", version_rows)
        _write_jsonl(records / "nodes.jsonl", nodes)
        _write_jsonl(records / "requests.jsonl", [])
        _write_jsonl(records / "failed_questions.jsonl", [])
        _write_json(records / "summary.json", {
            "format": "din-current-records-v1", "batch_id": "base-batch",
            "manifest_questions": 4, "complete_questions": 4, "nodes": 24,
            "request_events": 0,
            "states": {"succeeded": 4},
            "groups": {group: {"questions": 4, "states": {
                "succeeded": 4, "failed": 0, "pending": 0, "missing": 0}}},
        })
        record_names = ["questions.jsonl", "versions.jsonl", "nodes.jsonl",
                        "requests.jsonl", "failed_questions.jsonl", "summary.json"]
        _write_json(records / "verification.json", {
            "format": "din-current-records-v1", "ok": True,
            "manifest_questions": 4, "exported_questions": 4,
            "current_versions": 4, "six_node_refs_ok": True,
            "files": {name: {"bytes": (records / name).stat().st_size,
                              "sha256": _sha(records / name)} for name in record_names},
        })

        raw = root / "raw_records"
        _write_json(raw / "manifest.json", {
            "format": "din-sql-v1", "batch_id": "base-batch",
            "groups": {group: {"ids": ids}},
        })
        _write_json(raw / "prepared" / "inputs.json", {
            "tasks": [],
            "schemas": {"spider_dev:fixture": {
                # Reporting must not scrape this prompt-oriented text for the
                # physical catalog; the extension freezes canonical metadata.
                "context": "THIS TEXT IS DELIBERATELY NOT A SCHEMA CATALOG",
                "spider": "Table t, columns = [*,a,b]\nForeign_keys = []",
                "primary": "Primary_keys = []",
            }},
            "evaluation": {}, "templates": {}, "identities": {},
        })
        evaluation = root / "evaluation"
        _write_json(evaluation / "summary.json", {
            "format": "din-compact-evaluation-v1",
            group: {"generation": {"finished_questions": 4}, "revision": {"finished_questions": 4}},
            "groups": {group: {"generation": {"finished_questions": 4},
                               "revision": {"finished_questions": 4}}},
        })
        (evaluation / "tables.md").write_text("# Existing tables\n", encoding="utf-8")
        _write_json(evaluation / "versions.json", {
            "format": "din-compact-evaluation-v1",
            "versions": [{"group": group, "question_id": question_id,
                          "version_id": versions[question_id], "state": "succeeded"}
                         for question_id in ids],
        })
        _write_json(evaluation / "progress.json", {"phase": "complete"})
        (root / "guide.md").write_text("source guide\n", encoding="utf-8")

        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name not in {"index.json", "COMPLETE.json"}:
                files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                              "sha256": _sha(path)})
        _write_json(root / "index.json", {
            "format": "din-handoff-v1", "complete": True,
            "batch_id": "base-batch", "manifest_questions": 4,
            "current_questions": 4, "files": files,
            "raw_records_directory": "raw_records", "record_directory": "records",
            "evaluation_directory": "evaluation", "guide": "guide.md",
            "record_summary": "records/summary.json",
            "record_verification": "records/verification.json",
            "evaluation_summary": "evaluation/summary.json",
            "evaluation_versions": "evaluation/versions.json",
            "evaluation_progress": "evaluation/progress.json",
            "evaluation_tables": "evaluation/tables.md",
            "acceptance": {"fixture": True},
        })
        _write_json(root / "COMPLETE.json", {
            "complete": True, "index_sha256": _sha(root / "index.json"),
        })
        return root, versions

    def _make_extension(self, versions):
        root = self.root / "extension"
        group = "spider_dev"
        ids = ["0", "1", "2", "3"]
        source = self.root / "source/raw_records"
        manifest = {"format": "din-sql-linking-v1", "batch_id": "linking-batch",
                    "source": {"batch_id": "base-batch",
                               "manifest_sha256": _sha(source / "manifest.json"),
                               "prepared_sha256": _sha(source / "prepared/inputs.json")},
                    "groups": {group: {"ids": ids}}}
        _write_json(root / "manifest.json", manifest)
        _write_json(root / "prepared/inputs.json", {
            "tasks": [{"key": {"group": group, "question_id": question_id},
                       "schema_ref": "spider_dev:fixture"} for question_id in ids],
            "metadata": {"spider_dev:fixture": [
                {"table_name": "t", "columns": [
                    {"original_column_name": "a"},
                    {"original_column_name": "b"},
                ]},
            ]},
            "source": manifest["source"], "schemas": {}, "templates": {},
            "native_linking": [], "identities": {}, "groups": [group],
        })
        _write_json(root / "monitoring/implementation.json", {"implementation": "fixture"})
        _write_json(root / "monitoring/live.json", {"phase": "complete"})
        store_manifest = {"format": "din-sql-linking-v1", "group": group,
                          "batch_fingerprint": digest(manifest)}
        directory = root / f"group-{group}"
        filter_metadata = {
            question_id: [{"table_name": "t", "columns": [{"original_column_name": "a"}]}]
            for question_id in ids
        }
        with RunStore.create(directory, store_manifest) as store:
            for question_id in ids:
                attempt = store.begin_attempt(question_id, "din_linking_question", "input")
                # This parent is reserved for a prior extension attempt, not
                # the independently sealed six-node base version.
                store.append_event(attempt, "question_start", {"parent_version": None})
                node_refs = {}
                for node in ("schema_filter_rc3", "linking_rc3"):
                    identity = {"node": node, "request_id": f"request-{question_id}-{node}",
                                "attempt_no": 1, "batch_id": "linking-batch", "group": group,
                                "question_id": question_id, "round_execution_id": f"{attempt}:{node}",
                                "sample_position": 0}
                    store.append_event(attempt, "node_input", {
                        "node": node, "kwargs": {"messages": ["SECRET PROMPT"]},
                        "input_fingerprint": f"input-{question_id}-{node}"})
                    store.append_event(attempt, "attempt_queued", identity)
                    store.append_event(attempt, "request_attempt", identity)
                    store.append_event(attempt, "request_dispatch", {**identity, "telemetry": {"queue": 0.1}})
                    response_no = store.append_event(attempt, "request_result", {
                        **identity, "body": {"provider_body": "SECRET PROVIDER BODY"}})
                    succeeded = question_id != "3"
                    status = "succeeded" if succeeded else (
                        "failed" if node == "schema_filter_rc3" else "dependency_failed")
                    result = ({"selection": {"tables": [{"name": "t", "columns": ["a"]}]},
                               "filtered_metadata": filter_metadata[question_id]}
                              if node == "schema_filter_rc3" and succeeded else
                              ["t.*"] if node == "linking_rc3" and succeeded else None)
                    usage = ({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
                             if succeeded else None)
                    response_ref = {"group": group, "attempt_id": attempt, "event_no": response_no}
                    store.append_event(attempt, "request_outcome", {
                        **identity, "status": status, "response_ref": response_ref,
                        "usage": usage, "response_model": "fixture-model",
                        "error": None if succeeded else {"type": "fixture"}})
                    node_no = store.append_event(attempt, "node_result", {
                        "node": node, "status": status, "result": result,
                        "error": None if succeeded else {"type": "fixture"},
                        "input_fingerprint": f"input-{question_id}-{node}",
                        "response_ref": response_ref, "parent_refs": {}, "usage": usage,
                    })
                    node_refs[node] = {"group": group, "attempt_id": attempt, "event_no": node_no}
                store.finish_attempt(attempt, "failed" if question_id == "3" else "succeeded",
                                     {"nodes": {node: {"status": ("succeeded" if question_id != "3"
                                                  else "failed" if node == "schema_filter_rc3"
                                                  else "dependency_failed"), "ref": ref}
                                                for node, ref in node_refs.items()}})
        return root

    def _make_annotations(self):
        path = self.root / "annotations.jsonl"
        gold = {
            "0": (["t"], [["t", "a"]]),
            "1": (["t"], [["t", "b"]]),
            "2": (["t"], []),
            "3": (["t"], [["t", "a"]]),
        }
        _write_jsonl(path, [{
            "task_key": f"spider/dev/i:{question_id}", "status": "resolved",
            "required_tables": tables, "required_columns": columns,
            "required_table_ids": [], "required_column_ids": [],
            "evidence": [], "json_paths": [], "review_reasons": [],
        } for question_id, (tables, columns) in gold.items()])
        return path

    def test_export_builds_one_eight_node_view_with_lineage_and_correct_macro_recall(self):
        from scripts.rc_evaluation.din_sql_linking.reporting import export_unified_handoff

        source, versions = self._make_source_handoff()
        extension = self._make_extension(versions)
        annotations = self._make_annotations()
        output = self.root / "unified"
        source_before = {str(path.relative_to(source)): path.read_bytes()
                         for path in source.rglob("*") if path.is_file()}
        old_node_bytes = (source / "records/nodes.jsonl").read_bytes()
        result = export_unified_handoff(source, extension, annotations, output)

        self.assertEqual(Path(result), output.resolve())
        self.assertEqual(source_before, {str(path.relative_to(source)): path.read_bytes()
                                        for path in source.rglob("*") if path.is_file()})
        self.assertFalse((output / "linking").exists())
        self.assertTrue((output / "raw_records/group-spider_dev/linking_extension.sqlite3").is_file())
        self.assertTrue((output / "raw_records/linking_extension_manifest.json").is_file())
        self.assertTrue((output / "raw_records/linking_extension_prepared_inputs.json").is_file())
        self.assertTrue((output / "raw_records/linking_extension_monitoring_implementation.json").is_file())
        self.assertTrue((output / "raw_records/linking_extension_monitoring_live.json").is_file())
        self.assertTrue((output / "records/nodes.jsonl").read_bytes().startswith(old_node_bytes))
        nodes = _lines(output / "records/nodes.jsonl")
        by_question = {}
        for row in nodes:
            by_question.setdefault(row["question_id"], []).append(row)
        self.assertEqual(set(by_question), {"0", "1", "2", "3"})
        for question_id, rows in by_question.items():
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["node"] for row in rows}), 8)
            self.assertEqual({row["version_id"] for row in rows}, {versions[question_id]})
        versions_rows = {row["question_id"]: row for row in _lines(output / "records/versions.jsonl")}
        self.assertEqual(versions_rows["0"]["source_seals"]["base"]["finish_checksum"], "base-finish-0")
        self.assertEqual(versions_rows["0"]["source_seals"]["base"]["attempt_checksum"], "base-attempt-0")
        self.assertEqual(versions_rows["0"]["extension_lineage"]["state"], "succeeded")
        self.assertNotEqual(versions_rows["0"]["extension_lineage"]["finish_checksum"], "base-finish-0")
        self.assertIn("unified_record_checksum", versions_rows["0"])
        request_text = (output / "records/requests.jsonl").read_text()
        self.assertNotIn("SECRET PROMPT", request_text)
        self.assertNotIn("SECRET PROVIDER BODY", request_text)

        details = {row["question_id"]: row for row in _lines(output / "evaluation/linking_details.jsonl")}
        self.assertEqual(details["0"]["base"]["columns"], [["t", "a"], ["t", "b"]])
        self.assertEqual(details["0"]["rc3"]["columns"], [["t", "a"]])
        self.assertEqual(details["1"]["base"]["column_recall"], 1.0)
        self.assertEqual(details["1"]["rc3"]["column_recall"], 0.0)
        self.assertIsNone(details["2"]["base"]["column_recall"])
        self.assertEqual(details["3"]["base"]["column_recall"], 0.0)
        self.assertEqual(details["3"]["rc3"]["column_recall"], 0.0)
        summary = json.loads((output / "evaluation/summary.json").read_text())
        self.assertEqual(summary["base_evaluation_format"], "din-compact-evaluation-v1")
        linking = summary["linking"]["overall"]
        self.assertEqual(linking["base"]["columns"]["macro"]["recall"], 2 / 3)
        self.assertEqual(linking["base"]["columns"]["macro"]["recall_questions"], 3)
        self.assertEqual(linking["rc3"]["columns"]["macro"]["recall"], 1 / 3)
        self.assertEqual(linking["paired_columns"], {
            "basis": "per_question_recall", "eligible": 3,
            "improvements": 0, "regressions": 1, "unchanged": 2,
        })
        self.assertEqual(linking["filter"]["column_macro_recall"], 1 / 3)
        self.assertEqual(linking["filter"]["column_macro_recall_questions"], 3)
        self.assertEqual(linking["schema_reduction"]["macro_column_reduction"], 0.5)
        self.assertEqual(linking["tokens"]["base_linking"]["total_tokens"], {
            "sum": 15, "known_questions": 3, "unknown_questions": 1})
        self.assertEqual(linking["tokens"]["schema_filter_rc3"]["prompt_tokens"], {
            "sum": 21, "known_questions": 3, "unknown_questions": 1})
        self.assertEqual(linking["tokens"]["rc3_combined"]["total_tokens"], {
            "sum": 60, "known_questions": 3, "unknown_questions": 1})
        self.assertIn("## RC3 schema filtering and DIN Linking", (output / "evaluation/tables.md").read_text())
        guide = (output / "guide.md").read_text()
        self.assertIn("Eight-node unified view", guide)
        self.assertIn("Integrity verification", guide)
        self.assertIn("din-handoff-v2", guide)
        self.assertNotIn("din-handoff-v1", guide)
        self.assertNotIn("six_node_refs_ok", guide)
        self.assertNotIn("六节点流程", guide)

        index = json.loads((output / "index.json").read_text())
        self.assertEqual(index["format"], "din-handoff-v2")
        self.assertTrue(index["complete"])
        self.assertEqual(index["linking_details"], "evaluation/linking_details.jsonl")
        self.assertEqual(index["extension_lineage"]["audit_files"], {
            "manifest": "raw_records/linking_extension_manifest.json",
            "prepared_inputs": "raw_records/linking_extension_prepared_inputs.json",
            "monitoring_implementation": "raw_records/linking_extension_monitoring_implementation.json",
            "monitoring_live": "raw_records/linking_extension_monitoring_live.json",
        })
        listed = {row["path"] for row in index["files"]}
        actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
                  and str(path.relative_to(output)) not in {"index.json", "COMPLETE.json"}}
        self.assertEqual(listed, actual)
        for row in index["files"]:
            path = output / row["path"]
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], _sha(path))
        complete = json.loads((output / "COMPLETE.json").read_text())
        self.assertEqual(complete, {"complete": True, "index_sha256": _sha(output / "index.json")})

    def test_structured_catalog_and_prediction_support_punctuation_and_quoted_identifiers(self):
        from scripts.rc_evaluation.din_sql_linking.reporting import _prediction

        catalog = {"odd, \"table\"": ("space col", "comma,col", "quote\"col")}
        result = [
            '"odd, ""table"""."space col"',
            '"odd, ""table""".`comma,col`',
            '"odd, ""table"""."quote""col"',
        ]
        tables, columns = _prediction(result, catalog)
        self.assertEqual(tables, {"odd, \"table\""})
        self.assertEqual(columns, {
            ("odd, \"table\"", "space col"),
            ("odd, \"table\"", "comma,col"),
            ("odd, \"table\"", "quote\"col"),
        })

    def test_evaluation_catalog_trims_metadata_identifiers_consistently(self):
        from scripts.rc_evaluation.din_sql_linking.reporting import (
            _canonical_gold,
            _catalog_from_metadata,
            _filtered_catalog,
            _prediction,
        )

        catalog = _catalog_from_metadata([{
            "table_name": " frpm ",
            "columns": [{"original_column_name": " District Name "}],
        }])
        filtered = _filtered_catalog({"filtered_metadata": [{
            "table_name": " frpm ",
            "columns": [{"original_column_name": " District Name "}],
        }]}, catalog)
        annotation = {
            "status": "resolved",
            "required_tables": ["frpm"],
            "required_columns": [["frpm", "District Name"]],
        }

        expected_tables = {"frpm"}
        expected_columns = {("frpm", "District Name")}
        self.assertEqual(catalog, {"frpm": ("District Name",)})
        self.assertEqual(filtered, catalog)
        self.assertEqual(_canonical_gold(annotation, catalog),
                         (expected_tables, expected_columns))
        self.assertEqual(_prediction(["frpm.District Name"], catalog),
                         (expected_tables, expected_columns))

    def test_evaluation_catalog_rejects_empty_or_conflicting_trimmed_identifiers(self):
        from scripts.rc_evaluation.din_sql_linking.reporting import (
            _catalog_from_metadata,
            _filtered_catalog,
        )

        invalid_metadata = (
            [{"table_name": "   ", "columns": []}],
            [{"table_name": " t ", "columns": []},
             {"table_name": "T", "columns": []}],
            [{"table_name": "t", "columns": [
                {"original_column_name": " c "},
                {"original_column_name": "C"},
            ]}],
            [{"table_name": "t", "columns": [
                {"original_column_name": "", "column_name": "fallback"},
            ]}],
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "malformed|duplicate"):
                    _catalog_from_metadata(metadata)

        catalog = {"t": ("c",)}
        with self.assertRaisesRegex(ValueError, "duplicate|unknown"):
            _filtered_catalog({"filtered_metadata": [
                {"table_name": " t ", "columns": [{"original_column_name": " c "}]},
                {"table_name": "t", "columns": [{"original_column_name": "c"}]},
            ]}, catalog)
        with self.assertRaisesRegex(ValueError, "malformed"):
            _filtered_catalog({"filtered_metadata": "not-a-list"}, catalog)
        with self.assertRaisesRegex(ValueError, "duplicate|unknown"):
            _filtered_catalog({"filtered_metadata": [{
                "table_name": "t",
                "columns": [{"original_column_name": "", "column_name": "c"}],
            }]}, catalog)

    def test_rejects_corrupt_source_and_failed_build_preserves_existing_target(self):
        from scripts.rc_evaluation.din_sql_linking import reporting

        source, versions = self._make_source_handoff()
        extension = self._make_extension(versions)
        annotations = self._make_annotations()
        (source / "records/questions.jsonl").write_bytes(
            (source / "records/questions.jsonl").read_bytes() + b" \n")
        with self.assertRaisesRegex(ValueError, "source.*hash|hash.*source"):
            reporting.export_unified_handoff(source, extension, annotations, self.root / "corrupt")

        shutil.rmtree(source)
        shutil.rmtree(extension)
        source, versions = self._make_source_handoff()
        extension = self._make_extension(versions)
        output = self.root / "interrupted"
        output.mkdir()
        _write_json(output / "COMPLETE.json", {"complete": True, "index_sha256": "stale"})
        (output / "old-data.bin").write_bytes(b"old handoff remains byte-identical")
        before = {str(path.relative_to(output)): path.read_bytes()
                  for path in output.rglob("*") if path.is_file()}
        with patch.object(reporting, "_copy_regular", side_effect=RuntimeError("injected copy failure")):
            with self.assertRaisesRegex(RuntimeError, "injected copy failure"):
                reporting.export_unified_handoff(source, extension, annotations, output)
        self.assertEqual(before, {str(path.relative_to(output)): path.read_bytes()
                                  for path in output.rglob("*") if path.is_file()})

        absent = self.root / "interrupted-absent"
        with patch.object(reporting, "_copy_regular", side_effect=RuntimeError("injected copy failure")):
            with self.assertRaisesRegex(RuntimeError, "injected copy failure"):
                reporting.export_unified_handoff(source, extension, annotations, absent)
        self.assertFalse((absent / "COMPLETE.json").exists())

    def test_post_publish_backup_cleanup_failure_is_best_effort_and_retried(self):
        from scripts.rc_evaluation.din_sql_linking import reporting

        output = self.root / "published"
        output.mkdir()
        (output / "payload.txt").write_text("old", encoding="utf-8")
        first_stage = self.root / "first-stage"
        first_stage.mkdir()
        (first_stage / "payload.txt").write_text("new-one", encoding="utf-8")
        real_rmtree = shutil.rmtree
        failed = []

        def fail_first_backup_cleanup(path, *args, **kwargs):
            candidate = Path(path)
            if candidate.name.startswith(".published.old-") and not failed:
                failed.append(candidate)
                raise OSError("injected backup cleanup failure")
            return real_rmtree(path, *args, **kwargs)

        with patch.object(reporting.shutil, "rmtree", side_effect=fail_first_backup_cleanup):
            reporting._replace_output(first_stage, output)

        self.assertEqual((output / "payload.txt").read_text(), "new-one")
        stale = list(self.root.glob(".published.old-*"))
        self.assertEqual(stale, failed)
        self.assertEqual((stale[0] / "payload.txt").read_text(), "old")

        second_stage = self.root / "second-stage"
        second_stage.mkdir()
        (second_stage / "payload.txt").write_text("new-two", encoding="utf-8")
        reporting._replace_output(second_stage, output)
        self.assertEqual((output / "payload.txt").read_text(), "new-two")
        self.assertEqual(list(self.root.glob(".published.old-*")), [])

    def test_nested_index_and_complete_payloads_are_covered_by_new_index(self):
        from scripts.rc_evaluation.din_sql_linking.reporting import export_unified_handoff

        source, versions = self._make_source_handoff()
        extension = self._make_extension(versions)
        annotations = self._make_annotations()
        (source / "raw_records/audit").mkdir(parents=True)
        (source / "raw_records/audit/index.json").write_text("nested-index", encoding="utf-8")
        (source / "raw_records/audit/COMPLETE.json").write_text("nested-complete", encoding="utf-8")
        (source / "raw_records/audit/store.sqlite-wal").write_text("nested-wal", encoding="utf-8")
        _reseal_source(source)

        output = self.root / "nested-reserved"
        export_unified_handoff(source, extension, annotations, output)
        listed = {row["path"] for row in json.loads((output / "index.json").read_text())["files"]}
        actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
                  and str(path.relative_to(output)) not in {"index.json", "COMPLETE.json"}}
        self.assertEqual(listed, actual)
        self.assertTrue({"raw_records/audit/index.json",
                         "raw_records/audit/COMPLETE.json",
                         "raw_records/audit/store.sqlite-wal"} <= listed)

    def test_unindexed_source_guide_is_not_read_without_explicit_override(self):
        from scripts.rc_evaluation.din_sql_linking.reporting import export_unified_handoff

        source, versions = self._make_source_handoff()
        extension = self._make_extension(versions)
        annotations = self._make_annotations()
        (source / "guide.md").write_text("UNSEALED GUIDE MUST NOT CROSS BOUNDARY\n", encoding="utf-8")
        _reseal_source(source, exclude={"guide.md"})

        output = self.root / "guide-boundary"
        export_unified_handoff(source, extension, annotations, output)
        guide = (output / "guide.md").read_text(encoding="utf-8")
        self.assertNotIn("UNSEALED GUIDE", guide)
        self.assertIn("# DIN-SQL × Qwen3.8 2.4T", guide)
        self.assertIn("Unified RC3 Linking extension", guide)

    def test_copy_corruption_is_detected_before_new_seal_and_preserves_target(self):
        from scripts.rc_evaluation.din_sql_linking import reporting

        source, versions = self._make_source_handoff()
        extension = self._make_extension(versions)
        annotations = self._make_annotations()
        output = self.root / "copy-corruption"
        output.mkdir()
        _write_json(output / "COMPLETE.json", {"complete": True, "index_sha256": "old"})
        (output / "payload.bin").write_bytes(b"authenticated old target")
        before = {str(path.relative_to(output)): path.read_bytes()
                  for path in output.rglob("*") if path.is_file()}
        real_copy = reporting._copy_regular

        def copy_then_corrupt(source_path, destination_path):
            real_copy(source_path, destination_path)
            if Path(source_path).resolve() == (source / "raw_records/manifest.json").resolve():
                with Path(destination_path).open("ab") as stream:
                    stream.write(b"CORRUPTED AFTER COPY")

        with patch.object(reporting, "_copy_regular", side_effect=copy_then_corrupt):
            with self.assertRaisesRegex(ValueError, "copied source.*mismatch"):
                reporting.export_unified_handoff(source, extension, annotations, output)

        self.assertEqual(before, {str(path.relative_to(output)): path.read_bytes()
                                  for path in output.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
