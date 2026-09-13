"""Offline tests for frozen PostgreSQL values; no server or model is contacted."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


BASELINE = Path(__file__).resolve().parents[1] / "baselines" / "DeepEye-SQL"
sys.path.insert(0, str(BASELINE))


def _item(*, question: str = "Which value?", schema=None):
    if schema is None:
        schema = {
            "db_id": "DemoDB",
            "db_type": "postgresql",
            "tables": {
                "Items": {
                    "columns": {
                        "id": {"column_type": "INTEGER"},
                        "empty_value": {"column_type": "TEXT"},
                        "label": {"column_type": "VARCHAR(80)"},
                    }
                }
            },
        }
    return SimpleNamespace(
        db_type="postgresql",
        database_id="DemoDB",
        database_schema=schema,
        question=question,
    )


def _one_column_item():
    return _item(schema={
        "db_id": "DemoDB",
        "db_type": "postgresql",
        "tables": {
            "Items": {"columns": {"label": {"column_type": "TEXT"}}}
        },
    })


def _content_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strip_integrity_fields(output_dir: Path):
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("content_hash", None)
    for entry in manifest["columns"]:
        entry.pop("checkpoint_hash", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for checkpoint_path in (output_dir / "column_checkpoints").glob("*.json"):
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint.pop("content_hash", None)
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")


class _Cursor:
    def __init__(self, responses, queries):
        self._responses = iter(responses)
        self._rows = None
        self._queries = queries
        self.closed = False

    def execute(self, query, params=None):
        rendered = query.as_string(None)
        self._queries.append((rendered, params))
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        self._rows = response

    def fetchall(self):
        return list(self._rows)

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, responses, queries):
        self.handle = _Cursor(responses, queries)
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.handle

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class DatabaseValueCollectionTest(unittest.TestCase):
    def setUp(self):
        module_name = "scripts.baseline_adapters.deepeye.precompute_values"
        self.assertIsNotNone(
            importlib.util.find_spec(module_name),
            "database-value precomputation is not implemented",
        )
        self.collect = importlib.import_module(module_name).collect_database_values
        self.connection_kwargs = {
            "host": "pg.internal",
            "port": 5433,
            "user": "readonly",
            "password": "top-secret",
        }

    def test_collects_only_meta_text_columns_with_exact_counts_and_native_filters(self):
        """Broad type scans, approximate counts, or retaining ID-like text must fail."""
        schema = {
            "db_id": "DemoDB",
            "db_type": "postgresql",
            "tables": {
                "Odd\"Table": {
                    "columns": {
                        "id": {"column_type": "BIGINT"},
                        "empty_value": {"column_type": "TEXT"},
                        "label": {"column_type": "VARCHAR(80)"},
                        "numeric_code": {"column_type": "VARCHAR(20)"},
                        "payload": {"column_type": "JSONB"},
                        "uuid_code": {"column_type": "CHAR(36)"},
                    }
                }
            },
        }
        queries = []
        connection = _Connection(
            [
                [],
                [("Beta", 3), ("Alpha", 3)],
                [("007", 2), ("42", 2)],
                [
                    ("123e4567-e89b-12d3-a456-426614174000", 2),
                    ("123e4567-e89b-12d3-a456-426614174001", 2),
                ],
            ],
            queries,
        )
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "psycopg.connect", return_value=connection
        ) as connect:
            output_dir = Path(temporary_directory) / "values"
            manifest = self.collect(
                _item(schema=schema), output_dir,
                connection_kwargs=self.connection_kwargs,
                max_values_per_column=2, timeout_seconds=7,
            )

            self.assertEqual(connect.call_count, 1)
            self.assertEqual(len(queries), 4)
            self.assertTrue(connection.rolled_back)
            self.assertTrue(connection.closed)
            self.assertTrue(connection.handle.closed)
            for query, params in queries:
                self.assertIn('FROM "public"."Odd""Table"', query)
                self.assertIn("WITH eligible_values AS", query)
                self.assertIn("COUNT(*) OVER ()", query)
                self.assertIn("ORDER BY md5(value), value", query)
                self.assertIn("LENGTH(CAST(", query)
                self.assertIn("BETWEEN 1 AND 100", query)
                self.assertEqual(params, (2,))
                self.assertNotIn('"id"', query)
                self.assertNotIn('"payload"', query)

            by_name = {entry["column_name"]: entry for entry in manifest["columns"]}
            self.assertEqual(by_name["empty_value"]["eligible_distinct_count"], 0)
            self.assertEqual(by_name["empty_value"]["documents"], [])
            self.assertEqual(by_name["empty_value"]["status"], "empty")
            self.assertEqual(by_name["label"]["eligible_distinct_count"], 3)
            self.assertEqual(by_name["label"]["documents"], ["Beta", "Alpha"])
            self.assertEqual(by_name["label"]["status"], "collected")
            self.assertEqual(by_name["numeric_code"]["documents"], [])
            self.assertEqual(by_name["numeric_code"]["status"], "filtered_numeric")
            self.assertEqual(by_name["uuid_code"]["documents"], [])
            self.assertEqual(by_name["uuid_code"]["status"], "filtered_uuid")
            self.assertEqual(manifest["stats"], {
                "text_column_count": 4,
                "successful_column_count": 4,
                "eligible_distinct_count": 7,
                "document_count": 2,
                "empty_column_count": 1,
                "filtered_numeric_column_count": 1,
                "filtered_uuid_column_count": 1,
            })
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["policy"]["min_value_length"], 1)
            self.assertEqual(manifest["policy"]["max_value_length"], 100)
            self.assertIn("md5", manifest["policy"]["ordering"])
            self.assertEqual(manifest["policy"]["skipped_value_kinds"], ["uuid", "numeric"])
            self.assertNotIn("top-secret", json.dumps(manifest))
            self.assertNotIn("SELECT", json.dumps(manifest))
            self.assertEqual(json.loads((output_dir / "schema.json").read_text(encoding="utf-8")), schema)
            self.assertEqual(json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")), manifest)

    def test_quotes_declared_identifiers_as_identifiers(self):
        """Interpolating a Meta name as SQL syntax instead of an identifier must fail."""
        column_name = 'display"; DROP TABLE secrets; --'
        schema = {"db_id": "DemoDB", "db_type": "postgresql", "tables": {
            'odd"table': {"columns": {column_name: {"column_type": "TEXT"}}}
        }}
        queries = []
        connection = _Connection([[("safe", 1)]], queries)
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "psycopg.connect", return_value=connection
        ):
            result = self.collect(
                _item(schema=schema), Path(temporary_directory),
                connection_kwargs=self.connection_kwargs,
                max_values_per_column=1,
            )
        query = queries[0][0]
        self.assertIn('"public"."odd""table"', query)
        self.assertGreaterEqual(query.count('"display""; DROP TABLE secrets; --"'), 3)
        self.assertEqual(result["columns"][0]["column_name"], column_name)
        self.assertEqual(result["columns"][0]["documents"], ["safe"])

    def test_failed_column_is_not_empty_and_successful_column_resumes(self):
        """Turning errors into empty data or re-querying completed columns must fail."""
        first_queries = []
        failed_connection = _Connection(
            [[], RuntimeError("server said password=top-secret")], first_queries
        )
        second_queries = []
        resumed_connection = _Connection([[("Alpha", 1)]], second_queries)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch("psycopg.connect", return_value=failed_connection):
                with self.assertRaisesRegex(RuntimeError, "Items.label") as raised:
                    self.collect(
                        _item(), output_dir,
                        connection_kwargs=self.connection_kwargs,
                        max_values_per_column=2,
                    )
            self.assertNotIn("top-secret", str(raised.exception))
            partial = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(partial["status"], "failed")
            by_name = {entry["column_name"]: entry for entry in partial["columns"]}
            self.assertEqual(by_name["empty_value"]["status"], "empty")
            self.assertEqual(by_name["label"]["status"], "failed")
            self.assertNotEqual(by_name["label"]["status"], "empty")

            with patch("psycopg.connect", return_value=resumed_connection):
                complete = self.collect(
                    _item(question="A question must not affect samples."), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            self.assertEqual(len(first_queries), 2)
            self.assertEqual(len(second_queries), 1)
            self.assertIn('"label"', second_queries[0][0])
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["stats"]["document_count"], 1)
            self.assertTrue(failed_connection.rolled_back and failed_connection.closed)
            self.assertTrue(resumed_connection.rolled_back and resumed_connection.closed)

            with patch("psycopg.connect", side_effect=AssertionError("complete cache must resume")):
                repeated = self.collect(
                    _item(question="Entirely different question"), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            self.assertEqual(repeated, complete)

    def test_connection_failure_is_sanitized_and_recorded(self):
        """Leaking driver diagnostics or leaving failures pending must fail."""
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "psycopg.connect",
            side_effect=RuntimeError("password=top-secret; host internals"),
        ):
            output_dir = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "DemoDB") as raised:
                self.collect(
                    _item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            self.assertNotIn("top-secret", str(raised.exception))
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(
                {entry["status"] for entry in manifest["columns"]}, {"failed"}
            )
            self.assertEqual(manifest["stats"]["successful_column_count"], 0)
            self.assertEqual(manifest["stats"]["document_count"], 0)

    def test_rejects_cache_when_schema_policy_or_source_changes(self):
        """Reusing documents under a different collection identity must fail closed."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            connection = _Connection([[], [("Alpha", 1)]], [])
            with patch("psycopg.connect", return_value=connection):
                self.collect(
                    _item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )

            changed_schema = _item()
            changed_schema.database_schema = json.loads(json.dumps(changed_schema.database_schema))
            changed_schema.database_schema["tables"]["Items"]["columns"]["new"] = {"column_type": "TEXT"}
            attempts = [
                (_item(), self.connection_kwargs, 3),
                (_item(), {**self.connection_kwargs, "host": "other.internal"}, 2),
                (changed_schema, self.connection_kwargs, 2),
            ]
            for item, kwargs, cap in attempts:
                with self.subTest(kwargs=kwargs, cap=cap), patch(
                    "psycopg.connect", side_effect=AssertionError("must reject before connecting")
                ), self.assertRaisesRegex(ValueError, "cache configuration"):
                    self.collect(
                        item, output_dir, connection_kwargs=kwargs,
                        max_values_per_column=cap,
                    )

    def test_forces_read_only_connection_options_without_persisting_credentials(self):
        """Allowing caller options to weaken read-only or timeout settings must fail."""
        connection = _Connection([[], [("Alpha", 1)]], [])
        supplied = {
            **self.connection_kwargs,
            "options": "-c default_transaction_read_only=off -c statement_timeout=0",
            "autocommit": True,
            "dbname": "DemoDB",
        }
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "psycopg.connect", return_value=connection
        ) as connect:
            manifest = self.collect(
                _item(), Path(temporary_directory),
                connection_kwargs=supplied,
                max_values_per_column=2, timeout_seconds=9,
            )
        actual = connect.call_args.kwargs
        self.assertEqual(actual["dbname"], "DemoDB")
        self.assertFalse(actual["autocommit"])
        self.assertIn("default_transaction_read_only=on", actual["options"])
        self.assertIn("search_path=pg_catalog", actual["options"])
        self.assertIn("statement_timeout=9000", actual["options"])
        self.assertIn("lock_timeout=5000", actual["options"])
        self.assertIn("idle_in_transaction_session_timeout=9000", actual["options"])
        self.assertNotIn("=off", actual["options"])
        self.assertNotIn("top-secret", json.dumps(manifest))

    def test_resume_rejects_legal_shape_checkpoint_document_tampering(self):
        """Changing a document while preserving valid JSON shape must fail."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            checkpoint_path = next(
                (output_dir / "column_checkpoints").glob("*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["documents"] = ["Bravo"]
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            with patch(
                "psycopg.connect", side_effect=AssertionError("must not connect")
            ), self.assertRaisesRegex(RuntimeError, "integrity"):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )

    def test_resume_rejects_manifest_tampering(self):
        """Changing a valid-shaped manifest must fail before checkpoint reuse."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["columns"][0]["documents"] = ["Bravo"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with patch(
                "psycopg.connect", side_effect=AssertionError("must not connect")
            ), self.assertRaisesRegex(RuntimeError, "integrity"):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )

    def test_resume_rejects_cross_source_checkpoint_with_rewritten_identity(self):
        """A checkpoint copied across sources cannot be made valid by editing its identity."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_dir = root / "first"
            second_dir = root / "second"
            first_kwargs = {**self.connection_kwargs, "host": "first.internal"}
            second_kwargs = {**self.connection_kwargs, "host": "second.internal"}
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                first_manifest = self.collect(
                    _one_column_item(), first_dir,
                    connection_kwargs=first_kwargs,
                    max_values_per_column=2,
                )
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                self.collect(
                    _one_column_item(), second_dir,
                    connection_kwargs=second_kwargs,
                    max_values_per_column=2,
                )

            first_checkpoint = next(
                (first_dir / "column_checkpoints").glob("*.json")
            )
            second_checkpoint = next(
                (second_dir / "column_checkpoints").glob("*.json")
            )
            shutil.copyfile(second_checkpoint, first_checkpoint)
            copied = json.loads(first_checkpoint.read_text(encoding="utf-8"))
            copied["collection_fingerprint"] = first_manifest[
                "collection_fingerprint"
            ]
            first_checkpoint.write_text(json.dumps(copied), encoding="utf-8")

            with patch(
                "psycopg.connect", side_effect=AssertionError("must not connect")
            ), self.assertRaisesRegex(RuntimeError, "integrity"):
                self.collect(
                    _one_column_item(), first_dir,
                    connection_kwargs=first_kwargs,
                    max_values_per_column=2,
                )

    def test_explicit_legacy_seal_adds_verified_integrity_chain(self):
        """Normal reads must reject legacy v1 until the trusted migration seals it."""
        module = importlib.import_module(
            "scripts.baseline_adapters.deepeye.precompute_values"
        )
        self.assertTrue(
            hasattr(module, "seal_legacy_collection"),
            "explicit legacy collection sealing is not implemented",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            _strip_integrity_fields(output_dir)

            with patch(
                "psycopg.connect", side_effect=AssertionError("must not connect")
            ), self.assertRaisesRegex(RuntimeError, "unsealed"):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )

            sealed = module.seal_legacy_collection(output_dir)
            checkpoint_path = next(
                (output_dir / "column_checkpoints").glob("*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_payload = dict(checkpoint)
            checkpoint_content_hash = checkpoint_payload.pop("content_hash")
            self.assertEqual(checkpoint_content_hash, _content_hash(checkpoint_payload))
            self.assertEqual(
                sealed["columns"][0]["checkpoint_hash"],
                _content_hash(checkpoint),
            )
            manifest_payload = dict(sealed)
            manifest_content_hash = manifest_payload.pop("content_hash")
            self.assertEqual(manifest_content_hash, _content_hash(manifest_payload))

            with patch(
                "psycopg.connect", side_effect=AssertionError("sealed cache must resume")
            ):
                resumed = self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            self.assertEqual(resumed, sealed)

    def test_legacy_seal_rejects_manifest_checkpoint_value_mismatch(self):
        """The migration must not legitimize inconsistent legacy documents."""
        module = importlib.import_module(
            "scripts.baseline_adapters.deepeye.precompute_values"
        )
        self.assertTrue(
            hasattr(module, "seal_legacy_collection"),
            "explicit legacy collection sealing is not implemented",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            _strip_integrity_fields(output_dir)
            checkpoint_path = next(
                (output_dir / "column_checkpoints").glob("*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["documents"] = ["Bravo"]
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "[Ll]egacy.*inconsistent"):
                module.seal_legacy_collection(output_dir)

    def test_verify_collection_checks_the_complete_integrity_chain(self):
        """CLI verification must use the manifest, schema, and checkpoint hashes."""
        module = importlib.import_module(
            "scripts.baseline_adapters.deepeye.precompute_values"
        )
        self.assertTrue(
            hasattr(module, "verify_collection"),
            "strict collection verification is not implemented",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch(
                "psycopg.connect",
                return_value=_Connection([[('Alpha', 1)]], []),
            ):
                manifest = self.collect(
                    _one_column_item(), output_dir,
                    connection_kwargs=self.connection_kwargs,
                    max_values_per_column=2,
                )
            self.assertEqual(module.verify_collection(output_dir), manifest)

            checkpoint_path = next(
                (output_dir / "column_checkpoints").glob("*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["documents"] = ["Bravo"]
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                module.verify_collection(output_dir)


if __name__ == "__main__":
    unittest.main()
