from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from result_contract.data_preprocess import preprocess_spider2_snow
from scripts.generate_rc import _load_instances


class Spider2SnowPreprocessTest(unittest.TestCase):
    def test_filters_large_databases_and_writes_compatible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Spider2.0"
            snow_root = root / "spider2-snow"
            database_root = snow_root / "resource" / "databases"
            document_root = snow_root / "resource" / "documents"
            document_root.mkdir(parents=True)
            (document_root / "metric.md").write_text(
                "Metric definition.\n",
                encoding="utf-8",
            )

            instances = [
                {
                    "instance_id": "sf_small_1",
                    "instruction": "Return the metric.",
                    "db_id": "SMALL",
                    "external_knowledge": "metric.md",
                },
                {
                    "instance_id": "sf_large_1",
                    "instruction": "Return the large metric.",
                    "db_id": "LARGE",
                    "external_knowledge": None,
                },
                {
                    "instance_id": "sf_small_2",
                    "instruction": "Return the identifier.",
                    "db_id": "SMALL",
                    "external_knowledge": None,
                },
            ]
            (snow_root / "spider2-snow.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in instances),
                encoding="utf-8",
            )

            self._write_table(
                database_root / "SMALL" / "PUBLIC" / "METRICS.json",
                "SMALL.PUBLIC.METRICS",
                ["id", "metric"],
                ["NUMBER", "FLOAT"],
                ["Identifier", "Metric value"],
                [
                    {"ID": float("nan"), "metric": "x" * 1001},
                    {"ID": 1, "metric": 2.5},
                    {"ID": 2, "metric": 2.5},
                ],
            )
            large_columns = [f"column_{index}" for index in range(3001)]
            self._write_table(
                database_root / "LARGE" / "PUBLIC" / "WIDE.json",
                "LARGE.PUBLIC.WIDE",
                large_columns,
                ["TEXT"] * len(large_columns),
                [None] * len(large_columns),
                [],
            )

            output_dir = Path(temporary_directory) / "output"
            summary = preprocess_spider2_snow(root, output_dir)

            self.assertEqual(summary["source_instance_count"], 3)
            self.assertEqual(summary["instance_count"], 2)
            self.assertEqual(summary["excluded_instance_count"], 1)
            self.assertEqual(summary["database_count"], 1)
            self.assertEqual(summary["excluded_database_count"], 1)
            self.assertEqual(summary["column_count"], 2)

            output_instances = json.loads(
                (output_dir / "spider2_snow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["index"] for row in output_instances],
                ["sf_small_1", "sf_small_2"],
            )
            self.assertNotIn("instance_id", output_instances[0])
            self.assertEqual(output_instances[0]["evidence"], "Metric definition.")
            self.assertEqual(output_instances[1]["evidence"], "")
            self.assertEqual(
                _load_instances(output_dir / "spider2_snow.json"),
                output_instances,
            )

            metadata_path = (
                output_dir / "meta" / "SMALL" / "SMALL.PUBLIC.METRICS.csv"
            )
            with metadata_path.open(encoding="utf-8", newline="") as file:
                metadata = list(csv.DictReader(file))
            self.assertEqual(metadata[0]["column_name"], "id")
            self.assertEqual(metadata[0]["column_description"], "Identifier")
            self.assertEqual(json.loads(metadata[0]["sample_value"]), [1, 2])
            self.assertEqual(json.loads(metadata[1]["sample_value"]), [2.5])
            self.assertFalse((output_dir / "meta" / "LARGE").exists())

            manifest = json.loads(
                (output_dir / "filter_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["filter"]["threshold"], 3000)
            self.assertEqual(
                [row["db_id"] for row in manifest["excluded_databases"]],
                ["LARGE"],
            )
            self.assertEqual(manifest["excluded_instance_ids"], ["sf_large_1"])

    @staticmethod
    def _write_table(
        path: Path,
        table_fullname: str,
        column_names: list[str],
        column_types: list[str],
        descriptions: list[str | None],
        sample_rows: list[dict[str, object]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "table_name": path.stem,
                    "table_fullname": table_fullname,
                    "column_names": column_names,
                    "column_types": column_types,
                    "description": descriptions,
                    "sample_rows": sample_rows,
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
