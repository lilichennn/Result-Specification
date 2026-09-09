from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SNOW_DIR = PROJECT_ROOT / "code" / "scripts" / "spider2_snow"
SNOW_DATA_PATH = SNOW_DIR / "preprocessed_data" / "spider2_snow.json"
SNOW_META_DIR = SNOW_DIR / "preprocessed_data" / "meta"
SNOW_RC_PATH = SNOW_DIR / "rc.json"

LITE_SOURCE_DIR = PROJECT_ROOT / "Spider2.0" / "spider2-lite"
LITE_DATASOURCE_PATH = LITE_SOURCE_DIR / "spider2-lite_datasource_map.csv"

LITE_OUTPUT_DIR = PROJECT_ROOT / "code" / "scripts" / "spider2_lite"
LITE_DATA_PATH = LITE_OUTPUT_DIR / "preprocessed_data" / "spider2_lite.json"
LITE_META_DIR = LITE_OUTPUT_DIR / "preprocessed_data" / "meta"
LITE_RC_PATH = LITE_OUTPUT_DIR / "rc.json"


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return rows


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _read_datasource_map(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _build_mapping(
    snow_rows: list[dict[str, Any]],
    datasource_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    lite_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in datasource_rows:
        key = (row["question"], row["matched_resource_db"].casefold())
        lite_by_key[key].append(row)

    mapping: dict[str, dict[str, str]] = {}
    for row in snow_rows:
        old_id = str(row["index"])
        key = (str(row["question"]), str(row["db_id"]).casefold())
        matches = lite_by_key.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected one Lite match for {old_id}, found {len(matches)}"
            )
        mapping[old_id] = matches[0]

    target_ids = [row["instance_id"] for row in mapping.values()]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Multiple Snow instances map to the same Lite instance_id")
    return mapping


def _transfer_and_filter(
    rows: list[dict[str, Any]],
    mapping: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    source_ids = [str(row["index"]) for row in rows]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(mapping):
        raise ValueError("Snow data and RC files do not contain the same indexes")

    output: list[dict[str, Any]] = []
    for row in rows:
        match = mapping[str(row["index"])]
        if match["backend"].casefold() == "snowflake":
            continue
        updated = dict(row)
        updated["index"] = match["instance_id"]
        output.append(updated)
    return output


def main() -> None:
    snow_data = _read_json(SNOW_DATA_PATH)
    snow_rc = _read_json(SNOW_RC_PATH)
    datasource_rows = _read_datasource_map(LITE_DATASOURCE_PATH)

    mapping = _build_mapping(snow_data, datasource_rows)
    lite_data = _transfer_and_filter(snow_data, mapping)
    lite_rc = _transfer_and_filter(snow_rc, mapping)

    data_ids = {row["index"] for row in lite_data}
    rc_ids = {row["index"] for row in lite_rc}
    if data_ids != rc_ids:
        raise ValueError("Transferred Lite data and RC files have different ID sets")
    if LITE_OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {LITE_OUTPUT_DIR}")

    _write_json(LITE_DATA_PATH, lite_data)
    _write_json(LITE_RC_PATH, lite_rc)

    retained_databases = sorted({str(row["db_id"]) for row in lite_data})
    LITE_META_DIR.mkdir(parents=True)
    for db_id in retained_databases:
        source = SNOW_META_DIR / db_id
        if not source.is_dir():
            raise FileNotFoundError(f"Metadata directory not found: {source}")
        shutil.copytree(source, LITE_META_DIR / db_id)

    removed_instances = len(snow_data) - len(lite_data)
    removed_databases = sum(
        path.is_dir() for path in SNOW_META_DIR.iterdir()
    ) - len(retained_databases)
    print(
        f"Created {LITE_OUTPUT_DIR}: retained {len(lite_data)} instances and "
        f"{len(retained_databases)} databases; excluded {removed_instances} "
        f"Snowflake instances and {removed_databases} databases."
    )


if __name__ == "__main__":
    main()
