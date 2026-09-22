from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


REQUIRED_INSTANCE_FIELDS = {"db_id", "question", "evidence"}


def preprocess_bird(
    bird_root: str | Path,
    split: str = "dev",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Preprocess one BIRD split and copy its database metadata."""
    bird_root = Path(bird_root).resolve()
    output_dir = Path(
        output_dir if output_dir is not None
        else Path(__file__).resolve().parents[2] / "data" / f"bird_{split}"
    ).resolve()
    split_root = bird_root / "data" / split
    dataset_path = split_root / f"{split}.json"
    database_root = split_root / f"{split}_databases"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"BIRD split file not found: {dataset_path}")
    if not database_root.is_dir():
        raise FileNotFoundError(f"BIRD database directory not found: {database_root}")

    raw_instances = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_instances, list):
        raise ValueError(f"BIRD split must be a JSON array: {dataset_path}")

    instances: list[dict[str, Any]] = []
    db_ids: set[str] = set()
    seen_indices: set[int] = set()

    for position, raw_instance in enumerate(raw_instances):
        if not isinstance(raw_instance, dict):
            raise ValueError(f"BIRD instance at position {position} must be an object")

        missing_fields = REQUIRED_INSTANCE_FIELDS - raw_instance.keys()
        if missing_fields:
            raise ValueError(
                f"BIRD instance at position {position} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        index = raw_instance.get("question_id", position)
        if not isinstance(index, int):
            raise ValueError(f"BIRD instance index at position {position} must be an integer")
        if index in seen_indices:
            raise ValueError(f"Duplicate BIRD instance index: {index}")
        seen_indices.add(index)

        db_id = raw_instance["db_id"]
        question = raw_instance["question"]
        evidence = raw_instance["evidence"]
        if not all(isinstance(value, str) for value in (db_id, question, evidence)):
            raise ValueError(
                f"db_id, question, and evidence must be strings at position {position}"
            )

        instances.append(
            {
                "index": index,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
            }
        )
        db_ids.add(db_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    instance_output_path = output_dir / f"bird_{split}.json"
    _write_json_atomic(instance_output_path, instances)

    meta_output_root = _normalize_meta_directory_name(output_dir)
    copied_meta_files = 0
    for db_id in sorted(db_ids):
        source_meta_dir = database_root / db_id / "database_description"
        if not source_meta_dir.is_dir():
            raise FileNotFoundError(f"BIRD metadata directory not found: {source_meta_dir}")

        meta_files = sorted(source_meta_dir.glob("*.csv"))
        if not meta_files:
            raise FileNotFoundError(f"No metadata CSV files found: {source_meta_dir}")

        target_meta_dir = meta_output_root / db_id
        target_meta_dir.mkdir(parents=True, exist_ok=True)
        for source_path in meta_files:
            shutil.copy2(source_path, target_meta_dir / source_path.name)
            copied_meta_files += 1

    return {
        "split": split,
        "instance_count": len(instances),
        "database_count": len(db_ids),
        "meta_file_count": copied_meta_files,
        "instance_output_path": str(instance_output_path),
        "meta_output_path": str(meta_output_root),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _normalize_meta_directory_name(output_dir: Path) -> Path:
    meta_output_root = output_dir / "meta"
    entry_names = {entry.name for entry in output_dir.iterdir()}
    if "Meta" not in entry_names:
        return meta_output_root
    if "meta" in entry_names:
        raise FileExistsError(f"Both Meta and meta exist in output directory: {output_dir}")

    legacy_path = output_dir / "Meta"
    temporary_path = output_dir / ".meta_case_rename"
    if temporary_path.exists():
        raise FileExistsError(f"Temporary metadata path already exists: {temporary_path}")
    legacy_path.rename(temporary_path)
    temporary_path.rename(meta_output_root)
    return meta_output_root
