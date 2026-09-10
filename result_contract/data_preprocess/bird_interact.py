from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any


META_FIELDS = [
    "original_column_name",
    "column_name",
    "column_description",
    "data_format",
    "value_description",
]

_SQL_IDENTIFIER = r'"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_$]*'
_CREATE_TABLE_PATTERN = re.compile(
    rf'(?im)^[ \t]*(?:"CREATE"|CREATE)\s+(?:"TABLE"|TABLE)\s+'
    rf'(?:IF\s+NOT\s+EXISTS\s+)?(?P<table>{_SQL_IDENTIFIER})\s*\('
)
_SAMPLE_HEADER_PATTERN = re.compile(
    r'(?im)^[ \t]*(?:First|"First")\s+3\s+rows:[ \t]*$'
)
_SAMPLE_END_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:\.\.\.|No data available in this table\.)[ \t]*$"
)
_TYPE_END_PATTERN = re.compile(
    r"\s+(?="
    r"NOT\s+NULL\b|NULL\b|DEFAULT\b|PRIMARY\s+KEY\b|REFERENCES\b|"
    r"UNIQUE\b|CHECK\b|COLLATE\b|GENERATED\b"
    r")",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _resolve_variant_root(interact_root: str | Path, variant: str) -> Path:
    root = Path(interact_root).resolve()
    expected_name = f"bird-interact-{variant}"
    return root if root.name == expected_name else root / expected_name


def _validate_database_id(db_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", db_id) is None:
        raise ValueError(f"Database id must be one safe path component: {db_id!r}")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_knowledge(variant_root: Path, db_id: str) -> dict[int, dict[str, Any]]:
    knowledge_path = variant_root / db_id / f"{db_id}_kb.jsonl"
    entries = _read_jsonl(knowledge_path)
    knowledge_by_id: dict[int, dict[str, Any]] = {}
    for entry in entries:
        knowledge_id = entry.get("id")
        if not isinstance(knowledge_id, int):
            raise ValueError(f"Knowledge id must be an integer: {knowledge_path}")
        if knowledge_id in knowledge_by_id:
            raise ValueError(f"Duplicate knowledge id {knowledge_id}: {knowledge_path}")
        knowledge_by_id[knowledge_id] = entry
    return knowledge_by_id


def _format_knowledge(entry: dict[str, Any]) -> str:
    knowledge_id = entry["id"]
    name = entry.get("knowledge", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Knowledge {knowledge_id} must have a non-empty name")
    lines = [f"[{knowledge_id}] {name.strip()}"]
    for label, field in (("Description", "description"), ("Definition", "definition")):
        value = entry.get(field, "")
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Knowledge {knowledge_id} field {field} must be a string")
        if value.strip():
            lines.append(f"{label}: {value.strip()}")
    return "\n".join(lines)


def _build_evidence(
    row: dict[str, Any],
    knowledge_by_id: dict[int, dict[str, Any]],
    missing_references: list[dict[str, Any]],
) -> str:
    raw_references = row.get("external_knowledge", [])
    if not isinstance(raw_references, list):
        raise ValueError(
            f"external_knowledge must be a list for {row.get('instance_id')!r}"
        )

    blocks: list[str] = []
    emitted: set[int] = set()
    missing: set[int] = set()

    def visit(knowledge_id: int, active_path: tuple[int, ...]) -> None:
        if not isinstance(knowledge_id, int):
            raise ValueError(
                f"Knowledge reference must be an integer for {row.get('instance_id')!r}"
            )
        if knowledge_id in emitted:
            return
        if knowledge_id in active_path:
            cycle = " -> ".join(str(value) for value in (*active_path, knowledge_id))
            raise ValueError(
                f"Knowledge dependency cycle for {row.get('instance_id')!r}: {cycle}"
            )
        entry = knowledge_by_id.get(knowledge_id)
        if entry is None:
            if knowledge_id not in missing:
                missing.add(knowledge_id)
                missing_references.append(
                    {
                        "instance_id": row["instance_id"],
                        "db_id": row["selected_database"],
                        "knowledge_id": knowledge_id,
                    }
                )
            return

        children = entry.get("children_knowledge", -1)
        if children not in (-1, None):
            if not isinstance(children, list):
                raise ValueError(
                    f"children_knowledge must be -1 or a list for knowledge {knowledge_id}"
                )
            for child_id in children:
                visit(child_id, (*active_path, knowledge_id))

        emitted.add(knowledge_id)
        blocks.append(_format_knowledge(entry))

    for knowledge_id in raw_references:
        visit(knowledge_id, ())
    return "\n\n".join(blocks)


def _load_livesqlbench_index(
    livesqlbench_root: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    if livesqlbench_root is None:
        raise ValueError("livesqlbench_root is required for the full variant")
    source_path = Path(livesqlbench_root).resolve() / "livesqlbench_data.jsonl"
    source_rows = _read_jsonl(source_path)
    index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(source_rows):
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(
                f"LiveSQLBench instance_id at position {position} must be a non-empty string"
            )
        if instance_id in index:
            raise ValueError(f"Duplicate LiveSQLBench instance_id: {instance_id}")
        index[instance_id] = row
    return index, len(source_rows)


def _unquote_identifier(identifier: str) -> str:
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier[1:-1].replace('""', '"')
    return identifier


def _consume_identifier(text: str) -> tuple[str, str]:
    value = text.lstrip()
    match = re.match(_SQL_IDENTIFIER, value)
    if match is None:
        raise ValueError(f"Expected SQL identifier at: {value[:80]!r}")
    return _unquote_identifier(match.group(0)), value[match.end() :].lstrip()


def _find_matching_parenthesis(text: str, opening_index: int) -> int:
    depth = 0
    quote: str | None = None
    index = opening_index
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                break
        index += 1
    raise ValueError("Unbalanced CREATE TABLE parentheses")


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
        index += 1
    final_part = text[start:].strip()
    if final_part:
        parts.append(final_part)
    return parts


def _is_table_constraint(first_identifier: str, remainder: str) -> bool:
    first = first_identifier.upper()
    if first in {"CONSTRAINT", "CHECK", "EXCLUDE"}:
        return True
    if first in {"PRIMARY", "FOREIGN"}:
        try:
            second, _ = _consume_identifier(remainder)
        except ValueError:
            return False
        return second.upper() == "KEY"
    if first == "UNIQUE":
        return remainder.lstrip().startswith("(")
    return False


def _extract_declared_type(remainder: str) -> str:
    match = _TYPE_END_PATTERN.search(remainder)
    declared_type = remainder[: match.start()] if match else remainder
    return " ".join(declared_type.strip().split())


def _parse_schema_tables(schema_path: Path) -> list[dict[str, Any]]:
    schema_text = schema_path.read_text(encoding="utf-8-sig")
    tables: list[dict[str, Any]] = []
    seen_tables: set[str] = set()
    search_position = 0
    while True:
        match = _CREATE_TABLE_PATTERN.search(schema_text, search_position)
        if match is None:
            break
        table_name = _unquote_identifier(match.group("table"))
        normalized_table = table_name.casefold()
        if normalized_table in seen_tables:
            raise ValueError(f"Duplicate table {table_name!r}: {schema_path}")
        seen_tables.add(normalized_table)

        opening_index = match.end() - 1
        closing_index = _find_matching_parenthesis(schema_text, opening_index)
        sample_header = _SAMPLE_HEADER_PATTERN.search(schema_text, closing_index + 1)
        if sample_header is None:
            raise ValueError(
                f"Missing First 3 rows marker after table {table_name!r}: {schema_path}"
            )
        intervening_create = _CREATE_TABLE_PATTERN.search(
            schema_text, closing_index + 1, sample_header.start()
        )
        if intervening_create is not None:
            raise ValueError(
                f"Missing sample block for table {table_name!r}: {schema_path}"
            )
        sample_end = _SAMPLE_END_PATTERN.search(schema_text, sample_header.end())
        if sample_end is None:
            raise ValueError(
                f"Missing sample-block terminator after table {table_name!r}: "
                f"{schema_path}"
            )

        body = schema_text[opening_index + 1 : closing_index]
        columns: list[dict[str, str]] = []
        seen_columns: set[str] = set()
        for definition in _split_top_level_commas(body):
            if definition.lstrip().startswith("--"):
                continue
            column_name, remainder = _consume_identifier(definition)
            if _is_table_constraint(column_name, remainder):
                continue
            declared_type = _extract_declared_type(remainder)
            if not declared_type:
                raise ValueError(
                    f"Missing declared type for {table_name}.{column_name}: {schema_path}"
                )
            normalized_column = column_name.casefold()
            if normalized_column in seen_columns:
                raise ValueError(
                    f"Duplicate column {table_name}.{column_name}: {schema_path}"
                )
            seen_columns.add(normalized_column)
            columns.append({"name": column_name, "data_format": declared_type})
        if not columns:
            raise ValueError(f"No columns parsed for table {table_name!r}: {schema_path}")
        tables.append({"name": table_name, "columns": columns})
        # Each source table is followed by a sample block. Advancing beyond its
        # terminator prevents CREATE-like cell values from becoming phantom DDL.
        search_position = sample_end.end()
    if not tables:
        raise ValueError(f"No CREATE TABLE statements found: {schema_path}")
    return tables


def _format_column_meaning(value: Any, source_key: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        raise ValueError(f"Column meaning {source_key!r} must be a string or object")
    column_meaning = value.get("column_meaning")
    if not isinstance(column_meaning, str):
        raise ValueError(f"Column meaning {source_key!r} lacks string column_meaning")
    description = column_meaning.strip()
    fields_meaning = value.get("fields_meaning")
    if fields_meaning:
        serialized_fields = json.dumps(
            fields_meaning,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        description = f"{description}\nFields: {serialized_fields}".strip()
    return description


def _load_column_meanings(
    meaning_path: Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    raw_meanings = json.loads(meaning_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw_meanings, dict):
        raise ValueError(f"Column meanings must be a JSON object: {meaning_path}")
    meanings: dict[tuple[str, str], tuple[str, str]] = {}
    for source_key, value in raw_meanings.items():
        if not isinstance(source_key, str):
            raise ValueError(f"Column meaning key must be a string: {meaning_path}")
        key_parts = source_key.split("|")
        if len(key_parts) < 3:
            raise ValueError(f"Invalid column meaning key {source_key!r}: {meaning_path}")
        normalized_key = (key_parts[-2].casefold(), key_parts[-1].casefold())
        if normalized_key in meanings:
            raise ValueError(
                f"Duplicate normalized column meaning {source_key!r}: {meaning_path}"
            )
        meanings[normalized_key] = (
            source_key,
            _format_column_meaning(value, source_key),
        )
    return meanings


def _write_meta_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _generate_meta(
    metadata_root: Path,
    db_ids: set[str],
    meta_output_root: Path,
) -> dict[str, Any]:
    meta_file_count = 0
    column_count = 0
    missing_meanings: list[dict[str, str]] = []
    extra_meanings: list[dict[str, str]] = []

    for db_id in sorted(db_ids):
        database_root = metadata_root / db_id
        schema_path = database_root / f"{db_id}_schema.txt"
        meaning_path = database_root / f"{db_id}_column_meaning_base.json"
        tables = _parse_schema_tables(schema_path)
        meanings = _load_column_meanings(meaning_path)
        consumed_meanings: set[tuple[str, str]] = set()

        for table in tables:
            table_name = table["name"]
            if Path(table_name).name != table_name:
                raise ValueError(f"Unsafe table name {table_name!r}: {schema_path}")
            meta_rows: list[dict[str, str]] = []
            for column in table["columns"]:
                column_name = column["name"]
                normalized_key = (table_name.casefold(), column_name.casefold())
                meaning_record = meanings.get(normalized_key)
                if meaning_record is None:
                    description = ""
                    missing_meanings.append(
                        {
                            "db_id": db_id,
                            "table": table_name,
                            "column": column_name,
                        }
                    )
                else:
                    consumed_meanings.add(normalized_key)
                    _, description = meaning_record
                meta_rows.append(
                    {
                        "original_column_name": column_name,
                        "column_name": "",
                        "column_description": description,
                        "data_format": column["data_format"],
                        "value_description": "",
                    }
                )
            _write_meta_csv(meta_output_root / db_id / f"{table_name}.csv", meta_rows)
            meta_file_count += 1
            column_count += len(meta_rows)

        for normalized_key, (source_key, _) in meanings.items():
            if normalized_key not in consumed_meanings:
                extra_meanings.append({"db_id": db_id, "source_key": source_key})

    return {
        "database_count": len(db_ids),
        "meta_file_count": meta_file_count,
        "column_count": column_count,
        "missing_column_meanings": missing_meanings,
        "extra_column_meanings": extra_meanings,
    }


def _validate_owned_output_directory(output_dir: Path) -> None:
    if output_dir.parent == output_dir:
        raise ValueError("Refusing to use a filesystem root as output_dir")
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")
    allowed_names = {
        ".DS_Store",
        "bird_interact_full.json",
        "bird_interact_lite.json",
        "meta",
        "preprocess_report.json",
    }
    unexpected_names = sorted(
        entry.name for entry in output_dir.iterdir() if entry.name not in allowed_names
    )
    if unexpected_names:
        raise ValueError(
            f"Refusing to replace output_dir with unowned entries: {unexpected_names}"
        )


def _publish_output_tree(staging_dir: Path, output_dir: Path) -> None:
    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.parent / (
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def preprocess_bird_interact(
    interact_root: str | Path,
    variant: str,
    output_dir: str | Path,
    livesqlbench_root: str | Path | None = None,
) -> dict[str, Any]:
    """Preprocess one BIRD-Interact variant for RC generation."""
    if variant not in {"lite", "full"}:
        raise ValueError("variant must be 'lite' or 'full'")

    variant_root = _resolve_variant_root(interact_root, variant)
    source_rows = _read_jsonl(variant_root / "bird_interact_data.jsonl")
    query_rows = [row for row in source_rows if row.get("category") == "Query"]
    livesqlbench_index: dict[str, dict[str, Any]] = {}
    livesqlbench_row_count = 0
    metadata_root = variant_root
    if variant == "full":
        if livesqlbench_root is None:
            raise ValueError("livesqlbench_root is required for the full variant")
        metadata_root = Path(livesqlbench_root).resolve()
        livesqlbench_index, livesqlbench_row_count = _load_livesqlbench_index(
            livesqlbench_root
        )

    knowledge_cache: dict[str, dict[int, dict[str, Any]]] = {}
    missing_references: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    query_join_count = 0
    for position, row in enumerate(query_rows):
        instance_id = row.get("instance_id")
        db_id = row.get("selected_database")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(
                f"BIRD-Interact instance_id at Query position {position} "
                "must be a non-empty string"
            )
        if instance_id in seen_instance_ids:
            raise ValueError(f"Duplicate BIRD-Interact Query instance_id: {instance_id}")
        seen_instance_ids.add(instance_id)
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError(
                f"selected_database for {instance_id!r} must be a non-empty string"
            )
        _validate_database_id(db_id)

        if variant == "full":
            livesqlbench_row = livesqlbench_index.get(instance_id)
            if livesqlbench_row is None:
                raise ValueError(f"Missing LiveSQLBench query for {instance_id}")
            if livesqlbench_row.get("category") != "Query":
                raise ValueError(
                    f"LiveSQLBench row for {instance_id} is not category 'Query'"
                )
            if livesqlbench_row.get("selected_database") != db_id:
                raise ValueError(
                    f"Database mismatch for {instance_id}: BIRD-Interact={db_id!r}, "
                    f"LiveSQLBench={livesqlbench_row.get('selected_database')!r}"
                )
            question = livesqlbench_row.get("query")
            query_join_count += 1
        else:
            question = row.get("query")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question for {instance_id!r} must be a non-empty string")

        if db_id not in knowledge_cache:
            knowledge_cache[db_id] = _load_knowledge(variant_root, db_id)
        instances.append(
            {
                "index": instance_id,
                "db_id": db_id,
                "question": question,
                "evidence": _build_evidence(
                    row,
                    knowledge_cache[db_id],
                    missing_references,
                ),
            }
        )

    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _validate_owned_output_directory(output_dir)
    output_path = output_dir / f"bird_interact_{variant}.json"
    report_path = output_dir / "preprocess_report.json"
    meta_output_root = output_dir / "meta"
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        meta_summary = _generate_meta(
            metadata_root=metadata_root,
            db_ids={instance["db_id"] for instance in instances},
            meta_output_root=staging_dir / "meta",
        )
        _write_json_atomic(
            staging_dir / f"bird_interact_{variant}.json",
            instances,
        )
        _write_json_atomic(
            staging_dir / "preprocess_report.json",
            {
                "variant": variant,
                "source_instance_count": len(source_rows),
                "query_instance_count": len(query_rows),
                "excluded_non_query_count": len(source_rows) - len(query_rows),
                "excluded_follow_up_count": sum(
                    1 for row in source_rows if row.get("follow_up")
                ),
                "written_instance_count": len(instances),
                "selection_policy": "top_level_query_only",
                "follow_up_included": False,
                "evidence_policy": "external_knowledge_with_dependency_closure",
                "question_source": (
                    "livesqlbench_data.jsonl"
                    if variant == "full"
                    else "bird_interact_data.jsonl"
                ),
                "livesqlbench_row_count": livesqlbench_row_count,
                "query_join_count": query_join_count,
                "missing_knowledge_references": missing_references,
                "metadata_source": (
                    "livesqlbench-base-full-v1"
                    if variant == "full"
                    else "bird-interact-lite"
                ),
                "database_count": meta_summary["database_count"],
                "meta_file_count": meta_summary["meta_file_count"],
                "column_count": meta_summary["column_count"],
                "column_meaning_mismatches": {
                    "missing": meta_summary["missing_column_meanings"],
                    "extra": meta_summary["extra_column_meanings"],
                },
            },
        )
        _publish_output_tree(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return {
        "instance_count": len(instances),
        "query_join_count": query_join_count,
        "missing_knowledge_reference_count": len(missing_references),
        "database_count": meta_summary["database_count"],
        "meta_file_count": meta_summary["meta_file_count"],
        "column_count": meta_summary["column_count"],
        "instance_output_path": str(output_path),
        "meta_output_path": str(meta_output_root),
        "report_output_path": str(report_path),
    }
