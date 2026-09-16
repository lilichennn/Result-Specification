"""One-time offline binding of five workloads to public Meta, RC3, and training pools."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.baseline_adapters.dail_sql.config import DailSettings
from scripts.rc_evaluation.dail_sql.contracts import select_rc3


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*.csv") if path.is_file())
    if not files:
        raise ValueError("public Meta has no CSV files")
    for path in files:
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _identity(row: dict, id_field: str, db_field: str) -> tuple[str, str, str, str]:
    if row.get(id_field) is None or not row.get(db_field) or not row.get("question"):
        raise ValueError("question identity is incomplete")
    return (str(row[id_field]), str(row[db_field]), row["question"], row.get("evidence", ""))


def _unique(rows: list[dict], id_field: str, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if row.get(id_field) is None:
            raise ValueError(f"{label} row has no ID")
        key = str(row[id_field])
        if key in result:
            raise ValueError(f"{label} has a normalized duplicate ID: {key}")
        result[key] = row
    return result


def _evaluation_rows(path: Path, group: str, targets: list[dict]) -> dict[str, dict]:
    source = _json(path)
    if not isinstance(source, list) or len(source) != len(targets):
        raise ValueError(f"{group} evaluation source count differs from target count")
    id_field = "index" if group.startswith("bird_interact") else "question_id" if group == "bird_dev" else None
    if id_field:
        return _unique(source, id_field, f"{group} evaluation")
    # Spider references have no ID; benchmark order is preserved and identity checked below.
    return {str(target["index"]): row for target, row in zip(targets, source)}


def _reference_sql(row: dict) -> str:
    value = row.get("gold_sql", row.get("SQL", row.get("query")))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evaluation reference SQL is absent")
    return value


def _training_pools(root: Path, config: dict) -> dict:
    pools = {}
    for pool_name, pool in config["training_pools"].items():
        resource_refs = {}
        for field in ("schema_sources", "database_roots"):
            values = pool.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"{pool_name} requires explicit {field}")
            paths = [_path(root, value) for value in values]
            for path in paths:
                if not (path.exists() if field == "schema_sources" else path.is_dir()):
                    raise ValueError(f"{pool_name} {field} source is absent: {path}")
            resource_refs[field] = [str(path) for path in paths]
        examples = []
        sources = []
        for source_name in pool["sources"]:
            source_path = _path(root, source_name)
            rows = _json(source_path)
            if not isinstance(rows, list):
                raise ValueError("training source must be a JSON array")
            sources.append({"path": str(source_path), "sha256": _sha256(source_path), "count": len(rows)})
            for position, row in enumerate(rows):
                sql = row.get("query", row.get("SQL"))
                if not isinstance(sql, str) or not sql.strip() or not row.get("question") or not row.get("db_id"):
                    raise ValueError("incomplete training example")
                examples.append({"example_id": f"{pool_name}/{source_path.name}/{position}",
                                 "source": str(source_path), "source_row": position,
                                 "database_id": row["db_id"], "question": row["question"],
                                 "evidence": row.get("evidence", ""), "sql": sql})
        pools[pool_name] = {"sources": sources, "examples": examples, **resource_refs}
    return pools


def build_manifest(workspace_root: Path, config: dict) -> dict:
    """Read every source once; return frozen references without credentials or physical schema."""
    root = Path(workspace_root).resolve()
    if config.get("format") != "dail-sql-inputs-v1":
        raise ValueError("unsupported DAIL input format")
    settings = DailSettings(**config["settings"]).validate()
    groups = {}
    group_order = []
    pools = _training_pools(root, config)
    for item in config["groups"]:
        group = item["name"]
        if group in groups or item["training_pool"] not in pools:
            raise ValueError("duplicate group or unknown training pool")
        if not isinstance(item["compute_cv_link"], bool):
            raise ValueError("compute_cv_link must be explicit per group")
        paths = {key: _path(root, item[key]) for key in
                 ("questions", "rc", "snapshot", "meta", "evaluation")}
        questions = _json(paths["questions"])
        rc_rows = _json(paths["rc"])
        if not isinstance(questions, list) or not isinstance(rc_rows, list):
            raise ValueError("question and RC files must contain JSON arrays")
        snapshot = _json(paths["snapshot"])
        if snapshot.get("format") != "structured_dataset_snapshot" or snapshot.get("version") != 1:
            raise ValueError("unsupported DeepEye snapshot version")
        if len(questions) != item["expected_count"] or len(rc_rows) != len(questions) or snapshot["num_items"] != len(questions):
            raise ValueError(f"{group} source counts differ")
        question_by_id = _unique(questions, "index", f"{group} questions")
        rc_by_id = _unique(rc_rows, "index", f"{group} RC")
        if set(question_by_id) != set(rc_by_id):
            raise ValueError(f"{group} RC IDs differ from questions")
        snapshot_file = paths["snapshot"].parent / snapshot["snapshot_root"] / "items.jsonl"
        snapshot_by_id = {}
        with snapshot_file.open(encoding="utf-8") as stream:
            for line in stream:
                source = json.loads(line)["input"]
                key = str(source["instance_id"] if group.startswith("bird_interact")
                          else source["question_id"])
                if key in snapshot_by_id:
                    raise ValueError(f"{group} snapshot has duplicate ID {key}")
                snapshot_by_id[key] = source
        if set(snapshot_by_id) != set(question_by_id):
            raise ValueError(f"{group} DeepEye snapshot IDs differ from questions")
        gold_by_id = _evaluation_rows(paths["evaluation"], group, questions)
        if set(gold_by_id) != set(question_by_id):
            raise ValueError(f"{group} evaluation IDs differ from questions")
        bound_rows = []
        for raw in questions:
            key = str(raw["index"])
            source = snapshot_by_id[key]
            rc = select_rc3(rc_by_id[key])
            gold = gold_by_id[key]
            identity = _identity(raw, "index", "db_id")
            if (key, rc["database_id"], rc["question"], rc["evidence"]) != identity:
                raise ValueError(f"{group}/{key} RC3 identity differs")
            if (key, source["database_id"], source["question"], source.get("evidence", "")) != identity:
                raise ValueError(f"{group}/{key} DeepEye snapshot identity differs")
            if gold.get("db_id", gold.get("selected_database")) != raw["db_id"]:
                raise ValueError(f"{group}/{key} evaluation database differs")
            if gold.get("question") != raw["question"]:
                raise ValueError(f"{group}/{key} evaluation question differs")
            if gold.get("evidence", raw.get("evidence", "")) != raw.get("evidence", ""):
                raise ValueError(f"{group}/{key} evaluation evidence differs")
            db_id = raw["db_id"]
            schema_dir = paths["meta"] / db_id
            if not schema_dir.is_dir() or not any(schema_dir.glob("*.csv")):
                raise ValueError(f"{group}/{key} public Meta is missing")
            db_type = source["database_schema"]["db_type"]
            if db_type == "postgresql":
                if source["database_path"] != db_id:
                    raise ValueError(f"{group}/{key} PostgreSQL database binding differs")
                db_path = None
            elif db_type == "sqlite":
                db_path = str(Path(source["database_path"]).resolve())
                if not Path(db_path).is_file():
                    raise ValueError(f"{group}/{key} SQLite database file is absent")
            else:
                raise ValueError(f"{group}/{key} unsupported database dialect")
            database_binding = {"dialect": db_type, "database_id": db_id, "path": db_path}
            bound_rows.append({"group": group, "question_id": key, "original_id": raw["index"],
                               "question": raw["question"], "evidence": raw.get("evidence", ""),
                               "schema_ref": str(schema_dir),
                               "database": database_binding,
                               "rc3_ref": {"source": str(paths["rc"]), "question_id": key,
                                           "database_id": db_id,
                                           "content": rc["rc_round3"]},
                               "training_pool": item["training_pool"],
                               "evaluation_binding": {"reference_sql": _reference_sql(gold),
                                                      "source": str(paths["evaluation"]),
                                                      "database_id": db_id,
                                                      "database": dict(database_binding),
                                                      "comparison_rule": "compare_results"}})
        source_bindings = {key: {"path": str(path), "sha256": _sha256(path)}
                           for key, path in paths.items() if path.is_file()}
        source_bindings["meta"] = {"path": str(paths["meta"]), "sha256": _sha256_tree(paths["meta"])}
        source_bindings["snapshot_items"] = {"path": str(snapshot_file), "sha256": _sha256(snapshot_file)}
        groups[group] = {"count": len(bound_rows), "ids": [row["question_id"] for row in bound_rows],
                         "training_pool": item["training_pool"],
                         "compute_cv_link": item["compute_cv_link"], "rows": bound_rows,
                         "sources": source_bindings}
        group_order.append(group)
    return {"format": "dail-sql-inputs-manifest-v1", "settings": vars(settings),
            "group_order": group_order, "training_pools": pools, "groups": groups}


def runtime_task(row: dict) -> dict:
    """Only the generation-side task fields; never expose evaluation_binding."""
    keys = ("group", "question_id", "original_id", "question", "evidence", "schema_ref",
            "database", "rc3_ref", "training_pool")
    if any(key not in row for key in keys):
        raise ValueError("incomplete runtime task")
    return {key: row[key] for key in keys}


def validate_targets(manifest: dict, group: str, ids: list[str]) -> list[str]:
    if not ids:
        raise ValueError("target ID list cannot be empty")
    if group not in manifest["groups"]:
        raise ValueError(f"unknown experiment group: {group}")
    rows = manifest["groups"][group]["rows"]
    actual = [row["question_id"] for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(manifest["groups"][group]["ids"]):
        raise ValueError("conflicting manifest identities")
    normalized = [str(value) for value in ids]
    if len(normalized) != len(set(normalized)) or set(normalized) - set(actual):
        raise ValueError("duplicate or unknown target ID")
    selected = {row["question_id"]: row for row in rows if row["question_id"] in normalized}
    group_binding = manifest["groups"][group]
    sources = group_binding.get("sources", {})
    rc_source = sources.get("rc", {}).get("path")
    meta_source = sources.get("meta", {}).get("path")
    if not rc_source or not meta_source:
        raise ValueError(f"{group} frozen RC/public Meta source binding is missing")
    for key in normalized:
        row = selected[key]
        rc = row.get("rc3_ref")
        db = row.get("database")
        gold = row.get("evaluation_binding")
        if (row.get("group") != group or row.get("original_id") is None or
                str(row["original_id"]) != key or not isinstance(rc, dict) or
                rc.get("question_id") != key or
                rc.get("source") != rc_source or
                row.get("training_pool") != group_binding.get("training_pool")):
            raise ValueError(f"{group}/{key} selected row identity conflicts with manifest")
        db_id = db.get("database_id") if isinstance(db, dict) else None
        if (not isinstance(db, dict) or not isinstance(gold, dict) or
                not isinstance(db_id, str) or not db_id or
                rc.get("database_id") != db_id or
                db != gold.get("database") or
                db_id != gold.get("database_id") or
                row.get("schema_ref") != str(Path(meta_source) / db_id)):
            raise ValueError(f"{group}/{key} selected database/RC3/evaluation binding conflicts")
    return normalized


def write_manifest(workspace_root: Path, manifest: dict) -> Path:
    """Atomically publish a synced content-addressed artifact without clobbering."""
    if manifest.get("format") != "dail-sql-inputs-manifest-v1":
        raise ValueError("unsupported manifest format")
    encoded = (json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    preparation_id = "inputs-v1-" + hashlib.sha256(encoded).hexdigest()[:24]
    destination = (Path(workspace_root).resolve() / "baselines_reproduce" / "dail_sql" /
                   "prepared" / preparation_id / "inputs_manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".inputs-manifest-", suffix=".tmp",
                                                   dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise ValueError("content-addressed preparation manifest conflicts with existing artifact")
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
