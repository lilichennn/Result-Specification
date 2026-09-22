"""Incremental local DAIL features, shared training pools and first prompts.

Artifacts publish atomically, with per-stage checksummed completion records.
Production readers need neither CoreNLP nor MPNet. The sole prompt renderer is
Task5's build_prompt(task_with_schema, examples); it must exist before preparation.
"""

from contextlib import ExitStack
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from .inputs import runtime_task
from . import native
from .retrieval import choose_examples, distance_order
from .tokenizer import LocalCoreNLP


ALGORITHM_VERSION = "dail-local-preparation-v1"
ROOT = Path(__file__).resolve().parents[3]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if isinstance(value, np.ndarray):
                np.save(stream, value, allow_pickle=False)
            else:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _mark(directory, marker, names):
    _publish(directory / marker, {name: _file_hash(directory / name) for name in names})


def _complete(directory, marker, names=None):
    path = directory / marker
    if not path.is_file():
        return False
    try:
        checksums = _json(path)
        if names is not None:
            if not set(names) <= checksums.keys():
                return False
            checksums = {name: checksums[name] for name in names}
        return all((directory / name).is_file() and _file_hash(directory / name) == checksum
                   for name, checksum in checksums.items())
    except (ValueError, OSError):
        return False


def _prompt_builder(resources):
    builder = resources.get("prompt_builder")
    if builder is not None:
        if not resources.get("prompt_builder_version"):
            raise ValueError("Explicit prompt_builder requires a version for cache identity")
        return builder, resources["prompt_builder_version"]
    try:
        from .prompts import build_prompt
    except ImportError as exc:
        raise RuntimeError("Task5 prompt builder is missing; full preparation is not ready") from exc
    return build_prompt, _file_hash(Path(__file__).with_name("prompts.py"))


def _load_schemas(manifest):
    public, training, fingerprints = {}, {}, {}
    for group in manifest["groups"].values():
        for row in group["rows"]:
            ref = row["schema_ref"]
            if ref not in public:
                public[ref] = native.load_public_schema(Path(ref))
                fingerprints[ref] = {str(p.name): _file_hash(p) for p in sorted(Path(ref).glob("*.csv"))}
    for name, pool in manifest["training_pools"].items():
        schemas = {}
        for source in pool["schema_sources"]:
            path = Path(source)
            if path.is_file():
                fingerprints[source] = _file_hash(path)
                for schema in _json(path):
                    schemas[schema["db_id"]] = schema
            elif path.is_dir():
                # BIRD's database_description CSVs are the explicit public
                # fallback for schemas absent from train_tables.json.
                fingerprints[source] = {str(p.relative_to(path)): _file_hash(p)
                                        for p in sorted(path.glob("*/database_description/*.csv"))}
                for db_dir in sorted(path.iterdir()):
                    description = db_dir / "database_description"
                    if db_dir.is_dir() and db_dir.name not in schemas and description.is_dir():
                        schemas[db_dir.name] = native.load_public_schema(description)
            else:
                raise ValueError(f"Training schema source missing: {source}")
        missing = {row["database_id"] for row in pool["examples"]} - schemas.keys()
        if missing:
            raise ValueError(f"Training schemas missing for {name}: {sorted(missing)}")
        training[name] = schemas
    return public, training, fingerprints


def _pool_cv(manifest, name):
    flags = {group["compute_cv_link"] for group in manifest["groups"].values() if group["training_pool"] == name}
    if len(flags) != 1:
        raise ValueError("A shared training pool requires one consistent CV setting")
    return flags.pop()


def _reject_wal(path):
    if Path(str(path) + "-wal").exists():
        raise ValueError(f"CV preparation requires a checkpointed, closed SQLite WAL: {path}")


def _cv_inputs(manifest):
    """Resolve shared training bindings and hash each CV-read database once.

    All manifest CV inputs, not only selected groups, enter the canonical key.
    Existing WAL files are rejected rather than hashing a stale base database.
    CV-off paths are neither inspected nor fingerprinted here.
    """
    training, paths = {}, set()
    for name, pool in manifest["training_pools"].items():
        if not _pool_cv(manifest, name):
            continue
        for database_id in sorted({row["database_id"] for row in pool["examples"]}):
            candidates = [Path(root) / database_id / (database_id + ".sqlite") for root in pool["database_roots"]]
            path = next((path.resolve() for path in candidates if path.is_file()), None)
            if path is None:
                raise ValueError(f"SQLite CV database file missing: {name}/{database_id}")
            training[name, database_id] = {"dialect": "sqlite", "database_id": database_id, "path": str(path)}
            paths.add(path)
    for group in manifest["groups"].values():
        if group["compute_cv_link"]:
            for row in group["rows"]:
                database = row["database"]
                if database["dialect"] != "sqlite" or not database["path"]:
                    raise ValueError("CV linking requires an explicit SQLite database")
                paths.add(Path(database["path"]).resolve())
    fingerprints = {}
    for path in sorted(paths):
        _reject_wal(path)
        before = path.stat()
        fingerprints[str(path)] = _file_hash(path)
        _reject_wal(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"SQLite CV database changed while fingerprinting: {path}")
    return training, fingerprints


def _group_dependencies_complete(directory, group, plan=None, pool_ready=None):
    plan = plan if plan is not None else _json(directory / "plan.json")
    dependency = plan.get("group_dependencies", {}).get(group)
    if dependency is None:
        return False
    pool = dependency["training_pool"]
    ready = (pool_ready[pool] if pool_ready is not None else
             _complete(directory / "pools" / pool, "ready.json"))
    return ready and _complete(directory, "schemas.ready.json", dependency["schemas"])


def status_preparation(directory: Path) -> dict:
    directory = Path(directory).resolve()
    plan = _json(directory / "plan.json")
    pools = {name: {"status": "ready" if _complete(directory / "pools" / name, "ready.json") else "pending"}
             for name in plan["pools"]}
    pool_ready = {name: state["status"] == "ready" for name, state in pools.items()}
    groups = {name: {"status": "ready" if (_complete(directory / "groups" / name, "ready.json") and
              _group_dependencies_complete(directory, name, plan, pool_ready)) else "pending"}
              for name in plan["groups"]}
    ready = (_complete(directory, "schemas.ready.json") and
             all(item["status"] == "ready" for item in [*pools.values(), *groups.values()]))
    return {"status": "ready" if ready else "pending", "directory": str(directory),
            "training_pools": pools, "groups": groups, "cache_key": plan["cache_key"],
            "schema_diagnostics": plan.get("schema_diagnostics", {}),
            "evaluation_source_ref": "evaluation/source.json"}


def _hydrate_schema(directory, row, schemas):
    ref = row["schema_ref"]
    if ref not in schemas:
        schema = _json(directory / ref)
        if _digest(schema) != Path(ref).stem:
            raise ValueError("Prepared schema failed integrity validation")
        schemas[ref] = schema
    return {**row, "schema": schemas[ref]}


def load_prepared_group(directory: Path, group: str) -> dict[str, dict]:
    """Batch preload: validate once, decode each block/schema once, then point-read."""
    directory = Path(directory)
    group_dir = directory / "groups" / group
    if not (_complete(group_dir, "ready.json") and _group_dependencies_complete(directory, group)):
        raise ValueError("Prepared group is not ready")
    index = _json(group_dir / "index.json")
    blocks, schemas, tasks = {}, {}, {}
    for question_id, ref in index.items():
        if ref["path"] not in blocks:
            blocks[ref["path"]] = _json(directory / ref["path"])
        tasks[question_id] = _hydrate_schema(directory, blocks[ref["path"]][ref["position"]], schemas)
    return tasks


def load_prepared_task(directory: Path, group: str, question_id: str) -> dict:
    directory = Path(directory)
    group_dir = directory / "groups" / group
    if not ((group_dir / "ready.json").is_file() and _group_dependencies_complete(directory, group)):
        raise ValueError("Prepared group is not ready")
    ref = _json(group_dir / "index.json")[str(question_id)]
    # Selected block only: runtime must not rescan a whole prepared group.
    block = (directory / ref["path"]).parent
    if not _complete(block, "done.json"):
        raise ValueError("Selected prepared block failed integrity validation")
    return _hydrate_schema(directory, _json(directory / ref["path"])[ref["position"]], {})


def load_training_pool(directory: Path, pool: str) -> list[dict]:
    directory = Path(directory)
    pool_dir = directory / "pools" / pool
    if not _complete(pool_dir, "ready.json"):
        raise ValueError("Prepared training pool is not ready")
    schemas = {}
    return [_hydrate_schema(directory, row, schemas) for row in _json(pool_dir / "examples.json")]


def prepare(manifest, output: Path, resources: dict) -> dict:
    """Resume one immutable manifest; resources may inject bounded local fixtures.

    Optional fixture boundaries: tokenizer object, encoder(texts)->float32 array,
    prompt_builder(task, examples) + explicit prompt_builder_version. Production
    resources are the verified asset manifest plus nltk_data and optional block_size.
    """
    builder, builder_version = _prompt_builder(resources)
    manifest_source = None
    if isinstance(manifest, (str, Path)):
        manifest_source = {"input_manifest": str(Path(manifest).resolve()), "sha256": _file_hash(manifest)}
        manifest = _json(manifest)
    block_size = resources.get("block_size", 128)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be positive")
    requested = resources.get("groups", manifest["group_order"])
    if not requested or set(requested) - manifest["groups"].keys():
        raise ValueError("Preparation requires known, nonempty selected groups")
    selected_groups = [group for group in manifest["group_order"] if group in requested]
    selected_pools = {manifest["groups"][group]["training_pool"] for group in selected_groups}
    public, training, source_hashes = _load_schemas(manifest)
    cv_training, cv_fingerprints = _cv_inputs(manifest)
    safe_groups = {name: {**group, "rows": [runtime_task(row) for row in group["rows"]]}
                   for name, group in manifest["groups"].items()}
    stopwords = Path(resources.get("nltk_data", ROOT / "cache/dail_sql/assets/nltk_data"))
    identity = {"algorithm": ALGORITHM_VERSION, "input_manifest_hash": _digest(manifest),
                "manifest": {**manifest, "groups": safe_groups},
                "schemas": source_hashes, "native": native.source_fingerprints(), "block_size": block_size,
                "cv_database_sha256": cv_fingerprints,
                "corenlp": resources["corenlp"], "mpnet": resources["mpnet"], "java": resources.get("java"),
                "encoder_device": resources.get("encoder_device", "cpu"),
                "adapter": {name: _file_hash(Path(__file__).with_name(name)) for name in
                            ("native.py", "preparation.py", "retrieval.py", "tokenizer.py", "LoopbackCoreNLP.java")},
                "stopwords": _file_hash(stopwords / "corpora/stopwords/english"), "prompt_builder": builder_version,
                "packages": {name: version(name) for name in ("sql-metadata", "sqlparse", "sqlglot", "nltk", "sentence-transformers", "numpy", "scikit-learn")}}
    key = _digest(identity)
    directory = Path(output).resolve() / ("preparation-" + key[:24])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _publish(directory / "identity.json", identity)
        if manifest_source is None:
            bindings = {group: {row["question_id"]: row["evaluation_binding"] for row in spec["rows"]}
                        for group, spec in manifest["groups"].items()}
            _publish(directory / "evaluation/bindings.json", bindings)
            manifest_source = {"bindings": "evaluation/bindings.json",
                               "sha256": _file_hash(directory / "evaluation/bindings.json")}
        _publish(directory / "evaluation/source.json", manifest_source)
        registry = {_digest(schema): schema for schema in [*public.values(),
                    *(schema for pool in training.values() for schema in pool.values())]}
        for schema_key, schema in registry.items():
            _publish(directory / "schemas" / (schema_key + ".json"), schema)
        _mark(directory, "schemas.ready.json", [f"schemas/{key}.json" for key in registry])
        diagnostics = {ref: schema["unresolved_foreign_keys"] for ref, schema in public.items()
                       if schema.get("unresolved_foreign_keys")}
        diagnostics.update({f"training/{pool}/{db}": schema["unresolved_foreign_keys"]
                            for pool, schemas in training.items() for db, schema in schemas.items()
                            if schema.get("unresolved_foreign_keys")})
        pool_schema_refs = {name: {f"schemas/{_digest(training[name][db])}.json"
                                  for db in {row["database_id"] for row in pool["examples"]}}
                            for name, pool in manifest["training_pools"].items()}
        _publish(directory / "plan.json", {"cache_key": key, "pools": list(training),
            "groups": manifest["group_order"], "schema_diagnostics": diagnostics,
            "group_dependencies": {name: {"training_pool": group["training_pool"], "schemas": sorted(
                pool_schema_refs[group["training_pool"]] | {f"schemas/{_digest(public[ref])}.json"
                    for ref in {row["schema_ref"] for row in group["rows"]}})}
                for name, group in manifest["groups"].items()}})
        tokenizer = resources.get("tokenizer")
        encoder = resources.get("encoder")
        schema_tokens, connections = {}, {}

        def tokens_for(schema):
            nonlocal tokenizer
            if tokenizer is None:
                tokenizer = stack.enter_context(LocalCoreNLP(resources, directory / "corenlp", root=ROOT))
            schema_key = _digest(schema)
            if schema_key not in schema_tokens:
                cache = directory / "schema_tokens" / (schema_key + ".json")
                if cache.is_file():
                    schema_tokens[schema_key] = _json(cache)
                else:
                    schema_tokens[schema_key] = native.tokenize_schema(schema, tokenizer)
                    _publish(cache, schema_tokens[schema_key])
            return schema_tokens[schema_key]

        def encode(texts):
            nonlocal encoder
            if encoder is None:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(resources["mpnet"]["snapshot"],
                    device=resources.get("encoder_device", "cpu"), local_files_only=True)
                encoder = lambda batch: model.encode(batch, batch_size=32, normalize_embeddings=False,
                                                     convert_to_numpy=True, show_progress_bar=False)
            vectors = np.asarray(encoder(texts), dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
                raise ValueError("MPNet returned invalid vectors")
            if "encoder" not in resources and vectors.shape[1] != 768:
                raise ValueError("MPNet must produce 768 dimensions")
            return vectors

        def connection_for(path):
            if not path or not Path(path).is_file():
                raise ValueError("SQLite CV database file missing")
            path = str(Path(path).resolve())
            if path not in connections:
                _reject_wal(path)
                connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
                connection.execute("PRAGMA query_only=ON")
                stack.callback(connection.close)
                connections[path] = connection
            return connections[path]

        def features(row, schema, cv, database):
            tokens = tokens_for(schema)
            question = row["question"]
            if row.get("evidence"):
                question = (question + " " + row["evidence"]).strip()
            if cv and database["dialect"] != "sqlite":
                raise ValueError("Cell linking is SQLite-only")
            linked = native.link_question(question, schema, tokenizer, compute_cv_link=cv,
                connection=connection_for(database["path"]) if cv else None,
                stopwords_path=stopwords, tokenized_schema=tokens)
            return {"mask": native.mask_question(linked), "linking": linked}

        pools = {}
        for name, pool in manifest["training_pools"].items():
            if name not in selected_pools:
                continue
            pool_dir = directory / "pools" / name
            if not _complete(pool_dir, "ready.json"):
                cv = _pool_cv(manifest, name)
                all_examples, vector_blocks = [], []
                for start in range(0, len(pool["examples"]), block_size):
                    block = pool_dir / "blocks" / f"{start:06d}"
                    if not _complete(block, "features.done.json"):
                        examples = []
                        for row in pool["examples"][start:start + block_size]:
                            schema = training[name][row["database_id"]]
                            database = {"dialect": "sqlite", "database_id": row["database_id"], "path": None}
                            if cv:
                                database = cv_training[name, row["database_id"]]
                            example = {**row, "schema_ref": f"schemas/{_digest(schema)}.json", "database": database,
                                       **features(row, schema, cv, database),
                                       "query_skeleton": native.sql_skeleton(row["sql"], schema, "sqlite")}
                            examples.append(example)
                        _publish(block / "examples.json", examples)
                        _mark(block, "features.done.json", ["examples.json"])
                    examples = _json(block / "examples.json")
                    if not _complete(block, "vectors.done.json"):
                        _publish(block / "vectors.npy", encode([row["mask"] for row in examples]))
                        _mark(block, "vectors.done.json", ["vectors.npy"])
                    all_examples.extend(examples)
                    vector_blocks.append(np.load(block / "vectors.npy", allow_pickle=False))
                if not all_examples:
                    raise ValueError("Training pool is empty")
                _publish(pool_dir / "examples.json", all_examples)
                _publish(pool_dir / "ids.json", [row["example_id"] for row in all_examples])
                _publish(pool_dir / "vectors.npy", np.concatenate(vector_blocks))
                _mark(pool_dir, "ready.json", ["examples.json", "ids.json", "vectors.npy"])
            pools[name] = (load_training_pool(directory, name), np.load(pool_dir / "vectors.npy", mmap_mode="r"))

        for name in selected_groups:
            group = manifest["groups"][name]
            group_dir = directory / "groups" / name
            if _complete(group_dir, "ready.json"):
                continue
            examples, train_vectors = pools[group["training_pool"]]
            ids = [row["example_id"] for row in examples]
            by_id = {row["example_id"]: row for row in examples}
            index, ready_files = {}, []
            for start in range(0, len(group["rows"]), block_size):
                block = group_dir / "blocks" / f"{start:06d}"
                rows = group["rows"][start:start + block_size]
                if not _complete(block, "features.done.json"):
                    items = [{"task": runtime_task(row), "schema_ref": f"schemas/{_digest(public[row['schema_ref']])}.json",
                              **features(row, public[row["schema_ref"]], group["compute_cv_link"], row["database"])}
                             for row in rows]
                    _publish(block / "features.json", items)
                    _mark(block, "features.done.json", ["features.json"])
                items = _json(block / "features.json")
                if not _complete(block, "vectors.done.json"):
                    _publish(block / "vectors.npy", encode([item["mask"] for item in items]))
                    _mark(block, "vectors.done.json", ["vectors.npy"])
                vectors = np.load(block / "vectors.npy", allow_pickle=False)
                if not _complete(block, "done.json"):
                    tasks, files = [], []
                    for position, (item, vector) in enumerate(zip(items, vectors)):
                        order = distance_order(train_vectors, vector)
                        selected = choose_examples([ids[i] for i in order], k=manifest["settings"]["k_shot"], qualified_ids=None)
                        order_name = f"order-{position}.npy"
                        prompt_name = f"prompt-{position}.json"
                        _publish(block / order_name, np.array(order, dtype=np.int32))
                        schema = registry[Path(item["schema_ref"]).stem]
                        _publish(block / prompt_name, builder({**item["task"], "schema": schema}, [by_id[i] for i in selected]))
                        tasks.append({**item, "distance_order_ref": str((block / order_name).relative_to(directory)),
                                      "training_ids_ref": f"pools/{group['training_pool']}/ids.json",
                                      "first_example_ids": selected,
                                      "first_prompt_ref": str((block / prompt_name).relative_to(directory))})
                        files.extend([order_name, prompt_name])
                    _publish(block / "tasks.json", tasks)
                    _mark(block, "done.json", ["tasks.json", *files])
                tasks = _json(block / "tasks.json")
                for position, task in enumerate(tasks):
                    index[task["task"]["question_id"]] = {"path": str((block / "tasks.json").relative_to(directory)), "position": position}
                ready_files.extend(str((block / filename).relative_to(group_dir)) for filename in _json(block / "done.json"))
            _publish(group_dir / "index.json", index)
            _mark(group_dir, "ready.json", ["index.json", *ready_files])
        return status_preparation(directory)
