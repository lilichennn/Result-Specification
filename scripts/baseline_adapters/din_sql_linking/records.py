"""Durable two-node records for the DIN-SQL RC3 linking campaign."""
from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading

from scripts.baseline_adapters.deepeye.run_store import (
    RunStore,
    restore_jsonable,
    to_jsonable,
)
from scripts.baseline_adapters.din_sql.inputs import TaskKey, digest


NODES = ("schema_filter_rc3", "linking_rc3")
OUTPUT_NODES = NODES

def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(to_jsonable(value), stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    return restore_jsonable(json.loads(Path(path).read_text()))


class LinkingRecords:
    """Append-only per-question records backed by one RunStore per group."""

    def __init__(self, root, manifest, *, read_only=False):
        self.root = Path(root)
        self.manifest = manifest
        self.read_only = read_only
        self.stores = {}
        self.rows = {}
        self.views = {}
        self.latest = {}
        self.pending = {}
        self._lock = None
        self._mutex = threading.RLock()
        self._request_locks = {}
        try:
            manifest_path = self.root / "manifest.json"
            if read_only:
                if _read_json(manifest_path) != manifest:
                    raise ValueError("Read-only manifest differs")
            else:
                self.root.mkdir(parents=True, exist_ok=True)
                self._lock = (self.root / ".writer.lock").open("a+")
                try:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("DIN Linking batch already has a writer") from exc
                if manifest_path.exists() and _read_json(manifest_path) != manifest:
                    raise ValueError("Frozen manifest differs")
                if not manifest_path.exists():
                    _write_json(manifest_path, manifest)

            for group in manifest["groups"]:
                directory = self.root / f"group-{group}"
                identity = {
                    "format": manifest["format"],
                    "group": group,
                    "batch_fingerprint": digest(manifest),
                }
                if read_only and not directory.exists():
                    raise FileNotFoundError(directory)
                store = (
                    RunStore.open(directory, expected_manifest=identity, read_only=read_only)
                    if directory.exists()
                    else RunStore.create(directory, identity)
                )
                self.stores[group] = store
                for row in store.attempts():
                    row["group"] = group
                    version = row["attempt_id"]
                    self.rows[version] = row
                    key = TaskKey(group, row["item_key"])
                    index = self.latest if row["status"] in ("succeeded", "failed") else self.pending
                    previous = index.get(key)
                    if previous is None or self.rows[previous]["attempt_no"] < row["attempt_no"]:
                        index[key] = version
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def key(self, version):
        row = self.rows[version]
        return TaskKey(row["group"], row["item_key"])

    def request_lock(self, version, node):
        """Share one async critical section across all requesters for a node."""

        with self._mutex:
            if version not in self.rows or node not in NODES:
                raise ValueError("Unknown request node")
            return self._request_locks.setdefault((version, node), asyncio.Lock())

    def current(self, key):
        return self.latest.get(key)

    def unfinished(self, key):
        return self.pending.get(key)

    def begin(self, key, *, parent_version=None):
        with self._mutex:
            if self.read_only:
                raise ValueError("Read-only records")
            try:
                known_ids = self.manifest["groups"][key.group]["ids"]
            except (AttributeError, KeyError) as exc:
                raise ValueError("Unknown task") from exc
            if key.question_id not in known_ids:
                raise ValueError("Unknown task")
            if self.unfinished(key) is not None:
                raise ValueError("Resume unfinished version instead")
            if parent_version is not None and self.key(parent_version) != key:
                raise ValueError("Parent belongs to another question")
            store = self.stores[key.group]
            version = store.begin_attempt(
                key.question_id, "din_linking_question", digest(self.manifest)
            )
            self.rows[version] = {**store.attempt(version), "group": key.group}
            self.pending[key] = version
            self.views[version] = {
                "nodes": {},
                "inputs": {},
                "attempts": {},
                "outcomes": {},
                "history_loaded": True,
            }
            self.append(version, "question_start", {"parent_version": parent_version})
            return version

    def _view(self, version):
        with self._mutex:
            if version not in self.views:
                view = {
                    "nodes": {},
                    "inputs": {},
                    "attempts": {},
                    "outcomes": {},
                    "history_loaded": False,
                }
                self.views[version] = view
                row = self.rows[version]
                if row["status"] in ("succeeded", "failed"):
                    view["nodes"] = self._persisted_nodes(
                        version, expected_summary=row["payload"]
                    )
                    return view
                for event in self.stores[row["group"]].iter_events(
                    version, kinds=("question_start", "node_result")
                ):
                    ref = {
                        "group": row["group"],
                        "attempt_id": version,
                        "event_no": event["event_no"],
                    }
                    self._index(view, event["kind"], event["payload"], ref)
            return self.views[version]

    def view(self, version):
        """Return an isolated snapshot; callers may not mutate cached state."""

        with self._mutex:
            return copy.deepcopy(self._view(version))

    def _request_history(self, version):
        with self._mutex:
            view = self._view(version)
            if not view["history_loaded"]:
                row = self.rows[version]
                view.update(inputs={}, attempts={}, outcomes={})
                for event in self.stores[row["group"]].iter_events(
                    version,
                    kinds=("node_input", "request_attempt", "request_outcome"),
                ):
                    ref = {
                        "group": row["group"],
                        "attempt_id": version,
                        "event_no": event["event_no"],
                    }
                    self._index(view, event["kind"], event["payload"], ref)
                view["history_loaded"] = True
            return view

    def request_history(self, version):
        """Return an isolated snapshot of durable request provenance."""

        with self._mutex:
            return copy.deepcopy(self._request_history(version))

    @staticmethod
    def _index(view, kind, payload, ref):
        node = payload.get("node")
        if kind == "question_start":
            view["parent_version"] = payload.get("parent_version")
        elif kind == "node_result":
            view["nodes"][node] = {**payload, "ref": ref}
        elif kind == "node_input":
            view["inputs"][node] = {
                "input_fingerprint": payload["input_fingerprint"],
                "ref": ref,
            }
        elif kind == "request_attempt":
            view["attempts"].setdefault(node, []).append({**payload, "ref": ref})
        elif kind == "request_outcome":
            view["outcomes"].setdefault(node, []).append({**payload, "ref": ref})

    def append(self, version, kind, payload):
        with self._mutex:
            view = self._view(version)
            row = self.rows[version]
            event_no = self.stores[row["group"]].append_event(version, kind, payload)
            ref = {
                "group": row["group"],
                "attempt_id": version,
                "event_no": event_no,
            }
            self._index(view, kind, payload, ref)
            return ref

    def read_ref(self, ref):
        return self.stores[ref["group"]].event(
            ref["attempt_id"], ref["event_no"]
        )["payload"]

    def node(self, version, node):
        with self._mutex:
            return copy.deepcopy(self._view(version)["nodes"].get(node))

    def save_node(self, version, node, result):
        with self._mutex:
            if node not in NODES or result.get("status") not in (
                "succeeded",
                "failed",
                "dependency_failed",
            ):
                raise ValueError("Invalid terminal node")
            if self._view(version)["nodes"].get(node) is not None:
                raise ValueError("Node is already terminal")
            return self.append(version, "node_result", {**result, "node": node})

    @staticmethod
    def _validate_nodes(nodes):
        if set(nodes) != set(OUTPUT_NODES):
            raise ValueError("Exactly two target outcomes are required before seal")
        filtered = nodes["schema_filter_rc3"]
        linked = nodes["linking_rc3"]
        if filtered.get("status") not in ("succeeded", "failed"):
            raise ValueError("Invalid schema-filter terminal status")
        if linked.get("status") not in ("succeeded", "failed", "dependency_failed"):
            raise ValueError("Invalid Linking terminal status")
        for node, value in nodes.items():
            if not isinstance(value.get("input_fingerprint"), str) or not value["input_fingerprint"]:
                raise ValueError(f"Missing input fingerprint for {node}")
            if not isinstance(value.get("parent_refs"), dict):
                raise ValueError(f"Invalid parent_refs for {node}")
        if filtered["parent_refs"]:
            raise ValueError("Schema filter cannot have parent refs")
        expected_parent = {"schema_filter_rc3": filtered["ref"]}
        if linked["parent_refs"] != expected_parent:
            raise ValueError("Invalid Linking parent ref")
        if filtered["status"] == "succeeded":
            if linked["status"] == "dependency_failed":
                raise ValueError("Invalid dependency topology after successful filter")
        elif linked["status"] != "dependency_failed":
            raise ValueError("Invalid dependency topology after failed filter")

    def _persisted_nodes(self, version, *, expected_summary=None):
        """Rebuild and validate the node chain only from verified event rows."""

        row = self.rows[version]
        nodes = {}
        for event in self.stores[row["group"]].iter_events(
            version, kinds=("node_result",)
        ):
            payload = event["payload"]
            node = payload.get("node")
            if node not in NODES or node in nodes:
                raise ValueError("Invalid or duplicate durable node result")
            ref = {
                "group": row["group"],
                "attempt_id": version,
                "event_no": event["event_no"],
            }
            nodes[node] = {**payload, "ref": ref}
        self._validate_nodes(nodes)
        if expected_summary is not None:
            summaries = expected_summary.get("nodes") if isinstance(expected_summary, dict) else None
            expected = {
                node: {"status": value["status"], "ref": value["ref"]}
                for node, value in nodes.items()
            }
            if summaries != expected:
                raise ValueError("Finish summary disagrees with durable node results")
        return nodes

    def seal(self, version):
        with self._mutex:
            nodes = self._persisted_nodes(version)
            history = self._request_history(version)
            for node, request_input in history["inputs"].items():
                if node in nodes and nodes[node]["input_fingerprint"] != request_input["input_fingerprint"]:
                    raise ValueError(f"Node result input fingerprint differs for {node}")
            state = (
                "succeeded"
                if all(nodes[node]["status"] == "succeeded" for node in OUTPUT_NODES)
                else "failed"
            )
            row = self.rows[version]
            summary = {
                "nodes": {
                    node: {"status": value["status"], "ref": value["ref"]}
                    for node, value in nodes.items()
                }
            }
            self.stores[row["group"]].finish_attempt(version, state, summary)
            row.update(status=state, payload=summary)
            self.views[version]["nodes"] = copy.deepcopy(nodes)
            key = self.key(version)
            self.latest[key] = version
            self.pending.pop(key, None)

    def current_rows(self):
        return [copy.deepcopy(self.rows[version]) for version in self.latest.values()]


DinLinkingRecords = LinkingRecords
