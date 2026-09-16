"""DAIL version provenance on the append-only RunStore.

One process owns each group writer. Results are stored inline using RunStore's
lossless serializer; this API does not accept external attachment references.
"""

import copy
from dataclasses import asdict, astuple
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from .config import DailSettings, MODES, TaskKey


def _encoded(value):
    return json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def aggregate_observed_usage(values: list[dict | None]) -> dict | None:
    """Sum observed numeric leaves; absent sources/fields remain unknown.

    Empty input and any wholly unknown source produce None. Nested fields are
    summed only when every source provides a numeric value for that field.
    """
    if not values:
        return None
    if all(isinstance(value, dict) for value in values):
        return {key: aggregate_observed_usage([value.get(key) for value in values])
                for key in set().union(*(value.keys() for value in values))}
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return sum(values)
    return None


class DailRecords:
    def __init__(self, root: Path, manifest: dict, *, read_only: bool = False):
        self.root = Path(root)
        self._read_only = read_only
        self._manifest = copy.deepcopy(manifest)
        self._batch = manifest.get("batch_id")
        if not isinstance(self._batch, str) or not self._batch.strip():
            raise ValueError("manifest requires a nonempty batch_id")
        groups = manifest.get("groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("manifest requires groups mapping")
        self._members = {}
        for group, binding in groups.items():
            if not isinstance(group, str) or not group.strip() or not isinstance(binding, dict):
                raise ValueError("invalid manifest group")
            ids = binding.get("ids")
            if not isinstance(ids, list) or any(not isinstance(q, str) or not q.strip() for q in ids) or len(set(ids)) != len(ids):
                raise ValueError("manifest group requires unique normalized string ids")
            self._members[group] = frozenset(ids)
        self._settings = DailSettings(**manifest.get("settings", {})).validate()
        data = _encoded(self._manifest)
        self._fingerprint = hashlib.sha256(data).hexdigest()
        manifest_path = self.root / "manifest.json"
        if read_only:
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            if manifest_path.read_bytes() != data:
                raise ValueError("frozen batch manifest does not match")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                # No clobber: a concurrent constructor cannot replace a manifest.
                try:
                    os.link(temporary, manifest_path)
                except FileExistsError:
                    if manifest_path.read_bytes() != data:
                        raise ValueError("frozen batch manifest does not match")
                directory = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                os.unlink(temporary)
        self._stores = {}
        self._indexes = {}
        self._lock = threading.RLock()
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self._lock:
            for store in self._stores.values():
                store.close()
            self._stores.clear()
            self._indexes.clear()
            self._closed = True

    def _membership(self, key):
        if key.batch_id != self._batch or key.question_id not in self._members.get(key.group, ()):
            raise ValueError("task key is not in frozen batch manifest")

    def _store(self, group):
        if self._closed:
            raise RuntimeError("DailRecords is closed")
        if group not in self._members:
            raise ValueError("unknown group")
        if group not in self._stores:
            path = self.root / ("group-" + group.encode().hex())
            metadata = {"format": "dail-records-v1", "batch_id": self._batch, "group": group,
                        "manifest_sha256": self._fingerprint, "manifest_ref": "../manifest.json"}
            self._stores[group] = (RunStore.open(path, expected_manifest=metadata, read_only=self._read_only)
                                   if self._read_only else RunStore.open(path, expected_manifest=metadata)
                                   if path.exists() else RunStore.create(path, metadata))
        return self._stores[group]

    def _version(self, version):
        try:
            group_hex, attempt_id = version.split(":")
            group = bytes.fromhex(group_hex).decode()
            if str(uuid.UUID(attempt_id)) != attempt_id:
                raise ValueError("noncanonical attempt ID")
        except (AttributeError, TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("invalid version ID") from exc
        store = self._store(group)
        attempt = store.attempt(attempt_id)
        key = TaskKey(*json.loads(attempt["item_key"]))
        self._membership(key)
        if key.group != group or attempt["stage"] != "composite" or attempt["input_fingerprint"] != self._fingerprint:
            raise ValueError("version provenance mismatch")
        return store, attempt, key

    def begin_version(self, key: TaskKey) -> str:
        self._assert_writable()
        with self._lock:
            self._membership(key)
            attempt = self._store(key.group).begin_attempt(json.dumps(astuple(key)), "composite", self._fingerprint)
            version = key.group.encode().hex() + ":" + attempt
            self._indexes[version] = self._empty_index()
            return version

    @staticmethod
    def _empty_index():
        return {"sources": {}, "attempts": {}, "results": {}, "verified": {}}

    @staticmethod
    def _index_event(index, event_id, kind, payload):
        field = {"round_result": "round_execution_id", "candidate": "candidate_id", "mode_result": "mode"}.get(kind)
        if field:
            identity = (kind, payload[field])
            if identity in index["sources"]:
                raise ValueError("duplicate source identity")
            index["sources"][identity] = event_id
        elif kind == "request_attempt":
            slot = (payload["round_execution_id"], payload["sample_position"])
            index["attempts"].setdefault(slot, []).append(event_id)
        elif kind == "request_result":
            attempt_id = payload["request_attempt_id"]
            if attempt_id in index["results"]:
                raise ValueError("duplicate request result")
            index["results"][attempt_id] = (event_id, payload["status"])
        # Accepted immutable events only: retain provenance, never response/SQL
        # table bodies. Digests use the same canonical lossless encoding as above.
        if kind == "request_attempt":
            fact = {key: payload.get(key) for key in ("round_execution_id", "sample_position", "attempt_no")}
        elif kind == "request_result":
            fact = {key: payload.get(key) for key in ("request_attempt_id", "status")}
            fact.update(choice_digest=_digest(payload.get("choice")), usage_digest=_digest(payload.get("usage")))
        elif kind == "candidate":
            fact = _digest(payload)
        elif kind == "vote_execution":
            fact = None
        elif kind == "round_result":
            fact = {key: copy.deepcopy(payload[key]) for key in (
                "round_execution_id", "round_no", "status", "rc_injected",
                "actual_parent_round_id", "example_ids", "next_example_ids")}
            fact["selection"] = ({"candidate_id": payload["selection"].get("candidate_id")}
                                 if isinstance(payload["selection"], dict) else None)
        else:
            return
        index["verified"][event_id] = (kind, fact)

    def _index(self, version):
        if version not in self._indexes:
            store, attempt, _ = self._version(version)
            index = self._empty_index()
            # Reopen is the integrity boundary: RunStore verifies every checksum
            # once, then validate consumers in event order against accepted facts.
            # Publish temporarily for validator lookups; discard on any failure.
            self._indexes[version] = index
            try:
                for event in store.iter_events(attempt_id=attempt["attempt_id"]):
                    if event["kind"] == "round_result":
                        self._validate_round(version, event["payload"])
                    elif event["kind"] == "mode_result":
                        self._validate_mode(version, event["payload"])
                    self._index_event(index, version + "#" + str(event["event_no"]), event["kind"], event["payload"])
            except BaseException:
                self._indexes.pop(version, None)
                raise
            if attempt["status"] != "interrupted":
                index["attempts"].clear()
                index["results"].clear()
            self._indexes[version] = index
        return self._indexes[version]

    def _fact(self, version, event_id, kind):
        """Internal compact proof, valid only within this append-only open.

        Public payload reads still verify the on-disk record. Out-of-band edits
        during a writer's lifetime are outside its single-owner contract; reopen
        reconstructs this index from checksummed events, without persisted trust.
        """
        accepted = self._index(version)["verified"].get(event_id)
        if accepted is None or accepted[0] != kind:
            raise ValueError("missing or foreign source event")
        return accepted[1]

    def _round_fact(self, version, identity):
        event_id = self._index(version)["sources"].get(("round_result", identity))
        return self._fact(version, event_id, "round_result")

    def _event(self, version, event_id, kind=None):
        try:
            source_version, sequence = event_id.rsplit("#", 1)
            number = int(sequence)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("invalid source event ID") from exc
        if source_version != version or str(number) != sequence:
            raise ValueError("foreign or noncanonical source event ID")
        store, attempt, _ = self._version(version)
        event = store.event(attempt["attempt_id"], number)
        if kind and event["kind"] != kind:
            raise ValueError("missing or foreign source event")
        return event["payload"]

    def get_event(self, version_id: str, event_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._event(version_id, event_id))

    def version_key(self, version_id: str) -> TaskKey:
        """Return the verified immutable batch/group/question identity."""
        with self._lock:
            return self._version(version_id)[2]

    def request_history(self, version_id: str, round_execution_id: str) -> list[dict]:
        """Five compact per-slot ledgers; payloads remain verified point reads."""
        with self._lock:
            self._version(version_id)
            index = self._index(version_id)
            slots = []
            for position in range(5):
                attempts = []
                for attempt in index["attempts"].get((round_execution_id, position), []):
                    result, status = index["results"].get(attempt, (None, None))
                    attempts.append({"request_attempt_id": attempt,
                                     "request_result_id": result, "status": status})
                slots.append({"sample_position": position, "attempts": attempts})
            return slots

    def events(self, version_id: str, kind: str | None = None) -> list[dict]:
        """Read this version only, for requester recovery and cost accounting."""
        with self._lock:
            store, attempt, _ = self._version(version_id)
            return [{"event_id": version_id + "#" + str(event["event_no"]), "kind": event["kind"], "payload": event["payload"]}
                    for event in store.iter_events(attempt_id=attempt["attempt_id"], kinds=kind)]

    def iter_events(self, version_id: str, kinds=None):
        """Stream verified events within RunStore's bounded observation window."""
        with self._lock:
            store, attempt, _ = self._version(version_id)
        for event in store.iter_events(attempt_id=attempt["attempt_id"], kinds=kinds):
            yield {"event_id": version_id + "#" + str(event["event_no"]),
                   "kind": event["kind"], "payload": event["payload"]}

    def iter_versions(self):
        """Verified attempt metadata, one existing group at a time, including pauses.

        Group attempt lists are observed sequentially, not a batch transaction.
        Missing stores are unstarted groups and are never created here.
        """
        for group in self._members:
            if not (self.root / ("group-" + group.encode().hex())).exists():
                continue
            with self._lock:
                attempts = self._store(group).attempts()
            for attempt in attempts:
                version = group.encode().hex() + ':' + attempt['attempt_id']
                key = self.version_key(version)
                yield {'version_id': version, 'task_key': asdict(key),
                       'attempt_no': attempt['attempt_no'], 'status': attempt['status']}

    def get_version(self, version_id: str) -> dict:
        """Read a sealed composite and hydrate its four canonical mode pointers."""
        with self._lock:
            _, attempt, key = self._version(version_id)
            if not self.is_sealed(key, version_id):
                raise ValueError('version is not sealed')
            result = copy.deepcopy(attempt['payload'])
            result['mode_event_ids'] = result['modes']
            result['modes'] = {mode: copy.deepcopy(self._event(version_id, event, 'mode_result'))
                               for mode, event in result['mode_event_ids'].items()}
            return result

    def _assert_writable(self):
        if self._read_only:
            raise PermissionError('DailRecords is read-only')

    def _source(self, version, kind, field, identity):
        event_id = self._index(version)["sources"].get((kind, identity))
        if event_id is None:
            raise ValueError(f"missing or duplicate {kind} source")
        return self._event(version, event_id, kind)

    def get_round(self, version_id: str, round_execution_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._source(version_id, "round_result", "round_execution_id", round_execution_id))

    def find_source(self, version_id: str, kind: str, identity: str) -> dict | None:
        """Verified point lookup for the three existing indexed source kinds."""
        if kind not in ("round_result", "candidate", "mode_result"):
            raise ValueError("Source kind is not indexed")
        with self._lock:
            self._version(version_id)
            event_id = self._index(version_id)["sources"].get((kind, identity))
            if event_id is None:
                return None
            return {"event_id": event_id, "kind": kind,
                    "payload": copy.deepcopy(self._event(version_id, event_id, kind))}

    def append(self, version_id: str, kind: str, payload: dict) -> str:
        self._assert_writable()
        with self._lock:
            store, attempt, _ = self._version(version_id)
            if attempt["status"] != "interrupted":
                raise ValueError("version is already sealed")
            payload = copy.deepcopy(payload)
            if not isinstance(payload, dict):
                raise ValueError("payload must be a dictionary")
            index = self._index(version_id)
            unique_field = {"round_result": "round_execution_id", "candidate": "candidate_id", "mode_result": "mode"}.get(kind)
            if unique_field:
                identity = payload.get(unique_field)
                if not isinstance(identity, str) or not identity or (kind, identity) in index["sources"]:
                    raise ValueError("source identity is missing or already recorded")
            if kind == "request_attempt":
                rid, pos, number = (payload.get(field) for field in ("round_execution_id", "sample_position", "attempt_no"))
                previous = index["attempts"].get((rid, pos), [])
                finished = (("round_result", rid) in index["sources"] or any(
                    index["results"].get(event_id, (None, None))[1] == "success" for event_id in previous))
                if (not isinstance(rid, str) or not rid or type(pos) is not int or pos not in range(5)
                        or type(number) is not int or number != len(previous) + 1 or number > 5 or finished):
                    raise ValueError("invalid, exhausted, or already successful sampling attempt")
            if kind == "request_result":
                self._event(version_id, payload.get("request_attempt_id"), "request_attempt")
                if payload.get("status") not in ("success", "failed") or payload["request_attempt_id"] in index["results"]:
                    raise ValueError("request result must be a unique terminal outcome")
                if payload["status"] == "success" and not isinstance(payload.get("choice"), dict):
                    raise ValueError("successful request requires its single choice")
            if kind == "round_result":
                self._validate_round(version_id, payload)
            if kind == "mode_result":
                self._validate_mode(version_id, payload)
            number = store.append_event(attempt["attempt_id"], kind, payload)
            event_id = version_id + "#" + str(number)
            self._index_event(index, event_id, kind, payload)
            return event_id

    def _validate_round(self, version, r):
        fields = {"round_execution_id", "round_no", "status", "rc_injected", "actual_parent_round_id", "example_ids", "samples", "request_attempt_ids", "successful_request_ids", "success_usage", "candidates", "selection", "next_example_ids", "error"}
        if not fields <= r.keys() or r["status"] not in ("success", "failed") or r["round_no"] not in (1, 2) or type(r["rc_injected"]) is not bool:
            raise ValueError("invalid round schema")
        if r["round_no"] == 1:
            if r["actual_parent_round_id"] is not None:
                raise ValueError("first round has parent")
        else:
            parent = self._round_fact(version, r["actual_parent_round_id"])
            if parent["round_no"] != 1 or parent["status"] != "success" or r["example_ids"] != parent["next_example_ids"] or r["next_example_ids"]:
                raise ValueError("invalid second-round parent or examples")
        samples = r["samples"]
        if not isinstance(samples, list) or [s.get("sample_position") for s in samples] != list(range(self._settings.samples_per_round)):
            raise ValueError("round must contain five ordered samples")
        attempts, successes, usages = [], [], []
        index = self._index(version)
        for s in samples:
            if not {"status", "request_attempt_ids", "successful_request_id", "success_usage", "error"} <= s.keys() or s["status"] not in ("success", "failed"):
                raise ValueError("invalid sample schema")
            ids = s["request_attempt_ids"]
            if not isinstance(ids, list) or len(ids) > self._settings.max_attempts or len(set(ids)) != len(ids):
                raise ValueError("invalid sample attempt budget")
            recorded = index["attempts"].get((r["round_execution_id"], s["sample_position"]), [])
            if ids != recorded:
                raise ValueError("sample omits or changes persisted attempts")
            retained = [index["results"][source_id][0] for source_id in recorded
                        if index["results"].get(source_id, (None, None))[1] == "success"]
            if retained != ([s["successful_request_id"]] if s["status"] == "success" else []):
                raise ValueError("sample does not match persisted terminal success")
            for number, source_id in enumerate(ids, 1):
                a = self._fact(version, source_id, "request_attempt")
                if (a.get("round_execution_id"), a.get("sample_position"), a.get("attempt_no")) != (r["round_execution_id"], s["sample_position"], number):
                    raise ValueError("foreign round/sample attempt source")
            attempts.extend(ids)
            if s["status"] == "success":
                source = self._fact(version, s["successful_request_id"], "request_result")
                if not ids or source.get("request_attempt_id") != ids[-1] or source.get("status") != "success" or source["choice_digest"] != _digest(s.get("choice")) or source["usage_digest"] != _digest(s["success_usage"]) or (s["success_usage"] is not None and not isinstance(s["success_usage"], dict)):
                    raise ValueError("sample success does not match durable request result")
                successes.append(s["successful_request_id"])
                usages.append(s["success_usage"])
            elif s["successful_request_id"] is not None or s["success_usage"] is not None:
                raise ValueError("failed sample claims success")
        if attempts != r["request_attempt_ids"] or successes != r["successful_request_ids"] or len(set(attempts)) != len(attempts):
            raise ValueError("round request provenance does not match samples")
        for candidate in r["candidates"]:
            event_id = index["sources"].get(("candidate", candidate.get("candidate_id")))
            if _digest(candidate) != self._fact(version, event_id, "candidate"):
                raise ValueError("candidate source mismatch")
            pos = candidate.get("choice_position")
            if type(pos) is not int or pos not in range(5) or samples[pos]["status"] != "success":
                raise ValueError("candidate has no successful sample source")
            source_id = candidate.get("source_request_id")
            if source_id != samples[pos]["successful_request_id"]:
                raise ValueError("candidate references a different successful request")
            choice = samples[pos]["choice"]
            content = choice.get("message", {}).get("content")
            if candidate.get("raw_text") != ("" if content is None else content) or candidate.get("provider_choice_index") != choice.get("index"):
                raise ValueError("candidate raw text or provider index differs from response")
            self._fact(version, candidate.get("vote_execution_ref"), "vote_execution")
        if r["status"] == "failed":
            if r["success_usage"] is not None or not r["error"]:
                raise ValueError("failed round needs error and null aggregate usage")
            return
        if len(successes) != 5 or r["success_usage"] != aggregate_observed_usage(usages) or len(r["example_ids"]) != 9:
            raise ValueError("successful round requires five successes and observed usage")
        if r["round_no"] == 1 and len(r["next_example_ids"]) != 9:
            raise ValueError("successful first round needs nine next examples")
        if len(r["candidates"]) != 5:
            raise ValueError("successful round requires five candidates")
        for pos, candidate in enumerate(r["candidates"]):
            if candidate.get("choice_position") != pos or candidate.get("provider_choice_index") != samples[pos]["choice"].get("index"):
                raise ValueError("candidate source or ordering mismatch")
            if not {"raw_text", "candidate_sql", "vote_sql", "vote_execution_ref"} <= candidate.keys():
                raise ValueError("incomplete candidate")
        if not isinstance(r["selection"], dict) or r["selection"].get("candidate_id") not in [c["candidate_id"] for c in r["candidates"]]:
            raise ValueError("selection must reference a round candidate")

    def _validate_mode(self, version, m):
        if not {"mode", "status", "first_round_id", "second_round_id", "final_candidate_id", "failure_origin"} <= m.keys() or m["mode"] not in MODES or m["status"] not in ("succeeded", "failed", "dependency_failed"):
            raise ValueError("invalid terminal mode")
        rounds = [self._round_fact(version, rid) if rid is not None else None for rid in (m["first_round_id"], m["second_round_id"])]
        for number, r in enumerate(rounds, 1):
            if r is not None:
                rc = m["mode"] in (("rc_first", "rc_both") if number == 1 else ("rc_second", "rc_both"))
                if r["round_no"] != number or r["rc_injected"] != rc:
                    raise ValueError("mode references wrong round or RC condition")
        first, second = rounds
        if second is not None and (first is None or first["status"] != "success" or second["example_ids"] != first["next_example_ids"]):
            raise ValueError("mode second round has different ordered examples")
        if m["status"] == "succeeded":
            if not first or not second or any(r["status"] != "success" for r in rounds) or m["failure_origin"] is not None or m["final_candidate_id"] != second["selection"]["candidate_id"]:
                raise ValueError("mode success requires two successful source rounds")
        elif m["final_candidate_id"] is not None or not m["failure_origin"]:
            raise ValueError("failed mode requires failure origin and no candidate")

    def seal(self, version_id: str, modes: dict) -> str:
        self._assert_writable()
        with self._lock:
            store, attempt, key = self._version(version_id)
            if not isinstance(modes, dict) or set(modes) != set(MODES):
                raise ValueError("seal requires exactly four modes")
            if attempt["status"] != "interrupted":
                if attempt["payload"]["modes"] == modes:
                    return version_id
                raise ValueError("version already sealed differently")
            round_ids = []
            for mode, event_id in modes.items():
                m = self._event(version_id, event_id, "mode_result")
                if m["mode"] != mode:
                    raise ValueError("mode source mismatch")
                self._validate_mode(version_id, m)
                round_ids.extend(rid for rid in (m["first_round_id"], m["second_round_id"]) if rid is not None)
            store.finish_attempt(attempt["attempt_id"], "succeeded", {
                "version_id": version_id, "task_key": asdict(key), "attempt_no": attempt["attempt_no"],
                "round_ids": list(dict.fromkeys(round_ids)), "modes": modes, "sealed": True})
            # Terminal versions retain only compact source-ID lookup metadata.
            self._indexes[version_id]["attempts"].clear()
            self._indexes[version_id]["results"].clear()
            return version_id

    def is_sealed(self, key: TaskKey, version_id: str) -> bool:
        with self._lock:
            try:
                _, attempt, actual_key = self._version(version_id)
            except (ValueError, FileNotFoundError):
                return False
            payload = attempt["payload"]
            return (key == actual_key and attempt["status"] == "succeeded" and isinstance(payload, dict)
                    and payload.get("sealed") is True and payload.get("version_id") == version_id
                    and payload.get("task_key") == asdict(key) and set(payload.get("modes", {})) == set(MODES))
