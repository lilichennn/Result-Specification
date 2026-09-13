"""Read-only accounting for a fully-issued chain of draining capacity probes."""
import json
from pathlib import Path
import sqlite3

class DrainingProbe:
    def __init__(self, run, _seen=None):
        # core initializes the baseline import path before loading RunStore.
        from .core import RunStore
        self.path = Path(run).resolve()
        seen = set() if _seen is None else _seen
        if self.path in seen:
            raise ValueError('cyclic handoff source chain')
        self.parent = None
        self.db = sqlite3.connect((self.path/'run.sqlite3').as_uri()+'?mode=ro', uri=True, timeout=2)
        self.db.row_factory = sqlite3.Row
        self.cursor = 0
        self.requests, self.terminals = set(), set()
        try:
            # Reuse the authoritative decoder without opening a writer or
            # modifying RunStore. It verifies schema version and payload hash.
            manifest_json, _ = RunStore._read_manifest(self.path/'run.sqlite3')
            self.manifest = json.loads(manifest_json)
            if self.manifest.get('kind') != 'prechange_sdk_capacity_probe':
                raise ValueError('handoff source must be an independent capacity probe')
            self._read(initial=True)
            expected = self.manifest['budget']['max_requests']
            if type(expected) is not int or expected <= 0 or self.requests != set(range(expected)):
                raise ValueError('handoff source must have issued its entire finite request budget')
            parent = (self.manifest.get('handoff') or {}).get('source_run')
            if parent:
                self.parent = DrainingProbe(parent, seen | {self.path})
                if any(self.parent.manifest.get(k) != self.manifest.get(k)
                       for k in ('model', 'endpoint', 'workload_sha256')):
                    raise ValueError('handoff ancestors must share endpoint, model and workload')
        except BaseException:
            if self.parent:
                self.parent.close()
            self.db.close()
            raise

    def _read(self, initial=False):
        from .core import RunStore
        rows = self.db.execute("select * "
            "from events where event_id>? and kind in ('request','response','error') order by event_id",
            (self.cursor,))
        for row in rows:
            # Check both metadata and payload hashes before a terminal can
            # release capacity. Only newly consumed records are decoded.
            event = RunStore._event_dict(row)
            event_id, kind, number = event['event_id'], event['kind'], event['payload'].get('request_no')
            if type(number) is not int or number < 0:
                raise ValueError('invalid handoff request number')
            if kind == 'request':
                if not initial or number in self.requests:
                    raise ValueError('handoff source unexpectedly admitted another request')
                self.requests.add(number)
            else:
                if number not in self.requests or number in self.terminals:
                    raise ValueError('handoff source has unpaired or duplicate terminal records')
                self.terminals.add(number)
            self.cursor = event_id

    def pending(self):
        # A completed SDK call whose terminal has not committed still occupies a
        # reservation. With no further old admissions, this is conservative.
        self._read()
        return len(self.requests)-len(self.terminals)+(self.parent.pending() if self.parent else 0)

    def source_runs(self):
        return [str(self.path), *(self.parent.source_runs() if self.parent else [])]

    def close(self):
        if self.parent:
            self.parent.close()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
