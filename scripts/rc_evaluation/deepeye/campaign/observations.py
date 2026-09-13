"""Read checksummed outcomes without traversing model-response events."""
from pathlib import Path
from urllib.parse import quote, unquote

from scripts.baseline_adapters.deepeye.run_pipeline import STAGES
from scripts.baseline_adapters.deepeye.run_store import RunStore


def validate_items(items):
    if not isinstance(items, list) or not items:
        raise ValueError('items must be a nonempty list of task keys')
    for key in items:
        if not isinstance(key, str):
            raise ValueError('task keys must be strings')
        parts = key.split('/')
        legacy = len(parts) == 2 and parts[0] in ('lite', 'full') and parts[1] not in ('', '.', '..')
        from scripts.baseline_adapters.deepeye.workloads import SELECTIONS
        typed = False
        if len(parts) == 3 and tuple(parts[:2]) in SELECTIONS:
            suffix = parts[2]
            if suffix.startswith('i:'):
                try:
                    typed = str(int(suffix[2:])) == suffix[2:]
                except ValueError:
                    pass
            elif suffix.startswith('s:') and suffix[2:]:
                try:
                    typed = quote(unquote(suffix[2:], errors='strict'), safe='') == suffix[2:]
                except UnicodeDecodeError:
                    pass
        if (not (legacy or typed) or '\\' in key or key.strip() != key):
            raise ValueError(f'invalid task key: {key!r}')
    if len(set(items)) != len(items):
        raise ValueError('duplicate task keys')
    return list(items)


def manifest_items(manifest):
    bindings = manifest.get('items')
    if not isinstance(bindings, list) or any(not isinstance(row, dict) for row in bindings):
        raise ValueError('run manifest must contain item bindings')
    return validate_items([row.get('task_key') for row in bindings])


def _native_terminal(master, attempts):
    payload = master['payload']
    if not isinstance(payload, dict) or not isinstance(payload.get('stage_attempts'), list):
        raise ValueError('native terminal master has no stage references')
    links = payload['stage_attempts']
    stages = []
    for link in links:
        if not isinstance(link, dict):
            raise ValueError('malformed native stage reference')
        stage, attempt_id = link.get('stage'), link.get('attempt_id')
        if stage not in STAGES or not isinstance(attempt_id, str):
            raise ValueError('invalid native stage reference')
        attempt = attempts.get(attempt_id)
        if attempt is None or attempt['item_key'] != master['item_key'] or attempt['stage'] != stage:
            raise ValueError('native stage reference belongs to another item/stage or is missing')
        stages.append(stage)
        expected = 'failed' if master['status'] == 'failed' and stage == payload.get('failed_stage') else 'succeeded'
        if attempt['status'] != expected:
            raise ValueError('native master disagrees with referenced stage status')
    if master['status'] == 'succeeded':
        if stages != list(STAGES) or payload.get('failed_stage') is not None:
            raise ValueError('native success requires all four succeeded stage references')
    else:
        failed_stage = payload.get('failed_stage')
        if failed_stage not in STAGES or stages != list(STAGES[:STAGES.index(failed_stage) + 1]):
            raise ValueError('native failure requires a complete prefix ending at its failed stage')


class RunObservationReader:
    """Process-owned reader over immutable attempts and append-only finishes.

    Each stored body is decoded on first observation and when its finish is
    appended, not on every polling tick. Existing terminal data are immutable
    by RunStore triggers. A fresh inspection still starts from all records.
    Do not mutate the returned manifest; it belongs to this polling session.
    """
    def __init__(self, run_dir, *, kind, target_stage=None):
        if kind not in ('native', 'rc') or (kind == 'rc' and target_stage not in STAGES):
            raise ValueError('invalid run kind or RC target stage')
        self.store = RunStore.open(Path(run_dir).resolve(), read_only=True)
        try:
            self.manifest = self.store.manifest
            self.items = manifest_items(self.manifest)
            if kind == 'rc' and self.manifest.get('target_stage') != target_stage:
                raise ValueError('RC manifest target stage mismatch')
        except BaseException:
            self.store.close()
            raise
        self.kind, self.target_stage = kind, target_stage
        self._attempt_cursor = self._finish_cursor = 0
        self._attempts = {}

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def read(self):
        store = self.store
        with store._read_snapshot() as connection:
            attempt_cursor = connection.execute('SELECT COALESCE(MAX(rowid),0) FROM attempts').fetchone()[0]
            finish_cursor = connection.execute('SELECT COALESCE(MAX(rowid),0) FROM finishes').fetchone()[0]
            condition = ('WHERE a.rowid IN (SELECT rowid FROM attempts WHERE rowid>? '
                         'UNION SELECT a2.rowid FROM finishes f2 JOIN attempts a2 USING(attempt_id) WHERE f2.rowid>?)')
            args = [self._attempt_cursor, self._finish_cursor]
            if self.kind == 'rc':
                condition += ' AND a.stage=?'
                args.append(self.target_stage)
            rows = connection.execute(store._attempt_query(condition + ' ORDER BY a.rowid'), args).fetchall()
            for row in rows:
                decoded = store._attempt_dict(row)
                self._attempts[decoded['attempt_id']] = decoded
            result = self._snapshot()
            self._attempt_cursor, self._finish_cursor = attempt_cursor, finish_cursor
            return result

    def _snapshot(self):
        attempts = list(self._attempts.values())
        items, manifest, kind, target_stage = self.items, self.manifest, self.kind, self.target_stage
        members = set(items)
        if any(row['item_key'] not in members for row in attempts):
            raise ValueError('run contains an attempt for an unlisted task')
        if kind == 'native':
            successful_inputs = {}
            for attempt in attempts:
                if attempt['stage'] in STAGES and attempt['status'] == 'succeeded':
                    key = (attempt['item_key'], attempt['stage'])
                    fingerprint = attempt['input_fingerprint']
                    if successful_inputs.setdefault(key, fingerprint) != fingerprint:
                        raise ValueError(f'conflicting successful native input fingerprints for {key!r}')
        by_id = {row['attempt_id']: row for row in attempts}
        selected = {}
        for attempt in attempts:
            if attempt['stage'] == ('pipeline' if kind == 'native' else target_stage):
                if kind == 'native' and attempt['status'] in ('succeeded', 'failed'):
                    _native_terminal(attempt, by_id)
                selected[attempt['item_key']] = attempt
        states = {}
        for key in items:
            attempt = selected.get(key)
            states[key] = {
                'status': ('unfinished' if attempt['status'] == 'interrupted' else attempt['status']) if attempt else 'pending',
                'attempt_id': attempt['attempt_id'] if attempt else None,
                'finished_at': attempt['finished_at'] if attempt else None,
                'payload': attempt['payload'] if attempt else None,
            }
        return {'run_dir': str(self.store.run_dir.resolve()), 'manifest': manifest, 'items': items, 'states': states}


def read_run(run_dir, *, kind, target_stage=None):
    """One-shot verified observation; controllers retain their reader explicitly."""
    with RunObservationReader(run_dir, kind=kind, target_stage=target_stage) as reader:
        return reader.read()
