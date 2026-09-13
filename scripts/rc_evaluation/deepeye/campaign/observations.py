"""Read checksummed outcomes without traversing model-response events."""
from pathlib import Path

from scripts.baseline_adapters.deepeye.run_pipeline import STAGES
from scripts.baseline_adapters.deepeye.run_store import RunStore


def validate_items(items):
    if not isinstance(items, list) or not items:
        raise ValueError('items must be a nonempty list of task keys')
    for key in items:
        if not isinstance(key, str):
            raise ValueError('task keys must be strings')
        parts = key.split('/')
        if (len(parts) != 2 or parts[0] not in ('lite', 'full') or
                parts[1] in ('', '.', '..') or '\\' in key or key.strip() != key):
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


def read_run(run_dir, *, kind, target_stage=None):
    """Return one verified SQLite snapshot; missing/corrupt records are errors.

    RunStore has no public snapshot/filter API. Its existing internal snapshot
    and attempt decoder let RC read only the target records, with the same
    checksums as public ``attempts()``, and never read response event bodies.
    """
    if kind not in ('native', 'rc') or (kind == 'rc' and target_stage not in STAGES):
        raise ValueError('invalid run kind or RC target stage')
    path = Path(run_dir).resolve()
    with RunStore.open(path, read_only=True) as store, store._read_snapshot() as connection:
        manifest = store.manifest
        items = manifest_items(manifest)
        if kind == 'rc':
            if manifest.get('target_stage') != target_stage:
                raise ValueError('RC manifest target stage mismatch')
            rows = connection.execute(store._attempt_query('WHERE a.stage = ? ORDER BY a.rowid'),
                                      (target_stage,)).fetchall()
            attempts = [store._attempt_dict(row) for row in rows]
        else:
            attempts = store.attempts()
        if any(row['item_key'] not in items for row in attempts):
            raise ValueError('run contains an attempt for an unlisted task')
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
        return {'run_dir': str(path), 'manifest': manifest, 'items': items, 'states': states}
