"""Explicit, offline inheritance of verified native stage checkpoint prefixes.

This is an audited cross-version reuse claim. It never copies API events or
claims that historical artifacts were recomputed by the destination code.
"""
from __future__ import annotations

import copy
import errno
import fcntl

from .precompute_cache import fingerprint
from .precompute_pipeline import question_row
from .run_pipeline import STAGES, _restore, _valid_output, task_key
from .run_store import to_jsonable


_SCHEDULER = {'mode': 'pipeline_slots', 'concurrency_unit': 'question_pipeline'}
_SCHEDULING_FIELDS = ('workers', 'inner_workers', 'scheduler', 'admission')
_UPGRADED_CODE_FIELDS = ('adapter_python_sha256', 'entrypoints_sha256')


def _digest(value):
    return fingerprint(to_jsonable(value))


def _compatible_manifest(manifest, *, destination):
    """Remove only the explicitly permitted upgrade differences for comparison."""
    if (manifest.get('format') != 'deepeye-bird-interact-run-v1'
            or manifest.get('workflow') != list(STAGES)):
        raise ValueError('Inheritance requires a four-stage DeepEye run manifest')
    normalized = copy.deepcopy(manifest)
    config = normalized.get('effective_config')
    sources = normalized.get('sources')
    if not isinstance(config, dict) or not isinstance(sources, dict):
        raise ValueError('Inheritance requires effective configuration and source fingerprints')
    postgres = config.get('postgres')
    policy = postgres.get('execution_policy') if isinstance(postgres, dict) else None
    if (not isinstance(policy, dict) or not isinstance(policy.get('version'), str)
            or not policy['version'].strip()):
        raise ValueError('Inheritance requires an explicit versioned PostgreSQL execution policy')
    scheduler = config.get('scheduler')
    if (destination or scheduler is not None) and scheduler != _SCHEDULER:
        raise ValueError('Unsupported inheritance scheduler descriptor')
    code = sources.get('code')
    required_code = ('baseline_python_sha256', 'dependency_locks_sha256', *_UPGRADED_CODE_FIELDS)
    if not isinstance(code, dict) or any(not isinstance(code.get(key), str) or not code[key]
                                         for key in required_code):
        raise ValueError('Inheritance requires native, dependency, adapter and entrypoint source hashes')
    for key in ('precompute_inputs_content_hash', 'precompute_config_content_hash', 'few_shot_source_sha256'):
        if not isinstance(sources.get(key), str) or not sources[key]:
            raise ValueError(f'Inheritance source fingerprint is missing: {key}')
    for key in _SCHEDULING_FIELDS:
        config.pop(key, None)
    for key in _UPGRADED_CODE_FIELDS:
        code.pop(key)
    return normalized


def _bound_tasks(manifest, tasks):
    """Check manifest bindings and prepare untouched copies of complete inputs."""
    rows = manifest.get('items')
    if not isinstance(rows, list) or manifest.get('item_count') != len(rows):
        raise ValueError('Invalid inheritance manifest item bindings')
    bindings = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('task_key'), str):
            raise ValueError('Invalid inheritance item binding')
        key = row['task_key']
        if key in bindings:
            raise ValueError(f'Duplicate inheritance item binding: {key}')
        for field in ('question_sha256', 'schema_sha256', 'keywords_sha256',
                      'retrieval_sha256', 'few_shot_sha256'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Missing input fingerprint for {key}: {field}')
        bindings[key] = row
    prepared = {}
    for variant, original in tasks:
        key = task_key(variant, original)
        if key in prepared:
            raise ValueError(f'Duplicate inheritance task: {key}')
        if key not in bindings:
            raise ValueError(f'Inheritance target set differs from manifest: {key}')
        if original.gold_sql:
            raise ValueError('Target gold SQL must not enter inheritance')
        for stage in STAGES:
            if any(value is not None for value in original.get_stage_artifact(stage).model_dump().values()):
                raise ValueError(f'Inheritance requires pristine DataItems: {key}/{stage}')
        if not original.is_stage_complete('value_retrieval'):
            raise ValueError(f'Inheritance requires complete frozen precomputation: {key}')
        binding = bindings[key]
        actual = {'database_id': original.database_id,
                  'question_sha256': _digest(question_row(original)),
                  'schema_sha256': _digest(original.database_schema),
                  'few_shot_sha256': _digest(original.few_shot_examples)}
        for field, value in actual.items():
            if binding.get(field) != value:
                raise ValueError(f'Inheritance input differs from manifest: {key}/{field}')
        prepared[key] = copy.deepcopy(original)
    if set(prepared) != set(bindings):
        raise ValueError('Inheritance tasks must exactly match the manifest target set')
    return prepared


def _validated_restore(item, stage, payload, *, label):
    try:
        if not isinstance(payload, dict):
            raise ValueError('checkpoint payload must be a dictionary')
        _restore(item, stage, payload)
        if not _valid_output(item, stage):
            raise ValueError('checkpoint does not contain valid native stage output')
    except Exception as error:
        raise ValueError(f'Invalid inheritance checkpoint {label}/{stage}: {error}') from error


def _inherited_payload(source, source_manifest_hash, completed, input_hash):
    original = completed['payload']
    payload = copy.deepcopy(original)
    source_wall_time = original.get('attempt_wall_seconds', 0.0)
    provenance = {
        'source_run': str(source.run_dir.resolve()),
        'source_manifest_fingerprint': source_manifest_hash,
        'source_attempt_id': completed['attempt_id'],
        'source_input_fingerprint': input_hash,
        'source_payload_fingerprint': _digest(original),
        'source_attempt_wall_seconds': source_wall_time,
    }
    if 'inheritance' in original:
        provenance['prior_inheritance'] = copy.deepcopy(original['inheritance'])
    payload.update(execution_origin='inherited_successful_checkpoint',
                   attempt_wall_seconds=0.0, source_attempt_wall_seconds=source_wall_time,
                   inheritance=provenance)
    return payload


def inherit_checkpoints(source, destination, tasks):
    """Import verified contiguous successes; return imported/reused stage counts.

    ``source`` must be an immutable read-only RunStore; ``destination`` is fresh
    or contains matching earlier imports. All eligible payloads and existing
    destination records are validated before the first write. Each imported
    checkpoint is durable independently, so a later invocation can resume a
    partial import. Every target must have a source schema-linking attempt whose
    canonical fingerprint binds the full pristine input; older manifests alone
    cannot authenticate a never-started target's reconstructed precompute state.
    Caller DataItems and source records are untouched.
    """
    if source.run_dir.resolve() == destination.run_dir.resolve():
        raise ValueError('Source and destination inheritance runs must be different')
    # RunStore writers hold an exclusive flock on this existing file. A shared
    # read-only lease keeps the entire preflight/import stable without opening a
    # writer, changing the source, or leaving the source free to resume mid-copy.
    with (source.run_dir / '.writer.lock').open('rb') as source_lease:
        try:
            fcntl.flock(source_lease.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise ValueError('Inheritance source has an active writer; stop it before importing') from error
            raise
        return _inherit_verified_prefixes(source, destination, list(tasks))


def _inherit_verified_prefixes(source, destination, tasks):
    for label, store in (('source', source), ('destination', destination)):
        if not store.verify().get('ok'):
            raise ValueError(f'Inheritance {label} RunStore failed integrity verification')
    source_manifest, destination_manifest = source.manifest, destination.manifest
    if _compatible_manifest(source_manifest, destination=False) != _compatible_manifest(
            destination_manifest, destination=True):
        raise ValueError('Inheritance manifests differ in data, semantic configuration or protected sources')
    prepared = _bound_tasks(source_manifest, tasks)
    source_manifest_hash = _digest(source_manifest)
    source_first_inputs = {}
    for row in source.attempts():
        if row['stage'] == STAGES[0]:
            source_first_inputs.setdefault(row['item_key'], set()).add(row['input_fingerprint'])
    for key, original in prepared.items():
        source_identity = _digest({'manifest': source_manifest,
                                   'input': to_jsonable(original.model_dump(exclude={'gold_sql'}))})
        expected_input = _digest({'input': source_identity, 'stage': STAGES[0]})
        if key not in source_first_inputs:
            raise ValueError(f'Cannot verify full frozen input for unstarted source task: {key}')
        if source_first_inputs[key] != {expected_input}:
            raise ValueError(f'Inheritance pristine input fingerprint differs from source: {key}')
    existing = {}
    for row in destination.attempts():
        key = (row['item_key'], row['stage'])
        if (key in existing or row['status'] not in ('succeeded', 'interrupted')
                or row['stage'] not in STAGES or destination.events(row['attempt_id'])):
            raise ValueError(f'Destination is not a consistent partial inheritance: {key}')
        existing[key] = row

    planned = []
    seen_existing = set()
    for key, original in prepared.items():
        snapshot = to_jsonable(original.model_dump(exclude={'gold_sql'}))
        source_identity = _digest({'manifest': source_manifest, 'input': snapshot})
        destination_identity = _digest({'manifest': destination_manifest, 'input': snapshot})
        source_state, destination_state = copy.deepcopy(original), copy.deepcopy(original)
        destination_gap = False
        for stage in STAGES:
            source_input = _digest({'input': source_identity, 'stage': stage})
            prior = source.completed(key, stage, source_input)
            if prior is None:
                break
            _validated_restore(source_state, stage, prior['payload'], label=f'source:{key}')
            destination_input = _digest({'input': destination_identity, 'stage': stage})
            payload = _inherited_payload(source, source_manifest_hash, prior, source_input)
            _validated_restore(destination_state, stage, payload, label=f'destination:{key}')
            current = existing.get((key, stage))
            if current is not None:
                if (destination_gap or current['input_fingerprint'] != destination_input
                        or (current['status'] == 'succeeded'
                            and _digest(current['payload']) != _digest(payload))):
                    raise ValueError(f'Incompatible existing destination checkpoint: {key}/{stage}')
                seen_existing.add((key, stage))
            reused = current is not None and current['status'] == 'succeeded'
            if not reused:
                destination_gap = True
            planned.append((key, stage, destination_input, payload, reused,
                            current['attempt_id'] if current is not None else None))
            source_identity = _digest({'input': source_input, 'output': prior['payload']})
            destination_identity = _digest({'input': destination_input, 'output': payload})
    if set(existing) != seen_existing:
        raise ValueError('Destination contains checkpoints outside the eligible source prefixes')

    report = {'imported': 0, 'reused': 0,
              'stages': {stage: {'imported': 0, 'reused': 0} for stage in STAGES}}
    for key, stage, input_hash, payload, reused, existing_attempt in planned:
        kind = 'reused' if reused else 'imported'
        if not reused:
            attempt = existing_attempt or destination.begin_attempt(key, stage, input_hash)
            destination.finish_attempt(attempt, 'succeeded', payload)
        report[kind] += 1
        report['stages'][stage][kind] += 1
    return report
