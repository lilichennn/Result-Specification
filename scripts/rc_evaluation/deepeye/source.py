"""Offline, checksummed snapshots of successful native stage lineages."""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

from scripts.baseline_adapters.deepeye.precompute_cache import fingerprint
from scripts.baseline_adapters.deepeye.run_pipeline import STAGES, _restore, _valid_output, task_key, input_identity
from scripts.baseline_adapters.deepeye.run_store import RunStore, restore_jsonable, to_jsonable
from scripts.baseline_adapters.deepeye.workloads import dump_item, restore_item, external_id
from scripts.baseline_adapters.deepeye.run_usage import sampling_completeness

TRACE_KINDS = ('api_request', 'api_response', 'api_error', 'sampling_group_start',
               'sampling_group_result', 'sample_result', 'sample_attempt_started', 'sample_attempt')


def digest(value):
    return fingerprint(to_jsonable(value))


def api_trace(events):
    """Verify recorded outcomes; partial sampling is valid, not a broken trace.

    ``complete`` describes trace integrity. ``sampling.complete`` independently
    describes whether every requested sample succeeded; it is not an RC gate.
    """
    calls, responses, errors = {}, 0, 0
    events = list(events)
    for event in events:
        kind = event['kind']
        if kind not in ('api_request', 'api_response', 'api_error'):
            continue
        call = event['payload'].get('call_id')
        if not isinstance(call, str) or not call:
            raise ValueError('API trace has no call_id')
        if kind == 'api_request':
            if call in calls:
                raise ValueError('Duplicate API trace call_id')
            calls[call] = None
        else:
            if call not in calls or calls[call] is not None:
                raise ValueError('API trace terminal has no unique preceding request')
            calls[call] = kind
            responses += kind == 'api_response'
            errors += kind == 'api_error'
    unanswered = sum(value is None for value in calls.values())
    sampling = sampling_completeness(events)
    result = {'requests': len(calls), 'responses': responses, 'errors': errors,
            'unanswered_requests': unanswered, 'complete': unanswered == 0}
    restored = [e['payload'] for e in events if e['kind'] == 'sample_result'
                and e['payload'].get('restored_from_event') is not None]
    if restored:
        result['restored_samples'] = len(restored)
        result['restored_rc_samples'] = sum(p.get('rc_applied') is True for p in restored)
    if sampling['groups']:
        # A group summary cannot substitute for the actual retained slot records.
        targets, retained, terminal, indices = {}, {}, {}, {}
        started, fatal = {}, {}
        for event in events:
            payload = event['payload']
            key = (event.get('attempt_id'), payload.get('group_id'))
            if event['kind'] == 'sampling_group_start':
                targets[key] = payload['target_n']
            elif event['kind'] == 'sampling_group_result':
                terminal[key] = payload
            elif event['kind'] == 'sample_result':
                indices.setdefault(key, set()).add(payload['sample_index'])
                if payload.get('succeeded'):
                    retained.setdefault(key, set()).add(payload['sample_index'])
                elif payload.get('fatal') is True:
                    fatal.setdefault(key, set()).add(payload['sample_index'])
            elif event['kind'] in ('sample_attempt_started', 'sample_attempt'):
                started.setdefault(key, set()).add(payload['sample_index'])
        incomplete = {key for key, target in targets.items() if (
            retained.get(key, set()) != set(range(target)) or terminal.get(key, {}).get('complete') is not True
            or terminal.get(key, {}).get('success_count') != target
            or terminal.get(key, {}).get('target_n') != target)}
        incomplete.update(set(retained).difference(targets))
        sampling.update(complete=not incomplete, incomplete_groups=len(incomplete))
        # A documented exhausted/fatal sample is different from a missing or
        # inconsistent log record. Do not require target_n successful slots.
        consistent = (set(indices) | set(started)).issubset(targets) and all(
            key in terminal
            and indices.get(key, set()).issubset(range(target))
            and started.get(key, set()).issubset(indices.get(key, set()))
            and (indices.get(key, set()) == set(range(target))
                 or any(indices.get(key, set()) == set(range(index + 1))
                        for index in fatal.get(key, ())))
            and terminal[key].get('target_n') == target
            and terminal[key].get('success_count') == len(retained.get(key, set()))
            and terminal[key].get('complete') is (len(retained.get(key, set())) == target)
            for key, target in targets.items())
        result['complete'] = result['complete'] and consistent
        result['sampling'] = sampling
        from scripts.baseline_adapters.deepeye.run_usage import _effective_sampling
        result['effective_sampling'] = _effective_sampling(events)
    return result


def _source_trace(store, row, stage, seen=()):
    if not hasattr(store, '_source_sampling_index'):
        from scripts.baseline_adapters.deepeye.run_sampling import SamplingCheckpoints
        store._source_sampling_index = SamplingCheckpoints(store)
    provenance = row['payload'].get('inheritance')
    if row['payload'].get('execution_origin') == 'inherited_successful_checkpoint':
        if not isinstance(provenance, dict):
            raise ValueError('Inherited source has no provenance')
        identity = (provenance['source_run'], provenance['source_attempt_id'])
        if identity in seen or len(seen) >= 32:
            raise ValueError('Cyclic source inheritance')
        with RunStore.open(Path(identity[0]), read_only=True) as prior:
            with prior._read_snapshot():
                if not prior.verify()['ok'] or prior.manifest_fingerprint != provenance['source_manifest_fingerprint']:
                    raise ValueError('Inherited source manifest fingerprint mismatch')
                matches = [candidate for candidate in prior.attempts() if candidate['attempt_id'] == identity[1]]
                if len(matches) != 1 or matches[0]['status'] != 'succeeded':
                    raise ValueError('Inherited source attempt is not complete')
                original = matches[0]
                if (original['stage'] != stage or original['input_fingerprint'] != provenance['source_input_fingerprint']
                        or digest(original['payload']) != provenance['source_payload_fingerprint']):
                    raise ValueError('Inherited source payload fingerprint mismatch')
                return _source_trace(prior, original, stage, (*seen, identity))
    result = api_trace(store.iter_events(row['attempt_id'], kinds=TRACE_KINDS))
    if not result['complete']:
        raise ValueError('Source API trace has unanswered requests or inconsistent sampling records')
    if not result['requests'] and not result.get('restored_samples'):
        usage = restore_jsonable(row['payload']['artifact']).get(stage + '_llm_cost')
        if not isinstance(usage, dict) or any(usage.get(field) != 0 for field in
                ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            raise ValueError('Zero-call source trace conflicts with native token usage')
    return result


def _pristine(item):
    if item.gold_sql:
        raise ValueError('Target gold SQL must not enter RC runs')
    if not item.is_stage_complete('value_retrieval'):
        raise ValueError('Frozen value retrieval is incomplete')
    for stage in STAGES:
        if any(value is not None for value in item.get_stage_artifact(stage).model_dump().values()):
            raise ValueError('Source requires pristine inputs without native stage artifacts')


def _restore_checked(item, stage, payload):
    _restore(item, stage, payload)
    if not _valid_output(item, stage):
        raise ValueError(f'Invalid source stage artifact: {stage}')


def restore_seed(snapshot, target_stage):
    item = restore_item({'type': snapshot.get('input_type', 'bird_interact'),
                         'data': restore_jsonable(snapshot['input'])})
    _pristine(item)
    for stage in STAGES[:STAGES.index(target_stage)]:
        _restore_checked(item, stage, snapshot['stages'][stage]['payload'])
    return item


def snapshot_source(source_run, tasks, target_stage, *, continue_downstream=False):
    """Read a consistent source snapshot; unfinished targets are never accepted."""
    if target_stage not in STAGES:
        raise ValueError('Unknown target stage')
    tasks = list(tasks)
    if not tasks:
        raise ValueError('No selected tasks')
    with RunStore.open(Path(source_run).resolve(), read_only=True) as store:
        with store._read_snapshot():
            if not store.verify()['ok']:
                raise ValueError('Source RunStore verification failed')
            source_manifest = store.manifest
            if source_manifest.get('format') not in ('deepeye-bird-interact-run-v1', 'deepeye-run-v2') or source_manifest.get('workflow') != list(STAGES):
                raise ValueError('Source must be a native four-stage RunStore')
            bindings = {row['task_key']: row for row in source_manifest['items']}
            if len(bindings) != len(source_manifest['items']):
                raise ValueError('Duplicate source item binding')
            checkpoints = {}
            completed = {(row['item_key'], row['stage'], row['input_fingerprint']): row
                         for row in store.attempts() if row['status'] == 'succeeded'}
            stage_inputs = defaultdict(set)
            for item_key, stage, recorded_hash in completed:
                stage_inputs[item_key, stage].add(recorded_hash)
            for variant, original in tasks:
                _pristine(original)
                key = task_key(variant, original)
                if key in checkpoints or key not in bindings:
                    raise ValueError(f'Duplicate or unknown source task: {key}')
                snapshot = to_jsonable(original.model_dump(exclude={'gold_sql'}))
                identity = input_identity(source_manifest, store.manifest_fingerprint, snapshot)
                state = copy.deepcopy(original)
                records = {}
                upstream_hash = None
                limit = len(STAGES) if continue_downstream else STAGES.index(target_stage) + 1
                for index, stage in enumerate(STAGES[:limit]):
                    if stage == target_stage:
                        upstream_hash = digest(state.model_dump(exclude={'gold_sql'}))
                    input_hash = digest({'input': identity, 'stage': stage})
                    if stage_inputs[key, stage] - {input_hash}:
                        raise ValueError(f'Source checkpoint input fingerprint mismatch: {key}/{stage}')
                    prior = completed.get((key, stage, input_hash))
                    if prior is None:
                        if index <= STAGES.index(target_stage):
                            raise ValueError(f'Source target/prefix not successfully completed: {key}/{stage}')
                        break
                    _restore_checked(state, stage, prior['payload'])
                    records[stage] = {'attempt_id': prior['attempt_id'], 'input_fingerprint': input_hash,
                                      'payload': prior['payload'], 'payload_sha256': digest(prior['payload']),
                                      'api_trace': _source_trace(store, prior, stage)}
                    identity = digest({'input': input_hash, 'output': prior['payload']})
                checkpoints[key] = {'input': snapshot, 'input_sha256': digest(snapshot),
                                    'input_type': dump_item(original)['type'],
                                    'upstream_state_sha256': upstream_hash, 'stages': records}
            return {'source_run': str(store.run_dir.resolve()), 'source_manifest': source_manifest,
                    'source_manifest_fingerprint': store.manifest_fingerprint,
                    'items': [copy.deepcopy(bindings[key]) for key in sorted(checkpoints)],
                    'source_checkpoints': checkpoints}


def binding_partition(binding):
    if 'partition' in binding:
        return binding['partition']
    if binding.get('variant') in ('lite', 'full'):
        return binding['variant']
    # Legacy bindings explicitly carry split rather than partition.
    if binding.get('split') in ('lite', 'full'):
        return binding['split']
    if 'partition' not in binding and 'instance_id' in binding:
        for variant in ('lite', 'full'):
            if binding['task_key'] == f"{variant}/{binding['instance_id']}":
                return variant
    raise ValueError('Source binding has no explicit partition')


def validate_manifest(manifest, *, item_keys=None, seeds=None):
    """Validate embedded source chains without consulting mutable external state."""
    if manifest.get('format') != 'deepeye-rc-evaluation-run-v1':
        raise ValueError('Not an RC experiment manifest')
    target = manifest.get('target_stage')
    if target not in STAGES or manifest.get('condition') not in ('none', 'rc'):
        raise ValueError('Invalid experiment condition/target')
    if not isinstance(manifest.get('repeat_id'), str) or not manifest['repeat_id']:
        raise ValueError('repeat_id must be a nonempty string')
    if type(manifest.get('continue_downstream')) is not bool:
        raise ValueError('continue_downstream must be boolean')
    source = manifest['source_manifest']
    source_digest = digest(source)
    if source_digest != manifest['source_manifest_fingerprint']:
        raise ValueError('Source manifest fingerprint mismatch')
    expected_bindings = {row['task_key']: row for row in source['items']}
    if len(expected_bindings) != len(source['items']):
        raise ValueError('Duplicate source item binding')
    keys = [row['task_key'] for row in manifest['items']]
    if not keys or len(set(keys)) != len(keys) or set(keys) != set(manifest['source_checkpoints']):
        raise ValueError('Experiment item bindings differ from source checkpoints')
    contracts = manifest.get('contracts')
    if not isinstance(contracts, dict) or (set(contracts) != set(keys) if manifest['condition'] == 'rc' else bool(contracts)):
        raise ValueError('Experiment condition and bound contracts disagree')
    for binding in manifest['items']:
        if binding != expected_bindings.get(binding['task_key']):
            raise ValueError('Experiment item binding differs from source')
    for key, snapshot in manifest['source_checkpoints'].items():
        if digest(snapshot['input']) != snapshot['input_sha256']:
            raise ValueError('Source input fingerprint mismatch')
        binding = expected_bindings[key]
        partition = binding_partition(binding)
        raw = restore_jsonable(snapshot['input'])
        identity_field = 'question_id' if snapshot.get('input_type') == 'native' else 'instance_id'
        identity = raw[identity_field]
        expected_id = binding.get('external_id', binding.get('instance_id'))
        if type(identity) is not type(expected_id) or identity != expected_id or raw['database_id'] != binding['database_id']:
            raise ValueError('Source DataItem identity differs from binding')
        if item_keys is None or key in item_keys:
            seed = restore_seed(snapshot, target)
            if task_key(partition, seed) != key:
                raise ValueError('Source DataItem identity differs from binding')
            if digest(seed.model_dump(exclude={'gold_sql'})) != snapshot['upstream_state_sha256']:
                raise ValueError('Upstream state fingerprint mismatch')
            if seeds is not None:
                seeds[key] = seed
        if manifest['condition'] == 'rc':
            contract = contracts[key]
            for name, expected in (('task_key', key), ('db_id', raw['database_id']),
                                   ('question', raw['question']), ('evidence', raw['evidence'])):
                if contract.get(name) != expected:
                    raise ValueError(f'RC contract {name} differs from source input')
            from .injection import render_rc_block, manifest_prompt
            if 'rc_version' in manifest:
                from .contracts import resolve_rc_version, _record_sha256
                version = resolve_rc_version(manifest['rc_version'])
                if (contract.get('rc_version') != version or
                        contract.get('source_field') != f'rc_round{version}' or
                        contract.get('gold_corrected') != (version == 3) or
                        manifest.get('gold_corrected') != (version == 3)):
                    raise ValueError('RC contract version/provenance differs from manifest')
                if _record_sha256(contract.get('final_rc')) != contract.get('final_rc_sha256'):
                    raise ValueError('RC final content hash mismatch')
                if 'rc_prompt' not in manifest:
                    raise ValueError('Versioned RC manifest requires frozen prompt')
            render_rc_block(contract, prompt_template=manifest_prompt(manifest))
        identity = input_identity(source, source_digest, snapshot['input'])
        required = STAGES[:STAGES.index(target) + 1]
        if any(stage not in snapshot['stages'] for stage in required):
            raise ValueError('Missing source target/prefix')
        if set(snapshot['stages']) != set(STAGES[:len(snapshot['stages'])]):
            raise ValueError('Source checkpoints are not a contiguous prefix')
        for stage in STAGES[:len(snapshot['stages'])]:
            record = snapshot['stages'][stage]
            expected = digest({'input': identity, 'stage': stage})
            if record['input_fingerprint'] != expected or digest(record['payload']) != record['payload_sha256']:
                raise ValueError('Source checkpoint fingerprint mismatch')
            trace = record['api_trace']
            counts = [trace.get(field) for field in ('requests', 'responses', 'errors', 'unanswered_requests')]
            if (any(type(value) is not int or value < 0 for value in counts)
                    or trace.get('complete') is not True or counts[3] != 0 or counts[0] != counts[1] + counts[2]):
                raise ValueError('Source API trace is incomplete')
            identity = digest({'input': expected, 'output': record['payload']})
    return manifest
