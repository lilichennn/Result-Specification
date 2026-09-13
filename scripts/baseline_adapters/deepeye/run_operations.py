"""Explicit offline sample inspection/renewal shared by both run CLIs."""
from pathlib import Path

from .run_sampling import SamplingCheckpoints
from .run_store import RunStore


def add_sample_commands(commands):
    for name in ('samples', 'renew-samples'):
        command = commands.add_parser(name, help=('Inspect durable sampling slots offline' if name == 'samples'
            else 'Explicitly authorize four new attempts for exhausted slots; launches no work'))
        command.add_argument('--run-dir', type=Path, required=True)
        if name == 'renew-samples':
            command.add_argument('--group-id', required=True)
            command.add_argument('--decision', required=True)


def _inventory(checkpoints):
    groups = []
    for group_id, group in sorted(checkpoints.groups.items()):
        identity = group['identity']
        samples = []
        for index in range(group['target_n']):
            key = (group_id, index)
            prior = checkpoints.samples.get(key)
            outcome = prior['payload']['outcome'] if prior else None
            consumed = len(checkpoints.sample_starts.get(key, ()))
            limit = checkpoints.limit(group_id, index)
            succeeded = bool(outcome and outcome.succeeded)
            eligible = not succeeded and (consumed >= limit or (
                outcome and outcome.fatal and consumed > checkpoints.renewal_starts.get(key, -1)))
            samples.append({'sample_index': index, 'consumed_attempts': consumed,
                            'attempt_limit': limit, 'succeeded': succeeded,
                            'renewal_eligible': bool(eligible)})
        groups.append({'group_id': group_id, 'item_key': identity['item_key'], 'stage': identity['stage'],
                       'input_fingerprint': identity['input'], 'source_version': identity['source_version'],
                       'target_n': group['target_n'], 'samples': samples})
    return groups


def sample_command(args):
    renew = args.command == 'renew-samples'
    with RunStore.open(args.run_dir.resolve(), read_only=not renew) as store:
        if not store.verify()['ok']:
            raise ValueError('RunStore verification failed')
        checkpoints = SamplingCheckpoints(store)
        groups = _inventory(checkpoints)
        if not renew:
            return {'executed': False, 'groups': groups}
        if not isinstance(args.decision, str) or not args.decision.strip():
            raise ValueError('Renewal requires a nonempty explicit decision')
        selected = next((group for group in groups if group['group_id'] == args.group_id), None)
        if selected is None:
            raise ValueError('Exact sampling group_id not found; inspect with samples first')
        if selected['source_version'] != checkpoints.source_version:
            raise ValueError('Cannot renew samples from another implementation')
        if not any(sample['renewal_eligible'] for sample in selected['samples']):
            raise ValueError('No exhausted samples eligible for renewal')
        if store.completed(selected['item_key'], selected['stage'], selected['input_fingerprint']):
            raise ValueError('Cannot renew a completed stage')
        attempt = store.begin_attempt(selected['item_key'], selected['stage'], selected['input_fingerprint'])
        checkpoints.renew_exhausted(args.group_id, attempt_id=attempt, decision=args.decision)
        # Leave the audit stage unfinished; resume executes the normal lineage.
        return {'executed': False, 'group_id': args.group_id, 'audit_attempt_id': attempt,
                'renewed_indices': [s['sample_index'] for s in selected['samples'] if s['renewal_eligible']],
                'verification': store.verify()}
