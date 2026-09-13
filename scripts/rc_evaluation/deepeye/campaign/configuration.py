"""Offline campaign identity derived from the native prepare implementation."""
import hashlib
import os
from pathlib import Path
import sys

from scripts import deepeye_run as native
from .ledger import CampaignLedger


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def campaign_code_hash():
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths.extend(native.CODE_ROOT / 'scripts' / name for name in
                 ('deepeye_campaign.py', 'deepeye_bird_interact_campaign.py'))
    return hashlib.sha256(''.join(str(path.relative_to(native.CODE_ROOT)) + file_hash(path)
                                  for path in paths).encode()).hexdigest()


def derive(native_args, items):
    parser = native._build_parser()
    args = parser.parse_args(['prepare', '--run-dir', '/unused-campaign-derivation', *native_args,
                             *[value for key in items or [] for value in ('--item', key)]])
    native._validate_run_args(parser, args)
    environment = native.read_environment(args.env_file, args=args)
    tasks, bindings, sources = native.prepare_inputs(args.precompute_dir, args.few_shot_source,
        workload=args.workload, variants=args.variant, item_keys=args.item_keys)
    return tasks, native.build_manifest(native.build_effective_config(environment, args), sources, bindings)


def configure(args):
    from scripts.rc_evaluation.deepeye.cli import production_hash
    from scripts.rc_evaluation.deepeye.contracts import load_contracts
    native_args = ['--env-file', str(args.env_file.resolve())]
    if getattr(args, 'workload', None) is not None:
        from scripts.baseline_adapters.deepeye.workloads import load_workload
        workload = load_workload(args.workload)
        native_args.extend(['--workload', str(args.workload.resolve())])
        paths = {workload['partition']: Path(workload['rc'])}
        if any(getattr(args, name) is not None for name in ('precompute_dir', 'few_shot_source', 'rc_lite', 'rc_full')):
            raise ValueError('Use workload preparation and RC paths instead of BIRD-Interact-specific options')
    else:
        for field in ('precompute_dir', 'few_shot_source', 'rc_lite', 'rc_full'):
            if getattr(args, field) is None:
                raise ValueError('Without --workload, BIRD-Interact preparation and both RC paths are required')
        native_args.extend(['--precompute-dir', str(args.precompute_dir.resolve()),
                            '--few-shot-source', str(args.few_shot_source.resolve())])
        paths = {'lite': args.rc_lite.resolve(), 'full': args.rc_full.resolve()}
    tasks, manifest = derive(native_args, args.item_keys)
    load_contracts(paths, tasks)
    config = {'items': [row['task_key'] for row in manifest['items']],
              'native_args': native_args, 'native_manifest': manifest,
              'env_file': str(args.env_file.resolve()),
              'rc_sources': {key: str(path) for key, path in paths.items()},
              'rc_hashes': {key: file_hash(path) for key, path in paths.items()},
              'rc_code_hash': production_hash(), 'campaign_code_hash': campaign_code_hash(),
              # Resolving symlinks here loses the virtual environment's site-packages.
              'python': os.path.abspath(sys.executable), 'code_root': str(native.CODE_ROOT),
              'tail_fraction': args.tail_fraction, 'poll_seconds': args.poll_seconds}
    with CampaignLedger.create(args.campaign_dir, config) as ledger:
        (ledger.campaign_dir / 'jobs').mkdir()
        (ledger.campaign_dir / 'logs').mkdir()
        return {'campaign_dir': str(ledger.campaign_dir), 'items': len(config['items']), 'mode': 'configured'}


def validate_config(config, *, inputs=True):
    from scripts.rc_evaluation.deepeye.cli import production_hash
    if config['code_root'] != str(native.CODE_ROOT) or config['campaign_code_hash'] != campaign_code_hash():
        raise ValueError('Frozen campaign code differs from current checkout')
    if config['rc_code_hash'] != production_hash():
        raise ValueError('Frozen RC implementation changed')
    paths = config.get('rc_sources') or {key: config['rc_' + key] for key in ('lite', 'full')}
    for key, path in paths.items():
        if file_hash(path) != config['rc_hashes'][key]:
            raise ValueError('Frozen RC source changed: ' + key)
    if native.code_source_hashes() != config['native_manifest']['sources']['code']:
        raise ValueError('Frozen native code changed')
    if not inputs:
        # The actual child prepare/resume validates its inputs and compares the
        # prepared manifest. Supervisors must not prepare the same cohort again.
        return None
    tasks, manifest = derive(config['native_args'], config['items'])
    if manifest != config['native_manifest']:
        raise ValueError('Frozen native sources, bindings or effective configuration changed')
    return tasks
