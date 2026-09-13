"""Offline campaign identity derived from the native prepare implementation."""
import hashlib
import os
from pathlib import Path
import sys

from scripts import deepeye_bird_interact_run as native
from .ledger import CampaignLedger


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def campaign_code_hash():
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths.append(native.CODE_ROOT / 'scripts/deepeye_bird_interact_campaign.py')
    return hashlib.sha256(''.join(str(path.relative_to(native.CODE_ROOT)) + file_hash(path)
                                  for path in paths).encode()).hexdigest()


def derive(native_args, items):
    from scripts.deepeye_bird_interact_smoke import read_environment
    parser = native._build_parser()
    args = parser.parse_args(['prepare', '--run-dir', '/unused-campaign-derivation', *native_args,
                             *[value for key in items or [] for value in ('--item', key)]])
    native._validate_run_args(parser, args)
    environment = read_environment(args.env_file)
    tasks, bindings, sources = native.prepare_inputs(args.precompute_dir, args.few_shot_source,
                                                    variants=args.variant, item_keys=args.item_keys)
    return tasks, native.build_manifest(native.build_effective_config(environment, args), sources, bindings)


def configure(args):
    from scripts.rc_evaluation.deepeye.cli import production_hash
    from scripts.rc_evaluation.deepeye.contracts import load_contracts
    native_args = ['--precompute-dir', str(args.precompute_dir.resolve()),
                   '--few-shot-source', str(args.few_shot_source.resolve()),
                   '--env-file', str(args.env_file.resolve())]
    tasks, manifest = derive(native_args, args.item_keys)
    paths = {'lite': args.rc_lite.resolve(), 'full': args.rc_full.resolve()}
    load_contracts(paths, tasks)
    config = {'items': [row['task_key'] for row in manifest['items']],
              'native_args': native_args, 'native_manifest': manifest,
              'env_file': str(args.env_file.resolve()),
              'rc_lite': str(paths['lite']), 'rc_full': str(paths['full']),
              'rc_hashes': {key: file_hash(path) for key, path in paths.items()},
              'rc_code_hash': production_hash(), 'campaign_code_hash': campaign_code_hash(),
              # Resolving symlinks here loses the virtual environment's site-packages.
              'python': os.path.abspath(sys.executable), 'code_root': str(native.CODE_ROOT),
              'tail_fraction': args.tail_fraction, 'poll_seconds': args.poll_seconds}
    with CampaignLedger.create(args.campaign_dir, config) as ledger:
        (ledger.campaign_dir / 'jobs').mkdir()
        (ledger.campaign_dir / 'logs').mkdir()
        return {'campaign_dir': str(ledger.campaign_dir), 'items': len(config['items']), 'mode': 'configured'}


def validate_config(config):
    from scripts.rc_evaluation.deepeye.cli import production_hash
    if config['code_root'] != str(native.CODE_ROOT) or config['campaign_code_hash'] != campaign_code_hash():
        raise ValueError('Frozen campaign code differs from current checkout')
    if config['rc_code_hash'] != production_hash():
        raise ValueError('Frozen RC implementation changed')
    for key in ('lite', 'full'):
        if file_hash(config['rc_' + key]) != config['rc_hashes'][key]:
            raise ValueError('Frozen RC source changed: ' + key)
    tasks, manifest = derive(config['native_args'], config['items'])
    if manifest != config['native_manifest']:
        raise ValueError('Frozen native sources, bindings or effective configuration changed')
    return tasks
