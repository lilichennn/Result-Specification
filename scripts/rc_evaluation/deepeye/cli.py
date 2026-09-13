"""Prepare offline; run/resume/evaluate only through explicit commands."""
from __future__ import annotations

import argparse
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parents[3]
for directory in (CODE_ROOT, CODE_ROOT / 'baselines/DeepEye-SQL'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.baseline_adapters.deepeye.run_pipeline import STAGES
from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable
from scripts.baseline_adapters.deepeye.run_usage import observed_usage
from scripts.deepeye_run import (prepare_inputs, build_effective_config, code_source_hashes, bounded_runner_factory,
    build_runtime_config, admission_context, admission_settings, runtime_limits, sampling_runtime,
    read_environment, backend_context)
from scripts.rc_evaluation.deepeye.source import binding_partition, snapshot_source, validate_manifest


class _InlineReplayRuntime:
    """Runner-compatible scheduling for replay-only work without owned resources."""
    def __init__(self, stop_event, limits):
        self.stop_event = stop_event
        self._limits = dict(limits)

    def workflow_executor(self, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError('workflow workers must be a positive integer')
        return self

    def submit(self, function, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def snapshot(self):
        return {
            'limits': dict(self._limits),
            'workflows': {'cap': 0, 'active': 0, 'peak': 0},
            'coordinators': {'cap': self._limits['coordinator_workers'], 'active': 0, 'peak': 0},
            'samples': {'worker_cap': self._limits['request_workers'], 'active': 0, 'peak': 0, 'queued': 0},
            'requests': {'request_limit': self._limits['request_limit'],
                         'http_connections': self._limits['http_connections'],
                         'in_flight': 0, 'peak_in_flight': 0, 'submitted': 0, 'completed': 0},
        }


def production_hash():
    root = Path(__file__).resolve().parent
    paths = sorted([*root.glob('*.py'), *root.glob('*.txt')])
    hasher = hashlib.sha256()
    for path in paths:
        name, content = path.name.encode(), path.read_bytes()
        hasher.update(len(name).to_bytes(8, 'big'))
        hasher.update(name)
        hasher.update(len(content).to_bytes(8, 'big'))
        hasher.update(content)
    return hasher.hexdigest()


def _semantic(config):
    return {key: value for key, value in config.items()
            if key not in ('workers', 'inner_workers', 'scheduler', 'admission', 'runtime')}


def runtime_args(effective, overrides=None):
    """Restore the explicit shared runtime; old stores remain read-only inputs."""
    chat, stages = effective['chat'], effective['stages']
    admission = effective['admission']
    runtime = effective.get('runtime', {})
    if runtime.get('version') != 'shared-sampling-runtime-v1':
        raise ValueError('Legacy native runtime cannot be mixed with this implementation; prepare a new native source run')
    if overrides is not None:
        admission_settings(overrides)
        if any(getattr(overrides, key, None) is not None for key in ('workers', 'inner_workers', 'thinking_budget')):
            raise ValueError('Legacy workers/inner-workers/thinking-budget are unsupported in this version')
    values = {
        'pg_concurrency': admission['postgres_limit'],
        'max_tokens': chat['max_tokens'],
        'thinking_budget': chat.get('thinking_budget'), 'chat_timeout': chat['timeout_seconds'],
        'extractor_retries': chat['extractor_max_retries'], 'pg_sslmode': effective.get('postgres', {}).get('sslmode', 'prefer'),
        'direct_linking_budget': stages['schema_linking']['direct_linking_sampling_budget'],
        'reversed_linking_budget': stages['schema_linking']['reversed_linking_sampling_budget'],
        'dc_generation_budget': stages['sql_generation']['dc_sampling_budget'],
        'skeleton_generation_budget': stages['sql_generation']['skeleton_sampling_budget'],
        'icl_generation_budget': stages['sql_generation']['icl_sampling_budget'],
        'revision_checker_budget': stages['sql_revision']['checker_sampling_budget'],
        'selection_evaluator_budget': stages['sql_selection']['evaluator_sampling_budget'],
        **{key: value for key, value in runtime.items() if key not in ('version', 'request_timeout')},
    }
    for key in values:
        value = getattr(overrides, key, None) if overrides is not None else None
        if value is not None:
            values[key] = value
    for key, value in values.items():
        if key not in ('pg_sslmode', 'thinking_budget', 'start_rate', 'retry_delay'):
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
    if values['thinking_budget'] is not None:
        raise ValueError('thinking_budget is unsupported in this version')
    if values['chat_timeout'] > 1200:
        raise ValueError('chat_timeout exceeds the 1200-second bound')
    result = SimpleNamespace(**values)
    result.workload = effective.get('workload')
    runtime_limits(result)
    return result


def _check_source_config(source_manifest, environment, args):
    effective = build_effective_config(environment, args)
    if _semantic(effective) != _semantic(source_manifest['effective_config']):
        raise ValueError('Model, budget, PostgreSQL or semantic configuration differs from source baseline')
    if code_source_hashes() != source_manifest['sources']['code']:
        raise ValueError('Current baseline/adapter/dependency source hashes differ from frozen baseline')
    return effective


def _partition_path(value):
    partition, separator, raw_path = value.partition('=')
    if not separator or not partition or not raw_path:
        raise argparse.ArgumentTypeError('source must be PARTITION=PATH')
    return partition, Path(raw_path)


def _partition_paths(pairs, legacy, label):
    paths = {}
    for partition, path in pairs or ():
        if partition in paths:
            raise ValueError(f'Duplicate {label} source for partition {partition!r}')
        paths[partition] = path
    for partition, path in legacy:
        if path is not None:
            if partition in paths:
                raise ValueError(f'Duplicate {label} source for partition {partition!r}')
            paths[partition] = path
    return paths


def _contract_paths(args, baseline):
    paths = _partition_paths(getattr(args, 'rc', ()),
        (('lite', args.rc_lite), ('full', args.rc_full)), 'RC')
    workload = (baseline.get('sources', {}).get('workload') or
                baseline.get('effective_config', {}).get('workload'))
    if workload and workload.get('rc'):
        paths.setdefault(workload['partition'], Path(workload['rc']))
    return paths


def _evaluation_paths(args):
    return _partition_paths(getattr(args, 'reference', ()),
        (('lite', args.reference_lite), ('full', args.reference_full)), 'reference')


def prepare_command(args):
    with RunStore.open(args.source_run.resolve(), read_only=True) as source:
        baseline = source.manifest
    runtime = runtime_args(baseline['effective_config'], args)
    environment = read_environment(args.env_file.resolve(), args=runtime)
    effective = _check_source_config(baseline, environment, runtime)
    locators = baseline['sources']['locators']
    keys = args.item_keys or [row['task_key'] for row in baseline['items']]
    workload = baseline['sources'].get('workload') or baseline['effective_config'].get('workload')
    if workload is not None:
        tasks, bindings, sources = prepare_inputs(
            workload=workload, variants=args.variant, item_keys=keys)
    else:
        tasks, bindings, sources = prepare_inputs(
            args.precompute_dir or Path(locators['precompute_dir']),
            args.few_shot_source or Path(locators['few_shot_source']),
            variants=args.variant, item_keys=keys)
    if sources != baseline['sources']:
        raise ValueError('Input sources differ from the frozen native baseline')
    snapshot = snapshot_source(args.source_run, tasks, args.target_stage,
                               continue_downstream=args.continue_downstream)
    if {row['task_key']: row for row in bindings} != {row['task_key']: row for row in snapshot['items']}:
        raise ValueError('Prepared input bindings differ from baseline')
    contracts = {}
    if args.condition == 'rc':
        from scripts.rc_evaluation.deepeye.contracts import load_contracts
        contracts = load_contracts(_contract_paths(args, baseline), tasks)
    elif args.rc_lite is not None or args.rc_full is not None or args.rc:
        raise ValueError('RC files are only accepted for condition=rc')
    manifest = {
        'format': 'deepeye-rc-evaluation-run-v1', 'fingerprint_algorithm': 'manifest-digest-v2',
        'target_stage': args.target_stage,
        'condition': args.condition, 'repeat_id': args.repeat_id,
        'continue_downstream': args.continue_downstream, 'item_count': len(tasks),
        'effective_config': effective, 'sources': {**sources, 'rc_evaluation_code_sha256': production_hash()},
        'contracts': contracts, **snapshot,
    }
    validate_manifest(manifest)
    args.run_dir.resolve().parent.mkdir(parents=True, exist_ok=True)
    with RunStore.create(args.run_dir.resolve(), manifest) as store:
        return {'prepared': len(tasks), 'executed': False, 'run_dir': str(store.run_dir),
                'zero_call_targets': sum(row['stages'][args.target_stage]['api_trace']['requests'] == 0
                                         for row in manifest['source_checkpoints'].values()),
                'verification': store.verify()}


def _check_frozen(store, environment, *, prepared=None):
    from scripts.rc_evaluation.deepeye.runner import prepare_experiment
    prepared = prepared or prepare_experiment(store)
    prepared.check(store)
    manifest = prepared.manifest
    if set(prepared.plans) != set(manifest['source_checkpoints']):
        raise ValueError('Frozen execution check requires the full prepared experiment')
    if production_hash() != manifest['sources']['rc_evaluation_code_sha256']:
        raise ValueError('RC experiment production source hashes changed; prepare a new run')
    args = runtime_args(manifest['effective_config'])
    effective = _check_source_config(manifest['source_manifest'], environment, args)
    if effective != manifest['effective_config']:
        raise ValueError('Runtime configuration differs from prepared experiment')
    if manifest['condition'] == 'rc':
        from scripts.rc_evaluation.deepeye.contracts import load_contracts
        paths = {}
        tasks = []
        bindings = {row['task_key']: row for row in manifest['items']}
        for key in manifest['source_checkpoints']:
            partition = binding_partition(bindings[key])
            contract = manifest['contracts'][key]
            path = Path(contract['source_file'])
            if partition in paths and paths[partition] != path:
                raise ValueError('Multiple RC sources for one partition')
            paths[partition] = path
            tasks.append((partition, prepared.plans[key]['state']))
        if load_contracts(paths, tasks) != manifest['contracts']:
            raise ValueError('RC source records changed since preparation')
    return args


def execute_run(store, environment, *, item_keys=None, prepared=None):
    """The sole production execution path, always using the bounded factory."""
    from scripts.rc_evaluation.deepeye.runner import run_experiment, _preflight, prepare_experiment
    from scripts.baseline_adapters.deepeye.run_trace import TraceRecorder
    prepared = prepared or prepare_experiment(store)
    args = _check_frozen(store, environment, prepared=prepared)
    _, _, needed = _preflight(store, item_keys=item_keys, prepared=prepared)
    if item_keys == []:
        return {'succeeded': 0, 'failed': 0, 'executed': False}
    secrets = [environment.get(key) for key in ('DASH_API_KEY', 'EMBEDDING_API_KEY', 'PG_PASSWORD')]
    recorder = TraceRecorder(store, secrets=secrets, stop_event=threading.Event())
    if not needed:
        from scripts.baseline_adapters.deepeye.run_slots import WorkflowSlots
        def no_factory(*unused):
            raise RuntimeError('Replay-only run attempted to construct native resources')
        slots = WorkflowSlots(len(store.manifest['items']))
        runtime = _InlineReplayRuntime(recorder.stop_event, runtime_limits(args))
        result = run_experiment(store, no_factory, recorder, runtime=runtime,
                                slot_controller=slots, item_keys=item_keys, prepared=prepared)
        result['runtime'] = runtime.snapshot()
        result['admission'] = {'pipeline': slots.snapshot()}
        return result
    from scripts.rc_evaluation.deepeye.injection import install_rc_prompts
    config = build_runtime_config(environment, args, store.run_dir)
    with backend_context(environment, args):
        with recorder.install(), install_rc_prompts(), \
                admission_context(recorder, args, population=len(store.manifest['items'])) as controllers, \
                sampling_runtime(recorder, args) as runtime:
            factory = bounded_runner_factory(config, args.chat_timeout, runtime=runtime)
            result = run_experiment(store, factory, recorder, runtime=runtime,
                                    slot_controller=controllers['pipeline'], item_keys=item_keys,
                                    prepared=prepared)
            result['runtime'] = runtime.snapshot()
            result['admission'] = {key: gate.snapshot() for key, gate in controllers.items()}
            return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    from scripts.baseline_adapters.deepeye.run_operations import add_sample_commands
    add_sample_commands(commands)
    prepare = commands.add_parser('prepare', help='Offline source validation and immutable experiment manifest')
    prepare.add_argument('--source-run', type=Path, required=True)
    prepare.add_argument('--run-dir', type=Path, required=True)
    prepare.add_argument('--target-stage', choices=STAGES, required=True)
    prepare.add_argument('--condition', choices=('none', 'rc'), required=True)
    prepare.add_argument('--repeat-id', default='1')
    prepare.add_argument('--continue-downstream', action='store_true')
    prepare.add_argument('--precompute-dir', type=Path)
    prepare.add_argument('--few-shot-source', type=Path)
    prepare.add_argument('--variant', choices=('lite', 'full'), action='append')
    prepare.add_argument('--item', dest='item_keys', action='append')
    prepare.add_argument('--adaptive-concurrency', action='store_true',
                         help='Legacy adaptive throttle; rejected in the shared runtime')
    for name in ('initial', 'step', 'min', 'max'):
        prepare.add_argument('--concurrency-' + name, type=int,
                             help='Legacy adaptive policy; rejected in the shared runtime')
    prepare.add_argument('--concurrency-window', type=float,
                         help='Legacy adaptive policy; rejected in the shared runtime')
    for name in ('rc-lite', 'rc-full'):
        prepare.add_argument('--' + name, type=Path)
    prepare.add_argument('--rc', action='append', type=_partition_path, metavar='PARTITION=PATH',
                         help='RC source for an exact workload partition; repeat as needed')
    for name in ('workers', 'inner-workers', 'pg-concurrency', 'max-tokens', 'thinking-budget', 'chat-timeout',
                 'request-limit', 'request-workers', 'coordinator-workers', 'http-connections',
                 'extractor-retries', 'direct-linking-budget', 'reversed-linking-budget', 'dc-generation-budget',
                 'skeleton-generation-budget', 'icl-generation-budget', 'revision-checker-budget', 'selection-evaluator-budget'):
        prepare.add_argument('--' + name, type=int)
    prepare.add_argument('--request-start-rate', dest='start_rate', type=float)
    prepare.add_argument('--retry-delay', type=float)
    prepare.add_argument('--env-file', type=Path, default=CODE_ROOT / 'config/.env')
    for name in ('run', 'resume', 'inspect', 'export', 'evaluate', 'token-pairs'):
        command = commands.add_parser(name)
        command.add_argument('--run-dir', type=Path, required=True)
        if name == 'resume':
            command.add_argument('--unfinished-only', action='store_true',
                                 help='Execute only items without a canonical terminal outcome')
        if name in ('run', 'resume', 'evaluate'):
            command.add_argument('--env-file', type=Path, default=CODE_ROOT / 'config/.env')
        if name == 'export':
            command.add_argument('--export-dir', type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--output-dir', type=Path, required=True)
            command.add_argument('--reference-lite', type=Path)
            command.add_argument('--reference-full', type=Path)
            command.add_argument('--reference', action='append', type=_partition_path,
                                 metavar='PARTITION=PATH',
                                 help='Reference source for an exact workload partition; repeat as needed')
            command.add_argument('--database-version', required=True)
            command.add_argument('--source-run', type=Path)
    compare = commands.add_parser('compare')
    compare.add_argument('--left-dir', type=Path, required=True)
    compare.add_argument('--right-dir', type=Path, required=True)
    compare.add_argument('--output-dir', type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command in ('samples', 'renew-samples'):
            from scripts.baseline_adapters.deepeye.run_operations import sample_command
            result = sample_command(args)
        elif args.command == 'prepare':
            result = prepare_command(args)
        elif args.command in ('run', 'resume'):
            with RunStore.open(args.run_dir.resolve()) as store:
                frozen_args = runtime_args(store.manifest['effective_config'])
                environment = read_environment(args.env_file.resolve(), args=frozen_args)
                from scripts.rc_evaluation.deepeye.runner import prepare_experiment
                prepared = prepare_experiment(store)
                if getattr(args, 'unfinished_only', False):
                    from scripts.rc_evaluation.deepeye.runner import unfinished_keys
                    selected = unfinished_keys(store, prepared=prepared)
                    result = execute_run(store, environment, item_keys=selected, prepared=prepared)
                    result['recovery'] = {'selected': selected}
                else:
                    result = execute_run(store, environment, prepared=prepared)
                verification = store.verify()
                if not verification['ok']:
                    raise RuntimeError('Experiment RunStore verification failed after execution')
                result.update(verification=verification, observed_usage=observed_usage(store))
        elif args.command in ('inspect', 'export', 'token-pairs'):
            with RunStore.open(args.run_dir.resolve(), read_only=True) as store:
                validate_manifest(store.manifest)
                if args.command == 'token-pairs':
                    from scripts.rc_evaluation.deepeye.evaluation import stage_token_pairs
                    with store._read_snapshot():
                        if not store.verify()['ok']:
                            raise ValueError('Experiment RunStore verification failed')
                        result = stage_token_pairs(store.manifest, store.attempts(), store.events())
                elif args.command == 'export':
                    result = {'export_dir': str(store.export(args.export_dir.resolve(),
                                                           extra_reports={'usage.json': observed_usage}))}
                else:
                    result = {'summary': store.summary(), 'observed_usage': observed_usage(store),
                              'target_stage': store.manifest['target_stage'], 'condition': store.manifest['condition']}
        elif args.command == 'evaluate':
            from scripts.rc_evaluation.deepeye.evaluation import evaluate_run
            references = _evaluation_paths(args)
            result = evaluate_run(args.run_dir.resolve(), args.output_dir.resolve(), references,
                                  env_file=args.env_file.resolve(), database_version=args.database_version,
                                  source_run_dir=args.source_run)
        else:
            from scripts.rc_evaluation.deepeye.evaluation import compare_evaluations
            result = compare_evaluations(args.left_dir.resolve(), args.right_dir.resolve(), args.output_dir.resolve())
        print(json.dumps(to_jsonable(result), ensure_ascii=False, sort_keys=True))
        return 1 if args.command in ('run', 'resume') and result.get('failed', 0) else 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({'error_type': type(error).__name__, 'error': str(error)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
