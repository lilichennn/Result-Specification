"""One owner, shared resources, and durable whole-question campaign assignments."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading

import numpy as np

from scripts.baseline_adapters.dail_sql.config import MODES, TaskKey, DailSettings
from scripts.baseline_adapters.dail_sql.composite import CompositePaused, make_round_runner, run_composite
from scripts.baseline_adapters.dail_sql.current_index import CurrentIndex
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.dail_sql.preparation import load_prepared_group, load_training_pool
from scripts.baseline_adapters.dail_sql.transport import GroupRequester, classify_error
from scripts.baseline_adapters.dail_sql import native
from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits

ROOT = Path(__file__).resolve().parents[3]
PG_FIELDS = ('PG_HOST', 'PG_PORT', 'PG_USER', 'PG_PASSWORD', 'PG_SSLMODE')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment(env_file=None):
    values = dict(os.environ)
    if env_file:
        from dotenv import dotenv_values
        if not Path(env_file).is_file():
            raise ValueError('environment file unavailable')
        values = {**dotenv_values(env_file), **values}
    return values


def selected_groups(values, available):
    if values is None or values == 'all' or values == ['all']:
        return list(available)
    if isinstance(values, str):
        values = [values]
    names = [name for value in values for name in value.split(',')]
    if not names or len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError('unknown, duplicate, or empty selected groups')
    return [name for name in available if name in names]


def read_targets(path, batch_id):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(set(row) != {'group', 'question_id'} for row in rows):
        raise ValueError('targets require nonempty group/question_id JSONL')
    keys = [TaskKey(batch_id, row['group'], row['question_id']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate targets')
    return keys


def _validate_targets(manifest, targets):
    members = {TaskKey(manifest['batch_id'], group, q)
               for group, binding in manifest['groups'].items() for q in binding['ids']}
    if not targets or len(set(targets)) != len(targets) or set(targets) - members:
        raise ValueError('unknown, duplicate, or empty targets')


@contextmanager
def _owner(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('batch already has a live owner') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _configuration(config, prepared_root, batch_root):
    """One compatibility check before writers/transports; secrets stay in memory."""
    settings = DailSettings(**config['settings']).validate()
    values = environment(config.get('env_file'))
    if any(not isinstance(values.get(name), str) or not values[name].strip()
           for name in ('DASH_MODELS', 'DASH_BASE_URL', 'DASH_API_KEY')):
        raise ValueError('model environment required')
    model = values['DASH_MODELS'].strip()
    if ',' in model or model.startswith('[') or config.get('model', model) != model:
        raise ValueError('exactly one matching model required')
    identity = _read_json(prepared_root / 'identity.json')
    source = identity['manifest']
    if config.get('format') != 'dail-sql-inputs-v1' or source['settings'] != asdict(settings):
        raise ValueError('configuration differs from prepared settings')
    definitions = {row['name']: row for row in config['groups']}
    if len(definitions) != len(config['groups']) or set(definitions) != set(source['groups']):
        raise ValueError('configuration groups differ from preparation')
    def resolved(value):
        path = Path(value)
        return str((path if path.is_absolute() else ROOT / path).resolve())
    for group, binding in source['groups'].items():
        definition = definitions[group]
        if any(definition[field] != binding[field] for field in ('training_pool', 'compute_cv_link')) or definition['expected_count'] != binding['count']:
            raise ValueError('configuration group binding differs from preparation')
        for field, ref in binding['sources'].items():
            if field != 'snapshot_items' and resolved(definition[field]) != ref['path']:
                raise ValueError('configuration source differs from preparation')
    if set(config['training_pools']) != set(source['training_pools']):
        raise ValueError('configuration pools differ from preparation')
    for name, pool in source['training_pools'].items():
        for field in ('sources', 'schema_sources', 'database_roots'):
            actual = [ref['path'] for ref in pool[field]] if field == 'sources' and field in pool else pool.get(field, [])
            if [resolved(ref) for ref in config['training_pools'][name].get(field, [])] != actual:
                raise ValueError('configuration training sources differ from preparation')
    groups = selected_groups(config.get('selected_groups'), source['group_order'])
    resources = {**asdict(RequestLimits(request_timeout=settings.request_timeout_seconds)), 'sql_workers': 20}
    overrides = config.get('resources', {})
    if set(overrides) - set(resources) or 'request_timeout' in overrides and overrides['request_timeout'] != 910:
        raise ValueError('invalid runtime resource override')
    resources.update(overrides)
    RequestLimits(**{k: v for k, v in resources.items() if k != 'sql_workers'})
    if type(resources['sql_workers']) is not int or not 1 <= resources['sql_workers'] <= 20:
        raise ValueError('SQL workers must be within 1..20')
    timeout = config.get('sql_timeout_seconds', 60)
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive SQL timeout required')
    prepared_files = ['identity.json', 'evaluation/source.json', 'schemas.ready.json']
    prepared_files += ['groups/' + group + '/ready.json' for group in groups
                       if (prepared_root / 'groups' / group / 'ready.json').is_file()]
    prepared_files += ['pools/' + pool + '/ready.json'
                       for pool in sorted({source['groups'][group]['training_pool'] for group in groups})
                       if (prepared_root / 'pools' / pool / 'ready.json').is_file()]
    adapter = ROOT / 'scripts/baseline_adapters/dail_sql'
    implementation = {str(path.relative_to(ROOT)): _hash(path) for path in [
        *[adapter / name for name in ('config.py', 'records.py', 'current_index.py', 'transport.py',
                                      'composite.py', 'selection.py', 'execution.py', 'native.py', 'prompts.py', 'retrieval.py', 'preparation.py')],
        *[native.SOURCE_ROOT / name for name in (*native.ALLOWLIST, native.VOTE_SOURCE)],
        Path(__file__), Path(__file__).with_name('contracts.py'), Path(__file__).with_name('rc_prompt.txt'),
        ROOT / 'scripts/baseline_adapters/shared/transport.py']}
    frozen_config = {key: config[key] for key in ('format', 'settings', 'groups', 'training_pools')}
    manifest = {'format': 'dail-campaign-v1', 'batch_id': config.get('batch_id', batch_root.name),
        'purpose': config.get('purpose', 'formal'), 'settings': asdict(settings), 'model': model,
        'endpoint_sha256': hashlib.sha256(values['DASH_BASE_URL'].encode()).hexdigest(),
        'prepared_root': str(prepared_root), 'prepared_files': {name: _hash(prepared_root / name) for name in prepared_files},
        'implementation': implementation, 'config': frozen_config,
        'env_file': str(Path(config['env_file']).resolve()) if config.get('env_file') else None,
        'resources': resources, 'sdk_max_retries': 0, 'sql_timeout_seconds': timeout,
        'group_order': groups, 'groups': {group: {'ids': source['groups'][group]['ids']} for group in groups},
        'evaluation_source': str(prepared_root / 'evaluation/source.json')}
    targets = config.get('targets')
    if manifest['purpose'] == 'smoke':
        per_group = config.get('smoke_per_group')
        if type(per_group) is not int or per_group < 1 or resources['request_limit'] > 20:
            raise ValueError('smoke needs a positive per-group count and at most 20 requests')
        manifest['smoke_per_group'] = per_group
        if targets is None:
            targets = [TaskKey(manifest['batch_id'], group, q) for group in groups
                       for q in source['groups'][group]['ids'][:per_group]]
        if any(sum(key.group == group for key in targets) != per_group for group in groups):
            raise ValueError('smoke targets differ from per-group count')
    if targets is not None:
        _validate_targets(manifest, targets)
        for group in groups:
            manifest['groups'][group]['ids'] = [q for q in manifest['groups'][group]['ids'] if TaskKey(manifest['batch_id'], group, q) in targets]
        if any(not row['ids'] for row in manifest['groups'].values()):
            raise ValueError('every selected group requires targets')
    if not manifest['groups'] or any(not row['ids'] for row in manifest['groups'].values()):
        raise ValueError('nonempty groups required')
    bindings = {}
    for group in groups:
        pool = source['groups'][group]['training_pool']
        markers = [manifest['prepared_files'].get(f'groups/{group}/ready.json'),
                   manifest['prepared_files'].get(f'pools/{pool}/ready.json')]
        bindings[group] = {'training_pool': pool, 'fingerprint':
            hashlib.sha256(json.dumps(markers).encode()).hexdigest() if all(markers) else None}
    return manifest, values, {'groups': bindings}


def _run(manifest, values, source, batch_root, *, targets=None, explicit_resume=False):
    prepared_root = Path(manifest['prepared_root'])
    with _owner(batch_root):
        # Only open writers after the immutable manifest comparison.
        if (batch_root / 'manifest.json').exists() and _read_json(batch_root / 'manifest.json') != manifest:
            raise ValueError('frozen batch configuration mismatch')
        tasks, pools = {}, {}
        for group in manifest['group_order']:
            if (prepared_root / 'groups' / group / 'ready.json').is_file():
                tasks[group] = load_prepared_group(prepared_root, group)
                if not set(manifest['groups'][group]['ids']) <= tasks[group].keys():
                    raise ValueError('prepared task identities differ')
                pool = source['groups'][group]['training_pool']
                if pool not in pools:
                    examples = load_training_pool(prepared_root, pool)
                    pools[pool] = ({row['example_id']: row for row in examples},
                                   {row['example_id']: row['query_skeleton'] for row in examples},
                                   _read_json(prepared_root / 'pools' / pool / 'ids.json'))
        if targets:
            if any(key.group not in tasks for key in targets):
                raise ValueError('target preparation not ready')
        old_pg = {name: os.environ.get(name) for name in PG_FIELDS}
        for name in PG_FIELDS:
            if values.get(name) is not None:
                os.environ[name] = values[name]
        try:
            with DailRecords(batch_root, manifest) as records, CurrentIndex(batch_root / 'current.sqlite3') as index:
                for group in tasks:
                    index.bind_prepared_group(manifest['batch_id'], group, source['groups'][group]['fingerprint'])
                if targets:
                    for key in targets:
                        assignment = index.assignment(key)
                        if assignment and assignment['state'] != 'done':
                            raise ValueError('resume unfinished assignment before rerun')
                with ThreadPoolExecutor(max_workers=manifest['resources']['sql_workers']) as executor:
                    limits = RequestLimits(**{k: v for k, v in manifest['resources'].items() if k != 'sql_workers'})
                    dispatcher = RequestDispatcher(limits, fatal_policy=lambda error: classify_error(error)['pause'])
                    try:
                        client = dispatcher.make_client(api_key=values['DASH_API_KEY'], base_url=values['DASH_BASE_URL'])
                        requester = GroupRequester(dispatcher, client, DailSettings(**manifest['settings']), records,
                                                   resume_configuration_errors=explicit_resume)
                        definition = Path(__file__).with_name('rc_prompt.txt').read_text(encoding='utf-8')
                        def question_runner(key, version):
                            task = tasks[key.group][key.question_id]
                            examples, skeletons, ids = pools[task['task']['training_pool']]
                            messages = _read_json(prepared_root / task['first_prompt_ref'])
                            order = np.load(prepared_root / task['distance_order_ref'], allow_pickle=False)
                            return make_round_runner(task, version_id=version, requester=requester, records=records,
                                model=manifest['model'], examples_by_id=examples, skeletons_by_id=skeletons,
                                distance_ids=[ids[int(i)] for i in order], first_messages=messages,
                                rc3=task['task']['rc3_ref']['content'], sql_executor=executor,
                                sql_timeout_seconds=manifest['sql_timeout_seconds'], rc_definition=definition)
                        async def run_question(key, version, notify):
                            runner = await asyncio.get_running_loop().run_in_executor(None, question_runner, key, version)
                            task = tasks[key.group][key.question_id]
                            return await run_composite(task, version_id=version, run_round=runner,
                                                       records=records, on_mode_result=notify)
                        outcome = asyncio.run(_interruptible(_drive(manifest, records, index, run_question, set(tasks), targets)))
                    finally:
                        dispatcher.stop(cancel_active=True)
                        dispatcher.close()
        finally:
            for name, value in old_pg.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    result = status(batch_root)
    if outcome['status'] == 'paused':
        result['status'] = 'paused'
    return result


def run_batch(config: dict, prepared_root: Path, batch_root: Path) -> dict:
    root, prepared = Path(batch_root).resolve(), Path(prepared_root).resolve()
    manifest, values, source = _configuration(config, prepared, root)
    if (root / 'manifest.json').exists():
        raise ValueError('batch exists; use resume or choose a new batch ID')
    return _run(manifest, values, source, root)


async def _interruptible(operation):
    loop = asyncio.get_running_loop()
    installed = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if installed else None
    if installed:
        loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        return await operation
    except asyncio.CancelledError:
        return {'status': 'paused'}
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGTERM, previous)


def _resume_configuration(root):
    frozen = _read_json(root / 'manifest.json')
    config = {**frozen['config'], **{key: frozen[key] for key in (
        'batch_id', 'purpose', 'model', 'env_file', 'resources', 'sql_timeout_seconds')},
        'selected_groups': frozen['group_order']}
    if frozen['purpose'] == 'smoke':
        config['smoke_per_group'] = frozen['smoke_per_group']
        config['targets'] = [TaskKey(frozen['batch_id'], group, q)
                            for group, binding in frozen['groups'].items() for q in binding['ids']]
    manifest, values, source = _configuration(config, Path(frozen['prepared_root']), root)
    # A previously pending group may become ready; its prepared identity is
    # already frozen. Previously frozen ready artifacts may never change.
    for name, digest in frozen['prepared_files'].items():
        if manifest['prepared_files'].get(name) != digest:
            raise ValueError('prepared content changed')
    manifest['prepared_files'] = frozen['prepared_files']
    if manifest != frozen:
        raise ValueError('model, configuration, or implementation changed')
    with CurrentIndex(root / 'current.sqlite3', read_only=True) as index:
        for group, binding in source['groups'].items():
            previous = index.prepared_group(frozen['batch_id'], group)
            if previous is not None and binding['fingerprint'] != previous:
                raise ValueError('prepared group content changed')
    return manifest, values, source


def resume_batch(batch_root: Path) -> dict:
    root = Path(batch_root).resolve()
    return _run(*_resume_configuration(root), root, explicit_resume=True)


def rerun_questions(batch_root: Path, targets: list[TaskKey]) -> dict:
    root = Path(batch_root).resolve()
    manifest = _read_json(root / 'manifest.json')
    _validate_targets(manifest, targets)
    return _run(*_resume_configuration(root), root, targets=targets)


def status(batch_root: Path) -> dict:
    from .reporting import current_export
    root = Path(batch_root)
    manifest = _read_json(root / 'manifest.json')
    groups = {}
    with CurrentIndex(root / 'current.sqlite3', read_only=True) as index:
        for group in manifest['group_order']:
            counts = index.campaign_counts(manifest['batch_id'], group)
            total = len(manifest['groups'][group]['ids'])
            groups[group] = {**counts, 'total': total, 'started': index.group_started(manifest['batch_id'], group),
                             'ready_for_next': ready_for_next(total, counts['terminal']),
                             'pending': total - sum(counts['states'].values())}
    complete = all(g['states'].get('done', 0) == g['total'] for g in groups.values())
    export = current_export(root)
    return {'status': 'success' if complete else 'paused' if any(g['states'].get('paused') for g in groups.values()) else 'pending',
            'batch_root': str(root), 'batch_id': manifest['batch_id'], 'model': manifest['model'],
            'expected_questions': sum(g['total'] for g in groups.values()), 'groups': groups,
            'progress_basis': 'first_version_terminal_modes',
            'question_queue': {state: sum(g['states'].get(state, 0) for g in groups.values())
                               for state in ('active', 'paused', 'done')} | {
                                   'unstarted': sum(g['pending'] for g in groups.values())},
            'current_export': str(export) if export else None}


def ready_for_next(total, terminal_counts):
    return total > 0 and all(terminal_counts.get(mode, 0) >= (4 * total + 4) // 5 for mode in MODES)


def _interrupt_unanswered(records, version):
    rounds = {event['payload']['round_execution_id']
              for event in records.iter_events(version, kinds='request_attempt')}
    for round_id in rounds:
        for slot in records.request_history(version, round_id):
            for attempt in slot['attempts']:
                if attempt['request_result_id'] is None:
                    records.append(version, 'request_result', {
                        'request_attempt_id': attempt['request_attempt_id'], 'status': 'failed',
                        'usage': None, 'error': {'category': 'interrupted', 'retryable': True,
                                                'pause': False}, 'response': None})


async def _drive(manifest, records, index, run_question, ready_groups, targets=None):
    """Caller owns the batch lock and resources until every child has drained."""
    batch = manifest['batch_id']
    groups = manifest['group_order']
    counts = {g: index.campaign_counts(batch, g)['terminal'] for g in groups}
    changed = asyncio.Event()
    pending = set()
    launched = set()
    paused = False

    async def question(key, rerun=False):
        assignment = index.assignment(key)
        if rerun:
            if assignment and assignment['state'] != 'done':
                raise ValueError('resume unfinished assignment before rerun')
            assignment = None
        if assignment and assignment['state'] == 'done':
            return
        if assignment:
            version = assignment['version']
            lease = index.recover_assignment(key)
            if not records.is_sealed(key, version):
                _interrupt_unanswered(records, version)
        else:
            version = records.begin_version(key)
            lease = index.claim_assignment(key, version)
            assignment = index.assignment(key)
        seen = index.mode_states(key, version)

        def notify(mode, event_id):
            outcome = records.get_event(version, event_id)
            if outcome.get('mode') != mode:
                raise ValueError('mode source mismatch')
            index.record_campaign_mode(key, version, mode, event_id, lease, outcome['status'])
            if mode not in seen and assignment['first_version'] == version:
                counts[key.group][mode] += 1
            seen[mode] = event_id
            changed.set()

        try:
            if records.is_sealed(key, version):
                for mode, event in records.get_version(version)['mode_event_ids'].items():
                    notify(mode, event)
            else:
                await run_question(key, version, notify)
            index.finish_assignment(key, version, lease, records.is_sealed)
        except BaseException as error:
            index.pause_assignment(key, version, lease, type(error).__name__)
            raise

    def launch(key, rerun=False):
        task = asyncio.create_task(question(key, rerun))
        pending.add(task)
        task.add_done_callback(lambda _: changed.set())

    try:
        if targets is not None:
            for key in targets:
                launch(key, True)
        while True:
            changed.clear()
            if targets is None:
                for position, group in enumerate(groups):
                    if group in launched:
                        continue
                    started = index.group_started(batch, group)
                    previous = groups[position - 1] if position else None
                    if not started and previous and not ready_for_next(len(manifest['groups'][previous]['ids']), counts[previous]):
                        break
                    if group not in ready_groups:
                        index.set_waiting(batch, group, 'preparation_not_ready')
                        break
                    index.set_waiting(batch, group, None)
                    index.mark_group_started(batch, group)
                    launched.add(group)
                    for question_id in manifest['groups'][group]['ids']:
                        launch(TaskKey(batch, group, question_id))
            finished = [task for task in pending if task.done()]
            for task in finished:
                pending.remove(task)
                error = task.exception()
                if error is not None:
                    raise error
            if not pending:
                break
            await changed.wait()
    except CompositePaused:
        paused = True
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return {'status': 'paused' if paused else 'success'}
