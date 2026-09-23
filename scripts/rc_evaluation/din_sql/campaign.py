"""Five groups, one request dispatcher, per-target 80% group admission."""
import asyncio
from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
import signal
import time

from dotenv import load_dotenv
from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits
from scripts.baseline_adapters.din_sql.inputs import DinSettings, TaskKey, OUTPUT_NODES, file_hash
from scripts.baseline_adapters.din_sql.records import DinRecords, read_json, write_json, load_prepared
from scripts.baseline_adapters.din_sql.transport import DinRequester
from .runner import NodeExecutor, run_question, copy_reusable


def ready_for_next(total, terminal):
    return all(terminal.get(node,0)>=(4*total+4)//5 for node in OUTPUT_NODES)


def resolve_scope(operation, previous, targets, *, all_pending=False):
    if operation=='resume':
        if previous is None:
            raise ValueError('Batch has not been launched; use run')
        if all_pending:
            return None
        return [TaskKey(**key) for key in previous['targets']] if previous.get('targets') is not None else None
    return targets


def implementation_hashes(code_root):
    paths = []
    for directory in ('scripts/baseline_adapters/din_sql','scripts/rc_evaluation/din_sql'):
        paths.extend((code_root/directory).glob('*.py'))
    paths.extend(code_root/p for p in ('scripts/baseline_adapters/shared/transport.py',
                 'scripts/baseline_adapters/deepeye/run_store.py','scripts/baseline_adapters/dail_sql/execution.py',
                 'scripts/baseline_adapters/dail_sql/transport.py','scripts/rc_evaluation/deepeye/comparison.py',
                 'scripts/rc_evaluation/dail_sql/contracts.py','uv.lock'))
    return {str(p.relative_to(code_root)):file_hash(p) for p in paths}


def status(batch):
    batch = Path(batch)
    manifest = read_json(batch/'manifest.json')
    with DinRecords(batch,manifest,read_only=True) as records:
        result = {}
        for group, definition in manifest['groups'].items():
            nodes = {node:Counter() for node in OUTPUT_NODES}
            complete = 0
            for q in definition['ids']:
                key = TaskKey(group,q)
                version = records.unfinished(key) or records.current(key)
                complete += int(records.current(key) is not None)
                if version:
                    for node,value in records.view(version)['nodes'].items():
                        if node in nodes:
                            nodes[node][value['status']] += 1
            result[group] = {'total':len(definition['ids']),'complete_versions':complete,
                             'nodes':{n:dict(c) for n,c in nodes.items()},
                             'ready_for_next':ready_for_next(len(definition['ids']),{n:sum(c.values()) for n,c in nodes.items()})}
    live = batch/'monitoring/live.json'
    return {'groups':result,'last_controller_snapshot':read_json(live) if live.exists() else None,
            'http_inflight_now':None}


async def schedule(prepared, records, executor, *, targets=None, dispatcher=None):
    groups = list(records.manifest['groups'])
    tasks_by_group = {g:[] for g in groups}
    selected = set(targets) if targets is not None else None
    counts = {g:dict.fromkeys(OUTPUT_NODES,0) for g in groups}
    for key,task in prepared.tasks.items():
        if selected is not None and key not in selected:
            continue
        tasks_by_group[key.group].append(task)
        v = records.unfinished(key) or records.current(key)
        if v:
            if records.unfinished(key)==v and 'parent_version' not in records.view(v):
                parent = records.current(key)
                if records.rows[v]['attempt_no']>1 and parent is None:
                    raise ValueError('Interrupted version has no recoverable finished parent')
                records.append(v,'question_start',{'parent_version':parent,'recovered_after_interruption':True})
            parent = records.view(v).get('parent_version')
            if parent and records.unfinished(key)==v:
                await asyncio.to_thread(copy_reusable,task,parent,v,records)
            for node in OUTPUT_NODES:
                counts[key.group][node] += int(records.node(v,node) is not None)
    wake = asyncio.Event()
    running = set()
    started = []
    last_checkpoint = 0
    def terminal(key,node,state):
        if node in OUTPUT_NODES:
            counts[key.group][node] += 1
        wake.set()
    def finished(future):
        wake.set()
    async def launch(group):
        started.append(group)
        for task in tasks_by_group[group]:
            v = records.unfinished(task.key)
            if not v and records.current(task.key):
                continue
            if not v:
                v = await asyncio.to_thread(records.begin,task.key)
            future = asyncio.create_task(run_question(task,v,records,executor,on_terminal=terminal))
            future.add_done_callback(finished)
            running.add(future)
    async def checkpoint(state, *, force=False):
        nonlocal last_checkpoint
        now = time.monotonic()
        if not force and state=='running' and now-last_checkpoint < 2:
            return
        snapshot = {'state':state,'pid':os.getpid(),'started_groups':list(started),'terminal':counts,
                    'active_questions':sum(not f.done() for f in running),
                    'dispatcher':dispatcher.snapshot() if dispatcher else None}
        await asyncio.to_thread(write_json,records.root/'monitoring/live.json',snapshot)
        last_checkpoint = now
    next_group = 0
    try:
        while next_group < len(groups) or running:
            wake.clear()
            for future in list(running):
                if future.done():
                    future.result()
                    running.remove(future)
            while next_group < len(groups):
                previous = groups[next_group-1] if next_group else None
                if previous and not ready_for_next(len(tasks_by_group[previous]),counts[previous]):
                    break
                await launch(groups[next_group])
                next_group += 1
            await checkpoint('running')
            if next_group==len(groups) and not running:
                break
            try:
                await asyncio.wait_for(wake.wait(),timeout=10)
            except TimeoutError:
                pass
        await checkpoint('completed')
        return {'state':'completed','started_groups':started,'terminal':counts}
    except BaseException:
        if dispatcher:
            dispatcher.stop(cancel_active=True)
        for future in running:
            future.cancel()
        await asyncio.gather(*running,return_exceptions=True)
        await checkpoint('paused')
        raise


def run_batch(batch, env_file, *, operation='run', targets=None, all_pending=False):
    batch = Path(batch)
    manifest = read_json(batch/'manifest.json')
    prepared = load_prepared(batch)
    load_dotenv(env_file,override=True)
    settings = DinSettings(**manifest['settings'])
    model = os.environ.get('DASH_MODELS','').strip()
    if model != settings.model:
        raise ValueError('DASH_MODELS differs from the frozen DIN model')
    key, url = os.environ.get('DASH_API_KEY'),os.environ.get('DASH_BASE_URL')
    if not key or not url:
        raise ValueError('DASH_API_KEY and DASH_BASE_URL required')
    # Once per invocation, never inside a question/node loop.
    for path, expected in prepared.identities.items():
        if file_hash(path) != expected:
            raise ValueError(f'Input/template source changed: {path}; use a new batch')
    with DinRecords(batch,manifest) as records:
        code_root = Path(__file__).resolve().parents[3]
        implementation = implementation_hashes(code_root)
        frozen_code = batch/'monitoring/implementation.json'
        if frozen_code.exists() and read_json(frozen_code)!=implementation:
            raise ValueError('DIN implementation changed since launch; create a new batch')
        launch_file = batch/'monitoring/launch.json'
        previous = read_json(launch_file) if launch_file.exists() else None
        if operation=='run' and previous:
            raise ValueError('Batch was already launched; use resume or rerun')
        targets = resolve_scope(operation,previous,targets,all_pending=all_pending)
        endpoint = {'model':model,'base_url':url}
        if (batch/'monitoring/endpoint.json').exists() and read_json(batch/'monitoring/endpoint.json')!=endpoint:
            raise ValueError('Frozen endpoint differs')
        parents = previous.get('rerun_parents',{}) if operation=='resume' and not all_pending else {}
        if operation=='rerun':
            if not targets or any(t not in prepared.tasks for t in targets):
                raise ValueError('Explicit valid rerun targets required')
            if any(records.unfinished(t) for t in targets):
                raise ValueError('Target already has an unfinished version; resume it')
            if any(records.current(t) is None for t in targets):
                raise ValueError('Rerun requires finished previous versions for ALL targets')
            parents = {f'{t.group}/{t.question_id}':records.current(t) for t in targets}
        # Freeze intent before creating versions. Resume can finish an interrupted
        # multi-question rerun setup without expanding its paid scope.
        write_json(launch_file,{'operation':operation,'pid':os.getpid(),
                               'targets':[asdict(t) for t in targets] if targets is not None else None,
                               'rerun_parents':parents})
        if parents:
            for target in dict.fromkeys(targets):
                old = parents[f'{target.group}/{target.question_id}']
                new = records.unfinished(target)
                if new is None and records.current(target)!=old:
                    continue  # This target's rerun was already sealed.
                if new is None:
                    new = records.begin(target,parent_version=old)
                elif 'parent_version' not in records.view(new):
                    records.append(new,'question_start',{'parent_version':old,'recovered_after_interruption':True})
                elif records.view(new)['parent_version']!=old:
                    raise ValueError('Rerun parent does not match the persisted scope')
                copy_reusable(prepared.tasks[target],old,new,records)
        write_json(batch/'monitoring/endpoint.json',endpoint)
        write_json(frozen_code,implementation)
        limits = RequestLimits(request_limit=settings.request_limit,request_workers=settings.request_limit,
                               http_connections=settings.request_limit,start_rate=settings.start_rate,
                               request_timeout=settings.request_timeout_seconds)
        dispatcher = RequestDispatcher(limits)
        try:
            client = dispatcher.make_client(api_key=key,base_url=url)
            executor = NodeExecutor(prepared,DinRequester(dispatcher,client,settings,records))
            async def main():
                loop = asyncio.get_running_loop()
                current = asyncio.current_task()
                for sig in (signal.SIGINT,signal.SIGTERM):
                    loop.add_signal_handler(sig,current.cancel)
                try:
                    return await schedule(prepared,records,executor,targets=targets,dispatcher=dispatcher)
                finally:
                    for sig in (signal.SIGINT,signal.SIGTERM):
                        loop.remove_signal_handler(sig)
            return asyncio.run(main())
        finally:
            dispatcher.close()
