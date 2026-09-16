"""Append-only nodes on the existing RunStore; latest FINISHED version wins."""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading

from scripts.baseline_adapters.deepeye.run_store import RunStore, to_jsonable, restore_jsonable
from .inputs import TaskKey, DinTask, PreparedInputs, NODES, OUTPUT_NODES, digest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(to_jsonable(value), stream, ensure_ascii=False, separators=(',',':'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return restore_jsonable(json.loads(Path(path).read_text()))


class DinRecords:
    def __init__(self, root, manifest, *, read_only=False):
        self.root, self.manifest = Path(root), manifest
        self.read_only = read_only
        self.stores, self.rows, self.views = {}, {}, {}
        self.latest, self.pending = {}, {}
        self._lock, self._mutex = None, threading.RLock()
        try:
            if read_only and read_json(self.root/'manifest.json') != manifest:
                raise ValueError('Read-only manifest differs')
            if not read_only:
                self.root.mkdir(parents=True, exist_ok=True)
                self._lock = (self.root/'.writer.lock').open('a+')
                try:
                    fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError('DIN batch already has a writer') from exc
                path = self.root/'manifest.json'
                if path.exists() and read_json(path) != manifest:
                    raise ValueError('Frozen manifest differs')
                if not path.exists():
                    write_json(path, manifest)
            for group in manifest['groups']:
                directory = self.root/f'group-{group}'
                if read_only and not directory.exists():
                    raise FileNotFoundError(directory)
                identity = {'format':'din-sql-v1','group':group,'batch_fingerprint':digest(manifest)}
                store = (RunStore.open(directory, expected_manifest=identity, read_only=read_only)
                         if directory.exists() else RunStore.create(directory,identity))
                self.stores[group] = store
                for row in store.attempts():
                    row['group'] = group
                    self.rows[row['attempt_id']] = row
                    key = TaskKey(group,row['item_key'])
                    index = self.latest if row['status'] in ('succeeded','failed') else self.pending
                    old = index.get(key)
                    if old is None or self.rows[old]['attempt_no'] < row['attempt_no']:
                        index[key] = row['attempt_id']
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
        if self._lock:
            self._lock.close()
            self._lock = None

    def key(self, version):
        row = self.rows[version]
        return TaskKey(row['group'],row['item_key'])

    def current(self, key):
        return self.latest.get(key)

    def unfinished(self, key):
        return self.pending.get(key)

    def begin(self, key, *, parent_version=None):
        if self.read_only:
            raise ValueError('Read-only records')
        if key.question_id not in self.manifest['groups'][key.group]['ids']:
            raise ValueError('Unknown task')
        if self.unfinished(key):
            raise ValueError('Resume unfinished version instead')
        if parent_version is not None and self.key(parent_version) != key:
            raise ValueError('Parent belongs to another question')
        store = self.stores[key.group]
        version = store.begin_attempt(key.question_id,'din_question',digest(self.manifest))
        self.rows[version] = {**store.attempt(version),'group':key.group}
        self.pending[key] = version
        self.views[version] = {'nodes':{},'inputs':{},'attempts':{},'outcomes':{},'history_loaded':True}
        self.append(version,'question_start',{'parent_version':parent_version})
        return version

    def view(self, version):
        with self._mutex:
            if version not in self.views:
                view = {'nodes':{},'inputs':{},'attempts':{},'outcomes':{},'history_loaded':False}
                self.views[version] = view
                row = self.rows[version]
                if row['status'] in ('succeeded','failed'):
                    for node,summary in row['payload']['nodes'].items():
                        ref = summary['ref']
                        view['nodes'][node] = {**self.read_ref(ref),'ref':ref}
                    return view
                for event in self.stores[row['group']].iter_events(version,
                        kinds=('question_start','node_result')):
                    ref = {'group':row['group'],'attempt_id':version,'event_no':event['event_no']}
                    self._index(view,event['kind'],event['payload'],ref)
            return self.views[version]

    def request_history(self, version):
        """Only request resumption needs prompts/budget history; read it once."""
        with self._mutex:
            view = self.view(version)
            if not view['history_loaded']:
                row = self.rows[version]
                view.update(inputs={},attempts={},outcomes={})
                for event in self.stores[row['group']].iter_events(version,
                        kinds=('node_input','request_attempt','request_outcome')):
                    ref = {'group':row['group'],'attempt_id':version,'event_no':event['event_no']}
                    self._index(view,event['kind'],event['payload'],ref)
                view['history_loaded'] = True
            return view

    @staticmethod
    def _index(view, kind, payload, ref):
        node = payload.get('node')
        if kind == 'question_start':
            view['parent_version'] = payload.get('parent_version')
        elif kind == 'node_result':
            view['nodes'][node] = {**payload,'ref':ref}
        elif kind == 'node_input':
            view['inputs'][node] = {'input_fingerprint':payload['input_fingerprint'],'ref':ref}
        elif kind == 'request_attempt':
            view['attempts'].setdefault(node,[]).append({**payload,'ref':ref})
        elif kind == 'request_outcome':
            view['outcomes'].setdefault(node,[]).append({**payload,'ref':ref})

    def append(self, version, kind, payload):
        with self._mutex:
            view = self.view(version)
            group = self.rows[version]['group']
            number = self.stores[group].append_event(version,kind,payload)
            ref = {'group':group,'attempt_id':version,'event_no':number}
            self._index(view,kind,payload,ref)
            return ref

    def read_ref(self, ref):
        return self.stores[ref['group']].event(ref['attempt_id'],ref['event_no'])['payload']

    def node(self, version, node):
        return self.view(version)['nodes'].get(node)

    def save_node(self, version, node, result):
        with self._mutex:
            if node not in NODES or result.get('status') not in ('succeeded','failed','dependency_failed'):
                raise ValueError('Invalid terminal node')
            if self.node(version,node) is not None:
                raise ValueError('Node is already terminal')
            return self.append(version,'node_result',{**result,'node':node})

    def seal(self, version):
        with self._mutex:
            nodes = self.view(version)['nodes']
            if not all(n in nodes for n in OUTPUT_NODES):
                raise ValueError('Four target outcomes required before seal')
            state = 'succeeded' if all(nodes[n]['status']=='succeeded' for n in OUTPUT_NODES) else 'failed'
            row = self.rows[version]
            summary = {'nodes':{n:{'status':v['status'],'ref':v['ref']} for n,v in nodes.items()}}
            self.stores[row['group']].finish_attempt(version,state,summary)
            row.update(status=state,payload=summary)
            key = self.key(version)
            self.latest[key] = version
            self.pending.pop(key,None)

    def current_rows(self):
        return [dict(self.rows[v]) for v in self.latest.values()]


def hydrate_batch(root, prepared, records):
    root = Path(root)
    destination = root/'prepared/inputs.json'
    payload = {'tasks':[asdict(t) for t in prepared.tasks.values()], 'schemas':prepared.schemas,
               'evaluation':prepared.evaluation,'templates':prepared.templates,'identities':prepared.identities}
    if destination.exists():
        if digest(read_json(destination)) != digest(payload):
            raise ValueError('Prepared inputs differ; create a new batch')
    else:
        write_json(destination,payload)
    counts = {g:{'imported':0,'sealed':0} for g in records.manifest['groups']}
    for key in prepared.tasks:
        if records.current(key):
            continue
        version = records.unfinished(key) or records.begin(key)
        imported = prepared.legacy.get('accepted',{}).get(f'{key.group}/{key.question_id}',{})
        for node in NODES:
            if node not in imported or records.node(version,node):
                continue
            value = dict(imported[node])
            # Only actual dependencies; deterministic decomposition has none.
            names = (('generation_base',) if node.startswith('revision') else
                     ('linking','decomposition') if node.startswith('generation') else
                     ('linking',) if node=='decomposition' and prepared.tasks[key].label=='NESTED' else ())
            value['parent_refs'] = {n:records.node(version,n)['ref'] for n in names}
            records.save_node(version,node,value)
            counts[key.group]['imported'] += 1
        if all(records.node(version,n) for n in OUTPUT_NODES):
            records.seal(version)
            counts[key.group]['sealed'] += 1
    report = {'counts':counts,'rejected':prepared.legacy.get('rejected',[])}
    if not (root/'prepared/import_report.json').exists():
        write_json(root/'prepared/import_report.json',report)
    return report


def load_prepared(root):
    data = read_json(Path(root)/'prepared/inputs.json')
    tasks = {}
    for row in data['tasks']:
        row['key'] = TaskKey(**row['key'])
        task = DinTask(**row)
        tasks[task.key] = task
    return PreparedInputs(tasks,data['schemas'],data['evaluation'],data['templates'],{},data['identities'])
