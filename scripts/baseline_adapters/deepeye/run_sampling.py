"""Append-only sampling checkpoints; retry policy remains in app.llm.sampling.

The in-memory index is verified once per recorder and updated only after durable
commits. A started attempt without a committed terminal is uncertain and spent.
No claim of exactly-once external inference is made.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
from pathlib import Path
import threading

from .precompute_cache import fingerprint
from .run_store import to_jsonable

_NODE = ContextVar('deepeye_stable_sampling_node', default=())
_KINDS = ('sampling_group_bound', 'sample_attempt_started', 'sample_attempt_finished',
          'sample_checkpoint', 'sample_attempt_uncertain', 'sample_result', 'sampling_retry_authorized')


def implementation_version():
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256(b'deepeye-durable-sampling-v1')
    for folder in ('baselines/DeepEye-SQL/app', 'scripts/baseline_adapters/deepeye',
                   'scripts/rc_evaluation/deepeye'):
        for path in sorted((root / folder).rglob('*.py')):
            if 'tests' not in path.parts:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def stable_sampling_node(node):
    token = _NODE.set((*_NODE.get(), node))
    try:
        yield
    finally:
        _NODE.reset(token)


class SamplingCheckpoints:
    def __init__(self, store, on_error=None):
        self.store = store
        self.on_error = on_error
        self.manifest = store.manifest
        self.manifest_fingerprint = fingerprint(to_jsonable(self.manifest))
        self.rc_blocks = {}
        self.source_version = implementation_version()
        self.lock = threading.RLock()
        self.groups, self.samples, self.starts, self.terminals, self.uncertain = {}, {}, {}, set(), set()
        self.sample_starts = {}
        self._attempts = {}
        self.allowances, self.renewal_starts = {}, {}
        self.live_attempts = set()
        self.sample_results = set()
        for event in store.iter_events(kinds=_KINDS):
            self._index(event)

    def _index(self, event):
        kind, payload = event['kind'], event['payload']
        group = payload['group_id']
        if kind == 'sample_result':
            key = (event['attempt_id'], group, payload['sample_index'])
            if key in self.sample_results:
                raise ValueError('duplicate sample result within stage attempt')
            self.sample_results.add(key)
            reference = payload.get('restored_from_event')
            if reference is not None:
                from .run_trace import _prepare_value
                prior = self.samples.get((group, payload['sample_index']))
                if prior is None or prior['event_id'] != reference:
                    raise ValueError('invalid restored sample reference')
                self._validate_stage_link(event, self.groups[group]['identity'])
                outcome = prior['payload']['outcome']
                if (not outcome.succeeded or payload.get('succeeded') is not True
                        or payload.get('result') != to_jsonable(_prepare_value(outcome.result))
                        or payload.get('usage') != outcome.usage
                        or payload.get('response_id') != outcome.response_id
                        or payload.get('rc_applied', False) != outcome.rc_applied):
                    raise ValueError('restored sample differs from durable success')
            return
        if kind == 'sampling_group_bound':
            if group in self.groups or fingerprint(payload['identity']) != group:
                raise ValueError('duplicate or invalid sampling group binding')
            if type(payload['target_n']) is not int or payload['target_n'] < 1 or type(payload['max_attempts']) is not int or not 1 <= payload['max_attempts'] <= 4:
                raise ValueError('invalid sampling group budget')
            if payload['identity']['manifest'] != self.manifest_fingerprint:
                raise ValueError('sampling group manifest mismatch')
            self._validate_stage_link(event, payload['identity'])
            self.groups[group] = payload
            return
        if group not in self.groups:
            raise ValueError('sample event without bound group')
        self._validate_stage_link(event, self.groups[group]['identity'])
        if kind == 'sampling_retry_authorized':
            if not isinstance(payload.get('decision'), str) or not payload['decision'].strip():
                raise ValueError('retry allowance requires explicit decision')
            seen = set()
            for allowance in payload['allowances']:
                index = allowance['sample_index']
                key = (group, index)
                limit = self.limit(group, index)
                prior = self.samples.get(key)
                consumed = len(self.sample_starts.get(key, ()))
                fatal = prior and prior['payload']['outcome'].fatal
                if (index in seen or type(index) is not int or not 0 <= index < self.groups[group]['target_n']
                        or allowance['prior_limit'] != limit or allowance['attempt_limit'] != consumed + 4
                        or (consumed < limit and not (fatal and consumed > self.renewal_starts.get(key, -1)))
                        or (prior and prior['payload']['outcome'].succeeded)):
                    raise ValueError('invalid sample retry allowance')
                seen.add(index)
                self.allowances[key] = allowance['attempt_limit']
                self.renewal_starts[key] = len(self.sample_starts.get(key, ()))
            return
        index = payload['sample_index']
        if type(index) is not int or not 0 <= index < self.groups[group]['target_n']:
            raise ValueError('invalid sample index')
        key = (group, index)
        number = payload['sample_attempt']
        if type(number) is not int or number < 1:
            raise ValueError('invalid sample attempt number')
        identity = (*key, number)
        if kind == 'sample_attempt_started':
            expected = 1 + len(self.sample_starts.get(key, ()))
            if (identity in self.starts or number != expected or number > self.limit(group, index)
                    or (key in self.samples and self.samples[key]['payload']['outcome'].succeeded)):
                raise ValueError('duplicate, exhausted or late sample attempt')
            self.starts[identity] = event
            self.sample_starts.setdefault(key, []).append(identity)
        elif kind == 'sample_attempt_uncertain':
            if identity not in self.starts or identity in self.terminals or identity in self.uncertain:
                raise ValueError('uncertain attempt lacks unique unfinished start')
            self.uncertain.add(identity)
        else:
            start = self.starts.get(identity)
            if start is None or start['attempt_id'] != event['attempt_id'] or identity in self.terminals:
                raise ValueError('sample terminal lacks unique same-stage-attempt start')
            outcome = payload['outcome']
            if (outcome.group_id != group or outcome.sample_index != index
                    or len(outcome.attempts) != number
                    or [a.attempt_number for a in outcome.attempts] != list(range(1, number + 1))
                    or outcome.succeeded != (kind == 'sample_checkpoint')
                    or (outcome.succeeded and (outcome.attempts[-1].status != 'succeeded'
                                              or outcome.usage != outcome.attempts[-1].usage))):
                raise ValueError('invalid durable sample outcome')
            prior = self.samples.get(key)
            if prior and prior['payload']['outcome'].succeeded:
                raise ValueError('duplicate sample success')
            self.terminals.add(identity)
            self.samples[key] = event

    def _validate_stage_link(self, event, identity):
        attempt_id = event['attempt_id']
        if attempt_id not in self._attempts:
            self._attempts[attempt_id] = self.store.attempt(attempt_id)
        attempt = self._attempts[attempt_id]
        if any(identity[field] != attempt[column] for field, column in (
                ('item_key', 'item_key'), ('stage', 'stage'), ('input', 'input_fingerprint'))):
            raise ValueError('sampling event crosses stage/input lineage')

    def validate_stage(self, row):
        version = row['payload'].get('sampling_implementation_version')
        if version is not None and version != self.source_version:
            raise ValueError('Completed stage sampling implementation changed; use a new run')
        if row['payload'].get('sampling') and version is None:
            raise ValueError('Legacy sampling checkpoint is read-only under this implementation')

    def limit(self, group, index):
        return self.allowances.get((group, index), self.groups[group]['max_attempts'])

    def renew_exhausted(self, group_id, *, attempt_id, decision):
        """Authorize another <=4-request round; never called by ordinary resume.

        The caller creates an audit stage attempt with the same task/input hash.
        Successful samples and unfinished indices with remaining budget retain
        their allowance. This API does not launch work or mutate old records.
        """
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError('retry allowance requires explicit decision')
        with self.lock:
            group = self.groups[group_id]
            self._validate_stage_link({'attempt_id': attempt_id}, group['identity'])
            if group['identity']['source_version'] != self.source_version:
                raise ValueError('cannot renew samples from another implementation')
            allowances = []
            for index in range(group['target_n']):
                key = (group_id, index)
                prior = self.samples.get(key)
                outcome = prior['payload']['outcome'] if prior else None
                if outcome and outcome.succeeded:
                    continue
                starts = self.sample_starts.get(key, ())
                if any(self.starts[identity]['attempt_id'] in self.live_attempts for identity in starts if identity not in self.terminals):
                    raise ValueError('cannot renew an in-flight sample')
                limit = self.limit(group_id, index)
                if len(starts) < limit and not (outcome and outcome.fatal and len(starts) > self.renewal_starts.get(key, -1)):
                    continue
                allowances.append({'sample_index': index, 'prior_limit': limit, 'attempt_limit': len(starts) + 4})
            if not allowances:
                raise ValueError('no exhausted samples eligible for renewal')
            self.append(attempt_id, 'sampling_retry_authorized', {
                'group_id': group_id, 'decision': decision, 'allowances': allowances})

    def append(self, attempt_id, kind, payload):
        # Callers hold the lock; publish the index only after the durable commit.
        event = {'attempt_id': attempt_id, 'kind': kind, 'payload': payload, 'event_id': None}
        try:
            if kind == 'sample_result' and (attempt_id, payload['group_id'], payload['sample_index']) in self.sample_results:
                raise ValueError('duplicate sample result within stage attempt')
            event['event_id'] = self.store.append_event(attempt_id, kind, payload)
            self._index(event)
        except BaseException as error:
            if self.on_error is not None:
                self.on_error(error)
            raise
        return event

    def session(self, attempt_id, recorder):
        return SamplingSession(self, self.store.attempt(attempt_id), recorder)


class SamplingSession:
    def __init__(self, checkpoints, attempt, recorder):
        self.checkpoints, self.attempt, self.recorder = checkpoints, attempt, recorder
        self.invocations = {}
        self.group_ids = set()
        self.closed = False
        with checkpoints.lock:
            checkpoints.live_attempts.add(attempt['attempt_id'])

    def close(self):
        with self.checkpoints.lock:
            self.closed = True
            self.checkpoints.live_attempts.discard(self.attempt['attempt_id'])

    def check_stop(self):
        from app.llm.sampling import SamplingPaused
        self.recorder.raise_if_failed()
        if self.closed:
            raise ValueError('sampling session is closed')
        if self.recorder.stop_event.is_set():
            raise SamplingPaused('Sampling paused at request/node boundary')

    def group(self, request_identity, n, max_attempts):
        self.check_stop()
        try:
            return self._group(request_identity, n, max_attempts)
        except Exception as error:
            self.recorder._remember_error(error)
            raise

    def _group(self, request_identity, n, max_attempts):
        if request_identity is None:
            raise ValueError('durable sampling requires explicit request/parser identity')
        cp = self.checkpoints
        with cp.lock:
            node = _NODE.get()
            position = self.invocations.get(node, 0)
            self.invocations[node] = position + 1
            identity = {'item_key': self.attempt['item_key'], 'stage': self.attempt['stage'],
                        'input': self.attempt['input_fingerprint'],
                        'manifest': cp.manifest_fingerprint,
                        'source_version': cp.source_version, 'node': node, 'invocation': position,
                        'request': fingerprint(to_jsonable(request_identity)), 'n': n, 'max_attempts': max_attempts}
            group_id = fingerprint(to_jsonable(identity))
            if group_id not in cp.groups:
                rc_applied = False
                manifest = cp.manifest
                contract = manifest.get('contracts', {}).get(self.attempt['item_key'])
                if (manifest.get('condition') == 'rc' and manifest.get('target_stage') == self.attempt['stage']
                        and contract is not None):
                    from scripts.rc_evaluation.deepeye.contracts import render_rc_block
                    item_key = self.attempt['item_key']
                    if item_key not in cp.rc_blocks:
                        cp.rc_blocks[item_key] = render_rc_block(contract)
                    block = cp.rc_blocks[item_key]
                    rc_applied = any(block in str(message.get('content', ''))
                                     for message in request_identity.get('messages', ()) if isinstance(message, dict))
                cp.append(self.attempt['attempt_id'], 'sampling_group_bound', {
                    'group_id': group_id, 'target_n': n, 'max_attempts': max_attempts,
                    'identity': to_jsonable(identity), 'rc_applied': rc_applied})
            self.group_ids.add(group_id)
            return group_id

    def restore(self, group, index):
        from app.llm.sampling import SampleOutcome, SampleAttempt
        cp = self.checkpoints
        with cp.lock:
            key = (group, index)
            prior = cp.samples.get(key)
            outcome = copy.deepcopy(prior['payload']['outcome']) if prior else SampleOutcome(group, index)
            outcome.rc_applied = cp.groups[group].get('rc_applied', False)
            if outcome.succeeded:
                outcome.restored_from_event = prior['event_id']
                return outcome
            pending = [identity for identity in cp.sample_starts.get(key, ()) if identity not in cp.terminals]
            for identity in pending:
                number = identity[2]
                if identity not in cp.uncertain:
                    cp.append(self.attempt['attempt_id'], 'sample_attempt_uncertain', {
                        'group_id': group, 'sample_index': index, 'sample_attempt': number,
                        'started_event': cp.starts[identity]['event_id'],
                        'reason': 'no_durable_terminal_remote_execution_unknown'})
                if number > len(outcome.attempts):
                    outcome.attempts.append(SampleAttempt(number, 'uncertain', None,
                                                         'No durable result; prior attempt consumed'))
            if len(outcome.attempts) == cp.renewal_starts.get(key):
                outcome.fatal = False
            return outcome

    def attempt_limit(self, group, index):
        with self.checkpoints.lock:
            return self.checkpoints.limit(group, index)

    def start_attempt(self, identity):
        cp = self.checkpoints
        with cp.lock:
            self.check_stop()
            if identity['group_id'] not in self.group_ids:
                raise ValueError('sample starts outside session group')
            key = (identity['group_id'], identity['sample_index'], identity['sample_attempt'])
            if key in cp.starts:
                raise ValueError('sample attempt already started')
            cp.append(self.attempt['attempt_id'], 'sample_attempt_started', identity)

    def finish_sample_attempt(self, identity, outcome):
        cp = self.checkpoints
        with cp.lock:
            key = (identity['group_id'], identity['sample_index'], identity['sample_attempt'])
            start = cp.starts.get(key)
            if self.closed or start is None or start['attempt_id'] != self.attempt['attempt_id'] or key in cp.terminals or key in cp.uncertain:
                raise ValueError('late or mismatched sample terminal')
            cp.append(self.attempt['attempt_id'], 'sample_checkpoint' if outcome.succeeded else 'sample_attempt_finished',
                      {**identity, 'outcome': copy.deepcopy(outcome)})
