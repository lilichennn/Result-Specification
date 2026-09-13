"""Frozen campaign configuration and atomic, uniquely claimed work cohorts."""
from contextlib import contextmanager
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from scripts.baseline_adapters.deepeye.run_pipeline import STAGES
from .observations import validate_items


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _config(config):
    if not isinstance(config, dict):
        raise ValueError('campaign config must be a dictionary')
    result = copy.deepcopy(config)
    validate_items(result.get('items'))
    for name in ('tail_fraction', 'poll_seconds'):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if result['tail_fraction'] > 1:
        raise ValueError('tail_fraction must be at most one')
    args = result.get('native_args')
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise ValueError('native_args must be a list of CLI strings')
    for arg in args:
        if arg.split('=', 1)[0] in ('--run-dir', '--item', '--item-key', '--item-keys'):
            raise ValueError('campaign owns run directory and item selection')
    for name in ('env_file', 'rc_lite', 'rc_full', 'python', 'code_root'):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f'{name} must be a nonempty string')
    _json(result)
    return result


_SCHEMA = """
CREATE TABLE config(singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL CHECK(version=1), payload TEXT NOT NULL, checksum TEXT NOT NULL);
CREATE TABLE items(item_key TEXT PRIMARY KEY, position INTEGER NOT NULL UNIQUE);
CREATE TABLE jobs(
 job_id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('native_first','native_retry','rc')),
 source_run TEXT REFERENCES jobs(run_dir), target_stage TEXT,
 run_dir TEXT NOT NULL UNIQUE, state TEXT NOT NULL CHECK(state IN ('planned','prepared','running','finished','blocked','paused')),
 process TEXT, detail TEXT,
 CHECK((kind='rc' AND source_run IS NOT NULL AND target_stage IN ('schema_linking','sql_generation','sql_revision','sql_selection'))
 OR (kind!='rc' AND source_run IS NULL AND target_stage IS NULL)));
CREATE UNIQUE INDEX one_native_first ON jobs(kind) WHERE kind='native_first';
CREATE TABLE claims(
 item_key TEXT NOT NULL REFERENCES items(item_key), scope TEXT NOT NULL,
 job_id TEXT NOT NULL REFERENCES jobs(job_id), position INTEGER NOT NULL,
 PRIMARY KEY(item_key,scope), UNIQUE(job_id,position));
CREATE TABLE gates(stage TEXT PRIMARY KEY CHECK(stage IN ('schema_linking','sql_generation','sql_revision','sql_selection')), anchor_set INTEGER NOT NULL DEFAULT 0 CHECK(anchor_set IN (0,1)));
CREATE TABLE anchors(stage TEXT NOT NULL REFERENCES gates(stage), job_id TEXT NOT NULL REFERENCES jobs(job_id), position INTEGER NOT NULL, PRIMARY KEY(stage,job_id), UNIQUE(stage,position));
CREATE TABLE observations(job_id TEXT NOT NULL REFERENCES jobs(job_id), item_key TEXT NOT NULL REFERENCES items(item_key), status TEXT NOT NULL CHECK(status IN ('pending','unfinished','succeeded','failed')), attempt_id TEXT, finished_at TEXT, PRIMARY KEY(job_id,item_key));
CREATE TABLE canonical(item_key TEXT PRIMARY KEY REFERENCES items(item_key), job_id TEXT NOT NULL REFERENCES jobs(job_id), attempt_id TEXT NOT NULL);
CREATE TABLE events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TRIGGER frozen_config_update BEFORE UPDATE ON config BEGIN SELECT RAISE(ABORT,'config is immutable'); END;
CREATE TRIGGER frozen_config_delete BEFORE DELETE ON config BEGIN SELECT RAISE(ABORT,'config is immutable'); END;
CREATE TRIGGER frozen_claims_update BEFORE UPDATE ON claims BEGIN SELECT RAISE(ABORT,'claims are immutable'); END;
CREATE TRIGGER frozen_claims_delete BEFORE DELETE ON claims BEGIN SELECT RAISE(ABORT,'claims are immutable'); END;
CREATE TRIGGER frozen_canonical_update BEFORE UPDATE ON canonical BEGIN SELECT RAISE(ABORT,'canonical sources are immutable'); END;
CREATE TRIGGER frozen_canonical_delete BEFORE DELETE ON canonical BEGIN SELECT RAISE(ABORT,'canonical sources are immutable'); END;
CREATE TRIGGER frozen_anchors_update BEFORE UPDATE ON anchors BEGIN SELECT RAISE(ABORT,'anchors are immutable'); END;
CREATE TRIGGER frozen_anchors_delete BEFORE DELETE ON anchors BEGIN SELECT RAISE(ABORT,'anchors are immutable'); END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
CREATE TRIGGER job_identity_no_update BEFORE UPDATE OF job_id,kind,source_run,target_stage,run_dir ON jobs BEGIN SELECT RAISE(ABORT,'job identity is immutable'); END;
"""


class CampaignLedger:
    def __init__(self, campaign_dir, connection, *, read_only=False):
        self.campaign_dir = Path(campaign_dir).resolve()
        self._db = connection
        self._read_only = read_only
        self._closed = False

    @staticmethod
    def _connect(path, read_only=False):
        mode = 'ro' if read_only else 'rw'
        db = sqlite3.connect(f'{path.as_uri()}?mode={mode}', uri=True, isolation_level=None, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        if read_only:
            db.execute('PRAGMA query_only=ON')
        else:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
        return db

    @classmethod
    def create(cls, campaign_dir, config):
        config = _config(config)
        path = Path(campaign_dir).resolve()
        path.mkdir()
        (path / 'runs').mkdir()
        db_path = path / 'campaign.sqlite3'
        # Exclusive creation prevents silently adopting another campaign.
        with db_path.open('xb'):
            pass
        db = cls._connect(db_path)
        ledger = cls(path, db)
        try:
            db.executescript(_SCHEMA)
            with ledger.transaction():
                payload = _json(config)
                db.execute('INSERT INTO config VALUES(1,1,?,?)', (payload, _digest(payload)))
                db.executemany('INSERT INTO items VALUES(?,?)', [(key, n) for n, key in enumerate(config['items'])])
                ledger.append_event('campaign_created', {'config_checksum': _digest(payload)})
                ledger.claim_job(kind='native_first', items=config['items'])
            return ledger
        except BaseException:
            ledger.close()
            raise

    @classmethod
    def open(cls, campaign_dir, *, read_only=False):
        path = Path(campaign_dir).resolve()
        if not (path / 'campaign.sqlite3').is_file():
            raise FileNotFoundError(path / 'campaign.sqlite3')
        ledger = cls(path, cls._connect(path / 'campaign.sqlite3', read_only), read_only=read_only)
        try:
            config = ledger.config
            if ledger._db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('campaign foreign-key corruption')
            stored = [row[0] for row in ledger._db.execute('SELECT item_key FROM items ORDER BY position')]
            if stored != config['items']:
                raise ValueError('campaign item bindings differ from config')
            jobs = ledger.jobs()
            if len([job for job in jobs if job['kind'] == 'native_first']) != 1:
                raise ValueError('campaign requires one first-pass job')
            for job in jobs:
                if job['job_id'] != ledger._job_id(job['kind'], job['items'], job['source_run'], job['target_stage']):
                    raise ValueError('campaign job identity mismatch')
            return ledger
        except BaseException:
            ledger.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self._closed:
            self._db.close()
            self._closed = True

    @property
    def config(self):
        row = self._db.execute('SELECT * FROM config WHERE singleton=1').fetchone()
        if row is None or row['version'] != 1 or _digest(row['payload']) != row['checksum']:
            raise ValueError('campaign config is missing or corrupt')
        return _config(json.loads(row['payload']))

    @contextmanager
    def transaction(self):
        """A tick and nested public mutations share an all-or-nothing commit."""
        if self._read_only:
            raise PermissionError('campaign ledger is read-only')
        nested = self._db.in_transaction
        self._db.execute('SAVEPOINT campaign_nested' if nested else 'BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            if nested:
                self._db.execute('ROLLBACK TO campaign_nested')
                self._db.execute('RELEASE campaign_nested')
            else:
                self._db.rollback()
            raise
        else:
            if nested:
                self._db.execute('RELEASE campaign_nested')
            else:
                self._db.commit()

    def jobs(self):
        result = []
        for row in self._db.execute('SELECT * FROM jobs ORDER BY rowid'):
            job = dict(row)
            job['items'] = [member[0] for member in self._db.execute('SELECT item_key FROM claims WHERE job_id=? ORDER BY position', (job['job_id'],))]
            validate_items(job['items'])
            if Path(job['run_dir']).resolve() != self.campaign_dir / 'runs' / job['job_id']:
                raise ValueError('job directory is outside its campaign location')
            for field in ('process', 'detail'):
                job[field] = json.loads(job[field]) if job[field] is not None else None
            result.append(job)
        return result

    @staticmethod
    def _job_id(kind, items, source_run, target_stage):
        payload = _json([kind, sorted(items), source_run, target_stage])
        return f'{kind}-{_digest(payload)[:24]}'

    def claim_job(self, *, kind, items, source_run=None, target_stage=None):
        items = sorted(validate_items(items))
        if kind not in ('native_first', 'native_retry', 'rc'):
            raise ValueError('unknown campaign job kind')
        if kind == 'rc':
            if target_stage not in STAGES or not isinstance(source_run, (str, Path)):
                raise ValueError('RC jobs require a native source and a target stage')
            source_run = str(Path(source_run).resolve())
        elif source_run is not None or target_stage is not None:
            raise ValueError('native jobs cannot specify source or target stage')
        with self.transaction():
            if not set(items).issubset(self.config['items']):
                raise ValueError('job contains tasks outside the campaign')
            if kind == 'native_first' and set(items) != set(self.config['items']):
                raise ValueError('first-pass job must contain every campaign task')
            if kind == 'rc':
                source = next((job for job in self.jobs() if job['run_dir'] == source_run), None)
                if source is None or source['kind'] == 'rc' or not set(items).issubset(source['items']):
                    raise ValueError('RC items do not belong to the native source job')
            job_id = self._job_id(kind, items, source_run, target_stage)
            run_dir = str(self.campaign_dir / 'runs' / job_id)
            self._db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL,NULL)',
                             (job_id, kind, source_run, target_stage, run_dir, 'planned'))
            scope = target_stage if kind == 'rc' else kind
            self._db.executemany('INSERT INTO claims VALUES(?,?,?,?)',
                                 [(key, scope, job_id, n) for n, key in enumerate(items)])
            self.append_event('job_claimed', {'job_id': job_id, 'kind': kind, 'items': items,
                                             'source_run': source_run, 'target_stage': target_stage})
            return next(job for job in self.jobs() if job['job_id'] == job_id)

    def update_job(self, job_id, *, state=None, process=None, detail=None):
        with self.transaction():
            job = self._db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if job is None:
                raise ValueError('unknown campaign job')
            if process is not None and not isinstance(process, dict):
                raise ValueError('process metadata must be a dictionary')
            self._db.execute('UPDATE jobs SET state=?,process=?,detail=? WHERE job_id=?',
                             (state if state is not None else job['state'],
                              _json(process) if process is not None else job['process'],
                              _json(detail) if detail is not None else job['detail'], job_id))
            self.append_event('job_updated', {'job_id': job_id, 'state': state, 'process': process, 'detail': detail})

    def open_stage(self, stage):
        if stage not in STAGES:
            raise ValueError('unknown RC stage')
        with self.transaction():
            if stage not in self.opened_stages():
                self._db.execute('INSERT INTO gates(stage) VALUES(?)', (stage,))
                self.append_event('stage_opened', {'stage': stage})

    def opened_stages(self):
        opened = {row[0] for row in self._db.execute('SELECT stage FROM gates')}
        return [stage for stage in STAGES if stage in opened]

    def set_anchor(self, stage, job_ids):
        if not isinstance(job_ids, list) or not job_ids or len(set(job_ids)) != len(job_ids):
            raise ValueError('anchor must contain unique jobs and be nonempty')
        with self.transaction():
            gate = self._db.execute('SELECT anchor_set FROM gates WHERE stage=?', (stage,)).fetchone()
            if gate is None:
                raise ValueError('cannot anchor an unopened stage')
            if gate[0]:
                if job_ids != self.anchor(stage):
                    raise ValueError('stage anchor is already frozen')
                return
            jobs = {job['job_id']: job for job in self.jobs()}
            if any(job_id not in jobs or jobs[job_id]['target_stage'] != stage for job_id in job_ids):
                raise ValueError('anchor contains a job for another stage')
            self._db.executemany('INSERT INTO anchors VALUES(?,?,?)', [(stage, key, n) for n, key in enumerate(job_ids)])
            self._db.execute('UPDATE gates SET anchor_set=1 WHERE stage=?', (stage,))
            self.append_event('stage_anchored', {'stage': stage, 'job_ids': job_ids})

    def anchor(self, stage):
        return [row[0] for row in self._db.execute('SELECT job_id FROM anchors WHERE stage=? ORDER BY position', (stage,))]

    def append_event(self, kind, payload):
        if not isinstance(kind, str) or not kind:
            raise ValueError('event kind must be a nonempty string')
        with self.transaction():
            self._db.execute('INSERT INTO events(kind,payload,created_at) VALUES(?,?,?)',
                             (kind, _json(payload), dt.datetime.now(dt.timezone.utc).isoformat()))
