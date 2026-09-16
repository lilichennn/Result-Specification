"""Transactional current-version pointers; execution facts remain in RunStore."""

from contextlib import contextmanager
from dataclasses import astuple
from pathlib import Path
import sqlite3
import threading
import uuid

from .config import MODES, TaskKey


class CurrentIndex:
    def __init__(self, path: Path, *, read_only: bool = False):
        path = Path(path)
        self._read_only = read_only
        self._lock = threading.RLock()
        if read_only:
            path = path.resolve(strict=True)
            self._db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30,
                                       isolation_level=None, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS current_versions (
                batch TEXT NOT NULL, grp TEXT NOT NULL, question TEXT NOT NULL,
                version TEXT, epoch INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(batch, grp, question));
            CREATE TABLE IF NOT EXISTS leases (
                batch TEXT NOT NULL, grp TEXT NOT NULL, question TEXT NOT NULL,
                token TEXT NOT NULL, expected_epoch INTEGER NOT NULL, expected_version TEXT,
                PRIMARY KEY(batch, grp, question));
            CREATE TABLE IF NOT EXISTS mode_states (
                batch TEXT NOT NULL, grp TEXT NOT NULL, question TEXT NOT NULL,
                version TEXT NOT NULL, mode TEXT NOT NULL, event_id TEXT NOT NULL,
                PRIMARY KEY(batch, grp, question, version, mode));
            CREATE TABLE IF NOT EXISTS group_starts (
                batch TEXT NOT NULL, grp TEXT NOT NULL, PRIMARY KEY(batch, grp));
            CREATE TABLE IF NOT EXISTS campaign_assignments (
                batch TEXT NOT NULL, grp TEXT NOT NULL, question TEXT NOT NULL,
                version TEXT NOT NULL, first_version TEXT NOT NULL,
                state TEXT NOT NULL, error TEXT,
                PRIMARY KEY(batch, grp, question));
            CREATE TABLE IF NOT EXISTS campaign_outcomes (
                batch TEXT NOT NULL, grp TEXT NOT NULL, question TEXT NOT NULL,
                version TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(batch, grp, question, version, mode));
            CREATE TABLE IF NOT EXISTS campaign_waits (
                batch TEXT NOT NULL, grp TEXT NOT NULL, reason TEXT,
                PRIMARY KEY(batch, grp));
            CREATE TABLE IF NOT EXISTS prepared_groups (
                batch TEXT NOT NULL, grp TEXT NOT NULL, fingerprint TEXT NOT NULL,
                PRIMARY KEY(batch, grp));
        """)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self._lock:
            self._db.close()

    @contextmanager
    def _transaction(self):
        if self._read_only:
            raise PermissionError('CurrentIndex is read-only')
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    def _check_lease(self, key, lease):
        row = self._db.execute("SELECT token,expected_epoch,expected_version FROM leases "
                               "WHERE batch=? AND grp=? AND question=?", astuple(key)).fetchone()
        current = self._db.execute("SELECT epoch,version FROM current_versions "
                                   "WHERE batch=? AND grp=? AND question=?", astuple(key)).fetchone()
        if (row is None or dict(row) != lease or current is None
                or current["epoch"] != lease["expected_epoch"]
                or current["version"] != lease["expected_version"]):
            raise ValueError("stale or foreign lease")

    def claim(self, key: TaskKey) -> dict:
        with self._transaction():
            return self._claim(key)

    def _claim(self, key):
        self._db.execute("INSERT OR IGNORE INTO current_versions(batch,grp,question) VALUES(?,?,?)", astuple(key))
        row = self._db.execute("SELECT epoch,version FROM current_versions WHERE batch=? AND grp=? AND question=?", astuple(key)).fetchone()
        lease = {"token": uuid.uuid4().hex, "expected_epoch": row["epoch"], "expected_version": row["version"]}
        try:
            self._db.execute("INSERT INTO leases VALUES(?,?,?,?,?,?)", (*astuple(key), *lease.values()))
        except sqlite3.IntegrityError as exc:
            raise ValueError("question already has an active claim") from exc
        return lease

    def recover(self, key: TaskKey, lease: dict) -> dict:
        """Rotate a known lease after the campaign proves exclusive ownership.

        Possession of the exact old lease is required. Opening the database never
        recovers leases and there is no time-based automatic stealing.
        """
        with self._transaction():
            return self._recover(key, lease)

    def _recover(self, key, lease):
        self._check_lease(key, lease)
        fresh = {**lease, "token": uuid.uuid4().hex}
        self._db.execute("UPDATE leases SET token=? WHERE batch=? AND grp=? AND question=?", (fresh["token"], *astuple(key)))
        return fresh

    def release(self, key: TaskKey, lease: dict) -> None:
        with self._transaction():
            self._check_lease(key, lease)
            self._db.execute("DELETE FROM leases WHERE batch=? AND grp=? AND question=?", astuple(key))

    def publish(self, key: TaskKey, version_id: str, lease: dict, verify_sealed) -> None:
        if self._read_only:
            raise PermissionError('CurrentIndex is read-only')
        if not isinstance(version_id, str) or not version_id or not verify_sealed(key, version_id):
            raise ValueError("version must be durably sealed for this question")
        with self._transaction():
            self._publish(key, version_id, lease)

    def _publish(self, key, version_id, lease):
        self._check_lease(key, lease)
        changed = self._db.execute(
                "UPDATE current_versions SET version=?,epoch=epoch+1 "
                "WHERE batch=? AND grp=? AND question=? AND epoch=? AND version IS ? "
                "AND EXISTS (SELECT 1 FROM leases WHERE batch=? AND grp=? AND question=? AND token=?)",
                (version_id, *astuple(key), lease["expected_epoch"], lease["expected_version"], *astuple(key), lease["token"])).rowcount
        if changed != 1:
            raise ValueError("publication compare-and-swap failed")
        self._db.execute("DELETE FROM leases WHERE batch=? AND grp=? AND question=?", astuple(key))

    def current(self, key: TaskKey) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT version FROM current_versions WHERE batch=? AND grp=? AND question=?", astuple(key)).fetchone()
            return row[0] if row else None

    def snapshot(self, group: str | None = None) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT batch,grp,question,version FROM current_versions" +
                                    (" WHERE grp=?" if group is not None else ""),
                                    (group,) if group is not None else ()).fetchall()
            return {TaskKey(*row[:3]): row[3] for row in rows}

    def record_mode(self, key: TaskKey, version_id: str, mode: str, event_id: str, lease: dict) -> None:
        """Cache an immutable terminal mode event pointer, never duplicate status."""
        if mode not in MODES or not version_id or not event_id:
            raise ValueError("valid version, mode and source event required")
        with self._transaction():
            self._record_mode(key, version_id, mode, event_id, lease)

    def _record_mode(self, key, version_id, mode, event_id, lease):
        self._check_lease(key, lease)
        args = (*astuple(key), version_id, mode)
        row = self._db.execute("SELECT event_id FROM mode_states WHERE batch=? AND grp=? AND question=? AND version=? AND mode=?", args).fetchone()
        if row and row[0] != event_id:
            raise ValueError("mode already points to a different source")
        self._db.execute("INSERT OR IGNORE INTO mode_states VALUES(?,?,?,?,?,?)", (*args, event_id))

    def claim_assignment(self, key, version_id):
        if not isinstance(version_id, str) or not version_id:
            raise ValueError('version required')
        with self._transaction():
            lease = self._claim(key)
            self._db.execute('INSERT INTO campaign_assignments VALUES(?,?,?,?,?,?,NULL) '
                'ON CONFLICT(batch,grp,question) DO UPDATE SET version=excluded.version,state=excluded.state,error=NULL',
                (*astuple(key), version_id, version_id, 'active'))
            return lease

    def assignment(self, key):
        with self._lock:
            row = self._db.execute('SELECT a.*,l.token,l.expected_epoch,l.expected_version '
                'FROM campaign_assignments a LEFT JOIN leases l USING(batch,grp,question) '
                'WHERE a.batch=? AND a.grp=? AND a.question=?', astuple(key)).fetchone()
            if row is None:
                return None
            value = dict(row)
            value['lease'] = {field: value.pop(field) for field in ('token', 'expected_epoch', 'expected_version')}
            if value['lease']['token'] is None:
                value['lease'] = None
            return value

    def recover_assignment(self, key):
        """Caller MUST hold the exclusive batch owner lock; never expires by age."""
        if self._read_only:
            raise PermissionError('CurrentIndex is read-only')
        with self._transaction():
            assignment = self.assignment(key)
            if not assignment or assignment['lease'] is None:
                raise ValueError('no active assignment')
            fresh = self._recover(key, assignment['lease'])
            self._db.execute("UPDATE campaign_assignments SET state='active',error=NULL WHERE batch=? AND grp=? AND question=?", astuple(key))
            return fresh

    def finish_assignment(self, key, version_id, lease, verify_sealed):
        if self._read_only:
            raise PermissionError('CurrentIndex is read-only')
        if not verify_sealed(key, version_id):
            raise ValueError('version must be durably sealed for this question')
        with self._transaction():
            self._check_assignment(key, version_id)
            self._publish(key, version_id, lease)
            self._db.execute("UPDATE campaign_assignments SET state='done',error=NULL WHERE batch=? AND grp=? AND question=?", astuple(key))

    def _check_assignment(self, key, version_id):
        row = self._db.execute('SELECT version FROM campaign_assignments WHERE batch=? AND grp=? AND question=?', astuple(key)).fetchone()
        if not row or row[0] != version_id:
            raise ValueError('foreign assignment')

    def pause_assignment(self, key, version_id, lease, error):
        with self._transaction():
            self._check_lease(key, lease)
            self._check_assignment(key, version_id)
            self._db.execute("UPDATE campaign_assignments SET state='paused',error=? WHERE batch=? AND grp=? AND question=?", (error, *astuple(key)))

    def record_campaign_mode(self, key, version_id, mode, event_id, lease, status):
        if mode not in MODES or not event_id or status not in ('succeeded', 'failed', 'dependency_failed'):
            raise ValueError('terminal source required')
        with self._transaction():
            self._check_assignment(key, version_id)
            self._record_mode(key, version_id, mode, event_id, lease)
            args = (*astuple(key), version_id, mode)
            row = self._db.execute('SELECT status FROM campaign_outcomes WHERE batch=? AND grp=? AND question=? AND version=? AND mode=?', args).fetchone()
            if row and row[0] != status:
                raise ValueError('terminal outcome changed')
            self._db.execute('INSERT OR IGNORE INTO campaign_outcomes VALUES(?,?,?,?,?,?)', (*args, status))

    def campaign_counts(self, batch_id, group):
        with self._lock:
            rows = self._db.execute('SELECT o.mode,COUNT(*),SUM(o.status != \'succeeded\') '
                'FROM campaign_outcomes o JOIN campaign_assignments a USING(batch,grp,question) '
                'WHERE o.batch=? AND o.grp=? AND o.version=a.first_version GROUP BY o.mode', (batch_id, group)).fetchall()
            states = dict(self._db.execute('SELECT state,COUNT(*) FROM campaign_assignments WHERE batch=? AND grp=? GROUP BY state', (batch_id, group)).fetchall())
            errors = dict(self._db.execute('SELECT error,COUNT(*) FROM campaign_assignments WHERE batch=? AND grp=? AND error IS NOT NULL GROUP BY error', (batch_id, group)).fetchall())
            waiting = self._db.execute('SELECT reason FROM campaign_waits WHERE batch=? AND grp=?', (batch_id, group)).fetchone()
            return {'terminal': {mode: next((r[1] for r in rows if r[0] == mode), 0) for mode in MODES},
                    'failed': {mode: next((r[2] for r in rows if r[0] == mode), 0) for mode in MODES},
                    'states': states, 'errors': errors, 'waiting': waiting[0] if waiting else None}

    def set_waiting(self, batch_id, group, reason):
        with self._transaction():
            self._db.execute('INSERT INTO campaign_waits VALUES(?,?,?) ON CONFLICT(batch,grp) DO UPDATE SET reason=excluded.reason', (batch_id, group, reason))

    def prepared_group(self, batch_id, group):
        with self._lock:
            row = self._db.execute('SELECT fingerprint FROM prepared_groups WHERE batch=? AND grp=?', (batch_id, group)).fetchone()
            return row[0] if row else None

    def bind_prepared_group(self, batch_id, group, fingerprint):
        if not batch_id or not group or not fingerprint:
            raise ValueError('prepared group binding required')
        with self._transaction():
            previous = self.prepared_group(batch_id, group)
            if previous is not None and previous != fingerprint:
                raise ValueError('prepared group content changed')
            self._db.execute('INSERT OR IGNORE INTO prepared_groups VALUES(?,?,?)', (batch_id, group, fingerprint))

    def mode_states(self, key: TaskKey, version_id: str) -> dict:
        with self._lock:
            return dict(self._db.execute("SELECT mode,event_id FROM mode_states WHERE batch=? AND grp=? AND question=? AND version=?", (*astuple(key), version_id)).fetchall())

    def mark_group_started(self, batch_id: str, group: str) -> None:
        if not batch_id or not group:
            raise ValueError("batch and group required")
        with self._transaction():
            self._db.execute("INSERT OR IGNORE INTO group_starts VALUES(?,?)", (batch_id, group))

    def group_started(self, batch_id: str, group: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM group_starts WHERE batch=? AND grp=?", (batch_id, group)).fetchone() is not None
