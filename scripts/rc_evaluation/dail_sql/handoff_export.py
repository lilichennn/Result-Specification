"""Seal current DAIL facts and an existing compact SQL evaluation for Linking."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import zlib

from scripts.analysis.export_dail_facts import export as export_facts
from .linking_package import _authenticate, _info, _read, _sha, _snapshot, _write


FORMAT = 'dail-offline-handoff-v3'


def _versions(rows):
    result = {}
    for row in rows:
        key = row['task_key']
        identity = (str(key['batch_id']), str(key['group']), str(key['question_id']))
        if identity in result or row['version_id'] is None:
            raise ValueError(f'duplicate or pending DAIL version: {identity}')
        result[identity] = row['version_id']
    return result


def _verify_compact(compact: Path, stage: Path, versions: dict) -> dict:
    description = _read(compact / 'versions.json')
    progress = _read(compact / 'progress.json')
    summary = _read(compact / 'summary.json')
    if (description.get('format') != 'dail-compact-evaluation-v1' or
            summary.get('format') != description['format'] or
            progress.get('phase') != 'complete' or
            description.get('scope') != 'full'):
        raise ValueError('compact SQL evaluation is incomplete')
    if _versions(description['versions']) != versions:
        raise ValueError('compact and record versions differ')
    copied = stage / 'evaluation' / compact.name
    copied.mkdir(parents=True)
    for name in ('versions.json', 'progress.json', 'summary.json'):
        source = compact / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'unsafe compact source: {source}')
        shutil.copy2(source, copied / name)
        if _info(copied / name) != _info(source):
            raise ValueError(f'compact copy hash mismatch: {name}')
    if _read(copied / 'versions.json') != description or _read(copied / 'summary.json') != summary:
        raise ValueError('compact metadata changed during handoff export')
    evaluation_store = compact / 'evaluation.sqlite3'
    if evaluation_store.is_symlink() or not evaluation_store.is_file():
        raise ValueError(f'unsafe compact source: {evaluation_store}')
    snapshot = _snapshot(evaluation_store, copied / 'evaluation.sqlite3')
    observed = {}
    with closing(sqlite3.connect((copied / 'evaluation.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        frozen_row = db.execute("SELECT value FROM metadata WHERE key='frozen'").fetchone()
        if frozen_row is None or not all(description.get(key) == value for key, value in
                                         json.loads(frozen_row[0]).items()):
            raise ValueError('compact SQLite frozen inputs differ from its versions file')
        for batch, group, question, version, state, raw, digest in db.execute(
                'SELECT batch,grp,question,version_id,state,payload_json,sha256 FROM items'):
            identity = str(batch), str(group), str(question)
            if (identity in observed or state != 'evaluated' or
                    hashlib.sha256(raw.encode()).hexdigest() != digest):
                raise ValueError(f'incomplete or corrupt compact item: {identity}')
            payload = json.loads(raw)
            if payload.get('task_key') != {'batch_id': batch, 'group': group, 'question_id': question} or \
                    payload.get('version_id') != version or payload.get('state') != state:
                raise ValueError(f'compact item identity differs: {identity}')
            observed[identity] = version
        for blob, digest in db.execute('SELECT result,sha256 FROM queries'):
            if hashlib.sha256(zlib.decompress(blob)).hexdigest() != digest:
                raise ValueError('compact SQL query checksum mismatch')
    if observed != versions or snapshot['table_rows']['items'] != len(versions):
        raise ValueError('compact SQL evaluation question coverage differs')
    if any(summary.get('storage', {}).get(name) != snapshot['table_rows'][table]
           for name, table in (('items', 'items'), ('queries', 'queries'))):
        raise ValueError('compact SQL summary counts differ from its store')
    return {'directory': str(Path('evaluation') / compact.name), 'sqlite_snapshot': snapshot,
            'questions': len(versions)}


def export_handoff(batch: Path, compact: Path, destination: Path) -> Path:
    """Create a fresh, complete source handoff without executing benchmark SQL."""
    batch = Path(batch).resolve(strict=True)
    compact = Path(compact).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if batch in destination.resolve().parents or compact in destination.resolve().parents:
        raise ValueError('handoff destination must be outside its sources')
    manifest = batch / 'manifest.json'
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError('batch manifest is missing or unsafe')
    description = _read(compact / 'versions.json')
    if description.get('manifest_sha256') != _sha(manifest):
        raise ValueError('compact and batch manifests differ')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.staging-', dir=destination.parent))
    try:
        shutil.copy2(manifest, stage / 'inputs_manifest.json')
        facts = export_facts(batch, stage, quiet=True)
        if not facts.get('record_export_complete'):
            raise ValueError('current record export is incomplete')
        with (stage / 'record_export/versions.jsonl').open(encoding='utf-8') as stream:
            versions = _versions([json.loads(line) for line in stream])
        compact_info = _verify_compact(compact, stage, versions)
        # Reread the live current map after extracting facts to catch a rerun
        # that landed during assembly.
        with closing(sqlite3.connect((batch / 'current.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
            current = {(str(b), str(g), str(q)): v for b, g, q, v in db.execute(
                'SELECT batch,grp,question,version FROM current_versions')}
        if current != versions or _sha(manifest) != description['manifest_sha256']:
            raise ValueError('batch versions or manifest changed during handoff export')
        _write(stage / 'evaluation/latest.json', {'directory': compact.name})
        now = datetime.now(timezone.utc).isoformat()
        verification = {
            'format': 'dail-offline-handoff-verification-v3', 'ok': True, 'created_at': now,
            'record_export_complete': True, 'reference_sql_evaluation_complete': True,
            'counts': facts['counts'], 'compact_evaluation': compact_info,
            'checked': ['manifest SHA256', 'current sealed record facts', 'matching current and compact versions',
                        'complete compact progress', 'compact item and query checksums',
                        'SQLite backup integrity_check', 'final file inventory and hashes'],
            'files': {str(path.relative_to(stage)): _info(path) for path in sorted(stage.rglob('*'))
                      if path.is_file() and path.relative_to(stage) not in {
                          Path('verification.json'), Path('index.json'), Path('COMPLETE.json')}},
        }
        _write(stage / 'verification.json', verification)
        index = {
            'format': FORMAT, 'complete': True, 'created_at': now,
            'record_export_complete': True, 'reference_sql_evaluation_complete': True,
            'raw_batch_directory': str(batch), 'record_directory': 'record_export',
            'input_manifest': 'inputs_manifest.json', 'input_manifest_sha256': _sha(stage / 'inputs_manifest.json'),
            'evaluation_directory': compact_info['directory'], 'verification': 'verification.json',
            'files': [{'path': str(path.relative_to(stage)), **_info(path)}
                      for path in sorted(stage.rglob('*')) if path.is_file()
                      and path.relative_to(stage) not in {Path('index.json'), Path('COMPLETE.json')}],
        }
        _write(stage / 'index.json', index)
        _write(stage / 'COMPLETE.json', {'format': FORMAT, 'complete': True,
                                       'index_sha256': _sha(stage / 'index.json'),
                                       'questions': len(versions)})
        _authenticate(stage)
        for entry in index['files']:
            if _info(stage / entry['path']) != {key: entry[key] for key in ('bytes', 'sha256')}:
                raise ValueError(f'final file hash mismatch: {entry["path"]}')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.replace(stage, destination)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)
