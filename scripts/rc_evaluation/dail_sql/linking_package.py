"""Publish the frozen DAIL Generation analysis plus a local Linking comparison.

No database queries against benchmarks or model calls are performed here. The
source handoff is read only; every SQLite file is copied with the backup API so
the published directory never depends on a WAL sidecar.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Any

from scripts.rc_evaluation.din_sql_linking.reporting import _summarize


FORMAT = 'dail-offline-handoff-v3'
GUIDE = 'dail_qwen38_2p4t_rc3_linking_handoff_guide.md'
GENERATION_FILES = ('modes.jsonl', 'rounds.jsonl', 'candidates.jsonl', 'failed_questions.jsonl')


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _info(path: Path) -> dict[str, Any]:
    return {'bytes': path.stat().st_size, 'sha256': _sha(path)}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b'\n')


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('wb') as stream:
        for row in rows:
            stream.write(_canonical(row) + b'\n')


def _relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'unsafe handoff path: {value!r}')
    return path


def _key(row: dict[str, Any]) -> tuple[str, str]:
    key = row['task_key']
    return str(key['group']), str(key['question_id'])


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'not a regular source file: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    cloned = False
    if os.uname().sysname == 'Darwin':
        cloned = subprocess.run(['/bin/cp', '-c', str(source), str(destination)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                check=False).returncode == 0
    if not cloned:
        shutil.copy2(source, destination)


def _snapshot(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as origin:
        with closing(sqlite3.connect(destination)) as target:
            origin.backup(target)
            target.execute('PRAGMA journal_mode=DELETE')
            integrity = [row[0] for row in target.execute('PRAGMA integrity_check')]
            if integrity != ['ok']:
                raise ValueError(f'SQLite snapshot integrity failure: {source}: {integrity}')
            tables = [row[0] for row in target.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            counts = {table: target.execute('SELECT count(*) FROM "' + table.replace('"', '""') + '"').fetchone()[0]
                      for table in tables}
    return {'method': 'sqlite_backup', 'integrity_check': 'ok', 'table_rows': counts}


def _ignored(path: Path) -> bool:
    return path.name == '.DS_Store' or path.name.endswith(('-wal', '-shm'))


def _authenticate(source: Path) -> tuple[dict, dict, dict]:
    index = _read(source / 'index.json')
    verification = _read(source / 'verification.json')
    records = _read(source / 'record_export/verification.json')
    if not index.get('record_export_complete') or not index.get('reference_sql_evaluation_complete'):
        raise ValueError('source Generation/evaluation handoff is incomplete')
    if not records.get('record_export_complete'):
        raise ValueError('source record export is incomplete')
    for report, directory in ((records, source / 'record_export'), (verification, source)):
        for name, expected in report.get('files', {}).items():
            relative = _relative(name)
            # The v2 top-level report used bare names for record_export files.
            if directory == source and len(relative.parts) == 1 and relative.suffix == '.jsonl':
                relative = Path('record_export') / relative
            path = directory / relative
            if path.is_symlink() or not path.is_file() or _info(path) != {
                    field: expected[field] for field in ('bytes', 'sha256')}:
                raise ValueError(f'source file hash mismatch: {path}')
    if index.get('input_manifest_sha256') != _sha(source / 'inputs_manifest.json'):
        raise ValueError('source input manifest hash mismatch')
    for name in ('versions.jsonl', *GENERATION_FILES):
        if name not in records.get('files', {}):
            raise ValueError(f'source verification omits {name}')
    return index, verification, records


def _load_run_store(path: Path, versions: list[dict]) -> tuple[list[dict], dict]:
    expected = {_key(row): row for row in versions}
    if len(expected) != len(versions):
        raise ValueError('source contains duplicate questions')
    rows = {}
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
        metadata = {key: json.loads(value) for key, value in connection.execute('SELECT key, value FROM metadata')}
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schema_ids = set()
        if 'schemas' in tables:
            for schema_id, raw, checksum in connection.execute('SELECT schema_id, payload_json, sha256 FROM schemas'):
                if hashlib.sha256(raw.encode('utf-8')).hexdigest() != checksum or schema_id != checksum:
                    raise ValueError(f'Linking schema checksum mismatch: {schema_id}')
                json.loads(raw)
                schema_ids.add(schema_id)
        for group, question, raw, checksum in connection.execute(
                'SELECT grp, question, payload_json, sha256 FROM items ORDER BY grp, question'):
            if hashlib.sha256(raw.encode('utf-8')).hexdigest() != checksum:
                raise ValueError(f'Linking payload checksum mismatch: {group}/{question}')
            payload = json.loads(raw)
            key = str(group), str(question)
            if key != _key(payload) or key not in expected or key in rows:
                raise ValueError(f'Linking question identity mismatch: {key}')
            original = expected[key]
            if payload['task_key'] != original['task_key'] or payload['version_id'] != original['version_id']:
                raise ValueError(f'Linking version identity mismatch: {key}')
            detail = payload['evaluation']
            if (str(detail['group']), str(detail['question_id'])) != key:
                raise ValueError(f'Linking evaluation question mismatch: {key}')
            if detail.get('version_id', payload['version_id']) != payload['version_id']:
                raise ValueError(f'Linking evaluation version mismatch: {key}')
            if not isinstance(payload['linking'], dict):
                raise ValueError(f'Linking record is not an object: {key}')
            schema_id = payload['linking'].get('full_schema_id')
            if schema_id is not None and schema_id not in schema_ids:
                raise ValueError(f'Linking full schema reference is missing: {key}')
            rows[key] = payload
    if set(rows) != set(expected):
        raise ValueError('Linking/source question coverage differs')
    return [rows[_key(row)] for row in versions], metadata


def _source_snapshots(stage: Path) -> None:
    project = Path(__file__).resolve().parents[3]
    for relative in (
        'scripts/rc_evaluation/dail_sql/linking_package.py',
        'scripts/rc_evaluation/dail_sql/linking_handoff.py',
        'scripts/baseline_adapters/dail_sql/linking_comparison.py',
        'scripts/baseline_adapters/dail_sql/native.py',
        'baselines/DAIL-SQL/utils/linking_utils/spider_match_utils.py',
        'baselines/DAIL-SQL/utils/linking_utils/application.py',
        'scripts/rc_evaluation/din_sql_linking/reporting.py',
    ):
        path = project / relative
        if path.is_file():
            _copy_file(path, stage / 'source_snapshot/linking' / relative)


def _verify_final(stage: Path, index: dict, versions: list[dict], rows: list[dict]) -> None:
    listed = {row['path'] for row in index['files']}
    actual = {str(path.relative_to(stage)) for path in stage.rglob('*')
              if path.is_file() and path.relative_to(stage) not in {Path('index.json'), Path('COMPLETE.json')}}
    if actual != listed or len(listed) != len(index['files']):
        raise ValueError('final file inventory mismatch')
    for row in index['files']:
        if _info(stage / row['path']) != {field: row[field] for field in ('bytes', 'sha256')}:
            raise ValueError(f'final file hash mismatch: {row["path"]}')
    loaded_versions = _jsonl(stage / 'record_export/versions.jsonl')
    loaded_linking = _jsonl(stage / 'record_export/linking.jsonl')
    if len(loaded_versions) != len(versions) or len(loaded_linking) != len(rows):
        raise ValueError('final Linking counts mismatch')
    for version, link, original in zip(loaded_versions, loaded_linking, versions):
        ref = version['linking_ref']
        if ({key: version[key] for key in original} != original or
                link['task_key'] != version['task_key'] or
                link['version_id'] != version['version_id'] or
                ref['sha256'] != hashlib.sha256(_canonical(link)).hexdigest()):
            raise ValueError('final Linking reference mismatch')


def export_package(source: Path, destination: Path, run_store: Path, *, guide: Path | None = None) -> Path:
    """Publish one self-contained analysis directory, refusing an existing target."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    run_store = Path(run_store).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if source in destination.resolve().parents:
        raise ValueError('destination cannot be inside the immutable source')
    source_index, source_verification, source_records = _authenticate(source)
    versions = _jsonl(source / 'record_export/versions.jsonl')
    latest = _read(source / 'evaluation/latest.json')['directory']
    if _relative(latest).name != latest:
        raise ValueError('invalid evaluation latest pointer')
    evaluation_relative = Path('evaluation') / latest
    evaluation_versions = _read(source / evaluation_relative / 'versions.json')
    evaluation_rows = evaluation_versions['versions']
    if (len(evaluation_rows) != len(versions) or
            {_key(row): row for row in evaluation_rows} != {_key(row): row for row in versions}):
        raise ValueError('source evaluation versions differ from Generation versions')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.staging-', dir=destination.parent))
    snapshots, preserved = {}, {}
    try:
        for path in sorted(source.rglob('*')):
            if _ignored(path):
                continue
            if path.is_symlink():
                raise ValueError(f'source symlink is not supported: {path}')
            if not path.is_file() or path.name in ('COMPLETE.json',):
                continue
            relative = str(path.relative_to(source))
            target = stage / relative
            if path.suffix == '.sqlite3':
                snapshots[relative] = _snapshot(path, target)
            else:
                _copy_file(path, target)
                original_info = _info(path)
                if _info(target) != original_info:
                    raise ValueError(f'copied source hash mismatch: {relative}')
                preserved[relative] = original_info
        snapshots['raw_records/linking.sqlite3'] = _snapshot(run_store, stage / 'raw_records/linking.sqlite3')
        payloads, metadata = _load_run_store(stage / 'raw_records/linking.sqlite3', versions)
        rows, updated_versions, details = [], [], []
        for line, (version, payload) in enumerate(zip(versions, payloads), 1):
            row = {**payload['linking'], 'task_key': version['task_key'], 'version_id': version['version_id']}
            rows.append(row)
            details.append(payload['evaluation'])
            updated_versions.append({**version, 'linking_ref': {
                'file': 'record_export/linking.jsonl', 'line': line,
                'task_key': version['task_key'], 'version_id': version['version_id'],
                'sha256': hashlib.sha256(_canonical(row)).hexdigest()}})
        records, evaluation = stage / 'record_export', stage / evaluation_relative
        _write_jsonl(records / 'linking.jsonl', rows)
        _write_jsonl(records / 'versions.jsonl', updated_versions)
        _write_jsonl(evaluation / 'linking_details.jsonl', details)
        groups = sorted({row['group'] for row in details})
        linking = {
            'format': 'dail-local-linking-comparison-v1', 'primary_metric': 'column_macro_recall',
            'groups': {group: _summarize([row for row in details if row['group'] == group]) for group in groups},
            'overall': _summarize(details), 'local_llm_tokens': 0, 'new_paid_requests': 0,
            'cost_boundary': 'Filter usage is reused historical DIN schema-filter usage, not a new request; '
                             'base_linking and linking_rc3 are local DAIL execution with zero model tokens. '
                             'rc3_combined retains the conceptual filter plus Linking cost.',
            'generation_boundary': 'Separate Linking comparison; historical four Generation modes retain their full-schema inputs.',
            'details': str(evaluation_relative / 'linking_details.jsonl'),
            'raw_store': 'raw_records/linking.sqlite3',
            'status_counts': {condition: dict(Counter(row[condition]['status'] for row in details))
                              for condition in ('base', 'rc3', 'filter')},
        }
        summary = _read(evaluation / 'summary.json')
        summary['linking'] = linking
        _write(evaluation / 'summary.json', summary)
        evaluation_versions['linking_extension'] = {
            'format': linking['format'], 'questions': len(rows),
            'record_file': 'record_export/linking.jsonl',
            'details_file': linking['details'], 'raw_store': linking['raw_store'],
            'run_metadata': metadata, 'original_generation_versions_preserved': True,
        }
        _write(evaluation / 'versions.json', evaluation_versions)
        generation_preserved = {name: _info(records / name) for name in GENERATION_FILES}
        if any(generation_preserved[name] != preserved['record_export/' + name] for name in GENERATION_FILES):
            raise ValueError('Generation bytes changed during Linking export')
        counts = {}
        for path in records.glob('*.jsonl'):
            with path.open('rb') as stream:
                counts[path.stem] = sum(1 for _ in stream)
        for name, count in source_records.get('counts', {}).items():
            if name in counts and counts[name] != count:
                raise ValueError(f'source/export record count mismatch: {name}')
        record_verification = {
            **source_records, 'format': 'dail-offline-facts-v2',
            'created_at': datetime.now(timezone.utc).isoformat(), 'ok': True,
            'linking_complete': True, 'counts': counts, 'linking_refs_ok': True,
            'generation_payload_bytes_preserved': generation_preserved,
            'source_verification_sha256': _sha(source / 'record_export/verification.json'),
            'files': {path.name: _info(path) for path in sorted(records.glob('*.jsonl'))},
            'checked': ['authenticated source record hashes', 'preserved Generation payload bytes',
                        'one Linking record and current-version reference per question',
                        'Linking SQLite payload SHA256', 'record counts'],
            'source_checked': source_records.get('checked', []),
        }
        _write(records / 'verification.json', record_verification)
        _source_snapshots(stage)
        guide = Path(guide) if guide is not None else Path(__file__).resolve().parents[3] / 'docs/evaluation.md'
        _copy_file(guide, stage / GUIDE)
        verification = {
            'format': 'dail-offline-handoff-verification-v3', 'ok': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'record_export_complete': True, 'reference_sql_evaluation_complete': True,
            'linking_complete': True, 'counts': counts,
            'linking': {'questions': len(rows), 'unique_questions': len({_key(row) for row in rows}),
                        'version_refs_ok': True, 'status_counts': linking['status_counts'],
                        'local_llm_tokens': 0, 'new_paid_requests': 0},
            'source': {'directory': source.name, 'index_sha256': _sha(source / 'index.json'),
                       'verification_sha256': _sha(source / 'verification.json'),
                       'verification': source_verification},
            'sqlite_snapshots': snapshots,
            'generation_payload_bytes_preserved': generation_preserved,
            'checked': ['source indexed hashes', 'input manifest SHA256', 'SQLite backup integrity_check',
                        'Generation payload byte preservation', 'Linking question/version coverage',
                        'Linking payload SHA256', 'export file inventory and hashes'],
            'not_reexecuted': ['historical SQL benchmark queries', 'historical Generation requests',
                               'historical schema-filter requests'],
            'files': {str(path.relative_to(stage)): _info(path) for path in sorted(stage.rglob('*'))
                      if path.is_file() and path.relative_to(stage) not in {
                          Path('index.json'), Path('COMPLETE.json'), Path('verification.json')}},
        }
        _write(stage / 'verification.json', verification)
        raw_batch = Path(source_index['raw_batch_directory']).name
        index = {
            **source_index, 'format': FORMAT, 'complete': True,
            'path_base': 'this analysis directory (the directory containing index.json)',
            'raw_batch_directory': '../' + raw_batch,
            'raw_batch_required_for': 'historical complete LLM response bodies and deep request tracing only',
            'record_directory': 'record_export', 'input_manifest': 'inputs_manifest.json',
            'verification': 'verification.json', 'evaluation_directory': str(evaluation_relative),
            'guide': GUIDE, 'linking_complete': True, 'linking_questions': len(rows),
            'linking_records': 'record_export/linking.jsonl', 'linking_details': linking['details'],
            'linking_run_store': linking['raw_store'],
            'source_analysis': {'directory': source.name, 'index_sha256': _sha(source / 'index.json')},
            'notes': ['All quality and recorded usage analyses are available inside this directory.',
                      'Four historical Generation modes retain their original full-schema inputs.',
                      'New DAIL Linking is a separate local stage, not a parent of historical Generation.',
                      'Schema filtering reuses historical DIN filter records; no DIN Linking output is substituted.',
                      'Local Linking uses zero LLM tokens; conceptual RC cost includes reused filter usage.',
                      'All SQLite files are standalone backup snapshots; sidecars are not required.'],
        }
        index['files'] = [{'path': str(path.relative_to(stage)), **_info(path)}
                          for path in sorted(stage.rglob('*')) if path.is_file()
                          and path.relative_to(stage) not in {Path('index.json'), Path('COMPLETE.json')}]
        _write(stage / 'index.json', index)
        _verify_final(stage, index, versions, rows)
        _write(stage / 'COMPLETE.json', {'format': FORMAT, 'complete': True,
               'index_sha256': _sha(stage / 'index.json'), 'questions': len(rows)})
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.replace(stage, destination)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)
