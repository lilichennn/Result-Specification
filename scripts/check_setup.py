"""Check bundled RS inputs and source dependencies without network or databases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from result_contract.rc.rc_round3 import Round3RC

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('bird_dev', 'spider_dev', 'spider_test',
          'bird_interact_lite', 'bird_interact_full')


def identities(rows):
    values = [(str(row['index']), row['db_id']) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError('Duplicate question identities')
    return set(values)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def check_filtered_metadata(directory, questions, contracts, entry):
    """Check a reusable filter snapshot without executing filtering or SQL."""
    path = directory / 'filtered_meta.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Filtered metadata file hash differs from its provenance')
    if _digest(questions) != entry['questions_sha256']:
        raise ValueError('Filtered metadata question inputs changed')
    specifications = {str(row['index']): {'db_id': row['db_id'], 'rc_round3': row['rc_round3']}
                      for row in contracts}
    if _digest(specifications) != entry['round3_sha256']:
        raise ValueError('Filtered metadata Round-3 inputs changed')
    rows = json.loads(raw)
    if not isinstance(rows, dict) or set(rows) != {str(row['index']) for row in questions}:
        raise ValueError('Filtered metadata question identities differ')
    succeeded = failed = 0
    for row in rows.values():
        status = row.get('status', {})
        if type(status.get('success')) is not bool or not isinstance(status.get('reason'), str):
            raise ValueError('Malformed filtered metadata status')
        if status['success']:
            if not isinstance(row.get('result'), list):
                raise ValueError('Successful filtering requires a metadata list')
            succeeded += 1
        else:
            if row.get('result') is not None:
                raise ValueError('A failed filter must not provide a schema')
            failed += 1
    if (len(rows), succeeded, failed) != (entry['questions'], entry['succeeded'], entry['failed']):
        raise ValueError('Filtered metadata counts differ from provenance')
    for relative, expected in entry.get('metadata_sha256', {}).items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Filtered metadata source changed: {relative}')
    return succeeded, failed


def check_group(group, filter_entry=None):
    directory = ROOT / 'data' / group
    questions = json.loads((directory / f'{group}.json').read_text())
    contracts = json.loads((directory / 'rc.json').read_text())
    if not questions or identities(questions) != identities(contracts):
        raise ValueError(f'{group}: questions and RS identities differ')
    if not any((directory / 'meta').rglob('*.csv')):
        raise ValueError(f'{group}: no public metadata found')
    successful = 0
    for row in contracts:
        if row.get('round3_status') == 'succeeded':
            Round3RC.from_value(row['rc_round3'])
            successful += 1
    if successful != len(questions):
        raise ValueError(f'{group}: missing successful Round-3 specifications')
    if filter_entry is not None:
        check_filtered_metadata(directory, questions, contracts, filter_entry)
    return len(questions), successful


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append('Python 3.12 is required by pyproject.toml; run uv sync --locked.')
    required = [
        'baselines/DeepEye-SQL/app/pipeline/__init__.py',
        'baselines/DAIL-SQL/data_preprocess.py',
        'baselines/DIN-SQL/DIN-SQL.py',
        'scripts/rc_evaluation/deepeye/rc_prompt.txt',
        'data/reference/schema_linking_annotations.jsonl',
    ]
    required += [f'result_contract/rc/rc_round{number}_{role}.txt'
                 for number in (1, 2, 3) for role in ('system', 'user')]
    required += [f'scripts/baseline_adapters/din_sql/stages/{stage}.py' for stage in
                 ('schema_linking', 'difficulty_decomposition', 'sql_generation', 'self_correction')]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f'Missing {relative}; for submodules run git submodule update --init --recursive.')
    total = 0
    try:
        filters = json.loads((ROOT / 'data/reference/schema_filter_manifest.json').read_text())['groups']
        if set(filters) != set(GROUPS):
            raise ValueError('Filtered metadata groups differ from the five evaluation groups')
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f'Filtered metadata manifest: {error}')
        filters = {}
    for group in GROUPS:
        try:
            count, ready = check_group(group, filters.get(group))
            print(f'{group}: {count} questions, {ready} successful Round-3 specifications')
            if group in filters:
                print(f"  Reusable schema filters: {filters[group]['succeeded']} succeeded, "
                      f"{filters[group]['failed']} failed (preserved source statuses)")
            total += count
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'{group}: {error}')
    print(f'Main five-group question total: {total}')
    print('No network or database checks were performed; API credentials and downloaded resources are not validated.')
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
