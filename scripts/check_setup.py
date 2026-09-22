"""Check bundled RS inputs and source dependencies without network or databases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from result_contract.rc.rc_round3 import Round3RC

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('bird_dev', 'spider_dev', 'spider_test',
          'bird_interact_lite', 'bird_interact_full', 'spider2_lite')


def identities(rows):
    values = [(str(row['index']), row['db_id']) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError('Duplicate question identities')
    return set(values)


def check_group(group):
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
    if group != 'spider2_lite' and successful != len(questions):
        raise ValueError(f'{group}: missing successful Round-3 specifications')
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
    for group in GROUPS:
        try:
            count, ready = check_group(group)
            print(f'{group}: {count} questions, {ready} successful Round-3 specifications')
            if group != 'spider2_lite':
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
