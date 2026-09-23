"""Score saved Linking predictions against explicit final reference annotations.

Accepts five-group DeepEye offline.json files or DIN/DAIL linking_details.jsonl.
Uses the existing annotation reporting set metric, with exact physical names:
resolved annotations only; empty gold columns excluded from macro recall;
empty predictions score zero for nonempty gold. No model calls or benchmark SQL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scripts.rc_evaluation.schema_linking_gold.reporting import (
    _annotation_sets, _prediction_sets, _set_metrics,
)
from scripts.rc_evaluation.schema_linking_gold.source import load_offline_groups

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def detail_predictions(path):
    """The normalized physical-column details exported by DIN and DAIL."""
    for row in read_jsonl(path):
        predictions = {}
        for side in ('base', 'rc3'):
            columns = row[side]['columns']
            if not isinstance(columns, list) or any(
                    not isinstance(pair, list) or len(pair) != 2 or
                    not all(isinstance(name, str) for name in pair) for pair in columns):
                raise ValueError(f'Invalid physical columns for {row.get("task_key")}/{side}')
            predictions[side] = {tuple(pair) for pair in columns}
            if row[side].get('status') != 'succeeded' and predictions[side]:
                raise ValueError(f'Failed Linking prediction must have no columns: {row.get("task_key")}/{side}')
        yield row['task_key'], row['group'], predictions


def deepeye_predictions(directory):
    for task in load_offline_groups(directory):
        yield task.task_key, task.group, {
            'base': _prediction_sets(task, 'native_linked_schema')[1],
            'rc3': _prediction_sets(task, 'rc_linked_schema')[1],
        }


def score(annotations, predictions):
    labels = {}
    for row in annotations:
        key = row['task_key']
        if not isinstance(key, str) or key in labels:
            raise ValueError(f'Invalid or duplicate annotation key: {key!r}')
        if row.get('status') not in ('resolved', 'needs_review', 'invalid_sql', 'pending'):
            raise ValueError(f'Invalid annotation status for {key}')
        labels[key] = row
    groups = defaultdict(list)
    seen = set()
    excluded = Counter()
    for key, group, predicted in predictions:
        if key in seen or key not in labels:
            raise ValueError(f'Duplicate prediction or missing annotation: {key!r}')
        seen.add(key)
        annotation = labels[key]
        if annotation['status'] != 'resolved':
            excluded[annotation['status']] += 1
            continue
        gold = _annotation_sets(annotation)[1]
        groups[group].append((gold, predicted))

    def summarize(rows):
        result = {'resolved_questions': len(rows)}
        for side in ('base', 'rc3'):
            metrics = _set_metrics([(gold, prediction[side]) for gold, prediction in rows])
            result[side] = {'column_macro_recall': metrics['macro']['recall'],
                            'eligible_questions': metrics['macro']['recall_questions']}
        return result

    return {
        'primary_metric': 'column_macro_recall',
        'definition': 'Arithmetic mean of per-question physical-column set recall over resolved annotations with nonempty required_columns.',
        'prediction_conditions': {'base': 'native (DeepEye) or base (DIN/DAIL)',
                                  'rc3': 'rc (DeepEye) or rc3 (DIN/DAIL)'},
        'annotation_questions': len(labels), 'prediction_questions': len(seen),
        'annotations_without_predictions': len(set(labels) - seen),
        'excluded_prediction_questions': dict(excluded),
        'groups': {group: summarize(rows) for group, rows in sorted(groups.items())},
        'overall': summarize([row for rows in groups.values() for row in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path,
                        default=ROOT / 'data/reference/schema_linking_annotations.jsonl')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--deepeye-analysis', type=Path, help='Root with all five GROUP/offline.json files.')
    inputs.add_argument('--linking-details', type=Path, help='DIN/DAIL normalized evaluation/linking_details.jsonl.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/column_recall.json',
                        help='New JSON report; refuses to overwrite an existing file.')
    args = parser.parse_args()
    predictions = (deepeye_predictions(args.deepeye_analysis) if args.deepeye_analysis else
                   detail_predictions(args.linking_details))
    result = score(read_jsonl(args.annotations), predictions)
    result['annotations_sha256'] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    result['prediction_source'] = str(args.deepeye_analysis or args.linking_details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['overall'], ensure_ascii=False))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
