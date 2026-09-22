"""Verify the active RC mixed-pair Selection subset against saved assessments.

Print one JSON object per group followed by the pooled total. No model calls,
benchmark SQL execution, or source writes. Requires selection_metrics.json.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from .paths import analysis_directory


def run(root):
    data = json.loads((root / 'selection_metrics.json').read_bytes())
    all_rows = []
    for group, value in data['groups'].items():
        selected = [r for r in value['items'] if r['active']
                    and r['rc_shortlist']['length'] == 2
                    and r['rc_shortlist']['profile'] == 'mixed']
        assert len(selected) == value['active_profiles']['rc']['mixed']
        by_key = {r['item_key']: r for r in selected}
        assert len(by_key) == len(selected)
        verified = set()
        with sqlite3.connect(f'file:{root / group / "evaluation.sqlite3"}?mode=ro', uri=True) as db:
            for raw, checksum in db.execute('select payload_json,sha256 from items'):
                item = json.loads(raw)
                if item['item_key'] not in by_key:
                    continue
                row = by_key[item['item_key']]
                assert hashlib.sha256(raw.encode()).hexdigest() == checksum == row['source_payload_sha256']
                short = item['targets']['rc/shortlist']
                assert len(short) == 2 and sorted(s['bag_equal'] for s in short) == [False, True]
                for side in ('native', 'rc'):
                    assert item['targets'][side + '/sql_selection'][0]['bag_equal'] is row[side]
                verified.add(item['item_key'])
        assert verified == set(by_key)
        counts = Counter((r['native'], r['rc']) for r in selected)
        assert all(type(r[s]) is bool for r in selected for s in ('native', 'rc'))
        n = len(selected)
        a, b = (sum(r[s] for r in selected) for s in ('native', 'rc'))
        gain, loss = counts[False, True], counts[True, False]
        assert b - a == gain - loss
        print(json.dumps({
            'group': group, 'n': n, 'native_correct': a, 'rc_correct': b,
            'gain': gain, 'loss': loss, 'net': b - a,
            'native_percent': 100 * a / n if n else None,
            'rc_percent': 100 * b / n if n else None,
            'delta_pp': 100 * (b - a) / n if n else None,
            'native_shortlist_profiles': dict(Counter(r['native_shortlist']['profile'] for r in selected)),
            'same_sql_order': sum(r['same_list_order'] for r in selected),
            'verified_items': len(verified),
        }, ensure_ascii=False))
        all_rows.extend(selected)

    counts = Counter((r['native'], r['rc']) for r in all_rows)
    n = len(all_rows)
    a, b = (sum(r[s] for r in all_rows) for s in ('native', 'rc'))
    assert b - a == counts[False, True] - counts[True, False]
    print(json.dumps({
        'TOTAL': n, 'native_correct': a, 'rc_correct': b,
        'gain': counts[False, True], 'loss': counts[True, False], 'net': b - a,
        'native_percent': 100 * a / n if n else None,
        'rc_percent': 100 * b / n if n else None,
        'delta_pp': 100 * (b - a) / n if n else None,
        'native_both_wrong_to_rc_mixed': sum(r['native_shortlist']['profile'] == 'all_mismatch' for r in all_rows),
    }, ensure_ascii=False))


if __name__ == '__main__':
    run(analysis_directory(argparse.ArgumentParser(description=__doc__)))
