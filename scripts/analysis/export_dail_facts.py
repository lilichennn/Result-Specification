"""Export latest sealed DAIL facts without SQL execution or model requests.

Standalone standard-library tool; original stores remain read-only.
Usage: python -m scripts.analysis.export_dail_facts --batch BATCH --output OUTPUT
"""
import hashlib
import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3
import tempfile


def export(batch, analysis):
    batch, analysis = Path(batch), Path(analysis)
    analysis.mkdir(parents=True, exist_ok=True)
    destination = analysis / "record_export"
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=".facts-", dir=analysis))
    counts, statuses = Counter(), Counter()
    groups = defaultdict(Counter)
    with ExitStack() as stack:
        def connect(path):
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            stack.callback(conn.close)
            return conn
        current = connect(batch / "current.sqlite3")
        versions = current.execute("SELECT batch,grp,question,version FROM current_versions ORDER BY grp,question").fetchall()
        manifest = json.loads((batch / "manifest.json").read_text())
        expected = {(manifest['batch_id'], g, str(q)) for g, spec in manifest['groups'].items() for q in spec['ids']}
        assert {(b,g,q) for b,g,q,v in versions} == expected
        outputs = {name: stack.enter_context((staging / f"{name}.jsonl").open('w'))
                   for name in ('versions', 'modes', 'rounds', 'candidates', 'failed_questions')}
        def emit(name, value):
            outputs[name].write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            counts[name] += 1
        stores = {}
        for batch_id, group, qid, version in versions:
            assert version is not None
            group_hex, attempt = version.split(':')
            assert bytes.fromhex(group_hex).decode() == group
            if group not in stores:
                stores[group] = connect(batch / f'group-{group_hex}' / 'run.sqlite3')
            conn = stores[group]
            finish = conn.execute('SELECT status FROM finishes WHERE attempt_id=?', (attempt,)).fetchone()
            assert finish == ('succeeded',), (group, qid, finish)
            provenance = {'task_key': {'batch_id':batch_id, 'group':group, 'question_id':qid}, 'version_id':version}
            kinds = dict(conn.execute('SELECT event_no,kind FROM events WHERE attempt_id=?', (attempt,)))
            def check_ref(ref, kind):
                prefix, number = ref.rsplit('#', 1)
                assert prefix == version and kinds[int(number)] == kind
            modes, rounds, round_events = [], {}, {}
            for number, kind, raw, checksum in conn.execute(
                    "SELECT event_no,kind,payload_json,payload_checksum FROM events WHERE attempt_id=? AND kind IN ('mode_result','round_result') ORDER BY event_no", (attempt,)):
                assert hashlib.sha256(raw.encode()).hexdigest() == checksum
                payload = json.loads(raw)
                event_id = f'{version}#{number}'
                if kind == 'mode_result':
                    modes.append({**provenance, **payload, 'mode_event_id':event_id})
                else:
                    rid = payload['round_execution_id']
                    assert rid not in rounds
                    rounds[rid], round_events[rid] = payload, event_id
            assert len(modes) == 4 and {m['mode'] for m in modes} == {'native','rc_first','rc_second','rc_both'}
            referenced = {m[k] for m in modes for k in ('first_round_id','second_round_id') if m.get(k)}
            assert referenced <= rounds.keys()
            candidates = {}
            for rid in sorted(referenced):
                payload = rounds[rid]
                compact = {k:v for k,v in payload.items() if k not in ('candidates','samples')}
                compact['samples'] = [{k:v for k,v in s.items() if k != 'choice'} for s in payload.get('samples', [])]
                emit('rounds', {**provenance, **compact, 'round_event_id':round_events[rid]})
                for c in payload['candidates']:
                    assert c['candidate_id'] not in candidates
                    check_ref(c['source_request_id'], 'request_result')
                    check_ref(c['vote_execution_ref'], 'vote_execution')
                    candidates[c['candidate_id']] = c
                    emit('candidates', {**provenance, 'round_execution_id':rid, **c})
            for m in modes:
                c = candidates.get(m.get('final_candidate_id'))
                if m['status'] == 'succeeded':
                    assert c is not None
                    assert any(x['candidate_id'] == m['final_candidate_id'] for x in rounds[m['second_round_id']]['candidates'])
                m['final_candidate_sql'] = c['candidate_sql'] if c else None
                emit('modes', m)
                statuses[m['status']] += 1
            failed = any(m['status'] != 'succeeded' for m in modes)
            if failed:
                emit('failed_questions', {**provenance, 'modes':{m['mode']:m['status'] for m in modes}})
            groups[group]['questions'] += 1
            groups[group]['any_mode_failed' if failed else 'all_modes_succeeded'] += 1
            emit('versions', provenance)
        assert counts['versions'] == len(expected) and counts['modes'] == 4 * len(expected)
    hashes = {p.name:{'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in staging.iterdir()}
    verification = {'format':'dail-offline-facts-v1', 'created_at':datetime.now(timezone.utc).isoformat(),
        'record_export_complete':True, 'reference_sql_evaluation_complete':False,
        'counts':dict(counts), 'mode_statuses':dict(statuses), 'groups':dict(groups),
        'checked':['manifest membership', 'latest sealed versions', 'mode and round payload SHA256',
                   'four modes per question', 'round/candidate/request/vote references', 'successful final candidate in second round'],
        'files':hashes, 'not_checked':['all raw request payload checksums', 'full gold result comparisons']}
    (staging / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n')
    os.replace(staging, destination)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True,
                        help='Sealed batch with manifest.json, current.sqlite3 and group RunStores.')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'outputs/analysis/dail',
                        help='Parent of the new record_export directory.')
    args = parser.parse_args()
    export(args.batch, args.output)
