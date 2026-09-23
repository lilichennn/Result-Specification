"""Read-only exhaustion audit of the five frozen qwen3.8-2.4t-a95b RS groups.

Reuse the prior checksum-verified offline extraction for membership/denominators;
reconcile every nonzero failure row with original sample outcomes and attempts.
No model calls, benchmark queries, or changes to original experiment records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from .paths import DEFAULT_ANALYSIS, add_analysis_argument

HERE = DEFAULT_ANALYSIS
from scripts import deepeye_run  # Registers baseline imports for serialized types.
from scripts.baseline_adapters.deepeye.run_store import RunStore
from scripts.rc_evaluation.deepeye.campaign.ledger import CampaignLedger
from scripts.rc_evaluation.deepeye.campaign.processes import atomic_json

GROUPS = {'bird_dev': 1534, 'spider_dev': 1034, 'spider_test': 2147,
          'bird_interact_lite': 195, 'bird_interact_full': 410}
STAGES = ('schema_linking', 'sql_generation', 'sql_revision', 'sql_selection')


def reason(attempt):
    error = attempt.error or ''
    if 'APITimeoutError' in error:
        return 'timeout'
    if 'RateLimitError' in error or 'limit_requests' in error:
        return 'rate_limit'
    if 'APIConnectionError' in error:
        return 'connection'
    if attempt.status in ('parse_rejected', 'empty'):
        return attempt.status
    return attempt.status + (':' + error.split(':', 1)[0] if error else '')


def summarize(slots):
    return {
        'exhausted_samples': len(slots),
        'affected_questions': len({(s['group'], s['item_key']) for s in slots}),
        'affected_condition_stages': len({(s['group'], s['item_key'], s['condition'], s['stage']) for s in slots}),
        'all_four_timeout_samples': sum(s['cause'] == 'all_timeout' for s in slots),
        'any_timeout_samples': sum('timeout' in s['attempt_reasons'] for s in slots),
        'any_timeout_questions': len({(s['group'], s['item_key']) for s in slots if 'timeout' in s['attempt_reasons']}),
        'last_attempt_timeout_samples': sum(s['attempt_reasons'][-1] == 'timeout' for s in slots),
        'no_timeout_samples': sum('timeout' not in s['attempt_reasons'] for s in slots),
        'causes': dict(Counter(s['cause'] for s in slots)),
        'attempt_reason_counts': dict(Counter(v for s in slots for v in s['attempt_reasons'])),
        'target_status_counts_by_sample': dict(Counter(s['target_status'] for s in slots)),
    }


def run(output, campaign_root=None, campaign_pattern='{group}'):
    output.mkdir(parents=True, exist_ok=False)
    summaries, stage_rows, slots, question_rows, provenance = [], [], [], [], []
    for group, count in GROUPS.items():
        offline = HERE / group / 'offline.json'
        raw = offline.read_bytes()
        data = json.loads(raw)
        offline_sha = hashlib.sha256(raw).hexdigest()
        del raw
        assert data['config']['chat']['model'] == 'qwen3.8-2.4t-a95b'
        assert data['config']['chat']['sample_max_attempts'] == 4
        records = data['records']
        assert len(records) == count * 8
        assert len({(r['condition'], r['stage'], r['item_key']) for r in records}) == len(records)
        assert all(r['status'] == 'succeeded' for r in records)
        campaign = (Path(campaign_root) / campaign_pattern.format(group=group)
                    if campaign_root is not None else Path(data['campaign']))
        with CampaignLedger.open(campaign, read_only=True) as ledger:
            jobs = {j['job_id']: j for j in ledger.jobs()}
            assert all(j['state'] == 'finished' for j in jobs.values())
            assert {j['kind'] for j in jobs.values()} == {'native_first', 'rc'}
            assert ledger._db.execute('SELECT COUNT(*) FROM canonical').fetchone()[0] == count
            failed_stage_observations = ledger._db.execute("SELECT COUNT(*) FROM observations WHERE status='failed'").fetchone()[0]
            assert failed_stage_observations == 0
        manifests = {m['job_id']: m for m in data['manifests']}
        failures = defaultdict(list)
        for r in records:
            if r['failed_samples']:
                failures[r['job_id']].append(r)
        group_slots = []
        for job_id, job in jobs.items():
            with RunStore.open(Path(job['run_dir']), read_only=True) as store, store._read_snapshot() as db:
                manifest = store.manifest
                checksum = db.execute('SELECT payload_checksum FROM manifest').fetchone()[0]
                assert checksum == manifests[job_id]['manifest_sha256']
                assert manifest['effective_config']['chat']['model'] == 'qwen3.8-2.4t-a95b'
                assert manifest['effective_config']['chat']['timeout_seconds'] == 660
                if job['kind'] == 'rc':
                    assert manifest['condition'] == 'rc' and manifest['rc_version'] == 3
                    assert not manifest['continue_downstream']
                for r in failures[job_id]:
                    finish = db.execute('SELECT status,payload_checksum FROM finishes WHERE attempt_id=?', (r['attempt_id'],)).fetchone()
                    assert finish[0] == r['status'] and finish[1] == r['payload_sha256']
                    results, outcomes, groups = {}, {}, {}
                    for row in db.execute("SELECT * FROM events WHERE attempt_id=? AND kind IN ('sample_result','sample_attempt_finished','sampling_group_bound','sampling_retry_authorized') ORDER BY event_id", (r['attempt_id'],)):
                        e = store._event_dict(row)
                        p = e['payload']
                        if e['kind'] == 'sampling_group_bound':
                            groups[p['group_id']] = p
                        elif e['kind'] == 'sampling_retry_authorized':
                            raise AssertionError('Unexpected renewal; requires separate accounting')
                        else:
                            key = (p['group_id'], p['sample_index'])
                            if e['kind'] == 'sample_result':
                                assert key not in results, 'Duplicate result within stage'
                                results[key] = e
                            else:
                                outcomes[key] = e
                    failed = {k:e for k,e in results.items() if not e['payload']['succeeded']}
                    assert len(failed) == r['failed_samples']
                    for key, event in failed.items():
                        p = event['payload']; outcome_event = outcomes[key]
                        outcome = outcome_event['payload']['outcome']
                        assert p['attempt_count'] == len(outcome.attempts) == groups[key[0]]['max_attempts'] == 4
                        assert not outcome.succeeded
                        assert [a.attempt_number for a in outcome.attempts] == [1, 2, 3, 4]
                        reasons = [reason(a) for a in outcome.attempts]
                        cause = ('all_timeout' if set(reasons) == {'timeout'} else
                                 'mixed_with_timeout' if 'timeout' in reasons else 'no_timeout')
                        group_slots.append({
                            'group': group, 'condition': r['condition'], 'item_key': r['item_key'],
                            'stage': r['stage'], 'run_dir': job['run_dir'], 'job_id': job_id,
                            'stage_attempt_id': r['attempt_id'], 'target_status': finish[0],
                            'group_id': key[0], 'sample_index': key[1], 'attempt_count': 4,
                            'sample_result_event_id': event['event_id'], 'outcome_event_id': outcome_event['event_id'],
                            'fatal': outcome.fatal, 'branch_path': p.get('branch_path'),
                            'attempt_reasons': reasons, 'cause': cause,
                            'attempts': [{'number': a.attempt_number, 'status': a.status, 'error': a.error} for a in outcome.attempts],
                        })
        assert len(group_slots) == sum(r['failed_samples'] for r in records)
        assert len({(s['run_dir'], s['group_id'], s['sample_index']) for s in group_slots}) == len(group_slots)
        for condition in ('native', 'rc'):
            for stage in STAGES:
                subset = [s for s in group_slots if s['condition'] == condition and s['stage'] == stage]
                rows = [r for r in records if r['condition'] == condition and r['stage'] == stage]
                denominator = sum(r['effective']['retained_samples'] + r['failed_samples'] for r in rows)
                stage_rows.append({'group': group, 'condition': condition, 'stage': stage,
                                   'question_count': count, 'actual_sample_results': denominator, **summarize(subset)})
        by_question = defaultdict(list)
        for s in group_slots:
            by_question[s['item_key']].append(s)
        for key, subset in sorted(by_question.items()):
            question_rows.append({'group': group, 'item_key': key, **summarize(subset),
                                 'condition_stages': [{'condition': c, 'stage': st, **summarize([s for s in subset if s['condition'] == c and s['stage'] == st])}
                                                      for c, st in sorted({(s['condition'], s['stage']) for s in subset})]})
        summary = {'group': group, 'question_count': count, **summarize(group_slots),
                   'by_condition': {c:summarize([s for s in group_slots if s['condition'] == c]) for c in ('native', 'rc')},
                   'terminal_failed_observations': failed_stage_observations}
        summaries.append(summary); slots.extend(group_slots)
        provenance.append({'group': group, 'campaign': str(campaign), 'offline': str(offline),
                           'offline_sha256': offline_sha, 'verified_run_manifests': len(jobs),
                           'verified_exhausted_stage_attempts': sum(len(rs) for rs in failures.values())})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'model': 'qwen3.8-2.4t-a95b',
              'question_count': sum(GROUPS.values()), 'timeout_seconds': 660, 'sample_max_attempts': 4,
              'definition': 'One unique fixed sampling slot with a terminal unsuccessful sample_result after all four attempts. Fourth-attempt success is not exhaustion. Counts exclude reused zero-call RS artifacts and historical unrelated runs.',
              'method': 'Membership and zero-failure rows reuse the checksum-verified offline extraction. All source run manifest hashes are checked; every nonzero-failure stage row and failed sample outcome is reconciled with checksummed source records. No whole-store event replay or SQL scoring is claimed.',
              'groups': summaries, 'total': summarize(slots),
              'total_by_condition': {c:summarize([s for s in slots if s['condition'] == c]) for c in ('native', 'rc')},
              'stage_rows': stage_rows, 'source_provenance': provenance}
    atomic_json(output / 'summary.json', report)
    atomic_json(output / 'questions.json', question_rows)
    atomic_json(output / 'exhausted_samples.json', slots)
    print(json.dumps({'output': str(output), 'total': report['total'], 'by_condition': report['total_by_condition']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_analysis_argument(parser)
    parser.add_argument('--campaign-root', type=Path, help='Optional parent overriding campaign locations in offline.json.')
    parser.add_argument('--campaign-pattern', default='{group}')
    parser.add_argument('--output-dir', type=Path, help='New output directory; defaults under --analysis-dir.')
    args = parser.parse_args()
    HERE = args.analysis_dir.resolve()
    output = args.output_dir or HERE / ('sampling_exhaustion_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run(output.resolve(), args.campaign_root, args.campaign_pattern)
