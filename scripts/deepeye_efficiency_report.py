"""Read-only analysis of completed capacity probes; no credentials or model calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3


def verified_summary(run):
    """Fresh, read-only checksum validation; exports are not the authority."""
    from scripts.efficiency_probe.core import RunStore
    run = Path(run)
    summary = json.loads((run/'summary.json').read_text())
    with RunStore.open(run, read_only=True) as store:
        integrity = store.verify()
        if not integrity['ok']:
            raise ValueError(f'current RunStore integrity failed: {run}')
        attempts = store.attempts()
        if len(attempts) != 1 or attempts[0]['status'] != 'succeeded':
            raise ValueError('capacity probe must have one completed recording attempt')
        if store.manifest != summary['manifest']:
            raise ValueError('exported manifest differs from verified RunStore')
        if attempts[0]['payload'] != {k:v for k,v in summary.items() if k != 'integrity'}:
            raise ValueError('exported summary differs from verified RunStore finish')
    if summary['outstanding']:
        raise ValueError('analyze only fully drained probes')
    return summary, integrity


def rolling_peak(times, window):
    if not math.isfinite(window) or window <= 0:
        raise ValueError('window must be finite and positive')
    ordered = sorted(times)
    first, peak = 0, 0
    for last, stamp in enumerate(ordered):
        while stamp-ordered[first] >= window:
            first += 1
        peak = max(peak, last-first+1)
    return peak


def interval_statistics(intervals, start, end, capacity):
    if end <= start or capacity <= 0:
        raise ValueError('positive window and capacity required')
    events = defaultdict(int)
    for lo, hi in intervals:
        lo, hi = max(start, lo), min(end, hi)
        if hi > lo:
            events[lo] += 1
            events[hi] -= 1
    active = peak = 0
    area = full = 0.
    previous = start
    for stamp, change in sorted(events.items()):
        duration = stamp-previous
        area += active*duration
        if active >= .9*capacity:
            full += duration
        active += change
        peak = max(peak, active)
        previous = stamp
    duration = end-start
    return dict(duration_seconds=duration, peak_inflight=peak,
                mean_inflight=area/duration, fraction_at_90pct_capacity=full/duration)


def handoff_statistics(source_intervals, new_intervals, capacity):
    start = min(lo for lo, _ in new_intervals)
    end = max(hi for _, hi in new_intervals)
    return {**interval_statistics([*source_intervals, *new_intervals], start, end, capacity),
            'source_pending_at_first_new_wire': sum(lo <= start < hi for lo, hi in source_intervals),
            'window': 'first new request headers through last new SDK terminal; combined client wire occupancy'}


def completed_handoff_runs(source):
    from scripts.efficiency_probe.handoff import DrainingProbe
    with DrainingProbe(source) as chain:
        if chain.pending():
            raise ValueError('all handoff ancestors must have terminal records before combined analysis')
        return [Path(p) for p in chain.source_runs()]


def analyze_campaign(base):
    """All-origin wire occupancy; phase boundaries are first headers of each new batch.

    Unlike per-lineage statistics, later descendants are included. Phase response
    counts include old requests and are NOT causal estimates of that capacity.
    """
    timestamp = lambda v: datetime.fromisoformat(v).timestamp()
    runs, intervals, headers, terminals = [], [], [], []
    totals, errors = Counter(), Counter()
    for path in Path(base).glob('remote_*/summary.json'):
        run = path.parent
        summary, integrity = verified_summary(run)
        with sqlite3.connect((run/'run.sqlite3').resolve().as_uri()+'?mode=ro', uri=True) as db:
            if db.execute('pragma integrity_check').fetchone()[0] != 'ok':
                raise ValueError('campaign SQLite integrity failed')
            request_ids = Counter(r[0] for r in db.execute("select json_extract(payload_json,'$.request_no') from events where kind='request'"))
            rows = db.execute("select json_extract(payload_json,'$.request_no'),json_extract(payload_json,'$.wire_started_at'),"
                "json_extract(payload_json,'$.finished_at'),json_extract(payload_json,'$.http_success') "
                "from events where kind in ('response','error')").fetchall()
        if request_ids != Counter(r[0] for r in rows) or any(n != 1 for n in request_ids.values()):
            raise ValueError('campaign request/terminal pairing failed')
        own_headers = [timestamp(r[1]) for r in rows if r[1]]
        for _, wire, finish, success in rows:
            if wire:
                intervals.append((timestamp(wire), timestamp(finish)))
                headers.append(timestamp(wire))
            terminals.append((timestamp(finish), bool(success)))
        start = min(own_headers, default=timestamp(summary['manifest']['created_at']))
        runs.append({'run': run.name, 'first_wire_timestamp': start,
                     'limit': summary['target_concurrency'], 'requests': summary['requests'],
                     'http_success': summary['http_success'], 'errors_by_type': summary['errors_by_type'],
                     'reported_tokens': summary['reported_tokens'],
                     'usage_unknown_requests': summary['usage_unknown_requests'],
                     'integrity_now': integrity})
        totals.update(summary['reported_tokens'])
        errors.update(summary['errors_by_type'])
    if not runs or len(runs) != len(list(Path(base).glob('remote_*/manifest.json'))):
        raise ValueError('campaign has no completed runs or contains an unfinished run')
    runs.sort(key=lambda r:r['first_wire_timestamp'])
    end = max(t for t, _ in terminals)
    phases = []
    for i, run in enumerate(runs):
        lo = run['first_wire_timestamp']
        hi = runs[i+1]['first_wire_timestamp'] if i+1 < len(runs) else end
        stats = interval_statistics(intervals, lo, hi, run['limit'])
        in_phase = lambda t: lo <= t < hi or (i == len(runs)-1 and t == hi)
        phases.append({'run':run['run'], 'limit':run['limit'], 'start_timestamp':lo, 'end_timestamp':hi,
            **stats, 'new_headers':sum(in_phase(t) for t in headers),
            'successful_terminals_all_origins':sum(ok and in_phase(t) for t,ok in terminals),
            'error_terminals_all_origins':sum(not ok and in_phase(t) for t,ok in terminals),
            'within_limit':stats['peak_inflight'] <= run['limit']})
    return {'campaign':str(Path(base).resolve()), 'requests':sum(r['requests'] for r in runs),
        'http_success':sum(r['http_success'] for r in runs), 'reported_tokens':dict(totals),
        'usage_unknown_requests':sum(r['usage_unknown_requests'] for r in runs),
        'errors_by_type':dict(errors), 'runs':runs, 'phases':phases,
        'all_phase_limits_respected':all(p['within_limit'] for p in phases),
        'global_peak_wire_inflight':max(p['peak_inflight'] for p in phases),
        'peak_headers_per_rolling_second':rolling_peak(headers,1),
        'notes':['Phases include human pauses and tail drain; not comparable steady-state throughput windows.',
                 'Response counts by phase include earlier cohorts; do not attribute errors to the current limit alone.',
                 'Only client-observed wire occupancy; remote GPU occupancy and cancellation are unknown.']}


def analyze(run):
    from scripts.efficiency_probe.core import summarize, distribution
    run = Path(run)
    summary, record_integrity = verified_summary(run)
    work = {w['source_event_id']:w for w in json.loads((run/'workload.json').read_text())}
    with sqlite3.connect((run/'run.sqlite3').resolve().as_uri()+'?mode=ro', uri=True) as db:
        rows = [json.loads(r[0]) for r in db.execute("select payload_json from events where kind in ('response','error') order by event_id")]
        requests = Counter(json.loads(r[0])['request_no'] for r in db.execute("select payload_json from events where kind='request'"))
        integrity = db.execute('pragma integrity_check').fetchone()[0]
    terminals = Counter(r['request_no'] for r in rows)
    if requests != terminals or any(n != 1 for n in terminals.values()):
        raise ValueError('request and terminal records are not one-to-one')
    timestamp = lambda v: datetime.fromisoformat(v).timestamp()
    wire = sorted(timestamp(r['wire_started_at']) for r in rows if r.get('wire_started_at'))
    intervals = [(timestamp(r['wire_started_at']), timestamp(r['finished_at'])) for r in rows if r.get('wire_started_at')]
    start = min(timestamp(r['started_at']) for r in rows)
    end = max(timestamp(r['finished_at']) for r in rows)
    cap = summary['target_concurrency']
    events = defaultdict(int)
    for lo, hi in intervals:
        events[lo] += 1
        events[hi] -= 1
    active = 0
    load_start = None
    for stamp, delta in sorted(events.items()):
        active += delta
        if active >= .9*cap:
            load_start = stamp
            break
    load = None
    if wire and load_start is not None and wire[-1] > load_start:
        load = interval_statistics(intervals, load_start, wire[-1], cap)
        load['start_seconds'] = load_start-start
        load['end_seconds'] = wire[-1]-start
        load['successful_responses'] = sum(r['http_success'] and load_start <= timestamp(r['finished_at']) < wire[-1] for r in rows)
        load['successes_per_minute'] = load['successful_responses']*60/load['duration_seconds']
        load['caveat'] = 'First >=90% capacity until last admission, NOT a sustained steady-state proof.'
    stages, inputs = defaultdict(list), defaultdict(list)
    minutes = defaultdict(lambda: Counter(sent=0, success=0, errors=0))
    for r in rows:
        w = work[r['source_event_id']]
        stages[w['stage']].append(r)
        inputs[r['source_event_id']].append(r)
        minutes[int((timestamp(r.get('wire_started_at',r['started_at']))-start)//60)]['sent'] += 1
        minutes[int((timestamp(r['finished_at'])-start)//60)]['success' if r['http_success'] else 'errors'] += 1
    fields = {'requests','http_success','parsed','errors_by_type','reported_tokens','usage_unknown_requests',
              'reasoning_tokens_reported','latency_seconds','success_latency_seconds'}
    rate_errors = [r for r in rows if r.get('status_code')==429]
    handoff = None
    source_path = (summary.get('handoff') or {}).get('source_run')
    if source_path:
        old_intervals, sources = [], []
        for ancestor in completed_handoff_runs(source_path):
            _, ancestor_integrity = verified_summary(ancestor)
            with sqlite3.connect((ancestor/'run.sqlite3').resolve().as_uri()+'?mode=ro', uri=True) as db:
                old = [json.loads(r[0]) for r in db.execute("select payload_json from events where kind in ('response','error')")]
                source_integrity = db.execute('pragma integrity_check').fetchone()[0]
            old_intervals.extend((timestamp(r['wire_started_at']), timestamp(r['finished_at'])) for r in old if r.get('wire_started_at'))
            sources.append({'run': str(ancestor), 'integrity': source_integrity,
                            'integrity_now': ancestor_integrity, 'pairing_ok': True})
        handoff = handoff_statistics(old_intervals, intervals, cap)
        handoff.update(source_run=str(source_path), sources=sources,
                       combined_limit=cap, within_combined_limit=handoff['peak_inflight'] <= cap,
                       source_manifest_sha256=summary['handoff']['source_manifest_sha256'])
    return dict(run=str(run.resolve()), summary=summary,
                pairing_ok=True, sqlite_integrity_now=integrity, record_integrity_now=record_integrity,
                stage={stage:{k:v for k,v in summarize(rs,1).items() if k in fields} for stage,rs in stages.items()},
                inputs=[dict(source_event_id=key,item_key=work[key].get('item_key'),branch=work[key].get('branch_path'),
                             attempts=len(rs),success=sum(r['http_success'] for r in rs),
                             errors=dict(Counter(r['error_type'] for r in rs if r.get('error_type')))) for key,rs in inputs.items()],
                send_rate=dict(headers_count=len(wire),peak_per_rolling_second=rolling_peak(wire,1),
                               peak_per_rolling_minute=rolling_peak(wire,60),
                               first_to_last_headers_seconds=wire[-1]-wire[0] if wire else None),
                client_wire_occupancy=interval_statistics(intervals,start,end,cap),
                load_window=load, handoff=handoff, minute_counts=dict(sorted(minutes.items())),
                cached_input_tokens=sum(((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens') or 0 for r in rows),
                rate_limit_errors=dict(count=len(rate_errors),elapsed_seconds=distribution([r['elapsed_seconds'] for r in rate_errors]),
                                       codes=dict(Counter(json.loads(r['error_body']).get('code') for r in rate_errors if r.get('error_body')))),
                notes=['Client wire occupancy is not server GPU occupancy.',
                       'Response-completion timestamps cannot reconstruct provider TPM accounting.',
                       'Different request budgets and admission rates confound direct batch-throughput comparisons.'])


def main():
    from scripts.efficiency_probe.core import write_json
    p = argparse.ArgumentParser(description=__doc__)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--run', type=Path)
    target.add_argument('--campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze_campaign(args.campaign) if args.campaign else analyze(args.run)
    write_json(args.output, result)
    fields = ['campaign','requests','http_success','errors_by_type','global_peak_wire_inflight','all_phase_limits_respected'] if args.campaign else ['run','pairing_ok','send_rate','client_wire_occupancy','load_window','handoff','rate_limit_errors']
    print(json.dumps({k:result[k] for k in fields},ensure_ascii=False))


if __name__ == '__main__':
    main()
