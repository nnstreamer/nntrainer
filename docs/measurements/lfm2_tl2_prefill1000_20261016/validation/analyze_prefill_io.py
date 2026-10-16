#!/usr/bin/env python3
"""Summarize opt-in prefill queue, direct IO, copy, and compute timestamps.

The application's final load/repack marker excludes warmup COMPUTE events,
including warmup M=1024 events. --after-line overrides that boundary. Without
either boundary, compute totals are explicitly marked as unfiltered.
"""
import argparse
import bisect
import json
import re
import statistics
from pathlib import Path


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError('reversed interval')
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def overlap(a, b):
    i = j = n = 0
    while i < len(a) and j < len(b):
        n += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return n


def distribution(values):
    values = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p90=values[int(.90 * len(values))],
                p99=values[int(.99 * len(values))], maximum=values[-1])


def analyze(path, after_line=None, first_moe_layer=None):
    lines = Path(path).read_text(errors='replace').splitlines()
    if after_line is None:
        boundaries = [i + 1 for i, line in enumerate(lines)
                      if '[init-trace]' in line and
                      ('model->load_weight' in line or
                       'model->repack_weight' in line)]
        after_line = max(boundaries, default=0)
    groups = {'BATCH': [], 'READ': [], 'COMPUTE': [], 'WARMUP': []}
    for i, line in enumerate(lines):
        m = re.search(r'\[HTP-PREFILL-(BATCH|READ|COMPUTE)\](.*)', line)
        if m:
            kind = m[1]
            if kind == 'COMPUTE' and i + 1 <= after_line:
                kind = 'WARMUP'
            values = {k: int(v) for k, v in
                      re.findall(r'(\w+)=(-?\d+)', m[2])}
            values['log_line'] = i + 1
            groups[kind].append(values)
    batches, reads, compute = (groups[k] for k in ('BATCH', 'READ', 'COMPUTE'))
    if not batches or not reads:
        raise ValueError('no prefill IO trace found')
    by_id = {b['id']: b for b in batches}
    assert len(by_id) == len(batches), 'duplicate batch ID'
    for r in reads:
        assert r['rc'] == 0, ('read failed', r)
        assert r['batch'] in by_id, ('unjoined read', r)
        assert r['end_us'] >= r['start_us'] >= by_id[r['batch']]['queued_us']
    service = sum(r['end_us'] - r['start_us'] for r in reads)
    ios = sum(r['io_us'] for r in reads)
    copies = sum(r['copy_us'] for r in reads)
    active = union([(r['start_us'], r['end_us']) for r in reads])
    comp = union([(c['start_us'], c['end_us']) for c in compute])
    wait = union([(b['demand_us'], b['joined_us']) for b in batches])
    start = min(r['start_us'] for r in reads)
    end = max(r['end_us'] for r in reads)
    origin = min(c['start_us'] for c in compute) if compute else start
    moes = [c for c in compute if c['kind'] == 0]
    moe_starts = [c['start_us'] for c in moes]
    events = []
    for r in reads:
        events.extend([(r['start_us'], 1), (r['end_us'], -1)])
    occupancy = {}
    now = start
    workers = peak = 0
    for t, delta in sorted(events):
        occupancy[workers] = occupancy.get(workers, 0) + t - now
        now = t
        workers += delta
        peak = max(peak, workers)
    rows = []
    for b in batches:
        rs = [r for r in reads if r['batch'] == b['id']]
        assert len(rs) == b['experts'], ('missing read record', b)
        late = [r for r in rs if r['end_us'] > b['demand_us']]
        queued = [r for r in late if r['start_us'] > b['demand_us']]
        running = [r for r in late if r['start_us'] <= b['demand_us']]
        ordinal = bisect.bisect_left(moe_starts, b['register_end_us'])
        row = dict(batch=b['id'], log_line=b['log_line'], experts=len(rs),
                   ready=b['ready'], jobs_ahead=b['jobs_ahead'],
                   queued_from_first_compute_us=b['queued_us'] - origin,
                   demand_from_first_compute_us=b['demand_us'] - origin,
                   queue_to_first_us=min(r['start_us'] for r in rs) - b['queued_us'],
                   queue_to_demand_us=b['demand_us'] - b['queued_us'],
                   batch_read_span_us=max(r['end_us'] for r in rs) -
                   min(r['start_us'] for r in rs),
                   exposed_wait_us=b['joined_us'] - b['demand_us'],
                   register_us=b['register_end_us'] - b['register_start_us'],
                   late_reads=len(late), starts_after_demand=len(queued),
                   running_at_demand=len(running),
                   last_running_finish_after_demand_us=max(
                       [r['end_us'] - b['demand_us'] for r in running], default=0),
                   last_new_start_after_demand_us=max(
                       [r['start_us'] - b['demand_us'] for r in queued], default=0),
                   mean_read_us=statistics.mean(r['end_us'] - r['start_us'] for r in rs),
                   mean_io_us=statistics.mean(r['io_us'] for r in rs),
                   mean_copy_us=statistics.mean(r['copy_us'] for r in rs))
        if after_line and ordinal < len(moes):
            row['moe_ordinal'] = ordinal
            if first_moe_layer is not None:
                row['model_layer'] = first_moe_layer + ordinal
        rows.append(row)
    warmup = groups['WARMUP']
    return dict(log=str(path), batches=len(batches), experts=len(reads),
                direct_calls=sum(r['calls'] for r in reads),
                requested_bytes=sum(r['bytes'] for r in reads),
                service_ms=service / 1000, io_ms=ios / 1000,
                copy_ms=copies / 1000, other_read_ms=(service - ios - copies) / 1000,
                mean_read_us=service / len(reads), mean_io_us=ios / len(reads),
                mean_copy_us=copies / len(reads),
                read_distribution_us=distribution([r['end_us'] - r['start_us'] for r in reads]),
                io_distribution_us=distribution([r['io_us'] for r in reads]),
                copy_distribution_us=distribution([r['copy_us'] for r in reads]),
                read_window_ms=(end - start) / 1000, peak_active_readers=peak,
                active_readers_time_ms={str(k): v / 1000 for k, v in sorted(occupancy.items())},
                effective_requested_gbps=sum(r['bytes'] for r in reads) / (end - start) / 1000,
                exposed_wait_ms=sum(r['exposed_wait_us'] for r in rows) / 1000,
                register_ms=sum(r['register_us'] for r in rows) / 1000,
                compute_after_line=after_line,
                compute_phase='after_load' if after_line else 'unfiltered',
                excluded_warmup_calls=len(warmup),
                excluded_warmup_ms=sum(c['end_us'] - c['start_us'] for c in warmup) / 1000,
                traced_compute_calls=len(compute), traced_moe_calls=len(moes),
                first_compute_us=origin,
                first_read_from_first_compute_ms=(start - origin) / 1000,
                traced_compute_ms=sum(b - a for a, b in comp) / 1000,
                read_any_active_ms=sum(b - a for a, b in active) / 1000,
                read_compute_overlap_ms=overlap(active, comp) / 1000,
                read_wait_overlap_ms=overlap(active, wait) / 1000,
                per_batch=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('log')
    p.add_argument('--out')
    p.add_argument('--after-line', type=int)
    p.add_argument('--first-moe-layer', type=int)
    a = p.parse_args()
    result = analyze(a.log, a.after_line, a.first_moe_layer)
    s = json.dumps(result, indent=2) + '\n'
    if a.out:
        Path(a.out).write_text(s)
    print(s, end='')
