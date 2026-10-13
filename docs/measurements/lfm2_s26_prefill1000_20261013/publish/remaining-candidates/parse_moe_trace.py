#!/usr/bin/env python3
"""Summarize existing NNTR_MOE_TRACE, with no device access or model changes.

For the published LFM2 virtual-expert path, each row is:
    layer rows | ascending active expert IDs | rows * min(top_k + 5, E) IDs
Only the first top_k entries of each ranking contribute to the histogram.
Initial-cache and read-ahead attribution is a source-derived reconstruction,
validated against aggregate profile counters; it is not a per-read IO trace.
"""
import argparse
import hashlib
import json
import pathlib
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_trace(text, experts=64, top_k=4, extra_topk=5):
    n_sel = min(top_k + extra_topk, experts)
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('|')
        if len(fields) != 3:
            raise ValueError(f'line {line_no}: expected three pipe-separated fields')
        header = [int(v) for v in fields[0].split()]
        if len(header) != 2 or header[1] < 1:
            raise ValueError(f'line {line_no}: bad layer/rows header')
        layer, rows = header
        active = [int(v) for v in fields[1].split()]
        rankings = [int(v) for v in fields[2].split()]
        if active != sorted(set(active)) or any(e < 0 or e >= experts for e in active):
            raise ValueError(f'line {line_no}: active IDs must be unique, sorted and in range')
        if len(rankings) != rows * n_sel:
            raise ValueError(f'line {line_no}: expected {rows * n_sel} ranking IDs, got {len(rankings)}')
        counts = [0] * experts
        for row in range(rows):
            selected = rankings[row * n_sel:(row + 1) * n_sel]
            if len(set(selected)) != n_sel or any(e < 0 or e >= experts for e in selected):
                raise ValueError(f'line {line_no}, row {row}: invalid or duplicate ranking IDs')
            for e in selected[:top_k]:
                counts[e] += 1
        derived_active = [e for e, n in enumerate(counts) if n]
        if active != derived_active:
            raise ValueError(f'line {line_no}: active IDs disagree with first top_k ranking entries')
        assert sum(counts) == rows * top_k
        records.append({'trace_line': line_no, 'layer_ordinal': layer, 'rows': rows,
                        'n_sel': n_sel, 'active_count': len(active), 'active_ids': active,
                        'zero_count': experts - len(active),
                        'zero_ids': [e for e, n in enumerate(counts) if not n],
                        'row_count_by_expert': counts, 'assignment_count': sum(counts)})
    return records


def initial_cache_by_layer(layers, experts, capacity):
    result = []
    remaining = min(capacity, layers * experts)
    for _ in range(layers):
        n = min(experts, remaining)
        result.append(list(range(n)))
        remaining -= n
    return result


def one_tag(log, tag):
    lines = [line for line in log.splitlines() if line.startswith('[' + tag + '] ')]
    if len(lines) != 1:
        return None
    return {k: int(v) for k, v in re.findall(r'(\w+)=([0-9]+)', lines[0])}


def self_test():
    # Extra ranking hints 4 and 5 are NOT active. Two rows have different top4.
    trace = '0 2 | 0 1 2 3 4 | 0 1 2 3 4 5 4 3 2 1 0 5\n'
    row = parse_trace(trace, experts=6, top_k=4, extra_topk=2)[0]
    assert row['row_count_by_expert'] == [1, 2, 2, 2, 1, 0]
    assert row['active_count'] == 5 and row['zero_ids'] == [5]
    assert initial_cache_by_layer(3, 6, 8) == [list(range(6)), [0, 1], []]
    for bad in [trace.replace('| 0 1 2 3 4 |', '| 0 1 2 3 4 5 |'),
                trace.replace('0 1 2 3 4 5 4', '0 1 2 3 4 4 4'),
                trace.replace('0 2 |', '0 3 |')]:
        try:
            parse_trace(bad, experts=6, top_k=4, extra_topk=2)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed trace was accepted')
    print('PASS: top-k vs extra hints, histogram, initial cache, malformed-record guards')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', type=pathlib.Path)
    ap.add_argument('--profile-log', type=pathlib.Path)
    ap.add_argument('--run-json', type=pathlib.Path)
    ap.add_argument('--output', type=pathlib.Path)
    ap.add_argument('--experts', type=int, default=64)
    ap.add_argument('--top-k', type=int, default=4)
    ap.add_argument('--extra-topk', type=int, default=5)
    ap.add_argument('--layers', type=int, default=38)
    ap.add_argument('--cache-per-layer', type=int, default=22)
    ap.add_argument('--expected-rows', type=int, default=1024)
    ap.add_argument('--tail-rows', type=int, default=64)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.trace is None or args.output is None:
        ap.error('--trace and --output are required')
    records = parse_trace(args.trace.read_text(), args.experts, args.top_k, args.extra_topk)
    prefill = [r for r in records if r['rows'] > 1]
    if len(prefill) != args.layers or [r['layer_ordinal'] for r in prefill] != list(range(args.layers)):
        raise ValueError('expected exactly one ordered prefill pass over all layer ordinals')
    wanted_rows = [args.expected_rows] * (args.layers - 1) + [args.tail_rows]
    if [r['rows'] for r in prefill] != wanted_rows:
        raise ValueError('prefill row shape differs from requested full/tail layout')
    log = args.profile_log.read_text() if args.profile_log else ''
    run = json.loads(args.run_json.read_text()) if args.run_json else {}
    env = run.get('env', {})
    observed_prefetch = re.search(r'expert prefetch: (\d+) experts read', log)
    observed_progress = re.search(r'expert prefetch progress: (\d+) of (\d+) read', log)
    m0 = re.findall(r'\[M0-PROF\].*?tokens=(\d+).*?miss=(\d+)', log)
    qs2 = one_tag(log, 'HTP-QS2-IO')
    host_stage = one_tag(log, 'HTP-HOST-STAGING-ACTIVE')
    host_done = one_tag(log, 'HTP-HOST-STAGING-DONE')
    capacity = args.cache_per_layer * args.layers
    initial = initial_cache_by_layer(args.layers, args.experts, capacity)
    expected_prefetch = args.layers * args.experts - sum(map(len, initial))
    expected_weight_reads = 2 * args.layers * args.experts
    conditions = {
        'matching_run_env_cache': env.get('NNTR_MOE_CACHE_EXPERTS') == str(args.cache_per_layer),
        'virtual_experts_enabled': args.cache_per_layer > 0,
        'read_ahead_enabled': int(env.get('NNTR_MOE_PREFETCH', '1048576')) > 0,
        'exact_prefetch_descriptor_total': bool(observed_prefetch) and int(observed_prefetch.group(1)) == expected_prefetch,
        'all_layer_m0_demand_misses_zero': len(m0) == args.layers and all(int(miss) == 0 for _, miss in m0),
        'exact_qs2_weight_attempt_total': bool(qs2) and qs2.get('weights') == expected_weight_reads,
        'profile_not_followed_by_decode': not run.get('generation', {}).get('tokens', 0),
    }
    reconstruction_validated = all(conditions.values())
    expert_bytes = None
    if host_stage and host_stage.get('count', 0):
        payload = host_stage['payload_bytes']
        if payload % host_stage['count'] == 0:
            expert_bytes = payload // host_stage['count']
    host_count = host_stage.get('count', 0) if host_stage else 0
    remaining_host = host_count
    totals = dict(initial_preload_experts=0, active_initial_cache_experts=0,
                  inactive_initial_preload_experts=0, requested_prefetch_descriptors=0,
                  active_prefetch_experts=0, inactive_prefetch_descriptors=0,
                  host_staged_experts=0, inactive_host_staged_experts=0)
    for row, cached in zip(prefill, initial):
        active = set(row['active_ids'])
        planned = [e for e in range(args.experts) if e not in cached]
        staged = planned[:remaining_host]
        remaining_host -= len(staged)
        fields = {
            'initial_cache_ids': cached,
            'active_initial_cache_ids': [e for e in cached if e in active],
            'inactive_initial_preload_ids': [e for e in cached if e not in active],
            'requested_prefetch_descriptor_ids': planned,
            'active_prefetch_ids': [e for e in planned if e in active],
            'inactive_prefetch_descriptor_ids': [e for e in planned if e not in active],
            'host_staged_ids': staged,
            'inactive_host_staged_ids': [e for e in staged if e not in active],
        }
        row['reconstructed_schedule'] = fields
        if expert_bytes is not None:
            fields['inactive_prefetch_logical_bytes'] = len(fields['inactive_prefetch_descriptor_ids']) * expert_bytes
            fields['inactive_preload_logical_bytes_outside_prefill'] = len(fields['inactive_initial_preload_ids']) * expert_bytes
        for key, ids in [('initial_preload_experts', cached),
                         ('active_initial_cache_experts', fields['active_initial_cache_ids']),
                         ('inactive_initial_preload_experts', fields['inactive_initial_preload_ids']),
                         ('requested_prefetch_descriptors', planned),
                         ('active_prefetch_experts', fields['active_prefetch_ids']),
                         ('inactive_prefetch_descriptors', fields['inactive_prefetch_descriptor_ids']),
                         ('host_staged_experts', staged),
                         ('inactive_host_staged_experts', fields['inactive_host_staged_ids'])]:
            totals[key] += len(ids)
    if expert_bytes is not None:
        totals['inactive_prefetch_logical_bytes'] = totals['inactive_prefetch_descriptors'] * expert_bytes
        totals['inactive_preload_logical_bytes_outside_prefill'] = totals['inactive_initial_preload_experts'] * expert_bytes
        totals['inactive_host_staging_logical_bytes_subset_of_prefetch'] = totals['inactive_host_staged_experts'] * expert_bytes
    report = {
        'status': 'DIAGNOSTIC ROUTING ANALYSIS; TPS excluded',
        'inputs': {str(p): sha(p) for p in [args.trace, args.profile_log, args.run_json] if p},
        'format': 'layer rows | ascending active IDs | n_sel ranking IDs per token; only first top_k counted',
        'config': {'experts': args.experts, 'top_k': args.top_k, 'extra_topk': args.extra_topk,
                   'layers': args.layers, 'cache_per_layer': args.cache_per_layer, 'total_cache_capacity': capacity},
        'prefill_layers': prefill,
        'aggregate_active_count': sum(r['active_count'] for r in prefill),
        'aggregate_zero_count': sum(r['zero_count'] for r in prefill),
        'full1024_active_counts': [r['active_count'] for r in prefill if r['rows'] == args.expected_rows],
        'tail_active_count': prefill[-1]['active_count'],
        'observed_profile': {'prefetch_descriptors': int(observed_prefetch.group(1)) if observed_prefetch else None,
                             'prefetch_done_on_entry': int(observed_progress.group(1)) if observed_progress else None,
                             'qs2': qs2, 'host_staging_active': host_stage, 'host_staging_done': host_done},
        'schedule_reconstruction_validated': reconstruction_validated,
        'schedule_validation_conditions': conditions,
        'reconstructed_totals': totals,
        'logical_bytes_per_expert_from_observed_host_bank': expert_bytes,
        'source_schedule': 'v17 9b2ced98f lfm2_moe_layer.cpp: initial preload fills global capacity in layer/expert order; future-layer resident weights remain pinned; queue_future requests every nonresident descriptor; take_batch registers all; stage(active) only demands missing active experts.',
        'limitations': [
            'Routing histogram is observed. Initial-cache/prefetch/host-bank IDs are reconstructed from the unchanged source schedule, not from per-descriptor IO events.',
            'Use reconstructed unneeded-read totals only if schedule_reconstruction_validated=true. Source changes to caching/prefetch would require a new reconstruction.',
            'A read for a count=0 expert is logically unnecessary for this prefill layer; it may still be useful for later decode tokens.',
            'Logical payload bytes are not physical storage bytes, page-rounded O_DIRECT bytes, or measured wall-time savings.',
            'Reads already finished cannot be undone after router returns; only unstarted unused reads can be cancelled without having consumed their traffic. In-flight slots remain owned until completion.',
            'Initial-preload bytes occur before timed prefill. Host-staged bytes are a subset of prefetch bytes and must not be added twice.',
            'This observer performs substantial integer formatting and file output; exclude its TPS from clean timing.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)
    print('active counts:', [r['active_count'] for r in prefill])
    print('observed inactive experts:', report['aggregate_zero_count'])
    print('schedule validated:', reconstruction_validated)
    print('reconstructed totals:', json.dumps(totals, sort_keys=True))


if __name__ == '__main__':
    main()
