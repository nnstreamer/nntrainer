#!/usr/bin/env python3
"""Validate a single-process native-LFM decode GU trace by monotonic sequence.

Repeated transport records are accepted ONLY for the same sequence and identical
payload. Missing/conflicting sequences, partial lines, zero/mismatched execution
counts and incomplete expected workloads are rejected. Log timestamps do not
identify invocations and are never used for deduplication.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

TAG = '[nntr-tl2-gu-direct]'
FIELDS = ('seq', 'M', 'K', 'I', 'N', 'E', 'native', 'lanes', 'fc', 'ft', 'uc', 'ut')
PATTERN = re.compile(re.escape(TAG) + ' ' + ' '.join(n + r'=(\d+)' for n in FIELDS) + r' !')
LOGCAT = re.compile(r'^(\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s+(\d+)\s+(\d+)\s+\S\s+')


def validate(text, tokens, layers, top_k):
    records, occurrences, pids, tids = {}, Counter(), set(), set()
    raw_count, first_seen = 0, []
    times, lengths = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if TAG not in line:
            continue
        raw_count += 1
        payload = line[line.index(TAG):].strip()
        match = PATTERN.fullmatch(payload)
        if not match:
            raise ValueError(f'line {number}: incomplete or malformed sequence record')
        d = dict(zip(FIELDS, map(int, match.groups())))
        seq = d['seq']
        if not 1 <= seq <= 0xffffffff:
            raise ValueError(f'line {number}: invalid sequence {seq}')
        prefix = LOGCAT.match(line)
        if not prefix:
            raise ValueError(f'line {number}: missing logcat timestamp/PID/TID')
        timestamp, pid, tid = prefix.groups()
        pids.add(int(pid)); tids.add(int(tid)); times.append(timestamp)
        lengths.append(len(payload.encode()))
        occurrences[seq] += 1
        if seq in records:
            if records[seq] != d:
                raise ValueError(f'conflicting duplicate sequence {seq}')
            continue
        records[seq] = d
        first_seen.append(seq)
        if (d['M'], d['K'], d['I'], d['N']) != (1, 2048, 1536, 2048):
            raise ValueError(f'sequence {seq}: unsupported shape')
        if not 1 <= d['E'] <= top_k or d['native'] != d['E']:
            raise ValueError(f'sequence {seq}: expected all-native active experts in 1..{top_k}')
        if not 1 <= d['lanes'] <= 48:
            raise ValueError(f'sequence {seq}: invalid actual lane count')
        actual = (d['fc'], d['ft'], d['uc'], d['ut'])
        expected = ((d['E'], 48 * d['E'], 0, 0) if d['native'] >= 4 else
                    (0, 0, d['E'] * d['lanes'], 48 * d['E']))
        if actual != expected:
            raise ValueError(f'sequence {seq}: actual helper counts {actual} != {expected}')
    if not records:
        raise ValueError('no complete sequence records')
    if len(pids) != 1:
        raise ValueError(f'multiple process cohorts {sorted(pids)}')
    # A fresh process loads one skel with the diagnostic sequence starting at 1.
    # No heuristic trimming of warmups, duplicate timestamps, or count ratios.
    # Keys are unique positive integers. This proves 1..N without allocating
    # a range proportional to an untrusted (possibly UINT32_MAX) sequence.
    if max(records) != len(records):
        raise ValueError('missing sequence(s); unique set is not contiguous from 1')
    active = sum(d['E'] for d in records.values())
    expected_active = tokens * layers * top_k
    if active != expected_active:
        raise ValueError(f'active-expert total {active} != complete workload {expected_active}')
    counts = {k: sum(d[k] for d in records.values()) for k in ('fc', 'ft', 'uc', 'ut')}
    tiles = counts['ft'] + counts['ut']
    if tiles != 48 * expected_active:
        raise ValueError('tile total does not cover expected workload')
    old_workers, new_workers = tiles, counts['fc'] + counts['uc']
    by_e = Counter(d['E'] for d in records.values())
    return {
        'valid': True, 'instrumented': True, 'performance_comparison_eligible': False,
        'raw_records': raw_count, 'unique_kernel_calls': len(records),
        'transport_duplicate_records': raw_count - len(records),
        'sequences_with_duplicates': sum(n > 1 for n in occurrences.values()),
        'maximum_transport_occurrences': max(occurrences.values()),
        'sequence_min': min(records), 'sequence_max': max(records),
        'first_seen_sequence_order_monotonic': first_seen == sorted(first_seen),
        'pid': next(iter(pids)), 'transport_tids': sorted(tids),
        'first_delivery_timestamp': min(times), 'last_delivery_timestamp': max(times),
        'payload_byte_max': max(lengths), 'active_experts': active,
        'decode_tokens': tokens, 'moe_layers_per_token': layers, 'top_k': top_k,
        'kernel_calls_by_active_experts': dict(sorted(by_e.items())),
        'actual_pool_lanes': sorted({d['lanes'] for d in records.values()}),
        'direct_counts': counts, 'tile_pairs': tiles,
        'removed_serial_wrapper_calls': old_workers,
        'old_swiglu_worker_calls': old_workers, 'new_swiglu_worker_calls': new_workers,
        'fewer_swiglu_worker_calls': old_workers - new_workers,
        'removed_stack_vector_stores': 2 * tiles,
        'removed_stack_vector_loads': 2 * tiles,
        'removed_stack_logical_access_bytes': 4 * 128 * tiles,
        'per_decode_token': {
            'kernel_calls': len(records) / tokens,
            'serial_wrapper_calls_removed': old_workers / tokens,
            'new_swiglu_worker_calls': new_workers / tokens,
            'swiglu_worker_calls_removed': (old_workers - new_workers) / tokens,
            'stack_logical_access_bytes_removed': 4 * 128 * tiles / tokens},
        'limits': ['Sequence validates transport identity, not kernel arithmetic.',
                   'Workload total assumes one fresh process, one prompt and the explicit tokens/layers/top-k.',
                   'Stack counts are logical vector accesses, not DDR bytes or measured bandwidth.',
                   'Instrumented timing is excluded from normal TPS comparisons.']}


def self_test():
    def line(seq, e, lanes=3):
        fc, ft, uc, ut = (e, 48 * e, 0, 0) if e >= 4 else (0, 0, e * lanes, e * 48)
        return (f'10-16 22:00:00.000 100 101 V adsprpc : {TAG} seq={seq} M=1 K=2048 '
                f'I=1536 N=2048 E={e} native={e} lanes={lanes} fc={fc} ft={ft} uc={uc} ut={ut} !')
    a, b, c = line(1, 4), line(2, 1), line(3, 3)
    good = '\n'.join([a, b, a, c, b])
    r = validate(good, 1, 2, 4)
    assert r['unique_kernel_calls'] == 3 and r['transport_duplicate_records'] == 2
    bad = {
        'missing': '\n'.join([a, c]),
        'conflicting': '\n'.join([a, b, c, b.replace('lanes=3', 'lanes=2')]),
        'zero-path': '\n'.join([a, b.replace('uc=3 ut=48', 'uc=0 ut=0'), c]),
        'truncated': '\n'.join([a, b[:-2], c]),
        'wrong-total': a,
        'wrong-tile-range': '\n'.join([a, b, c.replace('ut=144', 'ut=143')]),
        'wrong-native': '\n'.join([a.replace('native=4', 'native=3'), b, c]),
        'wrong-shape': '\n'.join([a.replace('K=2048', 'K=2816'), b, c]),
        'multiple-pids': '\n'.join([a, b, c.replace(' 100 101 ', ' 200 201 ')]),
        'nonfresh-sequence': '\n'.join([line(2, 4), line(3, 1), line(4, 3)]),
        'sparse-max-u32-sequence': line(0xffffffff, 4),
        'empty': ''}
    rejected = {}
    for name, value in bad.items():
        try:
            validate(value, 1, 2, 4)
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise AssertionError('accepted invalid trace: ' + name)
    return {'valid_transport_replay_fixture': True, 'rejected': rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--layers', type=int, default=38)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if not args.log or min(args.tokens, args.layers, args.top_k) <= 0:
            parser.error('--log and positive workload dimensions required')
        data = args.log.read_bytes()
        try:
            result = validate(data.decode(), args.tokens, args.layers, args.top_k)
        except ValueError as error:
            result = {'valid': False, 'error': str(error)}
        result.update({'source_log': str(args.log.resolve()),
                       'source_log_sha256': hashlib.sha256(data).hexdigest()})
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
    if result.get('valid') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
