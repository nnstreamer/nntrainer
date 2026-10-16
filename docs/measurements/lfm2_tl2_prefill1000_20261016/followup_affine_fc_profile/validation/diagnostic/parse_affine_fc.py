#!/usr/bin/env python3
"""Strict parser for this LFM2-24BA2B 1024-row phase diagnostic, not a benchmark."""
import argparse
import collections
import json
import re
from pathlib import Path

TAG = '[LFM-AFFINE-FC]'
FIELDS = set('v M K N nh widths rc stride bank_tiles pipeline reads joins brackets hz body_t pack_t sums_t read_t join_t drain_t clock_pairs clock_t'.split())
PHASES = ('pack_t', 'sums_t', 'read_t', 'join_t', 'drain_t')
EXPECTED = {(1024, 6144): 30, (1024, 2048): 40, (1024, 3072): 10,
            (512, 6144): 1, (64, 6144): 29, (64, 2048): 50, (64, 512): 20}


def record(line, lineno):
    payload = line.split(TAG, 1)[1].strip()
    pairs = [word.split('=', 1) for word in payload.split()]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f'line {lineno}: malformed key/value record')
    keys = [pair[0] for pair in pairs]
    if len(keys) != len(set(keys)) or set(keys) != FIELDS:
        raise ValueError(f'line {lineno}: duplicate, missing or unknown field')
    row = dict(pairs)
    for k in FIELDS - {'widths'}:
        if not re.fullmatch(r'-?\d+' if k == 'rc' else r'\d+', row[k]):
            raise ValueError(f'line {lineno}: invalid integer {k}')
        row[k] = int(row[k])
    if not re.fullmatch(r'\d+,\d+,\d+', row['widths']):
        raise ValueError(f'line {lineno}: invalid widths')
    row['widths'] = tuple(map(int, row['widths'].split(',')))
    widths = row['widths']
    if not (row['v'] == 1 and row['K'] == 2048 and row['M'] > 1 and
            row['hz'] == 19200000 and row['clock_pairs'] == 32):
        raise ValueError(f'line {lineno}: unsupported version/shape/clock')
    if not ((row['nh'] == 1 and widths in ((512, 0, 0), (2048, 0, 0))) or
            (row['nh'] == 3 and widths in ((2048, 2048, 2048), (2048, 512, 512)))):
        raise ValueError(f'line {lineno}: unsupported handle geometry')
    if row['N'] != sum(widths):
        raise ValueError(f'line {lineno}: N does not match handles')
    if row['rc'] or row['pipeline'] != 1 or row['stride'] != 32 or row['bank_tiles'] != 32:
        raise ValueError(f'line {lineno}: error/fallback/unexpected runtime layout')
    blocks = (row['M'] + 63) // 64
    reads = blocks * sum(n // 32 for n in widths)
    joins = 1 + blocks * sum((n // 32 + 31) // 32 for n in widths)
    brackets = reads + joins + row['nh'] + 4  # pack, sum, initial drain, body
    if (row['reads'], row['joins'], row['brackets']) != (reads, joins, brackets):
        raise ValueError(f'line {lineno}: missing/duplicate phase bracket or tile')
    if row['body_t'] <= 0 or sum(row[k] for k in PHASES) > row['body_t']:
        raise ValueError(f'line {lineno}: overlapping/invalid foreground intervals')
    row['line'] = lineno
    return row


def parse(text):
    rows = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if TAG in line:
            if line.count(TAG) != 1:
                raise ValueError(f'line {lineno}: repeated trace tag')
            rows.append(record(line, lineno))
        elif 'LFM-AFFINE-FC' in line:
            raise ValueError(f'line {lineno}: malformed trace tag')
    if not rows:
        raise ValueError('no affine FC records')
    counts = collections.Counter((r['M'], r['N']) for r in rows)
    if dict(counts) != EXPECTED:
        got = {str(k): v for k, v in sorted(counts.items())}
        raise ValueError(f'incomplete/unexpected exact-model cohort: {got}')
    out = {'scope': 'exact LFM2-24BA2B 1024-row prompt; separate load warmups',
           'records': len(rows), 'groups': {}, 'main': {}}
    for key in sorted(counts):
        out['groups'][f'M{key[0]}_N{key[1]}'] = summarize([r for r in rows if (r['M'], r['N']) == key])
    out['main'] = summarize([r for r in rows if r['M'] == 1024])
    if out['main']['calls'] != 80 or out['main']['reads'] != 148480:
        raise ValueError('main prompt does not contain 80 calls / 148480 tiles')
    out['limits'] = [
        'read_t includes the vendor readout and possibly completion of preceding HMX work; it is not pure int32-store cost.',
        'pack_t includes parameter scan, optional pre-norm and AH packing; initial weight DMA overlaps it.',
        'join_t and drain_t are exposed foreground waits; do not add worker/DMA service times.',
        'body_t starts after validation/allocation, before the first weight push; includes final frees, excludes calibration and logging.',
        'Timers alter the hot loop. Empty-pair intervals are a scale reference, not an exact overhead correction; no automatic subtraction.',
        'This parser checks only diagnostic records. The run must separately have exit0, 65 exact IDs, unchanged model/settings, and complete logcat.',
    ]
    return out


def summarize(rows):
    result = {'calls': len(rows)}
    for field in ('reads', 'joins', 'brackets'):
        result[field] = sum(r[field] for r in rows)
    result['qtimer_reads_including_calibration'] = sum(2 * (r['brackets'] + r['clock_pairs']) for r in rows)
    for field in ('body_t', *PHASES):
        result[field.removesuffix('_t') + '_ms'] = sum(r[field] for r in rows) / 19200
    result['other_foreground_ms'] = result['body_ms'] - sum(result[p.removesuffix('_t') + '_ms'] for p in PHASES)
    result['empty_pair_mean_ns'] = sum(r['clock_t'] for r in rows) / sum(r['clock_pairs'] for r in rows) * 1e9 / 19200000
    result['empty_interval_times_brackets_ms_NOT_overhead_correction'] = sum(r['clock_t'] / r['clock_pairs'] * r['brackets'] for r in rows) / 19200
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logcat', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        result = parse(args.logcat.read_text())
    except ValueError as e:
        p.exit(1, f'INVALID TRACE: {e}\n')
    data = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
