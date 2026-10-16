#!/usr/bin/env python3
"""Strict two-packet v2 trace parser for the exact 1024-row LFM2 model."""
import argparse
import collections
import json
import re
from pathlib import Path

PHASES = ('pack_t', 'sums_t', 'read_t', 'join_t', 'drain_t')
EXPECTED = {(1024, 6144): 30, (1024, 2048): 40, (1024, 3072): 10,
            (512, 6144): 1, (64, 6144): 29, (64, 2048): 50, (64, 512): 20}
MAX_PAYLOAD = 192


def number(s, base, bits, signed=False):
    pattern = r'-?\d+' if signed else (r'[0-9a-f]+' if base == 16 else r'\d+')
    if not re.fullmatch(pattern, s):
        raise ValueError('malformed integer')
    value = int(s, base)
    low = -(1 << (bits - 1)) if signed else 0
    high = (1 << (bits - 1 if signed else bits)) - 1
    if not low <= value <= high:
        raise ValueError('out-of-range integer')
    return value


def packet(line, lineno):
    start = line.find('[LFM-AFC2')
    if start < 0 or line.count('LFM-AFC2') != 1:
        raise ValueError(f'line {lineno}: malformed/repeated v2 tag')
    payload = line[start:].strip()
    if len(payload.encode('ascii', 'strict')) > MAX_PAYLOAD:
        raise ValueError(f'line {lineno}: payload exceeds bounded format')
    match = re.fullmatch(r'\[LFM-AFC2([GT])\] (.+) !', payload)
    if not match:
        raise ValueError(f'line {lineno}: incomplete packet/terminal marker')
    kind, body = match.groups()
    parts = [word.split('=', 1) for word in body.split()]
    if any(len(part) != 2 for part in parts):
        raise ValueError(f'line {lineno}: malformed key/value')
    fields = dict(parts)
    wanted = {'s', 'd', 'x', 'c'} if kind == 'G' else {'s', 't', 'k'}
    if len(parts) != len(fields) or set(fields) != wanted:
        raise ValueError(f'line {lineno}: duplicate/missing/unknown field')
    try:
        ident = number(fields['s'], 16, 64)
        groups = {k: v.split(',') for k, v in fields.items() if k != 's'}
        lengths = {'d': 7, 'x': 4, 'c': 3} if kind == 'G' else {'t': 6, 'k': 3}
        if any(len(groups[k]) != size for k, size in lengths.items()):
            raise ValueError('wrong grouped-field cardinality')
        if kind == 'G':
            d = [number(x, 10, 32) for x in groups['d']]
            x = [number(v, 10, 32, signed=i in (0, 3)) for i, v in enumerate(groups['x'])]
            c = [number(v, 10, 32) for v in groups['c']]
            row = dict(zip(('M', 'K', 'N', 'nh'), d[:4]))
            row['widths'] = tuple(d[4:])
            row.update(zip(('rc', 'stride', 'bank_tiles', 'pipeline'), x))
            row.update(zip(('reads', 'joins', 'brackets'), c))
        else:
            t = [number(x, 16, 64) for x in groups['t']]
            k = [number(v, 16 if i == 2 else 10, 64 if i == 2 else 32)
                 for i, v in enumerate(groups['k'])]
            row = dict(zip(('body_t', *PHASES), t))
            row.update(zip(('hz', 'clock_pairs', 'clock_t'), k))
    except ValueError as e:
        raise ValueError(f'line {lineno}: {e}') from e
    return kind, ident, row


def validate(row, ident):
    widths = row['widths']
    if not (row['K'] == 2048 and row['M'] > 1 and row['hz'] == 19200000 and row['clock_pairs'] == 32):
        raise ValueError(f'ID {ident:x}: unsupported shape/clock')
    if not ((row['nh'] == 1 and widths in ((512, 0, 0), (2048, 0, 0))) or
            (row['nh'] == 3 and widths in ((2048, 2048, 2048), (2048, 512, 512)))):
        raise ValueError(f'ID {ident:x}: unsupported handle geometry')
    if row['N'] != sum(widths):
        raise ValueError(f'ID {ident:x}: N does not match handles')
    if row['rc'] or row['pipeline'] != 1 or row['stride'] != 32 or row['bank_tiles'] != 32:
        raise ValueError(f'ID {ident:x}: error/fallback/unexpected runtime layout')
    blocks = (row['M'] + 63) // 64
    reads = blocks * sum(n // 32 for n in widths)
    joins = 1 + blocks * sum((n // 32 + 31) // 32 for n in widths)
    brackets = reads + joins + row['nh'] + 4
    if (row['reads'], row['joins'], row['brackets']) != (reads, joins, brackets):
        raise ValueError(f'ID {ident:x}: missing/duplicate phase bracket or tile')
    if row['body_t'] <= 0 or sum(row[k] for k in PHASES) > row['body_t']:
        raise ValueError(f'ID {ident:x}: overlapping/invalid foreground intervals')
    return dict(row, start_tick=ident)


def parse(text):
    pairs = {}
    trace_lines = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if 'LFM-AFFINE-FC' in line:
            raise ValueError(f'line {lineno}: rejected v1 trace mixed into v2 input')
        if 'LFM-AFC2' not in line:
            continue
        kind, ident, row = packet(line, lineno)
        parts = pairs.setdefault(ident, {})
        if kind in parts:
            raise ValueError(f'line {lineno}: duplicate/colliding ID {ident:x} part {kind}')
        parts[kind] = row
        trace_lines += 1
    if not pairs:
        raise ValueError('no v2 affine FC records')
    incomplete = [f'{ident:x}' for ident, parts in pairs.items() if set(parts) != {'G', 'T'}]
    if incomplete:
        raise ValueError('missing geometry/timing counterpart for IDs: ' + ','.join(incomplete))
    if len(pairs) != 180 or trace_lines != 360:
        raise ValueError(f'incomplete/duplicate cohort: {len(pairs)} IDs/{trace_lines} packets, expected180/360')
    rows = [validate(dict(parts['G'], **parts['T']), ident) for ident, parts in pairs.items()]
    counts = collections.Counter((r['M'], r['N']) for r in rows)
    if dict(counts) != EXPECTED:
        raise ValueError('incomplete/unexpected exact-model cohort: ' + str(dict(counts)))
    out = {'version': 2, 'scope': 'exact LFM2-24BA2B 1024-row prompt; separate load warmups',
           'records': len(rows), 'packets': trace_lines, 'groups': {}, 'main': {}}
    for key in sorted(counts):
        out['groups'][f'M{key[0]}_N{key[1]}'] = summarize([r for r in rows if (r['M'], r['N']) == key])
    out['main'] = summarize([r for r in rows if r['M'] == 1024])
    if out['main']['calls'] != 80 or out['main']['reads'] != 148480:
        raise ValueError('main prompt does not contain80 calls/148480 tiles')
    out['limits'] = [
        'read_t includes vendor readout and possibly completion of preceding HMX work; not pure int32-store cost or a saving prediction.',
        'pack_t includes parameter scan, optional pre-norm and AH packing; initial weight DMA overlaps it.',
        'join_t and drain_t are exposed foreground waits; do not add worker/DMA service times.',
        'body_t starts after validation/allocation, before the first weight push; includes final frees, excludes calibration and both log calls.',
        'Timer placement/count is unchanged from v1. Empty-pair intervals are a scale reference, not an exact overhead correction; no subtraction.',
        'Log transport must preserve both packets. The run separately requires exit0,65 exact IDs and unchanged model/settings.',
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
    except (ValueError, UnicodeError) as e:
        p.exit(1, f'INVALID TRACE: {e}\n')
    data = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
