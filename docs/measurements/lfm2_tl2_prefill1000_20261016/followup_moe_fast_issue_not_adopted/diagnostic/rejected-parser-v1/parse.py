#!/usr/bin/env python3
"""Validate actual per-call GU/DN HMX leaf counts; never derive performance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

PREFIX = '[LFM-MISSUE]'
PATTERN = re.compile(r'\[LFM-MISSUE\] q=([0-9]{1,10}) s=([0-9a-f]{1,16}) d=([0-9]{1,10}),([0-9]{1,10}) b=(-?[0-9]{1,10}) '
                     r't=([0-9]{1,10}),([0-9]{1,10}) f=([0-9]{1,10}),([0-9]{1,10}) r=(-?[0-9]{1,10}) !')


def packet(line):
    if PREFIX not in line:
        return None
    match = PATTERN.fullmatch(line[line.index(PREFIX):].strip())
    if not match:
        raise ValueError('Malformed/truncated dispatch packet: ' + line.rstrip())
    values = [int(match[1]), int(match[2], 16), *(int(x) for x in match.groups()[2:])]
    row = dict(zip(('seq', 'id', 'M', 'E', 'selected', 'gu', 'dn', 'fast_gu', 'fast_dn', 'rc'), values))
    for field in ('seq', 'M', 'E', 'gu', 'dn', 'fast_gu', 'fast_dn'):
        if not 0 <= row[field] < 2**32:
            raise ValueError('Out-of-range unsigned field ' + field)
    if not -(2**31) <= row['rc'] < 2**31 or row['selected'] not in (-1, 0, 1):
        raise ValueError('Out-of-range signed field')
    if not row['seq'] or row['M'] <= 1 or not 1 <= row['E'] <= 64:
        raise ValueError('Outside trace shape')
    if row['fast_gu'] > row['gu'] or row['fast_dn'] > row['dn']:
        raise ValueError('Fast count exceeds total')
    return row


def parse(lines, rows, calls=38):
    unique = {}
    duplicates = 0
    pids = set()
    packet_count = prefixed_packets = 0
    for line in lines:
        row = packet(line)
        if row is None:
            continue
        packet_count += 1
        pid = re.match(r'^\d{2}-\d{2}\s+\d\d:\d\d:\d\d\.\d+\s+([0-9]{1,10})\s+\d+\s+\w\s+', line)
        if pid:
            pids.add(int(pid[1]))
            prefixed_packets += 1
        if row['seq'] in unique:
            if row != unique[row['seq']]:
                raise ValueError('Conflicting duplicated call ID')
            duplicates += 1
        else:
            unique[row['seq']] = row
    if len(pids) > 1 or (pids and prefixed_packets != packet_count):
        raise ValueError('Mixed Android processes or packet prefix formats')
    if sorted(unique) != list(range(1, len(unique) + 1)):
        raise ValueError('Missing/noncontiguous dispatch sequence')
    starts = [unique[seq]['id'] for seq in sorted(unique)]
    if any(a >= b for a, b in zip([0] + starts, starts)):
        raise ValueError('Duplicate/nonmonotonic start tick by sequence')
    main = [r for r in unique.values() if r['M'] == rows]
    if len(main) != calls:
        raise ValueError(f'Expected {calls} main calls, got {len(main)}')
    for r in unique.values():
        if r['rc']:
            raise ValueError('Failed RPC/kernel included')
    for r in main:
        if (r['selected'] != 1 or not r['gu'] or r['gu'] % 96 or
            r['dn'] % 64 or r['gu'] // 96 != r['dn'] // 64 or
            r['fast_gu'] != r['gu'] or r['fast_dn'] != r['dn']):
            raise ValueError('Main call did not execute every expected native leaf')
    return {
        'pass': True, 'rows': rows, 'main_calls': len(main),
        'unique_calls': len(unique), 'identical_duplicate_packets': duplicates,
        'android_pids': sorted(pids),
        'cohorts_by_M': dict(sorted(Counter(r['M'] for r in unique.values()).items())),
        'main_compacted_experts': sum(r['E'] for r in main),
        'main_gu_calls': sum(r['gu'] for r in main),
        'main_dn_calls': sum(r['dn'] for r in main),
        'main_gu_hmx_issues': sum(r['fast_gu'] * 64 for r in main),
        'main_dn_hmx_issues': sum(r['fast_dn'] * 48 for r in main),
        'main_blocks': sum(r['gu'] // 96 for r in main),
        'all_main_leaf_used': True,
        'limit': 'Diagnostic dispatch counts only; not timing or normal TPS'
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--rows', type=int, choices=(1024, 2048, 4096), required=True)
    p.add_argument('--calls', type=int, default=38)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = parse(a.log.read_text().splitlines(), a.rows, a.calls)
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.write_text(text)
    print(text, end='')
