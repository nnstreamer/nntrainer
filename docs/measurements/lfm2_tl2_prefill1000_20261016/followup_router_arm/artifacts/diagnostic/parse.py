#!/usr/bin/env python3
"""Strict app prefill CPU-router completion records; not a timing parser."""
import argparse
import json
from pathlib import Path
import re

TAG = '[LFM-ARM-ROUTER]'
KEYS = ('q', 'm', 'k', 'e', 'enter', 'used', 'fallback', 'tasks', 'done',
        'rejected', 'finish')
LINE = re.compile(re.escape(TAG) + ' ' + ' '.join(
    key + r'=(\d{1,10})' for key in KEYS) + r' !')


def packets(text):
    records = []
    for line in text.splitlines():
        if TAG not in line:
            continue
        match = LINE.fullmatch(line[line.index(TAG):].strip())
        if not match:
            raise ValueError('Malformed or truncated CPU-router marker')
        values = tuple(map(int, match.groups()))
        if any(v > 0xffffffff for v in values):
            raise ValueError('Out-of-range field')
        records.append(dict(zip(KEYS, values)))
        if len(records) > 38:
            raise ValueError('More than 38 app prefill records')
    return records


def parse(text, rows, enabled, expect_fast=None):
    if rows not in (1024, 2048) or enabled not in (0, 1):
        raise ValueError('Unsupported cohort')
    records = packets(text)
    if len(records) != 38:
        raise ValueError(f'Expected 38 app prefill layers, observed {len(records)}')
    for q, r in enumerate(records, 1):
        if (r['q'], r['m'], r['k'], r['e'], r['enter'], r['finish']) != (
                q, rows, 2048, 64, 1, 1):
            raise ValueError('Unexpected order, geometry, helper entry or failed route')
        if r['used'] not in (0, 1) or r['fallback'] != 1 - r['used']:
            raise ValueError('Invalid completed route')
        if r['tasks'] not in (0, rows // 4) or r['done'] != r['tasks']:
            raise ValueError('Incomplete or incorrectly sized task dispatch')
        if r['rejected'] > r['done']:
            raise ValueError('More rejected tasks than completed tasks')
        if r['used'] and (r['tasks'] != rows // 4 or r['rejected'] != 0):
            raise ValueError('ARM result consumed without complete successful tasks')
        if not r['used'] and r['tasks'] and not r['rejected']:
            raise ValueError('Unexplained post-task fallback')
        if not enabled and any(r[k] for k in ('used', 'tasks', 'done', 'rejected')):
            raise ValueError('OFF cohort performed ARM work')
    fast = sum(r['used'] for r in records)
    if expect_fast is not None and fast != expect_fast:
        raise ValueError(f'Expected {expect_fast} ARM routes, observed {fast}')
    return {'calls': 38, 'rows': rows, 'enabled': enabled, 'fast': fast,
            'fallback': 38 - fast, 'tasks': sum(r['tasks'] for r in records),
            'completed': sum(r['done'] for r in records),
            'rejected': sum(r['rejected'] for r in records),
            'scope': 'Actual app helper and completed routes; no timing inference',
            'records': records}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('log', type=Path)
    p.add_argument('--rows', type=int, choices=(1024, 2048), required=True)
    p.add_argument('--enabled', type=int, choices=(0, 1), required=True)
    p.add_argument('--expect-fast', type=int, choices=range(39))
    a = p.parse_args()
    print(json.dumps(parse(a.log.read_text(), a.rows, a.enabled, a.expect_fast), indent=2))
