#!/usr/bin/env python3
"""Strict app-only row-normalization dispatch count; not a timing parser."""
import argparse
import json
from pathlib import Path
import re

TAG = '[LFM-ROW-NORM]'
LINE = re.compile(r'\[LFM-ROW-NORM\] q=(\d+) m=(\d+) k=(\d+) e=(\d+) used=(\d+) jobs=(\d+) !')


def parse(text, rows, enabled):
    records = []
    for line in text.splitlines():
        if TAG not in line:
            continue
        match = LINE.fullmatch(line[line.index(TAG):].strip())
        if not match:
            raise ValueError('Malformed/truncated row norm marker')
        fields = tuple(map(int, match.groups()))
        if any(value > 0xffffffff for value in fields):
            raise ValueError('Out-of-range field')
        records.append(fields)
    if len(records) != 38:
        raise ValueError(f'Expected 38 app prefill layers, observed {len(records)}')
    for q, record in enumerate(records, 1):
        expected = (q, rows, 2048, 64, enabled, rows // 64 if enabled else 0)
        if record != expected:
            raise ValueError(f'Unexpected record {record}; expected {expected}')
    return {'calls': 38, 'rows': rows, 'used': 38 * enabled,
            'jobs_per_call': rows // 64 if enabled else 0,
            'scope': 'Actual helper completion, app incremental path only; no timing inference'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('log', type=Path)
    ap.add_argument('--rows', type=int, choices=(1024, 2048), required=True)
    ap.add_argument('--enabled', type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    print(json.dumps(parse(args.log.read_text(), args.rows, args.enabled), indent=2))
