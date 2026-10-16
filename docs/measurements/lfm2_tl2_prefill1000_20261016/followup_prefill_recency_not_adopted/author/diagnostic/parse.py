#!/usr/bin/env python3
"""Strict per-process stderr readback; this format does not carry a PID.

The caller must supply one process's stderr, not mixed-process logcat output.
No timestamp or identical-delivery deduplication is used: every layer appears
exactly once. Counts prove actual hint compaction, not elapsed-time savings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

TAG = '[LFM-RECENCY]'
RECORD = re.compile(r'\[LFM-RECENCY\] layer=(\d+) M=(\d+) raw=(\d+) kept=(\d+) fast=([01]) !')


def parse(text, rows, enabled, layers=38):
    if rows not in (1024, 2048) or enabled not in (0, 1) or layers != 38:
        raise ValueError('expected one supported prompt and 38 LFM MoE layers')
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        if TAG not in line:
            continue
        payload = line[line.index(TAG):]
        match = RECORD.fullmatch(payload)
        if not match:
            raise ValueError(f'malformed/truncated trace at line {number}')
        layer, m, raw, kept, fast = map(int, match.groups())
        if layer != len(records):
            raise ValueError(f'duplicate, missing or reordered layer at line {number}')
        if m != rows or raw != 9 * rows or fast != enabled:
            raise ValueError(f'wrong geometry, raw count or dispatch at line {number}')
        if (enabled and not (9 <= kept <= 64)) or (not enabled and kept != raw):
            raise ValueError(f'wrong hint extent at line {number}')
        records.append(dict(layer=layer, rows=m, raw=raw, kept=kept, fast=fast))
    if len(records) != layers:
        raise ValueError(f'expected {layers} complete layers, observed {len(records)}')
    raw = sum(r['raw'] for r in records)
    kept = sum(r['kept'] for r in records)
    return dict(layers=len(records), rows=rows, enabled=enabled, fast_calls=sum(r['fast'] for r in records),
                raw_entries=raw, kept_entries=kept, eliminated_entries=raw-kept,
                records=records,
                scope='One process stderr is required; PID identity is not encoded or checked. Counts are not a time or TPS measurement.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--rows', type=int, choices=(1024, 2048), required=True)
    p.add_argument('--enabled', type=int, choices=(0, 1), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = parse(a.log.read_text(), a.rows, a.enabled)
    result['log_sha256'] = hashlib.sha256(a.log.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
