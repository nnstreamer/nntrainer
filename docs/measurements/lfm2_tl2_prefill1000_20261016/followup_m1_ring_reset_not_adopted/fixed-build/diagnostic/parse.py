#!/usr/bin/env python3
"""Validate one process's ring-reset attempt records; never infer timing."""
import argparse
import json
from pathlib import Path
import re

TAG = 'nntr-ring-reset '
LINE = re.compile(r'nntr-ring-reset q=(\d+) used=([01]) !')


def parse(text, expected_calls=None, min_hits=1, pid=None, host_unprefixed=False):
    if (pid is None) == (not host_unprefixed) or pid is not None and pid <= 0:
        raise ValueError('Specify one positive device PID or explicit host-unprefixed mode')
    records = {}
    raw_count = duplicates = 0
    for line in text.splitlines():
        if TAG not in line:
            continue
        if host_unprefixed:
            if not line.startswith(TAG):
                raise ValueError('Host fixture mode requires unprefixed records')
        else:
            prefix = re.match(r'^\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\s+(\d+)\s+\d+\s+[VDIWEF]\s+', line)
            if not prefix or int(prefix[1]) != pid:
                raise ValueError('Missing or mixed process identity')
        raw_count += 1
        record = line[line.index(TAG):].strip()
        match = LINE.fullmatch(record)
        if not match:
            raise ValueError('Malformed/truncated reset record')
        seq, used = map(int, match.groups())
        if not 1 <= seq <= 0xffffffff:
            raise ValueError('Sequence overflow/reset')
        if seq in records:
            if records[seq] != used:
                raise ValueError('Conflicting duplicate sequence')
            duplicates += 1
        else:
            # Require the first observation of each ID in order. Exact transport
            # replays of earlier IDs are accepted and counted explicitly.
            if seq != len(records) + 1:
                raise ValueError('Missing or reordered sequence')
            records[seq] = used
    if not records:
        raise ValueError('No ring-reset observations')
    if expected_calls is not None and len(records) != expected_calls:
        raise ValueError('Unique call count differs')
    hits = sum(records.values())
    if hits < min_hits:
        raise ValueError('Insufficient actual fast-reset hits')
    return {'observed_unique_attempts': len(records), 'hits': hits,
            'fallbacks': len(records) - hits, 'raw_records': raw_count,
            'exact_transport_duplicates': duplicates,
            'full_count_independently_specified': expected_calls is not None,
            'device_pid': pid, 'host_unprefixed': host_unprefixed,
            'logical_descriptor_clear_bytes_avoided': hits * 256 * 128,
            'scope': 'One process; actual try_reset returns only. An unspecified expected count proves a contiguous observed prefix, not an untruncated end. No later RPC-success, DMA-visibility, physical-traffic or TPS claim.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--expected-calls', type=int)
    p.add_argument('--min-hits', type=int, default=1)
    identity = p.add_mutually_exclusive_group(required=True)
    identity.add_argument('--pid', type=int, help='Child PID from the supervisor/one-process run metadata')
    identity.add_argument('--host-unprefixed', action='store_true', help='Explicit bare-line host-fixture mode')
    a = p.parse_args()
    if a.expected_calls is not None and a.expected_calls <= 0 or a.min_hits < 0:
        p.error('Counts must be nonnegative and expected calls positive')
    print(json.dumps(parse(a.log.read_text(), a.expected_calls, a.min_hits, a.pid, a.host_unprefixed), indent=2))
