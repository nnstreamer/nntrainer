#!/usr/bin/env python3
"""Strict actual-QK-readout trace parsing. Expected records are independent.
A readout is a complete four-plane tile, not a plane or a PV accumulator.
"""
import argparse
import json
from pathlib import Path
import re

TAG = '[LFM-RAW4]'
LINE = re.compile(r'\[LFM-RAW4\] q=(\d+) m=(\d+) on=([01]) raw=(\d+),(\d+) sdk=(\d+) b=(\d+) rc=(-?\d+) END')
PID = re.compile(r'^\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\s+(\d+)\s+\d+\s+[VDIWEAF]\s+')
FIELDS = ('m', 'on', 'raw_s', 'raw_d', 'sdk', 'b', 'rc')


def parse(text, expected, pid=None, host=False):
    if (pid is None) == (not host) or pid is not None and pid <= 0:
        raise ValueError('Require one positive device PID or explicit host mode')
    if not isinstance(expected, list) or not expected:
        raise ValueError('Require independent nonempty expected records')
    for row in expected:
        if set(row) != set(FIELDS) or any(type(row[k]) is not int for k in FIELDS):
            raise ValueError('Invalid expected record')
    seen = {}
    raw = duplicates = 0
    for line in text.splitlines():
        if '[LFM-RAW' not in line:
            continue
        if host:
            if not line.startswith(TAG):
                raise ValueError('Host line prefix')
        else:
            match = PID.match(line)
            if not match or int(match[1]) != pid:
                raise ValueError('Missing or mixed device PID')
        raw += 1
        at = line.find(TAG)
        if at < 0:
            raise ValueError('Truncated tag')
        match = LINE.fullmatch(line[at:].strip())
        if not match:
            raise ValueError('Malformed/truncated record')
        values = list(map(int, match.groups()))
        seq, row = values[0], dict(zip(FIELDS, values[1:]))
        if not 1 <= seq <= 0xffffffff or any(not 0 <= row[k] <= 0xffffffff for k in FIELDS[:-1]):
            raise ValueError('Unsigned overflow')
        if row['rc'] != 0:
            raise ValueError('Unsuccessful attention call')
        if seq in seen:
            if seen[seq] != row:
                raise ValueError('Conflicting sequence replay')
            duplicates += 1
            continue
        if seq != len(seen) + 1:
            raise ValueError('Missing/reordered/reset sequence')
        if seq > len(expected) or row != expected[seq - 1]:
            raise ValueError('Actual dispatch/counter differs from expected record ' + str(seq))
        seen[seq] = row
    if len(seen) != len(expected):
        raise ValueError('Incomplete run: expected %d, got %d' % (len(expected), len(seen)))
    return dict(calls=len(seen), raw_records=raw, transport_duplicates=duplicates,
                selected=sum(x['on'] for x in seen.values()),
                raw_serial=sum(x['raw_s'] for x in seen.values()),
                raw_deferred=sum(x['raw_d'] for x in seen.values()),
                sdk_qk=sum(x['sdk'] for x in seen.values()),
                blocks=sum(x['b'] for x in seen.values()), pid=pid,
                expected_complete=True,
                scope='Actual successful QK readout calls only; PV excluded. No timing, bandwidth or speedup claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--expected', type=Path, required=True)
    p.add_argument('--output', type=Path)
    identity = p.add_mutually_exclusive_group(required=True)
    identity.add_argument('--pid', type=int)
    identity.add_argument('--host-unprefixed', action='store_true')
    a = p.parse_args()
    data = parse(a.log.read_text(), json.loads(a.expected.read_text()), a.pid, a.host_unprefixed)
    result = json.dumps(data, indent=2) + '\n'
    if a.output:
        a.output.write_text(result)
    print(result, end='')
