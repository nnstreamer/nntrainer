#!/usr/bin/env python3
"""Count router handoffs in raw/filtered logcat; no device access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('log', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--single-block-on', action='store_true')
p.add_argument('--source-separated', action='store_true')
a = p.parse_args()
rows, producers, states, publishes = [], [], [], []
for line in a.log.read_text(errors='replace').splitlines():
    for tag, dest, expected in [
        ('handoff', rows, 10), ('handoff-producer', producers, 4),
        ('handoff-state', states, None), ('handoff-publish', publishes, None),
    ]:
        if '[nntr-router-' + tag + ']' not in line:
            continue
        values = dict(re.findall(r'(\w+)=([^\s]+)', line.split(']')[-1]))
        d = {k: int(v) if re.fullmatch(r'-?\d+', v) else v for k, v in values.items()}
        if expected is not None:
            assert len(d) == expected, line
        dest.append(d)
r = {
    'log': str(a.log), 'calls': len(rows), 'producers': len(producers),
    'requested': sum(x['request'] for x in rows),
    'stamp_matches': sum(x['owner'] for x in rows),
    'input_copies_skipped': sum(x['used'] for x in rows),
    'bytes_skipped': sum(x['bytes'] * x['used'] for x in rows),
    'requested_fallbacks': [x for x in rows if x['fallback']],
    'unticketed_calls': [i for i, x in enumerate(rows) if not x['request']],
    'state_masks': dict(Counter(str(x['match']) for x in states if x['request_ticket'])),
    'rows': rows, 'producer_rows': producers, 'state_rows': states, 'publish_rows': publishes,
}
if a.single_block_on:
    assert len(rows) == 39 and len(producers) == 38, r
    assert not rows[0]['request'] and r['requested'] == 38, r
    assert all(x['ticket'] and x['cached'] for x in producers), r
    assert all(x['rc'] == 0 for x in rows), r
    # First M2048 call may outgrow the E64 x 64-row warmup reserve.
    assert r['input_copies_skipped'] in (37, 38), r
    for x in r['requested_fallbacks']:
        assert x['owner'] and x['cap_after'] > x['cap_before'], x
if a.source_separated:
    assert a.single_block_on, '--source-separated requires --single-block-on'
    assert len(states) == 39 and len(publishes) == 38, r
    assert all(x['source_scope'] == 'arm' for x in states + publishes), r
    assert all(x['match'] == 31 for x in states if x['request_ticket']), r
if a.output:
    a.output.write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({k: v for k, v in r.items() if not k.endswith('rows')}, indent=2))
