#!/usr/bin/env python3
"""Positive and intentional-rejection checks for actual handoff trace parsing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--parser', type=Path, required=True)
p.add_argument('--capacity-table', type=Path, required=True)
p.add_argument('--model1024', type=Path, required=True)
p.add_argument('--model2048', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('handoff_trace_parser', a.parser)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
capacity = json.loads(a.capacity_table.read_text())
texts = {1024: a.model1024.read_text(), 2048: a.model2048.read_text()}
results = []
for M, text in texts.items():
    r = parser.validate(parser.parse_text(text), capacity)
    assert r['input_copies_skipped'] == (38 if M == 1024 else 35)
    results.append({'name': f'actual{M}', 'result': 'PASS',
                    'skipped': r['input_copies_skipped'], 'bytes': r['bytes_skipped']})


def target(line, tag, field, value):
    return ('[nntr-router-' + tag + ']') in line and re.search(
        r'\b' + re.escape(field) + '=' + str(value) + r'(?:\s|$)', line)


def edit(text, tag, field, value, changes=None, action=None):
    lines = text.splitlines()
    indexes = [i for i, line in enumerate(lines) if target(line, tag, field, value)]
    assert len(indexes) == 1, (tag, field, value, len(indexes))
    i = indexes[0]
    if action == 'delete':
        del lines[i]
    elif action == 'duplicate':
        lines.insert(i, lines[i])
    else:
        for key, val in changes.items():
            lines[i], n = re.subn(r'\b' + re.escape(key) + r'=[^\s]+',
                                  key + '=' + str(val), lines[i])
            assert n == 1, key
    return '\n'.join(lines) + '\n'


cases = [
    ('missing producer', 1024, 'handoff-producer', 'ticket', 1, None, 'delete', 'producer: expected'),
    ('wrong producer ticket', 1024, 'handoff-producer', 'ticket', 1, {'ticket': 999}, None, 'producer: ticket bijection'),
    ('missing publish', 1024, 'handoff-publish', 'sent', 1, None, 'delete', 'publish: expected'),
    ('duplicate publish', 1024, 'handoff-publish', 'sent', 1, None, 'duplicate', 'publish: duplicate'),
    ('wrong published stamp', 1024, 'handoff-publish', 'sent', 1, {'saved': 999}, None, 'producer stamp mismatch'),
    ('missing requested state', 1024, 'handoff-state', 'request_ticket', 1, None, 'delete', 'requested state: expected'),
    ('wrong saved ticket', 1024, 'handoff-state', 'request_ticket', 1, {'saved_ticket': 999}, None, 'consumer stamp mismatch'),
    ('wrong scratch owner', 1024, 'handoff-state', 'request_ticket', 1, {'owner': 'DEADBEEF'}, None, 'scratch owner changed'),
    ('changed scratch epoch', 1024, 'handoff-state', 'request_ticket', 1, {'epoch': 999}, None, 'scratch epoch changed'),
    ('incomplete stamp mask', 1024, 'handoff-state', 'request_ticket', 1, {'match': 15}, None, 'incomplete DSP stamp'),
    ('missing consumer', 1024, 'handoff', 'ticket', 1, None, 'delete', 'consumer: expected'),
    ('duplicate consumer', 1024, 'handoff', 'ticket', 1, None, 'duplicate', 'consumer: duplicate'),
    ('used0 without growth', 1024, 'handoff', 'ticket', 1, {'used': 0, 'fallback': 1}, None, 'fallback without scratch growth'),
    ('growth inconsistent with original sizing', 2048, 'handoff', 'ticket', 1, {'cap_after': 55976832}, None, 'differs from original native-LFM sizing'),
    ('used1 during growth', 2048, 'handoff', 'ticket', 1, {'used': 1, 'fallback': 0}, None, 'reuse crossed an allocation growth'),
    ('wrong extent', 1024, 'handoff-producer', 'ticket', 1, {'bytes': 8388607}, None, 'producer shape/extent mismatch'),
    ('uncached producer', 1024, 'handoff-producer', 'ticket', 1, {'cached': 0}, None, 'producer did not retain input'),
    ('failed RPC', 1024, 'handoff', 'ticket', 1, {'rc': 1}, None, 'request/stamp/RPC failure'),
]


def reject(name, text, expected):
    try:
        parser.validate(parser.parse_text(text), capacity)
    except parser.InvalidTrace as exc:
        assert expected in str(exc), (name, str(exc), expected)
        results.append({'name': name, 'result': 'EXPECTED_REJECTION', 'reason': str(exc)})
    else:
        raise AssertionError('invalid trace accepted: ' + name)


for name, M, tag, field, value, changes, action, expected in cases:
    reject(name, edit(texts[M], tag, field, value, changes, action), expected)
text = texts[1024]
for tag, field in [('handoff', 'ticket'), ('handoff-producer', 'ticket'),
                    ('handoff-state', 'request_ticket'), ('handoff-publish', 'sent')]:
    text = edit(text, tag, field, 38, action='delete')
reject('missing complete lifecycle', text, 'consumer: expected')
lines = texts[1024].splitlines()
zero = next(line for line in lines if target(line, 'handoff-state', 'request_ticket', 0))
i = next(i for i, line in enumerate(lines) if target(line, 'handoff-state', 'request_ticket', 1))
lines.insert(i, zero)
reject('unticketed call inside live stamp', '\n'.join(lines), 'unticketed call interposed')
lines = texts[1024].splitlines()
a1 = next(i for i, line in enumerate(lines) if target(line, 'handoff-state', 'request_ticket', 1))
b1 = next(i for i, line in enumerate(lines) if target(line, 'handoff', 'ticket', 1))
lines[a1], lines[b1] = lines[b1], lines[a1]
reject('consume before state', '\n'.join(lines), 'invalid lifecycle order')
require_count = len(cases) + 3
assert sum(x['result'] == 'EXPECTED_REJECTION' for x in results) == require_count
report = {'positive_cases': 2, 'expected_rejections': require_count,
          'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                     [a.parser, a.capacity_table, a.model1024, a.model2048]},
          'results': results}
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(f'PARSER 2 actual traces PASS; {require_count} intended failures rejected')
