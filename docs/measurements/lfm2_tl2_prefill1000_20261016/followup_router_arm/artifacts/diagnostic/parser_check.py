#!/usr/bin/env python3
"""Bounded records, missing/conflicting/partial lines and task error checks."""
import json
from pathlib import Path
import parse as parser


def line(q, m, on):
    return (f'{parser.TAG} q={q} m={m} k=2048 e=64 enter=1 used={on} '
            f'fallback={1-on} tasks={m//4 if on else 0} '
            f'done={m//4 if on else 0} rejected=0 finish=1 !')


passed = failed = 0
for rows in (1024, 2048):
    for enabled in (0, 1):
        lines = [line(q, rows, enabled) for q in range(1, 39)]
        good = '\n'.join(lines)
        parser.parse(good, rows, enabled, 38 * enabled)
        passed += 1
        bad = [lines[1:], lines[:-1], lines + lines[:1], lines[::-1],
               [lines[0], lines[0]] + lines[2:]]
        for old, new in [('q=1 ', 'q=0 '), ('q=1 ', 'q=4294967295 '),
                         ('q=1 ', 'q=4294967296 '), ('q=1 ', 'q=99999999999 '),
                         ('m='+str(rows), 'm=512'), ('k=2048', 'k=2816'),
                         ('e=64', 'e=128'), ('enter=1', 'enter=0'),
                         ('finish=1', 'finish=0'), ('used='+str(enabled), 'used=2'),
                         ('fallback='+str(1-enabled), 'fallback=2'),
                         ('done='+str(rows//4 if enabled else 0), 'done=1'),
                         ('tasks='+str(rows//4 if enabled else 0), 'tasks=1'),
                         ('rejected=0', 'rejected=513')]:
            bad.append([lines[0].replace(old, new)] + lines[1:])
        # A complete terminal number with no terminal marker must still fail.
        bad += [[lines[0][:-2]] + lines[1:], [lines[0] + ' junk'] + lines[1:]]
        for b in bad:
            try:
                parser.parse('\n'.join(b), rows, enabled, 38 * enabled)
            except ValueError:
                failed += 1
            else:
                raise AssertionError('invalid record accepted')
        # Every prefix containing a tag is rejected, not accidentally accepted
        # as a smaller valid number or ignored as unrelated output.
        for end in range(len(parser.TAG), len(lines[0])):
            try:
                parser.parse('\n'.join([lines[0][:end]] + lines[1:]), rows, enabled)
            except ValueError:
                failed += 1
            else:
                raise AssertionError('truncated marker accepted')
    # Legitimate complete DSP fallbacks are reported separately, never fast.
    lines = [line(q, rows, 1) for q in range(1, 39)]
    lines[0] = line(1, rows, 0)  # metadata or weight refusal before dispatch
    lines[1] = lines[1].replace('used=1', 'used=0').replace('fallback=0', 'fallback=1').replace('rejected=0', 'rejected=1')
    result = parser.parse('\n'.join(lines), rows, 1, 36)
    assert result['fallback'] == 2 and result['rejected'] == 1
    passed += 1
    try:
        parser.parse('\n'.join(lines), rows, 1, 38)
    except ValueError:
        failed += 1
    else:
        raise AssertionError('required full ARM dispatch not enforced')

result = {'valid': passed, 'negative': failed, 'pass': True}
Path(__file__).with_name('parser-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
