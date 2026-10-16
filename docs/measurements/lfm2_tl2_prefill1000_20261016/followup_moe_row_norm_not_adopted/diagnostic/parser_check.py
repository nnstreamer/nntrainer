#!/usr/bin/env python3
import json
from parse import parse

valid = rejected = 0
for rows in (1024, 2048):
    for enabled in (0, 1):
        lines = [f'[LFM-ROW-NORM] q={q} m={rows} k=2048 e=64 used={enabled} '
                 f'jobs={rows // 64 if enabled else 0} !' for q in range(1, 39)]
        assert parse('\n'.join(lines), rows, enabled)['used'] == 38 * enabled
        valid += 1
        for mutant in (lines[:-1], lines + [lines[-1]], lines[::-1],
                       [lines[0]] + lines[:-1],
                       [lines[0].replace('k=2048', 'k=2816')] + lines[1:],
                       [lines[0].replace('q=1 ', 'q=4294967296 ')] + lines[1:],
                       [lines[0].replace('used='+str(enabled), 'used='+str(1-enabled))] + lines[1:],
                       [lines[0][:-1]] + lines[1:]):
            try:
                parse('\n'.join(mutant), rows, enabled)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Accepted malformed dispatch capture')
print(json.dumps({'valid': valid, 'rejected': rejected, 'status': 'PASS'}))
