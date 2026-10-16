#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import parse_affine_fc as p

records = []
for (m, n), count in p.EXPECTED.items():
    widths = {6144: (2048, 2048, 2048), 3072: (2048, 512, 512),
              2048: (2048, 0, 0), 512: (512, 0, 0)}[n]
    nh = sum(x != 0 for x in widths)
    reads = ((m + 63) // 64) * n // 32
    joins = 1 + ((m + 63) // 64) * sum((x // 32 + 31) // 32 for x in widths)
    row = dict(v=1, M=m, K=2048, N=n, nh=nh, widths=','.join(map(str, widths)),
               rc=0, stride=32, bank_tiles=32, pipeline=1, reads=reads,
               joins=joins, brackets=reads + joins + nh + 4, hz=19200000,
               body_t=50000, pack_t=1000, sums_t=500, read_t=20000, join_t=500,
               drain_t=800, clock_pairs=32, clock_t=64)
    records += [copy.deepcopy(row) for _ in range(count)]

def text(rows):
    return '\n'.join('prefix ' + p.TAG + ' ' + ' '.join(f'{k}={v}' for k, v in r.items()) for r in rows)

valid = text(records)
r = p.parse(valid)
assert r['main']['calls'] == 80 and r['main']['reads'] == 148480
assert r['main']['joins'] == 4880 and r['main']['brackets'] == 153840
assert r['main']['qtimer_reads_including_calibration'] == 312800
mutants = {'missing': text(records[1:]), 'duplicate': valid + '\n' + text(records[:1]),
           'malformed': valid.replace('read_t=20000', 'read_t=garbled', 1),
           'truncated': valid.replace(' clock_pairs=32 clock_t=64', '', 1),
           'unknown': valid.replace('v=1', 'v=1 extra=0', 1),
           'duplicate_key': valid.replace('v=1', 'v=1 v=1', 1),
           'malformed_tag': valid.replace('[LFM-AFFINE-FC]', '[LFM-AFFINE-FC', 1),
           'empty': 'no trace here'}
for field, value in [('rc', 1), ('pipeline', 0), ('stride', 0), ('bank_tiles', 16),
                     ('reads', 1), ('joins', 1), ('brackets', 1), ('body_t', 1),
                     ('widths', '2048,2048,0'), ('hz', 1), ('M', 2048)]:
    changed = copy.deepcopy(records);changed[0][field] = value
    mutants[field] = text(changed)
for name, data in mutants.items():
    try:p.parse(data)
    except ValueError:continue
    raise AssertionError('accepted mutant ' + name)
Path(__file__).with_name('checks').joinpath('parser-result.json').write_text(json.dumps({'valid_main':r['main'],'rejected_mutants':sorted(mutants)},indent=2)+'\n')
print(f'parser PASS: exact180-record cohort; {len(mutants)} malformed/missing/layout/count mutants rejected')
