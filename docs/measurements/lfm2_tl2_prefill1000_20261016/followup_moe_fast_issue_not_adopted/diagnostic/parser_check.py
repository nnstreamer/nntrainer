#!/usr/bin/env python3
"""Main1024/2048 cohort, SDK replay, truncation and malformed-packet checks."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location('dispatch_parser', here / 'parse.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def line(r):
    return ('sdk-prefix [LFM-MISSUE] q={seq} s={id:x} d={M},{E} b={selected} '
            't={gu},{dn} f={fast_gu},{fast_dn} r={rc} !').format(**r)


def cohort(M):
    result = []
    for i, rows in enumerate([M] * 39, 1):
        blocks = (99 if M == 1024 else 160) + i
        gu, dn = blocks * 96, blocks * 64
        result.append(dict(seq=i, id=0x1234567890000 + i * 1000, M=rows,
                           E=(1, 4, 60, 64)[i % 4], selected=1,
                           gu=gu, dn=dn, fast_gu=gu, fast_dn=dn, rc=0))
    result[0].update(E=64, gu=6144, dn=4096, fast_gu=6144, fast_dn=4096)
    return result


bad = []
positives = []
truncations = 0
for M in (1024, 2048):
    rows = cohort(M)
    original = list(map(line, rows))
    positives.append(p.parse(original, M))
    replay = original + original[:7] + original
    random.Random(M).shuffle(replay)
    result = p.parse(replay, M)
    assert result['identical_duplicate_packets'] == 46
    positives.append(result)

    def reject(name, lines):
        try:
            p.parse(lines, M)
        except ValueError:
            bad.append(f'M{M}:{name}')
        else:
            raise AssertionError((M, name, lines))

    for i in (0, 1, 3, 19, 38):
        reject(f'missing{i}', original[:i] + original[i + 1:])
    reject('empty', [])
    android = ['10-16 22:10:11.123 1234 1255 V adsprpc : ' + x for x in original]
    positives.append(p.parse(android, M))
    reject('mixed_pid', android[:-1] + [android[-1].replace(' 1234 ', ' 1235 ')])
    reject('mixed_prefix', android[:-1] + [original[-1]])
    reject('extra_unique', original + [line(dict(rows[-1], seq=40))])
    for field, value in [('M', M // 2), ('E', 63), ('selected', 0),
                         ('gu', 6240), ('dn', 4160), ('fast_gu', 0),
                         ('fast_dn', 0), ('rc', 1)]:
        changed = dict(rows[0]); changed[field] = value
        reject('warmup_changed_' + field, [line(changed)] + original[1:])
    reordered = copy.deepcopy(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    for i, r in enumerate(reordered, 1):
        r['seq'] = i; r['id'] = rows[i - 1]['id']
    reject('warmup_not_first', list(map(line, reordered)))
    inserted = copy.deepcopy(rows)
    inserted.insert(1, dict(rows[0]))
    for i, r in enumerate(inserted, 1):
        r['seq'] = i; r['id'] = rows[0]['id'] + i * 1000
    reject('second_warmup', list(map(line, inserted)))
    no_warm = copy.deepcopy(rows[1:])
    for i, r in enumerate(no_warm, 1):
        r['seq'] = i
    reject('missing_warmup_resequenced', list(map(line, no_warm)))
    # An actual main call may have the same dimensions/counts as the warmup;
    # preserve it by logical position, never remove all matching signatures.
    equal_main = copy.deepcopy(rows)
    equal_main[10].update(E=64, gu=6144, dn=4096, fast_gu=6144, fast_dn=4096)
    assert p.parse(list(map(line, equal_main)), M)['main_calls'] == 38
    positives.append(p.parse(list(map(line, equal_main)), M))

    for field in ('id', 'M', 'E', 'selected', 'gu', 'dn', 'fast_gu', 'fast_dn', 'rc'):
        changed = copy.deepcopy(rows[-1])
        changed[field] += 1
        reject('conflicting_replay_' + field, original + [line(changed)])
    mutations = {
        'seq0': ('seq', 0), 'seqmax': ('seq', 2**32 - 1),
        'seqoverflow': ('seq', 2**32), 'unsigned_overflow': ('M', 2**32),
        'tick_zero': ('id', 0), 'tick_collision': ('id', rows[-2]['id']),
        'tick_backwards': ('id', rows[-2]['id'] - 1),
        'expert0': ('E', 0), 'expert65': ('E', 65),
        'selected_invalid': ('selected', 2), 'selected_unknown': ('selected', -1),
        'fallback': ('selected', 0), 'error': ('rc', 1),
        'error_overflow': ('rc', 2**31), 'fast_zero': ('fast_gu', 0),
        'fast_excess': ('fast_gu', rows[-1]['gu'] + 96),
        'gu_not_tiles': ('gu', rows[-1]['gu'] + 1),
        'dn_not_tiles': ('dn', rows[-1]['dn'] + 1),
        'DN_partial': ('fast_dn', rows[-1]['fast_dn'] - 64),
    }
    for name, (field, value) in mutations.items():
        changed = copy.deepcopy(rows[-1]); changed[field] = value
        reject(name, original[:-1] + [line(changed)])
    zero = dict(rows[-1], gu=0, dn=0, fast_gu=0, fast_dn=0)
    reject('zero_tiles', original[:-1] + [line(zero)])
    ratio = dict(rows[-1], gu=96, dn=128, fast_gu=96, fast_dn=128)
    reject('ratio', original[:-1] + [line(ratio)])
    reject('terminal_suffix', original[:-1] + [original[-1] + 'bad'])
    reject('timestamp_overflow', original[:-1] + [line(dict(rows[-1], id=2**64))])
    reject('long_decimal', original[:-1] + [original[-1].replace('q=39 ', 'q=' + '9' * 10000 + ' ')])
    for n in range(len(original[-1])):
        reject(f'truncated{n}', original[:-1] + [original[-1][:n]])
        truncations += 1
result = {'pass': True, 'valid_cohorts_and_replays': len(positives),
          'negative_cases': len(bad), 'all_prefix_truncations': truncations,
          'positive_results': positives, 'rejected': bad}
text = json.dumps(result, indent=2) + '\n'
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text)
print(f'MOE_ISSUE_TRACE parser {len(positives)} valid / {len(bad)} negative '
      f'({truncations} truncations included) PASS')
