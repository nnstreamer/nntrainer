#!/usr/bin/env python3
import copy
import json
import random
from pathlib import Path
import parse_affine_fc as p

records = []
for (m, n), count in p.EXPECTED.items():
    widths = {6144: (2048, 2048, 2048), 3072: (2048, 512, 512), 2048: (2048, 0, 0), 512: (512, 0, 0)}[n]
    nh = sum(x != 0 for x in widths)
    reads = ((m + 63) // 64) * n // 32
    joins = 1 + ((m + 63) // 64) * sum((x // 32 + 31) // 32 for x in widths)
    row = dict(M=m, K=2048, N=n, nh=nh, widths=widths, rc=0, stride=32, bank_tiles=32,
               pipeline=1, reads=reads, joins=joins, brackets=reads+joins+nh+4,
               hz=19200000, body_t=50000, pack_t=1000, sums_t=500, read_t=20000,
               join_t=500, drain_t=800, clock_pairs=32, clock_t=64)
    records += [copy.deepcopy(row) for _ in range(count)]


def emit(row, ident):
    d = ','.join(map(str, (row['M'], row['K'], row['N'], row['nh'], *row['widths'])))
    x = ','.join(str(row[f]) for f in ('rc', 'stride', 'bank_tiles', 'pipeline'))
    c = ','.join(str(row[f]) for f in ('reads', 'joins', 'brackets'))
    t = ','.join(f'{row[f]:x}' for f in ('body_t', *p.PHASES))
    return [f'prefix [LFM-AFC2G] s={ident:x} d={d} x={x} c={c} !',
            f'prefix [LFM-AFC2T] s={ident:x} t={t} k={row["hz"]},{row["clock_pairs"]},{row["clock_t"]:x} !']


def lines(rows):
    out = []
    for i, row in enumerate(rows):
        out += emit(row, (1 << 64) - 1 - i)
    return out


def replacement(text, old, new):
    assert old in text
    return text.replace(old, new, 1)


valid_lines = lines(records)
valid = '\n'.join(valid_lines)
summary = p.parse(valid)
assert summary['records'] == 180 and summary['packets'] == 360
assert summary['main']['calls'] == 80 and summary['main']['reads'] == 148480
assert summary['main']['joins'] == 4880 and summary['main']['brackets'] == 153840
assert summary['main']['qtimer_reads_including_calibration'] == 312800
assert p.parse('\n'.join(reversed(valid_lines))) == summary
shuffled = valid_lines[:];random.Random(19).shuffle(shuffled)
assert p.parse('\n'.join(shuffled)) == summary
mutants = {
    'missing_G': '\n'.join(valid_lines[1:]), 'missing_T': '\n'.join(valid_lines[:1]+valid_lines[2:]),
    'missing_record': '\n'.join(valid_lines[2:]), 'duplicate_G': valid+'\n'+valid_lines[0],
    'duplicate_T': valid+'\n'+valid_lines[1], 'duplicate_record': valid+'\n'+'\n'.join(valid_lines[:2]),
    'empty': 'no trace here', 'mixed_v1': valid+'\n[LFM-AFFINE-FC] v=1',
    'bad_tag': replacement(valid,'[LFM-AFC2G]', '[LFM-AFC2X]'),
    'truncated_tag': replacement(valid,'[LFM-AFC2G]', '[LFM-AFC2G'),
    'duplicate_tag': replacement(valid,'[LFM-AFC2G]', '[LFM-AFC2G][LFM-AFC2G]'),
    'duplicate_key': replacement(valid,'s=ffffffffffffffff', 's=ffffffffffffffff s=1'),
    'extra_key': replacement(valid,'s=ffffffffffffffff', 's=ffffffffffffffff extra=1'),
    'id_overflow': replacement(valid,'s=ffffffffffffffff', 's=10000000000000000'),
    'bad_hex_id': replacement(valid,'s=ffffffffffffffff', 's=gggg'),
    'missing_terminal_G': replacement(valid, valid_lines[0], valid_lines[0][:-2]),
    'missing_terminal_T': replacement(valid, valid_lines[1], valid_lines[1][:-2]),
    'truncated_clock_and_terminal': replacement(valid, valid_lines[1], valid_lines[1][:-4]),
    'data_after_terminal': replacement(valid, valid_lines[0], valid_lines[0]+' more'),
    'duplicate_complete_id': replacement(valid, 's=fffffffffffffffe', 's=ffffffffffffffff'),
    'unmatched_timing_id': replacement(valid, valid_lines[1], valid_lines[1].replace('ffffffffffffffff', '1')),
    'wrong_d_count': replacement(valid, 'd=1024,2048,6144,3,2048,2048,2048', 'd=1024,2048,6144,3,2048,2048'),
    'wrong_x_count': replacement(valid, 'x=0,32,32,1', 'x=0,32,32'),
    'wrong_c_count': replacement(valid, 'c=3072,97,3176', 'c=3072,97'),
    'wrong_t_count': replacement(valid, 't=c350,3e8,1f4,4e20,1f4,320', 't=c350,3e8,1f4,4e20,1f4'),
    'wrong_k_count': replacement(valid, 'k=19200000,32,40', 'k=19200000,32,40,0'),
    'negative_ticks': replacement(valid, 't=c350,', 't=-1,'),
    'ticks_overflow': replacement(valid, 't=c350,', 't=10000000000000000,'),
    'unsigned_overflow': replacement(valid, 'd=1024,', 'd=4294967296,'),
    'signed_overflow': replacement(valid, 'x=0,', 'x=2147483648,'),
}
for field,value in [('rc',1),('pipeline',0),('stride',0),('bank_tiles',16),('reads',1),
                    ('joins',1),('brackets',1),('body_t',1),('widths',(2048,2048,0)),
                    ('hz',1),('clock_pairs',31),('M',2048),('K',2816)]:
    changed = copy.deepcopy(records);changed[0][field] = value
    mutants[field] = '\n'.join(lines(changed))
for name, value in mutants.items():
    try:p.parse(value)
    except (ValueError, UnicodeError):continue
    raise AssertionError('accepted mutant '+name)
# A physically truncated final numeric token cannot pass because its mandatory
# terminal marker is also lost. Test EVERY prefix truncation of both packet kinds.
truncations = 0
for index in (0,1):
    line = valid_lines[index]
    for length in range(line.index('[LFM-AFC2'), len(line)):
        malformed = valid_lines[:];malformed[index] = line[:length]
        try:p.parse('\n'.join(malformed))
        except ValueError:truncations += 1;continue
        raise AssertionError(f'accepted prefix truncation kind{index} length{length}')
archived = Path(__file__).parent.parent/'v1-rejected/continue-affine-fc-trace-r1.affine.logcat.txt'
try:p.parse(archived.read_text())
except ValueError:pass
else:raise AssertionError('accepted real truncated v1 evidence')
result = {'valid_records':180,'valid_packets':360,'main':summary['main'],
          'reordered_and_shuffled_pairs':'PASS','rejected_mutants':sorted(mutants),
          'rejected_all_prefix_truncations':truncations,'real_v1_truncated180':'REJECTED; no phase analysis'}
(Path(__file__).parent/'checks/parser-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'v2 parser PASS:180 paired records/360 packets; {len(mutants)} mutants +{truncations} prefix truncations rejected; realv1 rejected')
