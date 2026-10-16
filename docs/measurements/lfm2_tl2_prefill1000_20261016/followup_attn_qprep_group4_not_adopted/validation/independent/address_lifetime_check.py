#!/usr/bin/env python3
"""Enumerate the reviewed geometry and compare unchanged production bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def function(text, name):
    match = re.search(r'^static[^;{]*\b' + re.escape(name) + r'\([^;{]*\{', text, re.M)
    assert match, name
    end, depth = match.end(), 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[match.start():end]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
base = Path('nntrainer/tensor/htp_backend')
records = []
for rel, names in [
    (base/'hmx/hexkl_attn_q.c', ['dequant_s_at', 'dequant_s', 'qx_dequant_available',
                               'dequant_worker', 'phase_qk', 'run_q_block', 'phase_store']),
    (base/'hvx/hvx_quant_u8.c', ['quant_row_params_one', 'quant_params_pack_worker',
                               'pack_group_rows', 'quant_pack_group4']),
]:
    old, new = (a.reference/rel).read_text(), (a.source/rel).read_text()
    for name in names:
        before, after = function(old, name), function(new, name)
        assert before == after, name
        records.append({'name': name, 'source': str(rel),
                        'function_sha256': hashlib.sha256(after.encode()).hexdigest()})

rows_seen = set()
stores = 0
max_end = 0
partitions = 0
for head in range(8):
    for qb0, blocks in [(0, 9), (9, 7)]:
        groups = blocks*128
        chunk_rows = groups*4
        destinations = set()
        for group in range(groups):
            for subhead in range(4):
                local_row = 4*group+subhead
                qi, r = divmod(local_row, 512)
                old_query = (qb0+qi)*128+r//4
                old_head = head*4+r%4
                old_start = old_query*2048+old_head*64
                new_start = qb0*128*2048+head*4*64+group*2048+subhead*64
                assert old_start == new_start
                assert 0 <= new_start and new_start+64 <= 2048*2048
                rows_seen.add((old_query, old_head))
                max_end = max(max_end, 4*(new_start+64))
            row = group*4
            for kt in range(2):
                byte_start = ((row//64)*2+kt)*2048+(row%64)*32
                assert byte_start % 128 == 0
                assert 0 <= byte_start and byte_start+128 <= chunk_rows*64
                assert byte_start not in destinations
                destinations.add(byte_start)
                stores += 1
        assert len(destinations)*128 == chunk_rows*64
        assert sorted(destinations) == list(range(0, chunk_rows*64, 128))
        for workers in range(9):
            threads = workers+1
            ranges = [(groups*i//threads, groups*(i+1)//threads) for i in range(threads)]
            assert ranges[0][0] == 0 and ranges[-1][1] == groups
            assert all(ranges[i][1] == ranges[i+1][0] for i in range(threads-1))
            partitions += 1
assert len(rows_seen) == 2048*32
assert max_end == 16*1024*1024
raw_bytes = 8*8*8192
qx_bytes = 9*512*64*4
assert raw_bytes <= qx_bytes
result = {'scope': 'Source/body equality and enumerated address proof; no HW arithmetic/timing test.',
          'unchanged_functions': records, 'query_head_rows': len(rows_seen),
          'ah_vector_stores': stores, 'pool_partitions': partitions,
          'query_last_byte_exclusive': max_end, 'qx_existing_bytes': qx_bytes,
          'deferred_raw_bank_bytes': raw_bytes,
          'source_sha256': {str(rel): hashlib.sha256((a.source/rel).read_bytes()).hexdigest()
                            for rel in [base/'hmx/hexkl_attn_q.c', base/'hvx/hvx_quant_u8.c']}}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ['query_head_rows', 'ah_vector_stores',
                                       'pool_partitions', 'qx_existing_bytes',
                                       'deferred_raw_bank_bytes']}))
