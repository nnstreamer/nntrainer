#!/usr/bin/env python3
"""Strict AFC2 G/T/B validation for the source-derived 1024/2048 LFM topology."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

P = Path(__file__).with_name('afc2_base.py')
spec = importlib.util.spec_from_file_location('afc2_base', P)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ARENA = (8 << 20) - 0x4000
K = 2048


def chunks(rows, step):
    return [min(step, rows - m) for m in range(0, rows, step)]


def expected_topology(prompt):
    if prompt not in (1024, 2048):
        raise ValueError('only source-audited 1024/2048 prompts are supported')
    arm_step = (((8 << 20) - (256 << 10) - K * 2048) // K) // 64 * 64
    dsp_step = ((ARENA - K * 2048 - (512 << 10)) // K) // 64 * 64
    assert arm_step == 1920 and dsp_step == 1728
    main = Counter()
    for m in chunks(prompt, arm_step):
        main[(m, 2048)] += 40
        for part in chunks(m, dsp_step):
            main[(part, 6144)] += 30
    for m in chunks(prompt, dsp_step):
        main[(m, 3072)] += 10
    warm = Counter({(512, 6144): 1, (64, 6144): 29,
                    (64, 2048): 50, (64, 512): 20})
    return main, warm


def packet(line, lineno):
    if '[LFM-AFC2B]' not in line:
        return base.packet(line, lineno)
    if line.count('LFM-AFC2') != 1:
        raise ValueError(f'line {lineno}: repeated branch tag')
    payload = line[line.index('[LFM-AFC2B]'):].strip()
    match = re.fullmatch(r'\[LFM-AFC2B\] s=([0-9a-f]+) f=(\d+),(\d+),(\d+) a=(\d+) !', payload)
    if not match or len(payload.encode('ascii')) > 192:
        raise ValueError(f'line {lineno}: malformed/truncated branch packet')
    ident = base.number(match.group(1), 16, 64)
    vals = [base.number(v, 10, 32) for v in match.groups()[1:]]
    return 'B', ident, dict(zip(('eligible', 'fast_calls', 'generic_calls', 'arena'), vals))


def validate(row, ident, expected_fast):
    widths = row['widths']
    if row['K'] != K or row['M'] <= 1 or row['hz'] != 19200000 or row['clock_pairs'] != 32:
        raise ValueError(f'ID {ident:x}: unsupported shape/clock')
    if not ((row['nh'] == 1 and widths in ((512, 0, 0), (2048, 0, 0))) or
            (row['nh'] == 3 and widths in ((2048, 2048, 2048), (2048, 512, 512)))):
        raise ValueError(f'ID {ident:x}: unsupported handles')
    if row['N'] != sum(widths) or row['arena'] != ARENA:
        raise ValueError(f'ID {ident:x}: width or actual VTCM arena differs from audited topology')
    blocks = (row['M'] + 63) // 64
    result_off = blocks * 64 * K + K * max(widths)
    bank = min(32, (ARENA - result_off) // (2 * 8192))
    if bank < 1 or row['rc'] or row['pipeline'] != 1 or row['stride'] != 32 or row['bank_tiles'] != bank:
        raise ValueError(f'ID {ident:x}: error/fallback/unexpected accumulator layout')
    reads = blocks * sum(n // 32 for n in widths)
    joins = 1 + blocks * sum((n // 32 + bank - 1) // bank for n in widths)
    brackets = reads + joins + row['nh'] + 4
    if tuple(row[k] for k in ('reads', 'joins', 'brackets')) != (reads, joins, brackets):
        raise ValueError(f'ID {ident:x}: missing or extra phase bracket/tile')
    expected_branch = (expected_fast, reads if expected_fast else 0, 0 if expected_fast else reads)
    if tuple(row[k] for k in ('eligible', 'fast_calls', 'generic_calls')) != expected_branch:
        raise ValueError(f'ID {ident:x}: actual selected branch/leaf count does not match requested variant')
    if row['body_t'] <= 0 or sum(row[k] for k in base.PHASES) > row['body_t']:
        raise ValueError(f'ID {ident:x}: invalid/overlapping foreground intervals')
    return dict(row, start_tick=ident)


def parse(text, prompt, expected_fast):
    pairs = {}; raw = 0; duplicated = 0; pids = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        if 'LFM-AFFINE-FC' in line:
            raise ValueError('v1 trace cannot be mixed with AFC2')
        if 'LFM-AFC2' not in line:
            continue
        prefix = re.match(r'^\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+(\d+)\s+\d+\s+\S\s+', line)
        if prefix:
            pids.add(int(prefix.group(1)))
        kind, ident, row = packet(line, lineno)
        raw += 1
        parts = pairs.setdefault(ident, {})
        if kind in parts:
            if parts[kind] != row:
                raise ValueError(f'ID {ident:x}: conflicting duplicate {kind}')
            duplicated += 1
        else:
            parts[kind] = row
    if not pairs or len(pids) > 1:
        raise ValueError('empty or mixed-process cohort')
    if any(set(parts) != {'G', 'T', 'B'} for parts in pairs.values()):
        raise ValueError('missing geometry/timing/branch counterpart')
    rows = [validate(dict(parts['G'], **parts['T'], **parts['B']), ident, expected_fast)
            for ident, parts in sorted(pairs.items())]
    for left, right in zip(rows, rows[1:]):
        if left['start_tick'] + left['body_t'] > right['start_tick']:
            raise ValueError('overlapping serialized call bodies')
    main_expected, warm_expected = expected_topology(prompt)
    counts = Counter((r['M'], r['N']) for r in rows)
    if counts != main_expected + warm_expected:
        raise ValueError('physical-kernel cohort differs from source-derived ARM/DSP chunking: ' + str(dict(counts)))
    main = [r for r in rows if (r['M'], r['N']) in main_expected]
    warm = [r for r in rows if (r['M'], r['N']) in warm_expected]
    assert len(main) + len(warm) == len(rows)
    out = {'valid': True, 'version': 'AFC2 plus actual branch/arena', 'prompt_rows': prompt,
           'expected_fast': expected_fast, 'actual_arena_bytes': ARENA,
           'records': len(rows), 'unique_packets': 3*len(rows), 'raw_packets': raw,
           'transport_duplicate_packets': duplicated, 'pids': sorted(pids),
           'groups': {}, 'main': base.summarize(main), 'load_warmup': base.summarize(warm),
           'main_calls_by_shape': {f'M{m}_N{n}': count for (m,n),count in sorted(main_expected.items())},
           'main_branch': {k:sum(r[k] for r in main) for k in ('fast_calls','generic_calls')},
           'performance_comparison_eligible': False}
    for key in sorted(counts):
        out['groups'][f'M{key[0]}_N{key[1]}'] = base.summarize([r for r in rows if (r['M'],r['N']) == key])
    if out['main']['reads'] != 148480 * (prompt // 1024):
        raise ValueError('main tile total differs from logical FC work')
    out['limits'] = ['Only one fresh-process LFM2 run; model IDs/settings checked separately.',
                    'Main kernel call counts include nested ARM and DSP row chunking; load warmups separate.',
                    'read includes HMX completion; join/drain are exposed foreground waits, not worker sums.',
                    'Timer placement is AFCv2; added branch counters execute once per issued tile on both variants.',
                    'All three logs occur after body measurement; trace TPS is excluded.',
                    'Identical start-ID/part duplicates are reported; conflicting/incomplete parts reject.']
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--prompt',type=int,choices=[1024,2048],required=True)
    p.add_argument('--expected-fast',type=int,choices=[0,1],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.log.read_bytes()
    try: result=parse(raw.decode(),a.prompt,a.expected_fast)
    except (ValueError,UnicodeError) as e: p.exit(1,'INVALID TRACE: '+str(e)+'\n')
    result['source_log_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
