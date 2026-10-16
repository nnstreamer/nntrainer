#!/usr/bin/env python3
"""Independent raw-field readback; does not import either AFC2 parser."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

D = Path(__file__).resolve().parents[1]
E = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000/docs/measurements/lfm2_tl2_prefill1000_20261016')
MAIN = {(1728, 6144): 30, (192, 6144): 30, (128, 6144): 30,
        (1920, 2048): 40, (128, 2048): 40, (1728, 3072): 10, (320, 3072): 10}
WARM = {(512, 6144): 1, (64, 6144): 29, (64, 2048): 50, (64, 512): 20}
PHASES = ['body_ms', 'pack_ms', 'sums_ms', 'read_ms', 'join_ms', 'drain_ms']

def readback(variant, fast):
    path = E / f'continue-fc-phase2048-{variant}.affine.logcat.txt'
    grouped = {}; payload_lengths = []; pids = set(); lines = 0
    for line in path.read_text().splitlines():
        if 'LFM-AFC2' not in line:
            continue
        lines += 1
        prefix, raw = line.split('[LFM-AFC2', 1)
        pids.add(int(prefix.split()[2]))
        kind, payload = raw.split('] ', 1)
        assert kind in ('G', 'T', 'B') and payload.endswith(' !')
        payload_lengths.append(len(('[LFM-AFC2' + raw).encode('ascii')))
        words = payload[:-2].split()
        fields = dict(w.split('=') for w in words)
        assert len(fields) == len(words)
        ident = int(fields.pop('s'), 16)
        parts = grouped.setdefault(ident, {})
        assert kind not in parts, 'This readback requires the observed zero-duplicate cohort.'
        parts[kind] = fields
    assert len(pids) == 1 and lines == 870 and len(grouped) == 290
    count = Counter(); main = []; warm = []; previous_end = 0; cohorts = []
    for ident, parts in sorted(grouped.items()):
        assert set(parts) == {'G','T','B'}
        g, t, b = (parts[k] for k in ('G','T','B'))
        assert set(g) == {'d','x','c'} and set(t) == {'t','k'} and set(b) == {'f','a'}
        d = list(map(int,g['d'].split(','))); x = list(map(int,g['x'].split(','))); c = list(map(int,g['c'].split(',')))
        timers = [int(v,16) for v in t['t'].split(',')]
        hz, pairs, empty = t['k'].split(',')
        branches = list(map(int,b['f'].split(',')))
        m,k,n,nh,*widths = d
        assert k == 2048 and n == sum(widths) and int(b['a']) == 8372224
        assert (nh,widths) in [(1,[2048,0,0]),(1,[512,0,0]),(3,[2048,2048,2048]),(3,[2048,512,512])]
        blocks = (m+63)//64
        bank = min(32,(8372224-blocks*64*k-k*max(widths))//16384)
        reads = blocks*sum(v//32 for v in widths)
        joins = 1+blocks*sum((v//32+bank-1)//bank for v in widths)
        assert x == [0,32,bank,1]
        assert c == [reads,joins,reads+joins+nh+4]
        assert branches == [fast, reads*fast, reads*(1-fast)]
        assert int(hz) == 19200000 and int(pairs) == 32 and int(empty,16) >= 0
        assert len(timers) == 6 and timers[0] >= sum(timers[1:]) and timers[0] > 0
        assert ident >= previous_end; previous_end = ident+timers[0]
        count[m,n] += 1
        record = {'shape': (m,n), 'ticks': timers, 'reads': reads, 'joins': joins}
        if (m,n) in MAIN: main.append(record); cohorts.append('main')
        elif (m,n) in WARM: warm.append(record); cohorts.append('warm')
        else: raise AssertionError('unsupported shape')
    assert count == Counter(MAIN)+Counter(WARM)
    assert cohorts == ['warm']*100+['main']*190, 'load and main chronology differ'
    phases = {name:sum(r['ticks'][i] for r in main)/19200 for i,name in enumerate(PHASES)}
    phases['other_foreground_ms'] = phases['body_ms']-sum(phases[k] for k in PHASES[1:])
    accepted = json.loads((D / f'{variant}-device.json').read_text())
    for key,value in phases.items(): assert abs(value-accepted['main'][key]) < 1e-9
    assert sum(r['reads'] for r in main) == 296960
    assert sum(r['joins'] for r in main) == 13390
    return {'valid': True, 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'records':len(grouped), 'packets': lines, 'transport_duplicates':0,
            'load_then_main_chronology': [len(warm),len(main)], 'pids':sorted(pids),
            'max_payload_bytes':max(payload_lengths), 'main_phase_ms':phases,
            'independent_of_parser_modules': True}

result = {name:readback(name,fast) for name,fast in [('control',0),('candidate',1)]}
(D/'root/raw-readback.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
