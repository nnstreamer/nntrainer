#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse phase-isolated prefill logs without adding nested profile counters."""
import argparse
import collections
import json
from pathlib import Path
import re


def parse(directory, name):
    raw = (directory / (name + '.log')).read_text()
    nodes = []
    for order, label, kind, calls, us in re.findall(
        r'\[OP-TIME\] node f=(\d+) name=(\S+) type=(\S+) calls=(\d+) sum_us=([\d.]+)', raw):
        assert int(calls) == 1
        nodes.append(dict(order=int(order), name=label, type=kind, ms=float(us) / 1000))
    assert len(nodes) == 205
    groups = collections.defaultdict(float)
    counts = collections.Counter()
    for row in nodes:
        groups[row['type']] += row['ms']
        counts[row['type']] += 1
    assert counts['lfm2_moe'] == 38 and counts['mha_core'] == 10 and counts['conv_block'] == 30
    prefill = int(re.search(r'^prefill: 1024 tokens, (\d+) ms', raw, re.M)[1])
    # Prefill is printed in whole milliseconds; node counters retain us.
    # Truncation can put the printed total slightly below the node sum.
    assert -1 < prefill - sum(groups.values()) < 20
    m0 = []
    for line in raw.splitlines():
        if line.startswith('[M0-PROF]'):
            fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
            fields['index'] = int(re.search(r'moe_layer\[(\d+)\]', line)[1])
            m0.append(fields)
    assert len(m0) == 38
    buckets = []
    for line in raw.splitlines():
        m = re.search(r'K=(\d+)\s+N=(\d+)\s+(M>1|M==1)\s*(.*?)\s*calls=(\d+)\s+rows=(\d+)\s+host=\s*([\d.]+) ms.*?dsp=\s*([\d.]+) us/call.*?transport=\s*([\d.]+) us/call', line)
        if not m:
            continue
        k, n, shape, kind, calls, rows, host, dsp, transport = m.groups()
        stages = {key: float(value) for key, value in re.findall(r'(quant|gather|requant|dequant|acc|push|scatter|alloc|stage|mm) ([\d.]+)', line)}
        drain = re.search(r'drain ([\d.]+)\+([\d.]+)', line)
        stages.update(drain=float(drain[1]), drain_dn=float(drain[2]))
        stages['rest'] = float(re.search(r'rest<=([\d.]+)', line)[1])
        buckets.append(dict(k=int(k), n=int(n), shape=shape, kind=kind.strip() or 'MoE', calls=int(calls), rows=int(rows), host_ms=float(host), dsp_us_per_call=float(dsp), transport_us_per_call=float(transport), stages_us_per_call=stages, raw=line))
    trace = (directory / (name + '-logcat.txt')).read_text()
    phases, outer = [], []
    for line in trace.splitlines():
        if 'n_q=1024 ' not in line:
            continue
        fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
        if 'HTP attn phases q8wire:' in line:
            phases.append(fields)
        if 'HTP attn trace q8wire:' in line:
            outer.append(fields)
            assert 'err=0x0' in line
    assert len(phases) == len(outer) == 10

    def sums(rows):
        return {key: sum(r.get(key, 0) for r in rows) for key in rows[0] if key not in ('n_q', 'handle', 'cache', 'err')}

    math = []
    for line in raw.splitlines():
        if line.startswith('[HTP-MATH]'):
            fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
            fields['op'] = re.search(r'op=(\S+)', line)[1]
            math.append(fields)
    pf = re.search(r'expert prefetch: (\d+) experts read under the layer calls, exposed wait ([\d.]+) ms.*?register rpc ([\d.]+) ms', raw)
    ready = re.search(r'expert prefetch progress: (\d+) of (\d+) read', raw)
    return dict(name=name, prefill_ms=prefill, node_total_ms=sum(groups.values()), unassigned_ms=prefill-sum(groups.values()), groups_ms=dict(groups), counts=dict(counts), nodes=nodes, m0=m0, m0_sum=sums(m0), htp_buckets=buckets, attention_phases=sums(phases), attention_outer=sums(outer), attention_calls=outer, math=math, prefetch=dict(experts=int(pf[1]), exposed_wait_ms=float(pf[2]), register_ms=float(pf[3]), ready=int(ready[1])), evidence_scope='Node/M0/attention rows isolate measured prefill; HTP buckets include load-time warmups and nested fused calls. Do not sum all HTP rows or interpret missing DSP samples as transport.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = {name: parse(args.directory, name) for name in ('profile-1', 'profile-2')}
    (args.directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, r in result.items():
        print(name, r['prefill_ms'], 'ms', {k: round(v, 2) for k, v in r['groups_ms'].items()})
        print('attention:', r['attention_outer'], r['attention_phases'])
        print('M0:', r['m0_sum'], 'prefetch:', r['prefetch'])


if __name__ == '__main__':
    main()
