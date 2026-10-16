#!/usr/bin/env python3
"""Strict readback of one completed native LFM 1024/2048 attention stage run.

Read existing files only. Canonical logcat records must belong to the supervised
child. No intra-stream deduplication: these tags have no unique call sequence.
An optional stdout mirror is accepted only as the same entire ordered stream.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PREFIX = re.compile(r'^(\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s+(\d+)\s+(\d+)\s+[A-Z]\s+nntrainer:\s+(.*)$')
WRAPPER = re.compile(
    r'^HTP attn trace q: handle=(\d+) n_q=(\d+) cache=(\d+)\.\.(\d+) '
    r'append=(\d+) err=0x([0-9a-fA-F]+) wall_us=(\d+) dsp_us=(\d+) '
    r'append_us=(\d+) attn_us=(\d+) quant_us=(\d+) stage_us=(\d+) bake_us=(\d+)$')
STAGES = re.compile(
    r'^HTP attn stages q: n_q=(\d+) qprep=(\d+) dma=(\d+) qk=(\d+) '
    r'dequant=(\d+) softmax=(\d+) pquant=(\d+) pv=(\d+) oupd=(\d+) '
    r'store=(\d+) blocks=(\d+) total=(\d+) kcycles=(\d+)$')
WK = ('handle', 'n_q', 'cache_from', 'cache_to', 'append', 'err',
      'wall_us', 'dsp_us', 'append_us', 'attn_us', 'quant_us', 'stage_us', 'bake_us')
SK = ('n_q', 'qprep', 'dma', 'qk', 'dequant', 'softmax', 'pquant', 'pv',
      'oupd', 'store', 'blocks', 'total', 'kcycles')
LAYERS = tuple(f'layer{i}_attention' for i in range(2, 39, 4))


def parse(logcat, stdout, pid, tokens):
    need(type(pid) is int and pid > 0, 'Explicit positive supervised PID required')
    need(tokens in (1024, 2048), 'Only audited native LFM prompt lengths are supported')
    records = []
    payloads = []
    for line_no, line in enumerate(logcat.splitlines(), 1):
        if 'HTP attn trace' not in line and 'HTP attn stages' not in line:
            if ('HTP quantized attention step failed' in line or
                    'attn_q_step: kernel failed' in line or
                    'attn_q_step: append failed' in line):
                raise ValueError(f'Attention error in canonical run at line {line_no}')
            continue
        m = PREFIX.fullmatch(line)
        need(m is not None, f'Malformed/unattributed attention record at line {line_no}')
        timestamp, rp, tid, payload = m.groups()
        need(int(rp) == pid and int(tid) == pid, f'Unexpected attention PID/TID at line {line_no}')
        kind = 'wrapper' if len(records) % 2 == 0 else 'stages'
        pattern, keys = (WRAPPER, WK) if kind == 'wrapper' else (STAGES, SK)
        v = pattern.fullmatch(payload)
        need(v is not None, f'Malformed or out-of-order {kind} at line {line_no}')
        fields = {k: int(x, 16 if k == 'err' else 10) for k, x in zip(keys, v.groups())}
        need(all(0 <= x <= 0xffffffff for k, x in fields.items() if k != 'wall_us'), 'Counter outside uint32 range')
        need(fields['n_q'] == tokens, f'Wrong {kind} prompt length at line {line_no}')
        need(not records or timestamp >= records[-1]['timestamp'], 'Canonical attention timestamps went backwards')
        records.append({'kind': kind, 'line': line_no, 'timestamp': timestamp, **fields})
        payloads.append(payload)
    need(len(records) == 20, f'Expected exactly 10 wrapper/stage pairs; found {len(records)} records')

    mirror = []
    for line_no, line in enumerate(stdout.splitlines(), 1):
        if 'HTP attn trace' in line or 'HTP attn stages' in line:
            # This file is already the single child stdout/stderr stream.
            starts = [i for i in (line.find('HTP attn trace'), line.find('HTP attn stages')) if i >= 0]
            mirror.append(line[min(starts):])
    need(not mirror or mirror == payloads, 'Stdout mirror is not the complete identical canonical ordered stream')

    lp = []
    layer_pattern = re.compile(r'^LAYERPROF\t(\d+)\t(layer\d+_attention)\t(\d+)$')
    for line_no, line in enumerate(stdout.splitlines(), 1):
        if f'LAYERPROF\t{tokens}\t' not in line or not re.search(r'layer\d+_attention(?:\t|$)', line):
            continue
        m = layer_pattern.fullmatch(line)
        need(m is not None, f'Malformed main attention LAYERPROF at line {line_no}')
        span, name, ns = m.groups()
        need(int(span) == tokens and int(ns) > 0, 'Invalid layer profile span/time')
        lp.append({'layer': name, 'layer_profile_ns': int(ns), 'layer_log_line': line_no})
    need(tuple(x['layer'] for x in lp) == LAYERS, 'Expected layer2/6/10/14/18/22/26/30/34/38 once in order')

    output = []
    for i in range(10):
        w, s = records[2 * i:2 * i + 2]
        need(w['cache_from'] == 0 and w['cache_to'] == tokens and w['append'] == tokens, f'Layer {i}: unexpected cache/append geometry')
        need(w['err'] == 0, f'Layer {i}: RPC failed')
        need(w['dsp_us'] == w['append_us'] + w['attn_us'], f'Layer {i}: RPC time identity failed')
        need(w['wall_us'] >= w['dsp_us'] >= w['attn_us'] >= s['total'] > 0, f'Layer {i}: invalid nested time bounds')
        need(s['blocks'] == (160 if tokens == 1024 else 576), f'Layer {i}: unexpected causal block count')
        need(s['kcycles'] > 0 and all(s[k] > 0 for k in SK[1:10]), f'Layer {i}: missing stage counters')
        need(w['quant_us'] <= w['append_us'], f'Layer {i}: quant time exceeds append')
        need(w['timestamp'] <= s['timestamp'], f'Layer {i}: stage logged before wrapper')
        need(lp[i]['layer_profile_ns'] >= w['wall_us'] * 1000, f'Layer {i}: layer time smaller than RPC wrapper')
        output.append({'ordinal': i + 1, **lp[i], 'wrapper': w, 'stages': s,
                       'wall_minus_dsp_us': w['wall_us'] - w['dsp_us'],
                       'attn_minus_inner_us': w['attn_us'] - s['total'],
                       'layer_minus_wrapper_ns': lp[i]['layer_profile_ns'] - w['wall_us'] * 1000})
    return {'pid': pid, 'tokens': tokens, 'calls': 10, 'raw_records': 20,
            'stdout_mirror_records': len(mirror), 'records_deduplicated_within_source': 0,
            'layers': output,
            'totals': {
                'wrapper_us': {k: sum(x['wrapper'][k] for x in output) for k in WK if k.endswith('_us')},
                'stages_us': {k: sum(x['stages'][k] for x in output) for k in SK[1:] if k not in ('blocks', 'kcycles')},
                'blocks': sum(x['stages']['blocks'] for x in output),
                'kcycles': sum(x['stages']['kcycles'] for x in output),
                'layer_profile_ns': sum(x['layer_profile_ns'] for x in output),
                **{k: sum(x[k] for x in output) for k in ('wall_minus_dsp_us', 'attn_minus_inner_us', 'layer_minus_wrapper_ns')}},
            'limitations': [
                'Ten calls are matched to ten serialized model layer records by order; neither attention tag contains a layer ID or unique call sequence.',
                'No ambiguous within-log deduplication is allowed. An optional stdout mirror must match the complete canonical PID stream exactly.',
                'Stage timings include overlap and waiting; do not sum stage fields as disjoint serial cost or subtract a field from full-model wall time.',
                'This is one profile1/attention-trace run per arm and length, not an uninstrumented TPS measurement or causal explanation of normal-run variance.',
                'The normal binary has no raw-plane selection counters. Separate matching diagnostic runs establish selection; these stage rows alone do not.'
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for x in ('logcat', 'log', 'metadata', 'memory', 'ids', 'output'):
        p.add_argument('--' + x, type=Path, required=True)
    p.add_argument('--tokens', type=int, required=True, choices=(1024, 2048))
    a = p.parse_args()
    meta = json.loads(a.metadata.read_text()); mem = json.loads(a.memory.read_text())
    need(meta['profile'] is True and meta['instrumented'] is True and
         meta['performance_comparison_eligible'] is False, 'Expected diagnostic-only metadata')
    need(meta['valid'] is True and meta['ids_match_reference'] is True and
         meta['returncode'] == 0 and meta['concurrent_processes'] == [], 'Invalid run result')
    need(meta['prefill_tokens'] == a.tokens and meta['decode_tokens'] == 64 and meta['moe_cache_experts'] == 16, 'Wrong model/run configuration')
    env = meta['environment']
    for name, value in {'NNTR_HTP_ATTN_TRACE':'1', 'NNTR_HTP_PROFILE':'1', 'NNTR_LAYER_PROFILE':'1',
                        'NNTR_HTP_PAGE_CACHE':'0', 'NNTR_MOE_CACHE_GROW':'0', 'NNTR_MOE_PRELOAD_EXPERTS':'128',
                        'NNTR_NUM_THREADS':'8'}.items():
        need(env.get(name) == value, 'Wrong environment ' + name)
    need(mem['child_waited'] is True and mem['child_exit_code'] == 0 and mem['child_signal'] is None, 'Child did not exit successfully')
    need(mem['mode'] == 'supervised-child' and mem['requested_period_ms'] == 20, 'Wrong memory monitor scope')
    need(sha(a.memory) == meta['full_run_memory_report_sha256'], 'Memory report hash mismatch')
    need(sha(a.ids) == meta['ids_sha256'] and len(a.ids.read_text().splitlines()) == 65, 'Wrong 65 dumped token IDs')
    result = parse(a.logcat.read_text(), a.log.read_text(), mem['pid'], a.tokens)
    result.update({'tag': meta['tag'], 'variant': meta['variant'], 'ids_sha256': meta['ids_sha256'],
                   'inputs': {x: {'file': getattr(a, x).name, 'sha256': sha(getattr(a, x))}
                              for x in ('logcat', 'log', 'metadata', 'memory', 'ids')},
                   'performance_comparison_eligible': False, 'passed': True})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'tag': result['tag'], 'pid': result['pid'], 'calls': 10, 'totals': result['totals'], 'passed': True}))


if __name__ == '__main__':
    main()
