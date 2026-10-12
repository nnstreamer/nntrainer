#!/usr/bin/env python3
# @brief Evaluate NNTR_MOE_LOOKAHEAD_DUMP snapshots, never model output substitutes.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate NNTR_MOE_LOOKAHEAD_DUMP snapshots, never model output substitutes.

Requires numpy. Capture FP32 LFM2 decode with NNTR_MOE_TRACE as routing.txt
in the snapshot directory. Supply the checkpoint config.json. Every prediction
uses the TARGET layer's gate and sigmoid+bias ranking. Reconstructed true
router inputs must reproduce at least 99.9% of routed expert IDs.

Cache replay has zero I/O latency: its demand-hit rates are optimistic, not
measured ready-in-time hits or speedups. The shared capacity includes prefetched
experts. Wrong predictions can evict useful experts; traffic counts include
both speculative reads and demand misses. Skip initial decode tokens to reduce
sensitivity to the simplified prefill (demand-only) cache initialization.
"""
import argparse
from collections import OrderedDict
import gzip
import json
from pathlib import Path
import numpy as np
from moe_expert_cache_sim import parse


def replay(calls, predictions, capacity, topn, skip_tokens=8):
    """Immediate-completion replay; predict after previous MoE demand+refresh.

For same-block predictors the prediction is just before that block's demand;
there is no intervening expert demand. Previous-router predictions are also
scheduled after the previous demand, protecting its experts until completion.
Existing speculative candidates are not touched, matching makeRoom semantics.
"""
    cache = OrderedDict()
    result = dict(demand=0, misses=0, prefetch_reads=0, useful_prefetch=0,
                  unused_prefetch_evictions=0)
    decode_i = 0
    layers = len({c[0] for c in calls})

    def insert(keys, prefetched, count):
        if len(keys) > capacity:
            raise ValueError('One call exceeds cache capacity; split prefill '
                             'is not modeled by this evaluator')
        pinned = set(keys)
        missing = [k for k in keys if k not in cache]
        if not prefetched:
            for key in keys:
                if key in cache:
                    cache.move_to_end(key)
        while len(cache) + len(missing) > capacity:
            victim = next(k for k in cache if k not in pinned)
            if cache.pop(victim) and count:
                result['unused_prefetch_evictions'] += 1
        for key in missing:
            cache[key] = prefetched
        return len(missing)

    for i, (layer, tokens, routed, ext) in enumerate(calls):
        count = tokens == 1 and decode_i // layers >= skip_tokens
        if tokens == 1:
            pred = predictions.get(i, [])[:topn]
            reads = insert([(layer, int(e)) for e in pred], True, count)
            if count:
                result['prefetch_reads'] += reads
        keys = [(layer, e) for e in routed]
        if count:
            result['demand'] += len(keys)
            result['useful_prefetch'] += sum(bool(cache.get(k)) for k in keys)
        misses = insert(keys, False, count)
        if count:
            result['misses'] += misses
        for key in keys:
            cache[key] = False
        for e in reversed(ext):
            if (layer, e) in cache:
                cache.move_to_end((layer, e))
        decode_i += tokens == 1
    if not result['demand']:
        raise ValueError('No measured decode demand remains after warmup')
    result['hit_pct'] = 100 * (1 - result['misses'] / result['demand'])
    result['total_reads'] = result['misses'] + result['prefetch_reads']
    return result


def evaluate(directory, config, caches, topns, skip, replay_output=None):
    hidden, experts = config['hidden_size'], config['num_experts']
    dense = config['num_dense_layers']
    eps = config['norm_eps']
    calls = parse((directory / 'routing.txt').read_text().splitlines())
    steps = sorted(int(p.stem.split('_')[-2]) for p in
                   directory.glob(f'layer{dense}_ffn_norm_*_input.f32'))
    layers = sorted({c[0] for c in calls})
    assert len(layers) == config['num_hidden_layers'] - dense
    assert sum(c[1] == 1 for c in calls) == len(steps) * len(layers)
    def read(name):
        a = np.fromfile(directory / (name + '.f32'), dtype='<f4')
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite tensor: {name}')
        return a
    def raw(name, step):
        key = name + '_' + str(step)
        x = read(key + '_input')
        if (directory / (key + '_addend.f32')).exists():
            x = x + read(key + '_addend')
        return x
    def norm(x, gamma):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma
    rankings = {name: {} for name in ['previous_router', 'block_input',
                'block_norm_input', 'branch_output', 'previous_token', 'oracle']}
    matched = total = 0
    per_layer = []
    for layer in layers:
        physical = layer + dense
        prefix = f'layer{physical}'
        weight = read(prefix + '_ffn_down_gate').reshape(hidden, experts)
        bias = read(prefix + '_ffn_down_bias')
        gamma = read(prefix + '_ffn_norm_gamma')
        x = np.stack([raw(prefix + '_ffn_norm', s) for s in steps])
        x = norm(x, gamma)
        block_kind = 'conv' if config['layer_types'][physical] == 'conv' else 'attention'
        blockname = prefix + ('_conv_norm' if block_kind == 'conv' else '_attention_norm')
        # Transformer names the attention's pre-normalization layer _pre_attention_norm.
        if not (directory / (blockname + '_gamma.f32')).exists():
            blockname = prefix + '_pre_attention_norm'
        blockraw = np.stack([raw(blockname, s) for s in steps])
        vectors = {'oracle': x, 'block_input': norm(blockraw, gamma),
                   'block_norm_input': norm(blockraw, read(blockname + '_gamma')),
                   'branch_output': norm(np.stack([raw(prefix + '_ffn_norm', s)
                                                   for s in steps]) - blockraw, gamma)}
        if layer > 0:
            prev = f'layer{physical - 1}_ffn_norm'
            vectors['previous_router'] = norm(np.stack([raw(prev, s) for s in steps]),
                                              read(prev + '_gamma'))
        indices = [i for i, c in enumerate(calls) if c[0] == layer and c[1] == 1]
        actual = np.array([calls[i][2] for i in indices])
        row = {'layer': physical, 'kind': block_kind, 'recall': {}}
        for name, v in vectors.items():
            logits = v @ weight
            scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80))) + bias
            ranking = np.argsort(-scores, axis=-1, kind='stable')
            for i, rank in zip(indices, ranking):
                rankings[name][i] = rank.tolist()
            row['recall'][name] = {}
            for n in topns:
                correct = sum(len(set(a) & set(p[:n])) for a, p in zip(actual, ranking))
                row['recall'][name][n] = correct / actual.size
            if name == 'oracle':
                matched += sum(len(set(a) & set(p[:actual.shape[1]]))
                               for a, p in zip(actual, ranking))
                total += actual.size
        for j, i in enumerate(indices):
            if j:
                rankings['previous_token'][i] = calls[indices[j-1]][3]
        per_layer.append(row)
    agreement = matched / total
    if agreement < .999:
        raise ValueError(f'Reconstructed actual route agreement only {agreement:.6f}')
    # Use the real route (not floating-point recomputation) as the oracle.
    rankings['oracle'] = {i: c[3] for i,c in enumerate(calls) if c[1] == 1}
    if replay_output:
        with gzip.open(replay_output, 'wt') as stream:
            json.dump({'calls': calls, 'predictions': rankings}, stream)
    summary = {}
    for name, pred in rankings.items():
        summary[name] = {}
        for n in topns:
            eligible = [i for i in pred if calls[i][1] == 1]
            hit = sum(len(set(calls[i][2]) & set(pred[i][:n])) for i in eligible)
            demand = sum(len(calls[i][2]) for i in eligible)
            summary[name][n] = {'recall_pct': 100 * hit / demand,
                                'eligible_calls': len(eligible)}
    rows = []
    for c in caches:
        rows.append({'cache': c, 'method': 'baseline', 'topn': 0,
                     **replay(calls, {}, c * len(layers), 0, skip)})
        for name, pred in rankings.items():
            for n in topns:
                if name in ('oracle', 'previous_token') and n > 9:
                    continue
                rows.append({'cache': c, 'method': name, 'topn': n,
                             **replay(calls, pred, c * len(layers), n, skip)})
    return {'steps': len(steps), 'moe_layers': len(layers), 'skip_tokens': skip,
            'actual_route_agreement': agreement, 'prediction': summary,
            'per_layer': per_layer, 'cache_replay': rows,
            'limitation': 'Zero I/O latency replay, not measured runtime hit rate. '
                          'Prefill demand-only initialization; warmup excluded. '
                          'Branch output is available after attention/conv, not before.'}


def selftest():
    calls = [(0, 1, [0], [0]), (0, 1, [1], [1]), (0, 1, [0], [0])]
    base = replay(calls, {}, 1, 0, 0)
    oracle = replay(calls, {i:c[2] for i,c in enumerate(calls)}, 1, 1, 0)
    assert base['misses'] == 3 and oracle['misses'] == 0
    assert base['total_reads'] == oracle['total_reads'] == 3
    pollution = replay(calls, {2: [2]}, 2, 1, 0)
    assert replay(calls, {}, 2, 0, 0)['misses'] == 2
    assert pollution['misses'] == 3 and pollution['total_reads'] == 4
    print('lookahead replay selftest passed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--cache', nargs='+', type=int, default=[4, 8, 16, 24])
    p.add_argument('--topn', nargs='+', type=int, default=[4, 6, 8, 12])
    p.add_argument('--skip-tokens', type=int, default=8)
    p.add_argument('--replay-output', type=Path,
                   help='Save calls and rankings as gzip JSON for independent replay')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.directory or not a.config or not a.output:
        p.error('directory, --config and --output are required')
    if min(a.cache + a.topn) <= 0 or a.skip_tokens < 0:
        p.error('cache/topn must be positive and skip-tokens nonnegative')
    result = evaluate(a.directory, json.loads(a.config.read_text()),
                      a.cache, a.topn, a.skip_tokens, a.replay_output)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_layer', 'cache_replay')}, indent=2))
    for row in result['cache_replay']:
        if row['cache'] == 8:
            print(row)


if __name__ == '__main__':
    main()
