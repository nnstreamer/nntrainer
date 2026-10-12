#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Alternate hybrid/strict decode or strict fusion off/on on one device.

The continuation file is consumed by NNTR_PPL_DECODE in both modes. This
fixes token inputs, not expert routing: numerical differences can change it.
The runtime and model must already be deployed. Logs and JSON retain the
actual settings and device binary hashes; no device files are modified.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--continuation', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--comparison', choices=['hybrid', 'fusion'],
                        default='hybrid')
    parser.add_argument('--conv-hvx', choices=['0', '1'], default='1')
    parser.add_argument('--attention-hmx', choices=['0', '1'],
                        help='Override attention dispatch; default follows strict/hybrid')
    parser.add_argument('--norm-fusion', choices=['0', '1'], default='1')
    parser.add_argument('--rope-fusion', choices=['0', '1'], default='1')
    parser.add_argument('--retain-router-input', choices=['0', '1'], default='1')
    parser.add_argument('--strict-spin-us', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not 0 <= args.strict_spin_us <= 1000000:
        parser.error('--strict-spin-us must be in [0, 1000000]')
    args.output.mkdir(parents=True, exist_ok=False)
    adb = ['adb', '-s', args.serial, 'shell']

    def read(command):
        return subprocess.run(adb + [command], text=True, capture_output=True,
                              check=True, timeout=30).stdout.strip()

    runtime = shlex.quote(args.runtime)
    manifest = read(f'cd {runtime} && sha256sum nntrainer_causallm '
                    'libnntrainer.so libcausallm_core.so libnntr_hvx_skel.so')
    report = {'serial': args.serial, 'model': args.model,
              'continuation': args.continuation, 'binary_hashes': manifest,
              'runs': [], 'note': 'Fixed tokens do not imply identical routing.'}
    common = {'NNTR_HTP_LMHEAD_CPU': '1', 'NNTR_HTP_QS2_VTCM': '1',
              'NNTR_HTP_CONV_M1_HVX': args.conv_hvx,
              'NNTR_HTP_NORM_CONSUMER_FUSION': args.norm_fusion,
              'NNTR_HTP_QKV_ROPE_FUSION': args.rope_fusion,
              'NNTR_HTP_RETAIN_ROUTER_INPUT': args.retain_router_input,
              'NNTR_MOE_CACHE_EXPERTS': '8', 'NNTR_MOE_PREFETCH_EARLY': '1',
              'NNTR_MOE_PREFETCH_READERS': '6', 'NNTR_NUM_THREADS': '8',
              'NNTR_OP_TIME': '1', 'NNTR_HTP_PROFILE': '1',
              'NNTR_PPL_DECODE': args.continuation,
              'LD_LIBRARY_PATH': '.', 'ADSP_LIBRARY_PATH': '.'}
    baseline = 'hybrid' if args.comparison == 'hybrid' else 'strict-unfused'
    for pair in range(args.repeats):
        for mode in ([baseline, 'strict'] if pair % 2 == 0
                     else ['strict', baseline]):
            busy = subprocess.run(adb + ['pidof nntrainer_causallm'],
                                  text=True, capture_output=True, timeout=30)
            if busy.returncode != 1 or busy.stdout.strip():
                raise RuntimeError('Device is busy or process check failed: '
                                   + busy.stdout + busy.stderr)
            strict = mode != 'hybrid'
            env = dict(common, NNTR_HTP_STRICT=str(int(strict)),
                       NNTR_HTP_SHARED_ACTIVATIONS=str(int(strict)),
                       NNTR_HTP_ATTN_DECODE_HMX=(args.attention_hmx if args.attention_hmx
                                                is not None else str(int(strict))),
                       NNTR_HTP_DSPQ_SPIN_US=(str(args.strict_spin_us)
                                              if strict else '0'))
            if mode == 'strict-unfused':
                env.update(NNTR_HTP_NORM_CONSUMER_FUSION='0',
                           NNTR_HTP_QKV_ROPE_FUSION='0',
                           NNTR_HTP_RETAIN_ROUTER_INPUT='0')
            command = f'cd {runtime} && ' + ' '.join(
                f'{key}={shlex.quote(value)}' for key, value in env.items())
            command += ' ./nntrainer_causallm ' + shlex.quote(args.model)
            name = f'{pair + 1}-{mode}.log'
            with (args.output / name).open('w') as log:
                subprocess.run(adb + [command], stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=300)
            text = (args.output / name).read_text()
            match = re.search(r'^generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS',
                              text, re.M)
            if not match or '[PPL] decode tokens=' not in text:
                raise RuntimeError(f'Incomplete fixed-token run: {name}')
            times = collections.defaultdict(float)
            for kind, calls, us in re.findall(
                    r'type=(\S+) calls=(\d+) sum_us=([\d.]+)', text):
                if int(calls) == int(match[1]):
                    times[kind] += float(us) / 1000
            row = {'mode': mode, 'log': name, 'env': env,
                   'tokens': int(match[1]), 'ms': int(match[2]),
                   'tps': float(match[3]), 'operator_ms': dict(times)}
            nll = re.search(r'\[PPL\] decode tokens=.*nll_sum=([^\s]+)', text)
            steps = re.findall(r'\[PPL\] decode step=.*', text)
            if not nll or len(steps) != row['tokens']:
                raise RuntimeError(f'Incomplete decode accuracy trace: {name}')
            row['nll_sum'] = float(nll[1])
            row['step_trace_sha256'] = hashlib.sha256(
                '\n'.join(steps).encode()).hexdigest()
            for key, pattern in {
                'queue_calls': r'dspq: close calls=(\d+)',
                'fused_norm_calls': r'norm consumer fusion calls=(\d+)',
                'fused_rope_calls': r'fused QKV RoPE calls=(\d+)',
                'retained_router_calls': r'retained router input calls=(\d+)',
            }.items():
                count = re.search(pattern, text)
                row[key] = int(count[1]) if count else 0
            report['runs'].append(row)
            (args.output / 'results.json').write_text(
                json.dumps(report, indent=2) + '\n')
            print(f'{name}: {row["tps"]:.4f} tok/s', flush=True)
    report['median_tps'] = {
        mode: statistics.median(row['tps'] for row in report['runs']
                                if row['mode'] == mode)
        for mode in [baseline, 'strict']}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
