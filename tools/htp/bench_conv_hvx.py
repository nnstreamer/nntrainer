#!/usr/bin/env python3
# @brief Interleave HMX/HVX M=1 convolution, FC or attention on one deployed Android runtime.
# SPDX-License-Identifier: Apache-2.0
"""Interleave HMX/HVX M=1 convolution, FC or attention on one deployed Android runtime.

Both variants use the same binaries and model. Profiles and fixed-token
quality runs are optional and kept separate from clean generation timings.
This script does not install binaries or change the model configuration.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['conv', 'fc', 'attn'], default='conv')
    parser.add_argument('--serial', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--chunk-tiles', type=int, choices=[4, 8, 16, 32], default=8)
    parser.add_argument('--spin-us', type=int, default=20)
    parser.add_argument('--norm-consumer-fusion', type=int, choices=[0, 1], default=0)
    parser.add_argument('--continuation', help='Device token ID file for a quality run')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or args.spin_us < 0:
        parser.error('repeats must be positive and spin-us nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    adb = ['adb', '-s', args.serial, 'shell']

    def read(command):
        return subprocess.run(adb + [command], check=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()

    def wait_idle():
        deadline = time.monotonic() + 60
        while True:
            result = subprocess.run(
                adb + ['pidof nntrainer_causallm unittest_hvx_mm_u8i4 unittest_hvx_attn_f16 attn_probe'],
                capture_output=True, text=True, timeout=15)
            if result.returncode == 1 and not result.stdout.strip():
                return
            if result.returncode not in (0, 1) or time.monotonic() > deadline:
                raise RuntimeError('Device busy or process check failed: '
                                   + result.stdout + result.stderr)
            time.sleep(1)

    runtime = shlex.quote(args.runtime)
    hash_command = (f'cd {runtime} && sha256sum nntrainer_causallm '
                    'libnntrainer.so libcausallm_core.so libnntr_hvx_skel.so')
    hashes = read(hash_command)
    model = shlex.quote(args.model)
    model_config_hashes = read(
        f'cd {model} && sha256sum config.json nntr_config.json')
    common = {
        'LD_LIBRARY_PATH': '.', 'ADSP_LIBRARY_PATH': '.',
        'NNTR_HTP_STRICT': '1', 'NNTR_HTP_LMHEAD_CPU': '1',
        'NNTR_HTP_SHARED_ACTIVATIONS': '1', 'NNTR_HTP_QS2_VTCM': '1',
        'NNTR_HTP_ATTN_DECODE_HMX': '1', 'NNTR_HTP_DSPQ': '1',
        'NNTR_HTP_DSPQ_MATH': '1', 'NNTR_HTP_DSPQ_ROUTER': '1',
        'NNTR_HTP_DSPQ_SPIN_US': str(args.spin_us),
        'NNTR_HTP_ADD_NORM_FUSION': '1', 'NNTR_HTP_BLOCK_NORM_FUSION': '1',
        'NNTR_HTP_NORM_CONSUMER_FUSION': str(args.norm_consumer_fusion),
        'NNTR_HTP_CONV_M1_HVX': '1',
        'NNTR_HTP_FC_M1_HVX': '0',
        'NNTR_HTP_CONV_CHUNK_TILES': '8',
        'NNTR_HTP_FC_CHUNK_TILES': '8',
        'NNTR_MOE_CACHE_EXPERTS': '8', 'NNTR_MOE_PREFETCH_EARLY': '1',
        'NNTR_MOE_PREFETCH_READERS': '6', 'NNTR_NUM_THREADS': '8',
    }
    if args.target == 'attn':
        common['NNTR_HTP_FC_M1_HVX'] = '1'
    else:
        common[f'NNTR_HTP_{args.target.upper()}_CHUNK_TILES'] = str(args.chunk_tiles)
    if args.continuation:
        common['NNTR_PPL_DECODE'] = args.continuation
    if args.profile:
        common.update(NNTR_OP_TIME='1', NNTR_HTP_PROFILE='2')
    report = {'target': args.target, 'serial': args.serial, 'runtime': args.runtime, 'model': args.model,
              'binary_hashes': hashes, 'model_config_hashes': model_config_hashes,
              'environment': common, 'runs': [],
              'kind': 'profile' if args.profile else 'fixed-token' if args.continuation
              else 'clean generation',
              'note': 'Same-sitting comparison; fixed tokens alone do not fix routing.'}
    if args.continuation:
        report['continuation_hash'] = read('sha256sum ' + shlex.quote(args.continuation))

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    for pair in range(args.repeats):
        for mode in (['hmx', 'hvx'] if pair % 2 == 0 else ['hvx', 'hmx']):
            wait_idle()
            if read(hash_command) != hashes:
                raise RuntimeError('Deployed runtime changed during the comparison')
            temperature = read("dumpsys battery | grep 'temperature:' | head -1")
            env = dict(common)
            if args.target == 'attn':
                env['NNTR_HTP_ATTN_DECODE_HMX'] = str(int(mode == 'hmx'))
            else:
                env[f'NNTR_HTP_{args.target.upper()}_M1_HVX'] = str(int(mode == 'hvx'))
            # Do not interpret NAME=0 as off for presence-based diagnostics.
            unset = ('unset NNTR_HTP_PROFILE NNTR_OP_TIME NNTR_HTP_MATH_PROFILE '
                     'NNTR_ATTN_TRACE NNTR_HTP_ATTN_TRACE NNTR_PPL_DECODE && ')
            command = f'cd {runtime} && ' + unset + ' '.join(
                key + '=' + shlex.quote(value) for key, value in env.items())
            command += ' ./nntrainer_causallm ' + model
            name = f'{pair + 1}-{mode}.log'
            with (args.output / name).open('w') as log:
                result = subprocess.run(adb + [command], stdout=log,
                                        stderr=subprocess.STDOUT, timeout=300)
            text = (args.output / name).read_text()
            match = re.search(r'^generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS',
                              text, re.M)
            label = {'conv': 'conv', 'fc': 'FC', 'attn': 'attention'}[args.target]
            if result.returncode or not match or f'{label} M1: {mode.upper()}' not in text:
                raise RuntimeError(f'Incomplete run or missing dispatch echo: {name}')
            if args.continuation and '[PPL] decode tokens=' not in text:
                raise RuntimeError(f'No fixed-token quality result: {name}')
            nll = re.search(r'nll_sum=([\d.eE+-]+)', text)
            prefill = re.search(r'^prefill: (\d+) tokens, (\d+) ms', text, re.M)
            row = {'pair': pair + 1, 'mode': mode, 'log': name,
                   'start_battery_temperature': temperature,
                   'tokens': int(match[1]), 'decode_ms': int(match[2]),
                   'tps': float(match[3]),
                   'prefill_ms': int(prefill[2]) if prefill else None,
                   'nll_sum': nll[1] if nll else None}
            report['runs'].append(row)
            save()
            print(f'{name}: {row["tps"]:.4f} TPS, {row["decode_ms"]} ms', flush=True)
    report['median_tps'] = {
        mode: statistics.median(row['tps'] for row in report['runs'] if row['mode'] == mode)
        for mode in ['hmx', 'hvx']}
    report['gain_percent'] = (report['median_tps']['hvx'] /
                              report['median_tps']['hmx'] - 1) * 100
    report['runtime_unchanged'] = read(hash_command) == hashes
    save()
    if not report['runtime_unchanged']:
        raise RuntimeError('Deployed runtime changed during the comparison')
    print(json.dumps(report['median_tps']), flush=True)


if __name__ == '__main__':
    main()
