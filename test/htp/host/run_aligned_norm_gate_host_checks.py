#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check actual HVX norm/gate source with lane emulation and explicit alignment.

Compile backend sources and emulation headers from the selected checkout.
No copied kernel, prebuilt reference object, SDK, or device is required.
These checks validate indexing, memory boundaries and the host arithmetic model;
silicon rounding and compiler lowering still require device parity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[2]
    host = root / 'test/htp/host'
    backend = root / 'nntrainer/tensor/htp_backend'
    cc = shlex.split(os.environ.get('CC', 'cc'))
    flags = ['-std=c11', '-O1' if args.sanitize else '-O2', '-Wall', '-Wextra',
             '-Werror', '-Wno-unused-parameter', '-fno-strict-aliasing',
             '-ffp-contract=off']
    env = os.environ.copy()
    if args.sanitize:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g1']
        env['ASAN_OPTIONS'] = env.get('ASAN_OPTIONS', '') + ':halt_on_error=1'
        env['UBSAN_OPTIONS'] = env.get('UBSAN_OPTIONS', '') + ':halt_on_error=1'
    includes = ['-I' + str(p) for p in
                (host / 'hvx_emu', host / 'stub', backend.parent, backend / 'hvx')]
    sources = {
        'norm': [backend / 'hvx/hvx_quant_u8.c',
                 backend / 'hvx/hvx_rmsnorm_rows_f32.c'],
        'gate': [backend / 'hvx/hvx_conv_gate_f32.c'],
    }
    report = {
        'repo_root': str(root), 'sanitize': args.sanitize,
        'kernel_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for files in sources.values() for p in files},
        'fixture_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [here / 'aligned_norm_host_check.c',
                                     here / 'aligned_gate_host_check.c',
                                     here / 'hvx_alignment_host_alloc.h']},
        'checks': [],
        'scope': 'Actual selected-checkout HVX sources, host lane emulation; no claim of silicon arithmetic equivalence.',
    }
    with tempfile.TemporaryDirectory(prefix='nntr-aligned-norm-gate-') as path:
        for kind, modes in [('norm', (0, 1)), ('gate', (0, 1, 2))]:
            binary = Path(path) / ('aligned_' + kind + '_host_check')
            cmd = cc + flags + includes + [str(here / ('aligned_' + kind + '_host_check.c'))]
            cmd += [str(p) for p in sources[kind]] + ['-lm', '-o', str(binary)]
            subprocess.run(cmd, check=True)
            for mode in modes:
                output = subprocess.check_output([str(binary), str(mode)], env=env, text=True)
                print(output, end='', flush=True)
                report['checks'].append({'kind': kind, 'mode': mode, 'output': output.strip()})
    report['result'] = 'PASS'
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
