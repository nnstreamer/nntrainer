#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile actual predictor headers/callbacks for host or native ARM checking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def block(text, at):
    start = text.index(at)
    left = text.index('{', start)
    depth = 1
    right = left + 1
    while depth:
        depth += (text[right] == '{') - (text[right] == '}')
        right += 1
    return text[left + 1:right - 1]


def extract(source, output):
    rel = Path('nntrainer/tensor/htp_backend')
    hashes = {}
    for variant, namespace in [('baseline', 'before'), ('candidate', 'after')]:
        header = (source[variant] / rel / 'htp_router_predict.h').read_text()
        core = (source[variant] / rel / 'htp_compute_ops.cpp').read_text()
        hashes[variant] = {'header': hashlib.sha256(header.encode()).hexdigest(),
                           'core': hashlib.sha256(core.encode()).hexdigest()}
        transformed = header.replace('namespace nntrainer {', 'namespace ' + namespace + ' {')
        transformed = transformed.replace('NNTR_HTP_ROUTER_PREDICT_H', 'CHECK_' + namespace.upper() + '_H')
        # SIMD aliases are loaded before the simulated architecture macro.
        transformed = transformed.replace('#include <arm_neon.h>', '#ifndef PREDICT_SIMDE\n#include <arm_neon.h>\n#endif')
        (output / (variant + '.h')).write_text(transformed)
        checked = transformed
        if variant == 'candidate':
            marker = 'float *scores) {'
            assert checked.count(marker) == 1
            checked = checked.replace(marker, marker + '\n  ++dot_calls;')
            checked = checked.replace('namespace after {', 'namespace after {\ninline unsigned dot_calls = 0;')
        (output / (variant + '-checked.h')).write_text(checked)
        body = block(header, 'htpPredictExperts(')
        raw = body[:body.index('  for (float score : scores)')] + '\n  return scores;\n'
        if variant == 'baseline':
            raw = '(void)lfm2_register_accum;\n' + raw
        (output / (variant + '-raw.inc')).write_text(raw)
        callback = block(core, '[this, cpus, blend')
        (output / (variant + '-callback.inc')).write_text(callback)
    core = (source['candidate'] / rel / 'htp_compute_ops.cpp').read_text()
    start = core.index('    const char *register_env =')
    end = core.index('    cpu_predictor_ =', start)
    guard = core[start:end] + 'return register_accum;\n'
    (output / 'guard.inc').write_text(guard)
    return hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--simde-include', type=Path)
    p.add_argument('--cxx', default='g++')
    p.add_argument('--sanitize', action='store_true')
    p.add_argument('--native-arm', action='store_true')
    p.add_argument('--fallback', action='store_true')
    p.add_argument('--ndk', type=Path)
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    hashes = extract({'baseline': a.reference, 'candidate': a.source}, a.output)
    fixture = Path(__file__).with_name('lfm2_predict_register_check.cpp')
    micro = Path(__file__).with_name('lfm2_predict_register_micro.cpp')
    for f in (fixture, micro):
        shutil.copy2(f, a.output / f.name)
    cmd = [a.cxx, '-std=c++17', '-O3', '-g', '-pthread', '-Wall', '-Wextra', '-Werror', '-ffp-contract=fast']
    if a.ndk:
        assert a.native_arm
        root = a.ndk / 'toolchains/llvm/prebuilt/linux-x86_64'
        cmd[0] = str(root / 'bin/aarch64-linux-android29-clang++')
        cmd += ['-static-libstdc++', '-march=armv8.2-a+fp16+dotprod+i8mm', '-Wl,-z,max-page-size=16384']
    elif a.native_arm:
        cmd += ['-march=armv8-a']
    elif a.fallback:
        cmd += ['-DPREDICT_FALLBACK']
    else:
        if not a.simde_include:
            p.error('--simde-include is required for host NEON checks')
        cmd += ['-DPREDICT_SIMDE', '-mfma', '-I' + str(a.simde_include.resolve())]
    if a.sanitize:
        cmd += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    commands = []
    for path in (fixture, micro):
        name = 'check' if path == fixture else 'micro'
        c = cmd + [str(a.output / path.name), '-o', str(a.output / name)]
        commands.append(c)
        subprocess.run(c, check=True)
    result = {'hashes': hashes, 'compile': commands,
              'scope': 'Actual predictor header, raw-score prefix and prediction callback. Queue and pin are stand-ins; no real DSP, file I/O or scheduling overlap is modeled.'}
    (a.output / 'commands.json').write_text(json.dumps(result, indent=2) + '\n')
    if not a.build_only:
        env = os.environ.copy()
        env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
        subprocess.run([str(a.output / 'check')], env=env, check=True)


if __name__ == '__main__':
    main()
