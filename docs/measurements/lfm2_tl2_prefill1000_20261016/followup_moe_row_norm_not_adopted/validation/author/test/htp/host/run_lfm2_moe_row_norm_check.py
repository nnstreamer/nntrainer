#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract the production norm/caller and ARM math for bounded host checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def function(text, start):
    first = text.index(start)
    brace = text.index('{', first)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[first:end] + '\n'


p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--simde-include', type=Path, required=True)
p.add_argument('--cxx', default='g++')
p.add_argument('--sanitize', action='store_true')
p.add_argument('--build-only', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rel = 'Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp'
source = (a.source / rel).read_text()
neon = (a.repo / 'nntrainer/tensor/cpu_backend/arm/neon_impl.cpp').read_text()
baseline = function(source, 'static void normRows(')
helper = 'static bool\n' + function(source, 'lfmPrefillRowNorm(')
caller = function(source, '    if (in_norm && !norm_on_accel) {')
math = (function(neon, 'static float hsum_f32x4(') +
        function(neon, 'void rms_norm_wrt_width_fp32_intrinsic(') +
        function(neon, 'void ele_mul(const unsigned int N,'))
parts = {'baseline.inc': baseline, 'helper.inc': helper,
         'caller.inc': caller, 'math.inc': math}
for name, data in parts.items():
    (a.output / name).write_text(data)
fixture = Path(__file__).with_name('lfm2_moe_row_norm_check.cpp')
shutil.copy2(fixture, a.output)
cmd = [a.cxx, '-std=c++17', '-O2', '-g', '-pthread', '-mfma',
       '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
       '-I' + str(a.simde_include.resolve()),
       str((a.output / fixture.name).resolve()), '-o', str((a.output / 'check').resolve())]
if a.sanitize:
    cmd[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
manifest = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'neon_source_sha256': hashlib.sha256(neon.encode()).hexdigest(),
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'extracts': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in parts.items()},
            'compile': cmd,
            'scope': 'Actual app helper/caller and ARM math. Tensor/ops metadata, gamma broadcast and worker scheduler stand-ins. Native core/Tensor validation is separate.'}
(a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run(cmd, check=True)
if not a.build_only:
    for value in ['1', '0', 'unset', 'true']:
        env = os.environ.copy()
        env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
        if value == 'unset':
            env.pop('NNTR_LFM2_MOE_ROW_NORM', None)
        else:
            env['NNTR_LFM2_MOE_ROW_NORM'] = value
        subprocess.run([str((a.output / 'check').resolve())], env=env, check=True)
