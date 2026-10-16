#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject deliberately broken Tile8 source with the actual-source fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--simde-include', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cxx', default='g++')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rel = Path('Applications/CausalLM/layers/conv_block_layer.cpp')
source = (a.source / rel).read_text()
start = source.index('static bool lfmConvGateTilePrefill(')
end = source.index('/** Preserve the wide ARM convolution', start)
helper = source[start:end]
mutations = {
    'lost_chunk_halo': ('if (r >= last - 2u)', 'if (r >= rows - 2u)'),
    'read_dead_gated_rows': ('if (begin == first) {', 'if (true) {'),
    'write_dead_gated_rows': ('if (r >= last - 2u)', 'if (true)'),
    'missing_postgate': ('vmulq_f32(v, vld1q_f32(a + channels))', 'v'),
    'reordered_fma': ('v = vfmaq_f32(v, prev1, w1);\n          v = vfmaq_f32(v, prev2, w2);',
                      'v = vfmaq_f32(v, prev2, w2);\n          v = vfmaq_f32(v, prev1, w1);'),
    'chunk_projection_global_row': ('static_cast<size_t>(r - first) * 3u * channels + c',
                                    'static_cast<size_t>(r) * 3u * channels + c'),
    'missing_output_overlap_guard': ('!disjoint(projection, projection_bytes, output, bytes) ||', 'false ||'),
    'partial_tile_accepted': ('first == 0 && last == 1024', 'first <= 1 && last == 1024'),
}
runner = Path(__file__).with_name('run_lfm2_conv_gate_tile_check.py').resolve()
results = {}
for name, (old, new) in mutations.items():
    if helper.count(old) != 1:
        raise RuntimeError(f'{name}: expected exactly one source anchor')
    directory = a.output / name
    altered = directory / 'source' / rel
    altered.parent.mkdir(parents=True, exist_ok=True)
    text = source[:start] + helper.replace(old, new) + source[end:]
    altered.write_text(text)
    command = [sys.executable, str(runner), '--repo', str(a.repo.resolve()),
               '--source', str((directory / 'source').resolve()), '--baseline', str(a.baseline.resolve()),
               '--simde-include', str(a.simde_include.resolve()), '--output', str((directory / 'check').resolve()),
               '--cxx', a.cxx, '--sanitize', '--build-only', '--tile-on']
    with (directory / 'compile.log').open('w') as log:
        built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if built.returncode:
        raise RuntimeError(f'{name}: compilation failure is not mutation detection')
    env = os.environ.copy()
    env.update(NNTR_LFM2_CONV_GATE_TILE='1', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1')
    binary = directory / 'check' / 'check'
    with (directory / 'run.log').open('w') as log:
        ran = subprocess.run([str(binary.resolve())], stdout=log, stderr=subprocess.STDOUT, env=env)
    output = (directory / 'run.log').read_text()
    intended = ran.returncode != 0 and (
        'CONV_GATE_TILE FAIL:' in output or
        ('ERROR: AddressSanitizer:' in output and 'lfmConvGateTilePrefill' in output))
    results[name] = {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                     'compile_command': command, 'compile_rc': built.returncode,
                     'run_rc': ran.returncode, 'intended_detection': intended}
    (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    if not intended:
        raise RuntimeError(f'{name}: failed to detect intended runtime fault')
    print(f'{name}: intended runtime rejection PASS', flush=True)
print(f'TILE_MUTATIONS {len(results)} PASS')
