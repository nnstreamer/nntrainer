#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an actual-library fixture; device execution is deliberately separate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def function(text, start):
    first = text.index(start)
    brace = text.index('{', first)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[first:end] + '\n'


p = argparse.ArgumentParser()
for key in ['source', 'output', 'compiler', 'nntrainer-include', 'nntrainer-lib-dir']:
    p.add_argument('--' + key, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rel = 'Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp'
source = (a.source / rel).read_text()
parts = {
    'baseline.inc': function(source, 'static void normRows('),
    'helper.inc': 'static bool\n' + function(source, 'lfmPrefillRowNorm('),
    'caller.inc': function(source, '    if (in_norm && !norm_on_accel) {'),
}
for name, data in parts.items():
    (a.output / name).write_text(data)
fixture = Path(__file__).with_name('lfm2_moe_row_norm_native_check.cpp')
shutil.copy2(fixture, a.output)
cmd = [str(a.compiler.resolve()), '-std=c++17', '-O3', '-ffast-math',
       '-frtti', '-fexceptions', '-fPIC', '-fopenmp',
       '-march=armv8.2-a+fp16+dotprod+i8mm', '-DENABLE_FP16=1',
       '-DUSE__FP16=1', '-DENABLE_HEXKL=1', '-Wall', '-Wextra',
       '-ffunction-sections', '-fdata-sections',
       '-I' + str(a.nntrainer_include.resolve()),
       str((a.output / fixture.name).resolve()),
       '-Wl,--gc-sections', '-Wl,-rpath,$ORIGIN',
       '-L' + str(a.nntrainer_lib_dir.resolve()), '-lnntrainer', '-llog', '-ldl',
       '-o', str((a.output / 'check').resolve())]
manifest = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'extracts': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in parts.items()},
            'compile': cmd,
            'runtime_nntrainer_sha256': hashlib.sha256((a.nntrainer_lib_dir / 'libnntrainer.so').read_bytes()).hexdigest(),
            'scope': 'Actual app helper/caller with real Tensor, RunLayerContext, ThreadManager and linked core functions; no DSP RPC.'}
(a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run(cmd, check=True)
(a.output / 'binary.sha256').write_text(hashlib.sha256((a.output / 'check').read_bytes()).hexdigest() + '\n')
