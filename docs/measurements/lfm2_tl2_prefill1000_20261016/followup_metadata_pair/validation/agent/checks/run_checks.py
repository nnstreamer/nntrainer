#!/usr/bin/env python3
"""Run the actual registry refresh fixture (Linux ELF linker required)."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', required=True, type=Path)
p.add_argument('--source', type=Path)
p.add_argument('--fixture', type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--sanitize', action='store_true')
p.add_argument('--mutations', action='store_true')
p.add_argument('--cc', default=os.environ.get('CC', 'gcc'))
a = p.parse_args()
if not sys.platform.startswith('linux'):
    raise SystemExit('This standalone check uses the Linux ELF section linker.')
a.repo = a.repo.resolve(); a.output = a.output.resolve()
b = a.repo / 'nntrainer/tensor/htp_backend'
h = a.repo / 'test/htp/host'
src = (a.source or b / 'hmx/hexkl_mm_u8i4_dma.c').resolve()
fixture = (a.fixture or h / 'tl2_refresh_pair_check.c').resolve()
a.output.mkdir(parents=True, exist_ok=True)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
env = dict(os.environ)
env['ASAN_OPTIONS'] = 'detect_leaks=1:abort_on_error=1'
env['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
results = []

def run_one(name, source, expected_pass):
    obj = a.output / name
    command = [a.cc, '-std=gnu11', '-O1' if a.sanitize else '-O2',
               '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
               '-ffunction-sections', '-fdata-sections', '-include', 'malloc.h']
    if a.sanitize:
        command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    for inc in [source.parent, h / 'stub', h / 'standin', h.parent,
                b, b.parent, b / 'hmx', b / 'hvx']:
        command += ['-I', str(inc)]
    command += [str(fixture), '-Wl,--gc-sections', '-lm', '-o', str(obj)]
    (a.output / (name + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
    compiled = subprocess.run(command, capture_output=True, text=True)
    (a.output / (name + '-compile.log')).write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(name + ' failed to compile: ' + compiled.stderr)
    result = subprocess.run([str(obj)], capture_output=True, text=True, env=env)
    (a.output / (name + '-run.log')).write_text(result.stdout + result.stderr)
    if (result.returncode == 0) != expected_pass:
        raise RuntimeError(name + ' had unexpected exit ' + str(result.returncode))
    if not expected_pass and 'Assertion' not in result.stderr and 'assertion' not in result.stderr:
        raise RuntimeError(name + ' failed without the intended assertion: ' + result.stderr)
    results.append({'name': name, 'exit': result.returncode,
                    'expected_pass': expected_pass})
    print(name + (': PASS' if expected_pass else ': MUTATION CAUGHT'))

run_one('baseline', src, True)
if a.mutations:
    original = src.read_text()
    start = original.index('int hexkl_weight_tl2_refresh_pair(')
    end = original.index('\n/**', start)
    body = original[start:end]
    mutations = {
        'wrong_dn_row_count': ('dn4, dn4, dn4, 2u, 0, 0);', 'dn4, dn4, dn4, 1u, 0, 0);'),
        'wrong_gu_stride': ('g->w_scale, gt, gn4, gn4, gn4, 2u', 'g->w_scale, gt, gn4 + 128u, gn4, gn4, 2u'),
        'missing_drain': ('    hexkl_dma_ring_drain();\n', ''),
        'early_generation': ('    const uint32_t gn4', '    g->tl2_generation_seen = generation;\n    const uint32_t gn4'),
        'missing_destination_invalidate': ('    hexkl_dcache_inva(d->w_scale, 2u * dn4);\n', ''),
        'redundant_bias_zero': ('    g->tl2_generation_seen = generation;',
                                '    memset(g->bias, 0, gn4);\n    g->tl2_generation_seen = generation;'),
    }
    for name, (old, new) in mutations.items():
        if body.count(old) != 1:
            raise RuntimeError('Mutation anchor count was not one: ' + name)
        folder = a.output / ('source-' + name)
        folder.mkdir(exist_ok=True)
        mutated = folder / src.name
        mutated.write_text(original[:start] + body.replace(old, new) + original[end:])
        run_one(name, mutated, False)
(a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
