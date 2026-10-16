#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deliberately break extracted production guards/arithmetic; require rejection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
for key in ['repo', 'source', 'simde-include', 'output']:
    p.add_argument('--' + key, type=Path, required=True)
p.add_argument('--cxx', default='g++')
a = p.parse_args()
rel = 'Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp'
source = (a.source / rel).read_text()
changes = {
    'dst_extent': ('dst.getDim() != src.getDim()', 'false', 'invalid helper accepted'),
    'gamma_alias': ('!disjoint(scale, 2048 * sizeof(float))', 'false', 'invalid helper accepted'),
    'last_expert': ('e < 64; ++e', 'e < 63; ++e', 'invalid helper accepted'),
    'missing_rms_row': ('output + first, rows_per_task, 2048, eps', 'output + first, rows_per_task - 1, 2048, eps', 'row norm numeric/canary mismatch'),
    'missing_gamma_row': ('row < rows_per_task; ++row', 'row < rows_per_task - 1; ++row', 'row norm numeric/canary mismatch'),
    'wrong_mul_beta': ('out, 1.0f, 0.0f, 1, 1', 'out, 1.0f, 1.0f, 1, 1', 'gamma call coefficients changed'),
    'gelu_policy': ('!router_norm && !gelu_act && !softmax_router', '!router_norm && (true || !gelu_act) && !softmax_router', 'caller policy dispatch'),
    'wrong_row_offset': ('task * rows_per_task * 2048', 'task * rows_per_task * 2048 + 1', 'row norm numeric/canary mismatch'),
}
a.output.mkdir(parents=True, exist_ok=True)
results = {}
for name, (old, new, expected) in changes.items():
    assert source.count(old) == 1, (name, source.count(old))
    out = a.output / name
    path = out / 'source' / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.replace(old, new))
    cmd = [sys.executable, str(Path(__file__).with_name('run_lfm2_moe_row_norm_check.py')),
           '--repo', str(a.repo), '--source', str(out / 'source'),
           '--simde-include', str(a.simde_include), '--output', str(out / 'check'),
           '--cxx', a.cxx, '--sanitize']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out / 'run.log').write_text(proc.stdout)
    assert proc.returncode != 0 and expected in proc.stdout, (name, proc.returncode, proc.stdout[-2500:])
    results[name] = {'returncode': proc.returncode, 'expected_assertion': expected, 'detected': True}
    print(name, 'REJECTED')
(a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
