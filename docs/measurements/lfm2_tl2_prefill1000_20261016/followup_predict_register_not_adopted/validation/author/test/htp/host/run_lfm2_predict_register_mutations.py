#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual-source guard/order mutations; every case must fail at runtime."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--simde-include', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rel = Path('nntrainer/tensor/htp_backend')
mutants = [
 ('reverse_h', 'htp_router_predict.h', 'for (unsigned h = 0; h < 2048u; ++h)', 'for (unsigned h = 2048u; h-- > 0u;)', 'score mismatch'),
 ('skip_lane_group', 'htp_router_predict.h', 's0 = vfmaq_n_f32(s0, vld1q_f32(w + 0u), x);', '(void)x;', 'score mismatch|Assertion'),
 ('wrong_weight_stride', 'htp_router_predict.h', 'weights + static_cast<size_t>(h) * 64u;', 'weights + static_cast<size_t>(h) * 63u;', 'score mismatch|Assertion'),
 ('missing_hidden_guard', 'htp_router_predict.h', 'hidden == 2048u && experts == 64u', 'experts == 64u', 'AddressSanitizer'),
 ('missing_expert_guard', 'htp_router_predict.h', 'hidden == 2048u && experts == 64u && sigmoid_bias', 'hidden == 2048u && sigmoid_bias', 'AddressSanitizer'),
 ('missing_bias_guard', 'htp_router_predict.h', 'experts == 64u && sigmoid_bias)', 'experts == 64u)', 'Assertion'),
 ('missing_option_guard', 'htp_router_predict.h', 'if (lfm2_register_accum && hidden', 'if ((lfm2_register_accum || true) && hidden', 'Assertion'),
 ('missing_exact_env', 'htp_compute_ops.cpp', 'std::strcmp(register_env, "1") == 0', 'register_env', 'Assertion'),
 ('missing_native_guard', 'htp_compute_ops.cpp', 'shape.w_bits == HTP_TL2_WEIGHT_BITS && shape.K == 2048u', '(shape.w_bits == HTP_TL2_WEIGHT_BITS || true) && shape.K == 2048u', 'Assertion'),
]
results = []
runner = Path(__file__).with_name('run_lfm2_predict_register_check.py')
for name, file, old, new, expected in mutants:
    out = a.output / name
    source = out / 'source'
    for path in (rel / 'htp_router_predict.h', rel / 'htp_compute_ops.cpp'):
        dest = source / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.source / path, dest)
    target = source / rel / file
    text = target.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    target.write_text(text.replace(old, new))
    cmd = ['python3', str(runner), '--source', str(source), '--reference', str(a.reference),
           '--output', str(out / 'build'), '--simde-include', str(a.simde_include), '--sanitize']
    with (out / 'run.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log = (out / 'run.log').read_text()
    killed = result.returncode != 0 and re.search(expected, log) is not None and (out / 'build/check').exists()
    results.append({'name': name, 'command': cmd, 'exit_code': result.returncode,
                    'expected': expected, 'runtime_killed': killed})
    print(name, 'PASS' if killed else 'FAIL', flush=True)
    (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    if not killed:
        print(log[-3000:])
        raise SystemExit(1)
