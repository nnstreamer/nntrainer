#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual assignment/guard/reshape/caller extracts plus actual ExpertLru.

Host runs use the real host STL/libm, with optional ASan/UBSan. --native-build
builds the same checks for an external Android run; it never invokes adb.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def function(text, token):
    start = text.index(token)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end] + '\n'


def extract(source, reference, output):
    rel = Path('Applications/CausalLM/models/lfm2_moe')
    new = (source / rel / 'lfm2_moe_layer.cpp').read_text()
    old = (reference / rel / 'lfm2_moe_layer.cpp').read_text()
    parts = {}
    for tag, code in (('baseline', old), ('candidate', new)):
        method = function(code, 'void Lfm2MoELayer::buildExpertAssignments(')
        parts[tag + '.inc'] = method
        parts[tag + '-decl.inc'] = method[:method.index('{')].replace(
            'Lfm2MoELayer::', '') + ';\n'
    parts['constants.inc'] = '\n'.join(re.findall(
        r'^static constexpr (?:bool NORM_TOPK_PROB|float ROUTED_SCALING_FACTOR|unsigned EXTRA_TOPK).*?;',
        new, re.M)) + '\n'
    if len(parts['constants.inc'].splitlines()) != 3:
        raise ValueError('missing actual router constants')
    parts['trace.inc'] = function(new, 'static std::FILE *moeTrace()')
    parts['helper.inc'] = 'static bool\n' + function(new, 'lfmCompactPrefillRecency(')
    start = new.index('    bool compact_recency = false;')
    end = new.index('    size_t max_assigned_tokens', start)
    parts['caller.inc'] = new[start:end]
    caller = new.index('void Lfm2MoELayer::incremental_forwarding(')
    statement = re.search(r'input\.reshape\(\{total_tokens,\s*1,\s*1,\s*hidden_size\}\);', new[caller:])
    if not statement or caller + statement.start() >= start:
        raise ValueError('actual input reshape must precede candidate caller')
    parts['reshape.inc'] = statement[0] + '\n'
    # A single include preserves source order even under clang-format's include
    # sorting. Separate adjacent includes would put caller before reshape.
    parts['prepared-caller.inc'] = parts['reshape.inc'] + parts['caller.inc']
    # Prove selection/FP body unchanged byte-for-byte, not merely in a mirror.
    def fp_body(method):
        a = method.index('    const float *lrow')
        b = method.index('    if (extra_top_k)', a)
        return method[a:b]
    if fp_body(parts['baseline.inc']) != fp_body(parts['candidate.inc']):
        raise ValueError('candidate changed existing per-row FP/selection body')
    before_lru = (reference / rel / 'expert_lru.h').read_bytes()
    candidate_lru = (source / rel / 'expert_lru.h').read_bytes()
    if before_lru != candidate_lru:
        raise ValueError('generic ExpertLru changed')
    parts['expert_lru.h'] = candidate_lru.decode()
    for name, text in parts.items():
        (output / name).write_text(text)
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in parts.items()}


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cxx', default='g++')
    p.add_argument('--sanitize', action='store_true')
    p.add_argument('--fast-math', action='store_true')
    p.add_argument('--generic', action='store_true')
    p.add_argument('--native-build', action='store_true')
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    hashes = extract(a.source, a.reference, a.output)
    here = Path(__file__).parent
    src = a.output / 'lfm2_prefill_recency_check.cpp'
    shutil.copy2(here / src.name, src)
    opt = '-O3' if a.fast_math or a.native_build else '-O2'
    cmd = [a.cxx, '-std=c++17', opt, '-g', '-pthread', '-Wall', '-Wextra',
           '-Wno-unused-parameter', '-Wno-unused-function', '-fstack-usage',
           '-I' + str(a.output.resolve()), str(src.resolve()),
           '-o', str((a.output / 'check').resolve())]
    if a.fast_math or a.native_build:
        cmd[1:1] = ['-ffast-math']
    else:
        cmd[1:1] = ['-fno-fast-math', '-ffp-contract=off']
    if a.generic:
        cmd[1:1] = ['-DRECENCY_CHECK_GENERIC=1']
    if a.native_build:
        cmd[1:1] = ['-march=armv8.2-a+fp16+dotprod+i8mm', '-static-libstdc++']
    if a.sanitize:
        cmd[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    result = {'compile': cmd, 'extract_sha256': hashes,
              'fixture_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
              'scope': 'Actual FP/std::partial_sort method and unchanged ExpertLru. Tensor metadata/getWeight and reshape are stand-ins; exact production reshape/caller/trace/guard text is extracted. No DSP, I/O, full-model or timing proof.',
              'runs': []}
    subprocess.run(cmd, check=True)
    result['binary_sha256'] = hashlib.sha256((a.output / 'check').read_bytes()).hexdigest()
    if not a.build_only and not a.native_build:
        for mode in ('1', '0', 'unset', 'true', 'trace-configured', 'trace-opened'):
            env = os.environ.copy()
            env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
            env.pop('NNTR_LFM2_COMPACT_PREFILL_RECENCY', None)
            env.pop('NNTR_MOE_TRACE', None)
            if mode != 'unset':
                env['NNTR_LFM2_COMPACT_PREFILL_RECENCY'] = mode if mode in ('0', 'true') else '1'
            args = [str((a.output / 'check').resolve())]
            if mode == 'trace-configured':
                env['NNTR_MOE_TRACE'] = str(a.output / 'nonexistent' / 'trace')
            if mode == 'trace-opened':
                env['NNTR_MOE_TRACE'] = str(a.output / 'trace.txt')
                args += ['opened-trace']
            run = subprocess.run(args, env=env, text=True, capture_output=True)
            (a.output / (mode + '.log')).write_text(run.stdout + run.stderr)
            if run.returncode != 0:
                raise RuntimeError(f'{mode} failed ({run.returncode}): {run.stdout}{run.stderr}')
            result['runs'].append({'mode': mode, 'output': run.stdout.strip(), 'returncode': run.returncode})
            print(mode + ': ' + run.stdout.strip(), flush=True)
    (a.output / 'commands.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
