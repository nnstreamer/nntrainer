#!/usr/bin/env python3
"""Fault-inject frozen actual production extracts, never edit production."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ('source', 'reference', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
tests = a.source / 'test/htp/host'
spec = importlib.util.spec_from_file_location('actual_fixture', tests / 'run_lfm2_prefill_recency_check.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
items = [
    ('first_order', 'candidate.inc', 'extra_top_k->push_back(id);',
     'extra_top_k->insert(extra_top_k->begin(), id);', '1',
     'hint first-occurrence/append semantics changed'),
    ('bit63', 'candidate.inc', 'uint64_t{1} << static_cast<unsigned int>(id)',
     'uint32_t{1} << static_cast<unsigned int>(id)', '1', 'shift exponent'),
    ('append', 'candidate.inc', 'extra_top_k && extra_top_k->empty()',
     'extra_top_k', '1', 'hint first-occurrence/append semantics changed'),
    ('trace_configured', 'helper.inc', '(trace && *trace) || ',
     '', 'trace-configured', 'actual flattened caller never qualified'),
    ('trace_opened', 'helper.inc', 'moeTrace() != nullptr ||',
     '', 'trace-opened', 'actual flattened caller never qualified'),
    ('flattened_shape', 'helper.inc', 'rows.height() != 1',
     'rows.height() != 1024', '1', 'actual flattened caller never qualified'),
]


def run(item):
    name, include, old, new, mode, intended = item
    out = a.output / name
    out.mkdir(exist_ok=True)
    hashes = fixture.extract(a.source, a.reference, out)
    target = out / include
    text = target.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    target.write_text(text.replace(old, new))
    src = out / 'lfm2_prefill_recency_check.cpp'
    shutil.copy2(tests / src.name, src)
    cmd = ['g++', '-std=c++17', '-O1', '-g', '-pthread',
           '-fno-fast-math', '-ffp-contract=off', '-fsanitize=address,undefined',
           '-fno-omit-frame-pointer', '-Wall', '-Wextra',
           '-Wno-unused-parameter', '-Wno-unused-function',
           '-I' + str(out.resolve()), str(src.resolve()),
           '-o', str((out / 'check').resolve())]
    record = {'name': name, 'include': include, 'before': old, 'after': new,
              'mode': mode, 'intended': intended, 'original_extract_sha256': hashes,
              'mutated_include_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'compile': cmd}
    with (out / 'build.log').open('w') as log:
        compiled = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    record['compile_returncode'] = compiled.returncode
    if compiled.returncode:
        record['pass'] = False
    else:
        env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                   UBSAN_OPTIONS='halt_on_error=1',
                   NNTR_LFM2_COMPACT_PREFILL_RECENCY='1')
        env.pop('NNTR_MOE_TRACE', None)
        argv = [str((out / 'check').resolve())]
        if mode == 'trace-configured':
            env['NNTR_MOE_TRACE'] = str(out / 'does-not-exist' / 'trace')
        if mode == 'trace-opened':
            env['NNTR_MOE_TRACE'] = str(out / 'trace.txt')
            argv += ['opened-trace']
        result = subprocess.run(argv, env=env, text=True, capture_output=True)
        output = result.stdout + result.stderr
        (out / 'run.log').write_text(output)
        record.update(run=argv, returncode=result.returncode,
                      pass_=result.returncode != 0 and intended in output)
        record['pass'] = record.pop('pass_')
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'name': name, 'pass': record['pass'],
                      'returncode': record.get('returncode')}), flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run, items))
(a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
if not all(r['pass'] for r in results):
    raise SystemExit(1)
