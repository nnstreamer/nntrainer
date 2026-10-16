#!/usr/bin/env python3
"""Recreate the rejected Tile8 actual-source host fixture in a fresh directory."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from verify_archive import verify

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent.parent)
p.add_argument('--repo', type=Path, required=True, help='nntrainer checkout at ccea0d476')
p.add_argument('--simde-include', type=Path, required=True, help='external SIMDe0.8.2 include root')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mutations', action='store_true')
a = p.parse_args()
manifest = verify(a.archive)
if a.output.exists():
    raise ValueError('Refuse to overwrite an existing output')
expected_core = manifest['reproduction']['core_sha256']
core = a.repo / 'nntrainer/tensor/htp_backend/htp_compute_ops.cpp'
if hashlib.sha256(core.read_bytes()).hexdigest() != expected_core:
    raise ValueError('Core differs from the frozen extraction input; use the recorded checkout')
a.output.mkdir(parents=True)
for relative, row in manifest['files'].items():
    target = None
    prefix = 'validation/tile8/author/'
    if relative.startswith(prefix):
        part = relative[len(prefix):]
        if part.startswith(('candidate/', 'test/htp/host/')):
            target = a.output / 'source' / part.removesuffix('.gz')
    elif relative.startswith('validation/baseline/'):
        target = a.output / 'baseline' / relative[len('validation/baseline/'):].removesuffix('.gz')
    if target is not None:
        data = (a.archive / relative).read_bytes()
        if row['gzip']:
            data = gzip.decompress(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
runner = a.output / 'source/test/htp/host/run_lfm2_conv_gate_tile_check.py'
common = ['--repo', str(a.repo.resolve()), '--source', str((a.output/'source/candidate').resolve()),
          '--baseline', str((a.output/'baseline').resolve()), '--simde-include', str(a.simde_include.resolve())]
cmd = [sys.executable, str(runner)] + common + ['--output', str(a.output/'checks'), '--sanitize', '--tile-on']
commands = [cmd]
with (a.output/'on.log').open('w') as log:
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
env = os.environ.copy()
env.update(NNTR_LFM2_CONV_GATE_TILE='0', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
with (a.output/'off.log').open('w') as log:
    subprocess.run([str((a.output/'checks/check').resolve())], env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
if a.mutations:
    runner = runner.with_name('run_lfm2_conv_gate_tile_mutations.py')
    cmd = [sys.executable, str(runner)] + common + ['--output', str(a.output/'mutants')]
    commands.append(cmd)
    with (a.output/'mutants.log').open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
(a.output/'reproduction.json').write_text(json.dumps({'commands': commands, 'on':112, 'off':112, 'mutations':8 if a.mutations else None}, indent=2)+'\n')
print('REPRODUCTION PASS: actual-source112 ON/OFF' + (' and8mutants' if a.mutations else ''))
