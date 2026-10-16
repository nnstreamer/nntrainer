#!/usr/bin/env python3
"""Execute actual recency-output blocks with the unchanged real ExpertLru."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def block(text, name, offset=0):
    start = text.index(name, offset)
    first = text.index('{', start)
    depth, end = 1, first + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end] + '\n'


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sanitize', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
folder = Path('Applications/CausalLM/models/lfm2_moe')
manifest = {'sources': {}, 'extracts': {}}
for label, root in (('candidate', a.source), ('reference', a.reference)):
    path = root / folder / 'lfm2_moe_layer.cpp'
    source = path.read_text()
    begin = source.index('void Lfm2MoELayer::buildExpertAssignments(')
    method = block(source, 'void Lfm2MoELayer::buildExpertAssignments(', begin)
    extra = block(method, 'if (extra_top_k) {')
    (a.output / (label + '_hint.inc')).write_text(extra)
    manifest['sources'][label] = hashlib.sha256(source.encode()).hexdigest()
    manifest['extracts'][label] = hashlib.sha256(extra.encode()).hexdigest()
header = a.source / folder / 'expert_lru.h'
assert header.read_bytes() == (a.reference / folder / 'expert_lru.h').read_bytes()
shutil.copy2(header, a.output / 'expert_lru.h')
shutil.copy2(Path(__file__).with_name('lru_stress.cpp'), a.output)
command = ['g++', '-std=c++17', '-O1', '-g', '-pthread', '-Wall', '-Wextra',
           '-Werror', '-I' + str(a.output.resolve())]
if a.sanitize:
    command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
command += [str((a.output / 'lru_stress.cpp').resolve()), '-o', str((a.output / 'check').resolve())]
manifest['command'] = command
manifest['expert_lru_sha256'] = hashlib.sha256(header.read_bytes()).hexdigest()
(a.output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
with (a.output / 'build.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
r = subprocess.run([str((a.output / 'check').resolve())], env=env, text=True, capture_output=True)
(a.output / 'run.log').write_text(r.stdout + r.stderr)
print(r.stdout + r.stderr, end='')
r.check_returncode()
