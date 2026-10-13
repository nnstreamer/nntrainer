#!/usr/bin/env python3
"""Build only a PUB-derived logits diagnostic; preserve frozen production."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
PUB = Path('/tmp/lfm2-prefill1000-20261013/publish')
BUILD = Path('/tmp/lfm2-prefill1000-20261013/publish-build')
WORK = Path('/tmp/lfm2-prefill1000-20261013')
PARITY = Path('/home/jungone-lee/workspace/nntrainer/docs/measurements/'
              'lfm2_s26_prefill1000_20261013/parity')
patch = WORK / 'parity-v14-diagnostic.patch'
relative = Path('Applications/CausalLM/models/causal_lm.cpp')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


production = json.loads((HERE / 'artifacts.json').read_text())
for name, record in production['artifacts'].items():
    assert digest(Path(record['source'])) == record['sha256'], name
    assert digest(Path(record['frozen'])) == record['sha256'], name
for rel, sha in production['host_source_sha256'].items():
    assert digest(PUB / rel) == sha, rel

source_root = HERE / 'diagnostic-source'
source = source_root / relative
source.parent.mkdir(parents=True, exist_ok=True)
original = (PUB / relative).read_text()
modified = original
lines = patch.read_text().splitlines(True)
hunks = []
i = 0
while i < len(lines):
    if not lines[i].startswith('@@ '):
        i += 1
        continue
    header = lines[i].strip()
    i += 1
    old, new = [], []
    while i < len(lines) and not lines[i].startswith(('@@ ', '--- ', '+++ ')):
        line = lines[i]
        if line.startswith((' ', '-')):
            old.append(line[1:])
        if line.startswith((' ', '+')):
            new.append(line[1:])
        i += 1
    before, after = ''.join(old), ''.join(new)
    assert modified.count(before) == 1, header
    modified = modified.replace(before, after, 1)
    hunks.append(header)
assert len(hunks) == 5, hunks
assert modified.count('auto prefill_work = beginPrefillWork(') == 1
assert modified.count('prefill_work.reset();') == 1
assert modified.index('prefill_work.reset();') < modified.index('auto finish_prefill')
source.write_text(modified)
header = source_root / 'nntrainer/utils/parity_dump.h'
header.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(PARITY / 'source/nntrainer/utils/parity_dump.h', header)

dest = HERE / 'diagnostic-artifacts'
dest.mkdir(exist_ok=False)
for name, record in production['artifacts'].items():
    shutil.copy2(record['frozen'], dest / name)
commands = json.loads((HERE / 'pub-app-commands.json').read_text())
compile_cmd = list(commands['compile'])
old_object = compile_cmd[compile_cmd.index('-o') + 1]
object_path = dest / 'causal_lm.o'
compile_cmd[compile_cmd.index('-c') + 1] = str(source)
compile_cmd[compile_cmd.index('-o') + 1] = str(object_path)
compile_cmd[compile_cmd.index('-MF') + 1] = str(object_path.with_suffix('.o.d'))
if '-MT' in compile_cmd:
    compile_cmd[compile_cmd.index('-MT') + 1] = str(object_path)
compile_cmd.insert(1, '-I' + str(header.parent))
link_cmd = [str(object_path) if arg == old_object else arg
            for arg in commands['link']]
unstripped = dest / 'unstripped-libcausallm_core.so'
link_cmd[link_cmd.index('-o') + 1] = str(unstripped)
objects = [Path(arg) for arg in commands['link'] if arg.endswith('.o')]
assert old_object in commands['link']
assert all(str(path).startswith(str(BUILD / 'app-obj')) for path in objects)
object_sha = {str(path): digest(path) for path in objects}
for kind, command in [('compile', compile_cmd), ('link', link_cmd)]:
    with (dest / (kind + '.log')).open('w') as log:
        subprocess.run(command, cwd=commands['cwd'], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
strip = Path(compile_cmd[0]).with_name('llvm-strip')
subprocess.run([str(strip), '--strip-unneeded', '-o',
                str(dest / 'libcausallm_core.so'), str(unstripped)], check=True)
assert all(digest(Path(path)) == sha for path, sha in object_sha.items())
assert all(digest(Path(record['frozen'])) == record['sha256']
           for record in production['artifacts'].values())
report = {
    'diagnostic_only': True,
    'tps_eligible': False,
    'production_manifest': str(HERE / 'artifacts.json'),
    'patch': str(patch), 'patch_sha256': digest(patch),
    'exact_unique_context_hunks': hunks,
    'prefill_scope_hook_preserved': True,
    'production_source_sha256': digest(PUB / relative),
    'diagnostic_source_sha256': {str(source.relative_to(source_root)): digest(source),
                               str(header.relative_to(source_root)): digest(header)},
    'compile': compile_cmd, 'link': link_cmd, 'cwd': commands['cwd'],
    'unchanged_pub_object_sha256': object_sha,
    'runtime_sha256': {name: digest(dest / name) for name in production['artifacts']},
    'production_artifacts_unchanged': True,
    'deployed_by_agent': False,
}
(HERE / 'diagnostic-build.json').write_text(json.dumps(report, indent=2) + '\n')
print('PUB diagnostic build PASS:', dest)
print('Diagnostic libcausallm_core.so SHA256:', digest(dest / 'libcausallm_core.so'))
