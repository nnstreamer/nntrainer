#!/usr/bin/env python3
"""Rebuild the complete composite runtime and a current-object diagnostic."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
PUB = Path('/tmp/lfm2-prefill1000-20261013/publish')
BUILD = Path('/tmp/lfm2-prefill1000-20261013/publish-build')
SDK = Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1')
ROOT = Path('/home/jungone-lee/workspace/nntrainer')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def run(kind, command, cwd, env=None):
    start = time.monotonic()
    log = HERE / (kind + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output,
                                stderr=subprocess.STDOUT)
    results.append(dict(kind=kind, command=command, cwd=str(cwd),
                        returncode=result.returncode,
                        seconds=time.monotonic() - start, log=str(log)))
    dump('matched-install-results.json', results)
    print(kind, result.returncode, flush=True)
    result.check_returncode()


commands = json.loads((HERE / 'matched-install-commands.json').read_text())
applied = json.loads((HERE / 'applied-source.json').read_text())
for relative, expected in applied['files'].items():
    assert sha(PUB / relative) == expected, relative
results = []
environment = os.environ.copy()
environment.update(HEXAGON_SDK_ROOT=str(SDK),
                   DEFAULT_HEXAGON_TOOLS_ROOT=str(SDK / 'tools/HEXAGON_Tools/19.0.04'),
                   HEXKL_ROOT=str(ROOT / 'hexkl_addon'),
                   HEXKL_SDK_VER='6.4.0.2', HEX_ARCH='v81')
dump('dsp-environment.json', {k: environment[k] for k in (
    'HEXAGON_SDK_ROOT', 'DEFAULT_HEXAGON_TOOLS_ROOT', 'HEXKL_ROOT',
    'HEXKL_SDK_VER', 'HEX_ARCH')})
run('stub-generation', ['bash', 'nntrainer/tensor/htp_backend/generate_stub.sh'],
    PUB, environment)
run('dsp-build', ['bash', 'test/htp/build.sh'], PUB, environment)

# Refresh the headers touched by this topic AND the discarded aligned-I/O
# topic. Never let an installed v18 header survive into the rebuilt v19 app.
exports = {}
for relative in ('nntrainer/utils/weight_direct_file.h',
                 'nntrainer/tensor/htp_backend/htp_expert_host_staging.h',
                 'nntrainer/tensor/htp_backend/htp_rope_prefill.h',
                 'nntrainer/tensor/cpu_backend/compute_ops.h',
                 'nntrainer/tensor/htp_backend/generated/nntr_hvx.h'):
    source = PUB / relative
    for prefix in ('include', 'include/nntrainer'):
        dest = PUB / 'builddir/android_build_result' / prefix / source.name
        shutil.copy2(source, dest)
        assert sha(source) == sha(dest)
        exports[str(dest)] = dict(source=str(source), sha256=sha(dest))
dump('exported-headers.json', exports)

before = {str(p): sha(p) for directory in ('core-obj', 'app-obj')
          for p in (BUILD / directory).rglob('*.o')}
for kind in ('core', 'app'):
    entry = commands[kind]
    run(kind + '-matched-install', entry['command'], entry['cwd'])
    if kind == 'core':
        installed = PUB / 'builddir/android_build_result/lib/arm64-v8a'
        for name in ('libnntrainer.so', 'libccapi-nntrainer.so',
                     'libc++_shared.so', 'libsdkl.so'):
            source = BUILD / 'core-libs/arm64-v8a' / name
            if not (installed / name).exists() or not source.samefile(installed / name):
                shutil.copy2(source, installed / name)
            assert sha(source) == sha(installed / name)
for name in ('libnntrainer.so', 'libccapi-nntrainer.so', 'libc++_shared.so'):
    assert sha(BUILD / 'core-libs/arm64-v8a' / name) == sha(BUILD / 'app-libs/arm64-v8a' / name)
after = {str(p): sha(p) for directory in ('core-obj', 'app-obj')
         for p in (BUILD / directory).rglob('*.o')}
dump('object-changes.json', dict(before=before, after=after,
     changed=[p for p, digest in after.items() if before.get(p) != digest]))

artifacts = {}
dest = HERE / 'artifacts'
dest.mkdir(exist_ok=False)
for name in ('libnntrainer.so', 'libccapi-nntrainer.so', 'libcausallm_core.so',
             'nntrainer_causallm', 'libc++_shared.so', 'libsdkl.so',
             'libnntr_hvx_skel.so'):
    if name == 'libnntr_hvx_skel.so':
        source = PUB / 'test/htp/build' / name
    elif name in ('libcausallm_core.so', 'nntrainer_causallm'):
        source = BUILD / 'app-libs/arm64-v8a' / name
    else:
        source = BUILD / 'core-libs/arm64-v8a' / name
    frozen = dest / name
    shutil.copy2(source, frozen)
    artifacts[name] = dict(source=str(source), frozen=str(frozen),
                           bytes=frozen.stat().st_size, sha256=sha(frozen))

# Retain dependency content hashes, including the reused, unchanged objects.
# This records root vendored/generated dependencies read by the Android build.
dependencies = {}
for directory in ('core-obj', 'app-obj'):
    for dep in (BUILD / directory).rglob('*.d'):
        content = dep.read_text().replace('\\\n', ' ')
        for token in shlex.split(content):
            if token.endswith(':'):
                continue
            path = Path(token)
            if path.is_absolute() and path.is_file():
                dependencies[str(path)] = sha(path)
dump('android-dependencies.json', dependencies)
paths = subprocess.check_output(
    ['git', 'ls-files', 'nntrainer', 'api', 'Applications/CausalLM', 'test/htp'],
    cwd=PUB, text=True).splitlines()
paths += list(applied['files'])
source_sha = {name: sha(PUB / name) for name in sorted(set(paths))
              if (PUB / name).is_file()}
dump('source-manifest.json', source_sha)
for relative, expected in applied['files'].items():
    assert source_sha[relative] == expected, relative
report = dict(publication_base=applied['base_commit'],
              publication_head_at_freeze=subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], cwd=PUB, text=True).strip(),
              production_only=True, host_source_sha256=source_sha,
              artifacts=artifacts, build_results=results, exported_headers=exports,
              dsp=dict(rebuilt=True, registry_limit=2048, sdk='6.4.0.1',
                       tools='19.0.04', hexkl='6.4.0.2', arch='v81'),
              device_deployed_by_agent=False, commit_by_agent=False,
              root_live_changes_by_agent=False)
dump('artifacts.json', report)
run('diagnostic-build', ['python3', str(HERE / 'build_publication_diagnostic.py')],
    HERE)
shutil.copy2(HERE / 'diagnostic-build.json', HERE / 'diagnostic-artifacts.json')
print('PASS: matching PUB runtime and current-object diagnostic frozen', flush=True)
