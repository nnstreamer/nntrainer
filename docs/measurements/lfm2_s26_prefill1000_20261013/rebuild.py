#!/usr/bin/env python3
"""Rebuild the frozen Android/DSP tree and retain named artifacts and logs."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
WORK = Path('/tmp/lfm2-prefill1000-20261013')
SRC = WORK / 'source'
SDK = Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1')
parser = argparse.ArgumentParser()
parser.add_argument('name')
args = parser.parse_args()
original = json.loads((OUT / 'initial-source-sha256.json').read_text())
extra = [
    'nntrainer/tensor/htp_backend/htp_expert_host_staging.h',
    'test/htp/router_bound_select.h',
    'test/htp/router_sigmoid_bounds.h',
    'test/htp/router_sigmoid_outer_bounds.h',
    'test/htp/conv_norm_device_check.cpp',
    'test/htp/host/router_bounds_reference.h',
    'test/htp/host/router_bounds_host_check.c',
    'test/htp/host/generate_router_sigmoid_bounds.c',
    'test/htp/host/generate_router_sigmoid_outer_bounds.c',
    'test/htp/host/run_router_bounds_check.sh',
    'test/htp/host/prepared_router_host_check.c',
    'test/htp/host/prepared_router_lease_host_check.cpp',
    'test/htp/host/run_prepared_router_host_checks.py',
]
changed = []
for name, digest in original.items():
    current = ROOT / name
    if not current.exists():
        continue
    current_digest = hashlib.sha256(current.read_bytes()).hexdigest()
    if current_digest != digest:
        old = OUT / 'initial-source' / name
        if not old.exists():
            assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == digest, name
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / name, old)
        changed.append(name)
    # Also synchronize a reverted candidate back to the initial version.
    if hashlib.sha256((SRC / name).read_bytes()).hexdigest() != current_digest:
        shutil.copy2(current, SRC / name)
for name in extra:
    assert name not in original, name
    if (ROOT / name).exists():
        (SRC / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, SRC / name)
        changed.append(name)
headers = {}
for directory in ['nntrainer', 'api']:
    for path in (SRC / directory).rglob('*.h'):
        headers.setdefault(path.name, []).append(path)
for path in (WORK / 'include').rglob('*.h'):
    matches = headers.get(path.name, [])
    if len(matches) == 1:
        shutil.copy2(matches[0], path)
# The initial dirty tree already exports qs4cx_tensor.h with this dependency;
# the initial installed Android include tree omitted its new header.
shutil.copy2(SRC / 'nntrainer/tensor/htp_tl2_packing.h',
             WORK / 'include/nntrainer/htp_tl2_packing.h')
subprocess.run([str(SDK / 'ipc/fastrpc/qaic/Ubuntu/qaic'), '-I', str(SDK / 'incs'),
                '-I', str(SDK / 'incs/stddef'), '-mdll', '-o',
                str(SRC / 'test/htp/generated'), str(SRC / 'test/htp/nntr_hvx.idl')],
               check=True)
for name in ['nntr_hvx.h', 'nntr_hvx_stub.c']:
    shutil.copy2(SRC / 'test/htp/generated' / name,
                 SRC / 'nntrainer/tensor/htp_backend/generated' / name)

def run(kind):
    env = dict(os.environ)
    if kind == 'dsp':
        env.update(HEXAGON_SDK_ROOT=str(SDK),
                   DEFAULT_HEXAGON_TOOLS_ROOT=str(SDK / 'tools/HEXAGON_Tools/19.0.04'),
                   HEXKL_ROOT=str(ROOT / 'hexkl_addon'), HEXKL_SDK_VER='6.4.0.2',
                   HEX_ARCH='v81')
        command = ['bash', str(SRC / 'test/htp/build.sh')]
        cwd = ROOT
    else:
        app = kind == 'app'
        env['ANDROID_NDK'] = str(ROOT / 'android-ndk-r30')
        cwd = SRC / 'Applications/CausalLM/jni' if app else ROOT / 'builddir'
        command = [str(ROOT / 'android-ndk-r30/ndk-build'), 'NDK_PROJECT_PATH=.',
                   'APP_BUILD_SCRIPT=' + ('Android.mk' if app else str(WORK / 'Android-core.mk')),
                   'NDK_APPLICATION_MK=' + ('Application.mk' if app else str(WORK / 'Application-core.mk')),
                   f'NDK_OUT={WORK}/{kind}-obj', f'NDK_LIBS_OUT={WORK}/{kind}-libs',
                   'APP_MODULES=' + ('causallm_core nntrainer_causallm' if app else 'nntrainer ccapi-nntrainer'),
                   'APP_ABI=arm64-v8a', '-j8']
    with (OUT / f'{args.name}-{kind}-build.log').open('w') as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    print('BUILD PASS', kind, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, ['dsp', 'core']))
run('app')
dest = OUT / (args.name + '-artifacts')
dest.mkdir(exist_ok=True)
paths = {name: WORK / directory / name for directory, names in [
    ('core-libs/arm64-v8a', ['libnntrainer.so', 'libccapi-nntrainer.so']),
    ('app-libs/arm64-v8a', ['libcausallm_core.so', 'nntrainer_causallm']),
    ('source/test/htp/build', ['libnntr_hvx_skel.so'])] for name in names}
for name, path in paths.items():
    shutil.copy2(path, dest / name)
manifest = {'changed': changed,
            'sources': {name: hashlib.sha256((SRC / name).read_bytes()).hexdigest()
                        for name in list(original) + extra if (SRC / name).exists()},
            'artifacts': {name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
                          for name in paths}}
(OUT / (args.name + '-build-manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
print('FROZEN', args.name, flush=True)
