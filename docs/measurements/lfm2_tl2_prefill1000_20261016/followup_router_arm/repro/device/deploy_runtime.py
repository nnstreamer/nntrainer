from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

repo = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
evidence = repo / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
variant, parent = sys.argv[1:3]
adb = ['adb', '-s', 'R3CY70LV96T']
dest = '/data/local/tmp/nntr_lfm2_prefill1000_20261016/' + variant
busy = subprocess.check_output(adb + ['shell', 'pidof nntr_causallm nntrainer_causallm || true'], text=True).strip()
assert not busy, ('Deployment deferred while device is busy', busy)
files = [repo / 'builddir/jni/arm64-v8a' / n for n in
         ['libnntrainer.so', 'libccapi-nntrainer.so', 'libc++_shared.so', 'libsdkl.so']]
files += [repo / 'builddir_app/htp' / n for n in ['libcausallm.so', 'nntr_causallm']]
files += [Path(os.environ.get('NNTR_DEPLOY_DSP', repo / 'test/htp/build/libnntr_hvx_skel.so'))]
subprocess.run(adb + ['shell', 'mkdir', '-p', dest], check=True)
manifest = {'parent': parent, 'binaries': {}}
for source in files:
    subprocess.run(adb + ['push', '-Z', str(source), dest + '/' + source.name],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    local = hashlib.sha256(source.read_bytes()).hexdigest()
    remote = subprocess.check_output(adb + ['shell', 'sha256sum', dest + '/' + source.name], text=True).split()[0]
    assert local == remote, source.name
    manifest['binaries'][source.name] = local
subprocess.run(adb + ['shell', 'chmod', '755', dest + '/nntr_causallm'], check=True)
patch = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo)
(Path(__file__).parent / (variant + '.source.patch')).write_bytes(patch)
manifest['source_patch_sha256'] = hashlib.sha256(patch).hexdigest()
manifest['base_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
(evidence / (variant + '-artifacts.json')).write_text(json.dumps(manifest, indent=2) + '\n')
print('Matched runtime deployed and SHA-256 verified:', variant)
