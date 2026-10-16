import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

E = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000/docs/measurements/lfm2_tl2_prefill1000_20261016')
T = Path(__file__).parent
adb = ['adb', '-s', 'R3CY70LV96T']
D = '/data/local/tmp/nntr_lfm2_prefill1000_20261016'
variant, tag = sys.argv[1:3]
model = sys.argv[3] if len(sys.argv) > 3 else 'model'
if model not in ('model', 'model2048'):
    raise ValueError(model)
names = 'nntr_causallm nntrainer_causallm repeat_prompt_check lfm2_snapshot_check'
for _ in range(120):
    if not subprocess.check_output(adb + ['shell', f'pidof {names} || true'], text=True).strip():
        time.sleep(5)
        if not subprocess.check_output(adb + ['shell', f'pidof {names} || true'], text=True).strip():
            break
    time.sleep(1)
else:
    raise RuntimeError('Device remains busy')
reference = json.loads((E / 'validation/lfm-repeat.json').read_text())
if model == 'model2048':
    reference['ids_sha256'] = '2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'
env = reference['environment']
env['LD_LIBRARY_PATH'] = D + '/' + variant
env['ADSP_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']
env['NNTR_REPEAT_PREFIX'] = D + '/' + tag
env['NNTR_LFM2_DECODE_DEMAND_HANDOFF'] = '1'
command = shlex.join(['env'] + [k + '=' + v for k, v in env.items()] +
                     ['timeout', '120', D + ('/fixture-support2048/repeat_prompt_check' if model == 'model2048' else '/fixture-support/repeat_prompt_check'), D + '/' + model])
with (T / (tag + '.log')).open('wb') as log:
    subprocess.run(adb + ['shell', command], stdout=log, stderr=subprocess.STDOUT, check=True)
result = json.loads(subprocess.check_output(adb + ['shell', 'cat', D + '/' + tag + '.summary.json'], text=True))
for run in result['runs']:
    ids = subprocess.check_output(adb + ['shell', 'cat', run['ids']])
    assert hashlib.sha256(ids).hexdigest() == reference['ids_sha256']
    (T / (tag + '-run' + str(run['run']) + '.ids')).write_bytes(ids)
result.update(environment=env, ids_match_normal_run=True, ids_sha256=reference['ids_sha256'])
(T / (tag + '.json')).write_text(json.dumps(result, indent=2) + '\n')
print(variant, model, 'same-handle 2 runs: all65 IDs exact PASS')
