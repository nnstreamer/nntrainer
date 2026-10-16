import json
from pathlib import Path
import subprocess
import sys

evidence = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000/docs/measurements/lfm2_tl2_prefill1000_20261016')
for job in json.loads(Path(sys.argv[1]).read_text()):
    variant, tag, model = (job[k] for k in ['variant', 'tag', 'model'])
    command = [sys.executable, str(evidence / 'bench.py'), variant, tag, '--model', model, '--memory']
    if job.get('profile'):
        command.append('--profile')
    for key, value in job.get('env', {}).items():
        command.extend(['--env', key + '=' + str(value)])
    subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    result = json.loads((evidence / (tag + '.json')).read_text())
    print({k: result[k] for k in ['tag', 'prefill_tps', 'decode_tps', 'full_run_sampled_peak_kib', 'ids_match_reference']}, flush=True)
    for marker in job.get('require_log', []):
        if marker not in (evidence / (tag + '.log')).read_text():
            raise RuntimeError('Expected runtime path did not execute: ' + marker)
