#!/usr/bin/env python3
"""Independently rebuild the four frozen DSP translation units in /tmp."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import shutil
import subprocess

D = Path('/tmp/nntr-lfm2-attn-strided-qprep-20261017')
O = Path('/tmp/nntr-lfm2-attn-strided-qprep-review-20261017')
expected_patch = 'a07332c9043b6760a42392cae06a237319e6a1e383829edc0bee5d838b435df8'
assert hashlib.sha256((D/'production.patch').read_bytes()).hexdigest() == expected_patch
hashes = json.loads((D/'source-hashes.json').read_text())
for rel, record in hashes.items():
    for side in ('baseline', 'candidate'):
        expected = record[side+'_sha256']
        if expected is not None:
            assert hashlib.sha256((D/side/rel).read_bytes()).hexdigest() == expected, (side, rel)

# Freeze the source dependency closure; no root files or author outputs change.
snapshot = O/'compile-snapshot'
for side in ('baseline', 'candidate'):
    for source in (D/side).rglob('*'):
        if source.is_file() and source.suffix in ('.c', '.h'):
            target = snapshot/side/source.relative_to(D/side)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
records = json.loads((D/'codegen/commands.json').read_text())
assert len(records) == 16


def run(record):
    side = record['side']
    out = O/'codegen'/side
    out.mkdir(parents=True, exist_ok=True)
    name = Path(record['source']).stem+'-'+record['mode']
    argv = [x.replace(str(D/'codegen'), str(O/'codegen'))
             .replace(str(D/'baseline'), str(snapshot/'baseline'))
             .replace(str(D/'candidate'), str(snapshot/'candidate'))
            for x in record['command']]
    cwd = record['cwd'].replace(str(D), str(snapshot))
    (out/(name+'-command.json')).write_text(json.dumps({'argv': argv, 'cwd': cwd}, indent=2)+'\n')
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    output = Path(argv[argv.index('-o')+1])
    answer = {'side': side, 'source': record['source'], 'mode': record['mode'],
              'returncode': result.returncode,
              'source_sha256': hashlib.sha256((snapshot/side/record['source']).read_bytes()).hexdigest()}
    if result.returncode == 0:
        answer['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(answer), flush=True)
    return answer


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run, records))
(O/'codegen/results.json').write_text(json.dumps(results, indent=2)+'\n')
assert all(record['returncode'] == 0 for record in results)
for rel, record in hashes.items():
    assert hashlib.sha256((D/'candidate'/rel).read_bytes()).hexdigest() == record['candidate_sha256']
