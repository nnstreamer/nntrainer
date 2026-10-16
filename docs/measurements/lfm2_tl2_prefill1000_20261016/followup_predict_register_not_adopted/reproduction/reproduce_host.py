#!/usr/bin/env python3
"""Extract exact prediction fixtures and reproduce host checks with explicit paths."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from verify_archive import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--simde-include', type=Path, required=True, help='External SIMDe 0.8.2 include root')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mutations', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    archive, output = a.archive.resolve(), a.output.resolve()
    manifest = verify(archive)
    if output.exists():
        raise ValueError('Refuse to overwrite output')
    output.mkdir(parents=True)
    prefix = 'validation/author/'
    for relative, row in manifest['files'].items():
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix):].removesuffix('.gz')
        if not name.startswith(('baseline/', 'candidate/', 'test/')):
            continue
        if name.startswith('test/'):
            name = 'candidate/' + name
        data = (archive / relative).read_bytes()
        if row['gzip']:
            data = gzip.decompress(data)
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    source, reference = output / 'candidate', output / 'baseline'
    runner = source / 'test/htp/host/run_lfm2_predict_register_check.py'
    base = [sys.executable, str(runner), '--source', str(source), '--reference', str(reference)]
    commands = [base + ['--simde-include', str(a.simde_include.resolve()), '--output', str(output / 'host'), '--sanitize'],
                base + ['--fallback', '--output', str(output / 'fallback'), '--sanitize']]
    if a.mutations:
        commands.append([sys.executable, str(runner.with_name('run_lfm2_predict_register_mutations.py')),
                         '--source', str(source), '--reference', str(reference),
                         '--simde-include', str(a.simde_include.resolve()), '--output', str(output / 'mutants')])
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if a.prepare_only:
        print('PREPARED: exact archived source and commands; no tests executed')
        return
    for name, command in zip(('host', 'fallback', 'mutants'), commands):
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    print('REPRODUCTION PASS: actual-header/callback and fallback fixtures')


if __name__ == '__main__':
    main()
