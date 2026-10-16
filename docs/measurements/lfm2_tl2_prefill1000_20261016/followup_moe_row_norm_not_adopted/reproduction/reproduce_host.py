#!/usr/bin/env python3
"""Extract exact row-norm fixtures and reproduce host checks with explicit paths."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from verify_archive import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--simde-include', type=Path, required=True, help='External SIMDe 0.8.2 include root')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mutations', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Extract and record commands; no build/run')
    a = p.parse_args()
    archive, repo, output = a.archive.resolve(), a.repo.resolve(), a.output.resolve()
    manifest = verify(archive)
    if output.exists():
        raise ValueError('Refuse to overwrite existing output')
    for name, expected in manifest['reproduction']['dependency_sha256'].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Recorded math dependency differs: ' + name)
    output.mkdir(parents=True)
    prefix = 'validation/author/'
    for relative, row in manifest['files'].items():
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix):].removesuffix('.gz')
        if not name.startswith(('before/', 'candidate/', 'test/')):
            continue
        if name.startswith('test/'):
            name = 'candidate/' + name
        data = (archive / relative).read_bytes()
        if row['gzip']:
            data = gzip.decompress(data)
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    source = output / 'candidate'
    runner = source / 'test/htp/host/run_lfm2_moe_row_norm_check.py'
    command = [sys.executable, str(runner), '--repo', str(repo), '--source', str(source),
               '--simde-include', str(a.simde_include.resolve()), '--output', str(output / 'checks'), '--sanitize']
    commands = [command]
    if a.mutations:
        commands.append([sys.executable, str(runner.with_name('run_lfm2_moe_row_norm_mutations.py')),
                         '--repo', str(repo), '--source', str(source),
                         '--simde-include', str(a.simde_include.resolve()), '--output', str(output / 'mutants')])
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if a.prepare_only:
        print('PREPARED: exact source and commands; no validation executed')
        return
    for index, command in enumerate(commands):
        with (output / ('checks.log' if index == 0 else 'mutants.log')).open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if (output / 'checks.log').read_text().count('89 cases') != 4:
        raise ValueError('Unexpected host fixture cohort')
    print('REPRODUCTION PASS: 89 cases in each of four option environments')


if __name__ == '__main__':
    main()
