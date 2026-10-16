#!/usr/bin/env python3
"""Recreate archived tail-gather host validation using explicit checkout/SDK paths."""
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
    p.add_argument('--repo', type=Path, required=True, help='Checkout with the recorded dependency hashes')
    p.add_argument('--tools-root', type=Path, required=True, help='Hexagon release containing Tools/libnative')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mutations', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Verify/extract sources and record commands without building')
    a = p.parse_args()
    archive, repo, output = a.archive.resolve(), a.repo.resolve(), a.output.resolve()
    manifest = verify(archive)
    if output.exists():
        raise ValueError('Refuse to overwrite an existing output')
    for relative, expected in manifest['reproduction']['dependency_sha256'].items():
        if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Recorded dependency differs: ' + relative)
    output.mkdir(parents=True)
    prefix = 'validation/author/'
    for relative, row in manifest['files'].items():
        if not relative.startswith(prefix):
            continue
        name = relative[len(prefix):].removesuffix('.gz')
        if not name.startswith(('candidate/', 'baseline/')):
            continue
        data = (archive / relative).read_bytes()
        if row['gzip']:
            data = gzip.decompress(data)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    source = output / 'candidate'
    runner = source / 'test/htp/host/run_lfm_tail_gather_checks.py'
    checks = output / 'checks'
    command = [sys.executable, str(runner), '--repo', str(repo), '--source', str(source),
               '--reference', str(output / 'baseline'), '--output', str(checks),
               '--tools-root', str(a.tools_root.resolve()), '--sanitize', '--whole']
    commands = [command]
    if a.mutations:
        commands.append([sys.executable, str(runner.with_name('run_lfm_tail_gather_mutants.py')),
                         '--source', str(source), '--checks', str(checks), '--output', str(output / 'mutants')])
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if a.prepare_only:
        print('PREPARED: exact archived source and commands; no checks executed')
        return
    for i, command in enumerate(commands):
        with (output / ('checks.log' if i == 0 else 'mutants.log')).open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    results = json.loads((checks / 'results.json').read_text())
    if (results['helper_cases'], results['guard_cases'], results['readout_cases'], results['whole_cases']) != (198, 16, 512, 21):
        raise ValueError('Unexpected fixture cohort')
    print('REPRODUCTION PASS: 198 gather + 16 guard + 512 readout + 21 whole-TU' +
          (' and 7 mutants' if a.mutations else ''))


if __name__ == '__main__':
    main()
