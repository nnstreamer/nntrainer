#!/usr/bin/env python3
"""Extract frozen ring sources; optionally run their existing Linux host checks."""
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, help='External source-equivalent repo; required for optional whole-M1 checks')
    p.add_argument('--hexagon-tools', type=Path, help='Tools/libnative root; enables optional whole-M1 checks')
    p.add_argument('--mutations', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    archive, output = a.archive.resolve(), a.output.resolve()
    manifest = verify(archive)
    if output.exists():
        raise ValueError('Refuse to overwrite output')
    if a.hexagon_tools and not a.repo:
        p.error('--hexagon-tools requires --repo')
    if a.hexagon_tools:
        dependencies = json.loads((archive / 'reproduction/repo-dependencies.json').read_text())
        for rel, expected in dependencies.items():
            actual = hashlib.sha256((a.repo / rel).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('External fixture dependency differs: ' + rel)
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
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    candidate, baseline = output / 'candidate', output / 'baseline'
    runner = candidate / 'test/htp/host/run_lfm2_m1_ring_reset_checks.py'
    cmd = [sys.executable, str(runner), '--repo', str(a.repo.resolve() if a.hexagon_tools else candidate),
           '--source', str(candidate), '--reference', str(baseline), '--output', str(output / 'checks'), '--sanitize']
    if a.hexagon_tools:
        cmd += ['--hexagon-tools', str(a.hexagon_tools.resolve())]
    if a.mutations:
        cmd += ['--mutations']
    (output / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    if a.prepare_only:
        print('PREPARED: exact archived source; no fixtures executed')
        return
    with (output / 'host.log').open('w') as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    print('PASS: frozen actual-ring fixture' + (' and whole-M1 caller' if a.hexagon_tools else ''))


if __name__ == '__main__':
    main()
