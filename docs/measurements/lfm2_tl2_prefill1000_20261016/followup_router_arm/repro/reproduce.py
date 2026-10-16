#!/usr/bin/env python3
"""Materialize frozen router source/tests; optionally run host checks. No phone I/O."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', action='store_true')
    p.add_argument('--repo', type=Path)
    p.add_argument('--simde-include', type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'Use a fresh output directory'
    subprocess.run([sys.executable, '-B', str(a.archive / 'validation/verify.py'), str(a.archive)], check=True)
    if a.host and (not a.repo or not a.simde_include):
        p.error('--host requires --repo and --simde-include')
    manifest = json.loads((a.archive / 'manifest.json').read_text())
    mappings = {
        'artifacts/production/source/': 'source/',
        'artifacts/tests/source/': 'tests/',
        'artifacts/diagnostic/': 'diagnostic/',
    }
    for name in manifest['files']:
        for prefix, target in mappings.items():
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            if prefix == 'artifacts/diagnostic/' and (
                    relative.startswith(('checks/', 'codegen/')) or
                    relative.endswith(('README.txt', 'validation.json'))):
                continue
            data = (a.archive / name).read_bytes()
            if name.endswith('.gz'):
                relative, data = relative[:-3], gzip.decompress(data)
            dest = a.output / target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    # The diagnostic overlay only changes the app TU. Supply unchanged strict
    # kernel/build files to the materialized diagnostic source tree too.
    for source in (a.output / 'source').rglob('*'):
        if source.is_file():
            target = a.output / 'diagnostic/source' / source.relative_to(a.output / 'source')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if a.host:
        tests = a.output / 'tests/test/htp/host'
        host = a.output / 'host'
        subprocess.run([sys.executable, '-B', str(tests / 'run_lfm2_router_arm_check.py'),
                        '--repo', str(a.repo.resolve()), '--source', str((a.output / 'source').resolve()),
                        '--output', str(host.resolve()), '--simde-include', str(a.simde_include.resolve()),
                        '--sanitize'], check=True)
        subprocess.run([sys.executable, '-B', str(tests / 'run_lfm2_router_arm_mutations.py'),
                        '--checked-output', str(host.resolve()), '--output', str((a.output / 'mutations').resolve())], check=True)
    print('Materialized source/tests at', a.output)


if __name__ == '__main__':
    main()
