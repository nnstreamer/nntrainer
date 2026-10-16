#!/usr/bin/env python3
"""Verify portable payload hashes and recompute normal/diagnostic raw readback."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
sys.dont_write_bytecode = True
from readback import audit


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = set(manifest['files']) | {'manifest.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if expected != actual:
        raise ValueError('Archive file set differs from manifest')
    payloads = {}
    for relative, row in manifest['files'].items():
        for name in (relative, row['logical_path']):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Non-relative archive target: ' + name)
        stored = (root / relative).read_bytes()
        original = gzip.decompress(stored) if row['gzip'] else stored
        if len(stored) != row['stored_bytes'] or len(original) != row['uncompressed_bytes']:
            raise ValueError('Archive byte count differs: ' + relative)
        if hashlib.sha256(stored).hexdigest() != row['sha256']:
            raise ValueError('Stored hash differs: ' + relative)
        if hashlib.sha256(original).hexdigest() != row['uncompressed_sha256']:
            raise ValueError('Original hash differs: ' + relative)
        if row['logical_path'] in payloads:
            raise ValueError('Duplicate logical path')
        payloads[row['logical_path']] = original
    result = audit(payloads.__getitem__)
    if result != json.loads(payloads['summary.json']):
        raise ValueError('Independent raw readback differs from summary.json')
    if manifest['decision'] != result['decision']:
        raise ValueError('Decision differs')
    print('ARCHIVE PASS:', len(manifest['files']), 'payloads; normal12 + diagnostic2 raw readback; 65 IDs each; whole memory; original seven restored')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    verify(p.parse_args().archive.resolve())
