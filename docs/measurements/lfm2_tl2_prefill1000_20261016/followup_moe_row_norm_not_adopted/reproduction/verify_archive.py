#!/usr/bin/env python3
"""Verify archived files and the exact decompressed bytes of each payload."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = set(manifest['files']) | {'manifest.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if expected != actual:
        raise ValueError('Archive file set differs from manifest')
    for relative, row in manifest['files'].items():
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Non-relative archive target: ' + relative)
        stored = (root / relative).read_bytes()
        original = gzip.decompress(stored) if row['gzip'] else stored
        if len(stored) != row['stored_bytes'] or len(original) != row['uncompressed_bytes']:
            raise ValueError('Archive byte count differs: ' + relative)
        if hashlib.sha256(stored).hexdigest() != row['sha256']:
            raise ValueError('Stored hash differs: ' + relative)
        if hashlib.sha256(original).hexdigest() != row['uncompressed_sha256']:
            raise ValueError('Original hash differs: ' + relative)
    print('ARCHIVE PASS:', len(manifest['files']), 'payloads')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    verify(p.parse_args().archive)
