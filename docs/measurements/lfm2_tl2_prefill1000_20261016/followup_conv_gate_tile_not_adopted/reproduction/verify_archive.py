#!/usr/bin/env python3
"""Verify every archived payload and its decompressed original bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = set(manifest['files']) | {'manifest.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Archive file set differs from manifest')
    for relative, row in manifest['files'].items():
        path = root / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Payload hash mismatch: ' + relative)
        original = gzip.decompress(data) if row['gzip'] else data
        if hashlib.sha256(original).hexdigest() != row['uncompressed_sha256']:
            raise ValueError('Original hash mismatch: ' + relative)
    print('ARCHIVE PASS:', len(manifest['files']), 'payloads')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    verify(p.parse_args().archive)
