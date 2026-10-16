#!/usr/bin/env python3
"""Compare ELF32 function bytes, separately from relocations/link addresses."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def functions(path):
    raw = path.read_bytes()
    assert raw[:6] == b'\x7fELF\x01\x01', 'expected little-endian ELF32'
    h = struct.unpack_from('<16sHHIIIIIHHHHHH', raw)
    sections = [struct.unpack_from('<IIIIIIIIII', raw, h[6] + i*h[11])
                for i in range(h[12])]
    out = {}
    for section in sections:
        if section[1] != 2:
            continue
        string_section = sections[section[6]]
        strings = raw[string_section[4]:string_section[4]+string_section[5]]
        for off in range(section[4], section[4]+section[5], section[9]):
            name, value, size, info, other, index = struct.unpack_from('<IIIBBH', raw, off)
            if info & 15 != 2 or not size or index >= len(sections):
                continue
            name = strings[name:strings.index(b'\x00', name)].decode()
            text = sections[index]
            data = raw[text[4]+value-text[3]:text[4]+value-text[3]+size]
            assert len(data) == size
            out[name] = {'bytes': size, 'sha256': hashlib.sha256(data).hexdigest()}
    return out


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
old, new = functions(a.baseline), functions(a.candidate)
rows = [{'name': name, 'before': old[name], 'after': new.get(name),
         'same_function_bytes': old[name] == new.get(name)} for name in old]
result = {'scope': 'Relocatable function bytes; not final linked addresses or performance.',
          'baseline_object_sha256': hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
          'candidate_object_sha256': hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
          'common_functions': rows,
          'new_functions': {name: new[name] for name in new.keys()-old.keys()}}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'common': len(rows), 'same': sum(x['same_function_bytes'] for x in rows),
                  'changed': [x['name'] for x in rows if not x['same_function_bytes']],
                  'new': list(result['new_functions'])}))
