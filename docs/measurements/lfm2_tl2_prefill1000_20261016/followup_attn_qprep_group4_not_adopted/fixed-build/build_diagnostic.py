#!/usr/bin/env python3
"""Build a trace-only quant TU through a VFS overlay; leave frozen inputs intact."""
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    here = Path(__file__).resolve().parent
    build = here / 'build'
    out = here / 'diagnostic'
    out.mkdir(exist_ok=False)
    author = Path('/tmp/nntr-lfm2-attn-strided-qprep-20261017')
    patch = author / 'diagnostic/diagnostic.patch'
    assert sha(patch) == 'af263c745b23b9215a5707f9bc60c008ef2c168ea0880d9b201adddef4871a39'
    validation = json.loads((build / 'validation.json').read_text())
    records = json.loads((build / 'compilations.json').read_text())
    record = next(r for name, r in records.items()
                  if name.startswith('candidate-') and r['source'].endswith('/hvx_quant_u8.c'))
    source = Path(record['source'])
    source_hash = sha(source)
    relative = 'nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c'
    assert source_hash == validation['production_source_sha256'][relative]
    overlay_tree = out / 'source'
    overlay_source = overlay_tree / relative
    overlay_source.parent.mkdir(parents=True)
    overlay_source.write_bytes(source.read_bytes())
    subprocess.run(['git', 'apply', str(patch)], cwd=overlay_tree, check=True)
    (out / 'diagnostic.patch').write_bytes(patch.read_bytes())
    overlay = out / 'overlay.json'
    save(overlay, {'version': 0, 'use-external-names': False,
                  'roots': [{'type': 'file', 'name': str(source),
                             'external-contents': str(overlay_source)}]})
    commands = []
    work = build / 'source/test/htp'

    def run(command, name):
        proc = subprocess.run(command, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = out / (name + '.log')
        log.write_bytes(proc.stdout)
        commands.append({'name': name, 'argv': command, 'cwd': str(work),
                         'returncode': proc.returncode, 'log_sha256': sha(log)})
        save(out / 'commands.json', commands)
        assert proc.returncode == 0, name
        return proc.stdout.decode(errors='replace')

    for enabled in (0, 1):
        obj = out / ('quant-' + str(enabled) + '.o')
        command = record['cc1'].copy()
        for flag, value in [('-o', obj), ('-dependency-file', obj.with_suffix('.d')), ('-MT', obj)]:
            command[command.index(flag) + 1] = str(value)
        command += ['-ivfsoverlay', str(overlay), '-DNNTR_LFM2_QPREP_GROUP4_TRACE=' + str(enabled)]
        run(command, 'compile-' + str(enabled))
    assert sha(out / 'quant-0.o') == record['object_sha256'], 'Macro-OFF object differs'
    command = [str(out / 'quant-1.o') if token == record['object'] else token
               for token in validation['candidate']['link_command']]
    command[command.index('-o') + 1] = str(out / 'libnntr_hvx_skel.so')
    run(command, 'link-1')
    first = sha(out / 'libnntr_hvx_skel.so')
    run(command, 'link-2')
    assert first == sha(out / 'libnntr_hvx_skel.so')
    tools = Path(record['cc1'][0]).parent
    imports = run([str(tools / 'hexagon-nm'), '-u', str(out / 'quant-1.o')], 'object-imports')
    assert '__atomic' not in imports and '__sync' not in imports
    header = run([str(tools / 'hexagon-readelf'), '-h', str(out / 'libnntr_hvx_skel.so')], 'elf')
    assert 'Flags:                             0x81' in header
    dyn = run([str(tools / 'hexagon-readelf'), '--dyn-syms', str(out / 'libnntr_hvx_skel.so')], 'imports')
    imports = []
    for line in dyn.splitlines():
        columns = line.split()
        if len(columns) >= 8 and columns[6] == 'UND':
            imports.append({'symbol': columns[7], 'binding': columns[4]})
    assert imports == validation['candidate']['imports']
    for row in records.values():
        assert sha(Path(row['object'])) == row['object_sha256']
    assert sha(source) == source_hash
    for relative, expected in validation['production_source_sha256'].items():
        assert sha(build / 'source' / relative) == expected
    save(out / 'validation.json', {
        'status': 'PASS', 'production_patch_sha256': validation['source_patch_sha256'],
        'diagnostic_patch_sha256': sha(patch), 'diagnostic_source_sha256': sha(overlay_source),
        'macro_off_object_exact': True, 'diagnostic_so_sha256': first,
        'repeat_link_identical': True, 'runtime_imports_identical': True,
        'source_and_all_objects_unchanged': True, 'arch': 'v81',
        'additional_diagnostic_metadata_bytes': 4,
        'scope': 'One successful synchronous group4 helper per marker. Not normal performance.'})
    print('PASS diagnostic-only one-TU overlay; macro-OFF exact; DSP SHA256', first)


if __name__ == '__main__':
    main()
