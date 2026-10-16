#!/usr/bin/env python3
"""Independent read-only input audit and reviewer-output-only relink of QF."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess

Q = Path('/tmp/nntr-lfm2-attn-qprep-fixed-objects-20261017')
B = Q / 'build'
O = Path('/tmp/nntr-lfm2-attn-strided-qprep-review-20261017/fixed-object-review')
D = Path('/tmp/nntr-lfm2-attn-strided-qprep-20261017')
R = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
T = B / 'source'
W = T / 'test/htp'
EXPECTED_PATCH = 'a07332c9043b6760a42392cae06a237319e6a1e383829edc0bee5d838b435df8'
EXPECTED_TUS = {
    'nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c',
    'nntrainer/tensor/htp_backend/hvx/hvx_attn_decode_q.c',
    'nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c',
    'test/htp/nntr_hvx_attn_q.c',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_rel(value):
    path = Path(value)
    path = path if path.is_absolute() else W / path
    return str(path.resolve().relative_to(T))


def normalized_cc1(argv):
    result = []
    it = iter(argv)
    for token in it:
        if token in ('-o', '-dependency-file', '-MT'):
            next(it)
        else:
            result.append(token)
    return result


def elf_sections(path):
    data = Path(path).read_bytes()
    require(data[:7] == b'\x7fELF\x01\x01\x01', 'not ELF32 little endian')
    h = struct.unpack_from('<16sHHIIIIIHHHHHH', data)
    shoff, shsize, shnum, names = h[6], h[11], h[12], h[13]
    headers = [struct.unpack_from('<10I', data, shoff + i * shsize) for i in range(shnum)]
    nameh = headers[names]
    nametable = data[nameh[4]:nameh[4] + nameh[5]]
    result = {}
    for row in headers:
        name = nametable[row[0]:].split(b'\0', 1)[0].decode()
        result[name] = {'type': row[1], 'flags': row[2], 'address': row[3],
                        'size': row[5], 'alignment': row[8],
                        'data_sha256': None if row[1] == 8 else hashlib.sha256(
                            data[row[4]:row[4] + row[5]]).hexdigest()}
    return result


def run(argv, label):
    p = subprocess.run(argv, cwd=W, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=False)
    (O / (label + '.log')).write_text(p.stdout)
    (O / (label + '.command.json')).write_text(json.dumps(argv, indent=2) + '\n')
    require(p.returncode == 0, label + ' failed')
    return p.stdout


def main():
    O.mkdir(exist_ok=True)
    config = read(Q / 'candidate-config.json')
    v = read(B / 'validation.json')
    inputs = read(B / 'inputs.json')
    compilations = read(B / 'compilations.json')
    commands = read(B / 'commands.json')
    frozen = read(B / 'frozen-object-hashes.json')
    external = read(B / 'link-tool-input-hashes.json')
    initial = read(B / 'baseline-source-hashes.json')
    entries = read(B / 'object-map.json')
    require(digest(D / 'production.patch') == digest(B / 'production.patch') == EXPECTED_PATCH,
            'production patch hash')
    changed = config['candidate_source_sha256']
    require(len(changed) == 5 and set(config['affected_translation_units']) == EXPECTED_TUS,
            'unexpected source/TU configuration')
    require(all(digest(T / rel) == digest(D / 'candidate' / rel) == h
                for rel, h in changed.items()), 'source map mismatch')
    require(len(frozen) == len(entries) == 63, 'baseline count')
    require(all(digest(p) == h for p, h in frozen.items()), 'baseline object drift')
    require(all(digest(p) == h for p, h in external.items()), 'external dependency drift')
    gcc_shared = Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HEXAGON_Tools/19.0.04/Tools/target/hexagon/lib/v81/G0/pic/libgcc.so')
    gcc_shared_hash = digest(gcc_shared)
    require(all(c.get('returncode') == 0 and digest(B / c['log']) == c['log_sha256']
                for c in commands), 'recorded command/log mismatch')
    require(all(digest(row['object']) == row['object_sha256']
                for row in compilations.values()), 'compile object drift')
    require(all(digest(W / p) == h for p, h in read(B / 'generated-hashes.json').items()),
            'generated ABI drift')
    revision = inputs['revision']
    require(revision == '742735f3c61803fac51189012ba4caf588a3bd50', 'baseline revision')
    original_tracked = subprocess.check_output(
        ['git', '-C', str(R), 'ls-tree', '-r', '--name-only', revision,
         'test/htp', 'nntrainer/tensor'], text=True).splitlines()
    for rel in original_tracked:
        raw = subprocess.check_output(['git', '-C', str(R), 'show', revision + ':' + rel])
        require(hashlib.sha256(raw).hexdigest() == initial[rel], 'git baseline differs: ' + rel)
    actual_sources = {str(p.relative_to(T)): digest(p) for p in T.rglob('*')
                      if p.is_file() and not str(p.relative_to(T)).startswith(
                          ('test/htp/build/', 'test/htp/generated/'))}
    require(set(actual_sources) == set(initial) | set(changed), 'unexpected source files')
    require({p for p in actual_sources if actual_sources[p] != initial.get(p)} == set(changed),
            'source changes outside five-file patch')
    affected = set()
    for name, row in compilations.items():
        for dependency, h in row['dependency_sha256'].items():
            p = Path(dependency)
            rel = str(p.relative_to(T)) if p.is_relative_to(T) else None
            if rel in changed:
                if name.startswith('baseline-'):
                    affected.add(source_rel(row['source']))
                    require(h == initial[rel], 'baseline dependency wrong: ' + rel)
                else:
                    require(h == changed[rel], 'candidate dependency wrong: ' + rel)
            else:
                require(digest(p) == h, 'unchanged compilation dependency drift: ' + dependency)
    require(affected == EXPECTED_TUS, 'compiler dependency closure')
    candidate_rows = {source_rel(r['source']): r for name, r in compilations.items()
                      if name.startswith('candidate-')}
    require(set(candidate_rows) == EXPECTED_TUS, 'candidate TU count')
    baseline_rows = {source_rel(r['source']): r for name, r in compilations.items()
                     if name.startswith('baseline-')}
    for rel, row in candidate_rows.items():
        require(normalized_cc1(row['cc1']) == normalized_cc1(baseline_rows[rel]['cc1']),
                'candidate cc1 flags differ: ' + rel)
    for label in ('ordinary-driver-plan', 'ieee-plan-0', 'ieee-plan-1'):
        plans = [shlex.split(line.strip()) for line in (B / 'logs' / (label + '.log')).read_text().splitlines()
                 if line.lstrip().startswith('"')]
        for plan in plans:
            if '-cc1' not in plan:
                continue
            rel = source_rel(plan[-1])
            require(normalized_cc1(plan) == normalized_cc1(baseline_rows[rel]['cc1']),
                    'baseline cc1 differs from driver: ' + rel)
    control_cmd = v['control']['link_command']
    candidate_cmd = v['candidate']['link_command']
    require(len(control_cmd) == len(candidate_cmd), 'link option count')
    changes = [(a, b) for a, b in zip(control_cmd, candidate_cmd) if a != b]
    require(len(changes) == 4, 'link has more than four substituted TUs')
    require({b for _, b in changes} == {r['object'] for r in candidate_rows.values()},
            'unexpected candidate object substitution')
    require(all(a in frozen for a, _ in changes), 'substituted baseline object missing')
    require(control_cmd.count('-no-threads') == candidate_cmd.count('-no-threads') == 1,
            'repeat link mode')
    require(len({a for a, b in zip(control_cmd, candidate_cmd) if a == b and
                 str((W / a).resolve() if not Path(a).is_absolute() else Path(a)) in frozen}) == 59,
            'common link object count')
    readelf = str(Path(control_cmd[0]).with_name('hexagon-readelf'))
    imports = {}
    relink_hashes = {}
    for label, original in (('control', control_cmd), ('candidate', candidate_cmd)):
        image = B / label / 'libnntr_hvx_skel.so'
        require(digest(image) == digest(B / label / 'repeat.so') == v[label]['sha256'],
                'recorded image or repeat drift')
        hashes = []
        section_results = []
        target = O / (label + '-relink.so')
        for repeat in (1, 2):
            cmd = original.copy()
            cmd[cmd.index('-o') + 1] = str(target)
            run(cmd, label + '-relink-' + str(repeat))
            hashes.append(digest(target))
            shutil.copyfile(target, O / (label + '-repeat-' + str(repeat) + '.so'))
            actual_sections, original_sections = elf_sections(target), elf_sections(image)
            require(set(actual_sections) == set(original_sections), 'different section names')
            differs = [name for name in actual_sections if actual_sections[name] != original_sections[name]]
            require(differs == ['.comment'], 'non-comment section differs: ' + str(differs))
            require(not (actual_sections['.comment']['flags'] & 2), 'comment unexpectedly loaded')
            section_results.append({'only_changed_section': '.comment',
                                    'alloc_sections_identical': True,
                                    'noncomment_sections_identical': True})
        require(hashes[0] == hashes[1], 'same-output independent relink is not repeatable')
        relink_hashes[label] = {'same_output_repeat_sha256': hashes,
                                'deployed_image_sha256': v[label]['sha256'],
                                'different_output_name_comparison': section_results}
        text = run([readelf, '--dyn-syms', str(image)], label + '-dynsym')
        symbols = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 8 and fields[6] == 'UND':
                symbols.append({'symbol': fields[7], 'binding': fields[4]})
        require(symbols and symbols == v[label]['imports'], 'independent import mismatch')
        require(not any(s['symbol'].startswith(('nntr_', 'hvx_', 'hexkl_')) for s in symbols),
                'unresolved project import')
        require(all(not s['symbol'].startswith('dspqueue_') or s['binding'] == 'WEAK'
                    for s in symbols), 'strong optional DSPQ import')
        imports[label] = symbols
        header = run([readelf, '-h', str(image)], label + '-elf-header')
        require(re.search(r'Flags:\s+0x81\b', header), 'not v81')
    require(imports['control'] == imports['candidate'], 'different runtime imports')
    require(all(digest(p) == h for p, h in frozen.items()), 'post-review object drift')
    require(digest(gcc_shared) == gcc_shared_hash, 'shared libgcc changed during review')
    result = {
        'status': 'PASS', 'source_revision': revision, 'patch_sha256': EXPECTED_PATCH,
        'five_source_hashes_verified': changed, 'dependency_closure': sorted(affected),
        'baseline_objects': 63, 'common_objects': 59, 'candidate_recompiled': 4,
        'baseline_git_source_files_checked': len(original_tracked),
        'external_hashes_checked': len(external), 'compile_records_checked': len(compilations),
        'cc1_options_identical_except_output_and_dependency_metadata': True,
        'independent_relink_sha256': relink_hashes,
        'additional_shared_link_dependency': {'path': str(gcc_shared),
                                              'sha256': gcc_shared_hash,
                                              'not_in_original_archive_only_manifest': True},
        'runtime_imports_identical': True, 'runtime_import_count': len(imports['control']),
        'notes': ['Reviewer writes only under own output directory.',
                  'Checks object/input/ABI closure; not original 3b image identity or device performance.',
                  'Changing output path changes only nonallocated .comment and section header offsets; same-path repeats are fully identical.',
                  'Compiler build.sh flags also manually compared with the frozen script.'],
    }
    (O / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
