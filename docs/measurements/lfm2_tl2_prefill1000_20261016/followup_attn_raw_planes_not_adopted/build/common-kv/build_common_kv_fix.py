#!/usr/bin/env python3
"""Relink the frozen three attention arms with one identical corrected KV TU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'fix-patch', 'fix-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    build = parent / 'build'
    normal = json.loads((build / 'validation.json').read_text())
    diagnostic = json.loads((parent / 'diagnostic/validation.json').read_text())
    records = json.loads((build / 'compilations.json').read_text())
    external = json.loads((build / 'link-tool-input-hashes.json').read_text())
    config = json.loads(args.fix_config.read_text())
    rel = 'nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c'
    assert config['relative_source'] == rel
    assert sha(args.fix_patch) == config['patch_sha256']
    row = next(r for name, r in records.items()
        if name.startswith('baseline-') and r['source'].endswith('/hexkl_kv_q.c'))
    source = Path(row['source'])
    assert sha(source) == config['baseline_source_sha256']
    parent_files = {str(Path(r['object'])): r['object_sha256'] for r in records.values()}
    for arm in ('control', 'candidate'):
        for name in ('libnntr_hvx_skel.so', 'repeat.so'):
            parent_files[str(build / arm / name)] = normal[arm]['sha256']
    parent_files[str(parent / 'diagnostic/libnntr_hvx_skel.so')] = diagnostic['diagnostic_so_sha256']
    diag_object = parent / 'diagnostic/attn-1.o'
    parent_files[str(diag_object)] = sha(diag_object)
    assert all(sha(Path(p)) == h for p, h in parent_files.items())
    assert all(sha(Path(p)) == h for p, h in external.items())
    save(out / 'parent-file-hashes.json', parent_files)
    save(out / 'external-input-hashes.json', external)
    overlay_source = out / 'source' / rel
    overlay_source.parent.mkdir(parents=True)
    overlay_source.write_bytes(source.read_bytes())
    overlay = out / 'overlay.json'
    save(overlay, {'version': 0, 'use-external-names': False, 'roots': [
        {'type': 'file', 'name': str(source), 'external-contents': str(overlay_source)}]})
    commands = []
    work = build / 'source/test/htp'
    def run(command, label, cwd=work):
        proc = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = out / (label + '.log')
        log.write_bytes(proc.stdout)
        commands.append({'label': label, 'argv': command, 'cwd': str(cwd),
                         'returncode': proc.returncode, 'log_sha256': sha(log)})
        save(out / 'commands.json', commands)
        assert proc.returncode == 0, label
        return proc.stdout.decode(errors='replace')
    for label in ('baseline', 'corrected'):
        if label == 'corrected':
            run(['git', 'apply', str(args.fix_patch.resolve())], 'apply-common-fix', out / 'source')
            assert sha(overlay_source) == config['corrected_source_sha256']
        obj = out / ('kv-' + label + '.o')
        cmd = row['cc1'].copy()
        for flag, value in [('-o', obj), ('-dependency-file', obj.with_suffix('.d')), ('-MT', obj)]:
            cmd[cmd.index(flag) + 1] = str(value)
        cmd += ['-ivfsoverlay', str(overlay)]
        run(cmd, 'compile-' + label)
    assert sha(out / 'kv-baseline.o') == row['object_sha256'], 'No-op overlay differs'
    corrected = out / 'kv-corrected.o'
    tool_dir = Path(row['cc1'][0]).parent
    plans = {arm: normal[arm]['link_command'] for arm in ('control', 'candidate')}
    diag_commands = json.loads((parent / 'diagnostic/commands.json').read_text())
    plans['diagnostic'] = next(x['argv'] for x in diag_commands if x['name'] == 'link-1')
    outputs = {}
    active_objects = {}
    known_objects = {Path(r['object']).resolve() for r in records.values()}
    known_objects.update([corrected.resolve(), diag_object.resolve()])
    for arm, original in plans.items():
        assert original.count(row['object']) == 1, (arm, 'Expected exactly one shared KV object')
        destination = out / arm / 'libnntr_hvx_skel.so'
        destination.parent.mkdir()
        cmd = [str(corrected) if x == row['object'] else x for x in original]
        cmd[cmd.index('-o') + 1] = str(destination)
        hashes = []
        for repetition in (1, 2):
            run(cmd, arm + '-link-' + str(repetition))
            hashes.append(sha(destination))
        assert hashes[0] == hashes[1]
        shutil.copy2(destination, destination.with_name('repeat.so'))
        header = run([str(tool_dir / 'hexagon-readelf'), '-h', str(destination)], arm + '-elf')
        assert 'Flags:                             0x81' in header
        dyn = run([str(tool_dir / 'hexagon-readelf'), '--dyn-syms', str(destination)], arm + '-imports')
        imports = []
        for line in dyn.splitlines():
            columns = line.split()
            if len(columns) >= 8 and columns[6] == 'UND':
                imports.append({'symbol': columns[7], 'binding': columns[4]})
        assert imports == normal['control']['imports'], arm
        objects = {}
        for token in cmd:
            if not token.endswith('.o'):
                continue
            path = Path(token)
            path = (work / path).resolve() if not path.is_absolute() else path.resolve()
            assert path.is_file()
            if path in known_objects:
                objects[str(path)] = sha(path)
            else:
                assert external[str(path)] == sha(path), 'Unrecorded toolchain object'
        active_objects[arm] = objects
        outputs[arm] = {'sha256': hashes[0], 'bytes': destination.stat().st_size,
                        'link_command': cmd, 'repeat_identical': True, 'imports': imports}
    # The active 63-object arms differ in their attention object only.
    control, candidate = active_objects['control'], active_objects['candidate']
    common = set(control).intersection(candidate)
    assert len(common) == 62 and len(control) == len(candidate) == 63
    assert all(control[x] == candidate[x] for x in common)
    assert str(corrected) in common
    assert len(set(candidate).intersection(active_objects['diagnostic'])) == 62
    assert all(sha(Path(p)) == h for p, h in parent_files.items())
    assert all(sha(Path(p)) == h for p, h in external.items())
    assert sha(source) == config['baseline_source_sha256']
    for path, expected in row['dependency_sha256'].items():
        assert sha(Path(path)) == expected, path
    shutil.copy2(args.fix_patch, out / 'common-kv-fix.patch')
    save(out / 'active-objects.json', active_objects)
    result = {'status': 'PASS', 'parent_source_revision': normal['source_revision'],
        'parent_normal_validation_sha256': sha(build / 'validation.json'),
        'parent_diagnostic_validation_sha256': sha(parent / 'diagnostic/validation.json'),
        'common_correction': config, 'corrected_common_object_sha256': sha(corrected),
        'no_op_overlay_object_exact': True, 'parent_files_and_external_inputs_unchanged': True,
        'active_objects_per_arm': 63, 'common_objects_between_normal_arms': 62,
        'retained_original_common_objects': 61,
        'normal_arm_difference': 'Only the attention TU; the corrected KV object is identical in all three arms.',
        'production_patch_sha256': normal['source_patch_sha256'], **outputs,
        'scope': 'Local common correctness-fix build. Actual corrected control and candidate checks are still required.'}
    save(out / 'validation.json', result)
    print(json.dumps({arm: outputs[arm]['sha256'] for arm in outputs}, indent=2))

if __name__ == '__main__':
    main()
