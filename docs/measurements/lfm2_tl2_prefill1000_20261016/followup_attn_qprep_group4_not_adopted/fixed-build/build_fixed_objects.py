#!/usr/bin/env python3
"""Freeze one DSP object set and rebuild only audited affected production TUs.

Uses the compiler driver's -### plan for the recorded build.sh command: actual
cc1 flags and link order are retained, replacing temporary object output paths.
No checkout, SDK, or phone is modified. Output must be a new directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile

PATCH_SHA = ''
REL = 'nntrainer/tensor/htp_backend/'
CHANGED = []
CANDIDATE = {}
CHANGED_TUS = []
ALLOW = re.compile(r'^(HAP_|compute_resource_|qurt_|dspqueue_|__hexagon_|__extendhfsf2$|__cxa_finalize$|__register_frame_info_bases$|malloc$|free$|calloc$|memalign$|memcpy$|memset$|lroundf$|nearbyintf$|snprintf$|vsnprintf$|strlcpy$|rintf$|sqrtf$|memcmp$|__truncsfhf2$|ceil$|ldexp$|ldexpf$|lround$|_Log$)')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replace_output(command, output):
    result = command.copy()
    result[result.index('-o') + 1] = str(output)
    return result


def main():
    global PATCH_SHA, CHANGED, CANDIDATE, CHANGED_TUS
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'candidate-patch', 'candidate-config', 'sdk-root', 'tools-root', 'hexkl-root', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--hexkl-sdk-ver', default='6.4.0.2')
    p.add_argument('--hex-arch', choices=['v81'], default='v81')
    p.add_argument('--extra-cflag', action='append', default=[])
    p.add_argument('--resume-baseline', action='store_true', help='Resume this frozen baseline after link-only validation failure')
    a = p.parse_args()
    config = json.loads(a.candidate_config.read_text())
    PATCH_SHA = config['patch_sha256']
    CANDIDATE = config['candidate_source_sha256']
    CHANGED = sorted(CANDIDATE)
    CHANGED_TUS = sorted(config['affected_translation_units'])
    require(CHANGED_TUS and all(x.endswith('.c') for x in CHANGED_TUS), 'Missing audited TU list')
    repo, sdk, tools, hexkl, out = [getattr(a, n).resolve() for n in
                                  ('repo', 'sdk_root', 'tools_root', 'hexkl_root', 'output')]
    require(out.is_dir() if a.resume_baseline else not out.exists(), 'Output existence/resume mismatch')
    require(sha(a.candidate_patch) == PATCH_SHA, 'Unexpected production patch')
    require(not a.extra_cflag, 'Frozen experiment has no extra compilation flags')
    out.mkdir(parents=True, exist_ok=a.resume_baseline)
    tree = out / 'source'
    tree.mkdir(exist_ok=a.resume_baseline)
    work = tree / 'test/htp'
    backend = tree / REL
    compiler = tools / 'Tools/bin/hexagon-clang'
    readelf = tools / 'Tools/bin/hexagon-readelf'
    commands = json.loads((out / 'commands.json').read_text()) if a.resume_baseline else []
    entries, dependencies = [], {}

    def run(argv, name, cwd=None, capture=False):
        cwd = cwd or work
        row = {'name': name, 'cwd': str(cwd), 'argv': [str(x) for x in argv]}
        commands.append(row)
        write(out / 'commands.json', commands)
        proc = subprocess.run(row['argv'], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = out / 'logs' / (name + '.log')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(proc.stdout)
        row.update(returncode=proc.returncode, log=str(path.relative_to(out)), log_sha256=sha(path))
        write(out / 'commands.json', commands)
        require(proc.returncode == 0, 'Command failed: ' + name + ' (see ' + str(path) + ')')
        return proc.stdout.decode(errors='replace') if capture else None

    def parse_plan(argv, name):
        raw = run(argv + ['-###'], name, capture=True)
        parsed = [shlex.split(line.strip()) for line in raw.splitlines() if line.lstrip().startswith('"')]
        require(parsed and all('-cc1' in c or Path(c[0]).name in ('hexagon-link', 'ld.lld') for c in parsed),
                'Unrecognized compiler driver plan')
        return parsed

    def compile_entry(original, obj, name, source):
        obj.parent.mkdir(parents=True, exist_ok=True)
        dep = obj.with_suffix('.d')
        command = replace_output(original, obj)
        # Dependency emission adds metadata only, without changing codegen flags.
        command += ['-dependency-file', str(dep), '-MT', str(obj)]
        run(command, name)
        dep_text = dep.read_text().replace('\\\n', ' ')
        dep_files = shlex.split(dep_text.split(':', 1)[1])
        inputs = {}
        for item in dep_files:
            path = Path(item)
            path = (work / path).resolve() if not path.is_absolute() else path.resolve()
            inputs[str(path)] = sha(path)
        row = {'source': str(source), 'object': str(obj), 'object_sha256': sha(obj),
               'dependency_sha256': inputs, 'cc1': command}
        dependencies[name] = row
        write(out / 'compilations.json', dependencies)
        obj.chmod(0o444)
        return row

    if a.resume_baseline:
        inputs = json.loads((out / 'inputs.json').read_text())
        revision = inputs['revision']
        require(revision == subprocess.check_output(['git', '-C', str(repo), 'rev-parse', a.revision], text=True).strip(), 'Resume revision differs')
        require(inputs['candidate_patch_sha256'] == PATCH_SHA, 'Resume patch differs')
        require(inputs['compiler_sha256'] == sha(compiler.resolve()), 'Resume compiler changed')
        require({r: sha(tree / r) if (tree / r).is_file() else None for r in CHANGED} == inputs['baseline_changed_sources'], 'Resume source is not baseline')
        require(not (out / 'candidate').exists(), 'Candidate already exists')
        entries = json.loads((out / 'object-map.json').read_text())
        dependencies = json.loads((out / 'compilations.json').read_text())
        frozen_objects = json.loads((out / 'frozen-object-hashes.json').read_text())
        external = json.loads((out / 'link-tool-input-hashes.json').read_text())
        original_source_hashes = json.loads((out / 'baseline-source-hashes.json').read_text())
        require(all(sha(Path(obj)) == h for obj, h in frozen_objects.items()), 'Frozen object changed')
        require(all(sha(tree / r) == h for r, h in original_source_hashes.items()), 'Frozen source changed')
        raw = (out / 'logs/ordinary-driver-plan.log').read_text()
        plan = [shlex.split(line.strip()) for line in raw.splitlines() if line.lstrip().startswith('"')]
        compile_plans, link_plan = plan[:-1], plan[-1]
        object_map = {e['original_object']: e['frozen_object'] for e in entries if not e['ieee']}
        link_output = work / 'build/libnntr_hvx_skel.so'
    else:
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', a.revision], text=True).strip()
        archive = subprocess.check_output(['git', '-C', str(repo), 'archive', revision, 'test/htp', 'nntrainer/tensor'])
        archive_path = out / 'source.tar'
        archive_path.write_bytes(archive)
        with tarfile.open(archive_path) as tar:
            tar.extractall(tree, filter='data')
        archive_path.unlink()
        work = tree / 'test/htp'
        backend = tree / REL
        (work / 'generated').mkdir(exist_ok=True)
        (work / 'build').mkdir(exist_ok=True)
        script = (work / 'build.sh').read_text()
        # Reject script drift rather than silently claiming flags still match it.
        expected_build_sha = '5d7b80853a2dbc8f8a31b0610ed316bbcaaa016f3ddcac2252b4c70fda4b4d76'
        require(sha(work / 'build.sh') == expected_build_sha, 'build.sh changed; review flag derivation')
        compiler = tools / 'Tools/bin/hexagon-clang'
        readelf = tools / 'Tools/bin/hexagon-readelf'
        qaic = sdk / 'ipc/fastrpc/qaic/Ubuntu/qaic'
        lib = hexkl / ('lib/' + a.hexkl_sdk_ver + '/hexagon_toolv19_' + a.hex_arch + '/libhexkl_micro.a')
        require(lib.is_file(), 'Missing HexKL archive')
        baseline_sources = {r: sha(tree / r) if (tree / r).is_file() else None for r in CHANGED}
        for rel in CHANGED:
            if not (tree / rel).is_file():
                continue
            dest = out / 'baseline-source' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tree / rel, dest)
        write(out / 'inputs.json', {'revision': revision, 'git_archive_sha256': hashlib.sha256(archive).hexdigest(),
                                   'build_script_sha256': sha(work / 'build.sh'),
                                   'candidate_patch_sha256': PATCH_SHA, 'baseline_changed_sources': baseline_sources,
                                   'environment': {k: os.environ.get(k) for k in ('PATH', 'LD_LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH')},
                                   'compiler_sha256': sha(compiler.resolve()), 'qaic_sha256': sha(qaic),
                                   'hexkl_archive_sha256': sha(lib), 'arch': a.hex_arch})
        shutil.copy2(a.candidate_patch, out / 'production.patch')
        run([compiler, '--version'], 'compiler-version')
        run([qaic, '-I', sdk / 'incs', '-I', sdk / 'incs/stddef', '-mdll', '-o', 'generated', 'nntr_hvx.idl'], 'qaic')
        write(out / 'generated-hashes.json', {str(p.relative_to(work)): sha(p)
                                             for p in sorted((work / 'generated').iterdir()) if p.is_file()})
        system = ['-I', str(sdk / ('rtos/qurt/compute' + a.hex_arch + '/include/qurt')),
                  '-I', str(sdk / ('rtos/qurt/compute' + a.hex_arch + '/include/posix')),
                  '-isystem', str(sdk / 'incs'), '-isystem', str(sdk / 'incs/stddef'),
                  '-isystem', str(sdk / 'ipc/fastrpc/incs')]
        common = ['-m' + a.hex_arch, '-mhvx', '-mhvx-length=128B']
        backend_includes = ['-I', str(backend / '..'), '-I', str(backend),
                            '-I', str(backend / 'hvx'), '-I', str(backend / 'hmx')]
        ieee_flags = common + ['-mhvx-ieee-fp', '-G0', '-O3', '-fPIC', '-Wall', '-Werror', '-I', 'generated']
        ieee_flags += backend_includes + ['-I', str(hexkl / 'include')] + system
        flags = common + ['-mhmx', '-G0', '-O3', '-fPIC', '-shared', '-Wall', '-Werror', '-I', 'generated', '-I', str(hexkl / 'include')]
        flags += backend_includes + system
        ordinary = ''
        for assignment in re.findall(r'^SRCS="([^"]*)"$', script, re.M):
            ordinary = assignment.replace('$SRCS', ordinary).replace('$BACKEND', str(backend))
        srclist = shlex.split(ordinary)
        require(len(srclist) == len(set(srclist)), 'Duplicate build source')
        require(all(s.endswith(('.c', '.o')) for s in srclist), 'Unknown source syntax')
        write(out / 'source-order.json', srclist)
        for index, source in enumerate([str(backend / 'hvx/hvx_q4_gemv_f32.c'), 'nntr_hvx_sf_probe.c']):
            obj = work / 'build' / (Path(source).stem + '.o')
            plan = parse_plan([str(compiler)] + ieee_flags + ['-c', source, '-o', str(obj)], 'ieee-plan-' + str(index))
            require(len(plan) == 1 and '-cc1' in plan[0], 'IEEE driver plan')
            compile_entry(plan[0], obj, 'baseline-ieee-' + str(index), source)
            entries.append({'source': source, 'original_object': str(obj), 'frozen_object': str(obj), 'ieee': True})
        link_output = work / 'build/libnntr_hvx_skel.so'
        plan = parse_plan([str(compiler)] + flags + srclist + [str(lib), '-o', str(link_output)], 'ordinary-driver-plan')
        compile_plans, link_plan = plan[:-1], plan[-1]
        require(all('-cc1' in c for c in compile_plans) and '-cc1' not in link_plan and
                len(compile_plans) == len([s for s in srclist if s.endswith('.c')]), 'Incomplete driver plan')
        object_map = {}
        for index, cc1 in enumerate(compile_plans):
            source = cc1[-1]
            require(source in srclist, 'Unexpected driver input')
            original_obj = cc1[cc1.index('-o') + 1]
            obj = out / 'objects/baseline' / (f'{index:02d}-' + Path(source).stem + '.o')
            compile_entry(cc1, obj, 'baseline-' + str(index), source)
            object_map[original_obj] = str(obj)
            entries.append({'source': source, 'original_object': original_obj, 'frozen_object': str(obj), 'ieee': False})
            if (index + 1) % 10 == 0:
                print('Baseline objects:', index + 1, '/', len(compile_plans), flush=True)
        frozen_objects = {e['frozen_object']: sha(Path(e['frozen_object'])) for e in entries}
        write(out / 'frozen-object-hashes.json', frozen_objects)
        write(out / 'object-map.json', entries)
        # Hash the toolchain objects/static archives that the actual linker can consume.
        external = {}
        for token in link_plan:
            if token.startswith('-L'):
                directory = Path(token[2:])
                if directory.is_dir():
                    for file in directory.glob('*.a'):
                        external[str(file.resolve())] = sha(file)
            elif token.endswith(('.o', '.a')) and Path(token).is_file() and token not in frozen_objects:
                external[str(Path(token).resolve())] = sha(Path(token))
        external[str(Path(link_plan[0]).resolve())] = sha(Path(link_plan[0]).resolve())
        write(out / 'link-tool-input-hashes.json', external)
        original_source_hashes = {str(p.relative_to(tree)): sha(p) for p in tree.rglob('*')
                                  if p.is_file() and not str(p.relative_to(tree)).startswith(('test/htp/build/', 'test/htp/generated/'))}
        write(out / 'baseline-source-hashes.json', original_source_hashes)

    def link(label, replacements):
        command = [replacements.get(v, object_map.get(v, v)) for v in link_plan]
        command = replace_output(command, link_output)
        command.insert(1, '-no-threads')
        results = []
        for rep in (1, 2):
            run(command, label + '-link-' + str(rep))
            target = out / label / ('libnntr_hvx_skel.so' if rep == 1 else 'repeat.so')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(link_output, target)
            results.append(sha(target))
        require(results[0] == results[1], label + ': linker repeat identity failed')
        dyn = run([readelf, '--dyn-syms', link_output], label + '-imports', capture=True)
        header = run([readelf, '-h', link_output], label + '-elf', capture=True)
        imports = []
        for line in dyn.splitlines():
            columns = line.split()
            if len(columns) >= 8 and columns[6] == 'UND':
                symbol = columns[7]
                require(ALLOW.match(symbol), 'Unsupported runtime import ' + symbol)
                require(not symbol.startswith('dspqueue_') or columns[4] == 'WEAK', 'Strong DSPQ import')
                imports.append({'symbol': symbol, 'binding': columns[4]})
        require(imports and re.search(r'Flags:\s+0x81\b', header), 'Runtime import/arch guard failed')
        for obj, expected in frozen_objects.items():
            require(sha(Path(obj)) == expected, 'Frozen baseline object changed')
        return {'sha256': results[0], 'bytes': link_output.stat().st_size, 'repeat_identical': True,
                'imports': imports, 'arch': 'v81', 'link_command': command}

    affected = set()
    for name, row in dependencies.items():
        if not name.startswith('baseline-'):
            continue
        source = Path(row['source'])
        source = (work / source).resolve() if not source.is_absolute() else source.resolve()
        for dep in row['dependency_sha256']:
            dep_path = Path(dep)
            if dep_path.is_relative_to(tree) and str(dep_path.relative_to(tree)) in CHANGED:
                affected.add(str(source.relative_to(tree)))
    require(sorted(affected) == CHANGED_TUS,
            'Audited TU list differs from actual compiler dependencies: ' + str(sorted(affected)))
    write(out / 'affected-tu-audit.json', {'actual': sorted(affected), 'expected': CHANGED_TUS})
    control = link('control', {})
    run(['git', 'apply', str(a.candidate_patch.resolve())], 'apply-candidate', cwd=tree)
    require({r: sha(tree / r) for r in CHANGED} == CANDIDATE, 'Applied source hashes differ')
    after_source_hashes = {str(p.relative_to(tree)): sha(p) for p in tree.rglob('*')
                          if p.is_file() and not str(p.relative_to(tree)).startswith(('test/htp/build/', 'test/htp/generated/'))}
    require(set(after_source_hashes) == set(original_source_hashes).union(CHANGED) and
            sorted(r for r in after_source_hashes if after_source_hashes[r] != original_source_hashes.get(r)) == sorted(CHANGED),
            'Unexpected source change')
    replacement = {}
    recompiled = []
    for index, cc1 in enumerate(compile_plans):
        source_path = Path(cc1[-1])
        relative = str(source_path.relative_to(tree)) if source_path.is_absolute() else 'test/htp/' + str(source_path)
        if relative not in CHANGED_TUS:
            continue
        obj = out / 'objects/candidate' / (f'{index:02d}-' + source_path.stem + '.o')
        compile_entry(cc1, obj, 'candidate-' + str(index), cc1[-1])
        replacement[cc1[cc1.index('-o') + 1]] = str(obj)
        recompiled.append(relative)
    require(sorted(recompiled) == CHANGED_TUS, 'Expected exactly the audited candidate TUs')
    candidate = link('candidate', replacement)
    require(control['imports'] == candidate['imports'], 'Candidate changed runtime imports')
    # External inputs and retained objects are rechecked after both links.
    require(all(sha(Path(p)) == h for p, h in external.items()), 'Link dependency changed during build')
    for name, compilation in dependencies.items():
        for path, expected in compilation['dependency_sha256'].items():
            if Path(path).is_relative_to(tree) and str(Path(path).relative_to(tree)) in CHANGED:
                continue
            require(sha(Path(path)) == expected, 'Compile dependency changed during build: ' + path)
    validation = {'status': 'PASS', 'source_revision': revision, 'source_patch_sha256': PATCH_SHA,
                  'control': control, 'candidate': candidate, 'frozen_objects': len(entries),
                  'unchanged_objects': len(entries) - len(CHANGED_TUS), 'candidate_recompiled_tus': recompiled,
                  'baseline_objects_unchanged': True, 'all_other_link_inputs_identical': True,
                  'original_build_sh_cc1_flags_used': True,
                  'altered_cc1_metadata_options': ['output path', 'dependency-file', 'MT'],
                  'shared_linker_setting': '-no-threads',
                  'original_threaded_link_was_not_repeatable': True,
                  'production_source_sha256': CANDIDATE,
                  'no_checkout_sdk_or_phone_changes': True,
                  'not_a_claim_of_original_3b_binary_identity': True,
                  'scope': 'A newly compiled fixed object-set pair; device numerical/performance validation remains required.'}
    write(out / 'validation.json', validation)
    print(json.dumps({k: validation[k] for k in ('status', 'frozen_objects', 'unchanged_objects', 'candidate_recompiled_tus')}, indent=2))
    print('CONTROL', control['sha256'])
    print('CANDIDATE', candidate['sha256'])


if __name__ == '__main__':
    main()
