#!/usr/bin/env python3
"""Collect frozen ARM-router evidence. Reads inputs; writes only a new archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from verify import verify


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'evidence', 'review', 'prescreen', 'shared', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse to overwrite an existing archive'
    here = Path(__file__).resolve().parent
    D, E, I, P, T = a.source, a.evidence, a.review, a.prescreen, a.shared
    pending = {}

    def add(source, target, compress=None):
        source = Path(source)
        assert source.is_file(), source
        data = source.read_bytes()
        assert not data.startswith((b'\x7fELF', b'!<arch>\n')), source
        data.decode('utf-8')
        if compress is None:
            compress = (source.suffix in {'.log', '.cpp', '.c', '.h', '.patch', '.diff',
                                         '.s', '.asm', '.dis', '.disasm', '.su', '.ll'} or
                        source.name.endswith('.disasm.txt') or len(data) > 128 * 1024)
        if compress:
            target += '.gz'
        assert target not in pending, target
        pending[target] = (gzip.compress(data, mtime=0) if compress else data,
                           {'source': str(source.resolve()), 'source_sha256': sha(data),
                            'uncompressed_sha256': sha(data), 'uncompressed_bytes': len(data)})

    def tree(root, target, predicate):
        assert root.is_dir(), root
        for path in sorted(root.rglob('*')):
            if path.is_file() and predicate(path):
                add(path, target + '/' + str(path.relative_to(root)))

    frozen = json.loads((D / 'frozen.json').read_text())
    source_hashes = json.loads((D / 'source-hashes.json').read_text())['files']
    for name, value in source_hashes.items():
        assert sha((D / 'candidate' / name).read_bytes()) == value
        add(D / 'candidate' / name, 'artifacts/production/source/' + name)
    for name, value in frozen['test_files'].items():
        assert sha((D / name).read_bytes()) == value
        add(D / name, 'artifacts/tests/source/' + name)
    tree(D / 'before', 'artifacts/production/before', lambda x: x.suffix == '.cpp' or x.name == 'meson.build')
    for name in ('production.patch', 'source-hashes.json'):
        add(D / name, 'artifacts/production/' + name)
    add(D / 'tests.patch', 'artifacts/tests/tests.patch')
    for name in ('README.txt', 'validation.json', 'frozen.json', 'host-v3b.log', 'host-v3.log'):
        add(D / name, 'artifacts/author/' + name)
    add(D / 'checks/host-v3b/commands.json', 'artifacts/author/checks/host-v3b/commands.json')
    tree(D / 'checks/mutations-v3-final', 'artifacts/author/checks/mutations-v3-final',
         lambda x: x.name in {'commands.json', 'results.json', 'build.log', 'run.log'})
    for name in ('app-command.json', 'app-build.log', 'moe.su'):
        add(D / 'codegen' / name, 'artifacts/author/codegen/' + name)
    tree(D / 'codegen/final', 'artifacts/author/codegen/final', lambda x: x.suffix != '.o')

    for name in ('root-applied.json', 'root-frozen.json', 'root-app-build.log',
                 'root-kernel.disasm.txt', 'root-normal-jobs.json',
                 'root-normal-runtime-identity.json', 'root-normal-summary.json',
                 'root-diagnostic-1024.json', 'root-diagnostic-2048.json',
                 'root-diagnostic-build.json', 'root-diagnostic-build.log',
                 'root-diagnostic-jobs.json', 'root-diagnostic-source.cpp',
                 'root-diagnostic-removed.json', 'root-normal-restored-build.log'):
        add(D / name, 'artifacts/root/' + name)
    for variant in ('lfm-router-arm-0', 'lfm-router-arm-1', 'lfm-router-arm-diagnostic'):
        add(E / (variant + '-artifacts.json'), 'artifacts/root/deployment/' + variant + '-artifacts.json')
        add(T / (variant + '.source.patch'), 'artifacts/root/deployment/' + variant + '.source.patch')
    jobs = json.loads((D / 'root-normal-jobs.json').read_text())
    jobs += [{'tag': 'continue-router-arm-diagnostic' + str(n), 'profile': True} for n in (1024, 2048)]
    for job in jobs:
        tag = job['tag']
        for suffix in ('.json', '.ids', '.log', '.memory.json'):
            add(E / (tag + suffix), 'raw/' + tag + suffix, suffix in ('.log', '.memory.json'))
        if job.get('profile'):
            add(E / (tag + '.logcat.txt'), 'raw/' + tag + '.logcat.txt', True)
    for path in sorted(D.glob('continue-router-arm-repeat*')):
        assert path.suffix in {'.json', '.log', '.ids'}
        add(path, 'artifacts/repeat/' + path.name)
    add(D / 'root-repeat.py', 'artifacts/repeat/root-repeat.py')
    add(E / 'validation/lfm-repeat.json', 'artifacts/repeat/prior-environment-reference.json')
    add(E / 'repro/build_repeat_check.py', 'repro/device/build_repeat_check_1024.py')
    for name in ('build_repeat_check.py', 'repeat_prompt_check.cpp', 'build.json', 'compile.log', 'link.log'):
        add(T / 'repeat2048' / name, 'repro/device/repeat2048/' + name)
    for name in ('run_matrix.py', 'deploy_runtime.py'):
        add(T / name, 'repro/device/' + name)
    add(E / 'bench.py', 'repro/device/bench.py')
    add(E / 'validation/memory_monitor.cpp', 'repro/device/memory_monitor.cpp')

    for name in ('root-native-v3-results.json', 'root-native-v3-0-1.log',
                 'root-native-v3-0-8.log', 'root-native-v3-1-1.log', 'root-native-v3-1-8.log'):
        add(D / name, 'artifacts/native/' + name)
    for name in ('commands.json', 'binary.sha256'):
        add(D / 'checks/native-v3-final' / name, 'artifacts/native/' + name, False)

    for name in ('diagnostic.patch', 'parse.py', 'parser_check.py', 'check.py', 'README.txt',
                 'parser-results.json', 'validation.json'):
        add(D / 'diagnostic' / name, 'artifacts/diagnostic/' + name)
    tree(D / 'diagnostic/candidate', 'artifacts/diagnostic/source', lambda x: x.suffix == '.cpp')
    tree(D / 'diagnostic/checks', 'artifacts/diagnostic/checks',
         lambda x: x.name in {'command.json', 'results.json', 'build.log', 'run-0.log', 'run-1.log'})
    # Retain normalized whole-TU OFF identity proof, but not duplicate multi-MiB
    # disassemblies. Exact compiler commands and source reproduce those files.
    tree(D / 'diagnostic/codegen', 'artifacts/diagnostic/codegen',
         lambda x: x.suffix in {'.json', '.log', '.su'})

    for name in ('review.txt', 'validation.json', 'manifest.json', 'reviewed-source-hashes.json',
                 'host-san.log', 'mutations.log', 'compile.log', 'recompile.py'):
        add(I / name, 'artifacts/independent/' + name)
    add(I / 'host-san/commands.json', 'artifacts/independent/host-san/commands.json')
    tree(I / 'mutations', 'artifacts/independent/mutations',
         lambda x: x.name in {'commands.json', 'results.json', 'build.log', 'run.log'})
    tree(I / 'codegen', 'artifacts/independent/codegen',
         lambda x: x.suffix in {'.json', '.log', '.su', '.ll'} or
                   (x.suffix == '.asm' and 'kernel-arm' in x.parts))

    tree(D / 'invalid-nonfinite', 'artifacts/rejected/v1',
         lambda x: x.suffix in {'.cpp', '.h', '.json', '.log'})
    tree(D / 'numeric-v2', 'artifacts/rejected/v2',
         lambda x: x.suffix in {'.cpp', '.h', '.json', '.log', '.su'} or x.name.endswith('.disasm.txt'))
    for stem in ('pragma', 'vector', 'strict'):
        for suffix in ('-probe.cpp', '-command.json', '-build.log', '-codegen.json', '-probe.disasm.txt'):
            add(D / (stem + suffix), 'artifacts/rejected/fp-pragmas/' + stem + suffix)
    for name in ('micro.cpp', 'build-command.json', 'build.log', 'codegen.json',
                 'micro.disasm', 'root-micro.json', 'root-micro.log'):
        add(P / name, 'artifacts/prescreen/' + name)
    for name in ('micro.cpp', 'build-command.json', 'build.log', 'micro.disasm',
                 'root-micro.json', 'root-micro.log'):
        add(P / 'invalid-fma' / name, 'artifacts/rejected/prescreen-fma/' + name)

    bundled = not (here / 'README.txt').exists()
    resources = [('README.txt', 'README.txt'), ('REPRODUCE.txt', 'repro/README.txt'),
                 ('verify.py', 'validation/verify.py'),
                 ('reproduce.py', 'repro/reproduce.py'),
                 ('curate_router_arm.py', 'validation/curate_router_arm.py')]
    for source, target in resources:
        path = here.parent / target if bundled else here / source
        add(path, target, False)
    manifest = {'format': 1, 'decision': 'retain_opt_in_prefill_router',
                'provenance': {key: str(getattr(a, key).resolve()) for key in
                               ('source', 'evidence', 'review', 'prescreen', 'shared')},
                'files': {}}
    for name, (data, entry) in sorted(pending.items()):
        path = a.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest['files'][name] = dict(entry, sha256=sha(data), bytes=len(data))

    def save_manifest():
        (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    save_manifest()
    result = verify(a.output)
    result_bytes = (json.dumps(result, indent=2) + '\n').encode()
    (a.output / 'results.json').write_bytes(result_bytes)
    manifest['files']['results.json'] = {'source': 'generated from archived raw records by validation/verify.py',
                                       'sha256': sha(result_bytes), 'uncompressed_sha256': sha(result_bytes),
                                       'bytes': len(result_bytes), 'uncompressed_bytes': len(result_bytes)}
    save_manifest()
    verify(a.output)
    total = sum(path.stat().st_size for path in a.output.rglob('*') if path.is_file())
    print(json.dumps({'archive': str(a.output.resolve()), 'payload_files': len(manifest['files']),
                      'total_bytes': total, 'manifest_sha256': sha((a.output / 'manifest.json').read_bytes()),
                      'validation': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
