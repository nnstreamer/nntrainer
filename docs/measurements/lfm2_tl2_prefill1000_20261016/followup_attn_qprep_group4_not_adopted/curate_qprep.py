#!/usr/bin/env python3
"""Curate existing Qprep evidence; no compilation, model run, or device access."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
sys.dont_write_bytecode = True
from readback import audit, CONTROL, CANDIDATE, DIAGNOSTIC


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'author', 'review', 'fixed', 'root-evidence', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    D, I, F, Q = (getattr(args, name).resolve() for name in
                  ('author', 'review', 'fixed', 'root_evidence'))
    E = args.repo.resolve() / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    A = Path(__file__).resolve().parent
    pending = {}
    checked = {}
    omitted = {}

    def add(path, relative, compress=None):
        rel = PurePosixPath(relative)
        need(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe archive path')
        need(path.is_file(), 'Missing archive input: ' + str(path))
        data = path.read_bytes()
        need(not data.startswith(b'\x7fELF'), 'Binary archive input: ' + str(path))
        if compress is None:
            compress = path.suffix in {'.log', '.c', '.cpp', '.h', '.inc', '.patch',
                                       '.s', '.su', '.d', '.dis', '.diff', '.asm'}
            compress |= path.name.endswith('.logcat.txt') or len(data) > 128 * 1024
        item = (data, str(path), bool(compress))
        if relative in pending:
            need(pending[relative] == item, 'Different duplicate target: ' + relative)
        else:
            pending[relative] = item

    def generated(relative, value):
        need(relative not in pending, 'Duplicate generated target: ' + relative)
        pending[relative] = ((json.dumps(value, indent=2) + '\n').encode(), 'generated', False)

    def text_tree(folder, relative, suffixes=None):
        need(folder.is_dir(), 'Missing evidence tree: ' + str(folder))
        suffixes = suffixes or {'.json', '.log', '.txt', '.c', '.h', '.inc', '.py', '.s', '.su', '.d', '.diff', '.patch'}
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in suffixes:
                add(path, relative + '/' + str(path.relative_to(folder)))

    # Freeze checks use text/source manifests. Do not run any original build/test script.
    author_manifest = read(D / 'manifest.json')
    for relative, row in author_manifest.items():
        path = D / relative
        need(sha(path.read_bytes()) == row['sha256'] and path.stat().st_size == row['bytes'],
             'Changed frozen author payload: ' + relative)
        add(path, 'validation/author/' + relative)
    checked['author_manifest_sha256'] = sha((D / 'manifest.json').read_bytes())
    add(D / 'manifest.json', 'validation/author/manifest.json')
    review_manifest = read(I / 'manifest.json')
    for row in review_manifest['files']:
        path = I / row['path']
        need(sha(path.read_bytes()) == row['sha256'] and path.stat().st_size == row['bytes'],
             'Changed frozen independent payload: ' + row['path'])
        add(path, 'validation/independent/' + row['path'])
    checked['independent_manifest_sha256'] = sha((I / 'manifest.json').read_bytes())
    add(I / 'manifest.json', 'validation/independent/manifest.json')
    # Original reviewer fatal log is also retained under its original relative name.
    add(I / 'quant-san/attn_qprep_native_check.log',
        'validation/independent/quant-san/attn_qprep_native_check.log')

    source_hashes = read(D / 'source-hashes.json')
    tests_hashes = read(D / 'tests-hashes.json')
    for name in ('production.patch', 'tests.patch', 'source-hashes.json', 'tests-hashes.json',
                 'tests-v1-ffb26.patch', 'tests-hashes-v1-ffb26.json'):
        add(D / name, 'source/' + name)
    for relative, hashes in source_hashes.items():
        add(D / 'candidate' / relative, 'source/candidate/' + relative)
        if hashes['baseline_sha256']:
            add(D / 'baseline' / relative, 'source/baseline/' + relative)
    for relative in tests_hashes:
        add(D / 'candidate' / relative, 'source/candidate/' + relative)
        if (D / 'baseline' / relative).is_file():
            add(D / 'baseline' / relative, 'source/baseline/' + relative)
    # Unmodified source needed to interpret the existing overlap fixture, not a full checkout.
    for relative in ('tools/lfm2_attn_overlap_check.py', 'test/htp/host/attn_overlap_native_check.c'):
        path = D / 'candidate' / relative
        if path.exists():
            add(path, 'source/support/' + relative)
    for folder in ('checks/frozen-strict', 'checks/rpc-final', 'checks/overlap', 'codegen'):
        text_tree(D / folder, 'validation/author/' + folder)
    for relative in ('checks/final/attn_qprep_native_check.log', 'checks/final/commands.json',
                     'checks/final/validation.json', 'checks/strict-final/commands.json',
                     'host-strict-final.log', 'host-strict-final-fixed.log', 'host-frozen-strict.log',
                     'host-overlap.log', 'host-rpc-final.log', 'baseline-hashes.json'):
        add(D / relative, 'validation/author/' + relative)
    # Trace code is separate from production and all normal run artifacts.
    for name in ('diagnostic.patch', 'parse_trace.py', 'check_parser.py', 'parser-check.json',
                 'README.txt', 'build-validation.json', 'host-strict-counts.json',
                 'host-counts.json', 'host-strict.log', 'host.log'):
        add(D / 'diagnostic' / name, 'diagnostic/' + name)
    for folder in ('build', 'host-strict', 'host-stub'):
        text_tree(D / 'diagnostic' / folder, 'diagnostic/' + folder)
    add(D / 'diagnostic/host/attn_qprep_native_check.log',
        'diagnostic/superseded-host/attn_qprep_native_check.log')
    add(D / 'diagnostic/source/nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c',
        'diagnostic/source/nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c')

    for name in ('README.txt', 'build_fixed_objects.py', 'build_diagnostic.py', 'candidate-config.json'):
        add(F / name, 'fixed-build/' + name)
    for path in sorted((F / 'build').iterdir()):
        if path.is_file() and path.suffix in {'.json', '.patch'}:
            add(path, 'fixed-build/build/' + path.name)
    text_tree(F / 'build/logs', 'fixed-build/build/logs')
    text_tree(F / 'build/objects', 'fixed-build/build/objects', {'.d'})
    text_tree(F / 'diagnostic', 'fixed-build/diagnostic')
    text_tree(F / 'shared-link-audit', 'fixed-build/shared-link-audit')
    # Objects/DSOs are rehashed but never copied into the evidence archive.
    frozen = read(F / 'build/frozen-object-hashes.json')
    need(len(frozen) == 63, 'Expected 63 baseline objects')
    for historical, expected in frozen.items():
        path = Path(historical)
        marker = '/build/'
        need(marker in historical, 'Unknown fixed object path')
        path = F / 'build' / historical.split(marker, 1)[1]
        data = path.read_bytes()
        need(sha(data) == expected, 'Frozen object changed: ' + str(path))
        omitted['fixed-build/' + str(path.relative_to(F))] = {'sha256': expected, 'bytes': len(data)}
    checked['current_frozen_object_sha256_verified'] = 63
    for folder in (F / 'build/objects/candidate', F / 'build', F / 'diagnostic'):
        for path in sorted(folder.glob('*')):
            if path.is_file() and path.suffix in {'.o', '.so'}:
                data = path.read_bytes()
                omitted['fixed-build/' + str(path.relative_to(F))] = {'sha256': sha(data), 'bytes': len(data)}

    for path in sorted(Q.iterdir()):
        if path.is_file() and (path.name.startswith('root-') or path.name.endswith('deployment.patch') or path.name == 'diagnostic-quant-source.c'):
            add(path, 'validation/root/' + path.name)
    # Full runtime identities; binary bodies are intentionally excluded.
    for variant in (CONTROL, CANDIDATE, DIAGNOSTIC):
        add(E / (variant + '-artifacts.json'), 'artifacts/' + variant + '.json')
    add(E / 'lfm-router-arm-1-artifacts.json', 'artifacts/accepted-restored-runtime.json')
    for n in (1024, 2048):
        for path in sorted((E / ('repro/models/lfm2-' + str(n))).glob('*.json')):
            add(path, 'configuration/lfm2-' + str(n) + '/' + path.name)
    for source, target in ((E / 'device.json', 'configuration/device.json'),
                           (E / 'bench.py', 'repro/bench.py'),
                           (E / 'repro/model-conversion.json', 'configuration/historical-model-conversion.json'),
                           (E / 'validation/memory_monitor.cpp', 'repro/memory_monitor.cpp')):
        add(source, target)
    for row in read(Q / 'root-normal-jobs.json'):
        for suffix in ('.json', '.ids', '.log', '.memory.json'):
            name = row['tag'] + suffix
            add(E / name, 'raw/' + name)
    for row in read(Q / 'root-diagnostic-summary.json'):
        tag = row['tag']
        for suffix in ('.json', '.ids', '.log', '.memory.json', '.logcat.txt'):
            add(E / (tag + suffix), 'diagnostic/raw/' + tag + suffix)
        for suffix in ('.live.logcat.txt', '.parsed.json', '.collector.stderr', '.runner.log'):
            add(Q / (tag + suffix), 'diagnostic/raw/' + tag + suffix)

    summary = audit(lambda relative: pending[relative][0])
    generated('summary.json', summary)
    generated('omitted-artifact-identities.json', {
        'scope': 'Identity only; object, binary, SDK, model, and unrelated source bodies are excluded.',
        'fixed_build_objects_and_binaries': omitted,
        'runtime_binary_maps': summary['runtime_binaries'],
        'sdk_tool_and_link_input_hashes': 'fixed-build/build/link-tool-input-hashes.json',
        'shared_library_supplement': 'fixed-build/shared-link-audit/validation.json',
        'model_weight_hashes': 'configuration/historical-model-conversion.json is prior provenance, not a fresh per-run weight hash; no model bytes copied',
    })
    generated('curation-validation.json', {
        'status': 'PASS', 'decision': 'not_adopted', 'new_tests_builds_or_device_runs': False,
        'raw_readback_independent_of_author_parser': True,
        'normal_runs_recomputed': 12, 'diagnostic_runs_recomputed': 2,
        'all_14_id_and_memory_hashes_recomputed': True,
        'source_and_frozen_manifests_verified': checked,
        'original_documents_unmodified': True,
        'summary_supersedes_original_pending_decision_text': True,
        'frozen_review_report_sha256': sha((I / 'review.txt').read_bytes()),
        'frozen_review_validation_sha256': sha((I / 'validation.json').read_bytes()),
        'all_root_metadata_at_curation': sorted(p.name for p in Q.glob('root-*') if p.is_file()),
        'scope': 'Existing records and hashes only; no correctness/performance test rerun.'
    })
    for name in ('curate_qprep.py', 'verify_archive.py', 'readback.py', 'README.txt'):
        add(A / name, name, False)
    output = args.output.resolve()
    need(not output.exists(), 'Refusing to overwrite existing archive: ' + str(output))
    output.mkdir(parents=True)
    files = {}
    for relative, (data, origin, compress) in sorted(pending.items()):
        stored_name = relative + '.gz' if compress else relative
        path = output / stored_name
        need(stored_name not in files, 'Duplicate stored name')
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = gzip.compress(data, compresslevel=9, mtime=0) if compress else data
        path.write_bytes(stored)
        files[stored_name] = {'logical_path': relative, 'sha256': sha(stored),
                              'uncompressed_sha256': sha(data), 'gzip': compress,
                              'stored_bytes': len(stored), 'uncompressed_bytes': len(data),
                              'origin': origin}
    manifest = {'schema': 1, 'decision': 'not_adopted',
                'topic': 'LFM2 cached group4 Qprep gather removal',
                'source_base_commit': '742735f3c61803fac51189012ba4caf588a3bd50',
                'scope': 'Portable text/source/raw evidence; no SDK, models, binaries, or unrelated snapshots.',
                'files': files}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archive': str(output), 'manifest_sha256': sha((output / 'manifest.json').read_bytes()),
                      'payload_files': len(files), 'stored_payload_bytes': sum(r['stored_bytes'] for r in files.values()),
                      'uncompressed_payload_bytes': sum(r['uncompressed_bytes'] for r in files.values()),
                      'summary_sha256': sha((output / 'summary.json').read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
