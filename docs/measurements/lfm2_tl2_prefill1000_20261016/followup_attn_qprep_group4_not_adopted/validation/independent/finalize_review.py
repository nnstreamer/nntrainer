#!/usr/bin/env python3
"""Freeze reviewer text evidence after all strict checks complete."""
import hashlib
import json
from pathlib import Path
import shutil

O = Path(__file__).resolve().parent
D = Path('/tmp/nntr-lfm2-attn-strided-qprep-20261017')
Q = Path('/tmp/nntr-lfm2-attn-qprep-fixed-objects-20261017')
PROD = 'a07332c9043b6760a42392cae06a237319e6a1e383829edc0bee5d838b435df8'
TEST = 'f962d48fabdb73ce2702333f67907b8c5f5ddd86e17c01c38d37048ea66db23f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def clean_log(path):
    text = Path(path).read_text()
    assert 'runtime error:' not in text and 'ERROR: AddressSanitizer' not in text
    assert 'ERROR: LeakSanitizer' not in text and 'DEADLYSIGNAL' not in text
    return text


def main():
    assert sha(D / 'production.patch') == PROD and sha(D / 'tests.patch') == TEST
    sources = load(D / 'source-hashes.json')
    tests = load(D / 'tests-hashes.json')
    for rel, pair in sources.items():
        assert sha(D / 'candidate' / rel) == pair['candidate_sha256']
        copy(D / 'candidate' / rel, O / 'frozen/candidate' / rel)
        if pair['baseline_sha256']:
            assert sha(D / 'baseline' / rel) == pair['baseline_sha256']
            copy(D / 'baseline' / rel, O / 'frozen/baseline' / rel)
    for rel, digest in tests.items():
        assert sha(D / 'candidate' / rel) == digest
        copy(D / 'candidate' / rel, O / 'frozen/candidate' / rel)
    for filename in ('production.patch', 'tests.patch', 'source-hashes.json', 'tests-hashes.json'):
        copy(D / filename, O / 'frozen' / filename)
    quant = load(O / 'quant-san-fixed/validation.json')
    rpc = load(O / 'rpc-san/validation.json')
    for dirname, results, count in [('quant-san-fixed', quant['results'], 6),
                                     ('rpc-san', rpc['results'], 4)]:
        assert len(results) == count
        assert results[0]['returncode'] == 0 and results[0]['expected_success']
        for row in results:
            text = clean_log(O / dirname / (row['name'] + '.log'))
            if not row['expected_success']:
                assert row['returncode'] == -6 and 'Assertion' in text
        for path in (O / dirname).glob('*.log'):
            clean_log(path)
    assert '576 cases PASS' in (O / 'quant-san-fixed/attn_qprep_native_check.log').read_text()
    assert '31 PASS' in (O / 'quant-san-fixed/attn_qprep_native_check.log').read_text()
    assert '15 actual-body' in (O / 'rpc-san/rpc_check.log').read_text()
    assert '460 pipeline/tail/error' in clean_log(O / 'overlap-san/overlap_check.log')
    for name in ('wrong_accumulator', 'missing_extent_guard', 'missing_pv_drain'):
        assert 'Assertion' in clean_log(O / 'overlap-san' / (name + '.log'))
    codegen = load(O / 'codegen/results.json')
    assert len(codegen) == 16 and all(row['returncode'] == 0 for row in codegen)
    assert all(row['same'] for row in load(O / 'format-review/validation.json'))
    function_bytes = load(O / 'final-quant-codegen.json')
    assert len(function_bytes['common_functions']) == 11
    assert all(r['same_function_bytes'] for r in function_bytes['common_functions'])
    fixed = load(O / 'fixed-object-review/validation.json')
    assert fixed['status'] == 'PASS'
    supplement = load(Q / 'shared-link-audit/validation.json')
    assert supplement['status'] == 'PASS' and supplement['same_path_relink_both_original_artifacts_exact']
    assert len(supplement['shared_linker_input_sha256']) == 7
    for path, digest in supplement['shared_linker_input_sha256'].items():
        assert sha(path) == digest
    assert all(row['returncode'] == 0 and row['matches_original_full_sha256']
               for row in supplement['commands'])
    copy(Q / 'shared-link-audit/validation.json', O / 'fixed-object-review/root-shared-link-supplement.json')
    summary = {
        'status': 'PASS', 'verdict': 'No unresolved production correctness/integration blocker; no performance approval.',
        'production_patch_sha256': PROD, 'tests_patch_sha256': TEST,
        'production_source_sha256': sources, 'test_source_sha256': tests,
        'independent_executions': {
            'actual_qprep_hvx': {'cases': 576, 'guards': 31, 'intended_assertion_mutants': 5,
                                 'sanitizers': 'ASan+UBSan; fatal; no suppression; SDK aligned(4) host shim',
                                 'validation': 'quant-san-fixed/validation.json'},
            'actual_rpc': {'cases': 15, 'intended_assertion_mutants': 3,
                           'sanitizers': 'ASan+UBSan; explicit fatal runtime options',
                           'validation': 'rpc-san/validation.json'},
            'qx_overlap': {'cases': 460, 'intended_assertion_mutants': 3,
                           'sanitizers': 'ASan+UBSan; explicit fatal runtime options',
                           'source_manifest': 'overlap-san/source_manifest.json'},
            'full_tu_v81': {'object_builds': 8, 'assembly_builds': 8, 'errors': 0,
                            'results': 'codegen/results.json'},
            'address_lifetime': 'final-address-lifetime.json',
            'quantizer_function_bytes': 'final-quant-codegen.json',
            'changed_line_clang_format_14': 'format-review/validation.json',
            'fixed_object_pair': 'fixed-object-review/validation.json',
        },
        'resolved_findings': [
            {'issue': 'Test HVX_UVector host alignment was 8 instead of SDK unaligned 4.',
             'initial_tests_patch': 'ffb26aa00636eead79efd076b019ed4cb23609a88391411dcb5ce56ebb8cafa2',
             'initial_strict_run': 'FAIL; preserved, not a sanitizer PASS',
             'resolution': 'Test-only aligned(4) typedef/static assertions and fatal sanitizer mode; production unchanged.',
             'evidence': 'rejected-host-shim'},
            {'issue': 'Fixed-object input inventory originally recorded archives but omitted shared libgcc.',
             'resolution': 'Independent hash check and parent seven-shared-library/same-path-relink supplement.',
             'evidence': 'fixed-object-review/root-shared-link-supplement.json'},
        ],
        'frame_bytes': {'attn_prefill_before_after': [1536, 1536],
                        'rpc_step_before_after': [336, 336],
                        'new_group4_entry': 32, 'new_group4_worker': 640},
        'memory': {'new_heap_bytes': 0, 'new_vtcm_bytes': 0, 'persistent_metadata_bytes': 0,
                   'dsp_local_io_bytes_before_after': [24, 28],
                   'existing_qx_bytes': 1179648, 'deferred_raw_bank_bytes': 524288},
        'limitations': ['SDK libnative is prebuilt and unsanitized.',
                        'HMX/DMA/cache behavior needs parent device validation.',
                        'Separate Qprep/overlap checks are not full-attention numerical replay.',
                        'Semantic fallback does not prove unchanged Gemma or 1024 performance.',
                        'No hardware TPS or causal cache-contention claim is made.'],
        'root_sdk_phone_or_author_source_mutated': False,
    }
    (O / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    selected = []
    prefixes = ('frozen/', 'codegen/', 'quant-san-fixed/', 'rpc-san/', 'overlap-san/',
                'fixed-object-review/', 'format-review/', 'rejected-host-shim/')
    top = {'review.txt', 'validation.json', 'final-address-lifetime.json', 'final-quant-codegen.json',
           'address_lifetime_check.py', 'compare_elf_functions.py', 'recompile.py',
           'verify_fixed_objects.py', 'finalize_review.py'}
    suffixes = {'.py', '.json', '.txt', '.log', '.c', '.h', '.s', '.su', '.patch', '.diff'}
    for p in sorted(O.rglob('*')):
        if not p.is_file():
            continue
        rel = str(p.relative_to(O))
        if rel in top or (rel.startswith(prefixes) and p.suffix in suffixes):
            selected.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    manifest = {'scope': 'Reviewer text/source evidence only; ELF, objects and vendor binaries excluded.',
                'files': selected}
    (O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name in ('review.txt', 'validation.json', 'manifest.json'):
        print(name, sha(O / name))
    print('manifest payload files', len(selected))


if __name__ == '__main__':
    main()
