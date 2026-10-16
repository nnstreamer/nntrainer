#!/usr/bin/env python3
"""Validate and curate the unadopted LFM tail-gather experiment; host files only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

REL = 'nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c'
CONTROL = 'lfm-tail-gather-control'
CANDIDATE = 'lfm-tail-gather-candidate'
PRODUCTION_SHA = '4008a7108a632d218fbb081fb16eb06a884db4a9c70bc4da1dbf45a3aadf7824'
TESTS_SHA = '75e264f53c041bc9ce91a1b9417ed7fcd04667a97d4660b2de5c03654fef147a'
SOURCE_SHA = '725b006d02ebfedd442279e54691c499ad1ebc7270959bf6dafa4b2c6f33e3ba'
EXPECTED_IDS = {
    1024: 'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
    2048: '2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33',
}
ENV = {
    'NNTR_NUM_THREADS': '8', 'NNTR_HTP_PAGE_CACHE': '0',
    'NNTR_LOAD_PREFETCH': '0', 'NNTR_MOE_CACHE_GROW': '0',
    'NNTR_MOE_PRELOAD_EXPERTS': '128', 'NNTR_INIT_TRACE': '1',
    'NNTR_HTP_PREDICT': '2', 'NNTR_MOE_PREFETCH_RESIDUAL': '0.5',
    'NNTR_HTP_TL2_STABLE_SLOTS': '1', 'NNTR_LFM2_PREFILL_EARLY': '1',
    'NNTR_LFM2_DECODE_PAIR_READ': '1', 'NNTR_LFM2_DECODE_DEMAND_HANDOFF': '1',
    'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def verify_hash_manifest(root, path):
    for relative, record in read(path).items():
        expected = record['sha256'] if isinstance(record, dict) else record
        require(sha(root / relative) == expected, 'Changed frozen input: ' + str(root / relative))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'author', 'independent', 'scratch', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--restoration', type=Path,
                   help='Root restoration JSON; required to write the final archive')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    D, I, T = a.author.resolve(), a.independent.resolve(), a.scratch.resolve()
    E = a.repo.resolve() / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    A = Path(__file__).resolve().parent
    pending = {}

    def add(source, target, compress=None):
        require(source.is_file(), 'Missing archive input: ' + str(source))
        if compress is None:
            compress = source.suffix in {'.log', '.c', '.h', '.inc', '.patch', '.s', '.su', '.dis', '.diff'}
            compress |= source.stat().st_size > 128 * 1024
        target += '.gz' if compress else ''
        require(target not in pending, 'Duplicate target: ' + target)
        pending[target] = (source.read_bytes(), str(source), compress)

    def generated(target, value):
        require(target not in pending, target)
        pending[target] = ((json.dumps(value, indent=2) + '\n').encode(), 'generated', False)

    def small_tree(root, target, extensions):
        require(root.is_dir(), 'Missing evidence directory: ' + str(root))
        for source in sorted(root.rglob('*')):
            if source.is_file() and source.suffix in extensions and source.stat().st_size <= 128 * 1024:
                add(source, target + '/' + str(source.relative_to(root)))

    # Freeze and cross-bind both authorship and independent validation.
    verify_hash_manifest(D, D / 'manifest.json')
    verify_hash_manifest(D, D / 'source-hashes.json')
    verify_hash_manifest(I, I / 'manifest.json')
    verify_hash_manifest(I / 'frozen', I / 'frozen-hashes.json')
    require(sha(D / 'production.patch') == PRODUCTION_SHA, 'Wrong production topic')
    require(sha(D / 'tests.patch') == TESTS_SHA, 'Wrong fixture topic')
    author, review = read(D / 'validation.json'), read(I / 'validation.json')
    require(author['source_hashes']['candidate/' + REL] == SOURCE_SHA and
            review['source_sha256'] == SOURCE_SHA and
            review['production_patch_sha256'] == PRODUCTION_SHA and
            review['tests_patch_sha256'] == TESTS_SHA, 'Review/source mismatch')
    for checks in (author['checks'], review['independent_checks']):
        require(checks['helper'] == checks['readout'] == checks['whole_tu'] == 'PASS' and
                checks['helper_cases'] == 198 and checks['guard_cases'] == 16 and
                checks['readout_cases'] == 512 and checks['whole_cases'] == 21 and
                checks['sanitizers'], 'Incomplete actual-source checks')
    require(author['mutants']['killed'] == author['mutants']['total'] == 7 and
            len(review['independent_mutants']['results']) == 7 and
            all(x['killed'] for x in review['independent_mutants']['results']), 'Mutants incomplete')
    require(review['function_bytes']['generic_worker_exact'] and
            review['stack_bytes']['main_before'] == review['stack_bytes']['main_after'] == 17152,
            'Codegen boundary changed')

    variants = {}
    for name in (CONTROL, CANDIDATE):
        record = read(E / (name + '-artifacts.json'))
        require(len(record['binaries']) == 7 and
                all(re.fullmatch('[0-9a-f]{64}', v) for v in record['binaries'].values()),
                name + ': seven runtime SHA256s required')
        require(sha(T / (name + '.source.patch')) == record['source_patch_sha256'], name + ': source patch')
        variants[name] = record
        add(E / (name + '-artifacts.json'), 'artifacts/' + name + '.json')
        add(T / (name + '.source.patch'), 'artifacts/' + name + '.source.patch')
    require(variants[CONTROL]['base_commit'] == variants[CANDIDATE]['base_commit'], 'Different source bases')
    require(set(variants[CONTROL]['binaries']) == set(variants[CANDIDATE]['binaries']), 'Runtime file set')
    changed = [k for k in variants[CONTROL]['binaries']
               if variants[CONTROL]['binaries'][k] != variants[CANDIDATE]['binaries'][k]]
    require(changed == ['libnntr_hvx_skel.so'], 'Only DSP runtime may change')
    require(not (T / (CONTROL + '.source.patch')).read_bytes(), 'Control has a source delta')
    base = read(D / 'base.json')
    integrated = read(D / 'root-frozen/source.json')
    control = read(D / 'root-control/source.json')
    require(control['source_sha256'] == base['baseline_sha256'] == sha(D / 'baseline' / REL) ==
            sha(D / 'root-control/hexkl_mm_u8i4_moe.c'), 'Control source mismatch')
    require(control['base_commit'] == variants[CONTROL]['base_commit'], 'Control commit mismatch')
    require(integrated['source_sha256'] == sha(D / 'candidate' / REL) == SOURCE_SHA, 'Integrated source mismatch')
    require(integrated['build_log_sha256'] == sha(D / 'root-dsp-build.log'), 'Build log mismatch')
    require(integrated['binary_sha256'] == sha(D / 'root-frozen/libnntr_hvx_skel.so') ==
            variants[CANDIDATE]['binaries']['libnntr_hvx_skel.so'], 'Candidate DSP mismatch')
    require(control['dsp_sha256'] == sha(D / 'root-control/libnntr_hvx_skel.so') ==
            variants[CONTROL]['binaries']['libnntr_hvx_skel.so'], 'Control DSP mismatch')
    base_source = subprocess.check_output(['git', '-C', str(a.repo), 'show', control['base_commit'] + ':' + REL])
    require(digest(base_source) == base['baseline_sha256'], 'Commit source mismatch')
    for patch in (D / 'production.patch', D / 'root-applied.patch', T / (CANDIDATE + '.source.patch')):
        with tempfile.TemporaryDirectory(prefix='tail-patch-proof-', dir=A) as temporary:
            target = Path(temporary) / REL
            target.parent.mkdir(parents=True)
            target.write_bytes(base_source)
            subprocess.run(['git', 'apply', str(patch)], cwd=temporary, check=True, capture_output=True)
            require(sha(target) == SOURCE_SHA, 'Patch does not reproduce source: ' + str(patch))

    # Twelve ordinary runs. Diagnostics and historical models cannot enter this cohort.
    jobs = read(D / 'root-normal-jobs.json')
    require(len(jobs) == 12 and len({j['tag'] for j in jobs}) == 12, 'Expected 12 unique normal runs')
    require([j['variant'] for j in jobs] == [CONTROL, CANDIDATE, CANDIDATE, CONTROL, CONTROL, CANDIDATE] * 2,
            'AB/BA/AB order changed')
    rows, memory_rows = [], []
    for index, job in enumerate(jobs):
        tag = job['tag']
        n = 1024 if index < 6 else 2048
        side = 0 if job['variant'] == CONTROL else 1
        repetition = (1, 1, 2, 2, 3, 3)[index % 6]
        prefix = 'continue-tail-gather' + ('2048' if n == 2048 else '')
        require(tag == f'{prefix}-{side}-r{repetition}', 'Unexpected normal-run tag')
        r, m = read(E / (tag + '.json')), read(E / (tag + '.memory.json'))
        require(r['tag'] == tag and r['variant'] == job['variant'] and r['model'] == job['model'] ==
                ('model' if n == 1024 else 'model2048'), tag + ': identity')
        require(r['valid'] and r['performance_comparison_eligible'] and not r['instrumented'] and
                not r['profile'] and r['returncode'] == 0 and not r['concurrent_processes'], tag + ': invalid run')
        require(r['moe_cache_experts'] == 16 and r['prefill_tokens'] == n and r['decode_tokens'] == 64,
                tag + ': shape/cache')
        require(set(r['environment']) == set(ENV) | {'LD_LIBRARY_PATH', 'ADSP_LIBRARY_PATH', 'NNTR_GENERATED_IDS'},
                tag + ': unexpected environment')
        for key, value in ENV.items():
            require(r['environment'].get(key) == value, tag + ': environment ' + key)
        for key, value in job['env'].items():
            require(r['environment'].get(key) == str(value), tag + ': job env ' + key)
        require(r['ids_match_reference'] and r['ids_sha256'] == sha(E / (tag + '.ids')) == EXPECTED_IDS[n] and
                len((E / (tag + '.ids')).read_text().splitlines()) == 65, tag + ': generated IDs')
        config = E / 'repro/models' / ('lfm2-' + str(n)) / 'nntr_config.json'
        cfg = read(config)
        require(r['model_config_sha256'] == sha(config) and cfg['moe_cache_experts'] == 16 and
                cfg['num_to_generate'] == 64 and cfg['init_seq_len'] == n and
                cfg['moe_layer_dtype'] == 'TL2CX_WH' and cfg['fc_layer_dtype'] == 'QS4CX_WH_AFFINE',
                tag + ': model configuration')
        peak = m['peak_dedup_sample']
        require(m['child_waited'] and m['child_exit_code'] == 0 and m['child_signal'] is None and
                m['requested_period_ms'] == 20 and m['mode'] == 'supervised-child' and
                peak['composite_available'] and peak['dedup_complete_for_readable_size_fds'], tag + ': monitor')
        require(not any(peak[k] for k in ('fdinfo_read_errors', 'unknown_identity_fds', 'inode_size_conflicts',
                                         'invalid_fields', 'counter_overflow')), tag + ': incomplete peak')
        require(r['full_run_memory_report_sha256'] == sha(E / (tag + '.memory.json')) and
                r['full_run_sampled_peak_kib'] * 1024 == m['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] ==
                peak['anon_plus_fdinfo_dedup_or_raw_fallback_bytes'], tag + ': peak hash/value')
        require(r['full_run_memory_sample_count'] == m['sample_count'] == len(m['samples']) and
                r['full_run_memory_mean_interval_ms'] == m['sample_interval_ms']['mean'], tag + ': sampling')
        log = (E / (tag + '.log')).read_text()
        for label, field in [('prefill', 'prefill'), ('generation', 'decode')]:
            occurrences = re.findall(label + r': (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS', log)
            require(len(occurrences) == 1 and tuple(map(float, occurrences[0])) ==
                    (r[field + '_tokens'], r[field + '_ms'], r[field + '_tps']), tag + ': raw throughput')
        for marker in ('[MoE] preload_experts=128 cache_slots=608',
                       '[HTP-PREFETCH] direct slots=32 bytes=64618496 readers=2 pair_read=1',
                       '[HTP] demand read pool: readers=4'):
            require(marker in log, tag + ': missing fixed read/cache marker')
        row = {k: r[k] for k in ('tag', 'variant', 'prefill_tokens', 'prefill_tps', 'decode_tps',
                                 'full_run_sampled_peak_kib', 'ids_sha256')}
        row['repetition'] = repetition
        rows.append(row)
        memory_rows.append({
            'tag': tag, 'requested_period_ms': m['requested_period_ms'],
            'sample_count': m['sample_count'], 'composite_samples': m['composite_available_sample_count'],
            'interval_ms': m['sample_interval_ms'], 'scan_ms': m['sample_scan_ms'],
            'dma_identity_verified': m['dma_buffer_identity_verified'],
            'peak_kib': r['full_run_sampled_peak_kib'], 'peak_dedup_complete': True,
            'peak_error_fields_zero': True,
        })
        for suffix in ('.json', '.ids', '.log', '.memory.json'):
            add(E / (tag + suffix), 'raw/' + tag + suffix, suffix in ('.log', '.memory.json'))

    computed = []
    for n in (1024, 2048):
        groups = {v: sorted((r for r in rows if r['prefill_tokens'] == n and r['variant'] == v),
                            key=lambda x: x['repetition']) for v in (CONTROL, CANDIDATE)}
        require(all([r['repetition'] for r in g] == [1, 2, 3] for g in groups.values()), 'Unbalanced matrix')
        row = {'tokens': n}
        for metric in ('prefill_tps', 'decode_tps'):
            x, y = [[r[metric] for r in groups[v]] for v in (CONTROL, CANDIDATE)]
            xm, ym = statistics.median(x), statistics.median(y)
            row[metric] = {'control': x, 'candidate': y, 'median_control': xm, 'median_candidate': ym,
                           'median_pct': 100 * (ym / xm - 1), 'pairs_faster': sum(b > a for a, b in zip(x, y)),
                           'paired_deltas_pct': [100 * (b / a - 1) for a, b in zip(x, y)]}
        for label, variant in (('control', CONTROL), ('candidate', CANDIDATE)):
            row[label + '_max_peak_kib'] = max(r['full_run_sampled_peak_kib'] for r in groups[variant])
        computed.append(row)
    summary = read(D / 'root-normal-summary.json')
    require(summary['all_checks_pass'] and summary['summary'] == computed and summary['runs'] == rows,
            'Root normal summary not independently reproduced')

    # Root supplies the exact source/runtime restoration evidence; no device calls here.
    restoration = None
    if a.restoration:
        restoration = read(a.restoration)
        require(restoration['restored_source_sha256'] == base['baseline_sha256'], 'Restored source mismatch')
        require(restoration['restored_runtime']['binaries'] == variants[CONTROL]['binaries'], 'Restored runtime mismatch')
        require(restoration['all_seven_original_binary_hashes_identical'], 'Runtime restoration not verified')
        require(restoration['control_runtime'] == variants[CONTROL], 'Restoration control mismatch')
        require(restoration['rebuilt_dsp_byte_identical'] is False, 'Unexpected rebuilt identity claim')
        add(a.restoration, 'validation/root-restoration.json')
        restored_name = 'lfm-tail-gather-restored-verified'
        restored_artifact = E / (restored_name + '-artifacts.json')
        require(read(restored_artifact) == restoration['restored_runtime'], 'Restored artifact mismatch')
        add(restored_artifact, 'artifacts/' + restored_name + '.json')
        restored_patch = T / (restored_name + '.source.patch')
        require(sha(restored_patch) == restoration['restored_runtime']['source_patch_sha256'] and
                not restored_patch.read_bytes(), 'Restored source delta is not empty')
        add(restored_patch, 'artifacts/' + restored_name + '.source.patch')
    require(a.check_only or restoration is not None, 'Final archive requires the root restoration record')

    # Keep canonical source/fixtures once. Large full disassembly and executable
    # payloads are omitted; exact hashes, commands and focused slices are retained.
    for name in ('README.txt', 'validation.json', 'manifest.json', 'base.json', 'source-hashes.json',
                 'dependency-hashes.json', 'production.patch', 'tests.patch', 'root-applied.patch',
                 'root-dsp-build.log', 'root-normal-jobs.json', 'root-normal-runs.log', 'root-normal-summary.json'):
        add(D / name, 'validation/author/' + name)
    for name in ('root-control/source.json', 'root-frozen/source.json'):
        add(D / name, 'validation/author/' + name)
    add(D / 'baseline' / REL, 'validation/author/baseline/' + REL)
    add(D / 'candidate' / REL, 'validation/author/candidate/' + REL)
    for source in sorted((D / 'candidate/test/htp/host').iterdir()):
        if source.is_file() and source.suffix in {'.c', '.py'}:
            add(source, 'validation/author/candidate/test/htp/host/' + source.name)
    for folder in ('final-san-v2', 'mutants-final'):
        small_tree(D / folder, 'validation/author/' + folder, {'.json', '.log'})
    for name in ('review.txt', 'validation.json', 'manifest.json', 'frozen-hashes.json'):
        add(I / name, 'validation/independent/' + name)
    for folder in ('san', 'mutants'):
        small_tree(I / folder, 'validation/independent/' + folder, {'.json', '.log'})
    for label, root in (('author', D), ('independent', I)):
        small_tree(root / 'codegen', 'validation/' + label + '/codegen',
                   {'.json', '.log', '.su', '.txt', '.s'})
    # Same-source whole-DSP rebuilding was observed to change unrelated code.
    # Preserve this confound and the recovered exact-runtime distinction.
    audit = A / 'restore-review'
    verify_hash_manifest(audit, audit / 'manifest.json')
    codegen = read(audit / 'validation.json')
    require(codegen['normal_tail_comparison_confounded'] and
            codegen['elf_sha256']['control'] == variants[CONTROL]['binaries']['libnntr_hvx_skel.so'] and
            codegen['elf_sha256']['candidate'] == variants[CANDIDATE]['binaries']['libnntr_hvx_skel.so'],
            'Codegen confound evidence mismatch')
    for name in ('review.txt', 'validation.json', 'manifest.json', 'input-source-hashes.json',
                 'compiler-version.txt', 'compare_disassembly.py', 'normalized-got-screen.json',
                 'got-offsets.json', 'size-differences.json'):
        add(audit / name, 'validation/build-identity/' + name)
    for name in ('commands.json', 'results.json', 'match-screen.json',
                 'moe-ring-commands.json', 'moe-ring-results.json'):
        add(audit / 'recompile' / name, 'validation/build-identity/recompile/' + name)
    for source in sorted(audit.glob('*')):
        if source.is_file() and source.suffix in {'.dis', '.diff'} and source.stat().st_size <= 128 * 1024:
            add(source, 'validation/build-identity/focused/' + source.name)
    for name in ('root-restored-dsp-build.log', 'root-restore-audit/objcopy-side-effect.json',
                 'root-restore-audit/sections.json'):
        add(D / name, 'validation/author/' + name)
    for n in (1024, 2048):
        small_tree(E / 'repro/models' / ('lfm2-' + str(n)), 'configuration/lfm2-' + str(n), {'.json'})
    for name in ('bench.py', 'device.json'):
        add(E / name, 'reproduction/' + name)
    add(E / 'validation/memory_monitor.cpp', 'reproduction/memory_monitor.cpp', True)
    for name in ('run_matrix.py', 'summarize_pair.py'):
        add(T / name, 'reproduction/original-' + name)
    for name in ('curate_tail_gather.py', 'verify_archive.py', 'reproduce_host.py'):
        add(A / name, 'reproduction/' + name)
    add(A / 'README.txt', 'README.txt', False)
    generated('summary.json', {
        'decision': 'not_adopted',
        'reason': 'No consistent normal full-model improvement; unrelated same-source DSP codegen also confounds attribution. Neither causal gain nor causal regression is established.',
        'normal': computed, 'all_65_ids_exact': True, 'normal_run_count': 12,
        'runtime_binaries': {k: v['binaries'] for k, v in variants.items()},
        'restoration_verified': restoration is not None,
        'logical_operation_savings_only': author['observed_1024_route'],
        'no_new_repeats_or_profile_runs': True,
        'unrelated_dsp_codegen_confounds_ab': True,
        'restore_review_sha256': sha(audit / 'review.txt'),
    })
    generated('validation/archive-checks.json', {
        'normal_runs_validated': 12, 'normal_summary_recomputed': True,
        'all_raw_memory_hashes_exact': True, '65_ids_each': True,
        'fixed_c16_page0_preload128': True, 'only_dsp_runtime_changed': True,
        'author_and_integrated_patches_replayed': True, 'live_root_clean_not_required': True,
        'no_device_actions': True, 'restoration_verified': restoration is not None,
        'source_sha256': SOURCE_SHA, 'production_patch_sha256': PRODUCTION_SHA,
        'tests_patch_sha256': TESTS_SHA, 'memory_sampling': memory_rows,
        'unrelated_dsp_codegen_confounds_ab': True,
    })
    stored_bytes = sum(len(gzip.compress(data, mtime=0) if compress else data)
                       for data, _, compress in pending.values())
    print(json.dumps({'validated_normal_runs': 12, 'payloads': len(pending),
                      'stored_bytes': stored_bytes, 'restoration_verified': restoration is not None,
                      'normal': computed}, indent=2))
    if a.check_only:
        return
    require(not a.output.exists(), 'Refuse to overwrite existing archive')
    manifest = {'decision': 'not_adopted', 'source_base_commit': control['base_commit'],
                'author_base_commit': base['base_head'],
                'reproduction': {'dependency_sha256': read(D / 'dependency-hashes.json')}, 'files': {}}
    for relative, (data, source, compress) in sorted(pending.items()):
        stored = gzip.compress(data, mtime=0) if compress else data
        dest = a.output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(stored)
        manifest['files'][relative] = {
            'sha256': digest(stored), 'uncompressed_sha256': digest(data), 'gzip': compress,
            'source': source, 'stored_bytes': len(stored), 'uncompressed_bytes': len(data),
        }
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Manifest SHA256', sha(a.output / 'manifest.json'))


if __name__ == '__main__':
    main()
