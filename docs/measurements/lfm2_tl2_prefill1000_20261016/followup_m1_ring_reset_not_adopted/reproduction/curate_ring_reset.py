#!/usr/bin/env python3
"""Validate and curate the rejected fixed-object LFM M1 ring-reset comparison."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

CONTROL = 'lfm-m1-ring-reset-control'
CANDIDATE = 'lfm-m1-ring-reset'
DIAGNOSTIC = 'lfm-m1-ring-reset-diagnostic'
PRODUCTION_SHA = '7803996b40175702f5d9978d29aa8ea31df5ec26282f592fbac6b97ae3ef2a89'
TESTS_SHA = '1e9b6b6394ed911db733ed60259aa4a727c81d7ae92182da827c37adf0707c61'
TRACE_SHA = '9aadb311233191a4c3cd1013ed55fcfd64b4cc8dbbe552088c9f2d0f6ae6c188'
RELS = ['nntrainer/tensor/htp_backend/hmx/' + name for name in
        ('hexkl_dma_ring.c', 'hexkl_dma_ring.h', 'hexkl_mm_u8i4_moe.c')]
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
    'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1', 'NNTR_LFM2_ROUTER_ARM': '1',
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
    for name in ('repo', 'author', 'independent', 'fixed', 'fixed-review', 'device-review', 'scratch', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--restoration', type=Path)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    D, I, T = a.author.resolve(), a.independent.resolve(), a.scratch.resolve()
    F, J, V = a.fixed.resolve(), a.fixed_review.resolve(), a.device_review.resolve()
    E = a.repo.resolve() / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    A = Path(__file__).resolve().parent
    pending, omitted = {}, {}

    def add(source, target, compress=None):
        require(source.is_file(), 'Missing archive input: ' + str(source))
        if compress is None:
            compress = source.suffix in {'.log', '.c', '.cpp', '.h', '.inc', '.patch', '.s', '.su', '.dis', '.diff', '.asm'}
            compress |= source.stat().st_size > 128 * 1024
        target += '.gz' if compress else ''
        data = source.read_bytes()
        if target in pending:
            require(pending[target] == (data, str(source), compress), 'Different duplicate target: ' + target)
            return
        pending[target] = (data, str(source), compress)

    def generated(target, value):
        require(target not in pending, target)
        pending[target] = ((json.dumps(value, indent=2) + '\n').encode(), 'generated', False)

    def small_tree(root, target, extensions):
        require(root.is_dir(), 'Missing evidence directory: ' + str(root))
        for source in sorted(root.rglob('*')):
            if source.is_file() and source.suffix in extensions and source.stat().st_size <= 128 * 1024:
                add(source, target + '/' + str(source.relative_to(root)))

    def omit(source, target):
        omitted[target] = {'sha256': sha(source), 'bytes': source.stat().st_size}

    for folder in (D, I, F, J, V):
        verify_hash_manifest(folder, folder / 'manifest.json')
    require(sha(D / 'production.patch') == PRODUCTION_SHA and sha(D / 'tests.patch') == TESTS_SHA,
            'Wrong frozen production/tests')
    baseline = read(D / 'baseline.json')
    sources = read(D / 'production-hashes.json')['source_sha256']
    author, independent = read(D / 'validation.json'), read(I / 'validation.json')
    require(author['production_patch_sha256'] == independent['production_patch_sha256'] == PRODUCTION_SHA and
            author['tests_patch_sha256'] == independent['tests_patch_sha256'] == TESTS_SHA and
            independent['verdict'] == 'no_unresolved_blocker_in_frozen_scope', 'Review binding')
    require(independent['report_sha256'] == sha(I / 'review.txt'), 'Independent report hash')
    require(author['ring_cases'] == independent['independent_checks']['actual_ring_cases'] == 175 and
            author['whole_moe_cases'] == independent['independent_checks']['whole_m1_caller_cases'] == 31,
            'Host fixture cohort')
    for rel, expected in sources.items():
        require(sha(D / 'candidate' / rel) == expected and
                sha(D / 'baseline' / rel) == baseline['sources'][rel], 'Source snapshot changed')
    for folder in (D / 'checks/final-integrated', I / 'independent-final'):
        validation = read(folder / 'validation.json')
        require(validation['sanitize'] and validation['actual_ring_source'], 'Actual ring sanitizer scope')
        require('175 lifecycle/reset checks, 5957 fills, 50011639 polls PASS' in (folder / 'ring.log').read_text(),
                'Ring raw result')
        require('31' in (folder / 'caller.log').read_text() and 'PASS' in (folder / 'caller.log').read_text(),
                'Caller raw result')
        require(len(validation['mutants']) == 8 and len(validation['caller_mutants']) == 2, 'Mutation counts')
        for row in validation['mutants']:
            log = (folder / ('mutant-' + row['name'] + '.log')).read_text()
            require(row['exit'] != 0 and ('Assertion' in log or 'AddressSanitizer' in log), 'Undetected ring mutant')
        for row in validation['caller_mutants']:
            log = (folder / ('caller-mutant-' + row['name']) / 'check.log').read_text()
            require(row['exit'] != 0 and 'Assertion' in log and 'try_reset_calls' in log, 'Undetected caller mutant')
    with tempfile.TemporaryDirectory(prefix='ring-source-patch-', dir=A) as tmp:
        tree = Path(tmp)
        for rel in RELS:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes((D / 'baseline' / rel).read_bytes())
        subprocess.run(['git', 'apply', str(D / 'production.patch')], cwd=tree, check=True, capture_output=True)
        require({rel: sha(tree / rel) for rel in RELS} == sources, 'Production patch replay')
    with tempfile.TemporaryDirectory(prefix='ring-tests-patch-', dir=A) as tmp:
        tree = Path(tmp)
        for name in ('moe_layer_host_check.c', 'hvx_scalar_stubs.c'):
            rel = Path('test/htp/host') / name
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes((F / 'build/source' / rel).read_bytes())
        subprocess.run(['git', 'apply', str(D / 'tests.patch')], cwd=tree, check=True, capture_output=True)
        for name in ('moe_layer_host_check.c', 'hvx_scalar_stubs.c', 'lfm2_m1_ring_reset_check.c', 'run_lfm2_m1_ring_reset_checks.py'):
            rel = Path('test/htp/host') / name
            source = D / ('candidate' if name in ('moe_layer_host_check.c','hvx_scalar_stubs.c') else '') / rel
            require(sha(tree / rel) == sha(source), 'Tests patch replay')

    fixed = read(F / 'build/validation.json')
    diagnostic_build = read(F / 'diagnostic/build/validation.json')
    require(fixed['status'] == 'PASS' and fixed['frozen_objects'] == 63 and fixed['unchanged_objects'] == 61 and
            fixed['baseline_objects_unchanged'] and fixed['all_other_link_inputs_identical'], 'Frozen object contract')
    require(diagnostic_build['diagnostic_off_object_exact'] and diagnostic_build['runtime_imports_identical'] and
            not diagnostic_build['atomic_library_imports'] and diagnostic_build['metadata_bytes'] == 4 and
            diagnostic_build['diagnostic_patch_sha256'] == TRACE_SHA, 'Diagnostic build contract')
    device = read(V / 'readback/validation.json')
    require(device['status'] == 'PASS' and device['independent_parser'] and not device['new_tests_or_device_runs'],
            'Independent device readback status')
    readback_roots = {
        'nntr-lfm2-m1-ring-reset-20261017': D,
        'nntr-lfm2-ring-fixed-objects-20261017': F,
        'nntr-lfm2-tl2-prefill1000-20261016': T,
        'lfm2_tl2_prefill1000_20261016': E,
    }
    for path, value in device['input_sha256'].items():
        parts = Path(path).parts
        matches = [(parts.index(name), root) for name, root in readback_roots.items() if name in parts]
        require(len(matches) == 1, 'Unknown historical input root: ' + path)
        index, root = matches[0]
        relocated = root.joinpath(*parts[index + 1:])
        require(sha(relocated) == value, 'Device readback input changed: ' + str(relocated))
    variants = {}
    for name, arm in ((CONTROL, 'control'), (CANDIDATE, 'candidate'), (DIAGNOSTIC, 'diagnostic')):
        record = read(E / (name + '-artifacts.json'))
        require(len(record['binaries']) == 7 and sha(T / (name + '.source.patch')) == record['source_patch_sha256'],
                'Runtime artifact/patch')
        require(record['binaries']['libnntr_hvx_skel.so'] == device['controlled_dsp'][arm], 'Wrong DSP pair')
        require({k: v for k, v in record['binaries'].items() if k != 'libnntr_hvx_skel.so'} ==
                device['common_six_host_binaries'], 'Changed host binary')
        variants[name] = record
        add(E / (name + '-artifacts.json'), 'artifacts/' + name + '.json')
        add(T / (name + '.source.patch'), 'artifacts/' + name + '.source.patch')
    require(variants[CONTROL]['base_commit'] == variants[CANDIDATE]['base_commit'], 'Root base mismatch')
    require(device['not_original_3b_binary_comparison'], 'Controlled pair mislabeled')

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
        prefix = 'continue-m1-ring-reset' + ('2048' if n == 2048 else '')
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

    diagnostic_checks = []
    for observed in device['diagnostic_records']:
        n = observed['rows']
        tag = 'continue-m1-ring-reset-diagnostic' + str(n)
        record, memory = read(E / (tag + '.json')), read(E / (tag + '.memory.json'))
        require(record['valid'] and record['instrumented'] and not record['performance_comparison_eligible'] and
                record['profile'] and record['ids_match_reference'] and record['ids_sha256'] == EXPECTED_IDS[n],
                'Diagnostic run scope')
        require(record['ids_sha256'] == sha(E / (tag + '.ids')) and len((E / (tag + '.ids')).read_text().splitlines()) == 65,
                'Diagnostic exact IDs')
        require(record['full_run_memory_report_sha256'] == sha(E / (tag + '.memory.json')) and
                memory['child_waited'] and memory['child_exit_code'] == 0, 'Diagnostic lifetime')
        cmd = ['python3', str(F / 'diagnostic/parse.py'), str(D / (tag + '.live.logcat.txt')),
               '--pid', str(observed['pid']), '--expected-calls', str(observed['independent_total'])]
        parsed = json.loads(subprocess.check_output(cmd, text=True))
        require(parsed['observed_unique_attempts'] == observed['raw_records'] == observed['independent_total'] and
                parsed['hits'] == observed['hits'] and parsed['fallbacks'] == observed['fallbacks'] == 0 and
                parsed['exact_transport_duplicates'] == observed['duplicates'] == 0, 'Actual trace count')
        require(sha(D / (tag + '.live.logcat.txt')) == observed['live_log_sha256'], 'Live source hash')
        diagnostic_checks.append(observed)
        for suffix in ('.json', '.ids', '.log', '.memory.json', '.logcat.txt'):
            add(E / (tag + suffix), 'diagnostic/raw/' + tag + suffix,
                suffix in ('.log', '.memory.json', '.logcat.txt'))
        for suffix in ('.live.logcat.txt', '.parsed.json', '.collector.stderr', '.runner.log'):
            add(D / (tag + suffix), 'diagnostic/raw/' + tag + suffix, suffix != '.parsed.json')

    restoration = None
    if a.restoration:
        restoration = read(a.restoration)
        require(restoration['decision'] == 'not_adopted_no_decode_TPS_gain' and
                restoration['restored_sources'] == baseline['sources'] and
                restoration['all_seven_local_binaries_equal_accepted_ARM_router_runtime'] and
                restoration['phone_seven_hashes_verified'], 'Restoration status')
        accepted = read(D / 'root-accepted-control.json')
        require(restoration['binaries'] == restoration['phone_hashes'] == accepted and len(accepted) == 7,
                'Accepted runtime identity')
        require(accepted['libnntr_hvx_skel.so'] == '3baddb92d14d1d23fe9b6c6a7cacfe3a6cd4708001c110bb8b15e1b0b7276f7c',
                'Original frozen DSP not restored')
        require(all(sha(D / 'root-accepted-control' / name) == value for name, value in accepted.items()),
                'Accepted snapshot hash mismatch')
        require({k:v for k,v in accepted.items() if k != 'libnntr_hvx_skel.so'} == device['common_six_host_binaries'],
                'Unexpected app/core restoration')
        restored_variant = restoration['phone_variant']
        require(read(E / (restored_variant + '-artifacts.json'))['binaries'] == accepted,
                'Restored phone manifest')
        add(a.restoration, 'validation/root-restoration.json')
        add(E / (restored_variant + '-artifacts.json'), 'artifacts/accepted-restored-runtime.json')
    require(a.check_only or restoration is not None, 'Final archive requires final restoration')

    # Frozen author and source reviewer. Historical pre-device status is kept.
    for name in ('README.txt', 'validation.json', 'manifest.json', 'baseline.json', 'production-hashes.json',
                 'production.patch', 'tests.patch', 'tests-hashes.json'):
        add(D / name, 'validation/author/' + name)
    for side in ('baseline', 'candidate'):
        for rel in RELS:
            add(D / side / rel, 'validation/author/' + side + '/' + rel)
    for rel in ('test/htp/host/moe_layer_host_check.c', 'test/htp/host/hvx_scalar_stubs.c'):
        add(D / 'candidate' / rel, 'validation/author/candidate/' + rel)
    for source in sorted((D / 'test/htp/host').iterdir()):
        add(source, 'validation/author/test/htp/host/' + source.name)
    for folder in ('checks/final-integrated', 'checks/existing-suite'):
        small_tree(D / folder, 'validation/author/' + folder, {'.json', '.log'})
    small_tree(D / 'history/pre-host-integration', 'validation/author/history/pre-host-integration',
               {'.json', '.txt', '.patch', '.py'})
    for name in ('review.txt', 'validation.json', 'manifest.json', 'final-hashes.json'):
        add(I / name, 'validation/independent/' + name)
    for folder in ('independent-final', 'existing-suite', 'existing-suite-fixed'):
        small_tree(I / folder, 'validation/independent/' + folder, {'.json', '.log'})
    for label, folder in (('author', D), ('independent', I)):
        for source in sorted((folder / 'codegen').glob('*')):
            if not source.is_file():
                continue
            target = 'validation/' + label + '/codegen/' + source.name
            if source.suffix in {'.json', '.log', '.su', '.c'} or source.name.endswith('.normalized.s') or source.name in (
                    'baseline-hexkl_dma_ring.s', 'candidate-hexkl_dma_ring.s'):
                add(source, target)
            elif source.suffix in {'.o', '.s', '.dis', '.asm'}:
                omit(source, target)

    # Full fixed-object compiler/link plans, exact input hashes, and compact
    # codegen evidence. Executable/complete object payloads are not duplicated.
    for relative in read(F / 'manifest.json'):
        source = F / relative
        if source.suffix in {'.o', '.so'}:
            omit(source, 'fixed-build/' + relative)
        else:
            add(source, 'fixed-build/' + relative)
    add(F / 'manifest.json', 'fixed-build/manifest.json')
    for folder in ('build/control', 'build/candidate', 'build/link-determinism', 'build/threaded-link-attempt',
                   'build/resume-path-filter-failure', 'diagnostic/initial-summary-check-failure'):
        if (F / folder).is_dir():
            small_tree(F / folder, 'fixed-build/' + folder, {'.json', '.log', '.txt'})
    for folder in ('build/normalized-functions', 'build/normalized-addresses-incomplete'):
        small_tree(F / folder, 'fixed-build/' + folder, {'.json'})
    # Capture exact object identities without copying the63-object payload.
    for path, value in read(F / 'build/frozen-object-hashes.json').items():
        require(sha(Path(path)) == value, 'Frozen baseline object changed')
    for source in (F / 'build/objects/candidate').glob('*.o'):
        omit(source, 'fixed-build/build/objects/candidate/' + source.name)
    for relative in read(J / 'manifest.json'):
        add(J / relative, 'fixed-build/independent/' + relative)
    add(J / 'manifest.json', 'fixed-build/independent/manifest.json')
    for arm in ('control', 'candidate'):
        small_tree(J / arm, 'fixed-build/independent/' + arm, {'.json', '.log'})
    for relative in read(V / 'manifest.json'):
        add(V / relative, 'diagnostic/independent/' + relative)
    add(V / 'manifest.json', 'diagnostic/independent/manifest.json')
    # Preserve only evidence of prior nondeterminism here; prior tail verdict/raw
    # remains in its own archive. This pair fixed it by common objects+link flags.
    for name in ('root-normal-jobs.json', 'root-normal-summary.json', 'root-runtime-identity.json',
                 'root-accepted-control.json', 'root-local-restored.json', 'root-run-diagnostic.py'):
        add(D / name, 'validation/root/' + name)
    if (D / 'root-review.txt').exists():
        add(D / 'root-review.txt', 'validation/root/root-review.txt')
    generated('validation/omitted-artifacts.json', omitted)

    # Basic ring checks are self-contained. Optional whole-M1 checks require a
    # source-equivalent external repo and SDK; hash-pin that repo dependency set.
    dependencies = read(F / 'build/baseline-source-hashes.json')
    dependencies = {p: h for p, h in dependencies.items() if p not in RELS and
                    (p.startswith('nntrainer/tensor/htp_backend/') or
                     p.startswith('test/htp/host/') or
                     (p.startswith('test/htp/') and p.endswith('.h')) or
                     (p.startswith('nntrainer/tensor/') and p.endswith('.h')))}
    generated('reproduction/repo-dependencies.json', dependencies)
    for n in (1024, 2048):
        small_tree(E / 'repro/models' / ('lfm2-' + str(n)), 'configuration/lfm2-' + str(n), {'.json'})
    for name in ('bench.py', 'device.json'):
        add(E / name, 'reproduction/' + name)
    add(E / 'validation/memory_monitor.cpp', 'reproduction/memory_monitor.cpp', True)
    for name in ('run_matrix.py', 'summarize_pair.py'):
        add(T / name, 'reproduction/original-' + name)
    for name in ('curate_ring_reset.py', 'verify_archive.py', 'reproduce_host.py'):
        add(A / name, 'reproduction/' + name)
    add(A / 'README.txt', 'README.txt', False)
    generated('summary.json', {
        'decision': 'not_adopted',
        'reason': 'No decode TPS gain established;2048 is lower in all3pairs. Prefill variation is not attributed to this M1-only branch.',
        'normal': computed, 'normal_run_count': 12, 'all_65_ids_exact': True,
        'six_host_binaries_identical': True,
        'controlled_dsp_pair_not_original_3b': True,
        'runtime_binaries': {k: v['binaries'] for k, v in variants.items()},
        'diagnostic_actual_dispatch': diagnostic_checks, 'diagnostic_tps_excluded': True,
        'logical_clear_bytes_not_ddr_or_latency': True,
        'frozen_objects': 63, 'unchanged_common_objects': 61, 'common_linker_setting': '-no-threads',
        'original_threaded_link_failure_preserved': True,
        'three_sources_and_accepted_seven_binary_runtime_restored': restoration is not None,
        'restored_original_dsp_selected_not_recompiled': True,
        'independent_source_build_and_device_readback_reviews': True,
        'new_tests_or_phone_runs_during_curation': False,
    })
    generated('validation/archive-checks.json', {
        'normal_runs_validated': 12, 'normal_summary_recomputed': True,
        'all_raw_memory_hashes_exact': True, '65_ids_each': True,
        'fixed_c16_page0_preload128_readers': True, 'six_host_binaries_identical': True,
        'production_and_tests_patch_replayed': True,
        'device_manifest_sources_replayed_by_frozen_independent_readback': True,
        'frozen_object_hashes_verified': True, 'fixed_build_manifest_and_reviewer_hashes_verified': True,
        'restoration_verified': restoration is not None,
        'source_sha256': sources, 'production_patch_sha256': PRODUCTION_SHA,
        'tests_patch_sha256': TESTS_SHA, 'memory_sampling': memory_rows,
        'diagnostics_parsed_from_live_pid_raw': True,
        'diagnostic_independent_count_from_same_run': True,
        'independent_review_hashes_verified': True,
        'validation_scope': 'Read-only evidence validation. Existing host/compiler/device experiments were not rerun.',
    })
    stored_bytes = sum(len(gzip.compress(data, mtime=0) if compress else data)
                       for data, _, compress in pending.values())
    print(json.dumps({'validated_normal_runs': 12, 'payloads': len(pending),
                      'stored_bytes': stored_bytes, 'restoration_verified': restoration is not None,
                      'normal': computed}, indent=2))
    if a.check_only:
        return
    require(not a.output.exists(), 'Refuse to overwrite existing archive')
    manifest = {'decision': 'not_adopted', 'source_base_commit': variants[CONTROL]['base_commit'],
                'author_source_base_commit': baseline['root_head'], 'fixed_build_revision': fixed['source_revision'],
                'reproduction': {'scope': 'Basic actual-ring fixture uses archived source. Optional whole-M1 fixture needs hash-pinned repo dependencies and external Hexagon SDK/libnative.'},
                'files': {}}
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
