#!/usr/bin/env python3
"""Validate and curate the rejected LFM row-normalization experiment; host files only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

REL = 'Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp'
CONTROL = 'lfm-moe-row-norm-0'
CANDIDATE = 'lfm-moe-row-norm-1'
PRODUCTION_SHA = '61b72f9a7dfd66a529d6642dad76bf9c2f3760ce01fefe78388abd302054fcaf'
TESTS_SHA = '08ae48ae9babb5121b52de4c6d5b0f1e037ea722f2fd0d8cbac185e64ec91623'
SOURCE_SHA = '50c587595d586f6fd2e1e2da930220f236747408992c7406abc091e10cabdca8'
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
    for name in ('repo', 'author', 'scratch', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--restoration', type=Path,
                   help='Root restoration JSON; required to write the final archive')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    D, T = a.author.resolve(), a.scratch.resolve()
    E = a.repo.resolve() / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    A = Path(__file__).resolve().parent
    pending = {}

    def add(source, target, compress=None):
        require(source.is_file(), 'Missing archive input: ' + str(source))
        if compress is None:
            compress = source.suffix in {'.log', '.c', '.cpp', '.h', '.inc', '.patch', '.s', '.su', '.dis', '.diff', '.asm'}
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

    frozen = read(D / 'frozen.json')
    for name, expected in frozen['files'].items():
        require(sha(D / name) == expected, 'Changed frozen input: ' + name)
    require(sha(D / 'production.patch') == PRODUCTION_SHA and
            sha(D / 'tests.patch') == TESTS_SHA and
            sha(D / 'candidate' / REL) == SOURCE_SHA, 'Wrong frozen topic')
    baseline_sha = frozen['files']['before/' + REL]
    author = read(D / 'validation.json')
    require(author['production_source_sha256'] == SOURCE_SHA and
            author['production_patch_sha256'] == PRODUCTION_SHA, 'Author source binding')
    require(author['host']['cases_per_environment'] == 89 and
            author['host']['environments'] == ['1', '0', 'unset', 'true'] and
            author['host']['sanitizers'] == ['address', 'undefined'] and
            not author['host']['suppressions'] and
            sha(D / 'final-host-test.log') == author['host']['log_sha256'], 'Host cohort')
    require((D / 'final-host-test.log').read_text().count('89 cases') == 4, 'Host raw cases')
    mutants = read(D / 'checks/final-mutations/results.json')
    require(len(mutants) == author['host']['runtime_mutants_rejected'] == 8 and
            sha(D / 'checks/final-mutations/results.json') == author['host']['mutation_results_sha256'],
            'Mutation results binding')
    for name, result in mutants.items():
        log = (D / 'checks/final-mutations' / name / 'run.log').read_text()
        require(result['detected'] and result['returncode'] != 0 and
                result['expected_assertion'] in log, 'Mutation not rejected: ' + name)
    require(all(author['builds'].values()) and author['codegen']['original_normRows_bytes_identical'] and
            author['codegen']['incremental_caller_stack_before'] ==
            author['codegen']['incremental_caller_stack_candidate'] == 3008, 'Codegen scope')
    for side in ('before', 'candidate'):
        require(sha(D / 'codegen' / side / 'normRows.bin') ==
                author['codegen']['original_normRows_code_sha256'], 'Original normRows code differs')

    # Preserve the initial harness failure separately from the corrected native run.
    require('Weight initializer cannot be none' in (D / 'root-native-1-8.log').read_text(),
            'Missing initial native failure')
    native = read(D / 'root-native-fixed-results.json')
    require(native['binary_sha256'] == sha(D / 'checks/native-v2/check') ==
            author['native']['binary_sha256'], 'Corrected native binary differs')
    require(len(native['runs']) == 4 and {(r['flag'], r['threads']) for r in native['runs']} ==
            {(0, 1), (0, 8), (1, 1), (1, 8)}, 'Native matrix')
    for run in native['runs']:
        logpath = D / run['log']
        log = logpath.read_text()
        require(run['returncode'] == 0 and sha(logpath) == run['log_sha256'] and
                f"ROW_NORM_NATIVE 30 cases threads={run['threads']} env={run['flag']} PASS" in log and
                len(re.findall(r'^EXACT M=(1024|2048) mode=[012] used=[01] PASS$', log, re.M)) == 6,
                'Corrected native log mismatch')
    for folder in ('checks/final-san', 'checks/final-native', 'checks/native-v2'):
        commands = read(D / folder / 'commands.json')
        require(commands['source_sha256'] == SOURCE_SHA, 'Fixture source binding: ' + folder)
        for name, expected in commands['extracts'].items():
            require(sha(D / folder / name) == expected, 'Extract changed: ' + folder + '/' + name)
    commands = read(D / 'checks/final-san/commands.json')
    dependency = {'nntrainer/tensor/cpu_backend/arm/neon_impl.cpp': commands['neon_source_sha256']}
    require(all(sha(a.repo / name) == expected for name, expected in dependency.items()), 'Core math dependency')

    # Runtime OFF and ON are the same complete binary set. Only the option differs.
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
    require(variants[CONTROL]['binaries'] == variants[CANDIDATE]['binaries'], 'OFF/ON runtime differs')
    require(variants[CONTROL]['base_commit'] == variants[CANDIDATE]['base_commit'] == frozen['base'],
            'Runtime source bases differ')
    require(read(D / 'root-frozen/runtime.json') == variants[CONTROL], 'Frozen runtime manifest differs')
    for name in ('libcausallm.so', 'nntr_causallm'):
        require(sha(D / 'root-frozen' / name) == variants[CONTROL]['binaries'][name], 'Frozen app differs')
    require(sha(D / 'root-frozen/lfm2_moe_layer.cpp') == SOURCE_SHA and
            sha(D / 'root-control/lfm2_moe_layer.cpp') == baseline_sha, 'Frozen source differs')
    base_source = subprocess.check_output(['git', '-C', str(a.repo), 'show', frozen['base'] + ':' + REL])
    require(digest(base_source) == baseline_sha, 'Recorded commit source differs')
    for patch in (D / 'production.patch', D / 'root-applied.patch', T / (CONTROL + '.source.patch'),
                  T / (CANDIDATE + '.source.patch')):
        with tempfile.TemporaryDirectory(prefix='norm-patch-proof-', dir=A) as temporary:
            target = Path(temporary) / REL
            target.parent.mkdir(parents=True)
            target.write_bytes(base_source)
            subprocess.run(['git', 'apply', str(patch)], cwd=temporary, check=True, capture_output=True)
            require(sha(target) == SOURCE_SHA, 'Patch does not reproduce candidate: ' + str(patch))
    with tempfile.TemporaryDirectory(prefix='norm-fixture-proof-', dir=A) as temporary:
        subprocess.run(['git', 'apply', str(D / 'tests.patch')], cwd=temporary, check=True, capture_output=True)
        for source in sorted((D / 'test/htp/host').glob('*')):
            require(sha(Path(temporary) / 'test/htp/host' / source.name) == sha(source),
                    'Fixture patch differs: ' + source.name)

    # CPU-only timing prescreen is archived, never merged with normal model timing.
    verify_hash_manifest(D, D / 'root-prescreen-manifest.json')
    micro_meta = read(D / 'root-micro.json')
    require(micro_meta['returncode'] == 0 and micro_meta['log_sha256'] == sha(D / 'root-micro.log'),
            'Micro raw log binding')
    micro_log = (D / 'root-micro.log').read_text()
    micro = []
    for rows_n in (1024, 2048):
        values = {}
        for option in (0, 1):
            samples = re.findall(rf'^SAMPLE rows={rows_n} rep=(\d+) parallel={option} us=([\d.]+)$',
                                 micro_log, re.M)
            require(len(samples) == 31 and [int(k) for k, _ in samples] == list(range(31)), 'Micro sample matrix')
            med = statistics.median(float(v) for _, v in samples)
            text_median = re.findall(rf'^MEDIAN rows={rows_n} parallel={option} us=([\d.]+)$', micro_log, re.M)
            require(len(text_median) == 1 and float(text_median[0]) == med, 'Micro median')
            values[str(option)] = {'median_us': med, 'samples': 31}
        micro.append({'rows': rows_n, 'parallel': values})

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
        prefix = 'continue-moe-row-norm' + ('2048' if n == 2048 else '')
        require(tag == f'{prefix}-{side}-r{repetition}', 'Unexpected normal-run tag')
        r, m = read(E / (tag + '.json')), read(E / (tag + '.memory.json'))
        require(r['tag'] == tag and r['variant'] == job['variant'] and r['model'] == job['model'] ==
                ('model' if n == 1024 else 'model2048'), tag + ': identity')
        require(r['valid'] and r['performance_comparison_eligible'] and not r['instrumented'] and
                not r['profile'] and r['returncode'] == 0 and not r['concurrent_processes'], tag + ': invalid run')
        require(r['moe_cache_experts'] == 16 and r['prefill_tokens'] == n and r['decode_tokens'] == 64,
                tag + ': shape/cache')
        require(set(r['environment']) == set(ENV) | {'LD_LIBRARY_PATH', 'ADSP_LIBRARY_PATH', 'NNTR_GENERATED_IDS', 'NNTR_LFM2_MOE_ROW_NORM'},
                tag + ': unexpected environment')
        require(r['environment']['NNTR_LFM2_MOE_ROW_NORM'] == str(side), tag + ': option')
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

    # Strict actual-dispatch evidence comes from a separate diagnostic application.
    # App stdout/stderr, rather than logcat, contains these completion markers.
    import importlib.util
    import sys
    sys.dont_write_bytecode = True
    specification = importlib.util.spec_from_file_location('row_norm_dispatch_parser', D / 'diagnostic/parse.py')
    parser = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(parser)
    diagnostic_checks = []
    diagnostic_manifest = read(E / 'lfm-moe-row-norm-diagnostic-artifacts.json')
    diag_variant = 'lfm-moe-row-norm-diagnostic'
    require(sha(T / (diag_variant + '.source.patch')) == diagnostic_manifest['source_patch_sha256'],
            'Diagnostic runtime patch hash')
    normal_binaries = variants[CONTROL]['binaries']
    require([k for k in normal_binaries if normal_binaries[k] != diagnostic_manifest['binaries'][k]] ==
            ['libcausallm.so'], 'Diagnostic changed unexpected runtime')
    diagnostic_source = read(D / 'root-diagnostic-source.json')
    require(diagnostic_source['normal_source_sha256'] == SOURCE_SHA, 'Diagnostic base source')
    for patch in (D / 'root-diagnostic.patch', T / (diag_variant + '.source.patch')):
        with tempfile.TemporaryDirectory(prefix='norm-diagnostic-proof-', dir=A) as temporary:
            target = Path(temporary) / REL
            target.parent.mkdir(parents=True)
            target.write_bytes(base_source)
            subprocess.run(['git', 'apply', str(patch)], cwd=temporary, check=True, capture_output=True)
            require(sha(target) == diagnostic_source['source_sha256'], 'Diagnostic patch/source mismatch')
    diagnostic_author = read(D / 'diagnostic/validation.json')
    require(sha(D / 'diagnostic/diagnostic.patch') == diagnostic_author['patch_sha256'] and
            sha(D / 'diagnostic/candidate' / REL) == diagnostic_author['source_sha256'] and
            diagnostic_author['off_disassembly_exact'] and diagnostic_author['off_and_on_ndk_compile_pass'] and
            diagnostic_author['parser_valid'] == 4 and diagnostic_author['parser_negative'] == 32,
            'Diagnostic author checks')
    for n in (1024, 2048):
        tag = 'continue-moe-row-norm-profile' + str(n)
        record = read(E / (tag + '.json'))
        require(record['valid'] and record['instrumented'] and record['profile'] and
                not record['performance_comparison_eligible'] and record['returncode'] == 0 and
                record['variant'] == diag_variant and record['prefill_tokens'] == n and
                record['moe_cache_experts'] == 16 and record['environment']['NNTR_HTP_PAGE_CACHE'] == '0' and
                record['environment']['NNTR_LFM2_MOE_ROW_NORM'] == '1', 'Diagnostic run scope')
        require(record['ids_match_reference'] and record['ids_sha256'] == EXPECTED_IDS[n] ==
                sha(E / (tag + '.ids')) and len((E / (tag + '.ids')).read_text().splitlines()) == 65,
                'Diagnostic IDs')
        memory = read(E / (tag + '.memory.json'))
        require(record['full_run_memory_report_sha256'] == sha(E / (tag + '.memory.json')) and
                record['full_run_sampled_peak_kib'] * 1024 ==
                memory['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] and
                memory['child_waited'] and memory['child_exit_code'] == 0, 'Diagnostic memory binding')
        check = parser.parse((E / (tag + '.log')).read_text(), n, 1)
        require(parser.TAG not in (E / (tag + '.logcat.txt')).read_text(), 'Unexpected duplicate marker stream')
        diagnostic_checks.append(check)
        for suffix in ('.json', '.ids', '.log', '.memory.json', '.logcat.txt'):
            add(E / (tag + suffix), 'diagnostic/raw/' + tag + suffix,
                suffix in ('.log', '.memory.json', '.logcat.txt'))
    require(read(D / 'root-diagnostic-dispatch.json')['all_pass'] and
            read(D / 'root-diagnostic-dispatch.json')['checks'] == diagnostic_checks, 'Dispatch summary mismatch')
    removed = read(D / 'root-diagnostic-removed.json')
    require(removed['diagnostic_removed'] and removed['normal_binaries_byte_identical_after_rebuild'] and
            removed['normal_source_sha256'] == SOURCE_SHA and
            all(removed[k] == normal_binaries[k] for k in ('libcausallm.so', 'nntr_causallm')),
            'Diagnostic removal')
    add(E / (diag_variant + '-artifacts.json'), 'artifacts/' + diag_variant + '.json')
    add(T / (diag_variant + '.source.patch'), 'artifacts/' + diag_variant + '.source.patch')

    restoration = None
    if a.restoration:
        restoration = read(a.restoration)
        require(restoration['decision'] == 'not_adopted' and
                restoration['source_restored_sha256'] == baseline_sha and
                restoration['all_seven_original_binary_hashes_identical'], 'Restoration status/source')
        original = restoration['control_runtime']
        restored = restoration['restored_runtime']
        require(original['binaries'] == restored['binaries'] and len(restored['binaries']) == 7,
                'Restored runtime does not match original')
        require([k for k in normal_binaries if normal_binaries[k] != restored['binaries'][k]] == ['libcausallm.so'],
                'Restoration changes an unexpected runtime')
        for name in ('libcausallm.so', 'nntr_causallm'):
            require(sha(D / 'root-control' / name) == restored['binaries'][name], 'Original runtime binding')
        restored_variant = 'lfm-moe-row-norm-restored'
        require(read(E / (restored_variant + '-artifacts.json')) == restored, 'Restored artifact differs')
        patch = T / (restored_variant + '.source.patch')
        require(not patch.read_bytes() and sha(patch) == restored['source_patch_sha256'], 'Restore has a delta')
        add(a.restoration, 'validation/root-restoration.json')
        add(E / (restored_variant + '-artifacts.json'), 'artifacts/' + restored_variant + '.json')
        add(patch, 'artifacts/' + restored_variant + '.source.patch')
    require(a.check_only or restoration is not None, 'Final archive requires root restoration evidence')

    # Canonical exact source/tests, report history and bounded text evidence only.
    # No executable or complete multi-megabyte disassembly payload is duplicated.
    for name in ('README.txt', 'validation.json', 'frozen.json', 'format-check.json', 'initial-source.json',
                 'production.patch', 'tests.patch', 'root-applied.patch', 'final-host-test.log',
                 'final-mutations.log', 'final-native-build.log', 'native-v2-build.log'):
        add(D / name, 'validation/author/' + name)
    for side in ('before', 'candidate'):
        add(D / side / REL, 'validation/author/' + side + '/' + REL)
    for source in sorted((D / 'test/htp/host').iterdir()):
        add(source, 'validation/author/test/htp/host/' + source.name)
    for folder in ('checks/final-san', 'checks/final-native', 'checks/native-v2'):
        small_tree(D / folder, 'validation/author/' + folder, {'.json', '.inc', '.cpp', '.sha256', '.log'})
    small_tree(D / 'checks/final-mutations', 'validation/author/checks/final-mutations', {'.json', '.log'})
    # Preserve per-TU commands, stack records and only focused disassembly slices.
    excluded_codegen = {}
    for source in sorted((D / 'codegen').rglob('*')):
        if not source.is_file():
            continue
        relative = str(source.relative_to(D))
        if source.suffix in {'.json', '.log', '.su'} or source.name.startswith('row-norm-'):
            add(source, 'validation/author/' + relative)
        elif source.suffix in {'.o', '.bin', '.asm', '.txt'}:
            excluded_codegen[relative] = {'sha256': sha(source), 'bytes': source.stat().st_size}
    generated('validation/author/codegen/omitted-large-artifacts.json', excluded_codegen)
    for name in ('root-app-build.log', 'root-normal-rebuild.log', 'root-baseline-rebuild.log',
                 'root-normal-jobs.json', 'root-normal-runs.log', 'root-normal-summary.json', 'root-review.txt',
                 'root-native-1-8.log', 'root-native-fixed-results.json', 'root-prescreen-manifest.json',
                 'root-micro.log', 'root-micro.json', 'micro.cpp', 'baseline.inc',
                 'micro-build-command.json', 'micro-build.log', 'root-frozen/runtime.json',
                 'root-diagnostic-build.log', 'root-diagnostic-dispatch.json', 'root-diagnostic-jobs.json',
                 'root-diagnostic-runs.log', 'root-diagnostic-source.json', 'root-diagnostic-removed.json',
                 'root-diagnostic.patch'):
        add(D / name, 'validation/root/' + name)
    for run in native['runs']:
        add(D / run['log'], 'validation/root/' + run['log'])
    for name in ('README.txt', 'validation.json', 'diagnostic.patch', 'parse.py', 'parser_check.py'):
        add(D / 'diagnostic' / name, 'diagnostic/' + name)
    add(D / 'diagnostic/candidate' / REL, 'diagnostic/candidate/' + REL)
    for option in ('off', 'on'):
        small_tree(D / 'diagnostic' / option, 'diagnostic/' + option, {'.json', '.log', '.su'})
    for n in (1024, 2048):
        small_tree(E / 'repro/models' / ('lfm2-' + str(n)), 'configuration/lfm2-' + str(n), {'.json'})
    for name in ('bench.py', 'device.json'):
        add(E / name, 'reproduction/' + name)
    add(E / 'validation/memory_monitor.cpp', 'reproduction/memory_monitor.cpp', True)
    for name in ('run_matrix.py', 'summarize_pair.py'):
        add(T / name, 'reproduction/original-' + name)
    for name in ('curate_row_norm.py', 'verify_archive.py', 'reproduce_host.py'):
        add(A / name, 'reproduction/' + name)
    add(A / 'README.txt', 'README.txt', False)
    generated('summary.json', {
        'decision': 'not_adopted',
        'reason': 'Normal full-model prefill medians decreased at both prompt sizes; CPU-only prescreen improvement did not transfer to the model. No specific slowdown mechanism is established.',
        'normal': computed, 'normal_run_count': 12, 'all_65_ids_exact': True,
        'all_seven_binaries_identical_off_on': True,
        'runtime_binaries': {k: v['binaries'] for k, v in variants.items()},
        'diagnostic_actual_dispatch': diagnostic_checks,
        'diagnostic_tps_excluded': True, 'cpu_only_prescreen': micro,
        'cpu_prescreen_does_not_predict_model_gain': True,
        'restoration_verified': restoration is not None,
        'native_initial_harness_failure_preserved': True,
        'corrected_native_runs': 4, 'corrected_native_cases_each': 30,
        'independent_agent_source_review_performed': False,
        'new_tests_or_phone_runs_during_curation': False,
    })
    generated('validation/archive-checks.json', {
        'normal_runs_validated': 12, 'normal_summary_recomputed': True,
        'all_raw_memory_hashes_exact': True, '65_ids_each': True,
        'fixed_c16_page0_preload128': True, 'all_seven_binaries_identical_off_on': True,
        'author_and_integrated_patches_replayed': True, 'tests_patch_replayed': True,
        'live_root_clean_not_required': True, 'no_device_actions': True,
        'restoration_verified': restoration is not None,
        'source_sha256': SOURCE_SHA, 'production_patch_sha256': PRODUCTION_SHA,
        'tests_patch_sha256': TESTS_SHA, 'memory_sampling': memory_rows,
        'diagnostics_parsed_from_raw_app_logs': True,
        'native_log_hashes_verified': True, 'initial_native_failure_preserved': True,
        'cpu_micro_medians_recomputed': True,
        'validation_scope': 'Read-only evidence validation; existing author/root checks were not rerun.',
    })
    stored_bytes = sum(len(gzip.compress(data, mtime=0) if compress else data)
                       for data, _, compress in pending.values())
    print(json.dumps({'validated_normal_runs': 12, 'payloads': len(pending),
                      'stored_bytes': stored_bytes, 'restoration_verified': restoration is not None,
                      'normal': computed}, indent=2))
    if a.check_only:
        return
    require(not a.output.exists(), 'Refuse to overwrite existing archive')
    manifest = {'decision': 'not_adopted', 'source_base_commit': frozen['base'],
                'author_source_base_equivalent': frozen['source_base_equivalent'],
                'reproduction': {'dependency_sha256': dependency,
                                 'simde_version': '0.8.2',
                                 'native_build_core_sha256': read(D / 'checks/native-v2/commands.json')['runtime_nntrainer_sha256'],
                                 'native_device_runtime_commands': 'validation/root/root-native-fixed-results.json'},
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
