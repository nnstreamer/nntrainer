#!/usr/bin/env python3
"""Validate and curate the unadopted LFM prediction register experiment; host files only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

RELS = ['nntrainer/tensor/htp_backend/htp_router_predict.h', 'nntrainer/tensor/htp_backend/htp_compute_ops.cpp']
CONTROL = 'lfm-predict-register-0'
CANDIDATE = 'lfm-predict-register-1'
PRODUCTION_SHA = 'ba6055d746a37aecbb98f38a260ba0671071c285f6a3d426ebf5ec3a6c364f3d'
TESTS_SHA = '7c24b11915f44d30ed6933ce42ee79b19f0122b643741135c7f329970c32401e'
SOURCE_SHA = {RELS[0]: 'cb9d5e9ac8fd04844b02652e8a782ac69805c93df3aa5c7511935fb54f4e0836', RELS[1]: '1c470ac6a9edcc1cbb91f9c1ccd85a5a18132ead41921636dfca4bbf4bf83418'}
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

    verify_hash_manifest(D, D / 'manifest.json')
    require(sha(D / 'production.patch') == PRODUCTION_SHA and sha(D / 'tests.patch') == TESTS_SHA,
            'Wrong frozen production/tests')
    baseline = read(D / 'baseline.json')
    for rel, expected in SOURCE_SHA.items():
        require(sha(D / 'candidate' / rel) == expected and
                sha(D / 'baseline' / rel) == baseline['files'][rel], 'Source snapshot changed')
    author, independent = read(D / 'validation.json'), read(I / 'validation.json')
    require(author['production_sha256'] == independent['production']['production.patch'] == PRODUCTION_SHA and
            author['tests_patch_sha256'] == TESTS_SHA and independent['verdict'] == 'no_blocker', 'Review binding')
    require(independent['review_sha256'] == sha(I / 'review.txt'), 'Independent review text hash')
    require(all(independent['production'][rel] == SOURCE_SHA[rel] for rel in RELS), 'Independent source binding')
    require(author['host']['cases'] == independent['host_cases'] == 130 and
            author['host']['raw_score_sets'] == independent['raw_score_arrays'] == 125 and
            author['host']['actual_callback_jobs'] == independent['actual_callback_jobs'] == 144 and
            author['host']['env_shape_guards'] == independent['env_shape_guards'] == 12,
            'Host fixture cohort')
    for logfile in (D / 'host-final.log', D / 'fallback-san.log', I / 'host/check.log', I / 'fallback/check.log'):
        log = logfile.read_text()
        require(all(str(n) in log for n in (130, 125, 144, 12)) and 'PASS' in log, 'Host raw evidence')
    for folder in (D / 'checks/mutations-final', I / 'mutations'):
        results = read(folder / 'results.json')
        require(len(results) == 9, 'Mutation cohort')
        for row in results:
            name = row['name']
            log = (folder / name / 'run.log').read_text()
            require(row['runtime_killed'] and row['exit_code'] != 0 and re.search(row['expected'], log),
                    'Mutant not rejected: ' + name)
    require(independent['author_final_object_equal'] and
            sha(I / 'codegen/candidate/htp_compute_ops.o') ==
            sha(D / 'codegen/candidate-final/htp_compute_ops.o') == independent['candidate_object_sha256'],
            'Whole-core object identity')

    variants = {}
    for name in (CONTROL, CANDIDATE):
        record = read(E / (name + '-artifacts.json'))
        require(len(record['binaries']) == 7 and
                all(re.fullmatch('[0-9a-f]{64}', v) for v in record['binaries'].values()), 'Runtime SHA set')
        require(sha(T / (name + '.source.patch')) == record['source_patch_sha256'], 'Runtime source patch hash')
        variants[name] = record
        add(E / (name + '-artifacts.json'), 'artifacts/' + name + '.json')
        add(T / (name + '.source.patch'), 'artifacts/' + name + '.source.patch')
    require(variants[CONTROL]['binaries'] == variants[CANDIDATE]['binaries'] and
            variants[CONTROL]['base_commit'] == variants[CANDIDATE]['base_commit'], 'Normal OFF/ON differ')
    frozen = read(D / 'root-frozen.json')
    for name, expected in frozen.items():
        require(sha(D / 'root-frozen' / name) == expected, 'Frozen runtime/source differs: ' + name)
    require({k: frozen[k] for k in variants[CONTROL]['binaries']} == variants[CONTROL]['binaries'],
            'Runtime/frozen identity')
    for rel in RELS:
        require(frozen[Path(rel).name] == SOURCE_SHA[rel] and
                sha(D / 'root-control' / Path(rel).name) == baseline['files'][rel], 'Root source snapshot')
    base_source = {rel: subprocess.check_output(['git', '-C', str(a.repo), 'show',
                                               variants[CONTROL]['base_commit'] + ':' + rel]) for rel in RELS}
    require({rel: digest(data) for rel, data in base_source.items()} == baseline['files'], 'Source-equivalent base')
    for patch in (D / 'production.patch', D / 'root-applied.patch', T / (CONTROL + '.source.patch'),
                  T / (CANDIDATE + '.source.patch')):
        with tempfile.TemporaryDirectory(prefix='predict-patch-proof-', dir=A) as temporary:
            for rel, data in base_source.items():
                target = Path(temporary) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            subprocess.run(['git', 'apply', str(patch)], cwd=temporary, check=True, capture_output=True)
            require({rel: sha(Path(temporary) / rel) for rel in RELS} == SOURCE_SHA,
                    'Applied source mismatch: ' + str(patch))
    with tempfile.TemporaryDirectory(prefix='predict-fixture-proof-', dir=A) as temporary:
        subprocess.run(['git', 'apply', str(D / 'tests.patch')], cwd=temporary, check=True, capture_output=True)
        for source in sorted((D / 'test/htp/host').iterdir()):
            require(sha(Path(temporary) / 'test/htp/host' / source.name) == sha(source), 'Fixture patch differs')

    native = read(D / 'root-native-results.json')
    require(native['runs'] == author['native']['runs'] and len(native['runs']) == 2, 'Native metadata binding')
    for run in native['runs']:
        logfile = D / ('root-native-' + run['name'] + '.log')
        require(run['returncode'] == 0 and sha(logfile) == run['log_sha256'], 'Native result/hash')
        require(sha(D / 'checks/native-final' / run['name']) == run['sha256'], 'Native binary hash')
    require('PASS' in (D / 'root-native-check.log').read_text(), 'Native fixture did not pass')
    micro_log = (D / 'root-native-micro.log').read_text()
    require(micro_log.startswith('PREDICT_MICRO parity=37 exact;'), 'Micro parity missing')
    micro = []
    for mode, field in (('warm', 'warm_off_on_us'), ('37layers', 'sweep37_off_on_us')):
        records = re.findall(rf'^PREDICT_MICRO mode={mode} pair=(\d+) order=(AB|BA) off_us=([\d.]+) on_us=([\d.]+) ratio=([\d.]+)$', micro_log, re.M)
        require(len(records) == 31 and [int(x[0]) for x in records] == list(range(31)) and
                all(x[1] == ('AB' if i % 2 == 0 else 'BA') for i, x in enumerate(records)), 'Micro paired order')
        off, on = [statistics.median(float(x[k]) for x in records) for k in (2, 3)]
        require([off, on] == author['micro'][field], 'Micro median differs')
        micro.append({'mode': mode, 'pairs': 31, 'off_median_us': off, 'on_median_us': on,
                      'scope': 'CPU0 only; synthetic input, no real queue/I/O/DSP/concurrent model'})

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
        prefix = 'continue-predict-register' + ('2048' if n == 2048 else '')
        require(tag == f'{prefix}-{side}-r{repetition}', 'Unexpected normal-run tag')
        r, m = read(E / (tag + '.json')), read(E / (tag + '.memory.json'))
        require(r['tag'] == tag and r['variant'] == job['variant'] and r['model'] == job['model'] ==
                ('model' if n == 1024 else 'model2048'), tag + ': identity')
        require(r['valid'] and r['performance_comparison_eligible'] and not r['instrumented'] and
                not r['profile'] and r['returncode'] == 0 and not r['concurrent_processes'], tag + ': invalid run')
        require(r['moe_cache_experts'] == 16 and r['prefill_tokens'] == n and r['decode_tokens'] == 64,
                tag + ': shape/cache')
        require(set(r['environment']) == set(ENV) | {'LD_LIBRARY_PATH', 'ADSP_LIBRARY_PATH', 'NNTR_GENERATED_IDS', 'NNTR_LFM2_PREDICT_REGISTER_ACCUM'},
                tag + ': unexpected environment')
        require(r['environment']['NNTR_LFM2_PREDICT_REGISTER_ACCUM'] == str(side), tag + ': option')
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

    # Separate trace core; actual fast helper increments after predictor work.
    diagnostic_manifest = read(E / 'lfm-predict-register-diagnostic-artifacts.json')
    diagnostic_variant = 'lfm-predict-register-diagnostic'
    require(sha(T / (diagnostic_variant + '.source.patch')) == diagnostic_manifest['source_patch_sha256'],
            'Diagnostic runtime patch')
    normal_binaries = variants[CONTROL]['binaries']
    require([k for k in normal_binaries if normal_binaries[k] != diagnostic_manifest['binaries'][k]] ==
            ['libnntrainer.so'], 'Diagnostic changes unexpected runtime')
    diagnostic_source = read(D / 'root-diagnostic-source.json')
    for rel in RELS:
        require(sha(D / ('root-diagnostic-' + Path(rel).name)) == diagnostic_source[Path(rel).name],
                'Diagnostic source snapshot')
    for patch in (D / 'root-diagnostic.patch', T / (diagnostic_variant + '.source.patch')):
        with tempfile.TemporaryDirectory(prefix='predict-trace-proof-', dir=A) as temporary:
            for rel, data in base_source.items():
                target = Path(temporary) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            subprocess.run(['git', 'apply', str(patch)], cwd=temporary, check=True, capture_output=True)
            require({Path(rel).name: sha(Path(temporary) / rel) for rel in RELS} == diagnostic_source,
                    'Diagnostic source patch mismatch')
    diagnostic_checks = []
    for option in (0, 1):
        tag = 'continue-predict-register-diagnostic-' + str(option)
        record = read(E / (tag + '.json'))
        require(record['valid'] and record['instrumented'] and record['profile'] and
                not record['performance_comparison_eligible'] and record['returncode'] == 0 and
                record['variant'] == diagnostic_variant and record['prefill_tokens'] == 1024 and
                record['moe_cache_experts'] == 16 and record['environment']['NNTR_HTP_PAGE_CACHE'] == '0' and
                record['environment']['NNTR_LFM2_PREDICT_REGISTER_ACCUM'] == str(option), 'Diagnostic scope')
        require(record['ids_match_reference'] and record['ids_sha256'] == EXPECTED_IDS[1024] ==
                sha(E / (tag + '.ids')) and len((E / (tag + '.ids')).read_text().splitlines()) == 65,
                'Diagnostic IDs')
        memory = read(E / (tag + '.memory.json'))
        require(record['full_run_memory_report_sha256'] == sha(E / (tag + '.memory.json')) and
                record['full_run_sampled_peak_kib'] * 1024 ==
                memory['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] and
                memory['child_waited'] and memory['child_exit_code'] == 0, 'Diagnostic memory')
        with tempfile.TemporaryDirectory(prefix='predict-parser-', dir=A) as temporary:
            output = Path(temporary) / 'result.json'
            subprocess.run(['python3', str(D / 'diagnostic/parse.py'), '--log', str(E / (tag + '.log')),
                            '--enabled', str(option), '--tokens', '64', '--moe-layers', '38',
                            '--output', str(output)], check=True, capture_output=True)
            check = read(output)
        require(check == read(D / ('root-diagnostic-' + str(option) + '.json')), 'Diagnostic summary differs')
        diagnostic_checks.append(check)
        for suffix in ('.json', '.ids', '.log', '.memory.json', '.logcat.txt'):
            add(E / (tag + suffix), 'diagnostic/raw/' + tag + suffix,
                suffix in ('.log', '.memory.json', '.logcat.txt'))
    removed = read(D / 'root-diagnostic-removed.json')
    require(removed['all_four_core_binaries_byte_identical_to_normal_frozen'] and
            removed['normal_libnntrainer_sha256'] == normal_binaries['libnntrainer.so'],
            'Diagnostic removal not verified')
    add(E / (diagnostic_variant + '-artifacts.json'), 'artifacts/' + diagnostic_variant + '.json')
    add(T / (diagnostic_variant + '.source.patch'), 'artifacts/' + diagnostic_variant + '.source.patch')

    restoration = None
    if a.restoration:
        restoration = read(a.restoration)
        require(restoration['decision'] == 'not_adopted_no_consistent_decode_TPS_improvement' and
                restoration['sources'] == {Path(k).name: v for k, v in baseline['files'].items()} and
                restoration['phone_seven_hashes_verified'], 'Restoration status/source')
        core_files = ('libnntrainer.so', 'libccapi-nntrainer.so', 'libc++_shared.so', 'libsdkl.so')
        core = restoration['all_four_restored_core_binaries']
        original = restoration['original_seven_binary_phone_runtime']
        require(set(core) == set(core_files) and len(original) == 7 and
                all(core[k] == original[k] == sha(D / 'root-control' / k) for k in core_files), 'Core restore hash')
        require([k for k in normal_binaries if normal_binaries[k] != original[k]] == ['libnntrainer.so'],
                'Unexpected original runtime difference')
        require(all(original[k] == sha(D / 'root-frozen' / k) for k in original if k not in core_files),
                'Original app/DSP snapshot')
        phone_variant = restoration['phone_variant']
        restored_record = read(E / (phone_variant + '-artifacts.json'))
        require(restored_record['binaries'] == original, 'Verified phone runtime manifest differs')
        add(a.restoration, 'validation/root-restoration.json')
        add(E / (phone_variant + '-artifacts.json'), 'artifacts/original-phone-runtime.json')
    require(a.check_only or restoration is not None, 'Final archive requires restoration evidence')

    for name in ('README.txt', 'validation.json', 'manifest.json', 'baseline.json', 'production-hashes.json',
                 'production.patch', 'tests.patch', 'host-final.log', 'fallback-san.log', 'mutations-final.log',
                 'native-final-build.log'):
        add(D / name, 'validation/author/' + name)
    for side in ('baseline', 'candidate'):
        for rel in RELS:
            add(D / side / rel, 'validation/author/' + side + '/' + rel)
    for source in sorted((D / 'test/htp/host').iterdir()):
        add(source, 'validation/author/test/htp/host/' + source.name)
    for folder in ('checks/host-final', 'checks/native-final', 'checks/fallback-san'):
        small_tree(D / folder, 'validation/author/' + folder, {'.json', '.inc', '.cpp', '.h', '.log'})
    for folder in ('checks/mutations-final', 'checks/existing'):
        small_tree(D / folder, 'validation/author/' + folder, {'.json', '.log'})
    for name in ('production-hashes.json', 'production.patch'):
        add(D / 'history/pre-capture-padding' / name, 'validation/author/history/pre-capture-padding/' + name)
    for name in ('review.txt', 'validation.json'):
        add(I / name, 'validation/independent/' + name)
    for folder in ('host', 'fallback', 'mutations'):
        small_tree(I / folder, 'validation/independent/' + folder, {'.json', '.log'})
    omitted = {}
    for label, root in (('author', D), ('independent', I)):
        for source in sorted((root / 'codegen').rglob('*')):
            if not source.is_file():
                continue
            relative = str(source.relative_to(root))
            if label == 'author' and relative.startswith('codegen/candidate/'):
                continue  # Earlier capture draft is represented by its patch.
            if source.suffix in {'.json', '.log', '.su'} or source.name in ('htpPredictExperts.asm', 'initCpuPredictor.asm', 'layouts.txt'):
                add(source, 'validation/' + label + '/' + relative)
            elif source.suffix in {'.o', '.bin', '.asm'}:
                omitted[label + '/' + relative] = {'sha256': sha(source), 'bytes': source.stat().st_size}
    generated('validation/omitted-codegen-artifacts.json', omitted)
    for name in ('root-applied.patch', 'root-core-build.log', 'root-normal-build.log', 'root-normal-rebuild.log',
                 'root-restored-build.log', 'root-normal-jobs.json', 'root-normal-summary.json',
                 'root-native-results.json', 'root-native-check.log', 'root-native-micro.log', 'root-frozen.json',
                 'root-diagnostic.patch', 'root-diagnostic-source.json', 'root-diagnostic-build.log',
                 'root-diagnostic-removed.json', 'root-diagnostic-jobs.json', 'root-diagnostic-0.json',
                 'root-diagnostic-1.json'):
        add(D / name, 'validation/root/' + name)
    if (D / 'root-review.txt').exists():
        add(D / 'root-review.txt', 'validation/root/root-review.txt')
    for name in ('README.txt', 'diagnostic.patch', 'parse.py', 'parser_check.py'):
        add(D / 'diagnostic' / name, 'diagnostic/' + name)
    for rel in RELS:
        add(D / 'diagnostic/candidate' / rel, 'diagnostic/candidate/' + rel)
    for source in sorted((D / 'diagnostic/codegen').rglob('*')):
        if source.is_file() and (source.suffix in {'.json', '.log', '.su'} or
                                 source.name in ('htpPredictExperts.asm', 'initCpuPredictor.asm')):
            add(source, 'diagnostic/codegen/' + str(source.relative_to(D / 'diagnostic/codegen')))
    small_tree(D / 'diagnostic/checks', 'diagnostic/checks', {'.json', '.log'})
    for n in (1024, 2048):
        small_tree(E / 'repro/models' / ('lfm2-' + str(n)), 'configuration/lfm2-' + str(n), {'.json'})
    for name in ('bench.py', 'device.json'):
        add(E / name, 'reproduction/' + name)
    add(E / 'validation/memory_monitor.cpp', 'reproduction/memory_monitor.cpp', True)
    for name in ('run_matrix.py', 'summarize_pair.py'):
        add(T / name, 'reproduction/original-' + name)
    for name in ('curate_predict_register.py', 'verify_archive.py', 'reproduce_host.py'):
        add(A / name, 'reproduction/' + name)
    add(A / 'README.txt', 'README.txt', False)
    generated('summary.json', {
        'decision': 'not_adopted',
        'reason': 'No consistent end-to-end decode TPS improvement. Prefill variation is observed but not causally attributed to the decode-only hint kernel.',
        'normal': computed, 'normal_run_count': 12, 'all_65_ids_exact': True,
        'all_seven_binaries_identical_off_on': True,
        'runtime_binaries': {k: v['binaries'] for k, v in variants.items()},
        'diagnostic_actual_dispatch': diagnostic_checks, 'diagnostic_tps_excluded': True,
        'isolated_cpu_micro': micro, 'cpu_micro_does_not_predict_model_gain': True,
        'two_sources_and_four_core_binaries_restored': restoration is not None,
        'original_seven_binary_phone_runtime_verified': restoration is not None,
        'live_root_app_restored_claim': False,
        'independent_agent_review_performed': True,
        'new_tests_or_phone_runs_during_curation': False,
    })
    generated('validation/archive-checks.json', {
        'normal_runs_validated': 12, 'normal_summary_recomputed': True,
        'all_raw_memory_hashes_exact': True, '65_ids_each': True,
        'fixed_c16_page0_preload128': True, 'all_seven_binaries_identical_off_on': True,
        'author_and_integrated_patches_replayed': True, 'tests_patch_replayed': True,
        'live_root_state_not_used': True, 'no_device_actions': True,
        'restoration_verified': restoration is not None,
        'source_sha256': SOURCE_SHA, 'production_patch_sha256': PRODUCTION_SHA,
        'tests_patch_sha256': TESTS_SHA, 'memory_sampling': memory_rows,
        'diagnostics_parsed_from_raw_app_logs': True,
        'native_log_hashes_verified': True, 'cpu_micro_medians_recomputed': True,
        'independent_review_hash_verified': True,
        'validation_scope': 'Read-only evidence validation; existing author/reviewer/root checks were not rerun.',
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
                'author_source_base_commit': baseline['head'],
                'reproduction': {'simde_version': '0.8.2', 'dependency_sha256': {},
                                 'scope': 'Predictor fixture consumes only archived baseline/candidate source and external SIMDe/NDK.'},
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
