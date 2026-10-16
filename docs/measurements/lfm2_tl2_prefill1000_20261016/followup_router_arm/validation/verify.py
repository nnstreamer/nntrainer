#!/usr/bin/env python3
"""Verify this frozen router archive without the original workspace or a device."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True

IDS = {
    1024: 'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
    2048: '2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33',
}
FIXED = {
    'NNTR_NUM_THREADS': '8', 'NNTR_HTP_PAGE_CACHE': '0',
    'NNTR_LOAD_PREFETCH': '0', 'NNTR_MOE_CACHE_GROW': '0',
    'NNTR_MOE_PRELOAD_EXPERTS': '128', 'NNTR_HTP_PREDICT': '2',
    'NNTR_MOE_PREFETCH_RESIDUAL': '0.5', 'NNTR_HTP_TL2_STABLE_SLOTS': '1',
    'NNTR_LFM2_PREFILL_EARLY': '1', 'NNTR_LFM2_DECODE_PAIR_READ': '1',
    'NNTR_LFM2_DECODE_DEMAND_HANDOFF': '1',
    'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1',
}
PRODUCTION = 'd5604ee3969f985032b607967870f3e8983a4b534572a0e2910a1746839c4796'
TESTS = '403448179df6daf166ef4c3df8ecc7fef0489d530bf253b1f257437fba520d76'
PARSER = '7f949e97b2cfd1fb51df865c6cbc58e8de451810f264bfd9f19d140223344e43'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = set(manifest['files']) | {'manifest.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual == expected, ('Unlisted/missing files', actual ^ expected)
    for name, entry in manifest['files'].items():
        path = root / name
        assert path.resolve().is_relative_to(root) and not path.is_symlink(), name
        data = path.read_bytes()
        assert digest(data) == entry['sha256'], name
        data = gzip.decompress(data) if name.endswith('.gz') else data
        assert digest(data) == entry['uncompressed_sha256'], name
        assert not data.startswith((b'\x7fELF', b'!<arch>\n')), name
        data.decode('utf-8')

    def blob(name):
        path = root / name
        if path.is_file():
            return path.read_bytes()
        return gzip.decompress((root / (name + '.gz')).read_bytes())

    def load(name):
        return json.loads(blob(name))

    assert digest(blob('artifacts/production/production.patch')) == PRODUCTION
    assert digest(blob('artifacts/tests/tests.patch')) == TESTS
    hashes = load('artifacts/production/source-hashes.json')['files']
    frozen = load('artifacts/root/root-frozen.json')
    for name, value in hashes.items():
        assert digest(blob('artifacts/production/source/' + name)) == value, name
        assert frozen['hashes'][name] == value, name
    applied = load('artifacts/root/root-applied.json')
    assert applied['patch_sha256'] == PRODUCTION
    assert applied['clang_format_14_byte_identical']
    assert frozen['zero_fmla_fmadd']
    command = frozen['kernel_compile_command']['command']
    assert command.rindex('-fno-fast-math') > command.index('-ffast-math')
    assert '-ffp-contract=off' in command
    runtime = load('artifacts/root/root-normal-runtime-identity.json')
    assert runtime['all_seven_same'] and len(runtime['binaries']) == 7
    assert runtime['binaries']['libcausallm.so'] == frozen['hashes']['builddir_app/htp/libcausallm.so']
    assert runtime['binaries']['nntr_causallm'] == frozen['hashes']['builddir_app/htp/nntr_causallm']
    restored = load('artifacts/root/root-diagnostic-removed.json')
    assert restored['normal_app_and_executable_byte_identical'] and restored['normal_source_exact']
    assert restored['libcausallm_sha256'] == runtime['binaries']['libcausallm.so']
    assert restored['decision'] == 'retain_opt_in_prefill_router'
    author = load('artifacts/author/frozen.json')
    assert author['production_patch_sha256'] == PRODUCTION
    assert author['tests_patch_sha256'] == TESTS
    for name, value in author['test_files'].items():
        assert digest(blob('artifacts/tests/source/' + name)) == value, name
    for name, field in (('README.txt', 'README_sha256'), ('validation.json', 'validation_sha256')):
        assert digest(blob('artifacts/author/' + name)) == author[field]
    independent = load('artifacts/independent/manifest.json')
    for name, value in independent.items():
        assert digest(blob('artifacts/independent/' + name)) == value, name
    for variant in ('lfm-router-arm-0', 'lfm-router-arm-1', 'lfm-router-arm-diagnostic'):
        prefix = 'artifacts/root/deployment/' + variant
        deployed = load(prefix + '-artifacts.json')
        assert digest(blob(prefix + '.source.patch')) == deployed['source_patch_sha256']
        expected_runtime = dict(runtime['binaries'])
        if variant.endswith('diagnostic'):
            built = load('artifacts/root/root-diagnostic-build.json')
            expected_runtime['libcausallm.so'] = built['diagnostic_app_sha256']
            assert digest(blob('artifacts/root/root-diagnostic-source.cpp')) == built['source_sha256']
        assert deployed['binaries'] == expected_runtime

    jobs = load('artifacts/root/root-normal-jobs.json')
    summary = load('artifacts/root/root-normal-summary.json')
    assert len(jobs) == len(summary['runs']) == 12 and summary['all_checks_pass']
    assert len({j['tag'] for j in jobs}) == 12
    configurations, normal = {}, []

    def run(tag, profile):
        result = load('raw/' + tag + '.json')
        assert result['tag'] == tag and result['valid'] and result['returncode'] == 0
        assert not result['concurrent_processes']
        assert result['profile'] == result['instrumented'] == profile
        assert result['performance_comparison_eligible'] == (not profile)
        assert result['moe_cache_experts'] == 16 and result['decode_tokens'] == 64
        rows = result['prefill_tokens']
        assert rows in IDS
        for key, value in FIXED.items():
            assert result['environment'][key] == value, (tag, key)
        assert result['ids_match_reference']
        ids = blob('raw/' + tag + '.ids')
        assert len(ids.splitlines()) == 65
        assert digest(ids) == result['ids_sha256'] == IDS[rows]
        configurations.setdefault(rows, set()).add(result['model_config_sha256'])
        memory_data = blob('raw/' + tag + '.memory.json')
        memory = json.loads(memory_data)
        assert digest(memory_data) == result['full_run_memory_report_sha256']
        assert memory['child_waited'] and memory['child_exit_code'] == 0
        assert memory['requested_period_ms'] == 20
        peak = memory['peak_dedup_sample']
        assert peak['dedup_complete_for_readable_size_fds'] and peak['composite_available']
        for key in ('fdinfo_read_errors', 'unknown_identity_fds', 'inode_size_conflicts',
                    'invalid_fields', 'counter_overflow'):
            assert not peak[key], (tag, key)
        assert result['full_run_sampled_peak_kib'] * 1024 == memory['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes']
        return result

    for job, recorded in zip(jobs, summary['runs']):
        result = run(job['tag'], False)
        assert result['variant'] == job['variant']
        assert result['environment']['NNTR_LFM2_ROUTER_ARM'] == str(job['env']['NNTR_LFM2_ROUTER_ARM'])
        for key, value in recorded.items():
            if key != 'repetition':
                assert result[key] == value, (job['tag'], key)
        normal.append(result)
    recomputed = []
    for expected_summary in summary['summary']:
        rows = expected_summary['tokens']
        item = {'tokens': rows}
        group = [[r for r in normal if r['prefill_tokens'] == rows and
                  r['environment']['NNTR_LFM2_ROUTER_ARM'] == str(flag)] for flag in (0, 1)]
        assert all(len(g) == 3 for g in group)
        for metric in ('prefill_tps', 'decode_tps'):
            values = [[r[metric] for r in g] for g in group]
            medians = [statistics.median(v) for v in values]
            computed = {'control': values[0], 'candidate': values[1],
                        'median_control': medians[0], 'median_candidate': medians[1],
                        'median_pct': (medians[1] / medians[0] - 1) * 100,
                        'pairs_faster': sum(b > a for a, b in zip(*values)),
                        'paired_deltas_pct': [(b / a - 1) * 100 for a, b in zip(*values)]}
            assert computed == expected_summary[metric]
            item[metric] = computed
        for index, prefix in enumerate(('control', 'candidate')):
            key = prefix + '_max_peak_kib'
            item[key] = max(r['full_run_sampled_peak_kib'] for r in group[index])
            assert item[key] == expected_summary[key]
        recomputed.append(item)

    parser_path = root / 'artifacts/diagnostic/parse.py'
    assert digest(parser_path.read_bytes()) == PARSER
    spec = importlib.util.spec_from_file_location('frozen_router_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    diagnostics = []
    for rows, wait_ms, register_ms in ((1024, 35.5, 2.6), (2048, 0.1, 3.0)):
        tag = f'continue-router-arm-diagnostic{rows}'
        result = run(tag, True)
        assert result['environment']['NNTR_LFM2_ROUTER_ARM'] == '1'
        assert result['environment']['NNTR_M0_PROFILE'] == '1'
        log = blob('raw/' + tag + '.log').decode()
        parsed = parser.parse(log, rows, 1, 38)
        assert parsed == load(f'artifacts/root/root-diagnostic-{rows}.json')
        assert parsed['tasks'] == parsed['completed'] == rows // 4 * 38
        assert parsed['rejected'] == parsed['fallback'] == 0
        waits = re.findall(r'expert prefetch: (\d+) experts read under the layer calls, exposed wait ([\d.]+) ms .*?register rpc ([\d.]+) ms', log)
        assert len(waits) == 1
        experts, wait, register = waits[0]
        assert (float(wait), float(register)) == (wait_ms, register_ms)
        diagnostics.append({'rows': rows, 'fast': 38, 'tasks': parsed['tasks'],
                            'completed': parsed['completed'], 'rejected': 0,
                            'experts_read': int(experts), 'exposed_wait_ms': float(wait),
                            'register_rpc_ms': float(register), 'performance_comparison_eligible': False})
    assert all(len(values) == 1 for values in configurations.values())

    for rows in IDS:
        prefix = f'artifacts/repeat/continue-router-arm-repeat{rows}'
        repeated = load(prefix + '.json')
        assert repeated['input_ids_equal'] and repeated['output_ids_equal']
        assert repeated['ids_match_normal_run'] and not repeated['skip_prefill']
        assert repeated['ids_sha256'] == IDS[rows]
        assert repeated['configured_decode_tokens'] == 64 and repeated['expected_dumped_ids'] == 65
        assert len(repeated['runs']) == 2
        for key, value in FIXED.items():
            assert repeated['environment'][key] == value
        assert repeated['environment']['NNTR_LFM2_ROUTER_ARM'] == '1'
        for number, recorded in enumerate(repeated['runs'], 1):
            assert recorded['run'] == number and recorded['input_tokens'] == rows
            assert recorded['decode_tokens'] == 64 and recorded['dumped_generated_ids'] == 65
            ids = blob(prefix + f'-run{number}.ids')
            assert len(ids.splitlines()) == 65 and digest(ids) == IDS[rows]

    native = load('artifacts/native/root-native-v3-results.json')
    assert native['binary_sha256'] == 'f956408ffd7f054be2507cedbef1cad72ef12f8f4ac457d4f7fd8975b46aee75'
    assert {(r['flag'], r['threads']) for r in native['runs']} == {(f, t) for f in (0, 1) for t in (1, 8)}
    for result in native['runs']:
        data = blob('artifacts/native/' + result['log'])
        assert result['returncode'] == 0 and digest(data) == result['log_sha256']
        rpc = 138 if result['flag'] else 188
        marker = f"ROUTER_ARM_NATIVE 94 cases threads={result['threads']} env={result['flag']} RPC={rpc} PASS"
        assert marker in data.decode()
    for label, relative, marker in (
        ('v1', 'artifacts/rejected/v1/root-numeric-native.log', '14 cases 6 failures'),
        ('v2', 'artifacts/rejected/v2/root-numeric-native-v2.log', '46 cases 8 failures')):
        assert marker in blob(relative).decode(), label
    for prefix in ('artifacts/author/checks/mutations-v3-final', 'artifacts/independent/mutations'):
        mutations = load(prefix + '/results.json')
        assert len(mutations) == 10 and all(r['pass'] and r['returncode'] != 0 for r in mutations)
        for result in mutations:
            assert result['expected_assertion'] in blob(prefix + '/' + result['name'] + '/run.log').decode()

    return {'pass': True, 'normal_runs': 12, 'diagnostic_runs': 2,
            'same_handle_prompts': {'1024': 2, '2048': 2}, 'native_cases': 376,
            'runtime_binaries_same_off_on': 7, 'normal_summary': recomputed,
            'diagnostics': diagnostics, 'model_config_sha256': {
                str(k): next(iter(v)) for k, v in configurations.items()},
            'diagnostic_input_contract': 'One child process stderr; parser does not authenticate PID',
            'decision': 'retain_opt_in_prefill_router', 'default_enabled': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    result = verify(a.archive)
    print(json.dumps(result, indent=2) if a.json else
          'PASS: manifest, 12 normal runs, 2 diagnostics, 4 same-handle prompts, 376 native cases, 7 runtime hashes')
