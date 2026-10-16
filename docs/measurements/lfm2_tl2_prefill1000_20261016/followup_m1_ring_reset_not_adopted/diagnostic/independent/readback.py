#!/usr/bin/env python3
"""Read-only audit of ring diagnostic records and frozen runtime/source evidence.
No compiler, host fixture, phone, or root-source writes are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from fractions import Fraction


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('repo', 'author', 'fixed', 'scratch', 'output'):
        ap.add_argument('--' + arg, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ev = a.repo / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    inp = {}

    def read(p):
        inp[str(p)] = sha(p)
        return p.read_text()

    def jread(p):
        return json.loads(read(p))

    models = []
    ids_ref = {1024: 'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
               2048: '2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
    fixed_env = {'NNTR_NUM_THREADS': '8', 'NNTR_HTP_PAGE_CACHE': '0', 'NNTR_LOAD_PREFETCH': '0',
                 'NNTR_MOE_CACHE_GROW': '0', 'NNTR_MOE_PRELOAD_EXPERTS': '128', 'NNTR_HTP_PREDICT': '2',
                 'NNTR_MOE_PREFETCH_RESIDUAL': '0.5', 'NNTR_HTP_TL2_STABLE_SLOTS': '1',
                 'NNTR_LFM2_PREFILL_EARLY': '1', 'NNTR_LFM2_DECODE_PAIR_READ': '1',
                 'NNTR_LFM2_DECODE_DEMAND_HANDOFF': '1', 'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1',
                 'NNTR_LFM2_ROUTER_ARM': '1'}
    # This deliberately does not import the author's parser or its count helper.
    record = re.compile(r'^(\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\s+(\d+)\s+(\d+)\s+[A-Z]\s+[^:]+:.*nntr-ring-reset q=(\d+) used=([01]) !$')
    for rows, model in ((1024, 'model'), (2048, 'model2048')):
        tag = f'continue-m1-ring-reset-diagnostic{rows}'
        meta = jread(ev / (tag + '.json'))
        mem = jread(ev / (tag + '.memory.json'))
        raw = read(a.author / (tag + '.live.logcat.txt'))
        log = read(ev / (tag + '.log'))
        existing = jread(a.author / (tag + '.parsed.json'))
        ids = read(ev / (tag + '.ids'))
        assert meta['valid'] and meta['ids_match_reference'] and meta['returncode'] == 0
        assert meta['model'] == model and meta['moe_cache_experts'] == 16
        assert meta['decode_tokens'] == 64 and meta['prefill_tokens'] == rows
        assert meta['instrumented'] and not meta['performance_comparison_eligible']
        assert len(ids.split()) == 65 and sha(ev / (tag + '.ids')) == ids_ref[rows] == meta['ids_sha256']
        assert mem['child_waited'] and mem['child_exit_code'] == 0 and mem['child_signal'] is None
        assert sha(ev / (tag + '.memory.json')) == meta['full_run_memory_report_sha256']
        assert mem['command'][0].endswith('/lfm-m1-ring-reset-diagnostic/nntr_causallm')
        assert mem['command'][1].endswith('/' + model)
        assert all(meta['environment'].get(k) == v for k, v in fixed_env.items())
        assert meta['profile'] and meta['environment']['NNTR_HTP_PROFILE'] == '1'
        assert not meta['concurrent_processes']
        assert log.count('[HTP] graph: forward calls=64 tokens=64 calls/token=1.00') == 1
        assert len(re.findall(r'\[HTP\] token driver: close tokens=64 hops/token=\S+ served=64 timeouts=0 stale=0 id_checked=0 id_mismatch=0 stop_err=0x0', log)) == 1
        assert len(re.findall(r'\[HTP\] dspq: close calls=64 served=64 bad=0 .* stop_err=0x0', log)) == 1
        assert log.count('[HTP] token driver: on ') == 1
        assert log.count('[HTP] token driver: first token ') == 1
        assert '[MoE] preload_experts=128 cache_slots=608' in log
        assert 'direct slots=32 bytes=64618496 readers=2 pair_read=1' in log
        assert '[HTP] demand read pool: readers=4' in log
        rates = re.findall(r'\[HTP\] token driver: moe calls/token=(\d+\.\d{2}) ', log)
        assert len(rates) == 1
        # Integer enumeration over the exact half-cent rounding interval. Neither
        # endpoint is an integer here, so printf's tie policy is immaterial.
        rate = Fraction(rates[0])
        lo, hi = (rate - Fraction(1, 200)) * 64, (rate + Fraction(1, 200)) * 64
        assert lo.denominator != 1 and hi.denominator != 1
        possible = [n for n in range(lo.numerator // lo.denominator, hi.numerator // hi.denominator + 2) if lo < n < hi]
        assert len(possible) == 1
        expected = possible[0]
        records = []
        for line in raw.splitlines():
            m = record.fullmatch(line)
            assert m, line
            timestamp, pid, tid, seq, used = m.groups()
            assert int(pid) == mem['pid']
            records.append((timestamp, int(tid), int(seq), int(used)))
        assert len(records) == expected
        assert all(seq == i + 1 for i, (_, _, seq, _) in enumerate(records))
        assert all(records[i][0] <= records[i + 1][0] for i in range(len(records) - 1))
        hits = sum(x[3] for x in records)
        assert hits == expected
        assert existing['observed_unique_attempts'] == expected == existing['independently_derived_calls']
        assert existing['hits'] == hits and existing['fallbacks'] == 0
        assert existing['raw_records'] == len(records) and existing['exact_transport_duplicates'] == 0
        assert existing['device_pid'] == mem['pid']
        assert existing['live_log_sha256'] == sha(a.author / (tag + '.live.logcat.txt'))
        models.append({'rows': rows, 'pid': mem['pid'], 'tid_set': sorted({x[1] for x in records}),
                       'same_run_rate': rates[0], 'integer_rounding_interval': [str(lo), str(hi)],
                       'independent_total': expected, 'raw_records': len(records),
                       'sequence': [records[0][2], records[-1][2]], 'hits': hits, 'fallbacks': 0,
                       'duplicates': 0, 'missing_or_reordered': 0,
                       'first_timestamp': records[0][0], 'last_timestamp': records[-1][0],
                       'logical_clear_bytes_avoided': 32768 * hits,
                       'ids_count': 65, 'ids_sha256': ids_ref[rows],
                       'complete_exit': True, 'performance_eligible': False,
                       'live_log_sha256': sha(a.author / (tag + '.live.logcat.txt'))})

    build = jread(a.fixed / 'build/validation.json')
    trace = jread(a.fixed / 'diagnostic/build/validation.json')
    expected_dsp = {'control': build['control']['sha256'], 'candidate': build['candidate']['sha256'],
                    'diagnostic': trace['diagnostic_so_sha256']}
    variants = {'control': 'lfm-m1-ring-reset-control', 'candidate': 'lfm-m1-ring-reset',
                'diagnostic': 'lfm-m1-ring-reset-diagnostic'}
    dsp_so = {'control': a.fixed / 'build/control/libnntr_hvx_skel.so',
              'candidate': a.fixed / 'build/candidate/libnntr_hvx_skel.so',
              'diagnostic': a.fixed / 'diagnostic/build/libnntr_hvx_skel.so'}
    sources = [Path('nntrainer/tensor/htp_backend/hmx') / name for name in ('hexkl_dma_ring.c', 'hexkl_dma_ring.h', 'hexkl_mm_u8i4_moe.c')]
    source_results, common, app = {}, None, None
    for arm, variant in variants.items():
        manifest = jread(ev / (variant + '-artifacts.json'))
        assert manifest['binaries']['libnntr_hvx_skel.so'] == expected_dsp[arm] == sha(dsp_so[arm])
        inp[str(dsp_so[arm])] = sha(dsp_so[arm])
        host = {k: v for k, v in manifest['binaries'].items() if k != 'libnntr_hvx_skel.so'}
        if common is None:
            common = host
        assert len(host) == 6 and host == common
        spec = manifest['dsp_build']
        assert spec['kind'] == 'frozen_object_pair' and spec['arm'] == arm
        assert spec['baseline_revision'] == build['source_revision']
        assert spec['frozen_objects'] == 63 and spec['common_objects'] == 61 and spec['common_link_flag'] == '-no-threads'
        assert spec['production_patch_sha256'] == build['source_patch_sha256']
        if arm == 'diagnostic':
            assert spec['diagnostic_patch_sha256'] == trace['diagnostic_patch_sha256']
            assert spec['compiler_define'] == 'NNTR_LFM2_M1_RING_RESET_TRACE=1'
        patch = a.scratch / (variant + '.source.patch')
        text = read(patch)
        assert sha(patch) == manifest['source_patch_sha256']
        files = re.findall(r'^diff --git a/(\S+) b/\1$', text, re.M)
        assert files and len(set(files)) == len(files)
        tree = a.output / ('source-replay-' + arm)
        assert not tree.exists()
        tree.mkdir()
        for relative in sorted(set(files) | {str(x) for x in sources}):
            out = tree / relative
            data = subprocess.run(['git', '-C', str(a.repo), 'show', manifest['base_commit'] + ':' + relative], capture_output=True)
            if data.returncode == 0:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data.stdout)
            else:
                assert ('new file mode ' in text) and relative not in [str(x) for x in sources]
        subprocess.run(['git', 'apply', str(patch)], cwd=tree, check=True, capture_output=True)
        app_hashes = {f: sha(tree / f) for f in files if f.startswith('Applications/')}
        if app is None:
            app = app_hashes
        assert app_hashes == app
        source_results[arm] = {}
        for relative in sources:
            actual = (tree / relative).read_bytes()
            if arm == 'control':
                want = subprocess.check_output(['git', '-C', str(a.repo), 'show', build['source_revision'] + ':' + str(relative)])
            elif arm == 'diagnostic' and relative.name == 'hexkl_dma_ring.c':
                want = (a.fixed / 'diagnostic/build/ring-diagnostic.c').read_bytes()
            else:
                want = (a.fixed / 'build/source' / relative).read_bytes()
            assert actual == want, (arm, relative)
            source_results[arm][str(relative)] = hashlib.sha256(actual).hexdigest()
    accepted = jread(a.author / 'root-accepted-control.json')
    for name, value in common.items():
        assert accepted[name] == value == sha(a.author / 'root-accepted-control' / name)
        inp[str(a.author / 'root-accepted-control' / name)] = value
    assert common['libnntrainer.so'].startswith('072b5ca2')
    assert common['libcausallm.so'].startswith('c34aa784')
    assert accepted['libnntr_hvx_skel.so'].startswith('3baddb92')
    assert all(s != accepted['libnntr_hvx_skel.so'] for s in expected_dsp.values())
    identity = jread(a.author / 'root-runtime-identity.json')
    assert identity['host_binaries'] == common and identity['ARM_router_option_for_all_runs'] == 1
    assert identity['normal_source_restored_exact_after_trace_overlay']
    jobs = jread(a.author / 'root-normal-jobs.json')
    assert len(jobs) == 12 and all(j['env']['NNTR_LFM2_ROUTER_ARM'] == 1 for j in jobs)
    assert all(j['variant'] in ('lfm-m1-ring-reset', 'lfm-m1-ring-reset-control') for j in jobs)
    assert all(j['model'] in ('model', 'model2048') for j in jobs)
    job_info = {'jobs': 12, 'ARM_router_option_all': 1,
                'same_environments': len({json.dumps(j['env'], sort_keys=True) for j in jobs}) == 1}
    assert job_info['same_environments']
    out = {'status': 'PASS', 'independent_parser': True, 'new_tests_or_device_runs': False,
           'diagnostic_records': models, 'common_six_host_binaries': common,
           'controlled_dsp': expected_dsp, 'source_patch_replay': source_results,
           'common_application_source_sha256': app,
           'not_original_3b_binary_comparison': True,
           'no_timing_claim': True, 'input_sha256': inp, 'normal_job_metadata': job_info}
    (a.output / 'validation.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('input_sha256','source_patch_replay','common_application_source_sha256')},indent=2))


if __name__ == '__main__':
    main()
