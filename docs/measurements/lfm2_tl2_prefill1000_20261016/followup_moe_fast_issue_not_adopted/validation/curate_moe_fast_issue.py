#!/usr/bin/env python3
"""Archive the unadopted MoE leaf experiment; validate before writing anything."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo', type=Path, default=Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000'))
    cli.add_argument('--author', type=Path, default=Path('/tmp/nntr-lfm2-moe-fast-issue-20261016'))
    cli.add_argument('--independent', type=Path, default=Path('/tmp/nntr-lfm2-moe-fast-issue-review-20261016'))
    cli.add_argument('--scratch', type=Path, default=Path(__file__).resolve().parent)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--check-only', action='store_true')
    args = cli.parse_args()
    M, I, T = args.author, args.independent, args.scratch
    E = args.repo / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
    D = M / 'diagnostic'
    out = args.output or E / 'followup_moe_fast_issue_not_adopted'
    copies = {}

    def add(src, relative, compress=False):
        require(src.is_file(), 'Missing input: ' + str(src))
        relative += '.gz' if compress else ''
        require(relative not in copies, 'Duplicate archive path: ' + relative)
        copies[relative] = (src, compress)

    def tree(src, relative, suffixes):
        require(src.is_dir(), 'Missing input directory: ' + str(src))
        for path in sorted(src.rglob('*')):
            if path.is_file() and path.suffix in suffixes:
                add(path, relative + '/' + str(path.relative_to(src)),
                    path.suffix in {'.log', '.s', '.su', '.c', '.h', '.patch'})

    final_parser_sha = '464ba98bb96d725c0d829f01f13c6c12d9d147ea25390ae71ed29c6bcfc971fa'
    require(sha(D / 'parse.py') == final_parser_sha, 'Unexpected final parser')
    for manifest_dir, mapping in [(M, read(M / 'manifest.json')['files']),
                                  (M, read(M / 'source-hashes.json')),
                                  (D, read(D / 'manifest.json')['files'])]:
        for rel, expected in mapping.items():
            require(sha(manifest_dir / rel) == expected, 'Frozen input changed: ' + rel)
    independent = read(I / 'diagnostic/validation.json')
    require(independent['independent_final_host']['pass'] and independent['independent_final_parser']['pass'], 'Independent final validation incomplete')
    for rel, expected in independent['final_parser_manifest'].items():
        require(sha(D / rel) == expected, 'Independent parser snapshot differs: ' + rel)
    spec = importlib.util.spec_from_file_location('moe_issue_trace', D / 'parse.py')
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)

    control, candidate, diagnostic = 'tl2-gu-direct-epilogue', 'lfm-moe-fast-issue', 'lfm-moe-fast-issue-trace'
    variants = {}
    for name in (control, candidate, diagnostic):
        artifact = read(E / (name + '-artifacts.json'))
        require(sha(T / (name + '.source.patch')) == artifact['source_patch_sha256'], 'Deployment patch mismatch: ' + name)
        variants[name] = artifact
        add(E / (name + '-artifacts.json'), 'artifacts/' + name + '.json')
        add(T / (name + '.source.patch'), 'artifacts/' + name + '.source.patch', True)
    require(variants[candidate]['parent'] == control and variants[diagnostic]['parent'] == candidate, 'Variant parent mismatch')
    for name in (candidate, diagnostic):
        require({k: v for k, v in variants[name]['binaries'].items() if k != 'libnntr_hvx_skel.so'} ==
                {k: v for k, v in variants[control]['binaries'].items() if k != 'libnntr_hvx_skel.so'}, 'ARM runtime changed')
    frozen = read(M / 'root-frozen/source.json')
    require(frozen['normal_dsp_sha256'] == variants[candidate]['binaries']['libnntr_hvx_skel.so'], 'Normal DSP mismatch')
    require(frozen['source_sha256'] == read(M / 'manifest.json')['candidate_c_sha256'], 'Integrated source mismatch')
    restored = read(M / 'root-normal-restored.json')
    require(restored['tracked_clean'] and restored['normal_dsp_sha256'] == variants[control]['binaries']['libnntr_hvx_skel.so'] and
            restored['normal_moe_source_sha256'] == read(M / 'manifest.json')['baseline_c_sha256'], 'Normal restore incomplete')

    env = {'NNTR_NUM_THREADS': '8', 'NNTR_HTP_PAGE_CACHE': '0', 'NNTR_LOAD_PREFETCH': '0',
           'NNTR_MOE_CACHE_GROW': '0', 'NNTR_MOE_PRELOAD_EXPERTS': '128', 'NNTR_HTP_PREDICT': '2',
           'NNTR_MOE_PREFETCH_RESIDUAL': '0.5', 'NNTR_HTP_TL2_STABLE_SLOTS': '1',
           'NNTR_LFM2_PREFILL_EARLY': '1', 'NNTR_LFM2_DECODE_PAIR_READ': '1',
           'NNTR_LFM2_DECODE_DEMAND_HANDOFF': '1', 'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1'}
    ids_sha = {1024: 'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
               2048: '2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
    jobs = read(M / 'root-normal-jobs.json')
    require(len(jobs) == 12 and len({j['tag'] for j in jobs}) == 12, 'Expected 12 unique normal jobs')
    order = [control, candidate, candidate, control, control, candidate]
    require([j['variant'] for j in jobs] == order * 2, 'Normal ABBAAB order changed')
    results, traces = {}, {}
    for job in jobs + [dict(tag='continue-moe-issue-trace' + str(n), variant=diagnostic) for n in (1024, 2048)]:
        tag = job['tag']
        result, memory = read(E / (tag + '.json')), read(E / (tag + '.memory.json'))
        normal = job['variant'] != diagnostic
        n = result['prefill_tokens']
        require(n in ids_sha and result['tag'] == tag and result['variant'] == job['variant'], tag + ': wrong run')
        require(result['valid'] and result['returncode'] == 0 and not result['concurrent_processes'], tag + ': failed/concurrent run')
        require(result['ids_match_reference'] and result['ids_sha256'] == sha(E / (tag + '.ids')) == ids_sha[n], tag + ': IDs differ')
        require(len((E / (tag + '.ids')).read_text().splitlines()) == 65 and result['decode_tokens'] == 64, tag + ': incomplete IDs')
        require(result['moe_cache_experts'] == 16 and all(result['environment'].get(k) == v for k, v in env.items()), tag + ': fixed settings changed')
        require('NNTR_INIT_SEQ_LEN' not in result['environment'], tag + ': unexpected warmup override')
        require(result['performance_comparison_eligible'] == normal and result['instrumented'] != normal and result['profile'] != normal, tag + ': timing cohort mismatch')
        config_name = 'nntr_config.json' if n == 1024 else 'nntr_config_2048.json'
        require(result['model_config_sha256'] == sha(D / 'config-source' / config_name) and
                read(D / 'config-source' / config_name)['init_seq_len'] == n, tag + ': warmup config mismatch')
        peak = memory['peak_dedup_sample']
        require(memory['child_waited'] and memory['child_exit_code'] == 0 and memory['requested_period_ms'] == 20, tag + ': incomplete monitor')
        require(peak['dedup_complete_for_readable_size_fds'] and peak['composite_available'] and
                not any(peak[k] for k in ('fdinfo_read_errors', 'unknown_identity_fds', 'inode_size_conflicts', 'invalid_fields', 'counter_overflow')), tag + ': peak accounting incomplete')
        require(result['full_run_memory_report_sha256'] == sha(E / (tag + '.memory.json')) and
                result['full_run_sampled_peak_kib'] * 1024 == memory['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'], tag + ': memory hash/value differs')
        for ext in ('json', 'ids', 'log', 'memory.json'):
            add(E / (tag + '.' + ext), 'raw/' + tag + '.' + ext, ext in ('log', 'memory.json'))
        if normal:
            results[tag] = result
        else:
            for ext in ('logcat.txt', 'issue.logcat.txt'):
                add(E / (tag + '.' + ext), 'raw/' + tag + '.' + ext, True)
            counts = json.loads(json.dumps(parser.parse((E / (tag + '.issue.logcat.txt')).read_text().splitlines(), n)))
            require(counts == read(M / ('root-trace-summary' + str(n) + '.json')) == read(D / ('device' + str(n) + '.json')), tag + ': strict reparse differs')
            require(counts['android_pids'] == [memory['pid']], tag + ': DSP log/child PID mismatch')
            require(restored['trace_capture_validation'][str(n)]['raw_sha256'] == sha(E / (tag + '.issue.logcat.txt')), tag + ': restored capture hash mismatch')
            traces[str(n)] = counts

    computed = []
    for n in (1024, 2048):
        group = {v: sorted((r for r in results.values() if r['prefill_tokens'] == n and r['variant'] == v), key=lambda r: r['tag']) for v in (control, candidate)}
        require(all(len(group[v]) == 3 for v in group), 'Unbalanced normal group')
        row = {'tokens': n}
        for metric in ('prefill_tps', 'decode_tps'):
            a, b = [[r[metric] for r in group[v]] for v in (control, candidate)]
            row[metric] = {'control': a, 'candidate': b, 'median_control': statistics.median(a),
                           'median_candidate': statistics.median(b),
                           'median_pct': (statistics.median(b) / statistics.median(a) - 1) * 100,
                           'pairs_faster': sum(y > x for x, y in zip(a, b)),
                           'paired_deltas_pct': [(y / x - 1) * 100 for x, y in zip(a, b)]}
        for label, v in [('control', control), ('candidate', candidate)]:
            row[label + '_max_peak_kib'] = max(r['full_run_sampled_peak_kib'] for r in group[v])
        computed.append(row)
    require(computed == read(M / 'root-normal-summary.json')['summary'], 'Normal summary mismatch')

    text_types = {'.txt', '.json', '.log', '.s', '.su', '.c', '.h', '.py', '.patch', '.diff'}
    log_types = {'.json', '.log'}
    for name in ['review.txt', 'validation.json', 'source-hashes.json', 'manifest.json', 'production.patch',
                 'baseline.s', 'candidate.s', 'baseline.su', 'candidate.su', 'run-device-trace.py']:
        add(M / name, 'validation/author/' + name, (M / name).suffix in {'.s', '.su', '.patch'})
    for folder in ['fixture', 'baseline', 'candidate', 'codegen', 'root-frozen']:
        tree(M / folder, 'validation/author/' + folder, text_types)
    for folder in ['checks/san', 'checks/mutants']:
        tree(M / folder, 'validation/author/' + folder, log_types)
    for path in sorted(M.glob('root-*')):
        if path.is_file() and path.suffix in log_types:
            add(path, 'validation/root/' + path.name, path.suffix == '.log')
    for name in ['review.txt', 'validation.json', 'input-manifest.json', 'run_codegen.py']:
        add(I / name, 'validation/independent/' + name)
    tree(I / 'codegen', 'validation/independent/codegen', text_types)
    for folder in ['host-san', 'mutants-san']:
        tree(I / folder, 'validation/independent/' + folder, log_types)
    for path in sorted(D.iterdir()):
        if path.is_file() and path.suffix in text_types:
            add(path, 'diagnostic/' + path.name, path.suffix == '.patch' or path.name == 'warmup-source-excerpts.txt')
    for folder in ['candidate', 'codegen', 'config-source', 'rejected-parser-v1']:
        tree(D / folder, 'diagnostic/' + folder, text_types)
    tree(D / 'checks/warmup-parser-final', 'diagnostic/checks/warmup-parser-final', log_types)
    # Canonical author sources plus independent input manifests avoid a second
    # copy of every generated kernel/mutant while retaining command/log proofs.
    for path in sorted((I / 'diagnostic').iterdir()):
        if path.is_file() and path.suffix in text_types:
            add(path, 'diagnostic/independent/' + path.name, path.suffix == '.log')
    for folder in ['codegen', 'final-actual-host-san']:
        tree(I / 'diagnostic' / folder, 'diagnostic/independent/' + folder, text_types if folder == 'codegen' else log_types)
    add(I / 'diagnostic/final-parser-frozen/hashes.json', 'diagnostic/independent/final-parser-hashes.json')
    add(T / 'summarize_pair.py', 'validation/summarize_pair.py')
    add(Path(__file__).resolve(), 'validation/curate_moe_fast_issue.py')

    readme = '''LFM native MoE issue leaf -- not adopted

The proposed production patch and diagnostic patches in this directory are
evidence only. They were removed; the retained GU normal source and DSP were
restored. See validation/root/root-normal-restored.json and artifacts/ for
exact source, deployment patch and binary hashes. No production change is
requested by this archive. The candidate's normal throughput gain was not
established, despite exact output and actual dispatch confirmation.

Control is retained GU production36a75df1a. Candidate e8bbf0ae uses the same
ARM runtime and exact native LFM prefill math, accumulation order, DMA and
worker synchronization. It permits1..64 compacted experts, exact K2048,
inter1536/N2048, M>1, all-native TL2/SiLU and existing validated VTCM bounds.
An earlier draft wrongly required E64; the final candidate includes E1/4/60
whole-caller tests. No buffer, cache, reader, preload or persistent metadata
was added to the proposed production code.

v81 scalar issue packets per two HMX issues fall from6 to4; the leaf has no
inner stack reload. Caller frame remains17152B, leaf0. This is code-generation
evidence, not elapsed-time evidence. HMX stalls, call overhead and changed
overlap can hide or outweigh scalar issue savings. Gemma fallback keeps the
old kt loop but acquires a per-tile dispatch predicate and changed code layout.
No Gemma device runs were made and unchanged Gemma performance is not claimed.

S26 Ultra R3CY70LV96T; native TL2 experts and affine int4 dense weights.
C16, Direct I/O/PAGE_CACHE0, cache grow0, preload128, fixed readers, predict2,
residual0.5, stable slots/early prefill/pair reads/demand/router handoff1.
Normal order is A1 B1 B2 A2 A3 B3 separately for1024 and2048,64 generation
tokens. All12 normal runs and both diagnostic runs match all65 reference
dumped IDs (one prefill output token plus64 generation tokens).

                  control median   candidate median   paired prefill wins
1024 prefill TPS       838.657            819.200                 1/3
2048 prefill TPS       826.473            844.188                 2/3
1024 decode TPS         17.6260            17.5294
2048 decode TPS         16.8111            16.6018
Prefill median changes are -2.32% and +2.14%; paired2048 deltas range from
-12.02% to +8.10%. These results do not establish a reliable gain. The candidate
is not adopted. Decode is outside M>1; these differences are not speed claims.

Largest normal measured peaks(KiB), control->candidate:
1024:2733264->2733368;2048:3052252->3051360. No memory reduction is claimed.
The monitor covers the entire supervised child from beforeexec through exit,
requested20ms samples of RssAnon plus same-sample fdinfo allocated sizes,
deduplicated by mount/inode. All readable peak identities/sizes are complete.
This is process allocation accounting, not total physical device memory.
Unreported DSP/driver/VTCM and between-sample transients are excluded. The
separate app-run-only metric and ru_maxrss must not be added to this composite.

Validation: full DSP build and clang-format14; author and independent full-TU
ASan/UBSan11calls plus25guard checks;14 assertion-detected mutants;688128
byte-identical FP32 output bytes and matching packet addresses/control order.
The host uses scalar HMX/HVX/quant stand-ins and immediately executes jobs
while tracking pending waits. It does not model silicon arithmetic, parallel
cache timing or the Hexagon-only i24 staging path. E60 whole caller has2 active
experts. Only the pre-existing scalar swiglu signed-shift check is disabled
for hvx_scalar.c; production TUs retain full ASan/UBSan. Full repository and
Windows/Tizen/Yocto CI were not run for this rejected experiment.

Actual dispatch trace is separate and excluded from normal TPS. Its uint32
atomic sequence adds4B diagnostic-only metadata; local foreground counters
count actual GU/DN leaf entries. They are issue entries, not completed readout
tiles, so only successful rc0 records are aggregated. Compile-time trace off
is byte-identical to the normal v81 whole object. Maximum FARF payload is146
characters (211 including a64-character prefix and NUL), within256.

The first strict parser rejected both traces: it assumed38 matching-M calls,
but the unchanged shared loader issues a same-M warmup. This initial failure
and parser are preserved. Source-derived correction requires exactly39 unique
calls: seq1 is the single load warmup at M=max(512,INIT_SEQ_LEN), E64,64 rows
per expert, GU6144/DN4096; seq2..39 are the actual38 requested-M layers. The
captured config hashes bind INIT_SEQ_LEN to1024 or2048. It does not simply
relax the count or discard every matching warmup signature. Missing, moved,
duplicated and corrupted warmups, incomplete/failed calls, conflicting SDK
replays, mixed PIDs, noncontiguous seq, raw-tick collisions, overflow and
truncation are rejected. The final parser has8 positive and288 negative
checks including182 prefix truncations. Complete identical SDK replays alone
are deduplicated. Delivery reordering is allowed; logicalseq governs ticks.

Both unchanged raw captures pass the corrected parser, each with one warmup
and38 real calls, all actual leaf entries selected:
1024: compact E sum2283; GU362496/DN241664;3776 blocks.
2048: compact E sum2328; GU592128/DN394752;6168 blocks.
Warmup is excluded from those counts. The logs confirm actual dispatch, not
a speed benefit. Diagnostic timing is not included in the normal table.

Reproduction: decompress the .gz sources/patches beside their snapshots;
use validation/author/review.txt and diagnostic/README.txt for fixture/parser
commands. Compiler commands, sanitizer logs, hashes and independent reviews
are included. Regenerable intermediate kernel/mutant source copies and binary
outputs are omitted; canonical frozen source plus runner and input manifests
are retained. Gzip is deterministic/lossless. manifest.json hashes every
archived file except itself. This curator performs no device operation.
'''
    report = {'pass': True, 'adopted': False, 'normal_runs': len(results), 'normal': computed,
              'diagnostic': traces, 'parser_sha256': final_parser_sha,
              'restored_normal': restored, 'source_files_selected': len(copies)}
    if args.check_only:
        print(json.dumps(report, indent=2))
        return
    require(not out.exists() or not any(out.iterdir()), 'Output must be absent or empty: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    for relative, (src, compress) in copies.items():
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = src.read_bytes()
        destination.write_bytes(gzip.compress(payload, mtime=0) if compress else payload)
    (out / 'README.txt').write_text(readme)
    (out / 'validation/curation.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'files_hashed': len(manifest),
                      'manifest_sha256': sha(out / 'manifest.json'), 'pass': True}, indent=2))


if __name__ == '__main__':
    main()
