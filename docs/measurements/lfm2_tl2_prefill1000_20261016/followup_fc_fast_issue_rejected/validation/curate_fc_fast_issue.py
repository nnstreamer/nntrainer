from pathlib import Path
import gzip
import hashlib
import json

R = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
E = R / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
T = Path(__file__).parent
F = Path('/tmp/nntr-lfm2-fc-fast-issue-20261016')
I = Path('/tmp/nntr-lfm2-fc-fast-issue-review-20261016')
P = Path('/tmp/nntr-lfm2-fc-regression-diag-20261016')
PI = Path('/tmp/nntr-lfm2-fc-regression-diag-review-20261016')
D = E / 'followup_fc_fast_issue_rejected'
D.mkdir(exist_ok=True)

def copy(src, relative):
    dst = D / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = src.read_bytes()
    dst.write_bytes(gzip.compress(payload, mtime=0) if dst.suffix == '.gz' else payload)

def tree(src, target, suffixes=None):
    if suffixes is None:
        suffixes = {'.txt', '.json', '.log', '.s', '.su', '.c', '.h', '.inc', '.py', '.patch'}
    for f in sorted(src.rglob('*')):
        if f.is_file() and f.suffix in suffixes:
            compressed = f.suffix in {'.log', '.s', '.su', '.c', '.h', '.inc', '.patch'}
            copy(f, target + '/' + str(f.relative_to(src)) + ('.gz' if compressed else ''))

normal = [j['tag'] for j in json.loads((F / 'root-normal-jobs.json').read_text())]
profiles = [j['tag'] for j in json.loads((F / 'root-profile-jobs.json').read_text())]
phases = ['continue-fc-phase2048-control', 'continue-fc-phase2048-candidate']
for tag in normal + profiles + phases:
    result = json.loads((E / (tag + '.json')).read_text())
    memory = json.loads((E / (tag + '.memory.json')).read_text())
    peak = memory['peak_dedup_sample']
    assert result['valid'] and result['ids_match_reference'] and memory['child_exit_code'] == 0
    assert not result['concurrent_processes'] and result['moe_cache_experts'] == 16
    assert peak['dedup_complete_for_readable_size_fds']
    assert not any(peak[k] for k in ['fdinfo_read_errors', 'unknown_identity_fds', 'inode_size_conflicts'])
    assert hashlib.sha256((E / (tag + '.ids')).read_bytes()).hexdigest() == result['ids_sha256']
    for key, value in {'NNTR_HTP_PAGE_CACHE': '0', 'NNTR_MOE_CACHE_GROW': '0',
                       'NNTR_MOE_PRELOAD_EXPERTS': '128', 'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1'}.items():
        assert result['environment'][key] == value
    assert result['performance_comparison_eligible'] == (tag in normal)
    extensions = ['json', 'ids', 'log', 'memory.json']
    if result['profile']:
        extensions.append('logcat.txt')
    if tag in phases:
        extensions.append('affine.logcat.txt')
    for ext in extensions:
        copy(E / (tag + '.' + ext), 'raw/' + tag + '.' + ext + ('.gz' if ext in ['log', 'memory.json', 'logcat.txt', 'affine.logcat.txt'] else ''))

for variant in ['tl2-gu-direct-epilogue', 'lfm-fc-fast-issue', 'fc-issue-phase-control', 'fc-issue-phase-candidate']:
    copy(E / (variant + '-artifacts.json'), 'artifacts/' + variant + '.json')
    copy(T / (variant + '.source.patch'), 'artifacts/' + variant + '.source.patch.gz')

for name in ['review.txt', 'validation.json', 'source-hashes.json', 'manifest.json',
             'production.patch', 'root-normal-jobs.json', 'root-normal-runs.log',
             'root-normal-summary.json', 'root-profile-jobs.json', 'root-profile-runs.log',
             'root-profile-summary.json', 'root-dsp-build.log', 'root-normal-restored.json']:
    p = F / name
    copy(p, 'validation/author/' + name + ('.gz' if p.suffix in {'.log', '.patch'} else ''))
for folder in ['fixture', 'baseline', 'candidate', 'codegen', 'root-frozen']:
    tree(F / folder, 'validation/author/' + folder)
for folder in ['checks/final-san', 'checks/final-numeric-san', 'checks/mutations']:
    tree(F / folder, 'validation/author/' + folder, {'.json', '.log'})
for folder in ['codegen', 'frozen']:
    tree(I / folder, 'validation/independent/' + folder)
for folder in ['caller-san', 'numeric-san', 'mutations-san']:
    tree(I / folder, 'validation/independent/' + folder, {'.json', '.log'})
for name in ['review.txt', 'validation.json', 'frozen-manifest.json']:
    copy(I / name, 'validation/independent/' + name)

for name in ['README.txt', 'validation.json', 'manifest.json', 'source-hashes.json',
             'control-diagnostic.patch', 'candidate-diagnostic.patch', 'parse_fc_regression.py',
             'afc2_base.py', 'parser_check.py', 'control-device.json', 'candidate-device.json']:
    p = P / name
    copy(p, 'diagnostic/' + name + ('.gz' if p.suffix == '.patch' else ''))
for folder in ['fixture', 'control', 'candidate', 'v81', 'root']:
    tree(P / folder, 'diagnostic/' + folder)
tree(P / 'checks', 'diagnostic/checks', {'.json', '.log'})
tree(PI, 'diagnostic/independent', {'.json', '.txt', '.py', '.log'})
copy(T / 'summarize_pair.py', 'validation/summarize_pair.py')
copy(Path(__file__), 'validation/curate_fc_fast_issue.py')

readme = '''LFM affine FC issue leaf -- rejected normal throughput experiment

Control is retained GU source36a75df1a (binary built before its commit on207d).
Frozen candidate patch e1e45c09574d0b131fea13d2de3822b413effb47b550d22e81c718dfe93e9e0e.
Normal DSP9056191eed9d8766c033cccc379227e3f5d3e672a2e99b4c5301965b3b0fcf83.
The patch and both diagnostic variants are evidence ONLY, removed from production.
Exact source/client/DSP hashes and deployment patches are under artifacts/.

Only native LFM affine K2048 prefill geometry selects the proposed small issue
leaf. It preserves HMX packet order, math, clear/readout, worker and DMA ordering.
v81 assembly has six packets/issue in the old loop and two in the leaf, but
alignment mask calculation was already hoisted. Caller stack656->672B (+16B),
leaf0; no heap/VTCM/cache/reader/preload capacity changes. Assembly packet counts
are NOT cycle counts: stalls, call overhead and changed overlap can dominate.

S26 Ultra R3CY70LV96T; native TL2 experts / affine int4 dense weights. C16,
page cache0 (Direct I/O), cache grow0, preload128, predict2, fixed readers,
stable slots/early prefill/pair reads/demand handoff/router handoff1.
Order A1 B1 B2 A2 A3 B3 independently at1024 and2048,64 decode tokens.

                 control median  candidate median  paired prefill wins
1024 prefill TPS       840.033           827.141               1/3
2048 prefill TPS       888.889           785.276               0/3
1024 decode TPS        17.6309           17.6649
2048 decode TPS        16.8377           16.8200
Prefill median changes -1.53% / -11.66%; the candidate is not adopted.
Decode is outside its M>1 guard; the small differences are not speed claims.
All12 normal runs have exactly the reference65 saved IDs.

Largest normal measured peaks(KiB), control->candidate:
1024:2734508->2731308;2048:3052264->3065180. The larger2048 candidate peak
is recorded without claiming a memory reduction (about2.923GiB).
The monitor covers the entire supervised child from beforeexec through exit,
requested20ms samples of RssAnon plus same-sample allocated fdinfo sizes,
deduplicated by mount/inode. All peak identities/readable sizes are complete.
This is allocated process accounting, not total physical device memory;
unreported DSP/driver/VTCM and between-sample transients are excluded.

Validation: fullv81 DSP build; author and independent actual-caller46
ASan/UBSan cases plus8 guard/order mutants;7 numerical scalar cases with
233472 byte-identical FP32 output. Host primitives emulate arithmetic and
logical queue retirement, not silicon concurrency/cache behavior or timing.
No Gemma phone repeats or full repository/Windows/Tizen/Yocto CI were run.

A separate normal profile pair had total prefill2470->2512ms while aggregate
FC host timings decreased. These aggregates include100 load warmups, so a
second controlled phase diagnostic separates main work and actual dispatch.
The diagnostic instruments both variants identically and is excluded from
normal TPS. Its parser requires exact source-derived G/T/B cohorts, arena,
banks, counters, serial timing and complete records. The main2048 cohort has
190 kernel calls and296960 readouts;100 load warmups are separate. Existing
ARM1920 and DSP1728 row caps produce nested splits for conv FC. The reports
in diagnostic/root/ contain strict raw-device comparison and interpretation.
Readout time includes HMX completion; other includes required issue work and
instrumentation. Neither is a directly removable cycle budget. A timing-only
diagnostic cannot establish the cause of the uninstrumented12-run regression.

Source/DSP were restored exactly before starting the separate MoE candidate.
Reproduction commands and fixture limits are in validation/author/review.txt
and diagnostic/README.txt. Gzip files are lossless snapshots. manifest.json
hashes every file in this evidence bundle except the manifest itself.
'''
(D / 'README.txt').write_text(readme)
manifest = {str(p.relative_to(D)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(D.rglob('*')) if p.is_file() and p != D / 'manifest.json'}
(D / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Curated rejected FC candidate:', len(manifest), 'hashed files')
