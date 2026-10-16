LFM native TL2 ARM prediction register accumulation — not adopted

The normal model comparison did not establish consistent decode TPS improvement.
The exact same seven runtime binaries were used with
NNTR_LFM2_PREDICT_REGISTER_ACCUM=0/1. Only one of three decode pairs improved at
each prompt size. CPU-only improvement and confirmed actual helper execution
were insufficient to retain the candidate. Prefill variation is recorded but
is not causally attributed to this decode-only hint kernel.

Normal results (three OFF and three ON runs per prompt, AB/BA/AB order)
                        OFF median    ON median      change       ON wins
  1024 decode TPS         17.6066       17.5872        -0.11%        1/3
  2048 decode TPS         16.3683       16.4187        +0.31%        1/3
  1024 prefill TPS        848.384       827.141         -2.50%        0/3
  2048 prefill TPS        852.978       787.995         -7.62%        2/3
The full arrays and paired deltas are in summary.json. The mixed 2048 prefill
pairs and median differ because of run variation. No causal prefill regression
or universal decode benefit is claimed. All 12 runs retained the reference's
65 dumped IDs: one prefill token plus 64 decode iterations. All recorded run
validity and complete-run sampled-memory checks passed. No additional tests,
model repeats, or phone access were performed during this archive curation.

Identity and restoration
OFF and ON both use libnntrainer.so
297c9ff9b6a91b332eb98aaab67a4af28ab22c0275ef4521bcd215c5df2ada3a.
Their other six binaries are also identical, including original app c9efd442...
and DSP 3baddb92.... No DSP rebuild enters this comparison. The seven full
hashes, exact configuration, environment and source patch are preserved.
Production patch ba6055d746a37aecbb98f38a260ba0671071c285f6a3d426ebf5ec3a6c364f3d
and tests7c24b119... are evidence only; the candidate was not adopted.

After rejection, root restored the two core source files to their recorded
baseline hashes and rebuilt all four core binaries to their original hashes,
including libnntrainer.so072b5ca2.... The original seven-binary phone variant
lfm-moe-row-norm-restored was freshly verified. Its original app and DSP are
also preserved in this candidate's root-frozen snapshot. Root's current app
already contains a separate later ARM-router experiment; this archive does
not claim the live root app was restored. Only the two-source/four-core restore
and explicitly verified original phone runtime are asserted.

Fixed workload and memory scope
C16, PAGE_CACHE=0, preload128, grow0, eight CPU threads, native TL2CX_WH experts,
affine FC, prediction2/residual0.5, and unchanged reader/pool counts are fixed.
Startup logs confirm cache_slots=608, direct slots32/readers2 and demand readers4.
No model, quantization, activation/expert buffer, persistent cache or reader
capacity change was introduced. The captured bool fits existing lambda padding;
closure size72B and existing allocation count remain unchanged.

Maximum sampled full-run anon+deduplicated-fdinfo values, KiB:
  1024 OFF2733764 -> ON2734112 (+348)
  2048 OFF3036256 -> ON3037628 (+1372)
No mechanism is attributed to these small observed peak differences. The external
supervisor samples the child's lifetime including load, requested every20ms,
using RssAnon plus inode-deduplicated fdinfo size at each sample. It is not total
system/physical DSP memory; FD size being DMA is not independently verified.
Every selected peak has complete readable-size-FD deduplication and zero recorded
read/identity/conflict/field/overflow errors. Actual intervals, nonpeak/terminal
incomplete samples and limitations remain in raw memory JSON. Between-sample
peaks can be missed. VmHWM and application generation-only peak are not added
again to this composite. Per-run intervals and completeness are audited in
validation/archive-checks.json.

Candidate and validation scope
The new helper uses16 NEON vector accumulators for the same h0..2047 FMLA order,
then stores the existing scores once. Exact option1, native LFM geometry,
AArch64, hidden2048, E64 and sigmoid bias gate it. Quantization, DSP router,
mixture weights, rank/tie policy, residual state, prefetch filtering, ownership,
queue/cancellation behavior, cache and final accumulation order are unchanged.
Gemma, nonnative/other shapes, OFF and non-ARM retain the original path. No Gemma
device performance measurement is claimed.

The removed1,048,320B/prediction is a logical score-array load/store count, not
measured DRAM traffic. Whole-core NDK codegen shows no hot score load/store or
stack spill; predictor192B, callback336B, initializer288B and closure72B are
unchanged. An earlier appended-capture draft had a larger initializer frame;
its patch is retained separately and is not the final production source.

Author and independent reviewer each checked actual headers, raw-score prefixes
and extracted callback bodies:130 cases,125 raw score arrays,144 callback jobs,
12 env/shape guards, ASan/UBSan, non-ARM fallback and nine runtime mutants.
Both compiled whole baseline/candidate Android core; final candidate object
hashes agree. Root's native ARM fixture passed the same130/125/144/12 checks.
Native helper/callback extraction matches final code; capture padding was
subsequently finalized and whole-TU compiled. Queue/pinning in the fixture are
stand-ins; these checks do not prove real I/O/DSP scheduling or speed. Original
reports and historical pending-model status are preserved without rewriting.
This README and summary.json state the final model verdict.

Separate actual-dispatch diagnostics
At close, after the existing predictor drain, a compile-time diagnostic reports
callbacks/dots/qualified=2368 (64*(38-1)). OFF has fast0; ON has fast2368 from
inside the actual new helper. Both instrumented runs retained the reference IDs.
The strict single-session stderr parser rejects missing/duplicate/truncated or
unexpected extra dot-product cohorts. Diagnostic timing is not included in normal
results. Diagnostic counters add24B only to the diagnostic core. Macro-OFF
predictor/callback/initializer text is exact; three unrelated logging __LINE__
operands differ, so complete-object identity is not claimed. Root checked the
normal four-core hashes after removing the overlay; an initial timestamp-driven
no-op rebuild was detected by SHA and corrected before normal deployment.

CPU-only micro versus model
CPU0 isolated warm-weight median18.882242 ->9.527586us; sweep of37 synthetic
layer weights21.877459 ->14.139568us. Both have31 balanced AB/BA pairs and37 exact
top-k results. The micro includes dot/finite/sigmoid/top-k/existing allocations,
but excludes residual callback, filtering, real queue, I/O, DSP and concurrent
model work. It is not a decode latency/TPS bound. The model experiment did not
validate an end-to-end gain, despite this isolated improvement and actual use.

Archive/reproduction
manifest.json hashes stored and decompressed bytes for every payload; gzip
mtime=0. raw/ holds all12 ordinary logs/IDs/metadata/memory reports; diagnostic/
is separate. validation/ retains frozen source/tests, author and independent
reports, focused codegen, native/micro records and root build/restore evidence.
Large complete disassembly and executable payloads are omitted; exact hashes
and relevant command records are kept. No models/toolchains/SIMDe are bundled.

  python3 reproduction/verify_archive.py ARCHIVE
  python3 reproduction/reproduce_host.py --archive ARCHIVE \
    --simde-include EXTERNAL_SIMDE_0.8.2 --output NEW_DIR --prepare-only
Omit --prepare-only to run the archived host/fallback checks; --mutations adds
nine mutants. All source dependencies for these bounded fixtures are archived.
The original runner supports --native-arm --ndk NDK_ROOT --build-only; building
or deploying a native executable is separate from archive verification.
