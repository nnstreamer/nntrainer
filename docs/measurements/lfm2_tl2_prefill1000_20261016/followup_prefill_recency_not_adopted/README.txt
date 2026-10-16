LFM2 first-seen prefill recency candidate — NOT ADOPTED
2026-10-17 | base 742735f3c61803fac51189012ba4caf588a3bd50

This archive supersedes the pending-model wording in author/README.txt and the
not-yet-implemented wording in queue/README.txt. Those original reports remain
byte-for-byte frozen provenance. The candidate was implemented, independently
reviewed, run on S26 Ultra, and rejected for lack of a consistent whole-model
prefill benefit. Accepted production source and all seven local/phone runtime
artifacts were restored. No pending permission, test or deployment is implied.

The arithmetic and LRU parity checks passed. The larger 2048-token spread is
retained in full; no run is discarded and its cause is not established. This
record does not assert that recency compaction caused the entire observed
slowdown, nor that the work reduction guarantees a speedup. No PROFILE1 or
diagnostic TPS value is used in the normal performance conclusion.

Normal performance (same compiled binary, option OFF/ON)
-------------------------------------------------------
Each length ran A1 B1 B2 A2 A3 B3; A=0, B=1. All 12 runs generated the same 65
IDs as their respective reference. Exact environment/settings are in raw JSON.
Fixed C16, preload128, PAGE_CACHE=0, cache growth=0, eight existing CPU workers,
unchanged existing reader counts. No model/quantization/cache-capacity change.

1024 prefill TPS:
  OFF: 831.169 / 830.495 / 835.918
   ON: 861.228 / 849.088 / 803.768
  Median: 831.169 -> 849.088 (+2.156%); 2/3 paired wins.
  Paired differences: +3.616%, +2.239%, -3.846%.
  Decode median: 17.5535 -> 17.6649 TPS (+0.635%); 2/3 wins.
  Maximum sampled full-run peak: 2,724,044 -> 2,714,292 KiB.

2048 prefill TPS:
  OFF: 822.490 / 954.779 / 977.099
   ON: 833.876 / 868.901 / 799.688
  Median: 954.779 -> 833.876 (-12.663%); 1/3 paired wins.
  Paired differences: +1.384%, -8.995%, -18.157%.
  Decode median: 16.5631 -> 16.6104 TPS (+0.286%); 1/3 wins.
  Maximum sampled full-run peak: 3,019,488 -> 3,018,544 KiB.

The 2048 OFF range itself is wide (822.490..977.099 TPS). This small matrix
does not identify the source of variability. It does not demonstrate a stable
prefill win, so the candidate was not retained. Diagnostic/profile timing is
excluded from this table and from any claimed speedup.

Thermal observations are retained in device/root-thermal-observations.json and
checked against the original run JSON. All 24 before/after snapshots for the
12 normal runs report ThermalStatus 0. However, each dump repeats the SKIN
name/type in both its cached and HAL sections: cached remains 37.9 while HAL
observations range from 30.8 to 39.3. Matching labels do not establish a single
identical physical sensor or update cadence. These endpoint observations do
not prove stable frequencies, absence of DVFS or absence of throttling during
inference. They do not establish the cause of the large 2048 variation. No run
is discarded or reclassified based on these readings; the decision is unchanged.

Memory is the existing 20 ms supervised-child sampled composite: RssAnon plus
fdinfo allocation sizes deduplicated by inode/mount identity where available,
including loading. It is not total physical memory or total DSP memory. Private
DSP/driver/VTCM allocations and between-sample peaks may be absent; allocated
fd sizes need not be resident bytes. VmRSS/ru_maxrss must not be added to the
composite. All original samples and limitations are retained in .memory.json.

Actual dispatch and preserved behavior
-------------------------------------
The separate two diagnostic runs each show all 38 ordered MoE layers with
fast=1, complete 65-ID parity and valid memory records:
  M1024: 350,208 raw entries -> 2,405 retained; 347,803 removed.
  M2048: 700,416 raw entries -> 2,416 retained; 698,000 removed.
This establishes executed hint compaction, not a time bound. The parser requires
exactly 38 layers, geometry, terminal marker, counts and dispatch; it rejects
missing/duplicate/reordered records. Input is one process's stderr; the marker
does not itself encode or validate PID. Root collected each child separately.

NNTR_LFM2_COMPACT_PREFILL_RECENCY=1 is exact opt-in. AArch64+HTP, native TL2,
K2048/I1536/N2048/E64/top4, normal 1024/2048 prefill and original norm/router
conditions guard the path. All 128 GU/DN tensor metadata records are checked.
Configured or already-open routing trace, M1, Gemma and generic paths fall back.
Every sigmoid, partial_sort(9), top4 weight and expert/token append remains in
the same source order. Only first-seen ID hint materialization changes. The
existing global reverse plus mutex-protected refresh retains the same final
LRU/LRFU order, including Weight object aliases and absent keys. Generic
ExpertLru is unchanged. No persistent weight/activation cache, reader or buffer
capacity is added. Assignment stack is 528 -> 544 bytes; caller remains 3008.

Evidence and validation scope
-----------------------------
author/: exact production/tests patches, before/candidate source, portable
fixture, sanitizer/O3/generic results, whole-TU commands/stack records and
diagnostic source/parser/formatter results. Frozen source ac743dac..., patch
1f130610..., tests30397390.... No binary or vendor SDK is included.

review/: independent report, actual ExpertLru stress fixture/runner, six semantic
mutants and retained results. Actual method checks ran in six environments,
49 numeric + 162 guards + 72 LRU transitions + 2 observer cases each. Additional
2,880 LRU/LRFU transitions and six intended mutant failures passed. ARM before/
after, Android x86 and Linux GCC whole TUs compiled. No Windows whole build or
TSAN run is claimed.

The method/guard inspector uses actual extracted production method, reshape,
caller and guard text with Tensor metadata/getWeight/reshape stand-ins. Actual
std::partial_sort/libm and the unchanged ExpertLru execute. These checks alone
do not prove full Tensor/RunLayerContext ownership, I/O, DMA or DSP behavior.
The same fixture additionally passed all six environments on actual S26 ARM
(binary hash abf44674...). The full-model 12+2 runs provide separate output and
dispatch evidence. NaN parity exercises the same existing comparator; this is
not a new guarantee that arbitrary nonfinite comparator inputs are well-ordered.
The formatter check uses metadata/counter stand-ins and proves log format only.

device/: native six-environment raw logs, build commands/results, exact seven
runtime hashes, accepted/normal/diagnostic source snapshots, actual deployment
patches and restoration records. raw/: 12 normal + 2 diagnostic logs, metadata,
65-ID files and complete memory samples. Original thermal fields are preserved.

Normal OFF and ON both used app 7cd08a2d... with the other six artifacts identical.
Only app libcausallm.so differed from accepted c34aa784.... Diagnostic app was
071b82d0...; the original accepted DSP 3baddb92... never changed in this matrix.
Actual diagnostic source differs from the author's macro-disabled source by
the enable define and formatting; the deployed base-to-diagnostic patch is
retained and replayed byte-exactly, rather than assuming author/source identity.

device/root-normal-restored.json proves original CPP103b1e1d.../header5af03853...
were restored, the full app rebuilt to c34aa784..., all seven local artifact
hashes matched accepted, and all seven phone hashes were freshly checked after
both diagnostics for lfm-router-arm-1. verify_archive.py independently replays
the author production and actual normal/diagnostic deployment patches in memory.

Matched queue audit is a separate profile cohort
------------------------------------------------
queue/ contains the two earlier matched normal-app ARM-router OFF/ON 1024 logs,
their raw IDs/JSON/memory, original scripts and summaries. These are not recency
normal-performance runs and are not used in the TPS table above. The same ordered
2304 expert reads requested 4,656,726,016 bytes in both; six readers, 65 batches.
Read window 1178.197 -> 1179.362 ms; aggregate read-body 7063.616 -> 7064.392 ms;
exposed wait 28.807 -> 42.162 ms; registration bracket 1.652 -> 1.938 ms.

Read-body occupancy of 99.921% / 99.834% includes blocking pread, copy and other
work. It is neither CPU utilization nor a storage hardware bandwidth ceiling.
There is no all-readers-idle/empty-queue interval inside the recorded read window;
queued-but-not-started jobs exceed six for 99.513% / 99.440% of that window.
This is recorded mechanism evidence, not proof that faster I/O or other
implementation changes cannot improve throughput.

queued_us/jobs_ahead are captured under enqueue lock before push. READ start
is after pop/unlock/cpuBit, so submitted-minus-started may include up to six
already claimed jobs; the reconstruction uses >6 for definite FIFO work.
Read-end precedes done publication. Batch logging occurs after join in the
caller and can delay subsequent submission/compute. The register bracket is
the host registration section after join, not an isolated DSP metadata-DMA
timer; stable-slot reuse often avoids a registration RPC. No explicit held/free
slot timestamp exists. The parsers exclude 102 load/warmup compute calls by the
load/repack boundary (warmup can itself be M1024), retaining 120 main compute
calls including 38 MoE calls. Trace overhead and the unmatched worker/caller
durations prohibit treating these sums as a standalone speedup budget.

Verification / reproduction
---------------------------
From this directory:
  python3 verify_archive.py

This validates the complete manifest, unchanged copied evidence, all 12 normal
raw rates/IDs/settings/peak samples, two strict diagnostic cohorts, six native
and host environments, thermal-summary/raw equivalence, six mutant outcomes,
source patch replay, seven-artifact
restoration and both queue summaries. It recomputes results.json. It does not
compile or rerun any test, model, SDK operation or phone command.

Standalone diagnostic readback:
  python3 author/diagnostic/parse.py \
    --log raw/diagnostic/continue-prefill-recency-diagnostic1024.log \
    --rows 1024 --enabled 1 --output /tmp/recency-dispatch.json

Standalone queue readback (create the output directory first):
  python3 queue/analyze_matched.py --repo queue/repo --output-dir /tmp/queue-readback
The imported analyze_existing.py profile helper is retained unchanged. Its
historical all-input main command needs older unbundled evidence; it is not the
archive's replay entry point. Original absolute build paths are provenance,
not portable runtime requirements of verify_archive.py.

Optional fixture reproduction, not rerun during curation:
  python3 author/candidate/test/htp/host/run_lfm2_prefill_recency_check.py \
    --source author/candidate --reference author/before \
    --output /tmp/recency-fixture --sanitize
The independent run_lru_stress.py and run_mutants.py use the same --source,
--reference and --output options. Native compilation requires an external NDK;
whole-app compilation requires an external matching full repository/SDK.

manifest.json is the authoritative archive inventory. Original nested manifests
are provenance and may mention deliberately excluded objects/binaries. All
copied files match their source hashes in provenance.json. No ELF, library,
model weights, compiled Python, vendor SDK or phone runtime is bundled.
