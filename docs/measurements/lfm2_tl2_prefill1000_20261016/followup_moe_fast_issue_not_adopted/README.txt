LFM native MoE issue leaf -- not adopted

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
