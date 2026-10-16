LFM2 native TL2 / affine, S26 Ultra, 2026-10-16

This checkpoint improves LFM2 at fixed C16 without a weight page cache.
The 1000 prefill / 25 decode TPS target has not been reached.
Baseline is PR #4428 commit bd556b96030d903f2cc235f43f1795a5d9e2c345.
Implementation: bf10e0e04; exact source hashes are in repro/source-sha256.json.

Three normal runs per model/runtime, alternating A1 B1 B2 A2 A3 B3:
                                  Prefill TPS       Decode TPS      Peak GiB
LFM2 1024 input / 64 decode     598.131 -> 786.482  12.9267 -> 17.0621  2.605 -> 2.612
LFM2 2048 input / 64 decode     554.263 -> 707.182  12.1905 -> 16.0683  2.907 -> 2.938
Gemma4 1024 input / 64 decode   629.766 -> 629.379  12.7974 -> 12.7949  3.226 -> 3.215

TPS values are medians. Memory values are the largest sampled composite
across the three runs, not median peaks. LFM2-2048 optimized prefill varies
from 690.492 to 813.021 TPS. Gemma differences are within observed variance;
these runs do not establish a Gemma speedup. See results.json for every run,
ranges, runtime flags, peak samples and error counts. raw/ contains complete
compressed stdout/stderr, full memory samples and all generated IDs.

All 18 normal runs match their reference output IDs exactly. Each dump has
65 IDs: one output from prefill, followed by 64 decode steps. The legacy
token trace omits the prefill output when the prompt fills the prefill plane;
its 64 output IDs match the dump suffix. The test-only repeated-prompt CLI
also runs each LFM/Gemma model twice on one loaded handle: both inputs and
all 65 outputs match the normal-run reference. These are fixed-prompt
correctness checks, not a broad language-model quality evaluation.

Fixed conditions: S26 Ultra SM-S948U / SM8850 / Hexagon v81, 8 ARM threads,
moe_cache_experts=16, NNTR_MOE_CACHE_GROW=0, NNTR_HTP_PAGE_CACHE=0,
NNTR_LOAD_PREFETCH=0 and NNTR_MOE_PRELOAD_EXPERTS=128. There is no additional
host/victim weight cache. LFM has 608 global expert slots and 32 existing
decode spare slots; Gemma has 480 global slots. Prefill uses six readers;
decode prediction uses two. Initialization/preload time is outside prefill
TPS but inside the memory-monitor lifetime. Normal runs use the same memory
monitor and no layer/HTP/I/O profiling on either runtime.

LFM optimized flags:
  NNTR_HTP_PREDICT=2
  NNTR_MOE_PREFETCH_RESIDUAL=0.5
  NNTR_HTP_TL2_STABLE_SLOTS=1
  NNTR_LFM2_PREFILL_EARLY=1
  NNTR_LFM2_DECODE_PAIR_READ=1
These are opt-in. Gemma uses none of these LFM overrides.

The measured model file is nntr_lfm2_tl2_affine.bin, 5,912,746,496 bytes,
SHA256 ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d.
Its expert format is native TL2CX_WH; dense/attention/lmhead matrices use
QS4CX_WH_AFFINE, and embedding/norm/router/conv use FP32. The supplied
conversion report states exact verification of 4,864 expert tensors and
107 affine tensors, without requantization. We independently checked the
whole deployed file hash; we did not independently revalidate its original
source checkpoint. repro/model-conversion.json preserves that distinction.
Exact device configs and prompts are under repro/models/. Weight/tokenizer
payloads are not distributed with this record.

Implemented work includes compact i24/VTCM staging, reduced DMA descriptor
work, cached DSP attention/router scratch, guarded dynamic-Q8 attention/KV
paths, parallel M1 convolution, exact contiguous ARM convolution, projection
epilogues, and removal of redundant residual copies. Expert streaming pairs
GU/DN Direct I/O reads, starts future reads before the first prefill block,
and reuses registered physical-slot handles with generation refresh before
DSP compute. Decode uses LFM sigmoid/bias prediction hints and completes
paired reads in existing spare slots. Actual router decisions are unchanged.

The reader FIFO has no layer barrier: a worker starts the next queued expert
as soon as it finishes a read, including experts in subsequent layers.
Existing slot availability/LRFU rules still bound how far ahead jobs can be
queued. Stable slot registration removes repeated RPCs; metadata refresh
still performs work inside compute and is not concurrent use of the DSP RPC.
At load time it prepares metadata for all 608 physical slots instead of only
the 128 preload slots. This advances about 37.617 MiB of private DSP metadata
allocation to load time; it does not enlarge the expert weight pool. Shared
handle-table capacity also increases by about 32 KiB.

Two separate instrumented traces compare early/stable off versus on in the
same optimized runtime. Both read 2,304 experts using 2,304 Direct I/O calls
and request 4,656,726,016 bytes. Exposed read wait is 61.271 -> 0.038 ms and
registration time is 38.199 -> 1.920 ms. The first future read moves from
84.999 ms after first compute to 6.407 ms before it. The enabled read window
is 1,250.560 ms, with 3.724 GB/s effective requested-byte throughput. Profiling
is excluded from headline TPS. Near-continuous reader activity includes
pread waits and does not prove storage hardware saturation. Near-zero
exposed expert wait also means faster I/O alone need not reduce total prefill
time; compute, copies and memory contention remain important.

Gemma-specific early-read, pair-read and stable-slot extensions were tested
in 24 exploratory runs and removed. Early-only regressed median prefill by
about 5.46%; no tested extension established a useful improvement. The old
apparent early-read gain was invalid: a chunk guard made the hook a no-op.
repro/gemma-port-ablation.json records the correction and complete matrix;
candidate-latest-artifacts.json identifies that experimental runtime only.
Final optimized Gemma model code is unchanged from bd556b960.

Memory is sampled from before child exec through exit, including loading,
at a requested 20 ms period (actual mean intervals are in results.json).
The metric is same-sample RssAnon plus fdinfo allocation sizes deduplicated
by mount/inode. All selected peak samples have complete identities and no
fd read errors. Other scans may race fd closing. This is not exact total
physical device memory: private DSP heap, driver memory and VTCM may be
absent; allocation sizes need not be resident bytes. Do not add RSS/VmHWM or
mapped arena sizes to it. Brief peaks between samples can be missed.

Validation retained under validation/:
  Android NDK r30 core/app and Hexagon v81 builds pass.
  21 attention replay cases and 288 ARM convolution cases are byte-exact.
  15 DSP stable-slot cases cover generation changes and M1/M129 first use.
  Residual direct-add fixture passes all 10 cases.
  Stable-slot, paired-read and early-prefill ownership host sanitizer checks
  pass, including negative geometry and cancellation/EOF paths.
  Memory monitor: 10 normal and 10 ASan/UBSan cases pass; non-Linux guard
  compiles in a GNU syntax check. This is not an actual Windows/Tizen build.
  The broad host script initially stopped at a missing weak cache-invalidate
  stand-in; after that fix the remaining sections pass in the resumed log.
  This is not a single final uninterrupted full-suite run. A pre-existing
  signed-shift issue in the swiglu determinism fixture prevents claiming a
  clean sanitizer run of every graph test. Remote CI was not run here.

To reproduce, build both source revisions with the same toolchains/options
in repro/build-options.json. Regenerate the ARM RPC stub after IDL changes
using nntrainer/tensor/htp_backend/generate_stub.sh. Use matching NDK r30
core, app and libc++ libraries. Build the v81 DSP skeleton with test/htp's
build script and the recorded Hexagon/HexKL versions. Deploy the seven files
listed in build-options.json under separate baseline-latest/ and optimized/
device directories; repro/*-artifacts.json records the measured hashes.
Prepare model/, model2048/ and gemma/ directories using repro/models/ configs
and the corresponding native model/tokenizer payloads.

Compile validation/memory_monitor.cpp with the NDK aarch64 Android28 compiler,
-O2 -std=c++17 -static-libstdc++, and deploy it as fixture-support/memory_monitor.
Set ANDROID_SERIAL and NNTR_BENCH_DEVICE_ROOT to your device and parent
directory. From this directory run:
  python3 bench.py baseline-latest new-baseline-1 --model model --memory
  python3 bench.py optimized new-optimized-1 --model model --memory \
    --env NNTR_HTP_PREDICT=2 --env NNTR_MOE_PREFETCH_RESIDUAL=0.5 \
    --env NNTR_HTP_TL2_STABLE_SLOTS=1 --env NNTR_LFM2_PREFILL_EARLY=1 \
    --env NNTR_LFM2_DECODE_PAIR_READ=1
Use --model model2048 for 2048 rows. Use --model gemma without LFM flags for
Gemma. Alternate variants as above and retain all repetitions. bench.py
rejects changed C/preload/cache settings, output mismatches and profiling
for performance comparisons. It waits for known benchmark processes to
finish; device owners must also coordinate any unrelated workload.

Host checks run through test/htp/host/run_host_checks.sh. The additional
early_prefill_check.py accepts --help for replay options; decompress the
two repro/*.routes.gz files to a temporary directory first. The portable
repro/reproduce-stable-slot-checks.sh documents required SDK/NDK variables
and builds host/device stable-slot fixtures. repro/build_repeat_check.py
builds the test-only two-run CLI from an existing app build; it does not
modify production sources. validation/manifest.json records monitor sources
and executable identity; no executable binaries are committed.
