LFM native TL2 row-normalization experiment — not adopted

The normal model comparison rejected this candidate. With the exact same seven
runtime binaries and NNTR_LFM2_MOE_ROW_NORM=0/1, prefill medians decreased at both
prompt sizes. The earlier CPU-only prescreen improvement did not transfer to the
model. No specific scheduling, thermal, bandwidth, or allocation mechanism is
established by these observations. The original source and all seven original
runtime hashes were restored; no extra model runs followed the decision.

Normal model results (three OFF and three ON runs per prompt, AB/BA/AB order)
                       OFF median     ON median      change      ON wins
  1024 prefill TPS        843.493        828.479        -1.78%       0/3
  2048 prefill TPS        902.600        863.043        -4.38%       1/3
  1024 decode TPS          17.6600        17.6698       +0.06%       2/3
  2048 decode TPS          16.3016        16.5589       +1.58%       3/3
All 12 runs produced the same 65 dumped IDs per prompt as the reference: one
prefill token plus 64 decode iterations. There is no decode-kernel change or
claim of decode speedup. Three pairs provide a bounded experimental result,
not a universal performance prediction. summary.json retains every paired delta.

Runtime identity and restoration
OFF and ON both use app 9c1100d08db52276efbb48ceed652bb7831f87090e7b4e03ae03964aeb9857c8
and DSP 3baddb92d14d1d23fe9b6c6a7cacfe3a6cd4708001c110bb8b15e1b0b7276f7c.
All seven hashes are recorded in artifacts/ and summary.json. This comparison
has no different-DSP-build confound. The separate diagnostic changes only the
app binary; it was removed and normal app identity checked before normal runs.
Restoration returns the original app c9efd4428d5644389f0d7b76f5d5c0e6ad274a5309d1475fc6fd9e8da8656ba4
and source 4ed12667f11b5886648c18faf367ce1a1022b9baf81274f76045d028dd9dd5ec.
The restored runtime uses the original frozen DSP; source restoration is not
being substituted for binary identity. Root's restoration record is preserved.

Fixed conditions and sampled memory
C16, PAGE_CACHE=0, preload128, grow0, eight CPU threads, unchanged reader count,
TL2CX_WH experts, affine FC, and identical model/config/quantization are retained.
Raw startup logs confirm cache_slots=608, prefetch direct slots32, readers2,
and demand readers4. No activation/ION/expert-cache/reader capacity was added.
The existing task slot allocates a 48-byte libc++ callback object (40-byte
capture); replacement may briefly overlap the old object. New helper/dispatch/
callback frames and 16-byte merged object-BSS growth are documented separately.
This is not a claim of zero additional allocation.

Maximum sampled complete-run anon+deduplicated-fdinfo values (KiB):
  1024 OFF 2734000; ON 2733468
  2048 OFF 3034916; ON 3068796 (+33880)
The higher 2048 observed peak is reported without attributing its mechanism.
The supervisor samples the child's full lifetime, including load, every requested
20 ms. It sums RssAnon and fdinfo size after inode deduplication at the same sample;
this is neither total system memory nor exact physical DSP memory. FD size being
DMA is not independently verified. Peak records have complete readable-size-FD
deduplication and zero read/identity/conflict/field/overflow errors. Terminal or
incomplete nonpeak samples and actual sample intervals remain in raw memory JSON.
Between-sample peaks can be missed. Process VmHWM and application generation-only
peak are separate data, not added to this composite. No duplicate RSS+DMA sum is
used. The per-run sampling statistics are in validation/archive-checks.json.

Candidate and correctness scope
Only eligible AArch64 native LFM prefill M1024/M2048, K2048, E64/I1536 with the
exact option value 1 submits 64-row jobs to the existing ThreadManager. Each row
calls the original RMS and gamma multiplication functions/arguments. Other shapes,
dtypes, non-HTP, overlapping outputs, decode, Gemma, and options retain normRows.
The original normRows machine code is exact; the caller frame is unchanged.
Gemma was not benchmarked for this rejected candidate.

Author checks: actual helper/caller plus extracted ARM math through SIMDe, 89
cases in each env=1/0/unset/true with ASan/UBSan; eight runtime mutants rejected;
whole Android AArch64, Android x86, and Linux GCC translation units compiled.
Host Tensor/ops/scheduler are stand-ins; native checks use real deployed core,
Tensor, Context and ThreadManager but do not execute DSP or load expert weights.
No independent-agent source review was performed for this candidate. Root's
source review, native/device checks and decision are recorded as root evidence.
The archive validator independently checks existing records and hashes, not a
second execution of the author fixture. No new tests or phone runs were made
during this curation.

The first native fixture failed before entering the helper: a Weight initializer
of NONE is invalid even with deferred allocation. root-native-1-8.log and the
initial fixture/command/binary hash are preserved, not counted as a pass.
The corrected test uses ZEROS, alloc_now=false, and asserts !isAllocated().
Production was unchanged. Corrected binary e936c5201916b2f43f0bc86ece3b45b71563117d1ad2ef941f8bd3ac79f59943
passed 30 cases each for option0/1 and threads1/8 (four runs; six numeric exact
cases each). Native build-link core hash and actual deployment commands are
preserved distinctly; no claim that every linking/runtime file is byte identical
is inferred beyond the recorded normal runtime manifests.

Actual dispatch and CPU prescreen
Separate diagnostic app runs report 38 completed row-norm helpers each at 1024
and 2048, with 16/32 jobs per call. The strict parser requires contiguous q1..38,
exact geometry and completion fields. Markers reside in the raw app .log files;
logcat contains none. Diagnostic TPS is excluded from the normal comparison.

The CPU-only fixed-input prescreen measured medians 426.146 -> 200.729 us at1024
and 1176.354 -> 545.312 us at2048. Its 31 paired samples per shape/option and
arithmetic checks are preserved. It excludes expert I/O and DSP and does not
predict model speedup. The historical author README predates the normal verdict;
its arithmetic extrapolation over 38 layers is not adopted as a model estimate.
This README and summary.json state the final rejection decision.

Contents and reproduction
manifest.json hashes every stored payload and its exact decompressed bytes.
raw/ contains all 12 normal logs, IDs, metadata and memory reports. diagnostic/
contains separate instrumented records, source patch, parser and author checks.
validation/ contains frozen source/tests, initial and corrected native fixtures,
commands, focused codegen, reports and root evidence. Executables and enormous
whole-TU disassembly are omitted; their exact hashes and useful text are retained.
Gzip has deterministic mtime=0. Original reports are retained without rewriting
historical status; the archive binds them to final raw evidence.

Verify without compilation or device access:
  python3 reproduction/verify_archive.py ARCHIVE
Extract exact sources and record portable commands without executing tests:
  python3 reproduction/reproduce_host.py --archive ARCHIVE --repo CHECKOUT \
    --simde-include EXTERNAL_SIMDE_0.8.2 --output NEW_DIR --prepare-only
Remove --prepare-only to run host checks; --mutations also runs eight mutants.
The wrapper checks the recorded neon_impl.cpp dependency before extracting.
The corrected native build runner and its explicit compiler/core-path arguments
are archived under validation/author/test/htp/host; device execution is external.
No model files, full runtimes, toolchain, or vendored SIMDe are bundled.
