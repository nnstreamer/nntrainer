LFM2 cached group4 Qprep gather removal — NOT ADOPTED

The normal comparison did not establish a consistent improvement. The 2048
prefill median was 843.840 -> 829.150 TPS (-1.741%), with 1/3 paired wins.
Control values were 819.200, 983.197, 843.840 and candidate values were 829.150,
836.601, 807.890. The high control value remains in the evidence; no cause is
assigned to it. These observations do not establish a general regression or a
causal timing bound. The decision is to retain the accepted implementation.
This decision supersedes pending device-evaluation text in the unchanged
historical author/build reports archived below.

The 1024 prefill medians were 827.809 -> 833.198 TPS (+0.651%, 2/3 paired wins).
The new path was not selected at 1024, so this is not attributed to gather
removal. Decode medians were 17.5487 -> 17.6796 TPS at 1024 and 16.3390 ->
16.8332 TPS at 2048; these are likewise not attributed to a 2048-prefill helper.
All twelve ordinary runs and both diagnostics completed with exactly the same
65 generated IDs as their respective controls. No additional device run,
compilation, or correctness test was performed during this curation.

Scope and exact frozen candidate
Production patch: a07332c9043b6760a42392cae06a237319e6a1e383829edc0bee5d838b435df8
Corrected tests: f962d48fabdb73ce2702333f67907b8c5f5ddd86e17c01c38d37048ea66db23f
Diagnostic patch: af263c745b23b9215a5707f9bc60c008ef2c168ea0880d9b201adddef4871a39
Baseline source: 742735f3c61803fac51189012ba4caf588a3bd50

The candidate reads the four consecutive hd64 head rows directly from an
already completed cached Q slice, removing the intermediate Q -> Qx gather.
It retains the original quant_row_params_one and pack_group_rows arithmetic,
row order, initialization, padding and synchronous worker ownership. Selection
requires the cached-Q capability, dynamic Q8, n_q/cache_to=2048, cache_from=0,
32 Q / 8 KV heads, hd64, G4/Br128/Bc256, stride2048, complete chunks and valid
non-overlapping ranges. Direct prefill, decode, 1024, fixed-Q8/Gemma and other
unsupported conditions keep their previous path.

No activation buffer, persistent cache, reader, preload or staging capacity is
added. C16, PAGE_CACHE=0, cache growth0, preload128, existing six readers and
native TL2/QS4 affine model configuration are identical in both normal arms.
The archived model-conversion record is prior provenance (ceebe2a4 weight SHA),
not a new model-file rehash for each run; source checkpoint revalidation is
outside this archive.
The private DSP IO record grows by four call-local bytes (24 -> 28); no ARM or
IDL change. Qx remains allocated at 1,179,648 B, including the original deferred
QK readout workspace. The RPC's completed DMA and existing synchronous lifetime
protect the cached input; raw QK tiles overwrite their Qx area before dequant
reads it. Normal stack/frame and ownership evidence is in the author/reviewer
reports. Diagnostic-only global metadata is another four bytes; its logging
and larger helper frame are absent from both normal arms.

Device dispatch evidence, separate from performance
The trace-enabled 2048 run has PID16744 and 160 unique contiguous helper-success
records after worker drain: 80 chunks of 1152 groups and 80 of 896, alternating
in order. This is 10 attention layers * 8 KV heads * 2 chunks. The corresponding
1024 run, PID16926, has zero helper tags. Both live captures began before their
single supervised child and ended after successful child completion; the trace
binary's identity and exact IDs are recorded. Empty 1024 output alone is not
proof of a disabled branch without those completion/binary/scope records.

160 helper calls cover 167,772,160 bytes (160 MiB) of logical gather payload.
This is not measured DDR traffic or a speed estimate. Direct quantizer reads
now use cached Q instead of VTCM Qx; locality and dispatch costs can offset the
removed copy. The historical whole-Qprep time is not a gather-only estimate.
The optional existing-record stage summary reports qprep44.662 ms, qk84.217 ms,
dequant36.688 ms, pquant72.703 ms and inner total358.977 ms over ten full2048
layers. It is a single trace-enabled candidate run with logging inside Qprep,
not a matched stage comparison. Overlapping stage durations are not added into
a saving budget. Neither diagnostic run enters the ordinary TPS comparison.

Controlled DSP build and restoration
Earlier full DSP builds were non-deterministic, so this experiment did not
rebuild all TUs independently for each arm. One baseline set of 63 objects was
compiled and frozen. Only four dependent TUs were recompiled for the candidate:
  hvx/hvx_quant_u8.c
  hmx/hexkl_attn_q.c
  hvx/hvx_attn_decode_q.c
  test/htp/nntr_hvx_attn_q.c
The latter three consume/produce the changed private IO layout; no old-layout
consumer is left mixed into the candidate. Fifty-nine common objects are
identical. Both arms use the same source order, external inputs and -no-threads
linker setting, with exact repeated links. The supplemental inventory covers
all seven shared linker libraries, including the previously omitted libgcc.so;
same-path relinks preserved both original artifact hashes. The diagnostic uses
one quant TU VFS overlay; macro OFF is object-identical to normal, all other
objects are frozen, imports are checked and its repeated link is identical.
No new claim of a deterministic whole build or reproduction of original3b is
made. There is no independently rebuilt unrelated-TU confound between these arms.

DSP SHA256 values:
  control    f3fd2fd2f9c457d21ff13f0d1a9e6263a2a93a60fc9e0a6c75eb172d15387b06
  candidate  7fa6e9beec4b5972f9e1681cca9c676e965e123c4171bb4bd595f3e73849da54
  diagnostic ca3c46323c7ccf7d5c263fd530e4aefbab00ba089be7cc22941fbeec3337c1ed
  accepted   3baddb92d14d1d23fe9b6c6a7cacfe3a6cd4708001c110bb8b15e1b0b7276f7c
All six host-side runtime files are identical across both arms and diagnostics,
including core072b and accepted ARM-router appc34. Seven-file manifests accompany
every runtime. The controlled and trace DSPs were deployed to separate variants;
original accepted local binaries were never overwritten. After diagnostics,
five source states (the new header absent) and all seven original accepted local
and freshly rehashed phone files were verified. The accepted phone variant is
lfm-router-arm-1. Frozen original3b, not a fresh full rebuild, remains accepted.

Correctness evidence and the corrected sanitizer harness
The final author and independent checks cover 576 actual SDK libnative/Qprep
parity cases, 31 guards and five killed mutants; 15 actual RPC-body checks and
three mutants; and 460 overlap/readout/dequant checks with three mutants. Four
TUs compile for v81. Existing eleven quantizer function bodies are byte-identical,
and the decode object is byte-identical. Five production files pass the repo's
clang-format14 changed-lines check. Source/patch/fixture hashes are retained.
RPC DMA/reserve/append/kernel and overlap HMX/DMA/softmax/PV are explicit test
stand-ins. SDK libnative is a prebuilt uninstrumented library. These checks do
not independently prove hardware coherence or full model behavior; root's
separate device dispatch and exact-output records cover the measured runs.

The first tests-v1-ffb26 fixture incorrectly mapped HVX_UVector to a host type
with eight-byte alignment while deliberately testing a four-byte offset. The
author run emitted eight recoverable UBSan alignment errors and exited0; that
sanitizer-PASS claim is invalid and superseded. Independent halt-on-error caught
the failure. Both original raw logs, old patch/hash, and a failed intermediate
may_alias compiler attempt are preserved. Final tests use an aligned(4) shim
matching the SDK type, static size/alignment assertions, -fno-sanitize-recover
and explicit ASan/UBSan halt settings; no suppression. Author and independent
strict runs have zero runtime errors. Production did not change for this fix.

Whole-execution memory scope
All fourteen reports sample the supervised child from load through exit at a
requested 20 ms period, retain every sample and report actual scan/interval
statistics. Peak samples have complete readable fd identity, no fdinfo errors,
unknown identity, conflicting inode size or overflow. Normal maximum sampled
peaks are 2,714,368 -> 2,714,428 KiB at 1024 and 3,018,172 -> 3,019,064 KiB at
2048. These are RssAnon plus inode-deduplicated fdinfo allocated sizes; process
RSS is not added again. They are not total device/physical/DSP memory, do not
include unreported driver/VTCM allocation and can miss between-sample spikes.
The extra 60/892 KiB observed peak difference is not assigned a mechanism.

Portable archive and readback
Run with Python3 only, from any directory:
  python3 PATH/verify_archive.py PATH
This verifies every stored and uncompressed SHA256, exact file set and relative
paths, then independently rereads all raw TPS/ID/memory/live-log records and
recomputes summary.json. The readback does not call the author's trace parser.
It validates the parser SHA and its stored root results, fixed env/config,
source/runtime/build bindings, 63/4/59 objects and the restored seven binaries.
No SDK, phone, repository checkout or model is needed for this verification.

Original author/reviewer scripts and commands remain unchanged for historical
reproduction; their absolute paths are provenance, not archive dependencies.
To rerun kernel tests, use a complete checkout at the recorded base plus the
archived production/tests patches and the matching Hexagon SDK; uncompress the
selected source snapshots as needed. This archive itself does not rerun them.
The curator takes explicit --repo/--author/--review/--fixed/--root-evidence/
--output inputs and reads evidence only. The frozen manifests bind the original
input bundles. Raw logs, source, assembly and full memory JSON use deterministic
gzip; binaries, object bodies, SDK, models and unrelated checkout snapshots are
excluded. Their identities and external-input hashes remain in manifests.
