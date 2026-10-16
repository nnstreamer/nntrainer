LFM conv gate Tile16/Tile8 experiment — NOT ADOPTED

The Tile8 production change regressed normal full-model prefill and was removed.
All proposed source patches and tests below are historical evidence only. The
retained conv chunk v2 source at ccea0d476daf43ddd5dbe201178755e6ff978dfd and its
app were rebuilt and restored; all seven restored runtime hashes match the
previous lfm-conv-chunk-isolated runtime. No new model run was performed simply
to retest that byte-identical restoration. The restoration record and rebuild
log are retained under validation/tile8/author/.

The previous Tile8 author README and independent review were written before the
normal-model decision. Their correctness/codegen statements remain evidence;
any “performance pending” status there is superseded by this README and summary.
No correctness failure, memory-capacity growth, or failure to enable the option
was used to excuse the negative normal results. The cause of the slowdown has
not been isolated. Cache contention, scheduling and concurrent I/O effects are
possibilities, not measured explanations. No further tile size was tested.

Normal S26 Ultra measurements (R3CY70LV96T)
Same seven binary hashes for OFF and ON; only NNTR_LFM2_CONV_GATE_TILE differs
between equivalent runtime directories. The code is based on retained chunk v2.
For each prompt size the order is A1 B1 B2 A2 A3 B3, three pairs, with 64 generated
tokens. C16, PAGE_CACHE=0, cache grow=0, preload 128, eight CPU threads, direct I/O,
six prefill readers, two speculative decode readers and four demand participants
(including the caller), predict2/residual0.5, stable slots, early prefill and the
retained decode pair/demand/router handoff paths are unchanged. Startup logs show
“direct slots=32 bytes=64618496 readers=2 pair_read=1”. These are normal runs,
without layer/HTP/I/O timers; memory monitoring is identical in both conditions.

                      OFF median       Tile8 ON median     prefill pair wins
1024 prefill TPS         850.498             799.375                1/3
2048 prefill TPS         904.993             827.475                0/3
1024 decode TPS          17.7089             17.6649
2048 decode TPS          16.5846             16.6797

Prefill changes are -6.01095% and -8.56559%. At 2048 all three candidate runs are
slower. Small decode differences do not establish a decode effect; this helper
is outside the single-row decode path. All 12 runs match all 65 dumped reference
IDs (one prefill token plus 64 generation tokens). Every run's original stdout,
JSON including exact environment/command/thermal observations, 65 IDs and full
memory-monitor report is preserved in raw/. summary.json recomputes medians
and pair counts from these records; no runs were selected away.

Largest sampled whole-process peaks in KiB, OFF -> ON:
1024: 2,733,560 -> 2,733,244
2048: 3,036,944 -> 3,020,636
No memory-saving claim follows from these varying samples. This metric is
RssAnon plus same-sample fdinfo allocated sizes deduplicated by mount/inode,
from before child exec through exit including loading, requested every 20 ms.
Readable peak fd identities/sizes were complete in all 12 reports. It is not
plain RSS or total physical device memory. Allocated fd sizes may be nonresident;
unreported DSP/driver/VTCM and between-sample transients are excluded. The
separate app CausalLM::run-only metric excludes loading. Neither ru_maxrss nor
VmRSS/arena sizes should be added to the composite. Original sampling intervals,
all samples and limitations remain in each compressed memory JSON.

What the rejected candidate changed
The AArch64 native-LFM FP32 full 1024 and existing 2048 chunks of 1920+128 rows qualified.
Tile8 combined gate-pre, the original multiply/two-FMA convolution, and post gate
inside independent 8-row jobs. Halos were recomputed from the current ION view;
only the first chunk's final two gated rows crossed the next RPC. Those rows
and the final two state rows used the existing gated allocation. Jobs could run
in arbitrary order without reading each other's intermediate gated values.
The reference gate-pre compiles to C*A despite scalar source A*C; the new helper
preserved that actual operand order and exact rounding/NaN behavior. In the
three-way micro C is at byte offset 0x4000 (2*2048*sizeof(float)); 0x2000 is one
single-channel row's bytes. No claimed C-offset proof relies on 0x2000.

Tile8 recomputes 254/508 halo rows (+24.8% gate products) for 1024/2048, while
avoiding most gated tensor writes and reads. Existing activation and staging
capacities are unchanged. Main app frame of 1760 B and chunk frame of 464 B are unchanged,
but the new outlined helper (128 B) and parallel_for (96 B) add 224 B of nested caller
stack. Arithmetic callback is 448 bytes of code with zero stack references.
The existing ThreadManager task slot requests a 64 B function object versus 48 B
for the old full post helper or 56 B for its range helper. This small +16/+8 B
request and temporary old/new callback overlap are disclosed; “zero new memory”
is not claimed. No new expert/activation cache or data buffer was introduced.
Gemma/unsupported guards and seven old arithmetic callbacks were verified,
without a Gemma device benchmark or unchanged-performance claim.

Why CPU-only micro results did not justify adoption
First Tile16 native helper checks and a two-way cached-ION micro passed. Then
one Tile8 variant was compared with the old and frozen Tile16 helpers in a
single process, with identical arrays and eight real ThreadManager threads.
All 12 three-way exact-output/state cases passed, including sNaNs and poisoned
first-chunk ION views. 31 measured iterations per path used six cyclic orders.
Median microseconds for old / Tile16 / Tile8 were:
1024 warm          863.490 / 768.594 / 619.010
1024 host sweep   1020.312 / 925.157 / 790.729
2048 warm         2238.646 / 2144.271 / 1770.833
2048 host sweep   2502.552 / 2451.719 / 2066.615

These are CPU-region results, not model TPS. They include sampled-copy checks,
CPU gate/conv/post, completion barriers and state copy. They exclude projection
DSP work, accelerator coherency and concurrent expert I/O. ION was CPU-filled;
host sweep touched an existing projection array, not a guaranteed cache flush.
The preceding path can change cache residency; per-iteration values were not
saved, so no paired-difference interval can be recovered. The small isolated
absolute delta did not predict the observed full-model behavior. Both raw micro
logs, exact commands, OpenMP/runtime/binary hashes and independent arithmetic
readback are preserved separately from normal performance.

Correctness and build evidence
Tile16 and Tile8 each passed author and independent actual-source ASan/UBSan 112-case
ON/OFF checks and eight deliberately broken-source runtime rejections. Final
Tile8 passed 112 root integrated checks, 112 S26 real-NEON checks and 12
cached-ION cases.
The unchanged committed chunk checker still passed 82 cases with option unset/OFF.
One/eight-worker reverse/shuffled scheduling, finite and special FP32 inputs,
exact output/state bytes, alias/fallback guards, poisoned unused gated rows,
ION overwrite, failed first/second RPC/sync and retry, and same capacities were
checked. Native helpers used mock RPC, so these tests are not DSP/coherency
proof. Full Android AArch64/x86 and Linux GCC translation units compiled in the
independent review. Host ThreadManager/fake RPC/SIMDe limits remain explicit.
No full Windows/Tizen/Yocto CI claim is made for this rejected experiment.

Contents and reproducibility
artifacts/ keeps original parent/OFF/ON/restored runtime manifests and source
patches, including all seven binary SHA256 values. Frozen executables and model
weights are deliberately excluded. validation/ holds source, patches, canonical
tests (removed from production), native results, build/host/mutation commands
and logs, small assembly slices and author/independent reviews. Full generated
mutant sources and large disassembly dumps are omitted: the frozen mutation
runner regenerates them. A preliminary Tile16 native snapshot is distinctly
labelled and is not used as final Tile8 proof. Source patch replay was checked
against ccea; root source cleanliness at archive creation was not required.

Verify every archived payload and its original decompressed bytes:
  python3 ARCHIVE/reproduction/verify_archive.py ARCHIVE
Reproduce Tile8 host 112-case ON/OFF in a new output directory (optional 8 mutants):
  python3 ARCHIVE/reproduction/reproduce_host.py --archive ARCHIVE \
    --repo CHECKOUT_AT_ccea0d476 --simde-include EXTERNAL_SIMDE_0.8.2 \
    --output NEW_TEMP_DIRECTORY --mutations
This script validates the exact core extraction hash before compiling. It
never modifies the checkout or archive. Android compiler/build command JSONs
are preserved but refer to the original SDK/build environment; install/configure
those external dependencies when reproducing native objects. Runtime comparison
commands and exact model configs are retained; replaying on a device requires
rebuilding/deploying the recorded candidate and explicitly selecting its device
and runtime locations. No device action is performed by the curator or host
reproduction script. The historical original run/summarizer scripts retain
their recorded workspace paths for provenance.

manifest.json hashes every stored payload and its decompressed original.
The compressed memory reports are the complete original bytes, not summaries.
The top-level manifest's own hash is reported separately to the parent reviewer.
This archive adds evidence only; it does not request re-enabling this candidate.
