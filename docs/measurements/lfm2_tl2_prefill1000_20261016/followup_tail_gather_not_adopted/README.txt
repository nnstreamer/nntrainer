Native LFM TL2 prefill tail-gather experiment — NOT ADOPTED

The candidate was not retained because the normal model runs did not show a
consistent improvement. Rebuilding the whole DSP also changed unrelated machine
code despite unchanged source; the archive therefore makes no causal speedup or
regression claim for tail gather. The original control source and all seven
original runtime binary hashes were restored by selecting the saved original DSP
artifact. Rebuilding the original source did NOT reproduce that DSP byte-for-byte.
There were no further same-handle repeats or profile runs for this candidate.

The 12 normal runs used two input lengths, 1024 and 2048, three control/candidate
pairs each, in AB/BA/AB order. Every run produced the same 65 dumped IDs for its
prompt: one prefill-selected token plus 64 decode steps. Median TPS is descriptive:

  input    prefill control -> candidate    faster pairs    decode control -> candidate
  1024     832.520 -> 846.981 (+1.74%)      2 of 3          17.5391 -> 17.6503
  2048     820.184 -> 830.832 (+1.30%)      1 of 3          16.6363 -> 16.6018

The first two 2048 prefill pairs changed -1.18% and -2.23%; only the third improved.
The two median improvements alone do not establish a reproducible benefit.
Max sampled complete-run peaks were 2,734,164 -> 2,733,872 KiB for 1024 and
3,036,576 -> 3,036,740 KiB for 2048. All raw observations, thermal metadata and
pairwise differences are retained; summary.json is independently recomputed.

Scope and limits

Only native TL2 LFM prefill with K2048/I1536/N2048, SiLU, M>1 and 1..64 compacted
active experts takes the candidate gather worker. The last 16-row gather unit is
rounded to a 4-row endpoint. Rows that the existing epilogue ignores remain stale
in existing scratch. Quantization, HMX M64 work, DMA issue count, accumulation and
scatter order stay unchanged. The generic worker's 124-byte object is identical,
but shared entry/metadata code changes, so this is not a claim of whole Gemma
machine-code or performance identity. No Gemma phone benchmark was repeated.

C16, PAGE_CACHE=0, preload128, reader settings, weight quantization and all buffer
capacities were fixed. On the observed 1024 route, 3,721 row4 groups/14,884 gather
rows are omitted: 30,482,432 logical stored bytes (29.07 MiB) and 121,929,728
logical vector-load bytes (116.28 MiB). These are HVX instruction byte counts,
not measured DDR traffic, exposed wall time or a TPS estimate. Previous exposed
gather wait was small; parallel worker time must not be summed as serial latency.

Author and independent validation each passed 198 actual-HVX gather cases,
16 guards, 512 actual i24 readout/padding-poison cases and 21 whole-kernel cases,
with ASan/UBSan, plus 7 intended mutants. Whole-kernel tests use actual gather and
pthread workers but immediate DMA/scalar HMX/math stand-ins. Only the existing
scalar stand-in TU suppresses shift-base UBSan for its deterministic-exp helper.
Device correctness is the 12 normal runs' exact generated IDs, not host emulation.
v81 compile/codegen retains the 17,152-byte main frame; the new worker has frame0.
The archive preserves original author/independent reports as written before the
model verdict; those reports' no-device/no-speed claims refer to their own scope.

DSP identity investigation

Control DSP is 3baddb92d14d1d23fe9b6c6a7cacfe3a6cd4708001c110bb8b15e1b0b7276f7c;
candidate is 79ff958a00b6717b655310a344d31156f04d0ae7c8853a0cce3f83b6e06dc3e3.
The restored-source full rebuild was 72083514bdf2a07a3be1be4c6c9bbd627e513e3a10950e7597a8709a65ed04c8.
Identical source/argv/cwd/output compiler runs produced different probe_lane and
softmax object instructions. The frozen candidate additionally differs in other
attention-M1 and generic router functions outside the tail source patch. This
confounds attribution of normal timing differences. See validation/build-identity
for exact hashes, repeated compile commands, focused instructions and limitations.
Three repeated baseline MoE/ring TU compiles were each byte-identical; this bounded
check does not prove all future builds deterministic. Future DSP comparisons should
freeze baseline objects and recompile only the changed translation units.

A later section-dump audit accidentally rewrote two ELF files via llvm-objcopy
without an explicit output. Exact original copies were recovered and verified;
no model ran the rewritten binaries. This side effect happened after the rebuild
mismatch had been discovered. Its root record is retained separately. The selected
restored runtime uses original control3badd, not the different rebuild720835.

Memory interpretation

Each run uses the same 20 ms child-process monitor from before exec through exit.
Reported peak is max_t(RssAnon + deduplicated fdinfo size allocations), including
model loading. All peak samples have complete readable-fd identities and zero
recorded error fields. Actual sample intervals and unavailable terminal samples
are exposed in validation/archive-checks.json and the raw monitors. The monitor
does not verify that every size field is DMA memory and misses unreported private
DSP/driver/VTCM memory and between-sample transients. It is not total physical
device memory. Do not add VmRSS/VmHWM or separately logged HTP arenas to this peak;
they can overlap. The app's own peak excludes model loading.

Archive layout and reproduction

raw/ keeps all 12 exact run metadata, IDs, gzip logs and gzip memory reports.
artifacts/ binds seven runtime hashes and exact source deltas for each variant.
validation/ keeps original patches, canonical source/fixtures, host/readout/mutant
results, codegen commands/stack summaries, independent review and restoration.
Large full disassembly and executables are not copied. Their original hashes and
focused exact text remain; all archived payloads have both stored and decompressed
hashes in manifest.json. Gzip timestamps are zero for reproducibility.

Verify an archive:
  python3 reproduction/verify_archive.py /absolute/path/to/archive

Recreate actual host checks from an nntrainer checkout with the recorded dependency
hashes (the recorded base is f7b16e48; relevant source equals author ccea0d47):
  python3 reproduction/reproduce_host.py --archive /absolute/path/to/archive \
    --repo /absolute/path/to/nntrainer --tools-root /path/to/HEXAGON_Tools/19.0.04 \
    --output /new/temporary/directory --mutations

The proprietary SDK/libnative is an external prerequisite, not bundled. The helper
checks dependencies, extracts archived sources to the new directory and never edits
the checkout. --prepare-only validates extraction/commands without claiming execution.
Original root benchmark/matrix/monitor scripts and exact model configs are included
for evidence; this curator and verifier themselves never connect to a device.

Re-run curation from the recorded original evidence tree:
  python3 curate_tail_gather.py --repo /absolute/path/to/nntrainer \
    --author /tmp/nntr-lfm2-m-tail-gather-20261016 \
    --independent /tmp/nntr-lfm2-m-tail-gather-review-20261017 \
    --scratch /tmp/nntr-lfm2-tl2-prefill1000-20261016 \
    --restoration /tmp/nntr-lfm2-m-tail-gather-20261016/root-normal-restored.json \
    --output /new/followup_tail_gather_not_adopted

--check-only validates every input and recomputes the results without writing an
archive. Final creation refuses an existing destination and requires the exact
restoration record. Live root cleanliness is deliberately not used as evidence;
the recorded commit, frozen snapshots and runtime manifests bind this experiment.
