LFM native TL2 M1 completed-descriptor ring reset — not adopted

The controlled normal comparison did not establish a decode TPS benefit.
1024 median is essentially unchanged; 2048 is lower in all three pairs. The
candidate is therefore not retained despite exact output and confirmed actual
fast-path use. Prefill measurements are observations, not an attributed benefit
of this decode-only reset branch. No additional model runs or host fixtures
were executed while curating this evidence.

Normal comparison: three control and three candidate runs at each prompt,
AB/BA/AB order, same six ARM binaries and fixed-object DSP pair1ef/f3:
                          control median  candidate median  change    wins
  1024 decode TPS             17.6163        17.6066         -0.055%    2/3
  2048 decode TPS             16.3474        16.2066         -0.861%    0/3
  1024 prefill TPS           827.141        827.141           0.000%    1/3
  2048 prefill TPS           814.314        852.268          +4.661%    2/3
The normal arrays and paired changes are retained in summary.json. Large
prefill pair variation is visible; no causal prefill gain is claimed. All 12
runs retained 65 exact dumped IDs (one prefill token plus 64 decode iterations),
clean exit, valid run metadata and complete sampled peak records.

Maximum full-run sampled RssAnon+deduplicated-fdinfo peak, KiB:
  1024 control 2714944 -> candidate 2714636 (-308)
  2048 control 3019188 -> candidate 3019336 (+148)
These are observations without a memory-change mechanism claim. The supervisor
samples the child's whole lifetime including load, requested every 20 ms. It is
not total physical device/DSP memory: fdinfo sizes may not all be verified DMA,
unreported driver/DSP/VTCM and between-sample spikes are excluded. Selected peaks
have complete readable-size-FD identity deduplication and zero recorded read,
identity, conflict, field and overflow errors. Actual intervals and nonpeak or
terminal incomplete samples remain in raw memory JSON. VmHWM or the application's
generation-only peak are not added again. Per-run interval audits are preserved.

What changed and what stayed fixed
Production 7803996b adds try_reset and one caller under the existing native LFM
M1 guard: TL2, K2048/I1536/N2048, expert table 64, 1..4 active one-row experts,
SiLU, no affine. A completed last descriptor with correct tail/type/size permits
resetting only existing software indices. The old 256*128-byte descriptor memset
remains for BSS, reset sentinel, incomplete/unknown tail and all other paths.
Each push still fills all eight hardware words, clears done/reserved bits,
cleans/invalidates, releases and submits in the same order. The 96-byte stride
padding is not hardware descriptor payload. Math, weights, quant/LUT, weighting
order, lane partition, output zeroing, cache, reader and buffer capacities stay
unchanged. There is no new production state/allocation/API knob.

C16, PAGE_CACHE0, grow0, preload 128, eight CPU threads, prediction 2/residual 0.5,
native TL2 expert quantization, affine FC and retained ARM-router option 1 are
common to both arms. Startup confirms 608 expert slots,32 direct slots/two readers,
and four demand readers. The accepted ARM-router app c34aa784 and core 072b5ca2
are identical in control, candidate and diagnostic. The prior rejected predictor
register patch is absent. Generic/Gemma callers retain the old reset; no new
Gemma phone benchmark was requested or performed for this trial.

Existing completion/error contract, not a new safety fix
The optimization relies on the same serialized ring owner and ordered linked
completion assumption as existing waits. An incomplete tail causes fallback;
the old raw drain still has a bounded polling loop without an error result.
The candidate does not repair timeout retirement or re-entry after a timed-out
DMA. Host timeout fixtures retire their simulated engine before cleanup and do
not claim to make the existing unsafe timeout fallback safe.

Author and independent source validation
Frozen tests 1e9b6b63: actual-ring source with delayed ordered hardware stand-ins,
175 lifecycle checks / 5957 fills / 50011639 polls, eight rejected ring mutants;
separate whole-M1 actual-source/actual-HVX-quantizer caller 31 cases and two rejected
caller mutants. ASan/UBSan pass. The caller retains a pre-existing shift-base
sanitizer exclusion only in its scalar HVX stand-in; production ring/MoE and
actual quantizer remain sanitized. These are separate ring-lifecycle and caller
fixtures, not full real DMA/HMX behavior. Existing MoE/TL2-prefill/conv integration
passes after explicit test-stub additions; the initial missing-stub link failure
and corrected export are preserved. v81 confirms descriptor 128 B / src8 / dst12,
helper 144 code bytes / zero frame, MoE 17152 B frame unchanged, original reset bytes
exact. Other push/drain differences include alignment NOPs and relocated calls;
normalized packets agree, not entire object identity. Historical author/reviewer
reports retain their original pre-device status; this README records the final
model decision.

Build nondeterminism: why this is a new controlled pair
Earlier tail-gather evidence found unrelated probe/softmax recompilation
variation, and identical frozen objects also produced different threaded links.
Those observations are not imported as a confound in this ring comparison.
Instead, the isolated build snapshots b17d1d2f, compiles all 63 TUs once with the
original build driver's exact cc1 options/source order, and freezes them. Only
ring.c and MoE.c are recompiled for the candidate; the other 61 objects, libraries,
flags and input order are identical. The common explicit linker -no-threads setting
makes repeated links stable. Only the two original IEEE TUs use HVX IEEE flags.
Commands, source/object/external hashes, original unstable link attempt and
independent relink evidence are preserved. The header addition is a declaration,
with no changed structure/inline content seen by the shared objects.

Controlled DSPs:
  control 1efcd9dcde2109634be700bb778ce1ba802ea6bc69c3dec1612030edcd4ce1e4
  candidate f3e68660aff009b218efe16ab51e2ea93ba31e66611d555d768f5a82355c08bf
Each frozen same-path repeat is byte-identical. Independent relinks at another
output path have identical allocated code/data/relocation sections; nonallocated
.comment records the output path, so cross-path whole-file identity is not
claimed. This pair is not the historical DSP 3baddb92. Root manifests explicitly
record separate frozen-object compilation, not a build from its current source
directory. The common app patch was separately reconstructed against root's
80a8b754 base. Independent replay verifies the actual three DSP source files
against frozen control/candidate/diagnostic snapshots.

Separate actual-hit diagnostics
Diagnostic 9aadb311, macro NNTR_LFM2_M1_RING_RESET_TRACE, adds 4 B atomic sequence and
one short FARF record per actual try_reset return. Macro-OFF ring.o is exactly
the normal candidate object; no atomic runtime import appears. Diagnostic DSP
c19c6427... uses the same remaining objects and six host binaries. The logging
cost is excluded from all normal performance comparisons.

Live PID-specific logs prove:
  prompt 1024 PID 7354: q1..4345, 4345 used1, 0 fallback/duplicates/gaps.
  prompt 2048 PID 7567: q1..4389, 4389 used1, 0 fallback/duplicates/gaps.
The independently inspected same-run driver closes 64 successful tokens. Its
67.89/68.58 calls/token averages imply exactly one integer total in the rounded
intervals (4344.64,4345.28)/(4388.80,4389.44): 4345/4389. The actual callchain binds
these graph counts to the native M1 reset caller; dense calls and M>1 load warmup
are excluded. No earlier-run average or 64*38 assumption was reused. The author
strict parser and separate independent raw parser agree; 65 IDs remain exact.

Thus this cohort really bypassed 142376960 B / 143818752 B of logical descriptor clear
extent. These are not measured DDR bytes, latency savings or a TPS estimate.
All-fast model observations do not cover fallback hardware interleavings; host
fixtures provide fallback scope. The old bounded-wait failure limitation remains.

Restoration
Root restored all three production sources to their baseline hashes, and selected
the original frozen accepted DSP 3baddb92 rather than doing another full rebuild.
All seven local and phone lfm-router-arm-1 binaries were reverified against the
accepted ARM-router runtime: app c34aa / core 072b / original DSP 3b. The controlled 1ef/f3
pair remains evidence only. root-normal-restored.json distinguishes this exact
restoration from the earlier 72083514 full-rebuild artifact. Accepted ARM-router
behavior remains; the ring source change is absent.

Archive and reproduction
manifest.json hashes compressed and original bytes of every payload; gzip mtime 0.
raw/ contains 12 ordinary logs/IDs/metadata/memory records. diagnostic/ keeps both
live collection and post-run dumps with PID/count review. validation/ retains
source/test patches, original author/independent checks and root restoration.
fixed-build/ holds full compile/link plans and source/object/tool hashes plus
independent repeat-link checks. Executable payloads/large whole-TU disassembly
are omitted with hashes, not silently represented as bundled files. No SDK,
models or toolchains are included. Compiler codegen may vary on recompilation;
reproduction must create and freeze a new common-object pair instead of asserting
it recreates the historical whole ELF SHA at another path.

  python3 reproduction/verify_archive.py ARCHIVE
  python3 reproduction/reproduce_host.py --archive ARCHIVE \
    --output NEW_DIR --prepare-only
Omit prepare-only to run the archived basic ring sanitizer fixture; --mutations
adds eight ring mutants. Optional --repo SOURCE_EQUIVALENT_REPO --hexagon-tools
HEXAGON_TOOLS_ROOT adds whole-M1 checks/two caller mutants and verifies external
repository dependency hashes first. The SDK/libnative root contains Tools/.
The original fixed-object build harness accepts explicit paths; it requires the
recorded revision, SDK and HexKL inputs. Archive verification does not rerun it.
