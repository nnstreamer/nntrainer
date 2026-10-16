Fixed-object DSP pair for native LFM M1 ring reset

Status: host build/identity checks PASS; no device run or performance conclusion.
No root checkout, SDK, phone, commit, or push was changed by this work.
Base snapshot: b17d1d2fbda288c500f56072a7ad740c88f1c488.
Production patch: 7803996b40175702f5d9978d29aa8ea31df5ec26282f592fbac6b97ae3ef2a89.
The original ring candidate's correctness review remains separate; this work
controls DSP build identity and adds a diagnostic-only actual-hit counter.

Normal binaries
  build/control/libnntr_hvx_skel.so
    1efcd9dcde2109634be700bb778ce1ba802ea6bc69c3dec1612030edcd4ce1e4
  build/candidate/libnntr_hvx_skel.so
    f3e68660aff009b218efe16ab51e2ea93ba31e66611d555d768f5a82355c08bf
These form a newly compiled pair, not the historical DSP3b binary. Both must be
paired with the same unchanged host runtimes for model comparison. Device IDs,
actual dispatch, memory and speed still need root's serialized validation.

What is frozen
The pinned checkout's test/htp/build.sh source order and flags produce compiler
-### plans. The actual cc1 invocations are retained, including the two special
IEEE TUs; only their temporary output paths and dependency-emission metadata are
changed. 63 baseline objects are compiled once. The exact baseline object bytes
are retained and hashed. After applying the three-file production patch to this
isolated source snapshot, only hexkl_dma_ring.c and hexkl_mm_u8i4_moe.c are
recompiled; the new declaration in ring.h is used by those two TUs. The other
61 objects are linked unchanged. Source order, linker inputs, compilation
commands, external compiler/archive/headers and actual dependency hashes are
recorded. Frozen object hashes and unchanged external inputs are checked again
following both links. No extra codegen, math, cache, reader, or buffer setting is
introduced. There is no LTO in the recorded build.

Linker nondeterminism and common setting
Two default links of the same exact baseline object set differed. Both original
ELFs are preserved in build/threaded-link-attempt/control; a read-only section
comparison shows equal sizes but different .rela.dyn, .rela.plt and .text bytes.
Thus freezing compile outputs alone did not establish a repeatable final ELF.
The Qualcomm linker's documented -no-threads option is applied to both final
arms. Four exploratory links and both final links per arm are byte-identical.
All final imports match, pass the existing runtime allowlist/weak-DSPQ rule, and
ELF flags report v81. This common link-only setting is explicit; it is not
presented as an unchanged original build command. No objcopy was used on ELFs.

Within the two recompiled TUs, the symbol byte screen finds 38 unchanged MoE
functions and seven unchanged ring functions. The two MoE push wrappers differ
only by internal call displacement; main MoE contains the intended new call.
Ring push/drain contain two/one alignment nops plus corresponding branch offsets.
Focused raw and normalized disassembly are retained. Relocation addends and loop
addresses are normalized only in the supplemental instruction-text screen;
that screen is not a semantic proof or claim of byte-identical entire TUs.
This bounded difference is distinct from recompiling unrelated router/softmax/
attention TUs, which does not occur between these two final arms.

Diagnostic (never use diagnostic TPS as a normal result)
The separate diagnostic/build/libnntr_hvx_skel.so replaces only the candidate
ring object. NNTR_LFM2_M1_RING_RESET_TRACE enables one inline relaxed atomic32
sequence counter and one ERROR line at each actual try_reset return:
  nntr-ring-reset q=1 used=0 !
  nntr-ring-reset q=2 used=1 !
used=1 proves this invocation retained completed descriptor bytes instead of
clearing the ring. It is an attempt/return observation; later MoE RPC success or
failure is not inferred. The unchanged caller only tries this under its existing
native-LFM M1 guard. There is no worker logging or altered descriptor math.
The diagnostic adds a four-byte counter plus logging/code overhead. Counter
increment lowers inline with no atomic-library imports. The complete diagnostic
imports equal the normal pair's imports. Its two links are exact. With the macro
off, the ring object is byte-identical to the normal candidate object, proving
zero diagnostic machine-code overhead in normal builds. Normal source and all
normal objects/binaries are restored/retained exactly after this build.

The actual ring source runs under the existing delayed-DMA fixture, with a host
FARF sink checking the returned hit and monotonically incremented sequence at
both real call sites. ASan/UBSan: 175 lifecycle/reset checks PASS. This remains
host emulation, not real hardware cache/coherence testing. The separate strict
parser passes six positive and 30 negative records/cohorts; it rejects missing,
reordered, malformed, overflow and conflicting duplicate events. Exact transport
replays are counted explicitly. Device CLI requires --pid from the supervisor/run metadata and rejects tags
from any other process; --host-unprefixed is an explicit separate fixture mode.
--expected-calls checks
an independently known total; omitting it proves only a contiguous observed
prefix, not absence of truncation at the end. --min-hits defaults to one, so
merely enabling an unused fast path fails that diagnostic. Logical skipped
clear bytes are reported separately from physical bandwidth or speed.

Scope and limits inherited from the production candidate
Eligibility requires the actual last type9/size1 descriptor to be complete.
Otherwise the original reset remains. Successful reuse resets software indices;
every later push fills every hardware field before submission. This does not
repair the original ring timeout/drain fallback. No cache/capacity change or
new persistent content buffer is made. Production correctness cases and previous
review are in /tmp/nntr-lfm2-m1-ring-reset-20261017 and its independent review;
this build harness is not a replacement for those results.

Reproduction
  python3 build_fixed_objects.py --repo CHECKOUT --revision RECORDED_COMMIT \
    --candidate-patch FROZEN_PRODUCTION_PATCH --sdk-root HEXAGON_SDK_6.4.0.1 \
    --tools-root HEXAGON_TOOLS_19.0.04 --hexkl-root HEXKL_ADDON \
    --output NEW_OUTPUT
The script rejects build.sh drift and unexpected candidate hashes. It writes only
NEW_OUTPUT. A full fresh run uses the stable common link setting from the start.
--resume-baseline exists only to resume the retained original link-validation
failure without recompiling the baseline; it requires exact prior source/object
and compiler identity. The initial output path filter and host success-summary
assertion mistakes were caught by validation, preserved, and corrected. Neither
changed production or reused a failed normal result.

  python3 diagnostic/prepare.py --build-root BUILD_OUTPUT \
    --output NEW_DIAGNOSTIC_OUTPUT --fixture FROZEN_ACTUAL_RING_FIXTURE
  python3 diagnostic/parse.py ONE_PROCESS_LOG --pid CHILD_PID [--expected-calls N]
The normal snapshot is restored in a finally block; macro-off object equality
and unchanged frozen hashes are asserted. Source/metadata and exact command
manifests permit review without SDK or device access. Runtime files and full
object/disassembly outputs remain local artifacts, not proposed repository edits.
