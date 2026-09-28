# Plan #97: DSP skel fails to load (0x80000406) — undefined `hexkl_dma_trace_*` symbols

Issue: dlwlzzero/nntrainer#97 (p0). Contract:
`docs/plans/0001-htp-moe-decode-agent-system.md`. Tree read at `htp_moe` @
`d5eded3c`. Sources: `test/htp/build.sh`, `test/htp/host/run_host_checks.sh`,
`test/jni/Android.mk`, `.claude/skills/hexagon-gates/SKILL.md`,
`docs/plans/94-sitting-2.md` §3.1, the staged sets under
`/local/mnt/workspace/htp_moe/{77,94}/`, and the filled
`origin/htp/94-sitting2-anchor-trace:docs/measurements/94-sitting2-anchor-trace.md`.

**This is a one-line build fix plus a build-time guard.** No kernel, IDL,
`HtpComputeOps`, quantizer, loader or profile-table change; contract §2
(three walls, arena budget, no CPU fallback for `QS4CX_WH`) and doc 45 §3
are untouched by construction. The skel is the only artifact that changes.

## 1. Goal and gate

Acceptance (issue, made measurable):

| deliverable | what counts as done |
|---|---|
| Skel built by `./test/htp/build.sh` from the fix commit loads on `R3CY10WM83Y` | `unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*'` prints `[  PASSED  ]` with the new skel at `/data/local/tmp/htp_u8i4_layer_test/` (this is #94 block 5; no other new device work) |
| Host proof before the device | `hexagon-readelf --dyn-syms test/htp/build/libnntr_hvx_skel.so \| awk '$7=="UND"'` lists **no** `hexkl_*`, `hvx_*`, `nntr_*` symbol; the UND set equals the #77 skel's 45 entries plus `memalign` (verified in a scratch build, §3) |
| Guard | `build.sh` fails (non-zero, symbol names printed) when a referenced project `.c` is dropped from `SRCS`; prints `UNDEFINED SYMBOLS OK (<n> runtime imports)` when clean |
| #94 unblocked | the #94 handoff re-issued with the rebuilt skel; the 6 NPU A cells, block 3 (B), block 5, block 6 (C) become runnable |

Standing gates: prefill ≥ −5 % of variant A and text identical to the CPU
run are #94's own gates and are not re-defined here — this plan produces no
new variant (A stays A; only its skel becomes loadable). Bit-identity:
nothing computed changes; `run_host_checks.sh` must print the same
`ALL CHECKS PASS` / `WORKER POOL LANES OK` lines as before.

## 2. Where it lives

Root cause, verified (not re-derived):

* `test/htp/build.sh:77-89` — `SRCS`. Every `.c` under
  `nntrainer/tensor/htp_backend/{hmx,hvx}` (20 files) is listed **except**
  `hmx/hexkl_dma_trace.c` (added by `f4b9062e`, PR #93). Its seven exported
  functions (`hexkl_dma_trace.c:46,51,79,101,110,144,196`) are called from
  `hmx/hexkl_mm_u8i4_moe.c`, `test/htp/nntr_hvx_mm_u8i4.c` and
  `test/htp/nntr_hvx_dma_probe.c` (all in `SRCS`), so the link emits
  `R_HEX_JMP_SLOT` relocations against undefined
  `hexkl_dma_trace_{reset,push,sample,wait_begin,wait_end,finish,serialize}`
  (`hexagon-readelf -r` on the staged `94/libnntr_hvx_skel.A.so`). The
  Hexagon user-PD loader then refuses the object:
  `dlerror RX VA 0xFFF00000 outside ELF segment` → `0x80000406`.
* Diff of UND sets, #77 skel (`77/gtest/libnntr_hvx_skel.A.so`, loads) vs
  #94 skel: exactly the 7 `hexkl_dma_trace_*` plus `memalign`
  (`test/htp/nntr_hvx_dma_probe.c:477`, libc — legitimate runtime import).
* `-Wall -Werror` cannot catch it: a shared object may carry undefined
  symbols at link time; rung 2 was "passed" honestly and wrongly.
* Host side compiles the same file explicitly:
  `test/htp/host/run_host_checks.sh:33-34` (`moe_layer_host_check` links
  `hexkl_mm_u8i4_moe.c` + `hexkl_dma_trace.c`) and `:65-66`
  (`dma_trace_host_check`). That is why rung 1 passed: the host lists were
  updated by #93, the skel list was not.
* Nothing else is missing: a loop over all 20 backend `.c` files against
  `SRCS` names only `hexkl_dma_trace.c`.

Consumers that do **not** move: `test/htp/nntr_hvx.idl` and
`nntrainer/tensor/htp_backend/generated/nntr_hvx_stub.c` (unchanged, so no
`generate_stub.sh`), `htp_compute_ops.cpp`, `nntr_quantize_stream` format
tag, the loader check, `NNTR_HTP_PROFILE` stage tables,
`tools/htp_fc_report.py`. The two device gtests link only
`../htp/generated/nntr_hvx_stub.c` plus their own `.cpp`
(`test/jni/Android.mk:936-937`, `:960-961`), and the app links the same
stub through `nntrainer/tensor/htp_backend/meson.build:26,54` — so with an
unchanged IDL every ARM binary in `/local/mnt/workspace/htp_moe/94/` stays
valid.

Files this plan changes:

| file | change |
|---|---|
| `test/htp/build.sh` | add `$BACKEND/hmx/hexkl_dma_trace.c` to `SRCS` (line 81, next to `hexkl_dma_ring.c`); add the post-link undefined-symbol check before the `echo "built: ..."` at line 110 |
| `.claude/skills/hexagon-gates/SKILL.md` rung 2 | pass line gains the guard's line |
| `docs/measurements/94-sitting2-anchor-trace.md` (on `htp/94-sitting2-anchor-trace`) | Artifacts row 1 (skel md5 + commit), header sha note, Deviation 1 closed |
| `docs/htp_moe/BENCHMARK.md:73`, `LEDGER.md` | see §6 |

## 3. Design

**Fix.** One `SRCS` line. Verified in a scratch build (same compiler
command, `SRCS` = build.sh's list + `hexkl_dma_trace.c`, output in the
session scratchpad, not in `test/htp/build/`): `-Wall -Werror` clean,
4.6 s, 172 752 B, RX LOAD `0x22504`, zero `hexkl_*` UND, UND set ==
#77's + `memalign`.

**Guard: post-link allow-list check in `build.sh` (chosen).** After the
`hexagon-clang` line:

```
READELF="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-readelf"
# Runtime imports the DSP image provides: FastRPC HAP_*, compute_resource_*,
# QuRT, compiler-rt/CRT (__hexagon_*, __extendhfsf2, __cxa_finalize,
# __register_frame_info_bases) and libc. Anything else -- in particular a
# hexkl_*/hvx_*/nntr_* function -- is a project file missing from SRCS; the
# linker accepts it, the on-device loader does not (#97: 0x80000406).
BAD=$("$READELF" --dyn-syms build/libnntr_hvx_skel.so \
  | awk '$7=="UND" && $8!="" {print $8}' \
  | grep -Ev '^(HAP_|compute_resource_|qurt_|__hexagon_|__extendhfsf2$|__cxa_finalize$|__register_frame_info_bases$|malloc$|free$|calloc$|memalign$|memcpy$|memset$|lroundf$|nearbyintf$|snprintf$|vsnprintf$|strlcpy$)' || true)
if [ -n "$BAD" ]; then
    echo "Error: skel has undefined symbols the DSP image will not provide:" >&2
    echo "$BAD" | sed 's/^/  /' >&2
    echo "Add the defining .c to SRCS in $0 (see #97)." >&2
    exit 1
fi
echo "UNDEFINED SYMBOLS OK ($("$READELF" --dyn-syms build/libnntr_hvx_skel.so | awk '$7=="UND" && $8!=""' | wc -l) runtime imports)"
```

The allow-list is the measured UND set of the #77 skel (45 names, loads on
both units) plus `memalign`; every name is either a prefix family
(`HAP_`, `compute_resource_`, `qurt_`, `__hexagon_`) or an exact libc /
CRT name. No `hexkl_*` symbol is ever a legitimate import: `libhexkl_micro.a`
is linked statically (`build.sh:107`) and both the #77 and the fixed skel
have zero `hexkl_*` UND entries. `readelf --dyn-syms` rather than
`hexagon-nm -u` because `nm` marks the three `HAP_debug*` as weak (`w`)
and the rest `U`; the readelf column is uniform. A new libc import (e.g.
`posix_memalign`) fails the build with its name printed — the implementer
then extends the exact-name list in the same commit; that is the intended
friction (one line, visible in review), not a defect of the check.

**Rejected: `-Wl,--no-undefined` (`-z defs`).** The 45 legitimate imports
(`HAP_*`, `compute_resource_*`, `qurt_*`, libc, compiler-rt) are resolved
by the DSP image at load time and are not available as link inputs in this
build, so the flag would fail every build, and the work-around (linking
stub libraries from the SDK) adds toolchain plumbing for a check the
allow-list does in six lines. Also rejected: a generic "every backend `.c`
must be in `SRCS`" file-list check — it would have caught this instance but
not a symbol defined in a `test/htp/*.c` or a header-declared function that
lost its definition; the symbol check covers both and is what the loader
actually evaluates.

**`0xFFF00000`.** The failing VA is consistent with an unresolved
`R_HEX_JMP_SLOT` — the loader has to land the PLT jump somewhere, and a
high sentinel/unrelocated value outside every LOAD segment (RX ends at
`0x21c24`, RW at `0x56910`) is what "RX VA … outside ELF segment" says —
but the loader's exact sentinel is not documented in the SDK, so this is
consistent, not proven; the proof is that the skel with zero project-UND
symbols loads (§4 step 5).

## 4. Steps

Branch `htp/97-skel-undefined-symbols` off `htp_moe`; one commit
`[test] Link hexkl_dma_trace.c into the DSP skel and fail the build on undefined project symbols`.
Rung numbers = `.claude/skills/hexagon-gates`.

1. **Edit `test/htp/build.sh`**: `SRCS` line 81 becomes
   `SRCS="$SRCS $BACKEND/hmx/hexkl_dma_ring.c $BACKEND/hmx/hexkl_dma_trace.c $BACKEND/hmx/hexkl_kv_quant.c"`;
   insert the §3 guard block before line 110. Gate: rung 0 (`shellcheck`
   is not in the gates; just `bash -n test/htp/build.sh`).
2. **Rung 2, three runs.** (a) `./test/htp/build.sh` → `-Wall -Werror`
   clean, prints `UNDEFINED SYMBOLS OK (46 runtime imports)` and `built:`.
   Then `hexagon-readelf --dyn-syms test/htp/build/libnntr_hvx_skel.so | awk '$7=="UND" && $8 ~ /^(hexkl|hvx|nntr)_/'`
   prints nothing; `diff <(UND of /local/mnt/workspace/htp_moe/77/gtest/libnntr_hvx_skel.A.so) <(UND of the new skel)`
   shows only `> memalign`. (b) **Negative test of the guard**: temporarily
   drop `hexkl_dma_trace.c` from `SRCS` again, run, expect exit 1 with the
   7 names printed; restore. (c) Rebuild clean; `md5sum` the skel for the
   handoff. Record (a)–(c) output in the PR description.
3. **Rung 1** (`ninja -C build`, `run_host_checks.sh`, syntax check):
   unchanged pass lines — nothing on the host side moved; this is the "did
   I break rung 1 by mistake" check only. Rung 3 is **not** run: the IDL
   did not change, so stub, app and gtests are byte-for-byte the #94 set
   (§2). Confirm with
   `git diff --stat 2a75f7d9..HEAD -- Applications nntrainer test/unittest test/jni` = empty.
4. **Skill update.** `.claude/skills/hexagon-gates/SKILL.md` rung 2 pass
   line: "`test/htp/build/libnntr_hvx_skel.so` exists, `-Wall -Werror`
   clean, **and `build.sh` printed `UNDEFINED SYMBOLS OK`** (a project
   symbol left out of `SRCS` links fine and fails on the device with
   `0x80000406`, #97)". Open the PR against `htp_moe`.
5. **Re-issue #94 (device measurement unavoidable — this is the only
   step needing the phone, and it is #94's existing sitting, not a new
   one).** On `htp/94-sitting2-anchor-trace`: merge `htp_moe` once the fix
   PR lands (or cherry-pick the one commit if the sitting cannot wait;
   note which in the handoff). Rebuild **only the skel** from that head,
   restage `cp test/htp/build/libnntr_hvx_skel.so /local/mnt/workspace/htp_moe/94/libnntr_hvx_skel.A.so`,
   regenerate `md5.txt` (`cd $W && md5sum libnntr_hvx_skel.A.so tps/* gtest/* prompt512.txt > md5.txt`;
   only the skel line changes), `md5sum -c md5.txt` all OK. In
   `docs/measurements/94-sitting2-anchor-trace.md`: Artifacts row 1 md5 +
   "`./test/htp/build.sh` @ `<fix sha>`" and a one-line note that app and
   gtests remain from `2a75f7d9` because the fix touches only
   `test/htp/build.sh` (rule 3 satisfied: the diff between the two shas
   has no compiled source); close Deviation 1 with a pointer to #97; step
   1's provenance block now expects the new skel md5 in both device dirs.
   Comment on #94 with the new skel md5 and set `state:needs-measurement`.
   Handoff variants are unchanged from `docs/plans/94-sitting-2.md`: A =
   reference (unchanged binary, switch off), B = A with `NNTR_HTP_PROFILE`,
   C = A with `NNTR_MOE_HTP_M1_GEMV=1`; prompt 512, G 64/512/1024, two runs
   each. First device command of the sitting is block 5's
   `*DmaProbeShapes*` — a pass there is #97's acceptance; a second
   `0x80000406` means the stale skel is still on the device (`md5sum` both
   dirs) or a different loader fault (new issue, attach logcat).

## 5. Risks

* **Stale skel on the device.** Both device paths
  (`/data/local/tmp/nntrainer/causallm/` and `…/htp_u8i4_layer_test/`) hold
  a copy; the #94 handoff's step-1 `adb shell md5sum` of both is what makes
  a missed push visible. The staged `md5.txt` is the workstation side of
  that check.
* **Allow-list too tight for a future import.** A new libc call fails the
  build loudly with its name; fix is one token in the list. Too loose is
  the real risk and is closed by construction: no prefix in the list can
  match a project symbol (`hexkl_`, `hvx_`, `nntr_`, `htp_`).
* **Guard depends on `hexagon-readelf` column layout.** Verified on
  HEXAGON_Tools 19.0.04 (`$7` = `Ndx`, `$8` = name). A toolchain bump that
  changes columns would make `BAD` empty (silent pass) — the negative test
  in step 2(b) is the one-time proof; the skill's pass line asks for the
  `OK (<n> runtime imports)` count, which drops to 0 if the parse breaks.
* **Skel/app from different shas.** Rule 3 says one commit; step 5
  documents that the two shas differ only in `build.sh`/docs and proves it
  with the empty `git diff --stat`. If the #94 branch picks up anything
  else from `htp_moe` in the merge, rung 3 must be re-run and all app md5s
  restaged — the handoff's Artifacts table then changes in every row, not
  one.
* Host-vs-device gaps of the measurement itself (DMA rate, DVFS, thermal
  drift, address budget) belong to #94 and are already handled by its
  handoff table (thermal gate, run1 = run2 loop, post-B re-run of A).

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md:73` (#94 artifact row): skel md5
  `20fb9801…` → new, "built from" adds the fix sha for the skel; the
  rebuild note at `:75-` gets one sentence: "skel from `2a75f7d9` did not
  load (#97, `hexkl_dma_trace.c` missing from `SRCS`); rebuilt from
  `<fix sha>`, app/gtests unchanged".
* `docs/htp_moe/LEDGER.md` §1 rules: add a rule — "A DSP skel linking
  `-Wall -Werror` clean can still fail to load: undefined project symbols
  are legal in a shared object and rejected only by the on-device loader
  (`0x80000406`, `RX VA 0xFFF00000 outside ELF segment`). `build.sh`
  checks the UND set against the runtime-import allow-list since #97;
  rung 2's pass line includes it." §3a tooling note: `hexagon-readelf` /
  `hexagon-nm` live under `$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin`.
* `docs/htp_moe/LEDGER.md` cycle log: "#97 (p0) fixed; #94 re-issued with
  skel `<md5>`".
* `.claude/skills/hexagon-gates/SKILL.md` rung 2 pass line (step 4).
* `docs/measurements/94-sitting2-anchor-trace.md` on the #94 branch
  (step 5).
