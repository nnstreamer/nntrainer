# 80 — M=1 MoE dispatch on the existing HVX GEMV, no 64-row HMX block

Issue: dlwlzzero/nntrainer#80 (LEDGER §3 ⑤, wall 1; tracker #76; contract
`docs/plans/0001-htp-moe-decode-agent-system.md`). Branch
`htp/80-m1-moe-gemv-dispatch` from `htp_moe` @ `66a839c3` (source tree
identical to the PR head `2ce38d65`; only docs moved since). All
`path:line` references below were read against that tree.

This issue ends in a PR with host gates only (contract §9 rungs 0–3). No
handoff document is written here; §4 step 7 describes the later device A/B
so the implementer builds what it needs.

## 1. Goal and gate

Acceptance criterion (issue, verbatim, then made measurable):

| criterion | what counts as done |
|---|---|
| Host check: M=1 HVX path bit-identical to the HMX path at the tiny shape and the real shape, M = 1, 2, 4, empty experts and two rows of one token pair to the same expert covered | `bash test/htp/host/run_host_checks.sh` prints, for each of the 6 (shape × M) cases, `M1 GEMV ... f32 memcmp=0 i32 exact=yes blocks=0 dma_kb=0`, then the single line `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4; tiny+real)`, and still ends in `ALL CHECKS PASS` and `WORKER POOL LANES OK` |
| `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6, `ninja -C build`, `tools/htp_syntax_check.sh` | unchanged gates (the ARM decode path is not touched by this issue; the tiny fixture runs CPU only) |
| `test/htp/build.sh` (v79, `-Wall -Werror`) and `build_android.sh --htp` | md5s of `libnntr_hvx_skel.so`, `nntrainer_causallm`, `libnntrainer.so`, `unittest_hvx_mm_u8i4` in the PR body |
| Prefill untouched with the switch off | `git diff --stat` lists none of the shared kernels (§5 of this plan); the existing 37-row fixture gives `MOE KERNEL MATCHES REFERENCE` and `HMX blocks: 5` with the switch off **and** on (M = 37 > 4 must take the HMX path), and the two outputs `memcmp` to 0 |

Standing gates carried into the later handoff (not this issue's gate):
prefill ≥ −5 % of variant A (the switch never applies at M > 4, so B and
C must show the same prefill as A within noise); generated text identical
to the CPU `q40` run; `NNTR_L2_DIFF` not applicable to `QS4CX_WH`
(LEDGER rule 6) — the int32/f32 identity is carried by the host check and
the device gtest instead.

## 2. Where it lives

### 2.1 The kernel (DSP side)

* `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c`
  * `hexkl_mm_u8i4_moe_layer_run` `:573-1247`. The shared prologue the
    M=1 path reuses unchanged: validation `:594-628`, scratch sizing and
    carve `:694-773`, active-expert compaction `:794-805`, act copy-in and
    output zero `:806-815`, the activation quant scan
    `hvx_quant_rows_u8_params(act_c, M, m_pad, K, ...)` `:832` and the
    slot tables `:838-853`. **The branch goes after `:853` and before
    `:823`'s first `moe_push_gate_up_chunks`** — i.e. the first gate_up
    push must move below the branch (today it sits at `:823-827`, ahead
    of the scan). With the flag off the only new instruction on the HMX
    path is one `if`.
  * The tail path is the template: `moe_tail_pair_unit` `:320-336`
    (two `hvx_gemm_u8i4_wh_col` columns + `hvx_dequant_swiglu_acc_tiles_to_f32`),
    `moe_tail_requant_unit` `:340-359` (params with `m_pad = 64`, pack of
    `m4` rows with zeroed padding), `moe_tail_down_unit` `:362-376`
    (one column + `hvx_dequant_acc_tile_to_f32`), scatter
    `hvx_scale_add_rows_f32` `:234-241`. The weights are read straight
    from `g->wh_bytes` / `d->wh_bytes` (arena, `borrowed`), never DMA'd.
  * Probe columns filled by the HMX path that the M=1 path must fill or
    zero: `HEXKL_PROBE_BLOCKS` `:961`, `DMA_KB` `:138`, `MM` `:471-488`,
    `SWIGLU` (tail worker-time, `moe_tail_probe_add` `:310-316`),
    `QUANT` `:923`, `REQUANT` `:1123-1129`, `DEQUANT`, `SCATTER`,
    `ACC_COPY` (= `STAGE`) `:806-810`, `:1234-1236`.
* `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.h:114-120` —
  `hexkl_mm_u8i4_moe_layer_run` gains `uint32_t flags` (last parameter);
  new `#define HEXKL_MOE_FLAG_M1_GEMV 1u`.
* `nntrainer/tensor/htp_backend/hmx/hexkl_probe.h:34-122` — one new
  NOT-a-time slot `HEXKL_PROBE_PATH` before `HEXKL_PROBE_N` (0 = HMX
  block loop, 1 = M=1 GEMV).
* `nntrainer/tensor/htp_backend/hvx/hvx_gemm_u8i4_wh.{c,h}` — **not
  changed** (`gemm_rows4` `:122-169` already handles `rows ≤ 4` in one
  group and clamps the unused row reads to `r0`, `:130-132`; the
  `l2fetch` stride field holds `112*512` and `64*512`, `:96-99`).

### 2.2 The skel entry, IDL, session (contract consumers)

* `test/htp/nntr_hvx.idl:318-341` — the two `mm_u8i4_moe_layer*`
  signatures stay. One **additive** method at the end of the interface:
  `AEEResult moe_set_opts(in uint32 flags, rout uint32 applied);`.
  Both `nntrainer/tensor/htp_backend/generate_stub.sh` and
  `test/htp/build.sh:72-75` regenerate from it; both `generated/` dirs
  are gitignored (`test/htp/.gitignore:1`,
  `nntrainer/tensor/htp_backend/.gitignore:1`), so nothing generated is
  committed, but the ARM app must be rebuilt against the new stub (rung 3).
* `test/htp/nntr_hvx_session.h:51-62` — new field `uint32_t moe_flags;`
  (session-scoped; `nntr_hvx_open` `calloc`s the session,
  `test/htp/hvx_add_f32.c:44`, so it starts at 0 = HMX path).
* `test/htp/nntr_hvx_mm_u8i4.c`
  * `MOE_T_*` `:864-888`: new `MOE_T_PATH` before `MOE_N_STAGES`
    (mirrors `HEXKL_PROBE_PATH`), filled at `:1002-1021`. `MOE_N_STAGES`
    grows by one, which is what the existing `stage_usLen` check `:987`
    turns into `AEE_EBADPARM` on a stale skel.
  * `nntr_hvx_mm_u8i4_moe_layer` `:942-966` and `_timed` `:968-1022`
    pass `s->moe_flags` to the kernel.
  * new `nntr_hvx_moe_set_opts`: stores `flags & HEXKL_MOE_FLAG_M1_GEMV`,
    echoes the stored value in `applied`.

### 2.3 The ARM side (`HtpComputeOps`)

* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`
  * `HTP_MOE_T_*` `:149-170`: add `HTP_MOE_T_PATH` at the same position
    as `MOE_T_PATH`.
  * `HtpProfile::Bucket` (`:370-376` region) and `addInvokeMoeLayer`
    `:283-300`: new counter `m1_calls += (stage_us[HTP_MOE_T_PATH] != 0)`.
  * `dump()` `:429-480`: append ` m1_gemv=%llu/%llu` (m1_calls / calls)
    after `blocks=%llu` on the level ≥ 2 line.
  * The switch: `static const bool moe_m1_gemv = std::getenv("NNTR_MOE_HTP_M1_GEMV") != nullptr && std::atoi(...) != 0;`
    read once, same pattern as `NNTR_HTP_KEEP_ARM_WEIGHTS` `:1868`. Sent
    once per session from `gemm_qs4cx_moe_layer_fp32` `:856-893` (first
    call, guarded by a `std::once_flag`) through `nntr_hvx_moe_set_opts`;
    on any error or `applied != flags` **throw** with the rebuild-the-skel
    hint used at `:1520-1536`. Print one stderr line either way:
    `[HTP] moe m1 gemv: on (applied=0x1)` / `off`. That line is the
    proof of which path a non-profile tok/s run took; the
    `[HTP-PROFILE]` `m1_gemv=` and `blocks=` fields are the per-call proof.
  * `invokeMoeLayer` `:1458-1548` and the `[M0-PROF]` stages
    (`Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:432-439`)
    are not touched: the ARM-side stage list, the FastRPC call shape and
    the `gemm_qs4cx_moe_layer_fp32` gate `:498-505` stay as they are, so
    `[M0-PROF]` numbers remain comparable (only `ffn=` moves).
* Not touched, stated for completeness: the quantizer's format tag
  (`nntr_quantize_stream`, no layout change), the loader check
  (`nntrainer/tensor/float_tensor.cpp:774-775` message), and
  `tools/htp_fc_report.py` (it parses no MoE row: `grep -i moe` finds
  nothing; the MoE table is printed by `HtpProfile::dump()` directly).

### 2.4 Host check and device gtest

* `test/htp/host/moe_layer_host_check.c` — the stubs it already carries
  cover every primitive the new path calls (`hvx_gemm_u8i4_wh_col`
  `:63-77`, `hvx_dequant_swiglu_acc_tiles_to_f32` `:291-314`,
  `hvx_dequant_acc_tile_to_f32` `:182-197`, `hvx_quant_pack_u8_ah_rows`
  `:165-181`, `hvx_worker_pool_run` `:202-210`, `hvx_scale_add_rows_f32`
  `:247-258`); `hexkl_probe_on = 1` `:341` so `BLOCKS`/`DMA_KB`/`PATH`
  are counted. `run_host_checks.sh:27-33` builds it with
  `-DMOE_TAIL_MAX_ROWS=16u`; keep that, add `-O2` (the real-shape cases
  are scalar-heavy, §4 step 1).
* `test/unittest/unittest_hvx_mm_u8i4.cpp` — new
  `TEST_F(HmxMmU8I4Layer, MoeLayerM1GemvMatchesHmx)` modelled on
  `MoeLayerFromArenaMatchesHeap` `:2117-2240` (run the same call with
  opts off and on, `memcmp == 0`). Built in rung 3, run only on device.

## 3. Design

**Chosen: an in-kernel branch, session-scoped flag, foreground lane only,
arena reads.**

1. *Where.* Inside `hexkl_mm_u8i4_moe_layer_run`, a static
   `moe_m1_run(...)` called after the slot tables (`:853`) when
   `(flags & HEXKL_MOE_FLAG_M1_GEMV) && M <= MOE_M1_MAX_ROWS (4) &&
   n_active <= MOE_M1_MAX_EXPERTS (16)`; otherwise the code falls through
   to the HMX loop exactly as today. The first gate_up push (`:823-827`)
   moves below the branch so the M=1 path issues **no DMA at all**
   (`DMA_KB` = 0, the ring is untouched, `hexkl_dma_ring_reset` at `:807`
   stays). `n_active ≤ 16` is the top-4 bound at M = 4; larger routing
   falls back to the HMX path rather than growing scratch.
2. *Units and the doc 47 §21.1 trap.* The tail path lost because
   background units (20–40 µs each, non-preemptible) held workers that
   the foreground epilogue and the synchronous requant then waited for.
   At M ≤ 4 there is no HMX loop and therefore nothing for a background
   lane to hide under, so the M=1 path uses **only** `hvx_worker_pool_run`
   (caller + 3 workers = the 4 HVX contexts, `hvx_add_f32.c:113-118`),
   three barriers per call, no `submit_bg`, no `submit`:
   * stage A — one `run` over `n_active × inter_ntiles` (= 4 × 56 = 224
     at real shape) units, unit u = (expert `u / 56`, gate column
     `u % 56` and its up column `56 + u % 56`): two GEMV columns into the
     expert's int32 pair tile (row stride 32, `MOE_M1_MAX_ROWS` rows), then
     `hvx_dequant_swiglu_acc_tiles_to_f32(..., n_pairs = 1, g0 = j, m, ...)`
     into the expert's `gate_f32` — byte for byte the tail's
     `moe_tail_pair_unit`. Each lane takes a contiguous slice
     `[U*i/n, U*(i+1)/n)` (expert-major, so a lane's `l2fetch` stream stays
     inside one weight).
   * stage B — one `run` over `n_active` units: per expert the tail's
     `moe_tail_requant_unit` (zero rows `[m, m4)`, `hvx_quant_rows_u8_params`
     with `m_pad = 64`, `hvx_quant_pack_u8_ah_rows` into the expert's
     `mid_ah`). Same functions and arguments as the HMX path's `:1124-1128`
     for rows `< m`, hence the same bytes.
   * stage C — one `run` over `n_active × dn_ntiles` (= 256) units: the
     tail's `moe_tail_down_unit` into the expert's `res_f32`.
   * scatter — inline on the caller, expert order, row order:
     `hvx_scale_add_rows_f32(out_c + rows[r]*N_out, res_e + r*N_out,
     weights[r], N_out)`. This is the exact add sequence of
     `moe_scatter_worker` `:227-244` (blocks are sequential across experts
     and rows are distinct inside one), so the f32 bytes match.
   * Activation: rows `[slot_of[i], slot_of[i]+m)` packed inline with
     `hvx_quant_pack_u8_ah_rows(act_c, slot_row, ...)` per expert (≤ 16
     rows total); the block's padding rows are never read
     (`gemm_rows4` clamps). No background pack job.
3. *Scratch.* Per active expert: pair tiles `56 × 2 × 512 B`, down tiles
   `64 × 512 B`, `gate_f32` `4 × inter × 4 B`, `mid_ah` `56 × 2048 B`,
   `res_f32` `4 × N_out × 4 B`, `rq_scale/zp` `64 × 8 B` ≈ 261 KB; ×16
   bound = 4.2 MB, carved from the session scratch under the existing
   "reserve for the bound, never regrow" rule (`:686-693`). Address
   budget: ≤ 4.2 MB of the ≈ 182 MiB heap, and only when the decode call
   comes first (a prefill call already reserved 12.8 MB).
4. *Weight feed for this issue.* The GEMV reads `wh_bytes` in the arena
   with vector loads behind a per-column `l2fetch`, as the tail does. Per
   M=1 call at top-4: `4 × (2048·3584/2 + 1792·2048/2) = 22.0 MB`. Read
   at 18 GB/s → **1.22 ms**; at 38 GB/s → **0.58 ms**. HVX work on top:
   ≈ 0.16 ms if the 4 lanes stream (doc 48 §3), up to ≈ 0.35 ms because
   `gemm_rows4` issues all four accumulators' `vrmpy` even at `rows = 1`
   (three of four are dead at M=1). Expected `dsp` per call therefore
   0.7–1.4 ms against today's 1.35, with `blocks = 0` and `acc = 0` as the
   proof that wall 1 fell even when the total does not move — the DDR
   rate is what the handoff reads out of it. The DMA-into-VTCM feed and
   the row-specialised GEMV are LEDGER ⑥ after measurement C; the
   `hvx_gemm_u8i4_wh.c` object stays byte-identical in this PR.
5. *Switch.* `NNTR_MOE_HTP_M1_GEMV=1` (env, read once on the ARM side;
   default off). Wire = the additive IDL method `moe_set_opts`, called
   once per session; the DSP stores the flag and decides per call on `M`.
   Chosen over an `nntr_config.json` key because the backend does not
   see `nntr_config` (`moe_engine` is read in
   `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:51` and
   would need plumbing through `compute_ops.h`), and because the handoff
   flips it per run without editing files on the device.
6. *Profile.* On the M=1 path: `PATH = 1`, `BLOCKS = 0`, `DMA_KB = 0`,
   `ACC_READ = 0`, `DRAIN = 0`; `QUANT` = act params + inline pack;
   `MM` = wall of stages A + C on the caller (paired `T0`/`ADD` around
   each `run`); `SWIGLU` = summed worker-time inside A and C units (the
   tail's `moe_tail_probe_add` convention, atomic); `REQUANT` = wall of
   stage B; `SCATTER` = wall of the inline scatter; `STAGE` unchanged.
   Every timer stays paired; `SWIGLU / 4 ≈ MM` says the lanes were
   balanced, `22 MB / MM` is the effective arena read rate.

**Rejected: a separate IDL entry `mm_u8i4_moe_decode`.** It duplicates the
11-argument marshalling, needs its own ARM call site and profile bucket,
and moves the M-threshold decision to the ARM side where the skel version
is unknown; the in-kernel branch keeps one entry, one profile row and one
host harness, and the stale-skel failure stays the loud one
(`moe_set_opts` → `AEE_EUNSUPPORTED` on an old skel, `stage_usLen` →
`AEE_EBADPARM` on the timed path).

Also rejected: running the four experts as background jobs with the
caller waiting (the tail's structure) — doc 47 §21.1 ②③ apply verbatim and
there is no HMX shadow at M=1 to make it worthwhile.

## 4. Steps

Each step ends in the rung named from `.claude/skills/hexagon-gates`.
Commit split: **K1** kernel + skel (`hexkl_mm_u8i4_moe.{c,h}`,
`hexkl_probe.h`, `nntr_hvx.idl`, `nntr_hvx_session.h`,
`nntr_hvx_mm_u8i4.c`), **K2** ARM side (`htp_compute_ops.cpp`), **H**
host check (`moe_layer_host_check.c`, `run_host_checks.sh`), **G** device
gtest (`unittest_hvx_mm_u8i4.cpp`), **D** docs (`docs/htp_moe/*`, kept
apart from K/H/G per contract §5). Format every C/C++ file with
`clang-format-14` (rung 0) before each commit.

1. **Host check first, red.** Extend `moe_layer_host_check.c`: pass `0`
   as the new `flags` argument at the four existing call sites
   (`:482`, `:569`, `:584`, `:593`); add `run_m1_cases()` with
   * shapes: tiny (`K=64, inter=32, N_out=64`, 5 experts, as `:434`) and
     real (`K=2048, inter=1792, N_out=2048`, `n_experts = 32`, weights
     allocated only for the experts that get rows — the kernel validates
     handles only where `row_count != 0`, `:598-613`);
   * routings: M=1 → counts `{1,1,1,1}` on four experts; M=2 → `{2,2,1,1,1,1}`
     (both tokens share two experts: "two rows to the same expert");
     M=4 → `{4,3,2,1,1,1,1,1,1,1}` (one expert holds all four tokens); the
     remaining experts empty; at the real shape `row_index` are distinct
     within an expert;
   * for each case: run with `flags = 0` (HMX stand-in), record
     `HEXKL_PROBE_BLOCKS` (must be `n_active`) and the output; run with
     `flags = HEXKL_MOE_FLAG_M1_GEMV`, require `BLOCKS == 0`,
     `DMA_KB == 0`, `PATH == 1`, `memcmp(out_hmx, out_m1) == 0`, and the
     int32 tiles the GEMV stub logged (key `(wh, nt)`, rows `< m`) equal a
     scalar int32 reference computed in the check from the quantised row
     bytes and `wh_value`; the M1 output also passes the existing
     tolerance check against the f32 reference;
   * the existing 37-row fixture once more with the flag on: `BLOCKS == 5`
     and `memcmp` with the flag-off output `== 0` (the switch does not
     apply at M > 4).
   Pass lines: one `M1 GEMV shape=<tiny|real> M=<n> f32 memcmp=0 i32 exact=yes blocks=0 dma_kb=0`
   per case, then `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4; tiny+real)`.
   Add `-O2` to the check's compile line. Expect ≈ 1–2 min for the real
   shape (the HMX stand-in does 64 rows per tile). Gate: the check fails
   to compile (no `flags` yet) — commit H is finished later, after step 2.
2. **Kernel (K1, DSP-side C only).** `flags` parameter, `MOE_M1_*`
   constants, `moe_m1_run` with the three `run` stages and inline
   scatter, scratch sizing, the moved first gate_up push, probe fills
   (§3.6), `HEXKL_PROBE_PATH`. Gate: **rung 1** —
   `bash test/htp/host/run_host_checks.sh` prints the step-1 lines and
   `ALL CHECKS PASS`; `ninja -C build` unaffected (these files are not in
   the host build). Commit H and K1.
3. **Skel entry (K1 continued).** IDL method `moe_set_opts`, session
   field, `MOE_T_PATH`, `s->moe_flags` passed at `:962-965` and
   `:995-998`, the stage fill. Gate: **rung 2** — `./test/htp/build.sh`
   (`-Wall -Werror` clean), `md5sum test/htp/build/libnntr_hvx_skel.so`
   recorded.
4. **ARM side (K2).** Env read, once-per-session `moe_set_opts` with a
   throw on error/mismatch, the stderr line, `HTP_MOE_T_PATH`, the
   `m1_gemv=` field in `dump()`. Run `generate_stub.sh` (gitignored
   output). Gate: **rung 1** — `ninja -C build`, `tools/htp_syntax_check.sh`
   (note it declares FastRPC entries variadic, so the `moe_set_opts`
   argument list is checked only by rung 3), the two gtests (`*qs4cx*`,
   `*Lfm2Moe*` 6/6 after generating the tiny fixture).
5. **Device gtest (G).** `MoeLayerM1GemvMatchesHmx`: register four
   real-shape experts from the arena as `MoeLayerFromArenaMatchesHeap`
   does, M = 1 and M = 4 routings, `moe_set_opts(0)` run vs
   `moe_set_opts(1)` run, `memcmp == 0`, then `moe_set_opts(0)` again so
   later tests see the default. Gate: **rung 3** —
   `build_android.sh --htp`, `readelf -d` shows `libsdkl.so` and
   `libcdsprpc.so`, `ndk-build unittest_hvx_mm_u8i4`, md5s recorded.
6. **Prefill-untouched proof (§5) and PR.** `git diff --stat` against
   `htp_moe`; the step-1 M = 37 lines; PR body with one `<details>` per
   commit and the md5 table. Set `state:review`.
7. **Later device A/B (description only; a separate issue writes the
   handoff after #77 is measured).** Variants, ≤ 4, full-model E2E per
   the handoff skill, prompt 512, gen 64 / 512 / 1024, twice each,
   `NNTR_NUM_THREADS=8`, `q40-qs4cx-wh` model:
   * **A** — the unchanged reference: BENCHMARK.md's artifacts
     (`htp_moe` @ `2ce38d65` skel `0a3d4b81…`, app `53814a39…`), run
     first in the sitting.
   * **B** — this PR's skel + app, `NNTR_MOE_HTP_M1_GEMV` unset. Must
     equal A within the sitting's noise (proves the default is off and the
     IDL/profile changes cost nothing); stderr shows `moe m1 gemv: off`.
   * **C** — same binaries, `NNTR_MOE_HTP_M1_GEMV=1`. Stderr shows
     `moe m1 gemv: on (applied=0x1)`; text identical to B (and to A);
     prefill within −5 % of A; decode tok/s is the number.
   * One `NNTR_HTP_PROFILE=3` run per variant at gen 64: the `M==1` row
     must read `blocks=0 m1_gemv=<calls>/<calls>` for C and `blocks=4×calls
     m1_gemv=0/<calls>` for A/B; read `dsp`, `mm`, `swiglu`, `requant`,
     `scatter`, `transport`. `22 MB / mm` is the arena read rate that
     decides ⑥. Plus `unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1*'`
     once.
   * Expected reading: C `dsp` ≈ 0.7 ms if the arena streams at ≈ 38 GB/s,
     ≈ 1.2–1.4 ms if at 18 GB/s (wall 1 gone, wall 2 exposed); either way
     `blocks=0`. Decode tok/s moves by at most 22 × (1.35 − dsp_C) ms per
     token until wall 3 (transport 0.57 ms × 22) is addressed.

## 5. Prefill untouched — how it is proven

* Files that carry the prefill path and are not in the diff (so their
  objects are identical by construction; a `hexagon-clang -c` md5 adds
  nothing over `git diff --stat` and is not needed):
  `hvx/hvx_gemm_u8i4_wh.c`, `hvx/hvx_dequant_i32.c`, `hvx/hvx_quant_u8.c`,
  `hvx/hvx_scale_add_f32.c`, `hvx/hvx_worker_pool.c`,
  `hvx/hvx_gather_ah_u8.c`, `hmx/hexkl_dma_ring.c`,
  `hmx/hexkl_mm_u8i4_dma.c`, `hmx/hexkl_acc_tile.c`.
* `hexkl_mm_u8i4_moe.c` necessarily differs; for it the proof is the
  host check alone: the existing 37-row / 5-expert fixture with the flag
  off gives the unchanged `MOE KERNEL MATCHES REFERENCE`, `HMX blocks: 5`,
  `weight DMA` count; with the flag on, the same three lines and
  `memcmp == 0` against the flag-off output. The diff review shows the
  only code on the flag-off path is the added `if` and the relocated
  first gate_up push (same call, same arguments, after the scan instead of
  before it — the scan does not touch VTCM, `:817-822`, so the DMA still
  has the scan to hide behind; the implementer confirms `DMA_FIRST` in the
  handoff's B row is unchanged from A).
* On device: variant B vs A (§4 step 7) is the prefill gate proper.

## 6. Risks

| risk | how the plan makes it visible |
|---|---|
| **Arena reads from HVX are not the DMA rate.** The arena is CPU-uncached ION (`htp_rpcmem.h:86-90`) mapped on the DSP through `HAP_mmap_get` (`nntr_hvx_mm_u8i4.c:360`); whether that mapping is cacheable for vector loads has never been measured — the tail path's worker-time column was 0.0 in its only device run (doc 47 §21.1). If uncached, `l2fetch` is a no-op and the GEMV crawls far below 18 GB/s | `swiglu` (worker-time) and `mm` in the C profile row; `22 MB / mm` read against 18 and 38 GB/s. A rate under ≈ 10 GB/s sends ⑥ to DMA-into-VTCM (or a cached DSP mapping) before anything else |
| DVFS: with the HMX idle the DSP may clock down | `NNTR_HTP_PROFILE=3` (5× repeat, `min`); A/B in one sitting (LEDGER rule 9) |
| Thermal drift between sittings | variant A first, every sitting; B ≈ A is the sanity check that the sitting is readable |
| Stale skel | timed path → `AEE_EBADPARM` from `stage_usLen` (`:987`); non-timed path → `moe_set_opts` `AEE_EUNSUPPORTED` → the app **throws** when the env is set. A silent fallback would let C run the HMX path and report it as the GEMV |
| Address space | ≤ 4.2 MB more session scratch, bounded by `MOE_M1_MAX_EXPERTS`, never regrown; no VTCM change |
| Bit-identity on real silicon (host stubs share `wh_value`, so host identity is by construction) | device gtest `MoeLayerM1GemvMatchesHmx` (`memcmp == 0`) and identical generated text C vs B |
| Pool barrier cost at 3 `run`s per call | inside `mm`/`requant` columns; if `swiglu/4` is far below `mm`, the lanes are unbalanced or the barrier is the cost |

## 7. Docs to update

* `docs/htp_moe/BENCHMARK.md` — Artifacts: rows for this PR's
  `libnntr_hvx_skel.so`, `nntrainer_causallm`, `libnntrainer.so`,
  `unittest_hvx_mm_u8i4` with md5 and commit (supervisor, from the PR
  body). Results: nothing until the later handoff.
* `docs/htp_moe/LEDGER.md` — §3 ⑤: "dispatch built (#80, PR …), arena
  feed, switch `NNTR_MOE_HTP_M1_GEMV=1` default off; device A/B pending
  after #77"; ⑥ gains "and the M=1 GEMV feed (VTCM staging or cached
  mapping) plus the row-specialised `gemm_rows4`" as its scope; §1 gets
  no new rule until silicon speaks.
