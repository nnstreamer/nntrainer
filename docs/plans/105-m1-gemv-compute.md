# 105 — M=1 HVX GEMV MoE kernel, compute side: one-row loop, `l2fetch` lead, lane split

Issue #105 (p0, tracker #76, LEDGER ㉒ compute half). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. Ground truth: plan 101
§3.4 (why `mm` rose 783 → 975 µs), plan 80 (the path itself), #94 sitting 2
§C (`origin/htp/94-sitting2-anchor-trace` @ `dad0f476`). Base:
`htp_moe` @ `8beec31d`.

## 1. Goal and gate

**Goal.** At M==1 on the GEMV path, the `[HTP-PROFILE]` M==1 row's `mm`
goes from **975 µs/call** (#94 s2 C) towards **≤ 600 µs/call**. That target
is ㉒'s combined gate. This issue closes on "B's `mm` below A's by more than
±5 % and B's decode ≥ A's". If B lands above 600, the gap becomes the
feed half's target (§3.4).

**Gates (all must hold):**

| gate | where it is read | pass |
|---|---|---|
| kernel bit-identity, host | `bash test/htp/host/run_host_checks.sh` | the existing `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4; tiny+real)`, the new `HVX GEMV NATIVE BIT-IDENTICAL (libnative; m=1..16 rows1+rows4)` (step 1), the new `M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN (lanes=6; lead 0/64/192 KB)` (step 2), `ALL CHECKS PASS`, `WORKER POOL LANES OK` |
| kernel bit-identity, device | gtest `*MoeLayerM1GemvMatchesHmx*` under **every** B skel | `bad_elems_M1 value=0 of 2048`, `bad_elems_M4 value=0 of 8192`, `bit_identical value=yes` |
| host build | `ninja -C build`, `bash tools/htp_syntax_check.sh`, `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` | 6/6 PASSED, none skipped |
| skel | `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh` once per variant | `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (<n> runtime imports)` (rule 17), md5 in the handoff |
| text | every B cell against the A cell with the same G and run | byte-identical (`moe m1 gemv` line excluded). `NNTR_L2_DIFF` is n/a for `QS4CX_WH`, and so is text = CPU `q40` (different weights; BENCHMARK Results row 1) |
| prefill | prompt-512 prefill of each B cell against A's in the same sitting | ≥ −5 %. The tie-breaker is the level-2 M>1 row: `m1_gemv=0/23`, `dsp=` within 5 % of A's. A's own prefill spread was ±13 % in #94 s2, so a single cell below −5 % with the M>1 `dsp` unchanged is noise (BENCHMARK Goals, prefill row) |
| path proof | every A and B log | `[HTP] moe m1 gemv: on (applied=0x1)` (with or without `source=…`, whether or not #101 has landed). The level-2 M==1 row shows `blocks=0 m1_gemv=1408/1408`, otherwise the run is void |
| verdict | level-2 M==1 row `mm` / `dsp` (`min`, rule 2), decode tok/s means | **pass** = B `mm` < A `mm` × 0.95 and B decode ≥ A at G = 64 / 512 / 1024. **Closes ㉒** if `mm` ≤ 600 |

## 2. Where it lives

What the kernel does today, per M==1 call. LFM2 shape: K 2048, inter 1792,
N 2048, 4 active experts, 6 pool lanes (`test/htp/hvx_add_f32.c:113-118`:
`n_hvx − 1` workers plus the caller).

* `hexkl_mm_u8i4_moe.c:1100` stage A: `hvx_worker_pool_run(moe_m1_pair_worker)`
  over 4 × 56 = **224 units**. Each unit is gate column j plus up column
  56 + j (`:474-495`, the calls at `:484-488`), each column is 64 k-tiles
  × 512 B = 32 KiB at stride 112 × 512, then a fused dequant + SwiGLU.
  Stage B at `:1103` (4 units, one requant per expert, `:498-516`, owned
  by #95). Stage C at `:1106` is 4 × 64 = **256 units** of one down column
  each (`:518-537`, call at `:529`), 56 k-tiles × 512 B = 28 KiB at stride
  64 × 512. `MM` is the wall of A + C (`:1099-1107`). `SWIGLU` is Σ lane
  time inside A and C (`moe_tail_probe_add`, `:326-332`). Lanes take
  contiguous expert-major slices (`moe_m1_slice`, `:466-471`).
* Per column, `hvx_gemm_u8i4_wh_col` (`hvx_gemm_u8i4_wh.c:116-128`) first
  issues `prefetch_col` (`:38-51`). That is one 2D `l2fetch` of the whole
  strip (512 B × k_tiles), issued at `:124` directly before the loads of
  the same column, so it has no lead. The column then runs `gemm_rows4`
  (`:67-114`) once, with rows = 1.
* **Compiled inner loop at m = 1**, from `hexagon-clang -mv79 -mhvx
  -mhvx-length=128B -O3 -S` of the head file, read in this planning
  session. One k-tile iteration is **36 packets**: 32 `vrmpy`, 32 `vsplat`,
  32 `memw` activation loads, and 5 aligned `vmem` (the compiler's realign
  of the `HVX_UVector` loads). The 32 `vrmpy` rotate over the four
  accumulators v5, v4, v3, v2, so no `vrmpy` has its accumulator produced
  by the packet before it. The V79 HVX PRM (80-N2040-61 AB §5.6, "Avoiding
  accumulator stalls") names exactly that pattern as an accumulator stall.
  Per §5.1, `vsplat` (full 32-bit Rt) and the multiplies share slots 2/3.
  So 32 + 32 slot-2/3 ops per tile set a floor of ≥ 32 packets, plus up to
  32 accumulator stalls. Three quarters of it serves rows that are never
  stored (`:104-113`).
* **Per call:** 22.0 MB = **43,008 WH tiles** (7,168 per lane).
  1.38 M `vrmpy`, 1.55 M packets.
  The supervisor's count in the issue (8 `vrmpy` per *tile*) is 8 per
  *quarter-tile*. The dead work is 4× what the issue estimates.

**What changes (the diff stays inside these, per the #95 coordination):**

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/hvx/hvx_gemm_u8i4_wh.c` | new `gemm_row1` beside `gemm_rows4` (`:53-114`). The chunk loop at `:125-127` calls `gemm_row1` when `m − r0 == 1`. New `hvx_gemm_u8i4_wh_prefetch` (a box of `n_tiles` adjacent columns; the existing `prefetch_col` with width `n_tiles × 512`). New `hvx_gemm_u8i4_wh_col_nopf` (the column without its own `l2fetch`). `hvx_gemm_u8i4_wh_col` becomes prefetch(1) + nopf, so the prefill tail (`:342-344`, `:385`) is unchanged |
| `.../hvx/hvx_gemm_u8i4_wh.h` (`:13-53`) | declare the two functions. Document `gemm_row1`'s accumulation order (same `kt` → `g` → lo → hi into one accumulator, same `>> 4`) under the existing bit-identity paragraph |
| `.../hmx/hexkl_mm_u8i4_moe.c` | only the stage A/C workers `moe_m1_pair_worker` (`:474-495`) and `moe_m1_down_worker` (`:518-537`), plus the path comment (`:405-437`). The lead lives under `HVX_GEMV_PF_LEAD_KB` (default 64; 0 = today's per-column self-prefetch). `:417` "four HVX contexts" becomes "the pool's lanes (`n_hvx − 1` workers + the caller, 6 on v79)". The `ponytail` at `:434-437` keeps only the arena feed (㉒ feed half, #100). **Not touched:** stage B `:498-516`, the requant sites, `moe_m1_expert`/`moe_m1_ctx` (`:443-463`), the dispatch `:807`, the HMX loop |
| `test/htp/host/gemv_native_check.c` (new) + `run_host_checks.sh` | the real kernel compiled on x86 against the SDK's HVX emulation `$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/libnative` (step 1) |
| `test/htp/host/moe_layer_host_check.c` | stand-ins for the two new symbols (`:76-104` is the pattern). The `hvx_worker_pool_run` stand-in (`:234-242`) runs `func(n, i)` for i < n = min(n_units, 6) in sequence instead of `func(1, 0)`. Prefetch-coverage assertions (step 2) |
| `test/unittest/unittest_hvx_mm_u8i4.cpp` | new `TEST_F(HmxMmU8I4Layer, MoeM1GemvFeedVsCompute)` next to `MoeLayerM1GemvMatchesHmx` (`:2280-2405`), sharing its setup through one static helper (step 3) |

**Consumers checked, none move:**

* IDL `test/htp/nntr_hvx.idl` and its stub (`generate_stub.sh`): no new
  entry. The bench uses the existing `mm_u8i4_moe_layer_timed`
  (`:331-340`) and `moe_set_opts` (`:449`).
* `HtpComputeOps` (`htp_compute_ops.cpp`): unchanged, so the app is
  byte-for-byte the head's source.
* `nntr_quantize_stream` format tag and loader check: WH layout unchanged.
* `NNTR_HTP_PROFILE` stage tables (`HTP_MOE_T_*`, `htp_compute_ops.cpp:146-184`)
  and `tools/htp_fc_report.py`: no timer added or moved. `MM`/`SWIGLU`
  keep their meaning.
* `test/htp/build.sh` `SRCS` (`:89`): no new DSP `.c`.
* `test/jni/Android.mk` (`:930-936`): same gtest source file.

## 3. Design

### 3.1 B1 — `gemm_row1` (expected to matter most)

This is `gemm_rows4`'s `acc0` lane alone: one accumulator, 2 `vsplat` + 2
`vrmpy` per quarter-tile vector, `memw` of `a0[2g]`, `a0[2g+1]` only, and
the same `>> 4` store. The accumulation order is literally `acc0`'s, so the
int32 equals `gemm_rows4`'s row 0 by construction, and the HMX's by the
header's existing argument. A scratch prototype of this loop compiled with
the same flags gives **11 packets per k-tile** (8 `vrmpy`, 8 `vsplat`, 8
`memw`, 4 `vmem`), against 36. The single chain is the PRM's non-stalling
pattern wherever two `vrmpy` land in adjacent packets. The prototype left
about 3 gaps per tile.

**Budget.** 7,168 tiles/lane × (36 → 11) packets = 258 k → 79 k packets
per lane per call. Converting that to µs needs the per-context issue rate.
The PRM says contexts are "time interleaved to share the hardware"
(§5.6), so the rate falls as more lanes run, and it has never been
measured here. Two readings bracket it:

* *Issue-bound.* A's lanes were busy 96 % of 975 µs, which fits 258 k
  packets at ≈ 276 M packets/s per lane: 1.65 GHz ÷ 6 contexts,
  consistent with the 1.74 GHz pcycle clock `hvx_impl` read on this SoC.
  B1's compute is then ≈ 290–370 µs and `mm` falls until the feed binds:
  22 MB at 25–37 GB/s = 600–880 µs.
* *Feed-bound.* Each context issues every cycle, so A's compute was
  ≈ 160 µs, lanes waited on loads, and B1 saves ≤ 100 µs.

The microbench (§3.4) decides between them in the same sitting.

### 3.2 B2/B3 — `l2fetch` lead per lane (`HVX_GEMV_PF_LEAD_KB`)

Facts from the V79 PRM (80-N2040-60 AA, "Software-based l2fetch", p. 82):

* `l2fetch` is non-blocking and queued up to 3 deep per thread; a 4th
  stalls the thread until the oldest completes.
* It fetches only lines missing from L2, at lower priority than demand
  loads.
* A zero field cancels.
* The HVX PRM (§"L2FETCH") advises issuing it "at least several hundred
  cycles prior to using the data".

Today each column's box is issued 0 cycles before use, so between columns
no fetch is in flight.

Design:

* The worker turns its slice `[lo, hi)` into blocks of D consecutive
  units, never crossing an expert boundary.
  D = max(1, `HVX_GEMV_PF_LEAD_KB` × 1024 / unit_bytes). Unit_bytes is
  64 KiB in stage A and 28 KiB in stage C, so the default 64 KB gives
  D = 1 in A and D = 2 in C.
* A lane's consecutive units are adjacent 512 B tiles, so a block is one
  box per weight half: in A, gate columns `[j, j+D)` and up columns
  `[56+j, 56+j+D)`, width D × 512, height 64, stride 112 × 512; in C, one
  box width D × 512, height 56, stride 64 × 512. All fields fit the 16-bit
  Rtt fields.
* Prologue: boxes for blocks 0 and 1. Before computing block b: boxes for
  block b+1. The columns run through `hvx_gemm_u8i4_wh_col_nopf`, so no
  column re-issues its own box. Otherwise the 3-deep queue would stall the
  thread on every second column.
* At most two blocks are in flight per lane: ≤ 128 KiB (default) or
  ≤ 384 KiB (192 KB variant) × 6 lanes. L2 size is implementation-defined
  and not measured. B3 = 192 KB is where eviction would show, as B3 slower
  than B2.
* A hint only. It changes no byte, and `PF_LEAD_KB=0` restores today's
  behaviour exactly.

### 3.3 Lane split: decided, not a variant

* **Idle lanes are a small ceiling.** `swiglu / mm = 5.75 / 6` in #94 s2 C
  means the lanes were idle ≈ 4 % of `mm`, ≈ 40 µs. That is below the
  ±5 % (±49 µs) band this A/B can resolve.
* **Balance is already fine.** Static balance is 224/6 and 256/6 units:
  37/38 and 42/43 per lane, ≤ 2.7 % apart.
* **The slices stay contiguous and expert-major.** B2's per-lane box
  depends on that.
* Stage B's 4 units on 6 lanes are #95's code.
* The only lane change is the stale comment (§2).

**Rejected:** dynamic unit claiming (an atomic counter per unit). It
removes a measured-nil imbalance and breaks the per-lane box of adjacent
columns B2 needs.

### 3.4 Ride-along microbench (cheap: host code only, no DSP or IDL change)

`MoeM1GemvFeedVsCompute` calls `nntr_hvx_mm_u8i4_moe_layer_timed` at
M = 1, `moe_set_opts(1)`, 1 warm-up + 20 reps.

* Per rep it reads `stage[10]` = MM, `stage[2]` = Σ lane µs and
  `stage[29]` = PATH, all from 30 slots, the same indexing as
  `unittest_hvx_dma_probe.cpp:515-545`.
* It asserts `PATH == 1` and `AEE_SUCCESS` and makes no performance
  assertion.
* Per cell it prints `U8I4_FIELD path=m1_bench cell=<c> mm_us= lane_us=
  lanes= ns_per_tile= gbps=`, where ns_per_tile = lane_us × 1000 / tiles
  and gbps = tiles × 512 / mm_us / 1000. A real-shape `gbps` above 150
  prints `INVALID` (rule 12).

Cells:

| cell | weights | what it isolates |
|---|---|---|
| `arena` | real shape, 4 experts, arena handles over uncached rpcmem (production feed) | the in-situ number the profile row sees, without the rest of the layer |
| `heap` | real shape, the same 4 experts' heap handles (DSP `malloc`, cached) | whether the DSP mapping of the ION arena costs anything against a cached mapping (plan 80 §5's open question) |
| `hot` | K 2048, inter 128, N 256, one heap pair shared by all 4 experts: 272 KiB, L2-resident after the warm-up | the compute floor with no DDR in the loop: the same k-tile loop and epilogue per unit as stage A |

Under each skel the three cells give one row. A and B1 together are the
issue's "4-acc vs 1-acc" axis. How to read it:

* `hot` ≈ `arena` in ns/tile under A: A was issue-bound, and B1 is the
  lever.
* `hot` ≪ `arena` under B1/B2: the remainder is feed. `arena` − `hot` is
  the feed half's budget.
* `heap` ≪ `arena`: the ION mapping is the feed problem. A cacheable
  mapping becomes a cheaper feed fix than VTCM staging.

**Not included: the issue's "(ii) VTCM copy" cell.** It needs DSP code in
`hexkl_mm_u8i4_moe.c`'s M=1 block: a copy of each expert's `wh_bytes` into
VTCM under a new `moe_set_opts` bit, plus a VTCM budget check against the
region below `config_off`. That is outside the A/C glue #95 leaves us, and
it is the feed half's first step. `hot` stands in for it. An L2 hit is no
faster than VTCM, so `hot` is an upper bound on the VTCM floor, and a
`hot` ≪ `arena` conclusion holds for VTCM too.

### 3.5 Rejected alternative (compute)

Dropping the 8 `vsplat` per tile with the vector × scalar form
`Vx.w += vrmpy(Vu.ub, Rt.b)` does not work. The V79 PRM lists only
`ub`-vector × `b`-scalar, while our vector is the signed weight and the
scalar the unsigned activation. Flipping signedness needs a zero-point
shift and a colsum correction, which changes the int32 intermediate:
review-list violation, not bit-identical to the HMX.

Splitting the chain across two accumulators is rejected too: the issue
requires the order unchanged, and it recreates the PRM accumulator stall.

### 3.6 Contract and doc-45 rules

* Wall 1 (HMX 64-row tiles) is already off this path.
* Wall 2: this issue reads the arena directly and adds no DMA. The ring
  stays at 2 descriptors per call, the copy-out.
* No VTCM, arena or heap growth. Per-lane block state is on the lane's
  stack. There is still no CPU fallback for `QS4CX_WH`.
* No quantizer input changes, so no `_det` work.
* Bit-identity and text gates are in §1.

## 4. Steps

Commits follow `AGENTS.md`: `[HTP]` / `[test]` / `[docs]` subjects,
`git commit -s`, the `Co-authored-by` trailer, and `clang-format-14` on
changed lines (rung 0 before every commit). The branch is
`htp/105-m1-gemv-compute` from `htp_moe`, rebased on whichever of #95 and
#105 lands first. If #95 lands first, its move of the stand-ins into
`test/htp/host/hvx_scalar_stubs.c` means step 2's stand-ins go there.

1. **Kernel + native check, red first.**
   * Write `gemv_native_check.c` (`@file`/`@brief`). It is the scalar
     reference (`wh_value` as in `moe_layer_host_check.c:47-52`) against
     the real `hvx_gemm_u8i4_wh_col` and `_col_nopf` for:
     * m = 1..16,
     * k_tiles ∈ {1, 56, 64}, n_col ∈ {64, 112}, nt ∈ {0, 1, n_col−1},
     * random bytes plus an extremes case (activation 0xFF, nibble 0x8).
   * Wire it into `run_host_checks.sh`:

     ```
     gcc -std=gnu99 -O1 -fno-strict-aliasing -DHVX_UVector=HEXAGON_Vect1024
       -I $DEFAULT_HEXAGON_TOOLS_ROOT/Tools/libnative/include -I $BACKEND/hvx
     g++ … libnative.a
     ```

     Guard it on the library's presence. Without it the script prints
     `HVX GEMV NATIVE CHECK SKIPPED` and this issue's gate is not met.
     This session confirmed that the head kernel builds and matches this
     way (`bad=0`), and that a one-token mutation of it fails (`bad=512`).
   * Commit the check against the head kernel (it passes). Then add
     `gemm_row1`, the dispatch, `_prefetch` and `_col_nopf`, and record a
     mutation run in the commit body.
   * **Gate:** rung 1 `run_host_checks.sh` (the new line and
     `ALL CHECKS PASS`), then rung 2 skel with the default flags,
     `UNDEFINED SYMBOLS OK`.
2. **Glue + lead coverage check.**
   * The two workers get the block/box lead under `HVX_GEMV_PF_LEAD_KB`,
     and the comments of §2 are fixed.
   * The host stand-ins log prefetch boxes and nopf columns per lane
     (`g_lane` set by the new pool stand-in). They assert:
     * every nopf column was covered earlier by a box from its own lane,
       issued at most two blocks before;
     * no box leaves `[0, n_col)` or crosses an expert;
     * Σ box columns = Σ nopf columns (no double fetch, no over-fetch).
   * `run_host_checks.sh` compiles `moe_layer_host_check` three times, with
     `-DHVX_GEMV_PF_LEAD_KB=0`, default and `=192` (≈ 15 s each).
   * **Gate:** rung 1 in full. This worktree has no `build/` yet, so run
     `meson setup build` with the gates-skill flags first and generate the
     tiny fixture. Then rung 2 three times from one commit:
     * `HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=0` → `libnntr_hvx_skel.B1.so`
     * default → `.B2.so`
     * `HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=192` → `.B3.so`

     Each needs `UNDEFINED SYMBOLS OK` and an md5.
   * **A skel:** `git worktree add --detach <path> origin/htp_moe`
     (8beec31d or the head at build time), `HEXKL_SDK_VER=6.4.0.1
     ./test/htp/build.sh` → `.A.so`.
3. **Microbench gtest** (§3.4). **Gate:** rung 3 `ndk-build
   unittest_hvx_mm_u8i4`, md5.
4. **App.** One set, because the app source equals the head's:
   `build_android.sh --htp` (`--cache`), the `readelf -d` NEEDED lines,
   md5s (rung 3). **Gate:** rung 3 pass lines. Then open the PR into
   `htp_moe` (`Closes #105` only after the verdict; `state:review` →
   `state:needs-measurement`).
5. **Device measurement (unavoidable; the user runs it right after
   handoff 88).** Write `docs/measurements/105-m1-gemv-compute.md` with
   the `hexagon-handoff` template:
   * **Variants (4, one app set, skels swapped).** `NNTR_MOE_HTP_M1_GEMV=1`
     is set explicitly on every cell, including A and the gtests.
     * **A** = `htp_moe` head skel (`.A.so`), run first.
     * **B1** = one-row loop only (`LEAD_KB=0`).
     * **B2** = B1 + 64 KB lead (the PR default).
     * **B3** = B1 + 192 KB lead.
   * After every `adb push` of a skel, record
     `adb shell md5sum …/libnntr_hvx_skel.so` into the log. A B cell whose
     device md5 differs from the table is void (rule 21, different
     binary).
   * **Cells.** Prompt 512 (`prompt512.txt`, same file as #94/#88), NPU
     model `q40-qs4cx-wh`, `NNTR_NUM_THREADS=8`, G = 64 / 512 / 1024 × 2
     runs per variant. Mirrored order per G: A, B1, B2, B3, then B3, B2,
     B1, A.
   * **Profiles.** One `NNTR_HTP_PROFILE=2` run per variant at G = 64
     (not for tok/s). Paste the M==1 and M>1 rows and read `mm`, `dsp`,
     `swiglu`, `blocks`, `m1_gemv` and `qos_mode`.
   * **Gtests, once per skel:**

     ```
     unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'
     ```

     plus `cat /sys/class/thermal/thermal_zone0/temp` before and after
     (rule 18).
   * **Estimate ≈ 55 min**, calibrated on handoff 88's ≈ 45 min for 14
     E2E runs:
     * 24 E2E runs ≈ 30 min
     * 4 profile runs ≈ 4 min
     * 4 gtest runs ≈ 6 min
     * install and swaps ≈ 15 min

     If time runs short, B1 and B3 drop to G = 64 × 2 + profile + gtest
     (≈ 40 min). A and B2 stay complete.
   * Set `state:needs-measurement`.
6. **Read-back** (implementer + supervisor). Verdict per §1. The PR's
   default `HVX_GEMV_PF_LEAD_KB` becomes the better of B2/B3, or 64 if
   within ±5 %. If B1 ≥ B2 by more than the noise band, set 0. The knob
   stays as a documented measurement switch, like `hvx_impl`'s
   `HTP_MM_NO_PREFETCH`. The microbench table sizes ㉒'s feed half.

## 5. Risks

| risk | how the handoff makes it visible |
|---|---|
| **Issue-bound vs feed-bound is unmeasured** (§3.1): B1 may save 300 µs or 50 | `hot` vs `arena` ns/tile under A and B1 in the same sitting. B1's level-2 `mm` against A's |
| **The arena read rate caps everything** once compute drops. 600 µs needs ≥ 36.7 GB/s from vector loads. This unit has shown 15.6–18.6 GB/s in-situ through the ring and 22.6 GB/s direct. `hvx_impl` #59's "21–25" was a row-major DMA (21.2 GB/s) and a direct W8A16 read with compute (≈ 17.8 GB/s, 88.1 MB in 8.59 Mcyc at 1.74 GHz), not a pure vector-load feed on this layout | `arena` `gbps` per skel. B2/B3 vs B1 shows how far lead moves it |
| **DVFS**: less HVX work per call can lower the DSP's clock vote | `qos_mode` in each level-2 run. `hot` ns/tile would rise under B with unchanged code, which flags a clock change |
| **Thermal drift** inside the ≈ 55 min sitting | mirrored run order. `thermal_zone0` around each gtest. Verdict read from the profile `mm` (rule 20: kernel columns move ≤ 4 % while tok/s drifts) |
| **Stale or wrong skel** on the phone: all four print the same `moe m1 gemv: on` line | per-push device `md5sum` in every log. The gtest's `bit_identical` per skel |
| **L2 eviction by a too-long lead** | B3 < B2. The native check cannot see it (l2fetch is compiled out on x86) |
| **Host vs device gap of the bit-identity proof.** libnative emulates instruction semantics, not the l2fetch, the pool concurrency or timing | device gtest `MoeLayerM1GemvMatchesHmx` under every B skel. The routing M = 4 {4,3,2,1} covers `gemm_row1` and `gemm_rows4` in one call |
| **Address-space budget** | unchanged: no allocation added. The per-worker stack (`hvx_worker_pool.c:40`) holds a smaller frame than today's `gemm_rows4` |
| **Rebase against #95** (same file, stage B, and its host-stub move) | the diff is limited to `:405-437`, `:474-495`, `:518-537`. The host checks re-run after the rebase |

## 6. Docs to update (after the sitting)

* `docs/htp_moe/BENCHMARK.md`:
  * Results: 24 rows (A/B1/B2/B3 × G × run, unit-tagged, `= A text`).
  * Artifacts: 4 skels + the app set, md5 and commit.
  * Goals "now": the best-lever cell if B2 passes.
  * History: one line.
* `docs/htp_moe/LEDGER.md`:
  * ㉒ compute half: verdict, the level-2 `mm`/`dsp` per variant, the
    microbench table, and the feed half's budget = B's `arena` − `hot`.
    Correct the "8 `vrmpy` per tile" arithmetic to 32 per tile and note
    the accumulator-rotation stall.
  * ⑯: the path's new `mm`.
  * §1: a rule if the device contradicts §3.1's bracket (e.g. "at m = 1
    the u8×i4 GEMV was issue-bound: X ns/tile hot vs Y arena").
  * §3a tooling: "the real HVX kernels build on x86 against
    `Tools/libnative` (`-DHVX_UVector=HEXAGON_Vect1024`, link with g++)",
    since other M=1 kernels can reuse that host rung.
