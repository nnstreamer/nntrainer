# 113 — M=1 GEMV feed: the loop × `l2fetch`-lead matrix, and the lead swept past 192 KB

Issue #113 (p0, tracker #76, LEDGER ㉒ + rule 26). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. Ground truth:
`docs/plans/105-m1-gemv-compute.md` §3.2, the measured
`docs/measurements/105-m1-gemv-compute.md` (`origin/htp/105-m1-gemv-compute`
@ `da41340b`), LEDGER rules 26–30 and ㉒ (cycle 11). Base: **rebase on
`htp/105-m1-gemv-compute`** (PR #107, code @ `a0ae8b29`, branch head
`05cd6e07` / `da41340b`, worktree `/home/j2z0-lee/nntrainer-88-A`) — its
`gemm_row1`, `hvx_gemm_u8i4_wh_prefetch`, `_col_nopf`, the block/lead glue,
`gemv_native_check.c` and `MoeM1GemvFeedVsCompute` are all reused, not redone.

## 1. Goal and gate

**Goal (from the issue).** The four-row loop has never been measured with a
lead, and the lead was still monotonic at 192 KB. Fill the missing cells and
bring the level-2 M==1 `mm` from **980.9 µs/call** (#100 A_L2, GEMV default)
to **≤ 840 µs/call**.

| gate | where it is read | pass |
|---|---|---|
| **verdict** | `[HTP-PROFILE]` level 2, M==1 row, `mm` (`min`, rule 2) | **`mm` ≤ 840.0 µs on at least one variant** (−14.3 % vs #100 A_L2's 980.9, −9.7 % vs #105 B3's 930.9) |
| **decode** | E2E decode tok/s, mean of the two mirrored runs, same sitting | that variant's mean **≥ A at G = 64, 512 and 1024** |
| **text** | every B/C/D log vs the A log of the same G and run | byte-identical, excluding the `[HTP]` banner line. `NNTR_L2_DIFF` and "text = CPU q40" are n/a for `QS4CX_WH` (different weights) |
| **accuracy (a)** | device gtest `*MoeLayerM1GemvMatchesHmx*` under every measured (loop, lead) pair | `bit_identical value=yes`, `bad_elems_M1 = 0 of 2048`, `bad_elems_M4 = 0 of 8192`. The lead changes no arithmetic, so a single non-zero voids the variant |
| **accuracy (host)** | `bash test/htp/host/run_host_checks.sh` | `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH`, `HVX GEMV NATIVE BIT-IDENTICAL (libnative; …)`, and `M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN` **for each lead and each loop of the sweep**, `ALL CHECKS PASS`, `WORKER POOL LANES OK` |
| **prefill (standing)** | level-2 M>1 row `dsp=` and prompt-512 prefill tok/s vs A | M>1 `dsp` within **2.5 %** of A (the tie-breaker) and prefill tok/s ≥ **−5 %** of A, read with the mirrored-order rule (#105: a single cell's −5 % is position drift, rule 27) |
| **path proof** | every log | `[HTP] moe m1 gemv: on (applied=0x…)` with the **variant's** `applied` word; level-2 M==1 `blocks=0 m1_gemv=1408/1408`, M>1 `m1_gemv=0/23`, `qos_mode=2`. Otherwise the run is void |
| **host / skel build** | rungs 1–3 of `.claude/skills/hexagon-gates` | as printed there |

**Also decided by this sitting: PR #107.** If a four-row variant (B or C)
wins, #107 closes and only its lead machinery, `gemv_native_check.c` and the
microbench are kept on this branch. If D (one-row + the best lead) wins, #107
lands with that pair as the default.

**Ride-along (one line, free).** Re-read `DMA_REPLAY workers=1 load=0 pace=0`
from `unittest_hvx_dma_probe`: 561.5 µs / 40.2 GB/s (#94 s2) means rule 30's
drift is a session effect and ㉒'s feed half re-opens; ≈ 719 µs means 31.4
GB/s is real and the feed half stays closed.

## 2. Where it lives

Paths verified in this session. DSP files are quoted at their **#107 branch**
line numbers (`git show origin/htp/105-m1-gemv-compute:<path>`); ARM files at
`htp_moe`'s.

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/hvx/hvx_gemm_u8i4_wh.c:147-162` | `hvx_gemm_u8i4_wh_col_nopf` today picks the one-row loop by `m - r0 == 1u` (`:156`), so **every lead cell is forced onto `gemm_row1`** — this is the confound. Add a `rows1` parameter; `gemm_row1` is taken only when `rows1 != 0` and `m - r0 == 1u`. `gemm_row1` (`:62-82`) and `gemm_rows4` (`:98-145`) are untouched. `hvx_gemm_u8i4_wh_col` (`:164-169`, the prefill tail) passes `HVX_GEMV_M1_ROWS1` |
| `.../hvx/hvx_gemm_u8i4_wh.h:13-53` | the new parameter and the two compile-time defaults in the existing bit-identity paragraph |
| `.../hmx/hexkl_mm_u8i4_moe.h:131,141-144` | new flag bits beside `HEXKL_MOE_FLAG_M1_GEMV` (bit 0): `HEXKL_MOE_FLAG_GEMV_TUNE` (bit 7, "bits 8+ are authoritative"), lead in 64 KB units in bits [15:8], `rows1` in bit 16. `HVX_GEMV_PF_LEAD_KB` stays the compile-time default and is set to **0u** until the sitting names the winner; new `HVX_GEMV_M1_ROWS1` (default `0u` = four-row loop) |
| `.../hmx/hexkl_mm_u8i4_moe.c:463-467` (`moe_m1_ctx`), `:1159-1166` (setup), `:511-546` / `:579-613` (the two workers), `:490-493` (`moe_m1_lead_units`) | `moe_m1_ctx` gains `lead_kb` and `rows1`, resolved once from the call's `flags` (`:807`) with the macros as the default; the workers read `c->lead_kb` / `c->rows1` instead of the macro, and pass `c->rows1` down. `lead_kb == 0` keeps today's per-column self-prefetch (`hvx_gemm_u8i4_wh_col`) exactly |
| `test/htp/nntr_hvx_mm_u8i4.c:1057-1068` | `nntr_hvx_moe_set_opts` masks `flags & HEXKL_MOE_FLAG_M1_GEMV`; widen the mask to the new bits. The **echo is the variant proof**: `htp_compute_ops.cpp:1129-1138` throws when `applied != flags`, so a stale skel cannot silently run the wrong cell (rule 21's failure mode, removed by construction) |
| `nntrainer/tensor/htp_backend/htp_moe_opts.h:34-38` | `htp_moe_opts_flags` also reads `NNTR_MOE_HTP_GEMV_LEAD_KB` and `NNTR_MOE_HTP_GEMV_ROWS1` (unset = compile-time default, no `TUNE` bit). Rounding, clamping and the ≤ 127-unit bound live here, as pure functions |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1112-1142` | pass the two env strings; the banner gains `lead=<kb> rows1=<0|1>` beside `applied=0x…` |
| `test/htp/host/moe_opts_host_check.c:34-36` | the widened `htp_moe_opts_flags` (unset / 0 / 1 / lead rounding / clamp / rows1) |
| `test/htp/host/moe_layer_host_check.c:76-104,234-242` + `run_host_checks.sh:29-36` | the stand-ins take the runtime lead/rows1; the coverage assertions run over **{0, 192, 384, 768, 1536} KB × {rows4, rows1}** inside **one** build instead of three compiles of one configuration each |
| `test/htp/host/gemv_native_check.c` (#107) | add `rows1 = 0|1` to its case grid, so both loops are proved bit-identical against the scalar reference on x86 |
| `test/unittest/unittest_hvx_mm_u8i4.cpp:2447` (`MoeM1GemvFeedVsCompute`) | sweep `moe_set_opts` over the matrix and print `lead_kb=`, `rows1=`, `inflight_kb=` on every `U8I4_FIELD path=m1_bench` line. `MoeLayerM1GemvMatchesHmx` (`:2410`) loops the same pairs |

**Consumers checked; none move.** IDL `test/htp/nntr_hvx.idl:449`
(`moe_set_opts(in uint32 flags, rout uint32 applied)`) already carries a full
word — **no IDL change, no `generate_stub.sh` change, no new entry point**;
`HtpComputeOps` moves only in the two places above; `nntr_quantize_stream`'s
format tag and the loader check are untouched (the WH layout is unchanged);
`NNTR_HTP_PROFILE` stage tables (`htp_compute_ops.cpp:146-184`,
`HEXKL_PROBE_MM`/`SWIGLU`) and `tools/htp_fc_report.py` gain no timer and keep
their meaning; `test/htp/build.sh` `SRCS` gets no new `.c`; `test/jni/Android.mk`
keeps the same gtest sources.

## 3. Design

**Chosen: one skel, one app, the (loop, lead) pair selected at runtime through
the existing `moe_set_opts` word.** The lead is arithmetic done once per block
of columns (32 KiB of weights each), so a runtime value costs one load and one
compare per block and nothing in the `vrmpy` loop; the loop shape costs one
predictable branch per column. In exchange the whole 2 × 5 matrix runs in a
**single gtest invocation** (`MoeM1GemvFeedVsCompute`, ≈ 30 ms per matrix cell),
the E2E variants are env-var cells on one binary set, and the `applied` echo
proves per log which cell ran. Contract §2 and doc 45 §3 hold: no DMA, no ring,
no VTCM, no arena or heap growth (per-lane block state is on the lane's stack),
no CPU fallback for `QS4CX_WH`, no quantizer input touched (`_det` n/a), and the
int32 is unchanged by construction (`l2fetch` is a hint; `gemm_row1` is
`gemm_rows4`'s `acc0` in the same order).

**Rejected: keep `HVX_GEMV_PF_LEAD_KB` compile-time and build a skel per cell.**
It is the smaller diff, but the matrix is 2 × 5 = 10 skels; at ≈ 2 min per
push + gtest that is ≈ 16 min of the sitting for the sweep alone, pushing the
sitting past 60 min, and every swap re-imports the stale-skel failure mode
(rule 21, #97) that the `applied` echo otherwise removes. The compile-time
constants stay as the **defaults** so the landed binary is fixed at one pair.

**Why this cell is expected to win (issue table).** At lead 0 the four-row loop
is 111.72 ns/tile against the one-row loop's 145.23 (−23 %), because 8 `vrmpy`
per quarter-tile keep more loads in flight against DDR latency (rule 26); the
lead attacks the same latency from the other side and bought −17.2 % on the
loop that needs it most (145.23 → 120.23 at 192 KB), monotonically and without
saturating. Nothing has put the two together.

**The sweep's bounds (the L2 budget check).** In stage A a unit is
`2 × k_tiles × 512` = **64 KiB** (gate + up), so `moe_m1_lead_units`
(`:490-493`) gives `d = LEAD_KB / 64` units and the bytes a lane has in flight
are exactly **`LEAD_KB`** (two boxes of `d` columns); over 6 lanes
(`hvx_add_f32.c:113-118`) that is **1.1 / 2.25 / 4.5 / 9.0 MB** at 192 / 384 /
768 / 1536 KB. The top cell is deliberately past any plausible v79 L2, so the
turning point is **inside** the swept range and the issue can record it. Two
hard bounds the host check asserts instead of measuring:

* the `l2fetch` Rtt fields are 16 bits each (`hvx_gemm_u8i4_wh.c:35-40`):
  width `d × 512 < 65536` ⇒ `d ≤ 127`; strides 112 × 512 and 64 × 512 and
  heights 64 / 56 are fixed and fit. 1536 KB is `d = 24` (stage A) and
  `d = 54` (stage C, unit 28 KiB) — safe;
* `moe_m1_block_end` (`:479-487`) clamps every box to the lane's slice and to
  one expert, so the lead **saturates structurally** at ≈ 2368 KB in stage A
  (37 units of a 224/6 slice) and ≈ 1176 KB in stage C (42 of 256/6). The
  1536 KB cell is therefore already "fetch the rest of my slice"; a cell above
  it would be a no-op and is not swept.

**Expert-boundary crossing is out of scope, to #114.** `ex[e].g->wh_bytes` are
per-expert base pointers for arbitrarily routed experts, not one contiguous
run, so a box cannot span two experts: crossing means issuing a *second* box on
the next expert's base and tracking it per lane — more than the one-line change
the issue allows, and it would add variants to a sitting that is already full.
The issue's own fallback applies: it stays in **#114**.

## 4. Steps

Commits per `AGENTS.md` (`[HTP]` / `[test]` / `[docs]`, `git commit -s`, the
`Co-authored-by` trailer, `clang-format-14` on changed lines). Branch:
`htp/113-m1-gemv-lead-matrix`, **rebased on `htp/105-m1-gemv-compute`**
(`05cd6e07`), not on `htp_moe`.

1. **Rebase and split the two knobs.** Rebase; add the `rows1` parameter to
   `hvx_gemm_u8i4_wh_col_nopf` and the `HVX_GEMV_M1_ROWS1` default; extend
   `gemv_native_check.c` over `rows1 ∈ {0, 1}`.
   **Gate:** rung 1 — `run_host_checks.sh` prints
   `HVX GEMV NATIVE BIT-IDENTICAL`, `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH`,
   `ALL CHECKS PASS`, plus one mutant caught per loop; then rung 2 (skel,
   `-Wall -Werror`, `UNDEFINED SYMBOLS OK`).
2. **Runtime lead and loop through `flags`.** The flag bits, the widened
   `moe_set_opts` mask, `moe_m1_ctx.lead_kb/rows1`, the two workers, the
   `htp_moe_opts.h` env parsing (with the `d ≤ 127` clamp) and the banner.
   The host stand-ins loop the ten configurations and assert coverage, no
   over-fetch, no box outside `[0, n_col)` or across an expert, and the
   16-bit field bound.
   **Gate:** rung 1 in full (`M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN` once
   per configuration; `*Lfm2Moe*` 6/6; `tools/htp_syntax_check.sh` exit 0),
   then rung 2 — one skel, md5 recorded.
3. **Microbench + bit-identity sweep** in `unittest_hvx_mm_u8i4.cpp`: the
   matrix on the `arena` and `hot` cells, `heap` only at (rows4, 0) and
   (rows1, 192) as rule 27's control.
   **Gate:** rung 3 `ndk-build unittest_hvx_mm_u8i4`, md5.
4. **App.** One set (`build_android.sh --htp`, `--cache`), `readelf -d` NEEDED
   lines, md5s. **Gate:** rung 3 pass lines. Open the PR into `htp_moe`
   (`Closes #113` only after the verdict), `state:review` →
   `state:needs-measurement`.
5. **Device measurement — unavoidable, the user runs it.** Write
   `docs/measurements/113-m1-gemv-lead-matrix.md` from
   `.claude/skills/hexagon-handoff`. One app set, **one skel**, the NPU model
   `q40-qs4cx-wh` reused from the #105/#100 sitting (md5 check, no 4.3 GB
   push), prompt 512, `NNTR_NUM_THREADS=8`, `NNTR_MOE_HTP_M1_GEMV=1` set
   explicitly in every cell.
   * **Order:** (a) install + provenance ≈ 8 min; (b) the gtest sweep ≈ 6 min
     — it prints the whole 2 × 5 matrix and every pair's `bit_identical`;
     **read `L*` = the best arena lead of the four-row row** (if the column is
     still falling at 1536 KB, `L* = 1536`); (c) four level-2 profile runs at
     G = 64 ≈ 5 min; (d) the E2E block ≈ 26 min.
   * **Variants (4; contract §4.2):**
     | | loop | lead | cells |
     |---|---|---|---|
     | **A** (unchanged reference, runs first) | four-row | 0 (per-column self-prefetch) | full: G = 64/512/1024 × 2, mirrored |
     | **B** | four-row | 192 KB | full |
     | **C** | four-row | `L*` (> 192 KB) | full |
     | **D** | one-row | `L*` | G = 64 × 2 + profile + gtest only (the PR #107 tie-breaker; #105 already has one-row × 192 KB) |
   * A must reproduce the head: its `mm` within a few % of #100 A_L2's 980.9
     (a cross-sitting sanity check, rule 23 — not a gate), which is also the
     proof that the runtime knob costs nothing.
   * Mirrored order per G (A, B, C, then C, B, A), `thermal_zone0` at each
     checkpoint, device `md5sum` of the skel and the `applied=0x…` word in
     every log.
   * **Estimate ≈ 45–50 min.** Short form if time runs out: keep A and C at
     all three G, B at G = 64 and 1024.
   * Set `state:needs-measurement`.
6. **Read-back.** Verdict per §1. The landed defaults become the winning pair
   (`HVX_GEMV_PF_LEAD_KB`, `HVX_GEMV_M1_ROWS1`); the env vars stay as
   documented measurement switches, like `HTP_MM_NO_PREFETCH`. Record PR
   #107's fate and the lead at which the arena column turned.

## 5. Risks

| risk | how the handoff makes it visible |
|---|---|
| **Host cannot see the lead at all**: `l2fetch` is compiled out on x86, so the host checks prove coverage and bit-identity, never speed. Step 5 is the only place the matrix exists | the sweep prints all ten cells in one log; the verdict is read from the level-2 `mm`, not from the microbench |
| **L2 eviction by a too-long lead** (the sweep's whole point): 6 × 1536 KB = 9 MB in flight | the arena column turns non-monotonic; `inflight_kb=` on each line names the cell where it turned. `hot` (L2-resident) is swept beside it as the control |
| **The runtime knob itself costs time** (a branch per column, a non-folded lead) | A = four-row, lead 0 must reproduce the head's `mm` (≈ 980.9 / 973.0) in the same table |
| **DMA rate drift between sittings** (rule 30) and absolute GB/s (rule 28) | no absolute GB/s gate here — the gate is an in-sitting `mm` and an in-sitting decode A/B. The `DMA_REPLAY workers=1 load=0 pace=0` ride-along is the anchor cell |
| **DVFS**: less waiting can change the DSP clock vote | `qos_mode=` in all four profiles; the `hot` cell's ns/tile under unchanged code flags a clock change |
| **Thermal drift across ≈ 50 min** | mirrored order per G, `thermal_zone0` at every checkpoint, verdict read from the kernel column (rule 20: `mm` moves ≤ 4 % while tok/s drifts; #105's two opposite-order passes agreed ≤ 0.4 %) |
| **Stale skel / wrong variant** | removed by construction: `applied != flags` throws (`htp_compute_ops.cpp:1129-1138`), so a skel that ignores the new bits cannot run a cell silently. Device `md5sum` is still logged |
| **Prefill regression** (`hvx_gemm_u8i4_wh.c` is shared with the prefill tail) | with `HVX_GEMV_M1_ROWS1 = 0` the prefill path is today's `gemm_rows4` plus one branch; read as M>1 `dsp` within 2.5 % of A and prefill tok/s ≥ −5 % under rule 27 |
| **Address-space / arena budget** | unchanged: nothing is allocated, `l2fetch` is a hint over already-registered arena bytes |
| **Rebase on a held PR** (#107 may be closed by this sitting) | the diff is confined to the files of §2; the host checks and the native check re-run after the rebase, and step 6 records which half of #107 survives |

## 6. Docs to update (after the sitting)

* `docs/htp_moe/BENCHMARK.md`: Results — 20 rows (A/B/C × G × run, D at
  G = 64 × 2), unit-tagged, `= A text`; Artifacts — one skel + one app set
  with md5 and commit; Goals "now" if a variant passes; one History line.
* `docs/htp_moe/LEDGER.md`: ㉒ — the 2 × 5 matrix, the lead at which the arena
  column turns, the verdict, and PR #107's fate; ⑯ — the path's new `mm`;
  §1 — a rule for what the completed matrix says about rule 26 (whether the
  four-row loop plus a deep lead beats both diagonals), and the ride-along's
  answer on rule 30; the per-token budget row re-read with the new `mm`.
