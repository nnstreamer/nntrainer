# 117 — M=1 GEMV feed: stage each expert's `wh_bytes` into VTCM by DMA (the certified `f2` shape) under the one-row loop

Issue #117 (p1, tracker #76, LEDGER ㉒ feed half, rules 26–33). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. Ground truth read for
this plan: `docs/measurements/100-dma-chunk-list.md` (the `f2` cell, rule
29, the anchor cell), `docs/plans/100-dma-chunk-list.md` §3.2,
`docs/plans/113-m1-gemv-lead-matrix.md`, the filled
`origin/htp/113-m1-gemv-lead-matrix:docs/measurements/113-m1-gemv-lead-matrix.md`
(`d03c8929`). **Base: PR #115's branch `htp/113-m1-gemv-lead-matrix`**
(code `5be9c4ba`, one-row loop + lead machinery, runtime knobs, `applied`
echo, `gemv_native_check.c`, the swept `MoeM1GemvFeedVsCompute`). Every
`path:line` below was read on that branch. The D192 default flip (rule 33)
is not on the branch yet; §4 step 0 says what that means for the A cell.

**What `f2` is (the plan/handoff wording resolved).** Plan 100 §3.2 says
"2 descriptors per expert, whole matrices, contiguous, two buffers"; the
handoff line says `dst=strided`. Both are the same list:
`nntr_moe_dma_columns(... gu_row = (2·inter/32)·512, dn_row = (N_out/32)·512,
DST_STRIDED)` (`test/htp/nntr_moe_dma_plan.h:492-498`) makes one column as
wide as the matrix, so `dst_stride = src_stride = row_size` and each
descriptor is one contiguous copy: gate_up 57 344 B × 64 rows (3.5 MiB),
down 32 768 B × 56 rows (1.75 MiB). 8 descriptors per call, depth 2
(`nntr_moe_dma_depth2`, push k+1 then wait k). That list read **31.7 GB/s
alone and 31.6 with HVX streaming VTCM** (`f2` / `f2_load`, 695.5 / 697.6
µs per 22.02 MB). It is exactly the push `moe_push_weight_chunk`
(`hexkl_mm_u8i4_moe.c:133-152`) already issues when `cn = n_col`.

## 1. Goal and gate

Acceptance (from the issue), made measurable. One sitting, one app set,
one skel; `NNTR_MOE_HTP_M1_GEMV=1` explicit in every cell.

| gate | read from | pass |
|---|---|---|
| **verdict** | `[HTP-PROFILE]` level 2, G = 64, M==1 row, `mm` (`min`, rule 2) | **B `mm` ≤ 760.0 µs/call** (A ≈ 937); B `dsp` lower than A's by the same amount ± 20 µs (nothing else moved) |
| **decode** | E2E, mean of the two mirrored runs, same sitting | B ≥ A at G = 64, 512 **and** 1024 |
| **text** | every B log vs the A log of the same G and run, banner and md5 lines excluded | byte-identical (`NNTR_L2_DIFF` and text-vs-CPU are n/a for `QS4CX_WH`) |
| **accuracy (a), device** | `unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'` | `bit_identical value=yes`, `bad_elems_M1/M4 = 0`, now over the (loop, lead) matrix **× feed ∈ {arena, vtcm}**; the new `cell=vtcm` lines present, no `INVALID` |
| **accuracy, host** | `bash test/htp/host/run_host_checks.sh` | `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (… feed=arena,vtcm)`, new `M1 GEMV VTCM FEED SCHEDULE OK`, `HVX GEMV NATIVE BIT-IDENTICAL`, `ALL CHECKS PASS`, `WORKER POOL LANES OK` |
| **path proof** (rule 21) | every log | A: `[HTP] moe m1 gemv: on (applied=0x103c1) lead=192KB rows1=1 feed=default`; B: `(applied=0x303e1) … feed=vtcm`; level-2 M==1 row `blocks=0 m1_gemv=1408/1408`, B's row additionally `feed=1408/1408` and a `DMA ring: desc=10/call` line; M>1 row `m1_gemv=0/23 feed=0/23` |
| **prefill (standing)** | level-2 M>1 `dsp`, prompt-512 prefill tok/s | M>1 `dsp` within 2.5 % of A; prefill tok/s ≥ −5 % of A under the mirrored-order rule (rule 27) |
| **DMA anchor** (rules 30/32) | `unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*'`, cold, first thing | the `DMA_REPLAY workers=1 load=0 pace=0` line verbatim; ≈ 724 µs / 31.2 GB/s is the expected band, and the verdict is read *scaled by it* if it moves |
| **in-situ feed rate** | B's level-2 M==1 `DMA ring:` line, `engine lo..hi GB/s` | **lower bound ≥ 28 GB/s**. This is the number the issue's "`weight DMA:` line ≥ 28" means: the `weight DMA:` line's "averaged over the call" divides by `dsp` (`htp_compute_ops.cpp:626-640`) and will read ≈ 27 even at the gate; the ring line's `busy` bracket is the engine |
| host / skel / app | rungs 0–3 of `.claude/skills/hexagon-gates` | as printed there |

`mm` > 760 with the anchor at ≈ 31 GB/s closes ㉒ for good (issue text);
the path then does not land and the branch keeps only the host check and
the `vtcm` microbench cell as the record.

## 2. Where it lives

| file | change | verified at |
|---|---|---|
| `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c` | `moe_m1_expert` gains `const uint8_t *gu_wh, *dn_wh` (the bytes the workers read: the arena when the feed is off, the VTCM slab when on); `moe_m1_ctx` gains `feed`, `u_lo`/`u_hi` (the unit range of one pool run; 0..all today). The two workers slice `[u_lo, u_hi)` instead of `[0, n_active × per)`, read `e->gu_wh` / `e->dn_wh`, and take `_col_nopf` with no box at all when `feed` (§3). The M=1 block issues the eight pushes and the per-expert pool runs (§3 schedule), counts `HEXKL_PROBE_DMA_KB`, sets `DMA_FIRST(_KB)` on the first wait, and tags pushes/waits for the #87 trace | `:449-470` (structs), `:522-562` and `:591-621` (workers), `:840` (arena), `:893-896` (`use_m1`), `:1131-1216` (the M=1 block), `:133-152` (`moe_push_weight_chunk`, reused as is with `nt0 = 0, cn = n_col`) |
| `.../hmx/hexkl_mm_u8i4_moe.h` | `HEXKL_MOE_FLAG_GEMV_FEED_SET 0x20u` (bit 5), `HEXKL_MOE_FLAG_GEMV_FEED 0x20000u` (bit 17), `HVX_GEMV_M1_FEED` compile-time default (`0u` until step 6), added to `HEXKL_MOE_FLAGS_KNOWN`, `hexkl_moe_flags_feed()` beside `_lead_kb` / `_rows1` | `:131-207` |
| `nntrainer/tensor/htp_backend/htp_moe_opts.h` | the same three constants restated; `htp_moe_opts_flags` gains `feed_env` (`NNTR_MOE_HTP_GEMV_FEED`, unset = build default, `0`/`1`) | `:26-38`, `:66-88` |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `sendMoeOptsOnce` reads the fourth env and prints `feed=default|arena|vtcm`; the M==1 row prints `feed=<calls>/<calls>` from a new stage slot; the `weight DMA:` branch already prints when `dma_first_us != 0`, so B's row gets it for free and the `n/a (direct arena read…)` branch stays for A | `:1113-1163` (banner), `:596-660` (row and `weight DMA:`), `:660-695` (`DMA ring:` line, unchanged) |
| `test/htp/nntr_hvx_mm_u8i4.c` | `moe_set_opts` mask widens through `HEXKL_MOE_FLAGS_KNOWN` (no edit if the macro is the mask — it is, `:1068`); the timed entry copies the feed count into a stage slot; `MOE_N_STAGES` stays 30 if a free slot exists, else 31 with the gtest's `kMoeStages` moved with it | `:1057-1071`, `:980-1000` |
| `test/htp/host/moe_layer_host_check.c` | the DMA stand-in already copies on push (`:218-243`) and the pool stand-in runs lanes serially (`:341-360`), so the existing bit-compare covers the feed's data path; add: a scoreboard that records each push's VTCM `[dst, dst+bytes)` and each worker read's source range, asserting (i) every read hits a slab whose push was waited, (ii) no push lands on a slab a not-yet-finished pool run reads, (iii) every slab end ≤ `arena`, (iv) with `feed` the PF counters read `g_col_n = g_pf_cols = 0`, `g_nopf_n = cols`; `run_m1_cases` sweeps `feed ∈ {0,1}` over the existing matrix | `:655-760`, `:847-900` |
| `test/htp/host/moe_opts_host_check.c` | the fourth variable: unset / `0` / `1`, and that it does not touch the other two knobs | `:34-60` |
| `test/htp/host/run_host_checks.sh` | the new pass line in the first block's expectations; `-DMOE_TAIL_MAX_ROWS=16u` unchanged | `:29-40` |
| `test/unittest/unittest_hvx_mm_u8i4.cpp` | `MoeGemvOpts` gains a `feed` argument; `MoeLayerM1GemvMatchesHmx` loops the pairs × feed; `MoeM1GemvFeedVsCompute` gets a `cell=vtcm` line set: `arena` weights with the feed on, at (rows1 1, lead 0) and (rows1 1, lead 192) and (rows1 0, lead 0) — three lines, plus `feed=` on every `m1_bench` line | `:2364-2376`, `:2384-2465`, `:2489-2590` |
| `docs/measurements/117-m1-gemv-vtcm-feed.md` | the handoff (§4 step 5) | new |

**Consumers checked; none move.** IDL `test/htp/nntr_hvx.idl` `moe_set_opts(in uint32 flags, rout uint32 applied)` carries the whole word — no IDL edit, no `generate_stub.sh` run, no new entry (the `_timed` entry's `stage_us` is a `sequence<uint32>` of `MOE_N_STAGES`, also unchanged in shape); `HtpComputeOps` moves only in the two places above; the quantizer's format tag and the loader check are untouched (no layout change: the DMA copies the expert's WH blob byte for byte, as the HMX path does, `htp_compute_ops.cpp:2344` placement is 4 KiB-aligned); `NNTR_HTP_PROFILE` stage tables keep every existing slot's meaning (`DMA_KB`, `DMA_FIRST`, the ring trace slots are *populated* on the GEMV row for the first time, not redefined); `tools/htp_fc_report.py` parses FC rows only and reads nothing here; `hexkl_dma_ring.{c,h}` and `hvx_gemm_u8i4_wh.{c,h}` unchanged; `test/htp/build.sh` `SRCS` unchanged (no new `.c`).

## 3. Design

### 3.1 Chosen: whole-matrix slabs, two gate_up buffers, per-expert pool runs, the lead off

**Budget (the VTCM arithmetic).** The M=1 path uses no VTCM today: every
staging area is heap scratch (`:916-1046`), so at M ≤ 4 the whole arena
`min(vtcm_size, config_off)` (`:840`) is free — doc 46 §3 puts it at ≈ 8.30
MB (8 MiB minus the HMX config block). One expert is gate_up 3 670 016 B
+ down 1 835 008 B = 5.25 MiB; two experts (10.5 MiB) do not fit, which is
why the issue says "half an expert per half-slab". The split that keeps
`f2`'s 8 whole-matrix descriptors is **by matrix, not by half-expert**:

| slab | offset | bytes | holds |
|---|---|---|---|
| G0 | 0 | 3.5 MiB | gate_up of experts 0, 2 — then down of experts 0, 1 (two 1.75 MiB halves) |
| G1 | 3.5 MiB | 3.5 MiB | gate_up of experts 1, 3 — then down of experts 2, 3 |

7 MiB = 7 340 032 B ≤ ≈ 8.3 MB arena. The kernel checks
`2·gu_bytes ≤ arena` (and `2·dn_bytes ≤ gu_bytes`, true at this shape) once
per call; a shape that fails runs the arena read as today and the
`feed=<n>/<calls>` field shows it. No heap, no address-space growth: the
slabs are the HMX path's own weight region reused (contract §2, doc 46
§41).

**Schedule (stage-major, one expert ahead, depth 2).** The caller (the
FastRPC thread, lane 0 of every pool run) issues everything on the ring;
workers never touch the ring, so the single-owner rule of
`hexkl_dma_ring` holds by construction:

```
push gu(e0)->G0, gu(e1)->G1            (before the quant scan, where the HMX path issues its first gate_up)
A(e0): wait gu(e0); pool_run(pair, units of e0)          -- G0 read, 6 lanes over 56 pairs
push gu(e2)->G0                         (G0 free: the run's join is the barrier)
A(e1): wait gu(e1); pool_run(pair, e1);   push gu(e3)->G1
A(e2): wait gu(e2); pool_run(pair, e2);   push dn(e0)->G0[0], dn(e1)->G0[1.75M]
A(e3): wait gu(e3); pool_run(pair, e3);   push dn(e2)->G1[0], dn(e3)->G1[1.75M]
B: pool_run(requant, all experts)          (unchanged)
C(e): wait dn(e); pool_run(down, units of e)   for e = 0..3
```

Eight weight pushes + the two `moe_dma_copy` pieces = `desc=10/call`, in
issue order, on one dmlinked chain, so the ring retires them in order and
every wait is `hexkl_dma_ring_wait(idx)` on the slab's own index. The
engine never idles: a slab's compute (≈ 40 µs for a gate_up, ≈ 20 for a
down, from the `hot` one-row cell's 31.25 ns/tile × tiles / 6 lanes) is
shorter than the next transfer (116 / 58 µs at 31.6 GB/s), so the next
push always lands while the previous transfer is still moving. Bound:
22.02 MB / 31.6 GB/s = 697 µs + C(e3)'s ≈ 20 µs + 6 extra fork/joins
(§5) ≈ **720–750 µs**, under the 760 gate with little to spare — which is
what the gate was set to test.

**Two experts or more than four.** `n_active ≤ MOE_M1_MAX_EXPERTS = 16`;
the schedule is the same loop for any `n_active` (slab = e & 1; downs
pair up two per slab in order). For `n_active = 1` the second push is
just absent.

**The `l2fetch` lead under the feed: off, by construction.** HVX reads
VTCM; an `l2fetch` of the arena bytes would pull the same DDR bytes into
L2 beside the DMA that is already moving them (the issue's "must not both
fetch the same bytes"), and an `l2fetch` of a VTCM address is
meaningless. So with `feed` set the workers take `hvx_gemm_u8i4_wh_col_nopf`
(no self-prefetch either) and issue no box; `c->lead_kb` is forced to 0
for the call and the banner prints the requested lead beside
`feed=vtcm` so the log still names the cell. The host check's PF
scoreboard asserts zero boxes. No `hot`-cell measurement of a harmless
lead is needed; the question is removed rather than answered.

**The loop under the feed: one-row.** Rule 31's `hot` column is the
proxy for a VTCM read: one-row 31.25 ns/tile vs four-row 71.7–73.5. The
`vtcm` microbench cell runs both loops anyway (three lines) so the sitting
records it on silicon; the E2E B cell is the one-row loop.

**Accuracy.** The GEMV reads the same WH bytes from a different address;
`hvx_gemm_u8i4_wh.c` is untouched, so the int32 sums are the HMX's own by
the existing argument (`hvx_gemm_u8i4_wh.h:33-48`). The only new failure
mode is a stale or partial slab, which is a wrong byte, not a rounding
difference: the host scoreboard catches a read before its wait or a push
onto a live slab (§2), and on the device `bit_identical` over the ×feed
matrix plus text = A catch what the scoreboard cannot model (the DMA's
own timing). No quantizer input changes, so no `_det` question (doc 45
§3). No CPU fallback question: `QS4CX_WH` still has none.

**Profile.** The waits sit *inside* the `HEXKL_PROBE_MM` bracket (the
gate reads the whole feed + compute wall) and are **not** added to
`DRAIN` / `DRAIN_DN`, because the row's residual subtracts those
(`htp_compute_ops.cpp:600-606`) and would go negative. The first wait
(`gu(e0)`) sets `DMA_FIRST` / `DMA_FIRST_KB` so the `weight DMA:` line
prints, with the same caveat the HMX path has (the push went out before
the quant scan, so "first KB took" is the exposed remainder). Pushes and
waits are traced (`hexkl_dma_trace_push` with `HEXKL_DMA_KIND_GATE` /
`_DOWN`, chunk 0; `wait_begin/end` with `SITE_GU` / `SITE_DN`), which is
what makes the `DMA ring:` line's `engine lo..hi` the in-situ rate. It
costs nothing with probing off.

**Runtime switch (one skel serves A and B).** Bit 5 = "bit 17 is
authoritative", bit 17 = feed on, the same per-knob shape as #113's two
bits, so `NNTR_MOE_HTP_GEMV_FEED` alone leaves the loop and the lead at
their defaults. A = `lead=192 rows1=1` env, feed unset → `0x103c1`,
banner `feed=default` (= the build's `HVX_GEMV_M1_FEED`, 0 on this PR);
B = the same plus `FEED=1` → `0x303e1`, `feed=vtcm`. After a passing gate
the default flips to `1u` (step 6) and `FEED=0` (`0x103e1`,
`feed=arena`) keeps the old cell reachable, rule 33's condition.

### 3.2 Rejected: one pool run per stage with workers polling the ring

Keep today's three pool runs and let each lane wait for its slab inside
the worker (spin on the descriptor's `done` bit via
`hexkl_dma_ring_is_done`) while the caller-lane issues the next push
once a per-expert atomic countdown reaches zero. It saves the 6 extra
fork/joins (≈ 10–40 µs, §5) but (i) reads DMA descriptor `done` bits from
threads that did not `dmstart` the chain — the ring's comment and the
trace use it only from the issuing thread, and whether the engine's
descriptor write-back is visible to another hardware thread's cached
line is not something a host check can prove; (ii) puts a cross-lane
countdown and a spin in the hot loop; (iii) makes buffer reuse a
concurrency argument instead of a barrier. The per-expert runs make
every reuse a join, which the host scoreboard can check exactly. If the
`vtcm` microbench cell shows `mm − 697` well above the ≈ 20 µs compute
tail, this is the named upgrade (`ponytail:` at the schedule).

Also rejected, from the issue's own constraints: re-importing the HMX
path's paired 46-descriptor chunking so that compute can start on the
first column pair (rows f + g of #94's attribution). At M=1 the compute
per slab is a third of the transfer, so finer chunks buy at most the
≈ 20 µs tail and cost the list the ledger told us not to bring back.

## 4. Steps

Branch `htp/117-m1-gemv-vtcm-feed`, **rebased on `htp/113-m1-gemv-lead-matrix`**
(PR #115's head at the time; re-rebase if the D192 flip commit lands
first). Commits per `AGENTS.md`: `[HTP]` kernel + header + opts,
`[test]` host checks + gtest, `[docs]` handoff; `git commit -s`,
`Co-Authored-By` trailer, `clang-format-14` on changed lines.

0. **A's word.** If PR #115 carries the flip (`HVX_GEMV_M1_ROWS1=1u`,
   `HVX_GEMV_PF_LEAD_KB=192u`) the A cell may leave the two env vars
   unset; if not, every cell sets `NNTR_MOE_HTP_GEMV_LEAD_KB=192
   NNTR_MOE_HTP_GEMV_ROWS1=1` so the A log prints `0x103c1` either way,
   as the issue requires. Decide once, write it in the handoff.
1. **Flag bits, env, banner** (`hexkl_mm_u8i4_moe.h`, `htp_moe_opts.h`,
   `htp_compute_ops.cpp` banner only). Extend `moe_opts_host_check.c`.
   **Gate:** rung 1 — `run_host_checks.sh` prints
   `MOE GEMV TUNE OPTS: … feed`, `ALL CHECKS PASS`; `htp_syntax_check.sh`
   exits 0. (Nothing in the kernel reads the bit yet.)
2. **Kernel: unit ranges and source pointers, feed off.** `u_lo/u_hi`,
   `gu_wh/dn_wh`, the per-expert run helper, with `feed = 0` — the
   generated schedule is today's three runs. **Gate:** rung 1 —
   `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH` unchanged,
   `M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN` unchanged (the lane slices
   must be the same as before when `u_lo = 0`); rung 2 skel, `-Wall
   -Werror`, `UNDEFINED SYMBOLS OK (46 runtime imports)`.
3. **Kernel: the feed.** The budget check, the eight pushes, the waits,
   the per-expert runs, the `nopf` selection, the probes and the trace
   tags; the host scoreboard and the `feed ∈ {0,1}` sweep in
   `moe_layer_host_check.c`. **Gate:** rung 1 — new line
   `M1 GEMV VTCM FEED SCHEDULE OK (8 pushes, 8 waits, 2 slabs <= arena,
   no l2fetch under the feed)`, `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH
   (… feed=arena,vtcm)`, plus one deliberate mutant in the check itself
   (drop one wait → the scoreboard must fail); `*Lfm2Moe*` 6/6; rung 2
   skel, md5 recorded.
4. **Profile row and gtest.** The `feed=` field on the M==1 row, the
   stage slot in the `_timed` entry, `MoeGemvOpts(lead, rows1, feed)`,
   the ×feed loop in `MoeLayerM1GemvMatchesHmx`, the three `cell=vtcm`
   lines. **Gate:** rung 1 (`ninja -C build` for the ARM-side header
   change), rung 2 (the `_timed` entry changed — the IDL did not), rung 3
   — `build_android.sh --htp --cache` **after** `ninja -C builddir &&
   ninja -C builddir install` (the #113 handoff's stale-`libnntrainer.so`
   trap), `readelf -d` NEEDED lines, `ndk-build unittest_hvx_mm_u8i4
   unittest_hvx_dma_probe`, md5s recorded, `strings libnntrainer.so |
   grep -c 'feed=vtcm'` = 1. Open the PR into `htp_moe` (`Closes #117`
   only after the verdict), `state:review` → `state:needs-measurement`.
5. **Device measurement — unavoidable; the user runs it.** Write
   `docs/measurements/117-m1-gemv-vtcm-feed.md` from
   `.claude/skills/hexagon-handoff`. One app set, one skel, the NPU model
   already on the phone (md5 check, no push), prompt 512,
   `NNTR_NUM_THREADS=8`, `NNTR_MOE_HTP_M1_GEMV=1` in every cell.
   * **Order:** (a) install + provenance ≈ 8 min; (b) **anchor first,
     cold**: `unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*'`,
     the `DMA_REPLAY workers=1 load=0 pace=0` line verbatim ≈ 2 min;
     (c) `unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'`
     ≈ 4 min — bit-identity over the ×feed matrix and the `vtcm` lines
     next to `arena` and `hot`; (d) two level-2 profiles at G = 64 (A, B)
     ≈ 3 min; (e) E2E ≈ 30 min; (f) the workstation checks ≈ 1 min.
   * **Variants (2 of the allowed 4):**

     | | loop | lead | feed | `applied` | env beyond `M1_GEMV=1` | cells |
     |---|---|---|---|---|---|---|
     | **A** (unchanged reference, first) | one-row | 192 KB | arena (build default) | `0x103c1` | `LEAD_KB=192 ROWS1=1` (or unset, step 0) | full: G 64 / 512 / 1024 × 2, mirrored |
     | **B** | one-row | (ignored) | **VTCM** | `0x303e1` | A's + `NNTR_MOE_HTP_GEMV_FEED=1` | full |

     Mirrored order per G (A, B, B, A), `thermal_zone0` at every
     checkpoint, device `md5sum` of the skel in every run, 60 s cooldown
     before each G = 1024 pair. No CPU cell (the CPU "now" stands).
   * **Expected lines** (stop signals): A's banner `feed=default`, B's
     `feed=vtcm`; an `applied` mismatch throws (stale skel, rule 21);
     B's M==1 row `feed=1408/1408` and `DMA ring: desc=10/call …
     engine ≥ 28..`; A's row `feed=0/1408`, `weight DMA: n/a (direct
     arena read…)`; both M>1 rows `m1_gemv=0/23 feed=0/23`.
   * **Estimate ≈ 50 min.** Short form (≈ 35): (b)–(d) whole, E2E with
     one mirrored pair per G (A, B at G 64; B, A at 512; A, B at 1024)
     plus a second pair at G = 64.
   * Set `state:needs-measurement`.
6. **Read-back.** Verdict per §1. Pass: flip `HVX_GEMV_M1_FEED` to `1u`
   (one line + the banner's `default` meaning), re-run rungs 1–3, the
   host check's build-default line must read `feed=1`, PR closes #117;
   the `FEED=0` cell stays as the documented measurement switch. Fail:
   ㉒ closes at what the arena allows; the PR is reduced to the host
   scoreboard + the `vtcm` microbench cell (the record) or closed, the
   supervisor's call.

## 5. Risks

| risk | how the handoff makes it visible |
|---|---|
| **DMA rate below `f2`'s 31.6** on the day (rule 30's drift, now believed persistent at 31–32, rule 32) | the anchor cell runs first and cold; the verdict table carries `mm × (anchor_GB/s / 31.2)` beside the raw `mm`; the `DMA ring:` engine bound of B's own row is the in-situ number and is compared with the anchor in the same sitting, never with #100's |
| **Fork/join cost of 6 extra pool runs** (a futex wake per run, "several µs a worker", `hvx_worker_pool.c` worker-loop comment; workers spin 100 µs between runs so most wakes hit spinning threads) | the `vtcm` microbench cell prints `mm_us` beside `lane_us`: `mm − 22.02 MB / engine_GB/s` is the tail + fork/join; if it exceeds ≈ 40 µs the §3.2 upgrade is filed with the number |
| **Engine idles between slabs** (a slab's compute longer than the next transfer — not at this shape, 40 < 116 µs) | B's `DMA ring:` line: `busy` ≈ `mm` means no idle; `busy ≪ mm` names the gap; `depth max` should read 2 |
| **DVFS**: less waiting changes the DSP clock vote | `qos_mode=` in both profiles; the `hot` cell's ns/tile under unchanged code flags a clock change |
| **Thermal drift** across ≈ 50 min (this unit reaches 60 °C in the G = 1024 block) | mirrored order per G, `thermal_zone0` at every checkpoint, verdict from the kernel column (`mm` moves ≤ 4 % while tok/s drifts, rule 20); the profiles run at the coolest point after the gtests |
| **Stale skel / wrong cell** | removed by construction: `applied != flags` throws on the new bit; device `md5sum` in every log |
| **Address-space / VTCM budget** | no heap growth; the two slabs are the HMX path's weight region (7 MiB of ≈ 8.3 MB); the kernel refuses the feed per call if `2·gu_bytes > arena` and the row's `feed=` count says so; the host scoreboard asserts every slab end ≤ arena |
| **A partial slab read as data** (DMA timing the host cannot model) | `bit_identical` over the ×feed matrix on the device and text = A at all three G; a single non-zero voids B |
| **Prefill regression** (`hexkl_mm_u8i4_moe.c` is shared; the ring and the VTCM region are the prefill path's) | the M>1 path executes none of the new code (`use_m1` gates it, `:893`); read as M>1 `dsp` within 2.5 % and prefill tok/s ≥ −5 % under rule 27; `feed=0/23` on the M>1 row |
| **Rebase on a moving PR** (#115 may gain the flip commit) | step 0 fixes A's word by env; the diff is confined to §2's files; host checks re-run after any rebase |

## 6. Docs to update (after the sitting)

* `docs/htp_moe/BENCHMARK.md`: Results — 12 rows (A/B × G × run),
  unit-tagged, `= A text`; a "#117 side table": the two level-2 M==1
  rows with `mm`, `dsp`, `feed=`, the `DMA ring:` engine bound, the
  anchor line, and the `vtcm` / `arena` / `hot` microbench lines;
  Artifacts — one skel + one app set with md5 and commit; Goals "now" if
  B passes and lands (then B is a lever cell, so "now" moves only once a
  later sitting's A carries it, contract §4.2); one History line.
* `docs/htp_moe/LEDGER.md`: ㉒ — the feed half's verdict (pass: closed
  with the number; fail: closed "at what the arena allows"); §1 — a rule
  on what the in-situ feed rate is against the anchor and the `f2` cell
  (whether a per-call schedule under a real workload reproduces the
  replay's 31.6), and one on the fork/join cost if it was visible; the
  per-token budget row re-read with the new `mm` (the projected ≈ 30.5
  ms / ≈ 33 tok/s); #114 closed as moot if the feed lands.
* Contract §1 "Levers measured" row (3): the feed's number beside D192's.
