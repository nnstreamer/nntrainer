# Plan 99: `MoeChunkReplay` content check — the replay packs its VTCM rows, the kernel strides them

Issue: #99 (p1). Base: `htp_moe` @ `08afbb10`. Evidence: `docs/measurements/94-sitting2-anchor-trace.md` @ `dad0f476` (Deviation 6, replay table), `G_replay.log`.

## 1. Goal and gate

Acceptance criterion (from the issue, made measurable):

* `unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*:*DmaProbeShapes*'` prints `[  PASSED  ] 2 tests`; all 11 `DMA_REPLAY` lines end `checksum_ok=y`, and the line's byte field is `bytes_per_call=22560768 weight_bytes=22020096` (the two numbers the sitting could not reconcile, both now printed).
* A host check in `test/htp/host/` pins the arithmetic that produced every number in the issue: 46/30 items, 22 560 768 total = 22 020 096 weights + 524 288 activation + 16 384 copies, gate/up VTCM footprint 3 670 016 B under the kernel's stride (`want_sum` 9 461 760) and 569 344 B under the replay's packed stride (`res[6]` 1 467 840). Gate: `run_host_checks.sh` prints `ALL CHECKS PASS`, plus the new check's own line `REPLAY PLAN FOOTPRINT OK`.
* Verdict on #94 rows c, d, g, h: stated in §3 below; confirmed or replaced by the ride-along gtest of the next handoff (§4 step 4).

Standing gates: prefill ≥ −5 % of variant A, generated text identical to the CPU run. This change touches `test/htp/nntr_hvx_dma_probe.c` (a test-only skel entry), a gtest and a host check — no model path, so the standing gates are carried by the handoff's variant A as usual and cannot move.

## 2. Where it lives

Verified references (worktree `/home/j2z0-lee/nntrainer-moe`, `08afbb10`):

| File | What | Lines |
|---|---|---|
| `test/htp/nntr_hvx_dma_probe.c` | `replay_ring`: `hexkl_dma_ring_push2d(c->vtcm + it->dst_off, replay_src(c, it), it->row_size /*dst_stride*/, it->src_stride, it->row_size, it->nrows, 0, 1)` — destination stride = `row_size` (packed) | `:254-256` |
| same | `replay_worker`: `d->dst_stride = it->row_size;` — same packing on the N-worker path | `:301` |
| same | VTCM bound check uses the packed extent `dst_off + bytes` | `:432` |
| same | `gu_hi = dst_off + bytes` — the packed extent again; `memset` and the 64-stride sum run over `[gu_lo, gu_hi)`; `res[6] = sum` | `:446-447`, `:465`, `:538`, `:548` |
| same | `bytes_per_call += bytes` over **all** pushes (weights + activation + copies) | `:450` |
| `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c` | the kernel's push: `push2d(vtcm_base + dst_off + off, wh_bytes + off, stride /*dst_stride*/, stride /*src_stride*/, row, k_tiles, …)` — destination keeps the source's 2D layout ("the destination keeps the same layout so the matmul's indexing is unchanged") | `:145` (doc comment `:117-121`) |
| same | `HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_KB, …)` only in `moe_push_weight_chunk` — the in-situ `weight DMA:` line counts weights only | `:143` |
| `test/htp/nntr_moe_dma_plan.h` | gate/up `dst_off = vtcm_gu + g0 * T`, down `dst_off = vtcm_dn + nt0 * T` — column offsets, correct **for a strided destination** | `:171`, `:190`, `:206` |
| `test/unittest/unittest_hvx_dma_probe.cpp` | `want_sum = 0xA5 * (gu_bytes / 64)` — assumes the kernel's footprint | `:601` |
| same | `plan_shape_ok` = push/wait **counts** only | `:538-539` |
| same | `kMoeStages = 30`; `EXPECT_EQ(res[6], want_sum)` | `:516`, `:631` |
| `test/htp/host/moe_layer_host_check.c` | already compares the kernel's trace to the plan descriptor-for-descriptor (kind, expert, chunk, row_size, nrows, src_stride, ring index) — the trace carries no destination, so a dst-stride divergence is invisible to it | `:946-975` |
| `docs/plans/87-dma-in-situ-gap.md`, `docs/plans/94-sitting-2.md` | the expectation string `bytes_per_call=22020096` | `87:247`, `94:179` |

Consumers that do **not** move: the IDL `dma_replay` signature (`test/htp/nntr_hvx.idl:437`) and its stub — `res` keeps 12 words and the schedule keeps 8 words per item; `HtpComputeOps`, the quantizer tag, the loader, `NNTR_HTP_PROFILE` stage tables and `tools/htp_fc_report.py` are untouched (the replay is a test-only entry). The skel and gtest still must be rebuilt together, as always.

## 3. Design

### Root cause (code evidence)

Three separate facts were folded into one "content check fails":

1. **`res[6]` ≠ `want_sum` — the skel replay is wrong, the host expectation is right.** The kernel DMAs each gate/up chunk into VTCM with `dst_stride = src_stride = gu_ntiles × 512 = 57 344` (`hexkl_mm_u8i4_moe.c:145`), so the eight chunk descriptors of one expert tile a 3 670 016 B window exactly once. The replay passes `dst_stride = row_size` (`nntr_hvx_dma_probe.c:254-256`, `:301`), so each chunk lands **packed** at its column offset `g0 × 512`; the eight chunks overlap and the union is `[0, 569 344)`. The skel then sums `[gu_lo, gu_hi)` with `gu_hi = max(dst_off + row_size × nrows) = 569 344` (`:446-447`), i.e. 8 896 samples × 0xA5 = **1 467 840 = `res[6]`**. The host sums the kernel's window: 57 344 samples × 0xA5 = **9 461 760 = `want_sum`**. Ratio 6.446 = 3 670 016 / 569 344. Compiled from `nntr_moe_dma_plan.h` on the host (scratchpad, same shape K=2048 I=1792 N=2048 E=4 acc_tiles=32): `gu footprint packed=569344 -> sum 1467840 ; strided=3670016 -> sum 9461760`, and the strided layout covers `[0, gu)` 57 344/57 344 samples with no duplicate. Every sampled byte the replay wrote **is** 0xA5 — the source addresses, source strides, row sizes and byte counts are the traced ones; only the destination geometry differs. Not a stage-table offset (`res` is the replay's own 12-word array, `kMoeStages` only sizes the step-0 call), not a pre-fill read (the window has no zero sample), not a race (exact integer match).
2. **`bytes_per_call = 22 560 768` — correct; the handoff's comparator was wrong.** The plan's 46 pushes sum to 22 020 096 (weights, 4 × (3 670 016 + 1 835 008)) + 524 288 (4 activation blocks × 64 × 2048) + 16 384 (copy in + copy out at 4 × K, 4 × N floats) = 22 560 768; the extra 540 672 = 64 + 2 pieces of 8 192. The in-situ `weight DMA: 21504 KB/call` counts `HEXKL_PROBE_DMA_KB` only inside `moe_push_weight_chunk` (`:143`). The `bytes_per_call=22020096` text in plans 87 and 94 was hand-written from the `weight DMA:` line. Consequence: the replay's `gbs` is over 2.5 % more bytes than the in-situ line's — an apples-to-oranges 2.5 % that the plan makes explicit by printing both.
3. **`regions=32` — by design.** `n_regions = min(slots − 1, REPLAY_MAX_REGIONS)` with `slots = 256 MiB / 5 505 024 = 48` → 32; it is the `fresh=1` rotation pool, not the number of experts (`n_experts = 4` is derived from the schedule).

`plan_shape_ok` (issue Q2) is indeed weak — it compares two counts — but strengthening it to a per-descriptor comparison (kind, e, c, row, nrows, stride from the trace words) would **still** not have caught this: the trace records no destination. What caught it was the footprint checksum, which is the right instrument; §4 step 2 tightens `plan_shape_ok` anyway because it is six lines and the words are already in `trace[]`.

### Verdict on the provisional rows (issue Q3)

The replay issued the traced 46 descriptors with the traced source addresses, source strides, row sizes and byte counts, in the traced order with the traced 30 wait points; the DDR read stream — the thing wall 2 is about — was byte-for-byte the in-situ one. What differed is the VTCM write address pattern: 8 KiB / 16 KiB rows written contiguously and overlapping (569 KB + 934 KB of VTCM touched per expert) instead of at a 56 KiB / 32 KiB stride over 3.67 MB + 1.84 MB. `DmaProbeShapes iii`, the 106.9 GB/s comparator of row h, writes its destination packed too (`dma_probe_fill`, `nntr_hvx_dma_probe.c:75`: `d->dst_stride = p->row_size`), so packing cannot be what separates 40.2 from 106.9 GB/s. **Under this diagnosis rows c, d, g and h stand as measured**: the descriptor count and the bytes moved are what the trace says (46, 22.02 MB of weights + 0.54 MB of activation/copies), and the only known difference to in situ is on the VTCM side, where the DMA engine's write is SRAM-bound and stride-insensitive at these row sizes. What is *not* known from the code is whether eight in-flight descriptors targeting overlapping VTCM lines cost anything in the engine; that is a device question, and it is answered for free by step 4: the fixed replay re-runs the same 11 cells in the next handoff, and cell 1's `us_per_call` within ±10 % of 561.5 µs confirms row h as measured, while a larger move replaces the provisional numbers with the new ones. No separate sitting.

### Chosen approach

Fix the skel replay to the kernel's destination geometry: `dst_stride = src_stride` in both replay paths, and compute VTCM extents (bound check, `gu_hi`) as `dst_off + (nrows − 1) × src_stride + row_size`. Activation and copy pushes have `src_stride == row_size`, so they are unchanged. The host `want_sum` stays as is (it was right). Print `weight_bytes` next to `bytes_per_call` on the host side, computed from the plan (sum of gate/up/down pushes), so the handoff can compare the replay's GB/s with the in-situ `weight DMA:` GB/s on the same byte basis. Add a host check that pins the footprint arithmetic, including the packed-footprint negative control that reproduces `1 467 840`.

Why this side: the replay's stated purpose (plan 87 §3.3) is "the in-situ list with no compute between"; the kernel's VTCM layout is part of that list, and after the fix the replay's footprint is exactly the kernel's `w_gu_off`/`w_dn_off` windows. It also means the corrected replay exercises the strided-write case that `DMA_PROBE` does not — if the device shows a difference between them, that is new information for wall 2, not noise.

### Rejected alternative

Fix the expectation instead (`want_sum = 0xA5 × 569 344 / 64` computed on the host from the packed extents). Smallest possible diff and it would make the test pass, but it would enshrine a replay whose VTCM geometry is not the kernel's, and the checksum would then verify only that the union of overlapping windows got written — a chunk that failed to move would be masked by the chunk on top of it. The whole point of the content check is that the footprint is written exactly once.

## 4. Steps

Each step ends at a rung of `.claude/skills/hexagon-gates`.

1. **Host check first (rung 1).** New `test/htp/host/replay_plan_host_check.c` (header-only include of `nntr_moe_dma_plan.h`, no stub) and its four lines in `run_host_checks.sh`, in the style of `dma_probe_host_check`. At the LFM2 shape with the gtest's VTCM layout (`v_gu=0, v_dn=gu, v_act=v_dn+dn, v_copy=v_act+(K/32)×2048`) it asserts, as hand arithmetic: 46 pushes / 30 waits; byte split 22 020 096 / 524 288 / 16 384 (total 22 560 768); the strided gate/up footprint `[v_gu, v_gu + 3 670 016)` and down footprint `[v_dn, v_dn + 1 835 008)` covered exactly once at 64-byte sampling (bitmap, expert 0); the strided total footprint `< 6 MiB` (so it fits under `config_off − 1 MiB`, the `load=2` limit, on an 8 MiB VTCM); and the negative control: the **packed** gate/up extent is 569 344 → `0xA5 × 8896 = 1 467 840`, the number the device printed. Prints `REPLAY PLAN FOOTPRINT OK`. Gate: `bash test/htp/host/run_host_checks.sh` → `ALL CHECKS PASS` (with the new line); `bash tools/htp_syntax_check.sh` exits 0.
2. **Skel fix (rung 2).** `nntr_hvx_dma_probe.c`: `:254-256` third argument `it->src_stride`; `:301` `d->dst_stride = it->src_stride;`; introduce `const uint32_t extent = (it->nrows - 1u) * it->src_stride + it->row_size;` and use it at `:432` (`dst_off + extent > vtcm_limit`) and `:446-447` (`gu_hi = dst_off + extent`). Nothing else in the skel moves; `res` layout and the IDL are unchanged. Gate: `./test/htp/build.sh` → `libnntr_hvx_skel.so`, `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK`; record the md5.
3. **Gtest (rung 3).** `unittest_hvx_dma_probe.cpp`: compute `weight_bytes` and `total_bytes` from `plan[]` next to `want_sum` (`:601`); print `weight_bytes=` after `bytes_per_call=` (`:624`) and add `EXPECT_EQ(res[2], total_bytes)` beside the checksum assert (`:631`) so the byte count is checked, not only printed; tighten `plan_shape_ok` (`:538-539`) to compare each push's `kind, e, c, row, nrows, stride` words with `plan[]` (the words are in `trace[]` already; ≤ 10 lines). Update the expectation strings in `docs/plans/87-dma-in-situ-gap.md:247` and `docs/plans/94-sitting-2.md:179` to `bytes_per_call=22560768 weight_bytes=22020096`. Gate: `build_android.sh --htp --cache`, `ndk-build … unittest_hvx_dma_probe`, both `NEEDED` lines, md5s recorded. clang-format-14 on changed lines; commit `[HTP] Replay the MoE chunk list with the kernel's VTCM stride`.
4. **Device — ride-along, not a sitting (rung 4, user).** Add one row to the *next* handoff's gtest block (whichever plan is next measured; #94's re-issue if it is re-run, otherwise the next `state:needs-measurement`): the exact command from the issue's Repro with the new skel and gtest md5s. Pass: `[  PASSED  ] 2 tests`, 11 × `checksum_ok=y`, `bytes_per_call=22560768 weight_bytes=22020096`, `plan_shape_ok=y`. Read-out for #94: paste the 11 lines next to the provisional table; if cell 1 (`workers=1 load=0 pace=0`) lands within ±10 % of 561.5 µs / 40.2 GB/s and cell 4 within ±10 % of 1215 µs, drop "provisional" from rows c, d, g, h; otherwise the new cells replace them and the delta itself is a wall-2 finding (strided-write cost). No variant table of its own — the ride-along has no A/B and changes no model binary.

## 5. Risks

* **Stale skel / gtest pair.** The fix lives in the skel; the gtest changes only prints and asserts. An old skel with the new gtest still fails with `1467840`; a new skel with the old gtest passes but prints no `weight_bytes`. The handoff row lists both md5s and the `weight_bytes=` token, so a stale artifact is visible in the log.
* **VTCM budget after the fix.** The strided footprint grows the test's VTCM use from 1.5 MB to 5.6 MB (`v_copy + 8192 = 5 644 288`); on the 8 MiB v79 VTCM with `config_off ≈ 8 MiB − config` this fits under `vtcm_limit` for `load=2` as well; step 1 pins the `< 6 MiB` bound on the host, and the skel's extent check returns `AEE_EBADPARM` (printed as `skipped err=`) instead of writing over the HMX config if it ever does not.
* **Timing shift on device.** The fix changes only VTCM write addresses. If the 11 cells move by more than ±10 %, rows c, d, g, h are re-derived from the new cells in the same handoff (step 4); the plan does not assume they will not move, it makes the move measurable against the recorded 561.5 / 1215 µs.
* **Thermal drift between sittings.** The old and new cells come from different sittings; the ±10 % band is wider than the ≤ 7 % day-to-day drift LEDGER rule 9 quantified, and the ride-along's `DmaProbeShapes` line (hot vs cool 88.8 / 106.9 GB/s in #94) is in the same log to normalise against.
* **The trace still carries no destination.** `plan_shape_ok` after step 3 checks source geometry per descriptor; destination geometry is guarded only by the footprint checksum + the host coverage bitmap. Acceptable: the checksum is what found this.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: no tok/s row (no model binary changes). Artifact table: the new skel and `unittest_hvx_dma_probe` md5s when the ride-along handoff is built.
* `docs/htp_moe/LEDGER.md`: ⑥ (wall 2) — add "replay content check (#99): the replay packed its VTCM rows where the kernel strides them; DDR side was the traced one, rows c/d/g/h stand pending the ride-along"; a new rule under "Learned": *a checksum over the destination footprint catches what a trace of source geometry cannot — every replay harness sums its footprint, and the expected footprint is computed from the plan, never typed*. Cycle row for the day.
* `docs/measurements/94-sitting2-anchor-trace.md` (on its branch, by the supervisor when folding): Deviation 6 gets the root cause and the pointer here; "provisional" is lifted or replaced after step 4.
* `docs/plans/87-dma-in-situ-gap.md:247`, `docs/plans/94-sitting-2.md:179`: the expectation string (step 3).
