# 100 — Wall 2, row h: why the traced M=1 chunk list replays at 40 GB/s where probe iii does 107, and which descriptor shape can feed the GEMV from VTCM

Issue: dlwlzzero/nntrainer#100 (p1, LEDGER ⑥ / rule 19, tracker #76).
Contract `docs/plans/0001-htp-moe-decode-agent-system.md`. Base `htp_moe` @
`0c1a445e`; every `path:line` below was read against that tree. Evidence:
`origin/htp/94-sitting2-anchor-trace:docs/measurements/94-sitting2-anchor-trace.md`
@ `dad0f476` (replay table, `DMA_PROBE` table, attribution rows e/f/h),
plans 87 and 99, LEDGER rules 11/18–23, ⑥, ⑯, ㉒, upstream cycle 5b/6.

## 0. Read this first: what #100 serves for decode, and the re-scope

**The 46-descriptor list is not on the decode path any more.** With the
M=1 GEMV switch on (#94 s2 variant C; made the default by #101, in
progress; compute side #105 / PR #107 in review), the M==1 call issues
**2 descriptors** (the copies) and reads the weights straight from the
arena. `DMA ring: desc 46 → 2, wait 269 → 4.3 µs` (#94 s2 §C). So fixing
the HMX path's list serves:

* **(a) prefill (M>1):** its ring is hidden. Wait is 421.1 µs of
  16192.7 dsp at level 3 (2.6 %). A perfect list saves ≤ 2.6 % of the M>1
  call. That is not a lever. It is only a gate: whatever changes the ring
  must not regress prefill.
* **(b) the feed half of ㉒:** stage each expert's weights into VTCM so
  the GEMV reads VTCM instead of the arena. ㉒'s gate is `mm ≤ 600 µs` for
  22.02 MB, so the feed must move **≥ 36.7 GB/s while HVX computes**. The
  in-situ ring does 21.9–32.3, the isolated replay 40.2, and upstream
  measured "~12 GB/s beside the HMX, not the idle 33" (`aa371dd2`).
  Row h is exactly the question of whether a DDR→VTCM descriptor list can
  clear that bar.

**Re-scope (proposed; the supervisor edits the issue).** Row h's *cause*
is the deliverable. Its *fix* is a descriptor shape certified for the GEMV
feed, not a rewrite of `moe_push_weight_chunk`. Concretely:

1. Name the cause of the 2.7× with replay cells. These are test-only code
   (§3.2), so no model binary, IDL, quantizer or kernel changes.
2. Certify one feed shape: replay isolated ≥ 80 % of the **validated**
   ceiling cell of the same run, **and** ≥ 37 GB/s with HVX streaming VTCM
   (`load=2`). The ㉒ feed issue is then filed with that shape. If no shape
   passes, ㉒ closes on #105's compute alone, and the gap to 50 tok/s is
   carried by other items (bytes per token, §3.3 of the contract).
3. The issue's in-situ HMX-row gates (`engine GB/s ≥ 40`, `weight DMA ≥ 30`,
   M==1 `dsp ≤ 1000`) apply **only** if the M=1 HMX path is still the
   decode default when the verdict sitting runs (#101 not merged). In that
   case step 7 applies the named fix to `moe_push_weight_chunk`.

**The gate's denominator is itself suspect.** Probe iii reads
106.9 GB/s cool, ii reads 110–117. That is above the ≈ 85 GB/s peak of a
64-bit LPDDR5X-10667 bus. This is spec arithmetic: the S25 Ultra's DRAM
grade is not recorded in this repo, and nobody has measured its peak here.
Strided 2D also beats contiguous by ~40 % (iii 107 vs i 69–79), which DRAM
does not do. `dma_probe`'s content check also cannot see a dropped
transfer after the first pass:

* `vtcm` is zeroed once, before all passes (`nntr_hvx_dma_probe.c:155`).
* The fill has 0xA5 at every 64-byte sample, whatever the source
  (`unittest_hvx_dma_probe.cpp:65-67`).
* The sum covers slot 0 only (`:162-164`, `sum_bytes` = one payload).

Every per-call reading sits at 27–41 GB/s, dmstart'ed single transfers
included:

* doc 46 `DMA_FIRST` 116 µs = 31.6 GB/s for one 3.5 MB gate_up
* `arena_probe` 29.96 GB/s, one ring descriptor
* in-situ expert-0 gate/up burst 92.4 µs ≈ 38–40 GB/s (row e, L3)
* replay 40.2 GB/s

So "≥ 80 % of probe iii (≥ 85 GB/s)" may not be reachable by any list.
The plan replaces the denominator with a validated ceiling cell (§3.2
`c_star`) measured in the same run by the same timing code.

**Upstream `5731b6e5` does not need to be merged first.** It reorders the
HMX block loop so that down(n) is issued after gate_up(n+1). The M=1 list
keeps its geometry: 4 act + 32 gate/up + 8 down + 2 copies = 46
descriptors, same shapes. Only the push positions and wait points move.
The cells that decide row h are synthetic lists and geometry transforms,
and those do not depend on the order. The one order-dependent cell
(`traced`, and `traced_nowait` built from it) is re-traced by the gtest's
step 0 from whatever kernel is built. After a merge,
`IN-SITU CHUNK PLAN MATCHES KERNEL` in `moe_layer_host_check` fails first
and flags the planner. Its function names also differ from this tree's
(`hexkl_moe_push_weight_chunk`, `blk.first`), so it is not a clean apply
anyway. Decision: work on the frozen list. A later merge is a user
decision that re-opens only the `traced` rows.

Two corrections to the issue text:

* The list carries **8** down descriptors, not 4: `down` is 2 per expert
  × 4. Host arithmetic on `nntr_moe_dma_plan_m1`: act 4, gate 16, up 16,
  down 8, copy 2. Of these, 8 gate/up descriptors have 4 KiB rows.
* `regions=32` is the `fresh=1` rotation pool (plan 99 §3 item 3). A
  `fresh=0` call touches 4 expert regions plus the scratch slot, not 32.

## 1. Goal and gate

| gate | where | pass |
|---|---|---|
| host arithmetic | `bash test/htp/host/run_host_checks.sh` | `ALL CHECKS PASS`, `WORKER POOL LANES OK`, `DMA TRACE ARITHMETIC OK`, plus new `REPLAY CELLS PLAN OK (16 cells)` (§4 step 1); `bash tools/htp_syntax_check.sh` exits 0 |
| MoE sources untouched | `git diff --stat htp_moe -- nntrainer/ Applications/ test/htp/nntr_hvx.idl` | empty (test-only change: `test/htp/nntr_hvx_dma_probe.c`, `test/htp/nntr_moe_dma_plan.h`, `test/htp/host/*`, `test/unittest/unittest_hvx_dma_probe.cpp`). Bit-identity of model outputs holds by construction; the host line `MOE KERNEL MATCHES REFERENCE` is unchanged |
| skel | `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh` | `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (<n> runtime imports)` |
| device content | ride-along gtest (§4 step 5) | every new `DMA_REPLAY_X` line `checksum_ok=y` (tag-validated, §3.2); `DmaProbeShapes` `[  PASSED  ]` with the new tagged expectation |
| **cause named** | the §3.3 decision table over the 16 cells of one run | exactly one row of the table fires with the stated separation. "None fires" is a result too and is recorded as such |
| **feed shape certified** | cells `f1..f3` × `load ∈ {0,2}` | best cell: `gbs(load=0) ≥ 0.8 × gbs(c_star)` **and** `gbs(load=2) ≥ 37.0`, same run |
| standing gates | the host sitting's variant A (E2E, prompt 512, G 64/512/1024) | unchanged: #100 adds no model binary, so prefill ≥ −5 % and text = A are carried by that sitting's own A/B |
| conditional (only if the HMX M=1 path is still default, §0 item 3) | same sitting, level-2/3 M==1 row | the issue's original numbers: `engine GB/s` lower bound ≥ 40, `weight DMA:` avg ≥ 30, `dsp ≤ 1000`, M>1 `dsp` within 5 % of A, text = A, `MoeLayer*` gtests bit-identical |

## 2. Where it lives

| file | what changes | verified lines |
|---|---|---|
| `test/htp/nntr_hvx_dma_probe.c` | replay entry only: (1) decode `w[0]` as `(flags << 16) \| (op << 8) \| kind`. Today it is `it->op = w[0] >> 8` (`:413`), which would swallow flag bits. (2) new op `DRAIN = 2`, where `replay_ring` calls `hexkl_dma_ring_drain()` (`g_started = 0`, so the next push `dmstart`s, `hexkl_dma_ring.c:101-107`). It is rejected at parse when `workers > 1`. Today unknown ops are rejected at `:458`. (3) per-item destination mode: flag `DST_STRIDED` gives `dst_stride = src_stride`, `DST_PACKED` gives `row_size`, and neither keeps the entry's default. That default is today packed at `:254-256` / `:301`, and #99 makes it strided. (4) per-item extent `(nrows − 1) × dst_stride + row_size` in the VTCM bound (`:432`) and the gate/up window (`:444-447`). This is #99 step 2's rule. Whichever of #99/#100 lands second resolves the same four lines to the same rule. (5) per-descriptor cap `bytes > (1u << 20)` (`:431`) → `REPLAY_MAX_DESC_BYTES (4u << 20)`, for cell `f2` (one descriptor per matrix: 57 344 × 64, 32 768 × 56; inside the struct's 24-bit row / 16-bit nrows, `hexkl_dma_ring.h:24-44`) | `:200-203`, `:243-270`, `:274-351`, `:372-462` |
| `test/htp/nntr_moe_dma_plan.h` | `NNTR_MOE_DMA_OP_DRAIN = 2`; a `flags` field in `nntr_moe_dma_item`; header-only builders for the §3.2 cells: `iii` geometry over n regions (packed rotating slots / strided), the transforms `drop_waits`, `keep_kinds`, `serialize(op)`, and the three feed shapes; plus the fill tag `nntr_dma_tag(off)` and `nntr_dma_expected_sum(items, n, calls, fresh, window)` (last-writer simulation). They are shared by the gtest and the host check, as `nntr_moe_dma_plan_m1` already is | `:41-67`, `:124-227` |
| `test/unittest/unittest_hvx_dma_probe.cpp` | fill `pattern()` (`:65-67`) → tagged at 64-byte samples; `DmaProbeShapes`' `want` (`:226`) and `MoeChunkReplay`'s `want_sum` (`:601`) computed from the plan by `nntr_dma_expected_sum`; `words[0]` (`:589`) carries flags; 16 new cells after `:642`, one `DMA_REPLAY_X name=<id> …` line each | `:65-67`, `:210-258`, `:464-645` |
| `test/htp/host/replay_cells_host_check.c` (new) + 4 lines in `run_host_checks.sh` (after `:68`) | per cell: item and wait counts, bytes, every dst extent ≤ 6 MiB (fits `config_off − 1 MiB` for `load=2` on 8 MiB VTCM, the bound plan 99 uses), items ≤ `REPLAY_MAX_ITEMS` (128); the tag simulator equals a brute-force memcpy into a host "VTCM" array; and for every cell the stale-content sum ≠ the fresh sum (negative control). Prints `REPLAY CELLS PLAN OK (16 cells)` | — |

**Consumers that do not move:** the IDL `dma_replay` / `dma_probe`
signatures (`test/htp/nntr_hvx.idl:395`, `:437`; `schedule` stays
`sequence<uint32>`, 8 words per item) and hence the stub;
`HtpComputeOps`; the quantizer tag and the loader check (no layout
change: each expert's gate_up and down are already contiguous 3.5 / 1.75
MiB WH blobs, 4 KiB-aligned by `place`, `htp_compute_ops.cpp:2344`);
`NNTR_HTP_PROFILE` stage tables (`kMoeStages = 30` unchanged);
`tools/htp_fc_report.py`; `hexkl_dma_ring.{c,h}`, `hexkl_mm_u8i4_moe.c`.
The skel is rebuilt because `nntr_hvx_dma_probe.c` is in it. An old skel
paired with the new gtest prints `skipped err=` on the flagged or DRAIN
cells. There is no `0x8000040E`, because the IDL is the same.

## 3. Design

### 3.1 Hypotheses for the 2.7×, ranked by the data already in hand

Arithmetic: 22 560 768 B at 106.9 GB/s would take 211 µs. The call took
561.5, so 350 µs are unexplained. Spread over 46 descriptors that is
≈ 7.6 µs each. Probe iii itself pays ≈ 4.9 µs per 512 KiB descriptor,
dmstart and poll included. A fixed per-descriptor cost of that size would
therefore have to be chain-specific.

| # | hypothesis | for | against | rank |
|---|---|---|---|---|
| H0 | **The denominator is not a per-call rate.** Probe iii (sustained 800 MB stream, best of 3, weak content check) overstates what any call can get | 107–117 > ≈ 85 GB/s DRAM peak; strided > contiguous by 40 %; every per-call reading, single-descriptor dmstart ones included, is 27–41 GB/s (§0); the content check cannot see dropped rows after pass 1 | cool probe reproduces #77 within 4 % (stable, which does not make it right) | **1** |
| H1 | **A dmlinked chain with depth > 1 runs slower than one descriptor per `dmstart`** (the probe's discipline, `nntr_hvx_dma_probe.c:85-104`) | the only structural difference between probe iii and the replay; the in-situ expert-0 burst is a gate/up-only chain (+ act) at ≈ 38–40 with the thread idle in the wait; separate chains (workers 2/4) are not better (39.1 / 36.5) | doc 46's single-descriptor dmstart transfers were also ≈ 30 (cold, one-shot) | 2 |
| H3 | idle gaps at the 30 waits (26 of 30 block at `pace=0`) | `wait_us 551.4` of 561.5 | `busy_us 440.6..557.9` of 561.5: the engine had work ≥ 78 % of the call → at most 1.27× | 3 (bounded) |
| H2 | kind mix: 8 down (16 KiB × 56 @ 32 KiB, 7 MiB), 8 gate/up with 4 KiB rows, 4 act + 2 copies | down is 1/3 of the bytes; 4 KiB-row shapes probe at 66–79 | the gate/up-only in-situ burst is already ≈ 40 | 4 |
| H4 | region rotation / page walks | — | `fresh=0` touches 4 regions; the probe walks 256 MiB and is faster; cell 9 (`fresh=1`) = cell 4 | 5 |
| H5 | destination stride | — | probe iii and cell 1 both write packed (plan 99 §3); in situ writes strided and is also ≈ 40 | 6 |
| H6 | 2D vs 1D, ring size, trace hooks | — | every list compared is 2D; depth 9–11 of 256, no wrap; trace hooks only touch the issuing thread and are the same in every cell | 7 |

The per-kind in-situ rates from the level-3 `[HTP-DMA]` dumps would
settle H2 without a sitting. Those dumps are
`B_level3_dma_m1.txt`/`G_replay.log` on the user's workstation; they are
not in any branch. Step 0 asks for them. The quoted aggregate (row e,
expert-0 gate/up burst) already ranks H2 low.

### 3.2 The cells: one run of `MoeChunkReplay`, no new IDL

All run at `workers=1 pace=0 gap_us=0 calls=20`, and `fresh=0` unless
noted. The existing 11 cells stay as they are and remain #99's check.
Every new line is
`DMA_REPLAY_X name=<id> dst=packed|strided|mixed desc=<n> bytes_per_call=<b> us_per_call=… gbs=… busy_us=lo..hi depth_max=… checksum_ok=y|n`.

| id | list | separates |
|---|---|---|
| `traced` | today's cell 1 list, default dst mode (re-run beside the others) | control |
| `traced_f` | same, `fresh=1` | H4 |
| `traced_nowait` | the traced pushes, all waits removed, one final drain | H3 |
| `traced_gu` / `traced_dn` | the traced pushes filtered to gate+up / down, no waits | H2 (per-kind rate) |
| `iii_chain` | probe iii's own geometry through the ring: 4 regions × 7 columns (8 KiB × 64 @ 56 KiB, each region's gate_up band), 28 × 512 KiB, `DST_PACKED` into rotating 512 KiB slots as `nntr_dma_probe_plan` does, one chain | H0 vs H1: the probe's list with the replay's timing |
| `iii_dmstart` | `iii_chain` with `DRAIN` after every push: one descriptor in flight, each `dmstart`ed. This is the probe's discipline exactly | H1 (vs `iii_chain`) |
| `iii_link1` | `iii_chain` with `WAIT` after every push: one in flight, `dmlink` onto an idle tail | depth vs dmlink itself (vs `iii_dmstart`) |
| `iii_strided` | `iii_chain` with `DST_STRIDED` (the in-situ VTCM layout) | H5 |
| **`c_star`** | `iii_dmstart` with `fresh=1` (32 regions, 168 MiB rotation, tag-validated so a stale window fails) | **the validated ceiling**: the gate's denominator |
| `f1` × load 0/2 | per expert 6 contiguous pieces (5 × 1 MiB + 256 KiB, row 16 KiB), depth 2 (push k+1, wait k). This is `hvx_impl` `mm_pf_kick`'s shape and discipline | feed candidate |
| `f2` × load 0/2 | per expert 2 descriptors, whole matrices (gate_up 57 344 × 64, down 32 768 × 56, contiguous), two buffers (gu 3.5 MiB, dn 1.75 MiB): `gu0 dn0 \| wait gu0, gu1 \| wait dn0, dn1 \| …` | feed candidate ("two linear descriptors per expert", ㉒) |
| `f3` × load 0/2 | per expert 7 gate_up columns (8 KiB × 64 @ 56 KiB) + 2 down columns (16 KiB × 56 @ 32 KiB), `DST_STRIDED`, depth 2 | feed candidate (probe-iii shape, GEMV-readable layout) |

That is 10 diagnostic cells plus 6 feed cells. At ≤ 1 ms per call, the
whole block is under 0.5 s of DSP time.

**Content check that can fail.** The arena fill carries a tag at every
64-byte sample: a hash of the source 4 KiB page, chosen so that two
regions' pages differ. The host computes the expected window sum by
simulating the cell's last call: last writer per sample, in list order,
since a chain retires in order and every new cell has `workers=1`. With
`fresh=1` the last call's regions differ from the previous call's, so a
dropped or short transfer leaves a stale window and fails the sum. That
is H0's test. `DmaProbeShapes` gets the same fill and a host-computed
`want` (slot 0's last writer) so that it keeps passing. Its skel code is
unchanged.

### 3.3 Decision table (read in this order; ratios within the one run)

| row fires if | cause | consequence |
|---|---|---|
| `c_star` < 0.8 × the same log's `DMA_PROBE shape=iii workers=1`, with `c_star` `checksum_ok=y` | **H0**: probe iii is not reproducible through a validated per-call path | rule 11/19 rewritten: the "4–7×" and "2.7×" are against an unreproducible number. Row h's loss is `traced / c_star`. If that is ≥ 0.8, **row h dissolves**. The feed verdict uses `c_star` |
| `iii_dmstart` ≥ 0.8 × probe iii **and** `iii_chain` ≤ 0.6 × `iii_dmstart` | **H1**: chaining costs; `iii_link1` ≈ `iii_dmstart` → depth is the variable, `iii_link1` ≈ `iii_chain` → `dmlink` itself | the fix is issue discipline (depth ≤ 2, or dmstart per descriptor from a feeder thread), not descriptor shape. `f1`/`f3` (depth 2) should pass; `f2` shows whether fewer descriptors alone suffice |
| `iii_chain` ≥ 0.8 × `iii_dmstart` and `traced` ≤ 0.6 × `iii_chain` | the list's content | `traced_nowait` ≥ 1.25 × `traced` → waits (H3); `traced_gu` vs `traced_dn` ≥ 1.5× apart → the slower kind (H2); `traced_f` vs `traced` ≥ 15 % → H4; `iii_strided` vs `iii_chain` ≥ 15 % → H5 |
| none of the above | not decided by the replay | recorded; the next step is the in-situ per-kind dump (step 0) |

**Feed verdict:** the best `f` cell must pass both `load=0 ≥ 0.8 × c_star`
and `load=2 ≥ 37.0 GB/s`. Pass: the ㉒ feed issue is filed with that
shape, its depth and its VTCM footprint. Fail: ㉒'s feed half is closed
as "DMA side cannot deliver 37 GB/s beside HVX". The `load=2` rate is the
bound quoted.

### 3.4 Why this, and the alternative rejected

Chosen: decide on synthetic lists through the **existing** replay entry.
That costs an op code, a flag and a cap: ≈ 20 skel lines, no IDL. The
lists run under the same timing, trace and content check as the traced
list, so every ratio in §3.3 is within one binary, one run and one
thermal checkpoint (rule 18).

Rejected: **change `moe_push_weight_chunk` now to fewer, larger,
contiguous descriptors (the `mm_pf_kick` shape) and read the in-situ
line.** It fixes a path that decode no longer runs once #101 lands (§0).
The HMX needs whole column blocks: contiguous k-row chunks would make it
wait for the full 3.5 MB, the 110 µs doc 46 §26.4 removed by chunking. It
would move the prefill ring with prefill gains capped at 2.6 %. And under
H0 it lands at the same ~40 GB/s as everything else. `mm_pf_kick`'s
shape is tested where it belongs, as the feed candidate `f1`. It is the
fix **only** if H1 fires, and then because of its depth-2 discipline: its
contiguity is the probe's slower shape (i/i1 66–79 vs iii 107). The feed
design must also take upstream `aa371dd2`'s negative into account. A big
prefetch in the one FIFO ring put the activation behind it (gather
84 → 747 µs). So the feed goes on its own engine (per-thread queue, as
`hvx_impl` does) or keeps small urgent transfers ahead.

Contract §2 / doc 45 §3: no weight layout, no CPU-fallback question
(`QS4CX_WH` untouched), no arena or heap growth (static tables, VTCM
below `config_off`). No quantizer input changes, so there is no `_det`
question. The bit-identical and text gates are carried by the host
sitting's A because no model binary changes.

## 4. Steps

0. **(no code, optional, before any device time)** Ask the user on #100
   to attach `B_level3_dma_m1.txt` and `G_replay.log` from the #94 s2
   logs. An awk over `push … done=lo..hi` gives the per-kind in-situ
   rates (gate/up vs down vs 4 KiB-row pairs). If `down` alone runs below
   25 GB/s, H2 moves up and `traced_dn` is the cell to watch. No gate.
1. **Host builders and check.** `nntr_moe_dma_plan.h` gets the op, the
   flag, the builders, the tag and the simulator.
   `replay_cells_host_check.c` and its `run_host_checks.sh` lines are
   added. **Gate rung 1:** `ninja -C build`; `run_host_checks.sh` →
   `ALL CHECKS PASS`, `WORKER POOL LANES OK`, `DMA TRACE ARITHMETIC OK`,
   `IN-SITU CHUNK PLAN MATCHES KERNEL (46 descriptors)` unchanged, new
   `REPLAY CELLS PLAN OK (16 cells)`; `htp_syntax_check.sh` exits 0.
2. **Skel.** The five replay-entry edits of §2. **Gate rung 2:**
   `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`, `-Wall -Werror`,
   `UNDEFINED SYMBOLS OK`; md5 recorded. `git diff --stat htp_moe`
   touches only the §1 file list.
3. **Gtest.** Tagged fill, host-computed expectations for both tests, the
   16 cells, the flag word. **Gate rung 3:** `ndk-build …
   unittest_hvx_dma_probe` (and `build_android.sh --htp --cache` only if
   the host sitting needs a fresh app; #100 changes none); md5 recorded.
   clang-format-14 on changed lines. Commits:
   `[test] Replay cells that separate the probe's rate from the chunk list's`
   and `[test] Tag the DMA probe arena so a dropped transfer fails its sum`.
4. **PR into `htp_moe`** (test-only), `state:review`. It must merge before
   the host sitting's artifacts are built, so that every variant's skel
   carries the entry. If it misses one sitting, it takes the next.
5. **Device measurement (unavoidable; a ride-along, no variant of its
   own).** Added to the gtest block of the next handoff, next to #99's
   ride-along (the same binary). It runs under **that sitting's variant A
   skel**, once, right after `therm`:
   `unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*:*MoeChunkReplay*'`.
   Expected: `DmaProbeShapes` `[  PASSED  ]` (16 lines); 11 `DMA_REPLAY`
   lines plus 16 `DMA_REPLAY_X` lines, all `checksum_ok=y`; the probe iii
   `workers=1` line and `c_star` in the same log. Paste the
   `thermal_zone0` reading next to them. It adds about 1 minute. The
   sitting's own A (full E2E, prompt 512, G 64/512/1024 × 2, CPU and NPU
   as that plan has them) is the standing prefill/text control. #100
   contributes no variant to the ≤ 4. Stop signals: `skipped err=` on an
   X line means a stale skel (re-push A's skel, re-check md5); any X line
   with `checksum_ok=n` is filed, not read.
6. **Verdict.** Apply §3.3 to the pasted lines, then write LEDGER ⑥ /
   rule 19 (and rule 11 if H0 fires). File the ㉒ feed issue with the
   certified shape, or close ㉒'s feed half per §3.3. Close #100.
7. **(conditional, §0 item 3 only)** If the HMX M=1 path is still the
   decode default, apply the discipline §3.3 named (H1: cap outstanding
   gate/up chunks at 2 through the existing waits; H3/H2: the order or
   kind change the table names) in `moe_push_gate_up_chunks` / the down
   loop (`hexkl_mm_u8i4_moe.c:168-185`, `:1228-1260`). Gates: rungs 1–3,
   `moe_layer_host_check` `MOE KERNEL MATCHES REFERENCE` and
   `PROFILE ON/OFF BYTE-IDENTICAL`, then its own handoff with A = unchanged
   reference, B = the change, G 64/512/1024, prompt 512, both profile
   levels at G = 64.

## 5. Risks

| risk | how the run shows it |
|---|---|
| Thermal drift inside the sitting (rule 18: probe iii 106.9 cool / 88.8 hot) | every ratio in §3.3 is inside one gtest run, seconds apart; `therm` is pasted with it; the probe iii line and `c_star` sit in the same log |
| DVFS: short per-call bursts vs the probe's sustained 800 MB stream | `c_star` runs 20 back-to-back calls over 168 MiB. If `c_star` ≪ probe while `iii_dmstart` ≈ probe, the gap is "sustained vs bursty" and is recorded as H0's variant, not as H1 |
| Stale skel (rule 3/17) | new op and flags on an old skel → `skipped err=` on the X lines, never silent; md5 line before the run |
| VTCM budget | `f2` + `load=2` = 5.25 + 1 MiB; the host check bounds every cell ≤ 6 MiB; the skel's extent check returns `AEE_EBADPARM` (printed) rather than overwrite the HMX config |
| Address space / heap | none: static tables (`REPLAY_MAX_ITEMS` 128 covers the largest cell, `f3` ≈ 72 items); `load=1` is not used by new cells |
| #99 lands in between and flips the default dst mode | the new cells set `DST_PACKED` / `DST_STRIDED` explicitly; only the old 11 cells and `traced*` follow the default, and their line says which (`dst=` field) |
| Cross-sitting comparison with #94 s2's 40.2 / 106.9 | not used as a verdict (rules 20/23); only the ratios of this run are |
| The DRAM-peak figure is spec arithmetic, not measured | H0 is decided by `c_star` vs the probe line in the same log, not by the 85 GB/s figure |

## 6. Docs to update

* **BENCHMARK.md**: a side table under the host sitting, "#100 replay
  matrix (16 cells)" with `gbs`, `busy_us`, `checksum_ok` and the probe
  iii line of the same run. The #94 s2 row-h bullet gets the verdict
  pointer. Artifacts row: the skel and `unittest_hvx_dma_probe` md5s that
  ran. No tok/s row (no model binary).
* **LEDGER.md**:
  * ⑥: row h's verdict (which §3.3 row fired, with ratios).
  * rule 19 rewritten from "provisional" to the named cause; rule 11
    refined if H0 fires (the isolated rate it quotes is the unreproducible
    number).
  * ㉒: the feed half filed with the certified shape, or closed.
  * §4 `hvx_impl` table, `mm_pf_kick` row: measured as `f1`, with its
    number.
  * Upstream cycle 5b: "#100 did not need `5731b6e5`; a merge re-traces
    only the `traced*` cells."
  * New rule if H0 fires: *a DMA probe's rate is quoted only with a
    content check that fails on a dropped transfer and a figure below the
    DRAM peak*.
* **Contract §2 wall 2 sentence**: the cause as the verdict names it.
