# 87 — Wall 2 rewritten: instrument the MoE call's DMA ring use and attribute the 4–7× in-situ gap

Issue: dlwlzzero/nntrainer#87 (LEDGER §3 ⑥, wall 2; tracker #76; contract
`docs/plans/0001-htp-moe-decode-agent-system.md`). Branch
`htp/87-dma-in-situ-gap` from `htp_moe` @ `7ee82501`. Instrument, do not
fix: the deliverable is a handoff whose table attributes the gap between
the isolated 72–117 GB/s (#77 C) and the in-situ 16–18 GB/s (#77 B) to
named causes with a percentage each, and names the first fix.

## 0. What the code already says before any device time (read this first)

Two arithmetic facts, verified against the kernel, reshape the issue's
hypotheses. The instrumentation confirms them on silicon; it should not
rediscover them.

**(i) The in-situ descriptor geometry at the LFM2 shape is the probe's
best shape, not a small or contiguous one.** `K=2048, inter=1792,
N_out=2048` (`hf/config.json`: `moe_intermediate_size 1792`), `acc_tiles`
capped at 32 (`hexkl_mm_u8i4_moe.c:550-553`), `half = 16` (`:645`):

| push site | descriptors per expert | row_size × nrows @ src_stride | bytes |
|---|---|---|---|
| gate_up, `moe_push_gate_up_chunks` `:159-173` → `moe_push_weight_chunk` `:129-144` | 4 chunks × 2 (gate, up) = 8; `cn` = 16,16,16,8 | **8 KiB × 64 @ 56 KiB** (last pair 4 KiB × 64 @ 56 KiB) | 3.5 MiB |
| down, `:994-1000` | 2; `cn` = 32 | **16 KiB × 56 @ 32 KiB** | 1.75 MiB |
| activation block, `moe_push_act_block` `:189-198` | 1 | 16 KiB × 8 @ 16 KiB (contiguous, `moe_dma_row_size` `:106`) | 128 KiB |
| `moe_dma_copy` `:442-454` (act in, out out) | 1 each at M=1 | 8 KiB × 1 contiguous | 8 KiB |

Per M=1 call: 4 experts × 11 + 2 = **46 descriptors, 21,504 KB**
(= the profile's `21504 KB/call`). The gate_up shape is exactly #77 C's
`iii` (8 KiB × 64 @ 56 KiB, 108–117 GB/s isolated); down is between `ii`
and `iii`. Hypothesis (b) of the issue ("contiguous or small-row chunks")
is refuted by the source; the geometry capture below only proves the
kernel pushes what the arithmetic says. At M>1 the same weight
descriptors repeat per expert (weights are pushed once per expert, not per
block, `:994` and `:1109-1119`), only the activation blocks multiply.

**(ii) The `first 1024 KB took 0 us` line is a wait-order artefact, and
the exposed first transfer is filed under `gather`, not `drain`.** The
first expert's activation block is pushed *behind* its gate_up
(`:925-933`, the one deliberately reversed order), so the act wait at
`:977` (timed into `HEXKL_PROBE_GATHER`, `:976-982`) covers the whole 3.5
MiB gate_up[0] plus 128 KiB. By the time `:1024` waits on `gu_idx[0]` it
is done, `DMA_FIRST` reads 0 (`:1026-1032`), and the profile divides 1024
KB by 0. Read against #77 B at M==1: `gather 143.5 / 122.6 us` (level 2 /
3) for a 1-row scale copy is that wait; 3.6 MiB / 122 us ≈ **30 GB/s for a
cold 3.5 MiB transfer with nothing to hide behind** — the PR author's
38 GB/s probe, roughly. Meanwhile `drain 0.9+0.4 us` at level 3 says every
*other* wait finds its chunk already resident: behind 748 us of HMX issue
the ring is not exposed at all on a warm call. At level 2 (single-shot
calls, `drain 108.7+7.3`) the same waits block for ~110 us — the
warm/cold difference is the largest signal in the existing data and is
hypothesis (g) below.

So the "16–18 GB/s" is `bytes / dsp_us` with `dsp_us` dominated by wall 1
(HMX 64-row tiles). The engine's rate *while it runs* is unmeasured; that
number, and the cold-start share, are what this issue produces — they are
what wall 2 costs once #86 removes the HMX time and the DMA (if kept) is
fully exposed.

## 1. Goal and gate

From the issue, made measurable:

* **Host:** `test/htp/host/run_host_checks.sh` gains `dma_trace_host_check`
  (scripted push/wait sequence → expected union-of-intervals busy time,
  depth, blocked-wait count; pass line `DMA TRACE ARITHMETIC OK`) and
  `moe_layer_host_check` prints three new lines: `PROFILE ON/OFF
  BYTE-IDENTICAL` (M=37 fixture, `hexkl_probe_on` 1 vs 0, `memcmp == 0`),
  `PROFILE OFF: TRACE UNTOUCHED` (all trace counters 0 with probing off),
  `IN-SITU CHUNK PLAN MATCHES KERNEL (46 descriptors)` (the M=1 LFM2-shape
  run's push trace equals the header-only planner's list). Existing lines
  (`MOE KERNEL MATCHES REFERENCE`, `HMX blocks: 5`, `ALL CHECKS PASS`,
  `WORKER POOL LANES OK`) unchanged. `tools/htp_syntax_check.sh` exits 0;
  `ninja -C build` clean.
* **Build:** `test/htp/build.sh` (v79, `-Wall -Werror`) and
  `build_android.sh --htp`; `ndk-build unittest_hvx_dma_probe
  unittest_hvx_mm_u8i4`; md5s in the PR and the handoff.
* **Handoff (the real gate):** a filled `docs/measurements/87-dma-in-situ-gap.md`
  whose §"Attribution" table splits the M==1 call's DMA-related time into
  named rows (§3.2 hypotheses a–g), each with a percentage of `dsp_us`
  and of the isolated-vs-in-situ gap, and ends with one sentence naming
  the first wall-2 fix issue.
* **Standing gates:** prefill of the instrumented binary with
  `NNTR_HTP_PROFILE` unset within −5 % of variant A in the same sitting;
  generated text identical to A (and, on the NPU model, run1 = run2); the
  `M>1` profile row's existing columns unchanged beyond noise. No DSP
  arithmetic changes, so `NNTR_L2_DIFF` is not required; the byte-identity
  host line stands in.

## 2. Where it lives

Base first: the #77 probe sources are **not on `htp_moe`** (branch
`htp/77-first-handoff` has no PR; `git log htp_moe..origin/htp/77-first-handoff`
lists them). Step 1 cherry-picks the three code commits verbatim —
`ca231caa` (`build.sh` `HEX_EXTRA_CFLAGS`, bus-vote option), `a5996205`
(`dma_probe` IDL entry, `nntr_hvx_dma_probe.c`, `nntr_dma_probe_plan.h`,
the descriptor struct and asm helpers moved into `hexkl_dma_ring.h`),
`3c6e8397` (`unittest_hvx_dma_probe.cpp`, `dma_probe_host_check.c`,
`run_host_checks.sh`, `Android.mk`) — so the new gtest case has a file to
live in. If the user merges that branch first, the picks drop out on
rebase.

| file | change | lines verified |
|---|---|---|
| `nntrainer/tensor/htp_backend/hmx/hexkl_dma_ring.{c,h}` | **one pure query added, nothing else**: `int hexkl_dma_ring_is_done(uint32_t idx)` returning `g_ring[idx].done` (no `dmpoll`, no state change). The push/link/start/wait/drain paths are untouched — this file is shared with prefill (`hexkl_mm_u8i4_dma.c`) and the issue forbids ring behaviour changes | `hexkl_dma_ring.c:128-135` (`wait_idx_` spin), `:145-185` (`push2d`), `:189-195` (`wait`) |
| `nntrainer/tensor/htp_backend/hmx/hexkl_dma_trace.{c,h}` (new) | the bookkeeping: fixed static tables (no heap; §3.1), `hexkl_dma_trace_reset/push/wait_begin/wait_end/sample/finish`, and the union-of-intervals / depth arithmetic. Free of Hexagon headers so the host checks compile it as-is; the clock comes in through `hexkl_probe_now_ticks()` (below) | — |
| `nntrainer/tensor/htp_backend/hmx/hexkl_probe.{c,h}` | new slots appended before `HEXKL_PROBE_N` (§3.1); `hexkl_probe_now_ticks()` = raw `HAP_perf_get_qtimer_count()` (19.2 MHz, 52 ns) beside the µs one | `hexkl_probe.h:121` (`HEXKL_PROBE_N`), `:142-144` (`hexkl_probe_now`) |
| `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c` | the three push helpers and `moe_dma_copy` call `hexkl_dma_trace_push` after `hexkl_dma_ring_push2d`; the three waits (`:977`, `:1024`, `:1145`) and the two drains inside `moe_dma_copy` are wrapped `trace_wait_begin(idx, kind)` / `ring_wait` / `trace_wait_end`; one `trace_sample()` inside `MOE_MM_END()` (`:480-488`, already a probe point). Every call is inside `if (hexkl_probe_on)`. The HMX loop, the pool waits and all epilogues are not touched | `:129-144`, `:159-173`, `:189-198`, `:442-454`, `:976-982`, `:1022-1033`, `:1143-1147` |
| `test/htp/nntr_hvx_mm_u8i4.c` | `MOE_T_*` mirror extended (`:867-888`), `moe_layer_timed` fills the new slots (`:1013-1033`); new entry `nntr_hvx_moe_dma_trace_read` (copies the last timed call's trace tables); `NNTR_HTP_PROFILE` never reaches the DSP — `hexkl_probe_on` is set only inside `*_timed` (`:993`, reset `:996`), so "level < 2" on the DSP means "probing off" | `:864-888`, `:968-1034`, `:560` comment |
| `test/htp/nntr_hvx.idl` + `generate_stub.sh` (via `build.sh`) | two entries appended after `dma_probe` (and after #86's `moe_set_opts` if merged): `moe_dma_trace_read(rout sequence<uint32> words, rout uint32 n_words)` and `dma_replay(...)` (§3.3); `mm_u8i4_moe_layer_timed`'s `stage_us` grows by the new slot count → skel and app rebuilt together (rule 3) | `:331-341`, `:378-399` |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | `HTP_MOE_T_*` mirror (`:148-169`), `Bucket` fields and `addInvokeMoeLayer` (`:284-310`), the second line under `weight DMA:` (`:497-508`), the `[HTP-DMA]` per-descriptor dump after the `invokeMoeLayer` loop (`:1497-1520`) for the first `NNTR_HTP_DMA_TRACE` (default 3) calls of each bucket. Existing columns and their order unchanged | `:148-169`, `:284-310`, `:430-508`, `:1477-1545` |
| `tools/htp_fc_report.py` | reads only the FC `*_us` columns (`:45-59`); the MoE `weight DMA:` line is not parsed there — no change. Confirmed by grep | `:45-59` |
| `test/htp/nntr_moe_dma_plan.h` (new, header-only) | the in-situ descriptor list for (K, inter, N_out, M=1, n_active, acc_tiles) in push order with wait points — the same arithmetic as `moe_push_gate_up_chunks` / the down loop / `moe_push_act_block`, restated for the host check and the replay | — |
| `test/htp/nntr_hvx_dma_probe.c` | `dma_replay` entry: replays a descriptor+wait schedule from the ring (1 worker, one `dmstart`) or from N workers (own `dmstart` each), optional HVX load job, optional pacing (§3.3) | `:76-96` (per-worker loop to reuse), `:98-178` (entry shape) |
| `test/unittest/unittest_hvx_dma_probe.cpp` | new `TEST_F(HvxDmaProbe, MoeChunkReplay)`; lifts the arena weight registration from `unittest_hvx_mm_u8i4.cpp:2117-2185` (`MoeLayerFromArenaMatchesHeap`) into a small helper | `:78-274` (fixture) |
| `test/htp/host/moe_layer_host_check.c`, `test/htp/host/stub/HAP_perf.h`, `test/htp/host/dma_trace_host_check.c` (new), `run_host_checks.sh` | §4 steps 2–3; the stub clock becomes a settable counter (`HAP_perf.h:4-5` returns 0 today) | `moe_layer_host_check.c:84-103` (ring stubs), `:337-341` (probe globals), `:294-471` (main) |
| `test/jni/Android.mk` | nothing new (the #77 pick adds the gtest target `:954-975`) | — |

Not touched: the quantizer format tag (`QS4CX_WH` layout unchanged), the
loader check, `hvx/*`, `hexkl_mm_u8i4_dma.c`, the app's forward path.
Address-space budget: the trace tables are ≈ 14 KiB of skel `.bss` (§3.1),
no heap, no arena.

## 3. Design

### 3.1 What is recorded (only under `hexkl_probe_on`, i.e. `NNTR_HTP_PROFILE ≥ 2`)

Per timed call, in static tables sized for one call (`MOE_TRACE_MAX_PUSH
256` = the ring size, `MOE_TRACE_MAX_WAIT 64`, `MOE_TRACE_MAX_SAMPLE 64`):

* **push record** (8 × u32): `t_issue` (qtimer ticks since call t0),
  `kind` (0 act, 1 gate, 2 up, 3 down, 4 copy), `expert_ordinal`,
  `chunk`, `bytes`, `row_size`, `nrows`, `src_stride` (contiguity =
  `src_stride == row_size`), `ring_idx`, `depth_at_issue` (pushed − seen
  done, before this push), `t_done_lo`, `t_done_hi`.
* **completion bracket.** The engine does not timestamp completion, so at
  every instrumentation point (push, wait entry, wait exit, `MOE_MM_END`
  sample) the tracer advances a watermark: for each outstanding descriptor
  in ring order, `hexkl_dma_ring_is_done(idx)`; the first time it reads 1
  the record gets `t_done_hi = now`, and `t_done_lo` = the previous point's
  time. Busy time is therefore a **range**: `busy_lo` = union of
  `[t_issue, t_done_lo]`, `busy_hi` = union of `[t_issue, t_done_hi]`; the
  engine rate is reported as `bytes/busy_hi .. bytes/busy_lo`. A blocking
  wait tightens its own descriptor's bracket to the wait exit time.
* **wait record** (4 × u32): `t_in`, `t_out`, `ring_idx`, `site` (act
  `:977`, gu `:1024`, dn `:1145`, copy-in, copy-out) with the `blocked`
  bit = `!is_done(idx)` at entry. "Returned immediately" vs "blocked"
  is the issue's dry/blocking distinction.
* **per-call aggregates**, folded into new probe slots appended to
  `hexkl_probe.h` (and mirrored in `MOE_T_*` / `HTP_MOE_T_*`, positions
  after `HEXKL_PROBE_ACC_STRIDE` and after #86's `HEXKL_PROBE_PATH` when
  present): `DMA_DESC` (count), `DMA_WAITS`, `DMA_WAITS_BLOCKED`,
  `DMA_WAIT_ACT_US` (the `:977` share that today hides in `gather`; `gather`
  itself is left as is so #77's rows stay comparable), `DMA_BUSY_LO_US`,
  `DMA_BUSY_HI_US`, `DMA_DEPTH_MAX`, `DMA_FIRST_READY_US` (t0 → gate_up[0]
  fully resident: the cold-start cost), `DMA_LAST_ISSUE_US` (t0 → last
  weight push: how much of the call has DMA left to do at all).

Output. The existing `[HTP-PROFILE]` row and `weight DMA:` line are
byte-for-byte the same format; one new line follows each `weight DMA:`
line:

```
[HTP-PROFILE]     DMA ring: desc=46/call waits=14 (blocked 5) wait=124.3 us [act 118.9 gu 0.0+dn 5.4] busy=196..231 us -> engine 91..107 GB/s  depth max=11  first expert ready at 131 us  last issue at 902 us of 1222
```

and, for the first `NNTR_HTP_DMA_TRACE` calls (default 3) of each bucket,
`[HTP-DMA] call=<n> M=<M> push k=<i> t=<us> kind=gu e=<i> c=<i> bytes=524288 row=8192 nrows=64 stride=57344 depth=<d> done=<lo>..<hi>` and
`[HTP-DMA] call=<n> wait k=<i> site=act idx=<i> t=<in>..<out> blocked=y`
lines (one per record), read on the host through `moe_dma_trace_read`
after the `invokeMoeLayer` loop (at level 3 the trace is the last of the
5 repeats — a warm call — which is stated in the line header as `rep=5/5`).
Level 2 prints single-shot (cold) calls; level 3 prints warm ones; both
are wanted (hypothesis g).

Cost when on: ≈ 46 pushes × (1 timer read + ≤ depth `is_done` loads) +
14 waits × 2 timer reads + 24 samples per M=1 call — a few µs against
1.2 ms, and only in timed calls. Cost when off: one `if (hexkl_probe_on)`
per call site, the gate every existing probe already pays; the ring's
own code path does not change at all.

**Rejected alternative:** put the timestamps inside `hexkl_dma_ring.c`
(`push2d` / `wait`) behind a flag. It would give the same data with fewer
call sites, but it puts a branch on the prefill FC's ring path
(`hexkl_mm_u8i4_dma.c`) and makes "no ring behaviour changes" a matter of
review instead of a diff that does not touch the file. Also rejected:
FARF lines from the kernel (need mini-dm / logcat, perturb the loop, and
cannot be bound to a host call), and growing `stage_us` to carry the
whole trace (8 KiB of `rout` marshalling per call would move the
`transport` column this sitting also reads).

### 3.2 Hypotheses the table must decide (one signature each)

| # | hypothesis | decided by | signature that confirms |
|---|---|---|---|
| a | the ring is serialised behind compute: waits dominate, depth ≤ 1 | `wait=` vs `dsp`, `depth max` | `wait ≥ 25 %` of `dsp` **and** `depth max ≤ 2`; refuted if depth reaches 8–11 (a whole gate_up in flight) while waits stay small |
| b | chunk geometry is a shape the probe measured slow | `[HTP-DMA]` push lines, host line `IN-SITU CHUNK PLAN MATCHES KERNEL` | refuted on the host already (§0 i): gate_up is 8 KiB × 64 @ 56 KiB = probe `iii`; the device lines merely confirm |
| c | one dmlinked chain / one `dmstart` limits concurrency | replay `workers=1` vs `2/4`, same schedule, `pace=0` | confirmed if `gbs` rises ≥ 15 % with workers (note #77 C found 1→4 flat-to-negative for independent streams; the in-situ list has 46 dependent-order descriptors, so it can differ) |
| d | DDR contention with HVX epilogues / pack | replay `load=hvx_ddr` vs `load=none` at `pace=1`; in situ `M>1` vs `M==1` engine rate | confirmed if the paced replay's blocked-wait total moves ≥ 20 % with the load; at M=1 the epilogues touch DDR only for the 8 KiB scatter, so a large in-situ effect at M==1 with none in the replay points elsewhere |
| e | DVFS / bus ramp inside a call | per-descriptor rate of expert 0 vs expert 3 in one `[HTP-DMA]` dump; `busy` at level 2 vs 3 | confirmed if chunk rates climb monotonically through the call (first expert ≤ 60 % of the last) on cold calls and are flat on warm ones |
| f | cold start: the first expert's 3.5 MiB has nothing to hide behind (§0 ii) | `DMA_WAIT_ACT_US` of expert 0, `first expert ready at`, `gather` unchanged from #77 | confirmed if the `:977` wait of expert 0 ≈ `gather` (≈ 120 us) and every later wait is ≈ 0 on warm calls: then ~10 % of the M=1 call is cold start and the rest of the DMA is hidden behind wall 1 |
| g | warm/cold per call: single-shot calls (level 2, `drain 108`) block where 5× repeats (level 3, `drain 1.3`) do not — IOTLB / page state / bus clock after the ~0.6 ms FastRPC gap | level 2 vs level 3 `DMA ring:` lines; replay `fresh=1` (32 distinct 5.25 MiB regions in rotation) vs `fresh=0` (the same 4), and `gap_us=600` vs `0` between replayed calls | `fresh=1` alone reproduces the level-2 wait profile → translation/page state (fix: prefetch the next token's experts' first chunk, or larger mappings); `gap_us` alone reproduces it → clock ramp (fix: keep the engine busy across calls / bus vote timing; #77 C already showed the vote flag is ≤ 4 %, so the gap effect would be DSP-side clocking) |

The attribution table in the handoff has one row per letter, a
percentage of `dsp_us` (M==1, level 3 for warm and level 2 for cold), and
the fix issue name for the largest row. The expected outcome from §0 is
f + g large, a/b/c small — but the table is filled from the device, not
from this paragraph.

### 3.3 The replay gtest (`unittest_hvx_dma_probe`, `MoeChunkReplay`)

DSP entry `dma_replay(in uint32 arena, in sequence<uint32> schedule, in
uint32 workers, in uint32 load, in uint32 pace, in uint32 fresh, in uint32
gap_us, in uint32 calls, rout sequence<uint32> res)`:

* `schedule` = the push/wait list in in-situ order: for each item `kind,
  src_off, dst_off, row_size, nrows, src_stride, t_rel_ticks, wait_idx_or_
  none`. Built by the gtest from `nntr_moe_dma_plan.h` at the LFM2 shape
  with four expert regions in an attached 256 MiB chunk (weights are not
  needed for a DMA replay — the bytes are a pattern; `fresh=1` rotates
  the four regions through 32 slots of 5.25 MiB so each call touches new
  pages). The `t_rel` column comes from the `[HTP-DMA]` dump of a real
  timed call: the gtest first runs `mm_u8i4_moe_layer_timed` once at M=1
  on registered arena weights (helper lifted from
  `unittest_hvx_mm_u8i4.cpp:2117-2185`), reads the trace, prints it, and
  uses its timestamps; if that call is unavailable the gtest falls back to
  `pace=0` only and says so in the line.
* `workers=1`: the schedule goes through `hexkl_dma_ring_push2d` /
  `hexkl_dma_ring_wait` (the production ring, one chain, one `dmstart`).
  `workers=N`: descriptors round-robin over N pool workers, each with its
  own `dmstart` (the `dma_probe_worker` loop, `nntr_hvx_dma_probe.c:76-96`),
  waits mapped to the owning worker's descriptor.
* `pace=0`: issue as fast as the schedule's order allows, wait where the
  schedule waits — the engine's own rate on the in-situ list.
  `pace=1`: spin until `t_rel` before each push/wait — the in-situ
  timeline without compute, so the blocked-wait sum is directly comparable
  to the `DMA ring:` line's `wait=`.
* `load`: 0 none; 1 an `hvx_worker_pool_submit` job that streams a 4 MiB
  DDR heap buffer (read + write, 3 workers) for the call's duration; 2 the
  same over VTCM (no DDR traffic, HVX busy) — separates "DDR contention"
  from "HVX threads active".
* `res`: `us_total, calls, bytes_per_call, wait_us (sum of blocked time),
  n_blocked, depth_max, checksum` (the 64-stride sum of the last
  destination, as `dma_probe` does).

Printed line, one per cell: `DMA_REPLAY workers=<w> load=<l> pace=<p>
fresh=<f> gap_us=<g> calls=<n> us_per_call=<..> bytes_per_call=22020096
gbs=<..> wait_us=<..> blocked=<..>/<..> depth_max=<..> checksum_ok=y`.
Cells: `workers ∈ {1,2,4}` × `pace ∈ {0,1}` at `load=0 fresh=0 gap=0`
(6); then at `workers=1 pace=1`: `load ∈ {1,2}` (2), `fresh=1` (1),
`gap_us=600` (1), `fresh=1 gap_us=600` (1) — **11 lines**, ≈ 20 calls
each, well under a minute. The M=1 `[HTP-DMA]` dump from step 0 is printed
as `DMA_REPLAY_TRACE ...` lines above them so the handoff has the in-situ
schedule next to its replays.

### 3.4 What the design respects

Contract §2: no CPU fallback for `QS4CX_WH` is touched; no new heap or
arena use (14 KiB `.bss`); the three walls are not "fixed" here. Doc 45
§3: activation handles unchanged; DMA stays hidden behind compute exactly
as today (no push or wait moves); nothing before a quantizer changes, so
`_det` is unaffected and the bit-identity gate is the host `memcmp` plus
run1 = run2 on device. PR #86 (`htp/80-m1-moe-gemv-dispatch`): its M=1
GEMV path issues no DMA (`use_m1` skips `moe_push_gate_up_chunks` and
`goto out` before the block loop), so the trace calls never execute on
it; the only textual overlaps are the appended enum slots
(`HEXKL_PROBE_PATH`) and the IDL tail — put #87's after #86's and rebase.

## 4. Steps

1. **Branch and base.** `git checkout -b htp/87-dma-in-situ-gap htp_moe`;
   `git cherry-pick ca231caa a5996205 3c6e8397` (verbatim; conflicts none
   expected — `htp_moe` has not touched those files since). Gate: rung 1
   (`run_host_checks.sh` now ends with the `dma_probe_host_check` pass
   line) and rung 2 (skel builds with `nntr_hvx_dma_probe.c`).
2. **Trace module + host arithmetic check.** `hexkl_dma_trace.{c,h}`,
   `hexkl_probe_now_ticks()`, the settable stub clock, and
   `dma_trace_host_check.c`: a scripted sequence — pushes at t = 0, 10,
   20 (depth 1, 2, 3), samples that mark descriptor 0 done at 30, wait on
   descriptor 2 that blocks 30→50, a push at 60 that finds the ring empty
   (depth 0), a wait at 70 that returns immediately — expecting
   `busy_lo = 50, busy_hi = 50` (not 90: the union), `depth_max = 3`,
   `blocked = 1/2`, `first_ready` as scripted. Gate: rung 1, pass line `DMA
   TRACE ARITHMETIC OK` wired into `run_host_checks.sh`.
3. **Kernel call sites + `moe_layer_host_check` lines.** Wrap the pushes
   and the three waits; extend the host stub ring (`moe_layer_host_check.c:84-103`)
   with a settable `done` model so a scripted wait can "block"; add the
   three lines of §1. The LFM2-shape M=1 run uses the existing
   `hexkl_mm_u8i4_moe_layout` LFM2 case's shape (`:463`) with four random
   experts; at `-O2` the scalar HMX stand-in takes ≈ 1 min (PR #86's note in
   `run_host_checks.sh`). Gate: rung 1, all lines present; `ninja -C build`
   still clean (nothing in `build/` compiles the DSP files, but
   `htp_syntax_check.sh` does).
4. **Skel side: slots, IDL, trace read, replay entry.** `MOE_T_*` mirror,
   `moe_dma_trace_read`, `dma_replay`, `nntr_moe_dma_plan.h`. Gate: rung 2
   (`build.sh`, `-Wall -Werror`, md5 recorded). Static budget note in the
   commit body (`.bss` +14 KiB).
5. **Host side: profile line and dump, gtest.** `htp_compute_ops.cpp`
   mirror + `DMA ring:` line + `[HTP-DMA]` dump (`NNTR_HTP_DMA_TRACE`,
   default 3); `MoeChunkReplay`. Gate: rung 3 (`build_android.sh --htp`,
   `readelf` shows `libsdkl.so` and `libcdsprpc.so`; `ndk-build
   unittest_hvx_dma_probe unittest_hvx_mm_u8i4`; md5s recorded).
6. **Inertness proof and PR.** `git diff --stat htp_moe` shows
   `hexkl_dma_ring.c` +1 function only and no `hvx/*`, `hexkl_mm_u8i4_dma.c`
   change; the host lines `PROFILE OFF: TRACE UNTOUCHED` and `PROFILE
   ON/OFF BYTE-IDENTICAL`; PR into `htp_moe` with one `<details>` per
   commit. Set `state:review`.
7. **Handoff (device measurement unavoidable here).** Write
   `docs/measurements/87-dma-in-situ-gap.md` per the `hexagon-handoff`
   skill; set `state:needs-measurement`. Variants (≤ 4, full E2E,
   prompt 512, gen 64 / 512 / 1024, twice each, `NNTR_NUM_THREADS=8`,
   `q40-qs4cx-wh`; CPU `q40` only if bundled with #91):
   * **A** — unchanged reference: `htp_moe` head build (or the #77 device
     set with device md5s, LEDGER rule 14), run first.
   * **B** — this PR's skel + app, `NNTR_HTP_PROFILE` unset: tok/s cells;
     must equal A within the sitting's noise (prefill ≥ −5 %, text = A).
   * **B-profile** — same binaries at G=64 only: `NNTR_HTP_PROFILE=2` and
     `=3` runs (`NNTR_HTP_DMA_TRACE=3`), never read for tok/s; paste the
     `M==1` and `M>1` rows, both `weight DMA:` lines, both `DMA ring:`
     lines, and the `[HTP-DMA]` blocks (3 calls per bucket per level).
   * **gtest** — `unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*'`
     with the **B skel** (the `dma_replay` entry does not exist in A's):
     11 `DMA_REPLAY` lines + the `DMA_REPLAY_TRACE` dump.
   Estimated: install 5–10 min (models already on the device from #77 /
   #91; skel swaps only), A 6 NPU cells ≈ 10 min, B ≈ 10 min, profiles 4
   runs ≈ 6 min, gtest ≈ 3 min → **≈ 35–40 min** standalone.

**Commit split** (kernel/app commits separate from docs, contract §5):
(1–3) the three cherry-picks; (4) `[HTP] DMA ring trace: bookkeeping
module and host arithmetic check`; (5) `[HTP] Trace the MoE layer call's
ring use behind NNTR_HTP_PROFILE >= 2`; (6) `[htp] moe_dma_trace_read and
dma_replay skel entries, in-situ chunk planner`; (7) `[HTP] Print the DMA
ring line and the per-descriptor dump in [HTP-PROFILE]`; (8) `[test]
Device gtest: replay the M=1 MoE chunk sequence in isolation`; (9)
`[docs] Handoff 87` (separate, on the same branch, after the PR).

**Bundling (recommendation).** LEDGER ⑯ leaves it to the user. If PR #86
is merged before step 4 (so #87 rebases onto it and one binary carries
both switches), one sitting on `R3CY10WM83Y` covers #91 + #86 + #87 with
**three model-path variants** and fits the budget:

| block | what | min |
|---|---|---|
| install | both models present from #77/#91? then skel + app pushes only | 5–10 |
| A (#91 anchor) | CPU `q40` 6 cells + NPU 6 cells, twice each — the anchor table, replaces the provisional "now" | 25 |
| B (#86 + #87 binary, both switches off, profile unset) | NPU 6 cells — the prefill gate for both PRs at once (must equal A) | 10 |
| C (same binary, `NNTR_MOE_HTP_M1_GEMV=1`) | NPU 6 cells — #86's decode number | 10 |
| profiles at G=64 | B level 2 + 3 (the `DMA ring:` lines and dumps), C level 3 (`blocks=0`, arena read rate = `22 MB / mm`) | 6 |
| gtests | `MoeChunkReplay` (11 lines) + `MoeLayerM1` from #86 | 5 |
| **total** | | **≈ 60–70** |

Three variants, one control, every number an A/B inside the sitting.
If #86 is not merged in time, run #87 standalone (35–40 min) and leave
#91 + #86 as their own ≈ 45 min sitting; do not stack #87's branch on
#86's unmerged branch (linear history into `htp_moe`, contract §5).

## 5. Risks

| risk | how the handoff makes it visible |
|---|---|
| Thermal / DVFS drift between and within sittings (rule 9) | A first; B must equal A on tok/s and on the existing `M==1` columns (`mm`, `acc`, `gather`) before the new line is read; level 3 (warm, min-of-5) beside level 2 (cold) is itself the (e)/(g) discriminator, so drift is data, not noise, here |
| Stale skel (rule 3): the app expects `HTP_MOE_N_STAGES` + new slots | `AEE_EBADPARM` hint already names the slot count (`htp_compute_ops.cpp:1521-1535`); the handoff's install step pushes the skel and `md5sum`s it on the device before any run |
| Completion times are brackets, not instants | the line prints `busy=lo..hi` and the rate as a range; the replay at `pace=0` gives the point value for the same list; the attribution table quotes the range, never the midpoint |
| The trace itself changes the timeline (extra `is_done` loads) | the `M==1` existing columns at level 3 must match #77 B (`mm 746.6, acc 258.8, gather 122.6`) within noise on the same unit class; the host proves off-path inertness, the device proves on-path cost is ≤ a few µs |
| Unit mismatch (rule 13): #77's 16–18 GB/s came from `R3CY205ZMND` | the attribution is in percentages of the same sitting's `dsp_us`; absolute GB/s are unit-tagged |
| Address space / `.bss` | +14 KiB static, stated in the commit; no arena, no heap |
| The replay's DDR pattern is not the weight arena's page layout | `fresh=1` rotates through 32 regions of the 256 MiB chunk (pattern-filled, same ION path as the model's chunks, `unittest_hvx_dma_probe.cpp:119-155`); if `fresh` moves the number, the follow-up measures on the real model arena by reading the `[HTP-DMA]` dump at level 2 |

## 6. Docs to update

* **BENCHMARK.md:** B and C rows (6 + 6 cells, NPU) under the sitting's
  date; a new side table "#87 DMA ring (M==1, level 2 / 3)" with the
  `DMA ring:` line fields and the 11 `DMA_REPLAY` cells; artifact rows for
  the B skel / app / `unittest_hvx_dma_probe` md5s (table and device).
* **LEDGER.md:** §2 verdict for ③/⑥ rewritten from the attribution table
  (which letter, what percentage, the named fix); rule 11 refined with the
  engine rate *while running* vs the call-averaged number; §3 ⑥ step 2
  filed as the fix issue; if (g) confirms, a new rule on cold vs warm
  calls (`NNTR_HTP_PROFILE=3` min-of-5 hides a per-call cost the model
  pays every token) next to rule 15.
* **Contract §2 wall 2 sentence:** "Cause unknown … measurement C decides"
  → replaced by the attribution's one line.
