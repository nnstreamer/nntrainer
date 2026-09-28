# 90 — Two-reader DDR probe: the DSP side becomes a tag-validated DMA-ring DDR stream

Issue #90 (p2, tracker #76, LEDGER ④ / rule 12, contract §3.2 Q11 / ⑫).
Contract `docs/plans/0001-htp-moe-decode-agent-system.md`. Ground truth
read for this plan: `docs/measurements/77-first-handoff.md` §④ (the
invalid DSP side, the clean CPU side), LEDGER §1 rules 12, 26, 27, 28,
30, 32, 34 and §3 items ④ ⑫, `docs/plans/77-first-handoff.md` §3.5 (the
probe as built), `docs/plans/100-dma-chunk-list.md` §3.2 and
`docs/plans/117-m1-gemv-vtcm-feed.md` (the `f2` list and the rule-30/32
anchor cell), and the code on `htp/89-generation-last64` (= `htp_moe` +
PR #124 + PR #121 + #89; every `path:line` below was read there).

**What went wrong in #77, in one paragraph.** `TwoReaderDdr`
(`test/unittest/unittest_hvx_dma_probe.cpp:334-452`) streams the DSP side
through the `dma_probe` entry (`test/htp/nntr_hvx_dma_probe.c:113-193`):
per worker, one `dmstart` per 1 MiB descriptor, spin on `done`, 159 passes
over a 256 MiB chunk. It printed 3265 GB/s for 85 GB in 26 ms — 40× the
LPDDR5X peak (≈ 77–85 GB/s on this SoC), i.e. ≈ 0.6 µs per "completed"
descriptor. The entry's checksum (`:166-170`) reads only worker 0's first
VTCM slot, so it proves one descriptor of one pass landed and nothing
about the other 40 000. The same entry's 2–3-pass `DMA_PROBE` rows read
79–117 GB/s, also above peak, and rule 28 already established that no
tag-validated per-call list reproduces them (`c_star` = 0.36 × probe iii).
The cause is not separated (candidates: a `done` bit read before the
engine retired and the next `dmstart` erroring out instantly; a stale
descriptor line; an engine that skips a descriptor it considers done) and
this plan does **not** chase it: it retires `dma_probe` as the two-reader's
DSP side and uses the entry whose bytes are proven to land.

## 1. Goal and gate

Acceptance from the issue, made measurable. No model-path code moves; the
probe rides along any later sitting on that sitting's own skel.

| gate | read from | pass |
|---|---|---|
| **verdict cell** | the `DDR_TWO_READER` line of `unittest_hvx_dma_probe --gtest_filter='*TwoReaderDdr*'`, `cpu_threads=8` row | `valid=y`, `dsp_alone` in **30–120 GB/s**, `aggregate = cpu_with + dsp_with` printed, compared with **45 GB/s** (contract §3.2) |
| **validity (rule 12), inside the test** | the same line | `INVALID` (and `valid=n`, `EXPECT` failure) when `dsp_alone` or `dsp_with` > 150 GB/s, when any DSP cell's `checksum_ok=n` (tag sum ≠ host simulation), or when the long stream departs > 20 % from the same run's 20-call `f2` reference cell (`dsp_ref`) |
| **CPU side (unchanged, plan 77 §5 escape)** | `DDR_CPU` lines | `cpu_alone(8) ≥ 25 GB/s`, else "inconclusive (little cores)" |
| **components for the split rule** | `DDR_TWO_READER` rows at `cpu_threads=8`, `4`, `1` | each row carries `cpu_alone cpu_with dsp_alone dsp_with aggregate`; the supervisor writes the ⑫ rule from the `8` row (§3.4 formula) |
| **host** | `bash test/htp/host/run_host_checks.sh` | new line `TWO READER CELL SOUND (bytes=… footprint=… MiB)` and the old `SKEL REPLAY MATCHES TAG SIMULATOR (16 cells)`, `ALL CHECKS PASS`, `WORKER POOL LANES OK`; `tools/htp_syntax_check.sh` rc 0 |
| **skel** (rung 2, workstation) | `test/htp/build.sh` | `UNDEFINED SYMBOLS OK`; **the skel source does not change** (§3.1), so the md5 equals the sitting's own skel — recorded in the PR anyway |
| **app / gtest** (rung 3, workstation) | `ndk-build unittest_hvx_dma_probe` | binary exists, md5 in the PR and the handoff |
| **standing gates** | the sitting's E2E cells | untouched by this issue (probe entry and gtest only): prefill ≥ −5 % of A and text identical are the host sitting's gates, not this cell's |
| **device time** | the handoff step | ≤ 5 min: 7 stream cells × ≈ 1.5 s + fixture ≈ 40 s of run, once cold and once after the E2E block |

The `DDR_TWO_READER` prefix fields stay exactly as #77 printed them
(`cpu_alone= dsp_alone= cpu_with= dsp_with= aggregate= cpu_threads=
dsp_workers= chunk_bytes=`); new fields append after `chunk_bytes`.

## 2. Where it lives

| file | what changes | lines read |
|---|---|---|
| `test/unittest/unittest_hvx_dma_probe.cpp` | `TwoReaderDdr` (`:334-452`): the DSP side (`:401-433` sweep + `dsp_stream`) is replaced by a `dma_replay` stream of the `f2` cell (§3.1); the CPU side (`CpuStreamer` `:353-390`, `stream_xor` `:285-318`) is kept and run at 8 / 4 / 1 threads; the concurrent block (`:438-443`) loops over the three thread counts; the print (`:446-451`) gains the appended fields; one diagnostic `DDR_DSP_LEGACY` pair (§3.3). The `f2` list comes from `nntr_moe_dma_cell(12, …)` (`test/htp/nntr_moe_dma_plan.h:438-520`, case 12 at `:504-509`) exactly as `MoeChunkReplay` builds its cells (`:658-716`), with no weight registration and no trace (the `f2` cell does not derive from the traced list) | `:60-70` pattern, `:90-160` fixture (two 256 MiB chunks, falls back to 128 MiB), `:195-233` `PassesFor` / `Run`, `:334-452`, `:605-645` `cell` lambda, `:658-716` #100 cells |
| `test/htp/nntr_moe_dma_plan.h` | header-only additions the gtest and the host check share: `nntr_two_reader_cell()` (returns the `f2` list + `fresh = 1`, `workers = 1`, `calls_per_chunk = 500`, `chunks_max = 8`, `min_wall_us = 1 200 000`), `nntr_two_reader_footprint_bytes(n_regions, region_bytes)` (= `n_regions × region_bytes`, the distinct source the rotation touches), and `nntr_two_reader_verdict(bytes, us, ref_gbs, checksum_ok)` (the INVALID rule of §1 as a pure function) | `:35` `PLAN_MAX`, `:91` `region_bytes`, `:261` `REPLAY_MAX_DESC_BYTES`, `:321` `depth2`, `:354` `columns`, `:426-520` cells, `:534` `nntr_dma_pattern`, `:570-620` `tag_sum` |
| `test/htp/host/two_reader_host_check.c` (new) + `run_host_checks.sh` block after `:134` | compiled like `dma_replay_host_check` (`run_host_checks.sh:121-134`: the skel's `nntr_hvx_dma_probe.c` as-is against `replay_stub/`); §3.2 says what it asserts | `dma_replay_host_check.c:1-140`, `replay_stub/hexagon_protos.h:38-52` (`replay_stub_dma_run` lands a descriptor by `memcpy`) |
| `test/htp/host/replay_stub/hexagon_protos.h` | `replay_stub_dma_run` (`:38-47`) additionally accumulates `replay_stub_bytes_landed += n × row_size` and marks the 4 KiB source pages it read in a bitmap (`replay_stub_pages`, 256 MiB / 4 KiB = 65 536 bits); both reset by the check. Host-only; the device never compiles this file | `:38-52` |
| `test/htp/nntr_hvx_dma_probe.c` | **no change.** `dma_replay` (`:386-607`) already has everything the stream needs: `fresh` rotation (`replay_src` `:238-246`: region `(call × E + e) mod n_regions`), `calls` (`:543`), qtimer-derived `res[0]` µs (`:585`, `HAP_perf_qtimer_count_to_us`), `res[2]` bytes per call, `res[12]` tag sum (`:601-606`). `dma_probe` (`:113-193`) stays for `DmaProbeShapes` and the diagnostic pair, with rule 28 as its caveat | `:113-193`, `:196-260`, `:386-607` |
| `test/htp/nntr_hvx.idl` | **no change** (`dma_probe` `:424`, `dma_replay` `:466`); no stub regeneration, no `HtpComputeOps` consumer, no quantizer tag, no loader check, no `NNTR_HTP_PROFILE` stage table, no `tools/htp_fc_report.py` | `:395-480` |
| `test/jni/Android.mk` | **no change**: module `unittest_hvx_dma_probe` (`:954-975`) already links the stub and the plan headers | `:954-975` |
| `docs/measurements/90-two-reader-ddr-probe.md` (new) | the ride-along step (§4 step 5) | — |

Consumers that must move with a changed contract: none — the FastRPC
contract is untouched. The only cross-file coupling is
`nntr_two_reader_cell()` being the single source for the gtest and the host
check (the same pattern as `nntr_moe_dma_cell` for the #100 cells).

## 3. Design

### 3.1 Chosen: the DSP reader is `dma_replay` on the `f2` list, `fresh = 1`, ≥ 1.2 s, tag-validated

The DSP side of the split, if it ever exists, reads expert weights the way
decode reads them today: whole-matrix slabs through the production DMA
ring into VTCM (the feed default since PR #118, `f2` shape). So the probe
streams exactly that:

* **List:** `f2` = 8 descriptors per call (gate_up 57 344 B × 64 rows,
  down 32 768 B × 56 rows per expert, 4 experts, depth 2), 22.02 MB per
  call, `dst` strided = contiguous in VTCM (5.25 MiB window below the HMX
  config block; the fixture's VTCM is the session's, no heap).
* **Footprint:** `fresh = 1` rotates the source over `n_regions = 32`
  regions of 5.25 MiB (`slots = 256 MiB / 5.25 MiB = 48 → min(47, 32)`,
  `:409-415`), so 8 consecutive calls touch **168 MiB of distinct DDR**
  before any byte repeats — 20× VTCM (8 MiB) and far beyond any L2. With
  the 128 MiB fallback chunk `n_regions = 23`, footprint 121 MiB; the
  line prints `regions=` so the reader sees which. The 22 MB of one call
  alone already exceeds VTCM + L2, so even `fresh = 0` could not be
  cache-resident; `fresh = 1` is there so the tag check (next bullet)
  cannot be satisfied by a stale window.
* **Proof the bytes landed:** `res[12]` is the tag sum over every push's
  VTCM window after the last call, and the tag byte names the 4 KiB
  source page (`nntr_dma_pattern` `:534-539`); the gtest holds it against
  `nntr_moe_dma_tag_sum(items, n, calls, fresh, res[7], region, samples)`
  (`:570-620`), which for `fresh = 1` predicts the regions of call
  `calls − 1`. A stream whose last call did not land — or landed the
  previous call's regions — prints `checksum_ok=n` and the line is
  INVALID. This is the check `dma_probe` never had.
* **Bounded runtime, one blocking call ≤ 0.4 s:** the stream is
  `chunks ≤ 8` calls of `dma_replay(..., workers=1, load=0, pace=0,
  fresh=1, gap_us=0, calls=500, res, 13)`, each ≈ 0.3–0.35 s at 31–37 GB/s
  (0.55 s at a pathological 20 GB/s), summed until the summed `res[0]`
  ≥ 1.2 s or 8 chunks; the CPU streamer's window brackets all of them
  (the ≈ 100 µs FastRPC gap between chunks is < 0.1 % of the window). No
  1.5 s single FastRPC call, no watchdog exposure, `res[0]` (uint32 µs)
  cannot wrap. Alternating the arena chunk per call (`rep & 1`, as
  `Cell` does at `:240`) doubles the footprint to 336 MiB.
* **Rate:** `gbs = Σ(res[2] × res[1]) / Σ res[0] / 1e3` — bytes over
  qtimer-derived µs, the same arithmetic as the `DMA_REPLAY` lines
  (`:620-622`), so `dsp_alone` is directly comparable with the sitting's
  anchor (`DMA_REPLAY workers=1 load=0 pace=0`, 31.2 / 37.3 GB/s per
  unit, rules 30/32/34) and with `f2` (31.6 / 37.4).
* **Reference cell in the same run:** before the long stream, one
  `calls = 20, fresh = 0` `f2` call — the #100 cell verbatim — printed as
  `DDR_DSP_REF gbs=…`. It has been read on two units in four sittings
  (31.7 / 31.6 / 37.4 / 37.5) with `checksum_ok=y`; the long stream must
  sit within ±20 % of it or the line is INVALID (a long stream that reads
  2× the 20-call cell is the #77 defect in a new coat, not a discovery).
* **INVALID rule, in one pure function** (`nntr_two_reader_verdict`):
  `valid = checksum_ok && gbs ≤ 150 && |gbs / ref − 1| ≤ 0.20`; the
  gtest prints `valid=y|n` and `EXPECT_TRUE(valid)`, the host check feeds
  it the #77 pair (85 362 475 008 B / 26 140 µs → `n`) and a nominal pair
  (44 GB / 1.35 s with ref 32.6 → `y`).

**CPU side, unchanged code, three thread counts.** `CpuStreamer` at
8 threads (`NNTR_NUM_THREADS=8`, the contract's run condition and the
maximal contention), 4 (what a one-expert CPU GEMV would realistically
use beside the ARM remainder) and 1 (a single core's stream, the
lower bound of the split's CPU draw). Each `cpu_alone` runs 1.5 s (was
2.0; three of them now). The XOR is printed as before.

**Grid (7 stream cells, ≈ 12 s of streaming):**

| # | cell | prints | decides |
|---|---|---|---|
| 1–3 | `cpu_alone` t = 8, 4, 1 | `DDR_CPU threads=t …` | the CPU's own stream and the little-core escape |
| 4 | `dsp_ref` (f2, 20 calls, fresh 0) | `DDR_DSP_REF …` | the in-run anchor for the INVALID rule |
| 5 | `dsp_alone` (f2 stream) | `DDR_DSP workers=1 cell=f2 fresh=1 chunks= calls= bytes= us= gbs= regions= checksum_ok=` | the DSP's DDR rate through the ring; the rule-12 number |
| 6–8 | both, t = 8, 4, 1 | `DDR_DSP …` + `DDR_CPU …` + `DDR_TWO_READER … cpu_threads=t dsp_workers=1 chunk_bytes=… dsp_stream=ring cell=f2 fresh=1 dsp_ref=… valid=y|n` | `dsp_with`, `cpu_with`, `aggregate` per thread count |

`dsp_workers` stays 1: the production ring is one dmlinked chain on one
thread, and rule 30 says worker count moves single-queue issue rate, not
DDR bandwidth; the split would not change that.

### 3.2 Host check: the cell plan is sound before any phone time

`two_reader_host_check.c` runs the skel's `nntr_hvx_dma_replay` on the
host (same build line as `dma_replay_host_check`, `run_host_checks.sh:121-134`)
with `nntr_two_reader_cell()` and asserts:

1. **Byte accounting:** for a scripted `calls = 24` (three full rotations
   of 32 regions), `replay_stub_bytes_landed == res[2] × res[1]
   == 24 × 22 020 096` — bytes the gtest will divide by time are bytes
   the descriptors carried.
2. **Footprint:** the page bitmap the stub fills counts ≥ 168 MiB / 4 KiB
   distinct source pages over the 24 calls, and ≥ 121 MiB with a 128 MiB
   arena (`n_regions = 23`); `nntr_two_reader_footprint_bytes` returns
   the same numbers.
3. **Tag:** `res[12] == nntr_moe_dma_tag_sum(...)` for `calls ∈ {1, 20,
   500}` with `fresh = 1`, and the tag of call 500 differs from the tag
   of call 499's regions (a one-call-late landing is caught).
4. **Verdict function:** the two pairs of §3.1 (`n` for #77's, `y` for
   the nominal) plus the ±20 % edge (`ref × 1.21 → n`, `ref × 1.19 → y`).
5. **Time bound (arithmetic only):** `chunks_max × calls_per_chunk ×
   22.02 MB / 20 GB/s ≤ 4.5 s` — the worst-case DSP wall the phone step
   can take per stream cell.

Pass line: `TWO READER CELL SOUND (bytes=528482304 footprint=168 MiB)`.
The device's timing is not modelled (same caveat as every check in that
script).

### 3.3 Diagnostic, not a gate: the legacy pair

Two `dma_probe` calls on shape (i), workers 1, `passes = 2` and
`passes = 64`, printed as `DDR_DSP_LEGACY passes=2 gbs=…` /
`passes=64 gbs=…` (≈ 7 ms + ≈ 0.2 s if honest, ≈ 20 ms if not). Whether
the inflation scales with `passes` is the one bit that separates "the
engine skips repeated descriptors" from "the probe over-reports at any
pass count"; it costs nothing and closes rule 12's "why" in the LEDGER
without an issue. It never enters the verdict.

### 3.4 The rule the supervisor writes from the filled line

Per MoE layer at M = 1: 4 experts × 5.505 MB. NPU-only time
`T0 = 22.02 MB / dsp_alone`. Split (3 experts on the DSP, 1 on the CPU,
one sync per layer) `T1 = max(16.52 MB / dsp_with, 5.505 MB / cpu_with)
+ sync`. The split pays only if `T1 < T0`, i.e.

* `dsp_with ≥ 0.75 × dsp_alone` (the DSP may lose at most a quarter of
  its rate under the CPU's stream) **and**
* `cpu_with ≥ dsp_with / 3` (the CPU finishes its expert inside the
  DSP's three) **and**
* `aggregate ≥ 45 GB/s` (contract §3.2's threshold, kept as the
  headline number for Q11).

The supervisor fills `saving/token = 22 × (T0 − T1)` ms from the
`cpu_threads=8` row and reads the `4` and `1` rows as the sensitivity to
how many cores the CPU expert would take. #77's CPU side (−41 % at 8
threads) already says the sum is not additive; what this sitting adds is
whether the **DSP** keeps its rate, which is the term that decides.
Example with #77's CPU numbers and a hypothetical `dsp_with = 0.9 ×
37.3`: `T0 = 590 µs`, `T1 = max(492, 139) = 492` → −2.2 ms/token, pays;
with `dsp_with = 0.6 × 37.3`: `T1 = 738 > 590`, does not pay whatever the
aggregate reads.

### 3.5 Rejected: a direct HVX vector-read stream as a second DSP reader

A new IDL entry (or a `dma_probe` mode) in which pool workers read the
arena with `vmem` loads and fold a checksum. Rejected because (a) it is
not the reader the split would use — the feed default reads VTCM, and
the direct read is latency-bound at 21–27 GB/s (rules 26, 27), so a
split that read the arena directly would lose to NPU-only before any
contention; (b) it needs an IDL change → stub regeneration → a new skel
per handoff, which turns a ride-along into a variant (rule 3, rung 2 on
the workstation); (c) the question it would answer (does HVX direct read
degrade under CPU load) is moot once the feed is default. If Q11 is ever
decided in favour of the split, the CPU-side GEMV's own DDR draw is
measured in situ (`NNTR_M0_PROFILE`), not by a probe.

Also rejected: fixing `dma_probe`'s per-pass loop and keeping it as the
DSP side. Its rate would still lack a landing proof per descriptor and
rule 28 already bars it from being a ceiling; the replay is both the
proven reader and the production shape.

## 4. Steps

Branch `htp/90-two-reader-ddr-probe` off `htp_moe` head (rebased onto
whatever the user merges before the sitting; the branch touches only
`test/unittest/unittest_hvx_dma_probe.cpp`, `test/htp/nntr_moe_dma_plan.h`,
`test/htp/host/**`, `docs/measurements/90-*.md`). Rungs per
`.claude/skills/hexagon-gates`; the Mac runs rungs 0–1 in the container,
rungs 2–3 need the workstation.

| step | what | gate | rung |
|---|---|---|---|
| 0 | `nntr_two_reader_cell` / `_footprint_bytes` / `_verdict` in `nntr_moe_dma_plan.h`; `replay_stub` byte counter + page bitmap; `two_reader_host_check.c`; the `run_host_checks.sh` block | `TWO READER CELL SOUND (bytes=528482304 footprint=168 MiB)`, `SKEL REPLAY MATCHES TAG SIMULATOR (16 cells)` unchanged, `ALL CHECKS PASS`, `WORKER POOL LANES OK`; `clang-format-14` diff clean | 0, 1 (Mac ok) |
| 1 | `TwoReaderDdr` rewritten per §3.1 (CPU streamer kept, three thread counts, `f2` stream, ref cell, verdict, appended fields, legacy pair); `DmaProbeShapes` / `MoeChunkReplay` untouched (`git diff` shows no hunk in `:322-332` or `:466-`) | the file compiles in the host tree's syntax pass — `tools/htp_syntax_check.sh` rc 0 — and `g++ -fsyntax-only -std=c++17 -DNDK_BUILD=1 -I test/htp -I test/htp/generated -I nntrainer/tensor -I test/htp/host/replay_stub …` on the gtest (arm_neon path excluded; the NEON block is unchanged); `ninja -C build` green | 0, 1 (Mac ok) |
| 2 | Workstation: `./test/htp/build.sh` (nothing in `test/htp/*.c` or the IDL changed; run to prove it), md5 = the sitting's skel | `UNDEFINED SYMBOLS OK (<n> runtime imports)`; md5 recorded | 2 (workstation) |
| 3 | Workstation: `ndk-build … unittest_hvx_dma_probe unittest_hvx_mm_u8i4 -j8`, md5s recorded; `readelf -d` not needed (no app change) | binaries exist; md5 in the PR body | 3 (workstation) |
| 4 | PR into `htp_moe`: `[test] Two-reader DDR probe: DSP side streams the f2 list through the ring, tag-validated` (test commit) + docs commit (this plan's handoff). PR body: rungs run, md5s, "no DSP source / IDL change; rides any sitting's skel ≥ `d79c0efe`" | review; `state:review` | — |
| **5 (device, unavoidable)** | **Handoff `docs/measurements/90-two-reader-ddr-probe.md`**, a ride-along step for the next sitting (no device time exists now — the step is written so it costs ≤ 5 min and needs only the sitting's skel plus this gtest binary). Variants: **none of its own** — the sitting's A is the E2E control it rides with; the probe's cells are the grid of §3.1. Placement: **once cold, as the first thing after the sitting's anchor cell** (`*MoeChunkReplay*`, rules 30/32), and **once after the E2E block** (warm) — two `DDR_TWO_READER` triplets, same binary. Commands: `adb shell "cd $T && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*TwoReaderDdr*'" 2>&1 \| tee $W/logs/two_reader_{cold,warm}.log \| grep -E 'md5\|^DDR_\|PASSED\|FAILED'`; `therm` before and after each; `grep -c '^DDR_TWO_READER' = 3`, `grep -c 'valid=n' = 0`, `grep -c 'checksum_ok=n' = 0` | the §1 gate: `valid=y` × 3, `dsp_alone` 30–120 and within ±20 % of `dsp_ref`, `dsp_ref` within ±15 % of the sitting's anchor (else the note says which drifted, rule 30), `cpu_alone(8) ≥ 25`; the filled table gives `aggregate` with its two components for t = 8 / 4 / 1, cold and warm | 4 (user) |
| 6 | Supervisor: LEDGER ④ closes with the three-condition rule of §3.4 evaluated, rule 12 gets its "why" from the legacy pair, BENCHMARK side table row; Q11 goes to the user with the number | — | — |

Estimated device time for step 5: **≈ 2 min cold + ≈ 2 min warm**
(fixture: two 256 MiB `rpcmem_alloc` + pattern fill ≈ 3 s on the CPU; 7
stream cells ≈ 12 s; legacy pair < 0.3 s; the rest is adb and `therm`).

## 5. Risks

| risk | how the handoff makes it visible |
|---|---|
| **The replay stream's rate itself depends on the unit and the sitting** (rules 30/32/34: 31.2 vs 37.3 GB/s, ≈ 22 % drift once) | `dsp_ref` (the #100 `f2` cell, 20 calls) is printed in the same run and the sitting's anchor cell runs just before; `dsp_alone` is read as a ratio to both. A `dsp_alone` far from `dsp_ref` is INVALID, far from the anchor is noted (the two cells differ by ≤ 2 % historically) |
| **DVFS / bus vote**: the DSP stream may lift the DDR clock and the CPU threads may lift the CPU clock; `cpu_alone` at 1 thread is the most exposed | the three thread counts and the two placements (cold, warm) bracket it; the `DDR_CPU` lines carry `s=` and `passes=` so a governor stall shows as a short window |
| **Thermal drift between the cold and the warm triplet** | `therm` before/after each; the filled table keeps both triplets side by side and the verdict is read from whichever agrees with the anchor within 15 %; a > 10 % gap between them is written as a note, not averaged |
| **Stale skel / stale gtest**: a skel before `d79c0efe` (#100's `dma_replay` parse rules) returns `AEE_EBADPARM` on the `f2` list | the gtest prints `DDR_DSP … skipped err=0x8000040e` and the line reads `valid=n`; the md5 line is grepped first, as every handoff since #94 |
| **Address space / VTCM budget**: two 256 MiB arena chunks + the session's 8 MiB VTCM; the `f2` window is 5.25 MiB below the HMX config block | unchanged from #77 / #100 (the fixture and the cell already ran on both units); the fallback to 128 MiB chunks prints `chunk_bytes=134217728` and the footprint check accepts 121 MiB |
| **The CPU streamer's window vs the chunked DSP calls**: the CPU runs across ≤ 8 FastRPC calls with ≈ 100 µs gaps | gaps are < 0.1 % of ≥ 1.2 s; the line prints `chunks=` and the summed `us=` so the reader can bound it |
| **Host-vs-device gap the plan cannot close**: the host stub lands descriptors instantly, so nothing here predicts the rate or whether the DSP loses under contention — only that the bytes, the footprint and the tag are what the gtest assumes | that is the sitting's job; the plan's device step is unavoidable and marked so |
| **Rule 28 in reverse**: a reviewer may expect the two-reader DSP number to match `DMA_PROBE` (60–117) | the plan's band is 30–120 with the ±20 % tie to `dsp_ref`; `DMA_PROBE` is not a ceiling and the legacy pair documents why |

## 6. Docs to update (after the sitting)

* `docs/htp_moe/BENCHMARK.md`: a new side table "④ — two-reader DDR
  (#90)" with the cold and warm triplets (`cpu_alone / dsp_alone /
  cpu_with / dsp_with / aggregate` × `cpu_threads` 8 / 4 / 1, `dsp_ref`,
  the sitting's anchor, unit, skel md5), replacing the #77 line at
  `:237` ("DSP side … invalid (#90)") with the valid pair; the artifact
  row of the host sitting gets `unittest_hvx_dma_probe`'s new md5.
* `docs/htp_moe/LEDGER.md`: item ④ → **measured**, with the §3.4 rule
  evaluated and the projected `saving/token`; item ⑫ gets its
  precondition answered (pays / does not pay at the measured `dsp_with`);
  rule 12 gains the "why" from the legacy pair (or "cause still not
  separated" if the pair is flat); a new rule if the DSP loses more than
  the CPU under contention (device disagreed with the "DSP has priority
  on the NoC" reasoning) or if it does not.
* Contract §3.2 Q11: the supervisor hands the user the three numbers and
  the §3.4 rule; the decision stays the user's.
* `docs/htp_moe/guide/03-performance.html`: one sentence on the ceiling
  track (guide writer, only after the LEDGER row exists).

## 7. What is not done here

* No CPU+NPU split code, no per-layer sync design, no CPU-side GEMV
  measurement (Q11 is the user's; ⑫ is its own issue if the rule says
  "pays").
* No fix to `dma_probe`'s loop and no removal of `DmaProbeShapes`; rule 28
  remains its caveat and the legacy pair is diagnostic only.
* No IDL, skel, `HtpComputeOps`, quantizer, loader, profile-table or
  `tools/htp_fc_report.py` change — the FastRPC contract is untouched.
* No sitting of its own: the step rides the next handoff (candidates: the
  ⑨ wiring sitting or any control re-measure), which is why it must fit
  in ≤ 5 minutes and one existing gtest binary.
