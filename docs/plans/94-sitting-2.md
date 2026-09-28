# 94 — Sitting 2 on `R3CY10WM83Y`: anchor cells (12 × A) + the #87 trace / replay, variant C only if PR #86 is in `htp_moe`

Issue: dlwlzzero/nntrainer#94 (tracker #76; folds #91). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md` §1.1, §4.2–§4.3. Sources
read for this plan: LEDGER §1 rules 9–16 and §3 ⑥ ⑮ ⑯ ⑰, BENCHMARK.md
"Artifacts" and its #94 rebuild note, `docs/plans/77-first-handoff.md`
(§3.2, §4 step 8), the filled
`origin/htp/77-first-handoff:docs/measurements/77-first-handoff.md`
(Results, Deviations 2/3/6), `docs/plans/87-dma-in-situ-gap.md` (§0, §1,
§3.2, §3.3, §4 step 7), and on `origin/htp_moe` @ `49cc60f2`:
`test/unittest/unittest_hvx_dma_probe.cpp`, `htp_compute_ops.cpp`,
`test/htp/nntr_hvx.idl`, `test/htp/nntr_hvx_mm_u8i4.c`.

**This plan is assembly, not code.** No source file changes. The
implementer produces one branch with one docs commit (the handoff and the
prompt file), rebuilt artifacts staged on the workstation, and sets
`state:needs-measurement` in the same run. If any build rung fails, that
is a defect issue against #93 (or #86 if merged), not a reason to patch
here; if an env var or line the handoff needs turns out not to exist, that
is a bug against #87 — file it, do not add it on this branch.

Names (from the issue's acceptance criterion; the orchestrator's
shorthand "94-sitting-2" means the same artefacts):
branch **`htp/94-sitting2-anchor-trace`**, handoff
**`docs/measurements/94-sitting2-anchor-trace.md`**, staging
**`/local/mnt/workspace/htp_moe/94/`**.

## 1. Goal and gate

Acceptance criterion (issue), made measurable:

| deliverable | what counts as done |
|---|---|
| Handoff on `htp/94-sitting2-anchor-trace`, `state:needs-measurement` this cycle | estimated minutes at the top ≤ 90 (this plan budgets ≈ 50, ≈ 56 with C); every artifact row has an md5 from the build that is pushed (rule 14) and the commit it came from; every command is copy-pasteable with local `adb` (no SSH bridge this time) |
| (a) plan 87 attribution table filled | one row per hypothesis a–g of plan 87 §3.2, each with its share of the M==1 `dsp=` µs/call (level 3 = warm, level 2 = cold) and of the isolated-vs-in-situ gap, plus one sentence naming the first wall-2 fix issue — the data for every row comes from a named line of this sitting (§3.4 below) |
| (b) BENCHMARK.md "now" replaced | the 12 A cells (CPU `q40` and NPU `q40-qs4cx-wh` × G 64 / 512 / 1024 × 2 runs, prompt 512, `NNTR_NUM_THREADS=8`, profile unset) from the `prefill:` / `generation:` lines; the supervisor records the `R3CY10WM83Y` ÷ `R3CY205ZMND` ratio per cell (rule 13) — the #77 numbers are pre-filled as the reference column |
| (c) C verdict (only if C ran) | C vs A decode tok/s at G 64 / 512, text identical to A, `[HTP-PROFILE]` M==1 row shows `blocks=0 m1_gemv=<calls>/<calls>` |

Standing gates in the same sitting:

* **A denominator.** Prefill and decode of the `htp_moe` head binary with
  `NNTR_HTP_PROFILE` unset are the denominators for everything else. The
  −5 % prefill gate is not applied to B (B *is* A's binary with an env
  var; a profiled run is never read for tok/s, rule 1) — the post-B
  re-run of A (§3.3 block 4) shows that the sitting has not drifted.
* **Text.** NPU run1 = run2 in every cell (greedy decoding, `do_sample`
  false); CPU run1 = run2; CPU and NPU texts differ by construction
  (different expert quantisation, both recorded). For C: text identical
  to A (the GEMV's int32 accumulator is bit-identical to the HMX path,
  plan 80). `NNTR_L2_DIFF` is not applicable to `QS4CX_WH` (rule 6).
* **M>1 row unchanged.** The `[HTP-PROFILE]` `M>1` row's existing columns
  at level 2 must be within noise of #77 B scaled by the unit ratio; the
  trace hooks are under `hexkl_probe_on` only, so profile-off runs pay
  nothing (plan 87 §2, host line `PROFILE OFF: TRACE UNTOUCHED`).
* **Void rules (rule 14).** md5 mismatch with pushed-build provenance
  intact: a note in the A/B blocks (control-only), a void once C is in.
* `generation(last 64)` is `n/a` in every row until #89 lands (rule 16):
  `Lfm2MoeCausalLM` reports from `causal_lm.cpp:699`, which prints only
  `prefill` / `generation` / `total` / `peak memory`.

## 2. Where it lives

Nothing in the model path changes. What the handoff relies on, verified
on `origin/htp_moe` @ `49cc60f2` (code-identical to `750428e8` — the
diff is `docs/htp_moe/BENCHMARK.md` and `LEDGER.md` only):

| thing | where | fact the handoff uses |
|---|---|---|
| Report lines | `Applications/CausalLM/models/causal_lm.cpp:699-712` | `prefill: <n> tokens, <ms> ms, <tps> TPS`, `generation: <G> tokens, …`, `total:`, `peak memory: <KB> KB`; no `generation(last 64)` (#89) |
| G, greedy, exact length | `nntr_config.json` `num_to_generate`; `generation_config.json` `do_sample`; `bad_word_ids` (plan 77 §2.1/§3.2) | edits on the device with `sed`; `bad_word_ids: [124900]` (EOS) guarantees `generation: G` |
| Profile level, trace count | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:483-486` (`NNTR_HTP_PROFILE`, `NNTR_HTP_DMA_TRACE` default 3) | `NNTR_HTP_PROFILE=2` = one timed call per layer (cold); `=3` = 5 repeats, fastest kept (warm) |
| `[HTP-PROFILE]` M==1 row | `htp_compute_ops.cpp:572-600` (`dump()`, stderr at exit) | `K=2048 N=2048 M==1 calls=… host=… dsp=… us/call transport=… [quant … gather … drain …+… mm … blocks=…]`; **no `m1_gemv=` field on `49cc60f2`** — that field is PR #86's (`HTP_MOE_T_PATH` placeholder comment at `:171`) |
| `weight DMA:` line | `htp_compute_ops.cpp:604-609` | `weight DMA: <KB>/call, first <KB> took <us> = <GB/s>; averaged over the call <GB/s>` (the `first … took 0 us` artefact is explained in plan 87 §0 ii) |
| `DMA ring:` line (the "second line", #87) | `htp_compute_ops.cpp:618-646` | `DMA ring: desc=<n>/call waits=<n> (blocked <n>) wait=<us> us [act <us> gu <us>+dn <us>] busy=<lo>..<hi> us -> engine <lo>..<hi> GB/s  depth max=<n>  first expert ready at <us>  last issue at <us> of <dsp_us>`; `busy=n/a (trace truncated past 512 pushes)` when a prefill call overflows `HEXKL_DMA_TRACE_MAX_PUSH` |
| `[HTP-DMA]` per-descriptor dump | `htp_compute_ops.cpp:345-400` (`dmaTraceOrdinal`, `dumpDmaTrace`), call site `:1688-1710` | printed **at call time to stderr**, first `NNTR_HTP_DMA_TRACE` (3) timed calls of each bucket `(K, N_out, M==1)` → 3 M==1 dumps (the first decode token's layers 0–2) and 3 M>1 dumps; header `[HTP-DMA] call=<k> M=<m> rep=<r>/<r> desc=46 waits=… blocked=… depth_max=… dropped=0 end=<us>`, then one `push k=… t=… kind=gate|up|down|act|copy e=… c=… bytes=… row=… nrows=… stride=… idx=… depth=… done=<lo>..<hi>` line per descriptor and one `wait k=… site=act|gu|dn|copy_in|copy_out idx=… t=<in>..<out> blocked=y|n` per wait. At level 3 `rep=5/5` is the warm call; at level 2 `rep=1/1` the cold one |
| Trace inertness | `test/htp/nntr_hvx_mm_u8i4.c` (`hexkl_probe_on` set only inside `*_timed`), host lines `PROFILE OFF: TRACE UNTOUCHED`, `PROFILE ON/OFF BYTE-IDENTICAL` | the same skel and app serve A (untimed entry) and B (timed entry) |
| Replay gtest | `test/unittest/unittest_hvx_dma_probe.cpp:464-648` `TEST_F(HvxDmaProbe, MoeChunkReplay)` | reads **no env var and no argument**; the matrix is compiled in (§3.3); `kMoeStages = 29` (`:515`) must equal the skel's `MOE_N_STAGES` — a mismatch prints `DMA_REPLAY_NOTE moe_layer_timed err=…` and the `pace=1` cells silently become `pace=0` |
| Probe gtests kept | same file `:320` `DmaProbeShapes` (16 `DMA_PROBE` lines), `:332` `TwoReaderDdr` (DSP side invalid, rule 12 / #90 — **not run**) | |
| IDL | `test/htp/nntr_hvx.idl:395` `dma_probe`, `:411` `moe_dma_trace_read`, `:439` `dma_replay`; `mm_u8i4_moe_layer_timed` `stage_us` = 29 slots; **no `moe_set_opts`** (placeholder comment `:400`) | skel and app from one commit (rule 3); `AEE_EBADPARM (0x8000040E)` on the first MoE call = mix |
| Device gtest module | `test/jni/Android.mk:954` `unittest_hvx_dma_probe`, `:930` `unittest_hvx_mm_u8i4` | rung 3 |
| Skel build | `test/htp/build.sh:94` (`HEX_EXTRA_CFLAGS` hook, unused here), output `test/htp/build/libnntr_hvx_skel.so` | one skel, A |
| Prompt | `origin/htp/77-first-handoff:docs/measurements/77-prompt512.txt`, md5 `fc65c1588dc66dd764c7013fe96cbb75` — **not on `htp_moe`** (the #77 branch was never merged; `git log origin/htp_moe..origin/htp/77-first-handoff` lists 7 commits incl. `64a50bcf` last-64 line and `e909d113` prompt) | copied verbatim onto the 94 branch |
| Models | `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40/nntr_lfm2_8b_a1b_q40_arm.bin` 4,768,855,808 B `d28f55c5…`; `q40-qs4cx-wh/…_arm.bin` 4,316,133,120 B `7b7867fa…`; both dirs carry `generation_config.json` (`do_sample: true`) and `nntr_config.json` (`num_to_generate: 512`, `init_seq_len: 512`, `bad_word_ids: []`; the WH one `moe_engine: htp`, `moe_htp_layers: ""`) | unchanged since #78 (quantizer format untouched by #93) |
| PR #86 (`origin/htp/80-m1-moe-gemv-dispatch`) | `htp_compute_ops.cpp` `sendMoeOptsOnce` reads `NNTR_MOE_HTP_M1_GEMV`, prints `[HTP] moe m1 gemv: on|off (applied=0x…)`, adds ` m1_gemv=%llu/%llu` to the M==1 row; IDL `moe_set_opts`; gtest `HmxMmU8I4Layer.MoeLayerM1GemvMatchesHmx` | C only if merged (§3.2) |

Consumers that would have to move with a changed contract (IDL/stub,
`HtpComputeOps`, quantizer format tag, loader check, profile stage tables,
`tools/htp_fc_report.py`): **none change** — this plan adds no contract.
The stub is regenerated by `build.sh` from the IDL that is already on
`htp_moe`.

**Working-tree warning (verified 2026-09-22).** `/home/j2z0-lee/nntrainer`
is checked out on `htp/80-m1-moe-gemv-dispatch` (another agent's rebase
of PR #86, HEAD moving), and its `test/htp/build/`, `builddir/`,
`jni/libs/`, `test/jni/obj/` hold binaries from that work (dated
2026-09-21 19:36, before `49cc60f2`). **Nothing there is the #94
artifact set.** Build in a dedicated worktree (§4 step 1); the repo
already uses one (`/home/j2z0-lee/nntrainer-4334`).

## 3. Design

### 3.1 Artifacts: one build, one skel, no profile binary

Everything rebuilt from **`origin/htp_moe` head at the moment the
implementer starts** — `49cc60f2` unless #86 has merged (then the merge
head; record the sha in the handoff header and the artifact table):

| staged as (`W=/local/mnt/workspace/htp_moe/94`) | from | rung |
|---|---|---|
| `$W/libnntr_hvx_skel.A.so` | `test/htp/build/libnntr_hvx_skel.so` | 2 |
| `$W/tps/nntrainer_causallm`, `libcausallm_core.so` | `Applications/CausalLM/jni/libs/arm64-v8a/` | 3 |
| `$W/tps/libnntrainer.so`, `libccapi-nntrainer.so` | `Applications/CausalLM/jni/obj/local/arm64-v8a/` (NDK r30 leaves them there, gates skill) | 3 |
| `$W/tps/libc++_shared.so`, `$W/gtest/libc++_shared.so` | NDK r30 sysroot (== #77's `b1586b9b…`; `/local/mnt/workspace/htp_moe/77/tps/libc++_shared.so` may be copied after `md5sum` agrees) | — |
| `$W/tps/libsdkl.so`, `$W/gtest/libsdkl.so` | `$HEXKL_ROOT/lib/6.4.0.1/armv8_android26/libsdkl.so` (== `0ad4e22a…`) | — |
| `$W/gtest/unittest_hvx_dma_probe`, `unittest_hvx_mm_u8i4` | `test/jni/obj/local/arm64-v8a/` | 3 |
| `$W/prompt512.txt` | `docs/measurements/77-prompt512.txt` on the 94 branch (`fc65c158…`) | — |
| models | unchanged, workstation md5s above; never rebuilt | — |

`$W/md5.txt` = `md5sum` of every staged file, pasted into the handoff's
artifact table; the device `md5sum` lines after the push are pasted under
Notes (rule 14: the pushed build is the provenance).

**No `--profile` build this time.** The per-type decode table (LEDGER §2
①) was measured in #77; it is a *share* table (MoE 42 vs 22 ms/token, FC
28 vs 10) and the ~7 % clock offset between units scales every row
alike, so repeating it on `R3CY10WM83Y` would re-measure the same shares
for ≈ 10 min of workstation and ≈ 6 min of device time plus the
`builddir` juggling of plan 77 step 5 — and nothing in the ARM path has
changed since. The open follow-up ⑰ (FC 28 ms/token with no FC on the
HTP) needs a *different* cell (a 4-thread run), not a repeat of the
profile table; it gets its own issue, see §3.3 "ride-along" for the
zero-cost hint this sitting can leave. Rejected alternative: bundling the
profile build "because the binary is being rebuilt anyway" — it doubles
the artifact set and re-introduces the wrong-directory risk of plan 77 §5
for no new number.

### 3.2 Variant C rule (decided by a grep, not by memory)

At the start of step 1, after `git fetch origin`:

```
gh pr view 86 --json state --jq .state                 # MERGED or OPEN
git grep -q 'NNTR_MOE_HTP_M1_GEMV' origin/htp_moe -- nntrainer/tensor/htp_backend/htp_compute_ops.cpp && echo C-ON || echo C-OFF
```

`C-ON` only if both agree. **Do not use** `git log origin/htp_moe --oneline
| grep -c 'M=1 MoE dispatch'` — it already returns 1 today because of the
plan commit `4d900989` (`[docs] Plan the M=1 MoE dispatch …`), with #86
unmerged. Today (2026-09-22) PR #86 is OPEN with a rebase in progress, so
the expected outcome is **C-OFF**: the handoff header says "Variant C
(`NNTR_MOE_HTP_M1_GEMV=1`) not in this sitting — PR #86 not merged at
build time; it is sitting 3 (LEDGER ⑯)", and the C tables are written
with the heading "sitting 3" and left empty. Never build from the #86
branch (contract §5, linear history; issue text).

If `C-ON`: the head is newer than `49cc60f2` and the IDL gained
`moe_set_opts` → same rebuild rule (skel + app together). C = the same
binary with `NNTR_MOE_HTP_M1_GEMV=1`, NPU model, G 64 / 512 × 2 runs, one
`NNTR_HTP_PROFILE=2` run at G=64, and `unittest_hvx_mm_u8i4
--gtest_filter='*MoeLayerM1GemvMatchesHmx*'` once. Proof fields: the
stderr line `[HTP] moe m1 gemv: on (applied=0x1)` in every C log, and the
M==1 row `blocks=0 m1_gemv=1408/1408` in the C profile run (A's profile
run reads `blocks=5632 m1_gemv=0/1408`). A C run whose text differs from A
fails the accuracy gate and is filed, not averaged (handoff skill).

### 3.3 The sitting (≤ 90 min; budget ≈ 50 without C, ≈ 56 with)

Order is fixed: **A before anything else** (rule 9), CPU before NPU (no
registration, and the CPU decode rate is the thermal gate), B after A, A
re-run after B, gtests, then C.

| block | min | content |
|---|---|---|
| 0 device state | 2 | `adb devices` → `R3CY10WM83Y device`; `adb shell dumpsys battery \| grep -E 'level\|temperature'`; `adb shell cat /sys/class/thermal/thermal_zone0/temp`; `adb shell df -h /data` (≥ 10 GB free if the models must be pushed). Note battery %, °C, warm/cool. Screen off, charger in |
| 1 install | 5–15 | **Models present?** `adb shell ls -l /data/local/tmp/nntrainer/causallm/models/{q40,q40-qs4cx-wh}/nntr_lfm2_8b_a1b_q40_arm.bin` — sizes must read exactly 4768855808 and 4316133120; if both exist, `adb shell md5sum` on both (≈ 1 min each) must give `d28f55c5…` / `7b7867fa…`, and the 9 GB push is skipped. #77 ran on the *other* unit from another workstation, so on `R3CY10WM83Y` expect them absent → `adb push $M/q40 …/models/q40` and `adb push $M/q40-qs4cx-wh …/models/q40-qs4cx-wh` (≈ 5–8 min). Then binaries: `$W/tps/*` → `/data/local/tmp/nntrainer/causallm/`, `$W/libnntr_hvx_skel.A.so` → `…/causallm/libnntr_hvx_skel.so` and → `/data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.so`, `$W/gtest/*` → `/data/local/tmp/htp_u8i4_layer_test/`, `$W/prompt512.txt` → `…/causallm/prompt512.txt`; `chmod 755` the three executables. **Never** `builddir/.../libcdsprpc.so` (rule 5). **Config edits are re-applied unconditionally** (plan 77 §3.2, #77 step 1 block verbatim): `do_sample` → false in both `generation_config.json`, `bad_word_ids: []` → `[124900]` in both `nntr_config.json`, then the `grep -H` echo of `do_sample`, `bad_word_ids`, `num_to_generate`, `init_seq_len`, `moe_engine`, `moe_htp_layers` per model — even if the models were already there, since the last sitting leaves `num_to_generate` at some G and this unit's state is unknown. Expected echo: both `do_sample": false`, `bad_word_ids": [124900]`, `init_seq_len": 512`; WH shows `moe_engine": "htp"` + `moe_htp_layers": ""`, `q40` neither. Provenance: `adb shell md5sum` of the skel (both dirs), `nntrainer_causallm`, `libnntrainer.so`, `unittest_hvx_dma_probe`, `prompt512.txt`, both `tokenizer.json` (`7b8067a5…`) |
| 2 A control | 25 | the #77 `runA` helper verbatim (device-side `sed` on `num_to_generate`, `grep` echo, `NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/<m> "$(cat prompt512.txt)"`, `tee $W/logs/A_<m>_G<G>_r<n>.log`). Order: `q40` 64 ×2, 512 ×2, 1024 ×2; `q40-qs4cx-wh` 64 ×2, 512 ×2, `sleep 60` before each 1024. Expected lines exactly `prefill: 512 tokens`, `generation: <G> tokens`, `total:`, `peak memory:`; `prefill: N≠512` voids the prefill column (tokenizer md5s tell), `generation: N≠G` is a failed step (edit did not take), a `[HTP-PROFILE]` block in an A log means the env var leaked → void the run. **Thermal gate:** CPU decode < 30 tok/s ⇒ throttling → 5 min screen-off cool-down, redo that cell, note it. `diff` loop for run1 = run2 per cell (the #77 one, `sed -n '/^=====/q;p'`) |
| 3 B trace | 6 | NPU model, G=64 (set with the same `sed`), same binary: (i) `NNTR_HTP_PROFILE=2 NNTR_M0_PROFILE=1 …` → `$W/logs/B_level2.log`; (ii) `NNTR_HTP_PROFILE=3 NNTR_M0_PROFILE=1 …` → `$W/logs/B_level3.log` (level 3 runs ≈ 5× longer on the DSP side; ≈ 2 min). `NNTR_HTP_DMA_TRACE` left at its default 3. Capture **whole** logs (`2>&1 \| tee`); the pasted extract per level: the `[HTP-PROFILE] level=… qos_mode=…` header, the `M>1` and `M==1` rows, both `weight DMA:` and both `DMA ring:` lines, 3 `[M0-PROF]` lines, and every `[HTP-DMA] call=<k> M=1 …` block (3 per level, ≈ 60 lines each) — the M>1 dumps only by their header line. `qos_mode` must be 2 at both levels (#77 B) |
| 4 A re-run | 2 | `runA q40-qs4cx-wh 64 3` with `NNTR_HTP_PROFILE` unset: prefill and decode within the sitting's noise of block 2's G=64 cells → the trace binary and the trace runs left no residue (plan 87 §1 standing gate); text = run1 |
| 5 gtests | 3 | `adb shell "cd /data/local/tmp/htp_u8i4_layer_test && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*:*DmaProbeShapes*'"` → `$W/logs/G_replay.log`. Expected: one `DMA_REPLAY_TRACE dsp_us=… desc=46 waits=… plan_shape_ok=y` line plus 46 `push` and the `wait` lines, then **11 `DMA_REPLAY` lines** (the compiled-in matrix below), `bytes_per_call=22020096`, `checksum_ok=y` on every line, then 16 `DMA_PROBE` lines (`vote=1`, `checksum_ok=y`), `[  PASSED  ] 2 tests`. A `DMA_REPLAY_NOTE moe_layer_timed err=0x8000040e` line means a skel/gtest stage-count mismatch (stale artifact — stop, re-push, redo). `TwoReaderDdr` is excluded on purpose (rule 12, #90) |
| 6 C (only C-ON) | 6 | `NNTR_MOE_HTP_M1_GEMV=1 runA q40-qs4cx-wh 64 1/2`, `… 512 1/2` (logs `C_…`); then `NNTR_MOE_HTP_M1_GEMV=1 NNTR_HTP_PROFILE=2 …` at G=64 → `C_level2.log`; then the `*MoeLayerM1GemvMatchesHmx*` gtest once (`unittest_hvx_mm_u8i4`). Each C log must contain `[HTP] moe m1 gemv: on (applied=0x1)`; `diff` C text vs A text per G |
| 7 finish | 3 | fill the tables, texts to the appendix, `git commit -s`, push, `state:measured` |

Replay matrix as landed (`unittest_hvx_dma_probe.cpp:633-641`; 20 calls
per cell; not the full workers × load × fresh/gap cross product the issue
title abbreviates — plan 87 §3.3's 11 cells):

| # | workers | load | pace | fresh | gap_us | decides |
|---|---|---|---|---|---|---|
| 1–3 | 1 / 2 / 4 | 0 | 0 | 0 | 0 | c (engine rate on the in-situ list, 1 chain vs N `dmstart`) |
| 4–6 | 1 / 2 / 4 | 0 | 1 | 0 | 0 | a (blocked-wait sum on the in-situ timeline, comparable to `DMA ring:` `wait=`) |
| 7 | 1 | 1 (DDR stream, 3 workers) | 1 | 0 | 0 | d |
| 8 | 1 | 2 (VTCM stream, HVX busy) | 1 | 0 | 0 | d (separates DDR contention from HVX activity) |
| 9 | 1 | 0 | 1 | 1 | 0 | g (fresh pages) |
| 10 | 1 | 0 | 1 | 0 | 600 | g (FastRPC-sized gap) |
| 11 | 1 | 0 | 1 | 1 | 600 | g (both) |

Optional ride-along, **only if the clock shows ≥ 20 min left after block
6 and clearly labelled "not a variant of record"**: one NPU G=64 run with
`NNTR_NUM_THREADS=4` (same binary, profile unset). If decode tok/s does not
drop (or rises) against the 8-thread A cell, LEDGER ⑰'s over-splitting
candidate gains a data point for its own issue at zero build cost. It is
not part of any gate here and can be skipped without note.

### 3.4 Result tables the handoff carries (empty, reference column in)

**A — 12 cells + the re-run.** Columns: variant, model, G, run, prefill
ms / tok/s, decode tok/s (all), decode tok/s (last 64) = `n/a (#89)`, gen
tokens printed, peak RSS KB, text run1 = run2, skel md5 (device), and a
**reference column with #77's `R3CY205ZMND` value** so the supervisor
reads the ratio per cell:

| model | G | run | #77 prefill tok/s | #77 decode tok/s |
|---|---|---|---|---|
| q40 | 64 | 1 / 2 | 336.4 / 253.1 | 54.10 / 52.94 |
| q40 | 512 | 1 / 2 | 297.5 / 231.5 | 52.67 / 51.97 |
| q40 | 1024 | 1 / 2 | 287.3 / 288.0 | 46.44 / 48.53 |
| q40-qs4cx-wh | 64 | 1 / 2 | 403.5 / 475.4 | 19.68 / 21.27 |
| q40-qs4cx-wh | 512 | 1 / 2 | 485.8 / 494.7 | 18.38 / 18.24 |
| q40-qs4cx-wh | 1024 | 1 / 2 | 541.2 / 455.1 | 19.04 / 16.96 |

Goal line: ≥ 50 decode at every G (NPU); PR-era provisional now 20.8 / 48
/ 523. These 12 cells **replace** BENCHMARK's "now" (issue (b)).

**B — `weight DMA:` and `DMA ring:` fields per level (M==1 row).** One
row per level: `qos_mode`, `calls`, `host` / `dsp` / `transport` µs/call,
`gather`, `drain`+`dn`, `mm`, `blocks`; `weight DMA:` KB/call, first KB,
first us, first GB/s, averaged GB/s; `DMA ring:` desc, waits, blocked,
wait us, act / gu / dn us, busy lo..hi us, engine GB/s lo..hi, depth max,
first expert ready at, last issue at, of dsp_us. Reference (#77,
`R3CY205ZMND`): level 2 host 1948.6 / dsp 1360.9 / transport 587.7, gather
143.5, drain 108.7+7.3, mm 748.0, 21504 KB/call, avg 16.2 GB/s; level 3
1750.4 / 1222.7 / 527.7, gather 122.6, drain 0.9+0.4, mm 746.6, avg 18.0.
The `DMA ring:` fields have no reference — this sitting is their first.
Plus the M>1 row per level (existing columns only) for the standing gate.

**Attribution table (plan 87 §1 / §3.2), pre-readable.** Copied so the
user knows what each row is decided by before the sitting; the last two
columns are filled from the device:

| # | hypothesis | signature that confirms (plan 87 §3.2) | lines of this sitting that fill it | % of `dsp_us` (L3 warm / L2 cold) | % of the isolated-vs-in-situ gap |
|---|---|---|---|---|---|
| a | ring serialised behind compute: waits dominate, depth ≤ 1 | `wait ≥ 25 %` of `dsp` **and** `depth max ≤ 2`; refuted if depth reaches 8–11 while waits stay small | `DMA ring:` `wait=` vs the row's `dsp=`, `depth max=`; replay cells 4–6 `wait_us` / `blocked` | | |
| b | chunk geometry is a slow probe shape | already refuted on the host (§0 i: gate_up = 8 KiB × 64 @ 56 KiB = probe `iii`, down 16 KiB × 56 @ 32 KiB); device lines merely confirm | `[HTP-DMA] push … row=8192 nrows=64 stride=57344` (gate/up), `row=16384 nrows=56 stride=32768` (down), `desc=46` | | |
| c | one dmlinked chain / one `dmstart` limits concurrency | `gbs` rises ≥ 15 % with workers on the same schedule at `pace=0` | replay cells 1–3 `gbs` | | |
| d | DDR contention with HVX epilogues / pack | paced replay's blocked-wait total moves ≥ 20 % with `load=1`; `load=2` moving it instead points at HVX activity, not DDR | replay cells 7, 8 vs 4 (`wait_us`, `blocked`); in situ M>1 vs M==1 `engine` GB/s | | |
| e | DVFS / bus ramp inside a call | per-descriptor rates climb monotonically through the call (expert 0 ≤ 60 % of expert 3) on cold calls, flat on warm | `[HTP-DMA] push … e=0…3 … done=lo..hi` intervals per expert, L2 (`rep=1/1`) vs L3 (`rep=5/5`); `busy=` L2 vs L3 | | |
| f | cold start: expert 0's 3.5 MiB has nothing to hide behind | the act wait of expert 0 ≈ `gather` (≈ 120 µs) and every later wait ≈ 0 on warm calls | `DMA ring:` `[act <us> …]`, `first expert ready at`, the row's `gather`; `[HTP-DMA] wait k=0 site=act … blocked=y` | | |
| g | warm/cold per call: single-shot (L2, `drain 108`) blocks where 5× repeats (L3, `drain 1.3`) do not — IOTLB / page state / clock after the ≈ 0.6 ms FastRPC gap | `fresh=1` alone reproduces the L2 wait profile → translation/page state; `gap_us` alone → clock ramp | `DMA ring:` L2 vs L3; replay cells 9, 10, 11 vs 4 | | |

Below it one sentence: "The first wall-2 fix issue is: …" (largest row).
Plan 87 §0 expects f + g large and a / b / c small; the table is filled
from the device, not from that paragraph.

**Replay table.** One row per `DMA_REPLAY` line (11) with `us_per_call`,
`gbs`, `wait_us`, `blocked=<n>/<n>`, `depth_max`, `busy_us` lo..hi,
`regions`, `workers_used`, `load_units`, `checksum_ok`; the
`DMA_REPLAY_TRACE dsp_us=… desc=… waits=… blocked=… wait_us=…
wait_act_us=… busy_us=… depth_max=… first_ready_us=… last_issue_us=…`
line above it; the 16 `DMA_PROBE` lines with #77's `vote=1` GB/s
(contiguous 4 KiB × 256: 79.6 → 72.0 for workers 1 → 4; 1 MiB × 1: 78.4 →
72.4; 2D 16 KiB × 64 @ 56 KiB: 116.5 → 109.7; 2D 8 KiB × 64 @ 56 KiB:
111.0 → …) as the reference column — the isolated rate of *this* unit
sits in the same table as its in-situ rate.

**C — only if C-ON (else headed "sitting 3", empty).** C vs A per G (64,
512): prefill tok/s, decode tok/s, text = A (y/n, first differing token),
`m1_gemv=` proof field from `C_level2.log` (`blocks=0 m1_gemv=1408/1408`)
and the `dsp=` µs/call vs A's level-2 `dsp=` (plan 80: `22 MB / mm` is the
effective arena read rate the GEMV sees).

**Appendix.** Full texts per (model, G) (run1 = run2 → one block); the
six `[HTP-DMA] M=1` blocks; device md5 lines; `logs/` file list.

### 3.5 Thermal, drift and the rest of the protocol

* Battery %, battery °C and `thermal_zone0` before block 2, after block 2,
  after block 6; warm/cool note; USB powered, screen off.
* CPU decode < 30 tok/s ⇒ throttling (doc 49 §6) → cool down 5 min, redo
  the cell, record both.
* 60 s pause before each NPU G=1024 run (#77); no other pauses.
* A first; every B/C number is read against this sitting's A only. The
  #77 column is for the unit ratio (rule 13), never for a verdict.
* Local `adb` — device-side `sed` and `"$(cat prompt512.txt)"` quoting
  work (the #77 deviation 6 was the SSH bridge). If a `grep` echo shows an
  edit did not take, edit the two json files on the workstation and
  `adb push` them (the #77 fallback), then re-echo.

## 4. Steps (implementer; one run to `state:needs-measurement`)

`source tools/htp/env.sh` first in every shell. All rungs from
`.claude/skills/hexagon-gates`.

1. **Worktree and branch.** `cd /home/j2z0-lee/nntrainer && git fetch
   origin && git worktree add /home/j2z0-lee/nntrainer-94 -b
   htp/94-sitting2-anchor-trace origin/htp_moe`. Record `git rev-parse
   HEAD` (expect `49cc60f2` unless #86 merged). Run the §3.2 C-ON/C-OFF
   check and write the result down. Copy the gitignored inputs from the
   main tree: `Applications/CausalLM/lib/libtokenizers_android_c.a`
   (173 MB, built once by `build_tokenizer_android.sh`), and `ln -sfn
   $PWD/subprojects/googletest/googletest test/jni/googletest` (tracked
   subprojects come with the worktree). Do **not** check out anything in
   `/home/j2z0-lee/nntrainer` (another agent works there). Gate: `git
   status` clean in the worktree; `git log -1` = the fetched head.
2. **Rung 1 (host, seconds).** `bash test/htp/host/run_host_checks.sh`
   → `MOE KERNEL MATCHES REFERENCE`, `PROFILE ON/OFF BYTE-IDENTICAL`,
   `PROFILE OFF: TRACE UNTOUCHED`, `IN-SITU CHUNK PLAN MATCHES KERNEL (46
   descriptors)`, `DMA TRACE ARITHMETIC OK`, `DMA PROBE PLAN OK`, `ALL
   CHECKS PASS`, `WORKER POOL LANES OK`; `bash tools/htp_syntax_check.sh`
   exits 0. The meson host build and gtests are not repeated (no source
   change; they passed at the #93 merge) — if the implementer wants them,
   `meson setup build …` with the gates skill's flags in the worktree.
3. **Rung 2 (skel).** `./test/htp/build.sh` → `-Wall -Werror` clean,
   `test/htp/build/libnntr_hvx_skel.so`; `mkdir -p $W/{tps,gtest,logs}`;
   `cp test/htp/build/libnntr_hvx_skel.so $W/libnntr_hvx_skel.A.so`.
   Gate: file exists, md5 recorded. (`build.sh` regenerates the stub from
   the IDL, so rung 3 links against the same contract.)
4. **Rung 3 (app + device gtests, ≈ 10 min).** Fresh `builddir` in the
   worktree → the gates skill's workstation specifics: run
   `build_android.sh --htp` once; if `ninja install` fails on the
   googletest `prefix`, `cd builddir && meson configure
   -Dprefix=$PWD/android_build_result && ninja install`, then
   `build_android.sh --htp --cache`. `readelf -d
   builddir/android_build_result/lib/arm64-v8a/libnntrainer.so | grep -E
   'libsdkl|libcdsprpc'` → both NEEDED. `strings
   Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm | grep -c
   'per-layer-type totals'` → 0 (not a profile binary). Then
   `(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=.
   NDK_APPLICATION_MK=./Application.mk APP_BUILD_SCRIPT=./Android.mk
   NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT
   unittest_hvx_dma_probe unittest_hvx_mm_u8i4 -j8)`. Stage per §3.1;
   `cp $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/libsdkl.so` and the NDK
   `libc++_shared.so` into `tps/` and `gtest/`; `md5sum $W/tps/*
   $W/gtest/* $W/libnntr_hvx_skel.A.so $W/prompt512.txt > $W/md5.txt`.
   Gate: all files present, `md5.txt` written, no `libcdsprpc.so` in `$W`.
5. **Prompt file.** `git show
   origin/htp/77-first-handoff:docs/measurements/77-prompt512.txt >
   docs/measurements/77-prompt512.txt`; `md5sum` = `fc65c158…`; `grep -c
   '["`$\\]' docs/measurements/77-prompt512.txt` = 0; `cp` to
   `$W/prompt512.txt`. (The 512-id check snippet from #77 goes into the
   handoff for the user; the count on the device is proven by
   `prefill: 512 tokens`.)
6. **Handoff** `docs/measurements/94-sitting2-anchor-trace.md` from the
   `hexagon-handoff` template: header with sha, C-ON/C-OFF statement,
   estimated minutes (≈ 50 / 56); Why (two sentences: anchor + wall-2
   attribution); Artifacts (the §3.1 table filled from `md5.txt`, models
   with their fixed md5s); Steps = §3.3 blocks 0–7 as copy-pasteable
   shell (the #77 install block, config-edit block, `runA`, B commands,
   gtest command, `diff` loop, provenance `md5sum` line — adapted to `$W`
   and the two device dirs); Results = the §3.4 tables with the
   reference columns filled and the device columns empty; Notes;
   Appendix headings. Gate: every command in the file runs in `bash -n`
   (paste into a scratch script), the estimated total ≤ 90, every
   artifact row has an md5 and the sha.
7. **Commit, push, label.** One docs commit on the branch: `[docs] Handoff
   94: sitting 2 anchor cells and the #87 trace/replay on R3CY10WM83Y`
   (body ≥ 8 words; names the prompt file's provenance `e909d113` and the
   build sha), `git commit -s` with the `Co-Authored-By` trailer, `git
   push -u origin htp/94-sitting2-anchor-trace`. Set #94 to
   `state:needs-measurement` (keep `hexagon`, `prio:p0`), comment the
   handoff path, the staging dir and the C-ON/C-OFF result. No PR yet (the
   PR follows the filled handoff, as #77 did). Leave the worktree in place
   for the user's sitting (`git checkout htp/94-sitting2-anchor-trace` in
   the handoff's step 1 refers to `/home/j2z0-lee/nntrainer-94`).

The device measurement is the handoff itself (blocks 0–7); no other step
needs the phone. Variants: A (control, 12 cells), B (env only, same
binary), the gtest ride-along, C (env only, conditional) — three
model-path variants at most, one binary set.

## 5. Risks

| risk | how the handoff makes it visible |
|---|---|
| Wrong tree built (the main checkout is on the #86 rebase; its `test/htp/build/` and `jni/libs/` hold other binaries) | step 1 worktree; the handoff header's sha and `git log -1` of the worktree; the M==1 row on `49cc60f2` has **no** `m1_gemv=` field — if it appears in an A/B log while C-OFF, the app was built from the wrong tree → void |
| Stale skel / stub (rule 3): #93's IDL grew (`stage_us` 29 slots) | one commit for skel + app + gtests; device `md5sum` lines before block 2; `AEE_EBADPARM (0x8000040E)` on the first MoE call or `DMA_REPLAY_NOTE … err=0x8000040e` = mix → stop, re-push |
| md5 drift by build path (rule 14) | the table is filled from `$W/md5.txt` of the build that is pushed and the device lines are pasted; a mismatch is a note while C-OFF and a void once C ran |
| Thermal / DVFS drift within the sitting (rule 9) | A first; CPU < 30 tok/s gate; battery/°C at three points; block 4 re-runs an A cell after the trace runs; level 3 (warm) beside level 2 (cold) is itself hypothesis (g)'s discriminator, so drift is data there, not noise |
| Unit ratio (rule 13) | every table carries the `R3CY205ZMND` column; the supervisor records the ratio per cell and replaces "now" only from the `R3CY10WM83Y` column |
| Sampling left on / wrong G | unconditional config edits with `grep -H` echo; `generation: N≠G` = failed step; different run1/run2 text = failed cell |
| Prompt not 512 tokens on the device | `prefill: 512 tokens` is the check; tokenizer md5s (`7b8067a5…`) on both model dirs pasted |
| `[HTP-DMA]` dumps are the first decode token's layers 0–2 only (default 3 per bucket) and prefill dumps may print `busy=n/a (trace truncated past 512 pushes)` | stated in the handoff; the M>1 dumps are pasted by header only; if the user wants later layers, `NNTR_HTP_DMA_TRACE=6` is an env-only re-run of block 3 (not planned, not needed for a–g) |
| Completion times are brackets (`done=lo..hi`, `busy=lo..hi`) | the attribution quotes ranges; the replay `pace=0` cells give the point value for the same list |
| `MoeChunkReplay` pace cells fall back silently to `pace=0` if the step-0 timed call fails | the `DMA_REPLAY_NOTE` line and `plan_shape_ok=` are required pastes; `pace=1` rows with a note are marked "fell back" |
| Address space / budget | nothing new is allocated on the DSP; the trace tables are static `.bss` (#93) |
| Models absent on this unit (9 GB push) and `/data` space | block 1's `ls -l` / `df -h` decide; the push is budgeted (≈ 5–8 min) inside the 90 |
| PR #86 merges *during* the implementer's run | the C-ON/C-OFF check is taken once at step 1 and written into the header; a later merge does not change this sitting (LEDGER ⑯: then sitting 3) |

## 6. Docs to update (supervisor, from the filled handoff)

* **BENCHMARK.md**: Goals "now" cells (NPU decode 64 / 512 / 1024, CPU
  decode, NPU prefill at prompt 512) replaced by the `R3CY10WM83Y` A cells,
  status "provisional" → "measured on R3CY10WM83Y (#94)"; the unit ratio
  per cell next to the #77 rows; Results: 12 A rows (+ the block-4
  re-run) and, if C ran, 4 C rows; a side table "#87 DMA ring (M==1, level
  2 / 3)" with the `weight DMA:` / `DMA ring:` fields, the 11
  `DMA_REPLAY` cells and the 16 `DMA_PROBE` cells of this unit;
  Artifacts: the #94 set (skel A, `tps/*`, both gtests, `77-prompt512.txt`)
  with table and device md5s and the build sha; Method: note that the
  `generation(last 64)` column stays `n/a` until #89.
* **LEDGER.md**: §2 — verdict ③/⑥ rewritten from the attribution table
  (which letter, what share, the named fix issue); "Control on
  `R3CY205ZMND`" row gets its `R3CY10WM83Y` twin and rule 13's measured
  ratio; if C ran, ⑯'s wall-1 verdict. §1 — rule 11 refined with the
  engine rate *while running* vs the call-averaged number; a new rule if
  (g) confirms (cold vs warm calls; `NNTR_HTP_PROFILE=3` hides a per-call
  cost the model pays every token), and any silicon surprise. §3 — ⑥ step
  2 filed as the fix issue; ⑮ closed; ⑯ either closed (C ran) or "sitting
  3"; ⑰ unchanged unless the optional ride-along ran.
* **Contract §1** "now" table: the provisional PR numbers replaced by the
  A cells; the §1 status paragraph updated.
* **Contract §2 wall 2 sentence** ("Cause unknown … measurement C
  decides") → the attribution's one line.
* `docs/measurements/94-sitting2-anchor-trace.md`: written in step 6,
  filled by the user; `docs/measurements/77-prompt512.txt` lands on
  `htp_moe` with it.
* `.claude/skills/hexagon-gates/SKILL.md`: no change required by this
  plan (the worktree pattern is a recommendation, and #77's `81fd3ec9`
  gates note is still only on that branch — the supervisor may lift it
  when the handoff PR is made).
