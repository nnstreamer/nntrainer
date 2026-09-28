# 101 — Make the M=1 HVX GEMV MoE path the default (and make its profile row add up, #102)

Issue: dlwlzzero/nntrainer#101 (p1; LEDGER ⑯ follow-up, tracker #76) with
#102 (p2, LEDGER ㉑) folded in — §3.3 says why. Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. Branch
`htp/101-m1-gemv-default` from `htp_moe` @ `e9c2c907`; every `path:line`
below was read against that tree. Ground truth for the effect: #94 sitting 2
variant C (`origin/htp/94-sitting2-anchor-trace:docs/measurements/94-sitting2-anchor-trace.md`
@ `dad0f476`, §C): decode +6.2 / +7.7 / +1.7 % at G = 64 / 512 / 1024,
text byte-identical to A at every cell, `MoeLayerM1GemvMatchesHmx`
bit-identical, `blocks=0 m1_gemv=1408/1408`, dsp 1414 → 1044 µs/call,
`mm` 783 → 975, ring `wait` 269 → 4.3, transport 649 → 748.

This issue ends in a PR with host gates and an app build (rungs 0, 1, 3);
its device check is a ride-along in the next sitting, not a handoff of its
own (§4 step 6).

## 1. Goal and gate

From the issue, made measurable:

| criterion | what counts as done |
|---|---|
| Default resolves to **on** with nothing set, **off** with `NNTR_MOE_HTP_M1_GEMV=0`, on with `=1` | new host check line `MOE M1 GEMV OPTS: unset=on 0=off 1=on` from `run_host_checks.sh` (§2.3); no config key (§3.1) |
| Host gates unchanged | `ninja -C build`; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6 (the tiny fixture is CPU-only); `run_host_checks.sh` still prints `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4; tiny+real)`, `ALL CHECKS PASS`, `WORKER POOL LANES OK`; `tools/htp_syntax_check.sh` exits 0 |
| #102: the GEMV row adds up | new host check line `MOE PROFILE ROW: gemv rest>=0 hmx unchanged` (§2.3); on device (ride-along) the level-2 M==1 row shows `swiglu` as lane-time (≈ 5.7 × `mm`), `rest` in [0, 5 %] of host, and a `weight DMA: n/a (direct arena read inside mm)` line so the row block keeps its line count |
| Skel + app build | rung 3 md5s in the PR body (`nntrainer_causallm`, `libnntrainer.so`, `libcausallm_core.so`, `unittest_hvx_mm_u8i4`); **no DSP source changes → no new skel** (§3.3); the PR names the skel md5 it was built against (`htp_moe` head's `test/htp/build.sh` output) |
| Device, ride-along (next sitting's variant A) | A's stderr prints `[HTP] moe m1 gemv: on (applied=0x1) source=default` once per run; its level-2 M==1 row reads `blocks=0 m1_gemv=1408/1408`; the one opt-out cell (`NNTR_MOE_HTP_M1_GEMV=0`, G = 64) prints `off (applied=0x0) source=env`, `blocks=5632 m1_gemv=0/1408`, text identical to A. Decode tok/s is read against that sitting's numbers; the +6–8 % is #94's verdict and is not re-proven |
| Prefill gate | M>1 row `m1_gemv=0/<calls>` in both A and the opt-out cell; prompt-512 prefill of A within −5 % of the opt-out cell (same sitting). The switch never applies at M > 4 (`hexkl_mm_u8i4_moe.c:807-809`), so this is a check, not a risk |

Standing gates: text identical to variant A (for `QS4CX_WH` clause (c) is A = B text, LEDGER rule 6; the CPU `q40` text differs by construction); `NNTR_NUM_THREADS=8`; tok/s never from a `--profile` build.

## 2. Where it lives

### 2.1 The flip — `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`

* `sendMoeOptsOnce` `:1062-1094`. Today: `flags = (env && atoi(env) != 0) ? HTP_MOE_FLAG_M1_GEMV : 0` `:1064-1066`; a stale skel with `flags == 0` only logs `:1068-1077`; anything else that fails or echoes a different bit throws with the rebuild hint `:1078-1089`; the proof line `:1090-1091`. Changes: resolve `flags` through `htp_moe_opts_flags(std::getenv(...))` (§2.3); print `[HTP] moe m1 gemv: %s (applied=0x%x) source=%s` with `source` = `default` (env unset) or `env` — the old prefix `moe m1 gemv: on (applied=0x1)` stays byte-for-byte so sitting 2's greps (`grep -c 'moe m1 gemv: on (applied=0x1)'`) keep working. The "skel predates it, only log" branch now fires only when the user asked for **off**; with the default on, an old skel **throws** at the first MoE call (intended: rule 17 / doc 46 §48.7 — a silent HMX run reported as the GEMV is the failure mode plan 80 §6 refused).
* The DSP side is untouched: `nntr_hvx_moe_set_opts` (`test/htp/nntr_hvx_mm_u8i4.c:1057-1066`) stores the bit; the per-call decision `use_m1` (`hexkl_mm_u8i4_moe.c:807-810`: flag && `M ≤ 4` && widest expert ≤ 4 && `1 ≤ n_with_rows ≤ 16`) is unchanged, so prefill (M = 512) never takes the path.
* `HTP_MOE_FLAG_M1_GEMV` `:190` moves into the new header (§2.3) so the resolver and the caller share one definition.

### 2.2 The print (#102) — same file, `HtpProfile::dump()` `:508-660`

* M==1 / M>1 row `:545-605`: `mm_per` (the printed `rest`) `:582-586` is `dsp − (quant + swiglu + dequant + acc + drain + scatter + stage + gather + requant + mm + drain_dn + push + alloc)`. On the GEMV path `swiglu` is **not a stage**: `moe_tail_probe_add` (`hexkl_mm_u8i4_moe.c:326-332`) atomically sums each lane's wall time inside stages A and C (`moe_m1_pair_worker` `:478-494`, `moe_m1_down_worker` `:522-536`) into `HEXKL_PROBE_SWIGLU`, by design (plan 80 §3.6: "SWIGLU / lanes ≈ MM says the lanes were balanced"). Every `T0`/`ADD` pair on the path is paired (`:1099-1124`); nothing is cumulative. C's `5601.2 / 974.7 = 5.75` is six lanes (the pool is sized `n_hvx − 1` workers plus the caller, `test/htp/hvx_add_f32.c:112-118`; the 8 Elite cDSP has 6 HVX contexts) busy 96 % of the two GEMV stages' wall — the row's `mm` is lane-bound, which is exactly the ㉒ reading (§3.4). The print subtracts a 6-lane sum from a 1-lane clock and goes negative.
* `weight DMA:` block `:606-625` is gated on `b.dma_first_us != 0`; the GEMV path never waits on the ring (`DMA_FIRST` stays 0, `DMA_KB` = 0), so the line is skipped and the level-2 block loses one line.
* Changes, print only: `rest` through `htp_moe_row_rest_us(...)` (§2.3), which leaves `swiglu` out of the subtraction when `b.m1_calls != 0`; the format string keeps every field name and position (the sitting-2 extraction loop greps `blocks=`/`m1_gemv=` and pastes the row verbatim). After the row, when `b.m1_calls == b.calls && b.dma_first_us == 0`, print `\n[HTP-PROFILE]     weight DMA: n/a (direct arena read inside mm, no ring; swiglu = lane-time, %.2f lanes busy over mm)` with `swiglu_per / mm_meas_per`. The effective GB/s stays hand arithmetic (`22 MB / mm`, plan 80 §4.7): the host bucket knows `K`, `N_out` and `M` but not `inter` (`addInvokeMoeLayer` `:303`, key `(K, N_out, M==1)`), and giving it the bytes means either a kernel `DMA_KB` count on the GEMV path (a skel change this issue excludes) or a signature change to `addInvokeMoeLayer`, which is a #88 hunk (§2.4). `ponytail:` comment at the line naming both upgrade paths; ㉒'s plan takes one.

### 2.3 New: `nntrainer/tensor/htp_backend/htp_moe_opts.h` + one host check

Header-only, C99 (so the host check compiles it with `gcc -std=c99` like its siblings), `@file`/`@brief` header:

* `#define HTP_MOE_FLAG_M1_GEMV 1u` (moved from `htp_compute_ops.cpp:190`; the DSP's `HEXKL_MOE_FLAG_M1_GEMV` in `hmx/hexkl_mm_u8i4_moe.h` stays the DSP-side twin, as today).
* `static inline uint32_t htp_moe_opts_flags(const char *env)` — `NULL` → `HTP_MOE_FLAG_M1_GEMV`; otherwise `atoi(env) != 0 ? flag : 0`.
* `static inline double htp_moe_row_rest_us(double dsp, double named_sum_wo_swiglu, double swiglu, int gemv_row)` — `dsp − named_sum − (gemv_row ? 0 : swiglu)`.
* `test/htp/host/moe_opts_host_check.c`, added to `run_host_checks.sh` after the DMA trace check (`:66-76` pattern): asserts the three resolutions and two rows — the GEMV pattern from C (`dsp 1044.0`, `swiglu 5601.2`, `mm 974.7`, the other named stages summing to ≈ 60) gives `rest ≥ 0` and `≤ 0.05 × host (1792.2)`; the HMX pattern from A (`swiglu 0.0`, `dsp 1414.0`, named sum 1391.3) gives the unchanged `22.7 ± 0.1`. Pass lines: `MOE M1 GEMV OPTS: unset=on 0=off 1=on` and `MOE PROFILE ROW: gemv rest>=0 hmx unchanged`. `ALL CHECKS PASS` is printed by the first check today; the new binary exits non-zero on failure, which `set -eu` turns into a failed gate.

### 2.4 Not touched, and the #88 boundary

* `test/htp/nntr_hvx.idl` (`moe_set_opts` `:449` exists), `generate_stub.sh`, `test/htp/build.sh`, `test/htp/nntr_hvx_mm_u8i4.c`, `hmx/hexkl_mm_u8i4_moe.c`, `hmx/hexkl_probe.h`, `hvx/hvx_gemm_u8i4_wh.c`: no DSP or IDL change, so the skel and the 30-slot stage table (`HTP_MOE_N_STAGES` `:184` = `MOE_N_STAGES`) are the ones on the device today.
* Quantizer format tag, loader check (`float_tensor.cpp`), `tools/htp_fc_report.py` (parses no MoE row), `Applications/CausalLM/models/lfm2_moe/*` (no config key, §3.1), `[M0-PROF]`.
* Device gtest `TEST_F(HmxMmU8I4Layer, MoeLayerM1GemvMatchesHmx)` (`test/unittest/unittest_hvx_mm_u8i4.cpp:2280-2400`) calls `moe_set_opts(0)` / `(1)` explicitly and restores 0 at the end `:2392`; it does not read the env, so the flip does not change it. It is rebuilt in rung 3 only because the app build rebuilds everything.
* **#88** (`origin/htp/88-moe-call-marshalling` @ `d604177c`, 4 commits) touches `htp_compute_ops.cpp` at hunks `:301-312` (`addInvokeMoeLayer` gains byte arguments), `:488-497`, `:659-678` (the `staging:` line after the `DMA ring:` block), `:1428-1860` (`StagingPool`, six `invoke*` call sites), `:2546`, plus `htp_backend.cpp` and `htp_rpcmem.h`. #101's hunks are `:190` (one define), `:545-625` (row + `weight DMA:`), `:1062-1094` (`sendMoeOptsOnce`), the include block `:45-52`, and two new files. Nearest approach is `:625` vs `:659` — no overlapping context, so the two branches merge in either order without a conflict. The issue wants #101 first so #88's A is the GEMV path; if #88 merges first, this branch rebases with no edits. The implementer does not touch `addInvokeMoeLayer` or `invokeMoeLayer` for this reason.

## 3. Design

### 3.1 Chosen: flip the env default; the env var becomes the opt-out; no `nntr_config.json` key

One-line semantics change in the resolver: unset = on. The A/B control the contract needs (§4.2: every verdict is a same-sitting A/B) is `NNTR_MOE_HTP_M1_GEMV=0` on the same binary, set per `adb shell` command as sitting 2 already did for `=1`. The proof line carries `source=default|env` so a log says which of the two produced the path.

**Rejected: the issue's preferred `moe_m1_gemv: true|false` key in `nntr_config.json` with the env var as override.** The backend does not see `nntr_config.json` (plan 80 §3.5): `moe_engine` is read in `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:51` by the app, and a key would need a new virtual on `compute_ops.h` (outside the supervision scope, contract §6) or a `setenv` from the app (not cross-platform, AGENTS.md) — plumbing for a knob whose only "false" user is an A/B sitting, which sets the env per run anyway. If a file-based off switch is ever wanted for a deployment, it is a one-line follow-up in the app that sets the env when the key says so; not here. This deviates from the issue's "prefer" and is stated in the PR body.

### 3.2 Contract and doc 45 §3 checks

No kernel, IDL, layout, ring, VTCM, or heap change: walls untouched, arena budget untouched, no CPU fallback added for `QS4CX_WH`. No quantizer input changes, so no `_det` question. Bit-identity of the GEMV to the HMX path is already gated by the host check (`M1 GEMV PATH BIT-IDENTICAL…`, `test/htp/host/moe_layer_host_check.c:535-708`) and the device gtest, and the text gate was met at every sitting-2 C cell.

### 3.3 Fold #102 in: yes

Three reasons. (1) #101's own device gate reads the level-2 M==1 row of the next sitting's A; under the GEMV default that row prints `rest<=-5588` and drops a line until #102 lands, so #102 is a precondition of #101's acceptance, not a neighbour. (2) Both diffs live in the same ≈ 90 lines of `htp_compute_ops.cpp` (`dump()` and `sendMoeOptsOnce`), and the shared header (§2.3) hosts both pure functions; two PRs would rebase on each other for nothing. (3) The #102 fix is **print-only**: no stage slot moves, no kernel or skel change, so #101's "no kernel change" holds and rung 2 stays unnecessary. Kept as two commits (K2a flip, K2b print) plus the header/host-check commit, so either half can be reverted alone; the PR closes both issues.

The issue's other hypothesis for #102 — "a 30th slot written into the index the print reads as `swiglu`" — is ruled out by reading: `MOE_T_PATH` / `HTP_MOE_T_PATH` sit at the same index (`nntr_hvx_mm_u8i4.c:900`, `htp_compute_ops.cpp:183`), `PATH` reads `1408/1408` correctly, and `SWIGLU` is filled from `HEXKL_PROBE_SWIGLU` on both paths (`:1018`). The number is right and the subtraction is wrong.

**Rejected for #102: kernel-side.** Dropping `moe_tail_probe_add` from the GEMV workers loses the lane-balance reading that ㉒ needs; moving the sum to a new slot grows `MOE_N_STAGES`, changes the IDL contract of `stage_usLen` (`nntr_hvx_mm_u8i4.c:1001` → `AEE_EBADPARM` on a stale skel), and forces a skel + app rebuild for a print.

### 3.4 LEDGER ㉒ — why `mm` rose 783 → 975 µs, and the next lever (for a derived issue, not this one)

On the HMX loop, `mm` (783) times the HMX issue loop while the weights arrive through the DMA ring into VTCM behind it (46 descriptors/call, `gather` 146 + `drain` 122 + `acc` 260 sit in other columns, and the ring's 21.5 MB averaged 15.6 GB/s over the 1414 µs call). On the GEMV path there is no ring: per active expert, stage A issues 56 pair-units, each two `hvx_gemm_u8i4_wh_col` columns (`:484-488`), and stage C 64 down-units of one column (`:529`); every column does `prefetch_col` (an `l2fetch` of the `k_tiles × 512 B` strip, `hvx_gemm_u8i4_wh.c:38-46`, called at `:124`) and then `gemm_rows4` streams that strip from the **uncached ION arena through L2 with vector loads** and issues all four row accumulators even at `m = 1` (`:90-105`, three of four dead — the `ponytail` at `hexkl_mm_u8i4_moe.c:435-437`). So `mm` now contains the whole weight feed plus the vrmpy work: 22.0 MB in 975 µs = **22.6 GB/s effective**, below the 34–38 GB/s the DDR probe reaches and well below the ceiling arithmetic of doc 48 §1, and `swiglu / mm = 5.75` says all six lanes were busy for 96 % of that wall — the path is lane-bound, not barrier-bound, but whether the lanes wait on the loads or on the dead vrmpy issue is not separable from this row. The next lever inside ⑯ is therefore two-pronged and needs one measurement to choose: (a) **feed** — stage each expert's `wh_bytes` into VTCM by DMA (one linear descriptor per weight, 3.5 + 1.75 MiB, expert e+1 pushed while expert e computes, so the GEMV reads VTCM and the ring is used with two descriptors per expert instead of row h's 46-descriptor chunk list — this is why ㉒ waits for #100's cause); (b) **compute** — a `gemm_rows1` that issues one accumulator at `m = 1` (4× fewer `vrmpy`, `hvx_gemm_u8i4_wh.c` changes → its bit-identity gate re-runs). A device microbenchmark in `unittest_hvx_mm_u8i4` that runs `hvx_gemm_u8i4_wh_col` over the same 22 MB from (i) the arena and (ii) a VTCM copy, at `m = 1` with the four-accumulator and a one-accumulator loop, separates the two in one sitting; gate for the follow-up = `mm ≤ 600 µs` at M==1 with text identical (LEDGER ㉒). **Suggested derived issue:** "㉒ GEMV weight feed: VTCM staging vs one-row GEMV — microbench then fix", blocked on #100.

## 4. Steps

Each ends in a rung of `.claude/skills/hexagon-gates`. Commits: **H** header + host check (`htp_moe_opts.h`, `moe_opts_host_check.c`, `run_host_checks.sh`), **K2a** the flip (`sendMoeOptsOnce`), **K2b** the print (`dump()`), **D** docs (`docs/htp_moe/*`, apart from H/K per contract §5). `clang-format-14` on changed lines before each commit (rung 0).

1. **H, red first.** Write `htp_moe_opts.h` and `moe_opts_host_check.c` with the assertions of §2.3; wire it into `run_host_checks.sh`. Gate: rung 1's `bash test/htp/host/run_host_checks.sh` prints the two new pass lines and still ends in `ALL CHECKS PASS` / `WORKER POOL LANES OK` (the check is pure arithmetic, so it goes green with the header alone — the "red" is the missing header before this step).
2. **K2a, the flip.** `#include <htp_moe_opts.h>`, delete the local define `:190`, resolve `flags` through the header, add `source=` to the proof line, keep the throw path. Gate: rung 1 — `ninja -C build`, `tools/htp_syntax_check.sh`, `*Lfm2Moe*` 6/6 (generate the tiny fixture first if the three differential tests would skip), `*qs4cx*`.
3. **K2b, the print.** `rest` through `htp_moe_row_rest_us(..., b.m1_calls != 0)`; the `weight DMA: n/a (...)` line under the condition of §2.2 with the `ponytail:` comment. Gate: rung 1 again (`ninja -C build`, syntax check); paste a hand-computed row for C's numbers in the commit body (`rest ≈ 1044.0 − 14.8 − 41.9 − 974.7 − …`) so the reviewer can check the sign.
4. **Rung 3.** `build_android.sh --htp` (`--cache`), `readelf -d` NEEDED lines, `ndk-build unittest_hvx_mm_u8i4`, md5s. Rung 2 is not run (no file under `test/htp/*.c`, `hmx/`, `hvx/` or the IDL changed); the PR records `md5sum test/htp/build/libnntr_hvx_skel.so` of the head it was built from as the skel these binaries expect.
5. **PR** into `htp_moe`, `Closes #101, closes #102`, one `<details>` per commit, the md5 table, the §3.1 deviation, the §2.4 statement that no #88 hunk was touched. `state:review`.
6. **Device — ride-along, not a handoff of its own (the tok/s effect is #94's verdict).** The implementer adds to the next sitting's handoff (today: `docs/measurements/88-moe-call-marshalling.md` when #88 reaches step 4; otherwise whichever comes first) these lines, ≈ 6 min:
   * **A** = that sitting's reference set, built from `htp_moe` after this PR merged: every A log must print `[HTP] moe m1 gemv: on (applied=0x1) source=default` exactly once (`grep -c`), and the level-2 A run's M==1 row `blocks=0 m1_gemv=1408/1408`, `rest` in [0, 5 %] of host, `swiglu / mm` ≈ 5.7, a `weight DMA: n/a (direct arena read …)` line; M>1 row `m1_gemv=0/23`.
   * **A0** (one cell) = A's binaries with `NNTR_MOE_HTP_M1_GEMV=0`, NPU model, G = 64, run once: `off (applied=0x0) source=env`, text identical to A's G = 64 run, prefill of A within −5 % of A0, decode expected ≈ −6 % of A (sitting 2's C/A inverted; a 0 % here means the default did not take, and voids the sitting's A as a GEMV reference).
   * If the sitting's A is built before this PR merges, A is the HMX path and #101's ride-along moves to the following sitting; the supervisor keeps BENCHMARK.md's "variant A path" column honest either way.

## 5. Risks

| risk | how it is made visible |
|---|---|
| **Stale skel on a phone is now loud where it was silent**: an old `libnntr_hvx_skel.so` (pre-`2a75f7d9`) throws at the first MoE call with nothing set | intended (rule 17, doc 46 §48.7); the throw text carries the rebuild hint; the handoff's step 3 pushes the skel from the same commit |
| The sitting's A is built from a tree without this PR (ordering with #88) | the `source=default` grep in step 6 is mandatory; a missing line means A is the HMX path and every later A/B is read against the wrong reference |
| DVFS: the GEMV path leaves the HMX idle, so the DSP may clock differently between A and A0 | already inside sitting 2's C vs A (same sitting, same binary); A0 is one cell, read only for text identity and the sign of the decode delta |
| Thermal drift between sittings | no cross-sitting reading is made; the +6–8 % is not re-proven here |
| Host-vs-device gap of the print fix | the row arithmetic is checked on the host with C's numbers, but only the ride-along's level-2 row shows the real `rest`; until then the PR body says "arithmetic checked on host, row not yet seen on device" |
| DMA rate / address space | untouched: no ring, VTCM, arena or heap change in this PR |
| Transport +100 µs/call in C (649 → 748, unexplained, LEDGER ⑯) becomes every A's transport | recorded, not addressed: #88's A now carries it, and #88's B/A reads the staging fix on top of it; the supervisor notes the new transport baseline in ⑦ |

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md` — Goals "now" row: "from PR #<n> variant A = GEMV path (`source=default`)"; Artifacts: one row for this PR's app set (md5, commit, "skel unchanged: `<md5>` from `<head>`"); Results: the A0 cell when the ride-along fills, tagged with its sitting.
* `docs/htp_moe/LEDGER.md` — ⑯: "default since PR #<n>; opt-out `NNTR_MOE_HTP_M1_GEMV=0`"; ㉑: closed by the same PR, cause = lane-time sum subtracted as a stage; ㉒: add the §3.4 reading (22.6 GB/s effective, 5.75 lanes busy, feed vs one-row GEMV undecided, microbench first) and the derived-issue suggestion; §1 gains a reading rule: "in a `m1_gemv=calls/calls` row, `swiglu` is Σ lane-time over the pool (6 lanes on v79), not a stage; `swiglu / mm` is lane utilisation"; ⑦ (transport) notes the 748 µs baseline every A carries from now on.
* `.claude/skills/hexagon-handoff/SKILL.md` expected lines (supervisor): variant A prints `moe m1 gemv: on (applied=0x1) source=default`; the old "an `on` in an A log voids the run" rule of sitting 2 is inverted.
