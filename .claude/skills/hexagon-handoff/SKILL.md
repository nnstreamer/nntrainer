---
name: hexagon-handoff
description: Write a device measurement handoff document (docs/measurements/<issue#>-<slug>.md) that the user runs on the workstation with the Galaxy S25 Ultra, always as a full-model end-to-end LFM2.5-8B-A1B inference, and read a filled one back. Use whenever an htp_moe task needs silicon numbers.
---

Agents never touch the phone. A handoff turns "we need a device number"
into a file the user can execute top to bottom in one sitting, then fill
in. Contract: `docs/plans/0001-htp-moe-decode-agent-system.md` §4.2.

Hard rules (user decisions 2026-09-21):

* **Always a full-model E2E inference** (`nntrainer_causallm` with the real
  8B model on the phone). Kernel gtests or probes may be added, never
  substituted.
* **Variant A = the unchanged reference binary, run first in the same
  sitting.** Every number is read as A/B within that sitting.
* At most 4 variants. Prompt 512 tokens; generation 64 / 512 / 1024
  (contract §1.1). `NNTR_NUM_THREADS=8`. Never a `--profile` build for
  tok/s.
* Accuracy columns are mandatory: text identical to the CPU `q40` run
  (y/n, first differing token index), `NNTR_L2_DIFF` result when the
  variant changes DSP arithmetic. **Handoffs that run the per-token entry
  (`NNTR_HTP_FORWARD=1`, #130 onward; user decision 2026-09-28) add two
  things:** a `PPL` column (`NNTR_PPL=1` on one extra run per variant and
  G=64, the CPU `q40` PPL of the same prompt as the reference, fail above
  +2 % unless the plan says otherwise) and a **text approval** section —
  the filled handoff pastes A's and every variant's generated text (G=64,
  run 1) in full, and the user marks `text approved: y/n` per variant.
  Differing text is not a fail by itself; an unapproved text is not a
  pass; the supervisor folds only approved rows.
* Every artifact row has an md5 and the commit it was built from; the
  user copies the md5 the run prints or `md5sum` on the device shows.

## Template

```markdown
# Measurement <issue#>: <one-line purpose>

Branch `htp/<issue#>-<slug>` @ `<sha>` — estimated device time: <N> min

## Why
<two sentences: the question and the decision that hangs on it>

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)
| file | md5 | built with |
|---|---|---|
| test/htp/build/libnntr_hvx_skel.<variant>.so | | `HEX_EXTRA_CFLAGS=...` |
| Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm | | `build_android.sh --htp` |
| Applications/CausalLM/jni/libs/arm64-v8a/libnntrainer.so | | |
| /local/mnt/workspace/models/lfm2.5-8b-a1b/q40/*_ARM.bin | | CPU control |
| /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/*_ARM.bin | | NPU model |

## Steps (workstation, phone on USB)
1. `git fetch && git checkout htp/<issue#>-<slug>`; `source tools/htp/env.sh`; rebuild or reuse the artifacts above (`md5sum -c staged/md5.txt` before pushing; it exits 1 on a mismatch, and the device's `md5sum` is compared by hand).
2. `adb devices` lists exactly one device; record its serial under Notes (any S25 Ultra is allowed, contract §4.2 — the handoff never names one). Note battery % and whether the phone is warm.
3. Install once: `(cd Applications/CausalLM && ./install_android.sh --model=<model dir>)` for each model dir, then
   `adb push test/htp/build/libnntr_hvx_skel.A.so /data/local/tmp/nntrainer/causallm/libnntr_hvx_skel.so`.
   Never push `builddir/jni/arm64-v8a/libcdsprpc.so`.
4. For each generation length G in 64 512 1024 and each variant (A first):
   set `"num_to_generate": G` in the model's `nntr_config.json` on the device, swap the skel if the variant has one, then
   `adb shell "cd /data/local/tmp/nntrainer/causallm && NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/<model> '<prompt file or text>'"`
   Expected lines: `prefill: 512 tokens, <ms> ms, <tps> TPS`, `generation: <G> tokens, <ms> ms, <tps> TPS`, the generated text.
   Repeat twice; keep both.
5. Profiles (one run each, not for tok/s): `NNTR_HTP_PROFILE=2 ...`, `NNTR_M0_PROFILE=1 ...`; paste the `[HTP-PROFILE]` / `[M0-PROF]` lines.
6. Paste the summary lines below, commit this file on the same branch, push, set the issue to `state:measured`.

## Results (fill in)
| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS | text = CPU q40? | skel md5 (device) |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | | | |

Reference: NPU now 20.8 / CPU now 48 decode tok/s, NPU prefill 523 (PR doc 49; replaced by the first handoff). Goal ≥ 50, prefill ≥ 497.

## Text approval (per-token-entry handoffs only)
| variant | PPL (NNTR_PPL, G=64) | generated text (G=64, run 1) | text approved (user: y/n) |
|---|---|---|---|
| A | | <paste> | (reference) |
| B | | <paste> | |

## Notes from the run
<thermal, first-run page faults, FARF/AEE errors, anything stale>
```

## Dump ride-along (optional, never in a tok/s cell)

`NNTR_HTP_DUMP=/data/local/tmp/dump` on one decode run (prompt 16, G = 4)
writes every MoE call's input / output and a manifest; pull the directory
and run `tools/htp/htp_dump_eval.py <host dump of the same prompt on the
x86-packed twin> <device dump>` on the workstation. `bit_identical=0`
names the first call and side that differs (plan 84 §3.1 level (d)).

## Reading a filled one (supervisor / implementer)

1. Check md5s against the artifact table; a mismatch voids the run.
2. Read every B..D against A **from the same sitting**; classify
   goal-progress / regression / no-change with the plan's threshold (default
   ±5 %). A prefill below −5 % of A is a regression regardless of decode.
3. Copy rows to `docs/htp_moe/BENCHMARK.md`; design verdicts and any
   silicon rule (device disagrees with host reasoning) go to
   `docs/htp_moe/LEDGER.md` before anything else.
4. If the text differs from the CPU run, the variant failed the accuracy
   gate: file it, do not average it in. **Exception (per-token-entry
   handoffs):** a differing text with `text approved: y` and PPL within
   the threshold passes; with `text approved` empty the row is left
   unfolded and the issue stays `state:measured` with a `needs-user` label
   until the user fills it; with `n` or a PPL above the threshold it fails.
