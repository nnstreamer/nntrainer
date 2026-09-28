# Plan 89: print `generation(last 64)` from the base CausalLM report block

Issue: #89 (p2, tracker #76, LEDGER ⑭ / rule 16). Contract §1.1 asks for
decode tok/s "over the whole generation and separately over the last 64
tokens"; every NPU row in `docs/htp_moe/BENCHMARK.md` since #77 reads
`n/a (line not printed, #89)` in that column. Report-only change on the ARM
side. No IDL, no DSP code, no forward-path change.

## 1. Goal and gate

From the issue, made measurable:

* `ninja -C build` and
  `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` → `[  PASSED  ] 6`
  (3 differential + 3 tiny-model, none skipped). These tests run
  `num_to_generate: 1` and compare logits, so they prove "nothing broke",
  not "the line prints".
* The line prints for LFM2-MoE: proven by the next handoff's log — every
  A run (G = 64 / 512 / 1024) carries exactly one
  `generation(last 64): 64 tokens, <ms> ms, <tps> TPS` line — and by
  reading the diff (the timestamp pair lives in
  `Applications/CausalLM/models/causal_lm.cpp`, not in a subclass).
* The existing `generation:` line is byte-for-byte unchanged in format
  (BENCHMARK.md and every handoff's
  `grep -E '^(prefill|generation|total|peak memory)'` depend on it).
* G < 64 (or an early EOS / stop) prints the line over the whole
  generation and says so.

Standing gates: prefill ≥ −5 % of variant A and text identical to the CPU
run are unaffected by construction (one `now()` per generated token, taken
after `registerOutputs`; nothing in the forward path reads it). No
`NNTR_L2_DIFF` column: DSP arithmetic is untouched.

## 2. Where it lives

Verified on `htp_moe_cycle` @ `63a20f69`:

* `Applications/CausalLM/models/causal_lm.cpp:625` — `start_generation`
  timestamp; `:627-676` the decode loop; `:648` `++generation_cnt`
  (after `registerOutputs`, the point where a token is "done");
  `:686-689` `generation_duration`; `:699-711` the base report block
  (`prefill:` `:700`, `generation:` `:704`, `total:` `:708`,
  `peak memory:` `:709`). `<chrono>` and `<vector>` are already in use.
* `Applications/CausalLM/models/lfm2/lfm2_causallm.cpp:468-473` —
  `Lfm2CausalLM::run` delegates to `CausalLM::run` unless `USE_EMBEDDING`
  (`:327`, from `nntr_config.json`). None of the three device model dirs
  (`q40`, `q40-qs4cx-wh`, `q40-qs4cx-wh-had` under
  `/local/mnt/workspace/models/lfm2.5-8b-a1b/`) sets `use_embedding`, so
  `Lfm2MoeCausalLM` (`models/lfm2_moe/lfm2_moe_causallm.h:26`, does not
  override `run`) reports from the base block — exactly the issue's
  finding. The duplicate report block in `run_with_embeddings`
  (`lfm2_causallm.cpp:757-768`) is **not** touched (see §3).
* The LFM2-only copy the issue says to delete (`64a50bcf`) exists only on
  `htp/77-first-handoff`; it was never merged into `htp_moe`, so on this
  tree there is nothing to delete. The plan re-uses that commit's hunk,
  relocated.

Consumers of the changed contract — none of the listed ones move: no IDL
(`test/htp/nntr_hvx.idl`), no stub, no `HtpComputeOps`, no quantizer tag,
no loader check, no `NNTR_HTP_PROFILE` table, no `tools/htp_fc_report.py`.
`performance_metrics.h` (`TransformerPerformanceMetrics`) stays as is: no
API consumer asked for the last-64 number and the struct is C ABI shared
with the API layer; adding a field is the upgrade path if one does.

The handoff tooling picks the line up for free:

* Every handoff's per-run filter is
  `grep -E '^(prefill|generation|total|peak memory)…'` (77:133, 94:240,
  100:168, 113:306), which matches `generation(last 64):` as well, so the
  summary block the user pastes shows it without editing the recipe.
* The text gate is **not** affected. The recipe in use since #105
  (`105-m1-gemv-compute.md:255`) is
  `sed -n '/^=====/q;p' <log> | grep -v 'moe m1 gemv\|libnntr_hvx_skel'`,
  i.e. it stops at the `=================[ LLM with NNTrainer ]===` banner.
  The new line is printed inside that block, after `generation:`, so it is
  excluded from the comparison — which is what must happen: it is a timing
  line that legitimately differs run to run, not generated text. The
  earlier `grep -v '^\[HTP\]'` recipe (100:589) would include it and make
  every A/B `DIFFERENT`; the handoff skill should keep the `sed '/^=====/q'`
  form (the skill file itself says "byte-identical logs excluding the
  `[HTP]` banner" in prose only; the executable recipe is the one in the
  handoff documents).

## 3. Design

One `std::vector<time_point>` per `CausalLM::run`, reserved to
`NUM_TO_GENERATE + 1`, seeded with `start_generation`, `push_back(now())`
right after `++generation_cnt` (`:648`). In the report block, after the
`generation:` line and before `total:`:

```
generation(last 64): 64 tokens, <ms> ms, <tps> TPS
```

with `ms = ts[n-1] - ts[n-65]` when `n >= 65`; otherwise
`generation(last 64): <generation_cnt> tokens, <ms> ms, <tps> TPS (whole generation, fewer than 64)`
with `ms = ts[n-1] - ts[0]`. The prefix `generation(last 64):` is the same
in both cases so one grep finds it; the token count and the suffix say
which case it was. Seeding with `start_generation` means a G = 64 run has
its 64 intervals and equals the whole-run number by construction (the
#77 commit message states this; keep it). Same `milliseconds` cast and the
same `count / ms * 1000` arithmetic as the neighbouring lines, so a 0 ms
tiny-model run prints `inf` exactly as `generation:` already does — not a
new behaviour. Guarded by the same `if (log_output)`.

Cost: one `steady`-class `now()` per token (tens of ns) against ≈ 35 ms
per token; `NUM_TO_GENERATE + 1` × 8 bytes (8 KB at G = 1024) allocated
once before the loop, so no allocation inside it. Off nothing: no env
switch, no config key.

Rejected alternative: a single timestamp taken when
`generation_cnt == NUM_TO_GENERATE - 64`. Fewer bytes, but the loop can
leave early (EOS at `:664-666`, `stop_requested_` at `:672`), in which
case "64 tokens before the end" is not known in advance and the printed
number would silently be over a different window. The vector makes the
window always the true last 64 completed tokens and handles G < 64 with
the same data. A 65-slot ring buffer would do the same in fixed memory; not
worth the index arithmetic for 8 KB.

Not done, on purpose (`ponytail:` comment in the diff): the duplicate
report block in `Lfm2CausalLM::run_with_embeddings`
(`lfm2_causallm.cpp:757`) does not get the line. That path is reached only
with `use_embedding=true`, which no benchmark model dir sets; the day it
is, the same hunk moves there (or the two report blocks are folded into
one helper — the better fix, but a refactor the issue did not ask for).

## 4. Steps

1. **Edit `causal_lm.cpp`** — the three hunks above (declare + seed before
   the loop at `:625`, `push_back` after `:648`, the print after `:707`).
   Commit subject `[CausalLM] Print decode tok/s over the last 64 tokens
   from the base report block`, body ≥ 8 words, `-s`, `Co-authored-by`.
   Gate: rung 0 — `clang-format-14 -i Applications/CausalLM/models/causal_lm.cpp`,
   `git diff --stat` shows only that file.
2. **Host build and tests.** Gate: rung 1 —
   `ninja -C build`;
   `NNTR_QUANTIZE_BIN=… NNTR_QUANTIZE_STREAM_BIN=… ./build/Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2Moe*'`
   → `[  PASSED  ] 6`, none skipped (regenerate the tiny fixture weight
   first if the differential tests skip). `bash tools/htp_syntax_check.sh`
   exits 0. `run_host_checks.sh` is not required (no `test/htp/` change)
   but is free to run.
3. **Read the diff against the acceptance list**: `generation:` line
   untouched (`git diff` shows no `-` on `:704-707`), timestamps in
   `causal_lm.cpp` only, `(whole generation, fewer than 64)` branch
   present. This is the "by reading the diff" half of the criterion.
4. **Android app build** (rung 3, `build_android.sh --htp --cache`) only
   as part of the next sitting's artifact set — the change rides whichever
   handoff is next (cycle 15's flip of the VTCM feed to the default, or
   #85's wiring) at no extra device time. No skel rebuild (rung 2) — no
   `test/htp/` or `htp_backend/` file changes.

**Device step: none of its own.** The line is proven when the next
handoff's A logs carry it; the supervisor reading that handoff fills the
"decode tok/s (last 64)" column for A and every B..D and closes #89 with
the file name. Until then the acceptance rests on steps 2-3 plus the diff.
Say so in the PR: "line not yet seen on a device; rides the next sitting".

## 5. Risks

* **The number is a real-time reading of the tail, so it is more
  thermally sensitive than the whole-run mean.** At G = 1024 the last 64
  tokens are the hottest 2 s of a ≈ 40 s run; the handoff already records
  `thermal_zone0` and runs A first in the same sitting, so an A/B on this
  column is read the same way as the whole-run column (rule 13, same
  sitting only). Do not compare last-64 numbers across sittings.
* **Host cannot prove the line.** The gtests generate one token; there is
  no host E2E on this tree (#84 pending). The G < 64 branch is exercised
  only by a tiny-model run or an EOS-terminated device run; it is a print
  of the same data, reviewed by reading. Say "not run" rather than
  "verified" for that branch until a log shows it.
* **Two `generation` matches per log.** Any script that does
  `grep generation | awk` (rather than `^generation:`) now sees two lines;
  the recipes in the handoffs use `grep -E '^(prefill|generation|…)'` for
  display only and read numbers by hand, so nothing breaks, but the next
  handoff should quote both lines as expected output so the user is not
  surprised.
* **Text gate recipe drift.** If a future handoff reverts to
  `grep -v '^\[HTP\]'` for the text diff, the new line will make every
  comparison `DIFFERENT`. §2 fixes the recipe form; the hexagon-handoff
  skill's prose should be aligned with it when the skill is next edited
  (not in this PR — `.claude/` is out of this plan's write set).
* Stale binary: an A log without the line after this merges means the
  app was built from an older tree; the handoff's md5 table catches it.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: no new row from this PR. From the next
  sitting on, the "decode tok/s (last 64)" column stops reading
  `n/a (line not printed, #89)`; the row that first carries it cites this
  PR. Method paragraph (lines 9-12) already describes the column.
* `docs/htp_moe/LEDGER.md`: ⑭ → "landed in PR #<n>, first seen on
  device in <handoff>"; rule 16 stands as written (it is now enforced by
  the code location).
* `docs/plans/0001-…` §1.1: no change.
* Next handoff document: add `generation(last 64): …` to the "Expected
  lines" list of step 4 and keep the `sed -n '/^=====/q;p'` text-diff
  recipe.
