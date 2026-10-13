# PR 4416: aligning the branch with upstream main

Plan for bringing `Seunghui98:htp/s26-prefill-opt` (PR nntrainer#4416) onto
upstream `main`. Written 2026-10-13 from a read-only survey of both trees;
nothing in it is implemented yet. The rule that shaped it: **the HTP graph
that runs Gemma-4 26B on the S26 Ultra must come out byte-identical**, so
every choice below is judged first by "does the NPU path change".

## 1. Where the two trees stand

| | value |
|---|---|
| PR head | `1b6006673` (= local `htp/4416-remaining`, 0 ahead / 0 behind) |
| upstream main | `6ca5d2b77` (2026-10-13) |
| merge base | `aad932ce3` (2026-09-30) |
| PR commits past the base | 1454 (126 merges, 114 of them fork PR merges) |
| main commits past the base | 535 |
| PR files changed vs base | 860 (+196k lines); code only 319 (+86k) |
| files changed on both sides | 92 |
| files that conflict | 51 |
| GitHub state | CONFLICTING, DCO check failing, no reviews |

Upstream allows **rebase-merge only** (no squash, no merge commit), so the
PR has to end up as a linear series. Replaying 1454 commits with 126 merges
through `git rebase` is not a realistic path; see section 4.

The PR branch contains all of `origin/htp_decode` (htp_moe → htp_decode →
this PR), which is why the history is that long.

How the numbers were taken:

```bash
git fetch upstream main
git fetch https://github.com/Seunghui98/nntrainer.git htp/s26-prefill-opt
MB=$(git merge-base FETCH_HEAD refs/remotes/upstream/main)
git rev-list --count $MB..FETCH_HEAD                      # 1454
git merge-tree --write-tree refs/remotes/upstream/main FETCH_HEAD  # conflicts
```

(The worktree had a stale local ref `refs/upstream/main` at `7d14d001c`
that makes `upstream/main` ambiguous; delete it before starting.)

## 2. What conflicts, by cause

A conflict means both sides changed the same place of the same file since
`aad932ce3`. The 51 files group into four causes.

### 2.1 Gemma-4 MoE: the same code arrived twice (9 add/add files)

Upstream merged Jungwon-Lee's #4296 "[CausalLM] add Gemma4 MoE support" on
2026-10-13 (6 commits, 35 files, +2971 lines). Our branch had pinned the same
#4296 into htp_decode on 2026-10-06, then merged upstream #4415 on top. The
rebased SHAs differ, so git sees two unrelated additions of:

- `Applications/CausalLM/models/gemma4/gemma4_moe_causallm.{cpp,h}`
- `Applications/CausalLM/res/gemma4/gemma4_moe_weight_converter.py` (1 line apart)
- `test/unittest/models/causallm_reference/gemma4_moe_tiny/*` (different fixture)
- `test/unittest/models/causallm_reference/generators/generate_gemma4_moe_reference.py`

`gemma4_moe_layer.{cpp,h}` are **byte-identical** on both sides (main's
"[gemma4] optimize MoE prefill" is already in the pin). The real difference
is how the MoE block is built:

| | main (#4296, CPU) | this PR (via #4415, HTP) |
|---|---|---|
| MoE layer type | `gemma4_moe` | `lfm2_moe` with `router_type=softmax_scale` |
| built in | `Gemma4MoECausalLM::createFeedForwardBlock` (subclass) | `Gemma4Transformer::createTransformerDecoderBlock` behind `ENABLE_MOE_BLOCK` |
| norms | six separate `rms_norm` layers | folded into the MoE layer (`in_norm`, `router_norm`, `out_norm`) and the residual add (`out_add`) |
| subclass role | builds the block, registers the layer | only the one-PD decode list (`NNTR_HTP_E2E`, `load_weight`) |

As the PR stands, main's `createFeedForwardBlock` is gone and `gemma4_moe`
is compiled but never created: a rebase that keeps this would silently
disable code merged the day before. Resolution in section 3.

### 2.2 mha_core.cpp: both sides grew the same functions

| version | lines | what was added |
|---|---|---|
| base | 1602 | |
| PR | 2451 | +856: int8 KV cache, a16 / kv8 path, routing attention to the HTP |
| main | 6216 | +4848: CUDA / OpenCL / host flash attention arms |

Both insert branches into the same functions. Nothing is contradictory;
main's file is the base to keep and our 856-line delta is re-applied on it
(`git diff aad932ce3 1b6006673 -- Applications/CausalLM/layers/mha_core.cpp`
as a 3-way patch).

### 2.3 A file main deleted (1 modify/delete)

`Applications/CausalLM/layers/tie_word_embedding.cpp` was removed on main
and its content moved to `nntrainer/layers/llm/tie_word_embedding.h`. Our
edit moves to the header.

### 2.4 Shared files both sides touched a little (about 40 files)

- `nntrainer/tensor/qs4cx_tensor.{cpp,h}`: main packs a QS4CX weight once,
  in parallel, and releases the plain copy (zero-copy load); we pack
  "whatever the engine" and keep height-1 tensors on save. Both intents
  have to survive in one function.
- `cpu_ops_table.h`, `compute_ops.h`, `fallback.{cpp,h}`, `x86_compute_backend.cpp`:
  both sides appended new ops to the same tables. Keep both.
- `meson.build`, `jni/meson.build`, `Applications/CausalLM/jni/Android.mk`,
  `Applications/CausalLM/{,layers/,models/,models/gemma4/}meson.build`: both
  sides appended sources. Keep both.
- `main.cpp`, `transformer.{cpp,h}`, `causal_lm.cpp`, `kv_cache_manager.{cpp,h}`,
  `neuralnet.cpp`, `layer_devel.h`, `qkv_layer.h`, `tensor_dim.h`, `.gitignore`,
  `build_android.sh`, `build_tokenizer_android.sh`, `quantize.cpp`,
  `weight_converter.py`, `gemma3/lfm2/qwen3_causallm.cpp`, three unit tests:
  option and branch additions on both sides, read line by line.

### 2.5 Not conflicts

`nntrainer/tensor/htp_backend/` (107 files, main has 5 and 4 of them are
unchanged by us) and `test/htp/` (181 files) are ours alone and drop in.

## 3. Gemma-4 MoE resolution: HTP graph untouched, CPU falls back to main

### 3.1 Why the obvious split is wrong

Moving the `lfm2_moe` block into `Gemma4MoECausalLM::createFeedForwardBlock`
at main's seam (after `post_attention` exists) loses the decisions that the
PR takes **before** that seam in `createTransformerDecoderBlock`:

- `block_on_htp` and `out_to_host=false` on `post_attention_add` (the
  11.5 MB-per-layer host round trip of a 1024-row tensor at prefill),
- `keep_dense_out_` / `keep_moe_out_` (what `createMlp` and the next layer's
  qkv read directly on the DSP),

and **after** it:

- `createMlp` folding the dense out-norm and skipping the host copy,
- `createMoe(..., &ffn_out)` with `out_add`: residual, norms, scalar in the
  MoE call (one FastRPC call and three activation copies a layer, doc 57 §9.30).

That split would re-introduce a host copy per layer at prefill. Rejected.

### 3.2 The resolution

Keep the PR's `createTransformerDecoderBlock` as it is and add one early
return right after `post_attention` is built:

```cpp
// after post_attention_add; everything below is the PR's HTP path, unchanged
if (ENABLE_MOE_BLOCK && MOE_ENGINE != "htp")
  return createFeedForwardBlock(layer_id, post_attention, is_kv_shared_layer);
```

- `createFeedForwardBlock` comes back as the virtual main has; the subclass
  body is main's (`gemma4_moe`, six norms), unchanged.
- Our additions stay in the subclass next to main's `setupParameters` /
  `registerCustomLayers`: the `NNTR_HTP_E2E` flag, `set_moe_geglu`, the
  `load_weight` override that builds the decode list.
- The branch is on the **global** `MOE_ENGINE`, not per layer: a mixed
  `MOE_HTP_LAYERS` run keeps the PR graph (`lfm2_moe`, cpu engine) for its
  CPU layers, so any run that involves the HTP is the graph measured today.
  main's `gemma4_moe` serves CPU-only runs.
- Tests: main's `gemma4_moe_tiny` fixture and `unittest_causallm_gemma4.cpp`
  stay as main has them (CPU truth). Our head_dim-8 variant of
  `gemma4_moe_tiny` is dropped; `gemma4_moe_tiny_hd64` (already a separate
  directory) stays as the CPU bit-reference of the HTP topology and
  `unittest_causallm_gemma4_moe_reference.cpp` runs it with `moe_engine=cpu`.
- `gemma4_moe_weight_converter.py`: take main's, port the one differing line
  if still needed.

Cost accepted and to be stated in the PR: two CPU topologies for Gemma-4
MoE exist (main's `gemma4_moe`, and `lfm2_moe` on the cpu engine as the
reference for the HTP path). The HTP MoE kernel stays the one LFM2 already
uses; no second NPU kernel.

One check with new code possible: both topologies must load the same
converted file. main's topology stores the norms as separate layers
(`layerN_pre_ffn_norm_2`, `post_ffn_norm_2`, ...); the PR's holds them
inside `_sparse_moe` and its converter folds the router scale into a gamma.
If the names or the folding differ, a load-time alias map in the subclass
(and at most one converter flag) closes it. This affects the CPU-only path
and the fixtures, not the device.

Upstream #4415 (same author, base main) has no `gemma4_moe_*` files at all
and meets the same conflict; this branch is a superset of #4415, so the split
above should be the one #4415 takes too. Agree with the author before either
PR is rebased.

## 4. Method: re-apply on main, not replay

1. **Freeze.** Tag the PR head (`pr4416-pre-rebase`) and push the same
   commit to the fork as `htp/s26-prefill-opt-pre-rebase`.
2. **New branch from `upstream/main`.** Bring the PR's end-state tree over
   in a few logical commits instead of 1454:
   - (a) files only we touched: copied as they are
     (`htp_backend/`, `test/htp/`, `tools/htp/`, new CausalLM files);
   - (b) files both sides touched that auto-merge (about 40): take the
     `git merge-tree` result;
   - (c) the 51 conflicts: by hand, per section 2 and 3.
3. **Commit series** (DCO `-s`, `Co-Authored-By` on agent commits,
   `[<component>] <subject>`, `clang-format-14` on changed lines):
   - `[HTP]` backend kernels, graph, session (`nntrainer/tensor/htp_backend`)
   - `[HTP]` kernel tests (`test/htp`)
   - `[tensor]` cpu_backend ops, qs4cx packing, tensor_dim
   - `[CausalLM]` Gemma-4 on the HTP: fused layers, decode, the MoE split of §3
   - `[CausalLM]` attention: int8 / a16 KV and the HTP routing in mha_core
   - `[build]` meson / jni / Android.mk / build scripts
   - `[test]` unit tests and the hd64 fixture
   - `[docs]` the minimum (see §6)
4. **Push** to the fork's `htp/s26-prefill-opt` with `--force-with-lease`
   after the gates in §5 pass. This rewrites the PR branch: needs the
   user's go.

Alternative considered: merging main into the branch. Same conflict work,
keeps the 1454-commit history, and upstream refuses merge commits anyway.
Useful only as a scratch tree for measuring, not as the deliverable.

## 5. Gates: the device decides

Before / after on the S26 Ultra (R5KL309ALRK), same configs
(`$D/m_p{P}_g{G}`), alternating runs per the log's §6 rules.

| item | pass |
|---|---|
| output md5 | p1024 `fe44d523`, p2048 `aff26409` |
| first token | p1024 ≤ 506 ms, p2048 ≤ 636 ms |
| decode tok/s | before / after inside the alternating-run noise |
| `NNTR_HTP_PROFILE=1` stage lines | same calls, same order, same counts |
| host RSS, DSP maps | same as before (DSP ≈ 3.3 GB, staging 370 MiB) |
| host build + unit tests | `meson build -Denable-transformer=true`, main's Gemma-4 tests and the hd64 test green |
| Hexagon v81 build | clean |

The rebase is not done until this table is filled.

### 5.1 main-side changes that can move those numbers

These are not Gemma-4 MoE conflicts; they are the places where main
changed code the HTP path runs through.

1. `qs4cx_tensor`: main's pack-once / zero-copy / release-plain-copy vs our
   2026-10-12 "QS4CX FC copies dropped at load" (`660652c97`). Re-measure
   first token and host RSS after merging the two.
2. `transformer.cpp`: main packs QS4CX inside `load_weight()`. Order against
   our "quantized KV caches registered after the lm_head" and "MoE warm-up
   at load" decides the load-time peak. Re-measure.
3. `mha_core.cpp`: main's new arms and defaults ("CPU decode attention off
   the host flash path", "FP32-activation attention on the device arms",
   smaller ARM flash tiles). Keep our HTP attention branch (ATTN_M1, a16 /
   kv8) first in the dispatch and check that the Android default did not
   move to a new arm.
4. `neuralnet.cpp` / cpu_backend tables from main's "optimize MoE prefill"
   (`6ca5d2b77`) against our D5 (skip the 1 MiB logits copy). Mechanical.

## 6. Decisions still open

1. Gemma-4 MoE split of §3: agreed in principle here; the author of #4415
   should confirm since that PR needs the same split.
2. Docs to keep in the upstream PR. Recommendation: `docs/backend_guide`
   only; drop `docs/measurements` (194 files), `docs/htp_attention` (71),
   `docs/plans` (60), `docs/htp_moe` (7), `.claude` (7) from the PR.
   `tools/htp` (14) stays if the build scripts need it.
3. Commit granularity: the eight-commit series of §4, or finer.
4. Go for the force-push of §4 step 4.
