# LoRA Training for Qwen3 CausalLM

This document is a full account of the LoRA (Low-Rank Adaptation) fine-tuning
support added to the CausalLM application: why it was needed, what was built,
every bug found along the way (especially the NaN investigation), how each
was diagnosed and fixed, how it was all verified, and how to use it. It is
meant to be exhaustive enough that someone who wasn't in the room can
understand not just *what* changed but *why*, and can reproduce the
verification independently.

## Table of contents

- [Background](#background)
- [Scope](#scope)
- [Architecture: what was added, and why it works this way](#architecture-what-was-added-and-why-it-works-this-way)
- [The NaN investigation](#the-nan-investigation)
- [The LoRA initialization order bug](#the-lora-initialization-order-bug)
- [Was it gradient clipping?](#was-it-gradient-clipping)
- [Two memory bugs](#two-memory-bugs)
- [A silent-wrong-answer bug in validation](#a-silent-wrong-answer-bug-in-validation)
- [Verification performed](#verification-performed)
- [How to train](#how-to-train)
- [Fast on-device training: int8 backward and the GPU-trunk / CPU-head split](#fast-on-device-training-int8-backward-and-the-gpu-trunk--cpu-head-split)
- [Batch training (`--batch_size`)](#batch-training---batch_size)
- [Loss on every position (`lm_head_loss`)](#loss-on-every-position-lm_head_loss)
- [Decision models (RLCD)](#decision-models-rlcd-nntr_decide---task-decision)
- [Two silent training bugs: clipped gradients sharing memory, frozen inputs overwritten](#two-silent-training-bugs-clipped-gradients-sharing-memory-frozen-inputs-overwritten)
- [Commit structure](#commit-structure)
- [Known limitations / follow-ups](#known-limitations--follow-ups)

## Background

An earlier fork of nntrainer had backprop and LoRA training wired up for
Qwen3. By the time this work started, mainline had moved 344 commits past the
fork point, and the exact files the fork touched had been substantially
rewritten — most importantly, attention (`mha_core`) had moved from an
internal, self-managed KV cache to an externally-owned `KVCacheManager` with
absolute cache-index addressing. The fork's training code assumed the old
internal-cache-only design and could not be reapplied as a patch; the
backward math had to be re-derived from scratch against the *current* forward
implementations, not ported.

Before this change, the CausalLM application was inference-only end to end:
`calcDerivative`/`calcGradient` were either absent or empty stubs on every
layer in the Qwen3 stack, and `NeuralNetwork::initialize()` as called from
`Transformer` hardcoded `ExecutionMode::INFERENCE`.

## Scope

Plain FP32 LoRA training on Qwen3-0.6B: dense, causal attention, no sliding
window, no attention sink, no logit softcapping. Quantization-aware LoRA
(Q4_0 / Q6_K / W4A8) is a deliberately separate, later effort — the training
math here assumes full-precision weights throughout.

## Architecture: what was added, and why it works this way

### Backward math for six layers

None of these supported training before this change:

**`rms_norm.cpp` / `reshaped_rms_norm.cpp`** — standard RMSNorm backward:

```
dx      = inv_rms * (gamma*dy - x * mean(gamma*dy*x) * inv_rms^2)
dgamma  = sum_over_rows(dy * x * inv_rms)
```

`inv_rms` is cached in an `ITERATION_LIFESPAN` tensor during the forward pass
so `calcDerivative` doesn't recompute the norm from scratch. `gamma` is
frozen under the default LoRA-only recipe (it's never a LoRA target), so
`calcGradient` here only fires when `lora_train_norms` opts into training the
norms alongside the adapters. `reshaped_rms_norm` applies the same math
per `feature_size` chunk (used for Qwen3's per-head `q_norm`/`k_norm`) and is
a no-op when `use_gamma` is false.

**`swiglu.cpp`** — SwiGLU backward:

```
Swish(g) = g * sigmoid(g)
d_up     = Swish(gate) * dy
d_gate   = Swish'(gate) * up * dy
```

No weights, so no `calcGradient`. This is the layer where the [aliasing bug](#the-nan-investigation)
was first found.

**`lm_head.cpp` / `tie_word_embedding.cpp`** — both need to handle
right-padded training batches, where a sample's last *real* token isn't
necessarily the last row of the sequence (only that row's logits actually
feed the loss; every other row would just add noise gradient). A module-level
global (`g_lm_head_read_row` / `g_tie_embedding_lm_head_read_row`, sentinel
`UINT_MAX` meaning "use `height - 1`") tells `forwarding()`/`calcDerivative()`
which row to project to vocab logits. `tie_word_embedding` derives this value
itself, inside its own `forwarding()` embedding-mode branch, by scanning
token ids for the last non-pad position — see
[the thread_local bug](#bug-1-the-thread_local-thread-visibility-bug) for why
that design replaced an earlier one.

`calcGradient`:
  - `lm_head`: `dL/dW` and `dL/dbias` accumulate only from the read row.
  - `tie_word_embedding`: embedding mode scatter-adds `dy * scale` into the
    row named by each token id (repeated ids accumulate, matching the
    embedding lookup's read semantics); `lm_head` mode accumulates the same
    outer-product gradient as `lm_head`. Both are gated on
    `context.isGradientFirstAccess(...)` because the weight is *shared*
    between the two mode instances (tied embeddings) — without that gate,
    whichever mode's `calcGradient` runs second would silently double-count
    or overwrite the other's contribution.

**`mha_core.cpp`** — training forward reuses the *existing* internal-cache
machinery (`incremental_forwarding` over the full sequence, `from=0` to
`to=seq_len`) rather than reimplementing QKᵀ/softmax/·V from scratch, so
training numerics are provably the same code path as the trusted inference
prefill path. This is checked directly by a dedicated test
(`MhaCoreTrainForwardMatchesInferencePrefill`), not just assumed.
`calcDerivative` caches RoPE'd Q/K, raw V, and the softmax attention weights
during the forward pass (new tensors sized off `query_dim.height()`, i.e. the
actual sequence length — see [the memory bugs](#two-memory-bugs) for why this
matters), then computes: masked-causal `d_attn = dy·Vᵀ`, GQA-grouped `dV`,
softmax backward, `dQ`/`dK` through the cached attention weights, and finally
an inverse RoPE rotation on the `dQ`/`dK` gradients using the same cached
rotation angles the forward pass used. `mha_core` has no learnable weights of
its own (Q/K/V/O are separate FC layers), so `calcGradient` stays a no-op.

This layer's training path is plain scalar C++, not SIMD or multi-threaded
like the inference path — see [Known limitations](#known-limitations--follow-ups).

### A pre-existing crash fix (unrelated to training)

`nntrainer/layers/fc_layer.cpp`'s `incremental_forwarding()` unconditionally
read the LoRA scratch tensors `loraTmp`/`loraOut` via `context.getSharedDataTensor(...)`.
Those tensors use `FORWARD_GRAD_LIFESPAN`/`FORWARD_FUNC_LIFESPAN` and are
never allocated in a pure-inference (no-backward) compiled graph — so calling
existing mainline LoRA inference support (`lora_rank` was already a
recognized property before this work) with incremental decoding would crash.
Fixed by computing the LoRA scratch math into freshly-allocated local
`Tensor` objects instead of pulling from possibly-null context storage. This
is a standalone fix, independent of everything else in this document.

### LoRA config threading

`lora_rank` / `lora_alpha` / `lora_target` were already being read out of
`nntr_config.json` into unused fields; two more were added:
`lora_clip_grad_by_norm` and `lora_train_norms`. These are threaded through
every FC projection construction call in `Transformer::createAttention`/
`createMlp` and `Qwen3Transformer::createAttention`: a targeted layer gets
`lora_rank`/`lora_alpha` properties (and `clip_grad_by_norm` if set);
everything else gets `trainable=false` whenever LoRA is active. That
includes `lm_head` (never a LoRA target — see `causal_lm.cpp`) and, unless
`lora_train_norms` is set, every RMSNorm/`reshaped_rms_norm` instance
(attention norms, FFN norms, output norm, and Qwen3's per-head
`q_norm`/`k_norm`).

### Training entrypoint

`Transformer::initializeForTraining(lr, epochs)` builds the same model graph
as inference but in `ExecutionMode::TRAIN`, with the lm head replaced by the
`lm_head_loss` layer (see [Loss on every position](#loss-on-every-position-lm_head_loss)), and sets up
optimizer/batch/epoch properties; `train()`/`getTrainingStats()`/
`getValidStats()`/`setDataset()` are thin wrappers around the underlying
`NeuralNetwork` calls. `save_weight_lora()`/`load_weight_lora()` handle the
adapter file format — `load_weight_lora()` in particular loads a pretrained
*non-LoRA* checkpoint into a graph that *has* LoRA weights by diffing weight
names against a throwaway reference graph built with the exact same layer
construction calls as the real graph (so the two topologies can never
desync). Base-model weights are collected as **borrowed pointers** into that
reference graph, not clones — see [the memory bugs](#two-memory-bugs).

### Training data pipeline and CLI

`lora_train.cpp`/`.h` (`TrainingDataGenerator`) tokenizes a text file
(plain-text or chat-format, auto-detected), right-pads every sample to a
fixed sequence length, shuffles epochs with a seeded `mt19937`, and labels
every position with its next token (-1 on padding).
`train_qwen3_lora.cpp` is the `nntr_lora_train` CLI driver. Full flag list:

```
nntr_lora_train <model_dir> <train_data.txt> [options]
  --lr <float>          learning rate (default 1e-4)
  --epochs <int>        number of epochs (default 1)
  --output <path>       LoRA adapter output path (default <model_dir>/lora_adapter.bin)
  --lora_path <path>    resume from an existing LoRA adapter
  --lora_rank <int>     LoRA rank; overrides nntr_config.json
  --lora_alpha <int>    LoRA alpha; overrides nntr_config.json
  --max_samples <int>   cap the number of training samples
  --seq_len <int>       training sequence length; overrides nntr_config.json's
                        init_seq_len. Attention memory is quadratic in this,
                        so prefer the smallest length that fits your samples.
  --clip_grad <float>   clip LoRA gradients to this global norm (0 = off, default)
  --train_norms         also train the RMSNorm gammas alongside the LoRA
                        adapters (default: norms frozen)
  --seed <int>          RNG seed for epoch shuffling (default 42)
```

The adapter is saved whenever validation loss improves, so an interrupted
run still leaves a usable adapter on disk.

### Inference-side adapter loading

`main.cpp` reads an optional `lora_file_name` from `nntr_config.json`; when
present, `load_weight_lora(weight_file, lora_file)` is used instead of
`load_weight(weight_file)`, applying the trained adapter on top of the base
checkpoint before inference.

## The NaN investigation

This is the part worth reading carefully if you hit something similar in the
future: the loss went to NaN during real training, and the eventual root
cause was *not* what the first two fixes addressed. Both of those fixes were
real bugs and had to be fixed regardless, but neither was the reason for the
NaN. The actual cause was found only by bisection.

### Symptom

Training on Qwen3-0.6B produced `NaN` loss, generally within the first few
steps. The first, most decisive diagnostic used was: train with `lr=0`. At
`lr=0` no weight update happens, so a correct implementation must produce
*exactly* the same train loss and validation loss on the same data (both are
just forward passes through an unchanging model). If they don't match, the
bug is in the forward path, not backward or the optimizer. If they match but
loss is still garbage (e.g. ~15.38, far from a sane cross-entropy value),
there's a correctness bug somewhere in the read path. If loss is sane at
`lr=0` but goes NaN at `lr>0`, the bug is specifically in backward.

### Bug 1: the `thread_local` thread-visibility bug

**Symptom found via the `lr=0` test**: loss was ~15.38 instead of a sane
value — not NaN yet, just wrong.

nntrainer's `GENERATOR`-based dataset callback runs on a **separate producer
thread** from the training thread that actually runs the forward/backward
passes. The original design had `g_lm_head_read_row`/
`g_tie_embedding_lm_head_read_row` declared `thread_local` and set by the
data generator before yielding each sample, on the assumption that the value
would be visible to the layer code that reads it. It is not — `thread_local`
storage is per-thread, so the value the generator set was invisible on the
training thread, which always saw the sentinel (`UINT_MAX`, meaning
"use `height - 1`"). For right-padded samples where the last real token
isn't the last row, this meant the LM head was scoring a pad-token position
instead of the intended one: wrong logits, wrong loss, but not NaN.

**Fix**: made the globals plain (non-`thread_local`) statics, and moved the
read-row *derivation* into `TieWordEmbedding::forwarding()`'s embedding-mode
branch — scanning the input token ids for the last non-pad id, on the same
thread that will later read the value in `calcDerivative`/`calcGradient`.
This removed the need for the training driver to communicate the value
across threads at all.

This fixed the *incorrect loss* problem. It did **not** fix the NaN.

### Bug 2 (root cause): the tensor-aliasing bug

**How it was found**: rather than guessing, each layer's `calcDerivative`
was temporarily gated behind an env-var switch (`NNTR_SKIP_CD_RMSNORM`,
`NNTR_SKIP_CD_SWIGLU`, etc. — a scratch/diagnostic mechanism, not part of the
final code) so that individual layers' backward passes could be disabled one
at a time and the run re-tried. Disabling `SwiGLULayer::calcDerivative`
alone made the NaN disappear. That isolated the bug to one layer, out of six
newly-implemented ones.

**Root cause**: nntrainer aliases each layer's *outgoing derivative* (the
gradient it produces for a given input) onto the **same memory buffer** as
that input's own storage — a memory-reuse optimization in the tensor pool.
The original SwiGLU backward was written intuitively, computing `d_up`
first and writing it out, then computing `d_gate` from `up`:

```cpp
// WRONG — writes d_up (which aliases the up input's own buffer) before
// d_gate has finished reading up.
for (i) {
  du[i] = swish(gate[i]) * dy[i];       // du aliases up: this OVERWRITES up[i]
  dg[i] = swish_prime(gate[i]) * up[i] * dy[i]; // reads the now-corrupted up[i]
}
```

Because `d_up`'s output tensor is literally the same memory as `up`'s input
tensor, the first line destroys `up[i]` before the second line can read it.
The result: `d_gate` is computed from garbage, and depending on what
happened to be in that memory, this manifests as wildly wrong values that
compound into NaN within a step or two of training. This was confirmed by
printing the two tensors' data pointers and observing they were identical.

**Fix**: snapshot every value the computation still needs to read *before*
writing any output:

```cpp
// RIGHT — read gate[i]/up[i]/dy[i] into locals first, then it doesn't
// matter that writing du[i]/dg[i] aliases up's storage.
for (i) {
  float gi = gate[i], ui = up[i], dyi = dy[i];
  du[i] = swish(gi) * dyi;
  dg[i] = swish_prime(gi) * ui * dyi;
}
```

The same class of bug was proactively audited for in every other new
`calcDerivative`. It was found again in `mha_core.cpp`: `calcDerivative` was
calling `dV.setZero()` before finishing reading `V` (`dV` aliases the `V`
input), which would have caused the identical failure mode the first time
`mha_core`'s backward was exercised with real data. Fixed by `.clone()`-ing
`V` into a local snapshot before zeroing `dV`. `rms_norm`/`reshaped_rms_norm`
and `lm_head`/`tie_word_embedding` were also audited and found *not* to have
this problem (they either accumulate-then-write without an intervening read,
or never read their input in `calcDerivative` at all).

A dedicated regression test, `SwiGLUCalcDerivativeIsSafeWhenGradAliasesInput`,
was added specifically because the other gradient-check tests build each
`Var_Grad` with independently-allocated variable/gradient storage and
**never exercise the aliasing** — they would pass even with the buggy code
above. The regression test explicitly constructs both an aliased and a
non-aliased version of the same computation and asserts they agree,
reproducing the exact hazard directly rather than relying on it showing up
by chance in a broader test.

### Verification that the fix actually worked

Rather than just asserting "should be fixed now," the same `lr=0` decisive
test was rerun after both fixes: train loss and validation loss matched
*exactly* — `0.420231` both — confirming the forward path is correct and
deterministic. Training was then run at `lr>0` and the loss decreased
monotonically with no NaN, across multiple datasets and a learning-rate
sweep (see [Verification performed](#verification-performed)).

## The LoRA initialization order bug

Separately from the NaN, mainline's `fc_layer.cpp` initialized LoRA weights
as `loraA = zeros`, `loraB = random`. This does give `B·A = 0` at
initialization (so the adapter starts as a no-op, correctly reproducing the
pretrained model), which is *necessary* but not *sufficient* for a good
initialization.

**Why the original order is badly conditioned**: with `A = 0`, the only
nonzero gradient at step 0 is `dL/dA = xᵀ(dy·Bᵀ) * scaling` — note this
depends on `B`, which is *random*. So the very first update to `A` is
steered entirely by whatever random values `B` happens to have, i.e. an
essentially arbitrary direction with no relationship to the actual gradient
signal. This produces an uncontrolled large first step. (An earlier version
of this analysis mistakenly claimed `B` would never receive gradient because
`A = 0` — that's wrong: `dL/dB = (dyᵀ · xA)ᵀ`-style terms are zero only on
the very first step; once `A` moves away from zero on step 2 onward, `B`
does receive real gradient. The actual defect is the conditioning of `A`'s
first update, not "B never trains.") Empirically, this ordering diverged
above `lr ≈ 1e-6` on Qwen3-0.6B.

**Standard LoRA practice (Hu et al. 2021)** is the reverse: `A` drawn from a
zero-mean distribution (`LECUN_NORMAL` here), `B = zeros`. This still gives
`B·A = 0` at step 0, but now the first gradient that matters,
`dL/dA = xᵀ(dy·Bᵀ)`, is zero (since `B = 0`), while `dL/dB = (something
involving the real, non-random A)` is well-defined and driven by actual
signal. `A`'s random values only ever get *scaled* by gradients that are
themselves shaped by real data, never used as the sole source of direction
for an update. This ordering was stable at `lr = 1e-4` on the same model.

Both the crash fix and this init-order swap live in `nntrainer/layers/fc_layer.cpp`,
each as their own commit.

## Was it gradient clipping?

A natural question once NaN loss shows up: is this actually just an
exploding (very large but finite) gradient, and would gradient clipping have
masked or fixed it? The answer here is **no**, and the reasoning matters:

Gradient clipping (`clip_grad_by_norm`, computed globally across all flagged
weights in `NetworkGraph`/`layer_node.cpp`) rescales a gradient whose norm is
large but *finite*. It cannot help with a gradient that is already `NaN`,
because `‖g‖` of a `NaN` vector is itself `NaN` — there's no finite norm to
clip against. The decisive piece of evidence: the NaN was reproduced even at
`lr = 0`. At `lr = 0`, no weight update happens at all, so an exploding
(but finite) gradient literally cannot be the cause — the forward pass
itself was already producing corrupted numbers (from the aliasing bug),
independent of what the optimizer does with the resulting gradient.

Gradient clipping was still added (`--clip_grad` / `lora_clip_grad_by_norm`,
threaded into the LoRA-targeted layers' `clip_grad_by_norm` property) as a
genuinely useful, orthogonal feature for training stability at higher
learning rates or noisier data — just not as the fix for this specific bug.

## Two memory bugs

Neither of these caused incorrect results; both were found while checking
resource usage on a memory-constrained target and would have been fatal
there (though not necessarily on a workstation with abundant RAM).

**`mha_core` training cache oversized** (~7 GB-scale at the sequence lengths
tested): the training-mode cache tensors (`train_q_roped`, `train_k_roped`,
`train_attn_wt`) were originally sized off `max_timestep` — the maximum
*decode* horizon used for incremental inference — rather than the actual
training sequence length (`query_dim.height()`). Training never does
incremental decoding, so this was allocating buffers sized for a much longer
sequence than any training batch actually needs. Fixed by sizing these
tensors from `query_dim.height()`. Additionally, the KV cache itself
(`cache_key`/`cache_value`) is now only allocated when `!for_training` (a new
`internal_cache_requested` member gates this in `setBatch()` and
`updateTensorsByInputDimensions()`), since training never uses incremental
decode caching at all.

**`load_weight_lora` cloning the entire base model** (~2 GB-scale): the
original implementation collected every weight of a throwaway reference
`base_model` graph into an `unordered_map<string, Tensor>` by **value**
(i.e. cloning each tensor), so that peak memory during a checkpoint load was
roughly `base_model + the clone map + this` — nearly triple the model size.
Fixed by storing `const Tensor *` (borrowed pointers into `base_model`)
instead; this is safe because `base_model` is guaranteed to outlive the copy
loop within the same function scope. Verified the save/load round-trip test
still passes identically after the change.

## A silent-wrong-answer bug in validation

`mha_core::forwarding()` gated its real attention computation on the
`training` boolean flag rather than on whether the training-mode tensors had
actually been allocated (`train_tensors_requested`). Validation is called
with `training=false` (it's not a training step), so this caused validation
to **skip attention entirely** and score whatever was left in a stale output
buffer — producing a loss (~13.5) that happened to look plausible (close to
a random-guess baseline for the vocab size) but was not actually measuring
the model. This is the more dangerous kind of bug: it doesn't crash and
doesn't produce an obviously-wrong number, it just silently measures the
wrong thing. Caught by the same `lr=0` train-vs-validation-loss-must-match
test: at `lr=0` the two forward passes should be identical, and they weren't,
which is what motivated looking at what differs between a "training-mode"
and "validation-mode" call in the first place. Fixed by gating on
`train_tensors_requested` instead of `training`.

## Verification performed

- **13 unit tests** (`unittest_lora_backward_gradcheck`): finite-difference
  gradient checks (central difference, `L(x) := dot(forward(x), dy)`, so the
  exact gradient of `L` w.r.t. `x` is what a correct backward pass should
  produce) for every `calcDerivative`/`calcGradient` added, plus the
  dedicated aliasing regression test and the mha_core
  training-forward-vs-inference-prefill consistency check.
- **3 end-to-end tests** (`unittest_qwen3_lora_training_smoke`): a tiny
  synthetic Qwen3 graph compiles for training with LoRA enabled, the
  resulting trainable/frozen flags match the freeze policy exactly, and a
  few training steps followed by `save_weight_lora`/`load_weight_lora`
  round-trip correctly.
- **No regressions**: the full pre-existing suite (67 CausalLM model tests,
  17 KV-cache tests, 10 embedding-sidecar tests, and everything else in
  nntrainer's unit test suite — 38 test suites in total) continues to pass.
- **Real-model validation on Qwen3-0.6B**:
  - The `lr=0` train-loss-equals-validation-loss test (`0.420231` both,
    exact match) as the decisive correctness check described above.
  - Loss decreases monotonically at `lr > 0` on identity, SST-2, and LaMP-3
    style data, with no NaN.
  - Saved adapters scanned byte-for-byte for NaN/Inf across every parameter
    — clean.
  - A full train → save → load → generate round trip produces coherent,
    adapter-influenced output.
  - A learning-rate sweep confirms the expected qualitative behavior, not
    just "doesn't crash": a very small LR leaves generation ≈ unchanged from
    the base model; `1e-4` visibly picks up trained vocabulary/style while
    staying fluent; `1e-3` overfits a small (~20-sample) dataset into
    repetitive output — the textbook LoRA overfitting failure mode at too
    high an LR, not a bug.
  - Peak RSS was measured during a training run (`~8 GB`) to sanity-check
    the memory-bug fixes above actually brought usage into a reasonable
    range.

## How to train

```
nntr_lora_train <model_dir> <train_data.txt> \
  --lr 1e-4 --epochs 3 --lora_rank 8 --lora_alpha 16 \
  --output <model_dir>/lora_adapter.bin
```

`<model_dir>` must contain the same `nntr_config.json`/checkpoint layout used
for inference; `lora_rank`/`lora_alpha`/`lora_target` can be set there or
overridden on the command line. To also train the RMSNorm gammas alongside
the adapters, pass `--train_norms` (or set `lora_train_norms: true` in the
config). To use the resulting adapter at inference time, set
`lora_file_name` to the saved adapter's filename in `nntr_config.json` and
run `nntr_causallm` as usual — `main.cpp` will call `load_weight_lora`
instead of `load_weight` automatically.

Each epoch ends with a line such as
`[epoch 1] train_loss=4.98 valid_loss=3.02 time=8.31s (train 5.84s + valid 2.48s)`.
`time` is the epoch's wall time, from its first iteration to the end of
validation. The adapter save that follows is not included. The same two
numbers reach `trainLora()`'s `TrainProgressCallback` (`epoch_seconds`,
`train_seconds`) and the app's Training Log.

## Training with the OpenCL device (Android / Adreno)

The training graph is built from host layers: their `calcDerivative` /
`calcGradient` read and write the tensors directly, and the GPU variants
(`FullyConnectedLayerCl`, `rms_norm_gpu`, ...) have neither LoRA properties nor
a backward. So in an OpenCL build the training graph is always built with
`engine=cpu` (`Transformer::layerEngine()`), even though inference defaults to
`engine=gpu`. Before this, an OpenCL build died at graph construction with
`[LayerImpl] Unknown Layer Properties count 7` (the LoRA/QAT properties handed
to `FullyConnectedLayerCl`).

GPU work in training is requested per layer instead: with `NNTR_TRAIN_GPU=1`
(needs an OpenCL build and a live `gpu` context, i.e. `NNTR_ENGINE` unset or
`gpu`), every LoRA fully connected layer gets `lora_gpu=true` and sends its
large FP32 GEMMs through `ComputeOps::try_sgemm_fp32_host`
(`ClComputeOps` -> `sgemm_train_cl`, a register-tiled 64x64 kernel). The layer
stays a host layer, its tensors are ordinary host memory, and a product the
device declines runs on the CPU table exactly as before.

| env | meaning |
|---|---|
| `NNTR_TRAIN_GPU=1` | enable the offload (default off) |
| `NNTR_TRAIN_GPU_MIN_WORK` | smallest `M*N*K` sent to the device (default `2^26`: only the frozen-weight input-derivative GEMMs; the rank-32 adapter matmuls are ~10x slower on the device than on the CPU table) |
| `NNTR_TRAIN_GPU_CHECK=1` | also run every offloaded GEMM on the CPU and log the relative error and both timings; `=2` additionally copies the CPU result back (isolates the effect of the GPU values) |
| `NNTR_TRAIN_GPU_PROF=1` | log stage / dispatch / wait+read time per device GEMM |
| `NNTR_TRAIN_BASE_GPU=1` | route the frozen QS4CX base-weight **forward** GEMM of every host FC through the GPU v8c int8 x int4 kernel (the GPU trunk's kernel; default off, needs a live `gpu` context). While active, the payload-release levers `NNTR_V8C_DROP_PLAIN` and `NNTR_TRAIN_DROP_PLAIN` are forced off: a training step still reads those pages after the first offloaded forward (step 1's dX transposed copy is built from them), and the released pages read as zeros -- the loss then pins at the degenerate constant from step 2 on, with no crash and no mismatch in the forward output itself. |
| `NNTR_TRAIN_BASE_GPU_CHECK=0` | disable the self-check latch (default on): the first 8 offloaded GEMMs are cross-checked against the host dot and, on a mismatch -- or when the reference itself reads a released payload -- the GPU base path latches off for the process with a log line. `NNTR_TRAIN_BASE_GPU_CHECK=N` (N>8) checks N calls instead. |

The base-weight switch is opt-in because, measured on SM-S948N (Qwen3-0.6B
QS4CX, rank 8, seq 128), it does not pay off for training today: the round
trip (FP32 activation upload, fp16 result readback) makes
`fc.base_fwd.lora` roughly 5x slower than the host i8mm kernel (98 vs
19 ms per iteration at plateau) and the epoch roughly 2x slower end to end
(59-67 s vs 22-31 s over 90 samples) -- in inference the kernel wins because
the activations already live on the device. With the payload levers forced
off (see above) the loss trajectory matches the host path within the int8
activation quantisation noise (final train loss 1.96493 vs 1.96309).

Measured on SM-S918N (Adreno 740), Qwen3-0.6B Q4_0, rank 32, QAT, seq 192,
12 samples, 1 epoch:

- Correctness: worst per-GEMM relative error 1.9e-6 over every offloaded call.
  The first Adam step is sensitive to that (valid loss after one step differs
  between CPU and device runs), but over 12 samples the device runs
  (valid 2.56 / 2.40) sit inside the spread of CPU runs that differ only by
  `--seed` (2.33 - 2.50).
- Speed: 97 s with the offload vs 75 s on the CPU. A frozen-weight
  `dX` GEMM (192x1024x3072) takes ~11 ms on the device (staging ~1 ms, kernel
  ~10 ms, ~100 GFLOP/s) vs 10-15 ms on the CPU table, so on this device the
  offload is roughly break-even per GEMM; the end-to-end loss is the extra
  host/device synchronisation. The GPU is busy ~30% of the time in the
  backward passes.

What this does not do yet: the attention forward/backward (the largest share of
a step), the QAT fake-quantisation, the adapter matmuls and every activation
still run on the host; nothing stays resident on the device between ops.
Making the device pay off needs those moved over (device-resident frozen
weights and activations, a device attention backward, an on-device
fake-quant kernel) and the GEMM kernel tuned on the target GPU.

## Tuning only the last N layers (`--lora_last_n`)

`--lora_last_n N` (nntr_config key `lora_last_n_layers`) puts LoRA only on the
last N decoder layers. Layers before that have nothing trainable, and
`NetworkGraph::markNodesForBackwarding()` only marks the nodes at and after the
first trainable one for backward, so their `calcDerivative` never runs (their
forward still does). Measured on SM-S918N, Qwen3-0.6B Q4_0, rank 32, QAT,
seq 192, CPU engine, one epoch:

| `--lora_last_n` | per (train+valid) sample, 4 -> 12 samples | 90 samples, 1 epoch | train / valid loss |
|---|---|---|---|
| all (28) | 5.5 s | 539 s | 2.174 / 2.044 |
| 14 | - | 449 s (1.2x) | 2.468 / 2.085 |
| 7 | 2.4 s (2.3x) | 339 s (1.6x) | 2.301 / 2.192 |
| 3 | - | 275 s (2.0x) | 2.035 / 1.750 |

The 90-sample runs ran back to back on a warming device, so their ratios are
the conservative ones; the short runs were measured cool. The losses are one
epoch, one seed, and the valid set is the training set, so they show the
smaller adapters train without breaking, not that they match full-depth
quality; that needs several epochs and a held-out split.

## Fast on-device training: int8 backward and the GPU-trunk / CPU-head split

Measured on SM-S948N (Snapdragon 8 Elite Gen 5, Adreno 840), Qwen3-0.6B,
seq 256, rank 8, `--lora_last_n 7`, QAT, app-default environment
(OpenBLAS 1 thread). "iteration" is one train step (forward + backward +
update), averaged over the first iterations of a run unless noted:

| step | iteration |
|---|---|
| Q4_0 base, before | 1538 ms (50-iter avg; 1.0-1.3 s cool) |
| int8 dX + LoRA / SwiGLU / attention kernels (Q4_0, CPU) | 614 ms |
| QS4CX base, CPU only | ~395 ms |
| QS4CX base, GPU trunk + CPU head (`--hybrid_trunk`) | **175 ms** (165 ms min, ~300 ms thermally throttled) |

Where the time went, and what replaced it:

- **Forward was already integer.** Q4_0 / QS4CX FC weights go straight to
  the ggml Q4_0 x Q8_0 / KleidiAI qai8dxp x qsi4cxp i8mm GEMMs; nothing is
  dequantized on the forward path.
- **Backward dX** (`dY . W^T` against the frozen base) was a per-call FP32
  dequantize plus an FP32 GEMM on single-threaded OpenBLAS: 627 ms of a
  1.5 s step. It is now an int8 GEMM: W re-quantized once to Q8_0 along N
  (the dX reduction axis) for Q4_0 bases, or, for QS4CX, the same int4 codes
  transposed with unit scales and the per-channel scale folded into dY
  (lossless) on the KleidiAI GEMM. rel L2 vs FP32: ~1% (Q8_0) / ~2.7%
  (KleidiAI, per-row activation quantization); 1-epoch valid loss unchanged.
- **LoRA rank-r GEMMs, SwiGLU, attention**: multithreaded skinny kernels,
  row-block parallel SwiGLU (bit-identical), 4x4 NEON-tiled attention with
  race-free parallel backward passes.
- **Frozen trunk on the GPU**: with LoRA only on the last N layers, the first
  `L - N` blocks are an inference prefill. `--hybrid_trunk <dir>` runs them on
  the GPU (QS4CX-FP16 inference graph, `setTrunkLayers()`), on the dataset
  producer thread, one sample ahead of the CPU step; the training graph
  (`setHeadFromLayer()`) takes the trunk's hidden state as input0. GPU
  ~40-130 ms per sample, fully hidden under the CPU step.

Do not raise the OpenBLAS thread count (`NNTR_NUM_THREADS`) for this
workload: with the big GEMMs off BLAS, BLAS threads only contend with the
nntrainer thread pool (324 ms vs 175 ms hybrid, 828 vs 614 ms CPU).

### Preparing the two models

The head (training) model is QS4CX FC + FP32 embedding with FP32 norms; the
trunk model is the QS4CX-FP16 inference model. Both come from the same FP32
weights, so their QS4CX codes match:

```bash
# host: HF -> nntrainer FP32 (FP32 norms) and FP32 / FP16-norm stages
python3 res/qwen3/qwen3-0.6b/weight_converter_layergraph.py \
  --model_path <hf Qwen3-0.6B> --output_name stage32/nntr_qwen3_0.6b_fp32.bin \
  --data_type float32                       # nntr_config: FP32-FP32
python3 res/qwen3/qwen3-0.6b/weight_converter_layergraph.py \
  --model_path <hf Qwen3-0.6B> --output_name stage16/nntr_qwen3_0.6b_mixed.bin \
  --data_type float32 --norm_fp16           # nntr_config: FP32-FP16
# device (NNTR_ENGINE=cpu): quantize
nntr_quantize stage32 --fc_dtype QS4CX --embd_dtype FP32 --lmhead_dtype FP32 \
  --output /sdcard/q_qs4                    # head: QS4CX-FP32
nntr_quantize stage16 --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype Q6_K \
  --output /sdcard/q_qs4_gpu                # trunk: QS4CX-FP16
```

### Running it

```bash
nntr_lora_train /sdcard/q_qs4 /sdcard/t.txt --lr 5e-5 --epochs 5 \
  --lora_rank 8 --lora_alpha 16 --lora_last_n 7 --seq_len 256 \
  --max_samples 50 --clip_grad 1.0 --lora_qat --lora_weight_q4 \
  --hybrid_trunk /sdcard/q_qs4_gpu
```

SampleTestAPP: the Train tab's "GPU trunk + CPU head" switch (on by default)
and "GPU trunk model" path map to the `hybrid_trunk` field of the
`trainLora()` params JSON; the defaults are `/sdcard/q_qs4` and
`/sdcard/q_qs4_gpu`. The status line shows ms/iter.

The hybrid mode sets `NNTR_MHA_GPU=1`, `NNTR_KV_IMG_ATTN=1` and
`NNTR_V8C_DROP_PLAIN=0` unless already set (with the trunk's host payload
dropped, the first trunk prefill trips the v8c weight-cache rebuild guard),
and gives the dataset `buffer_size=2` (with one slot the slot is freed only
when the consuming step ends, which serializes trunk and step). Peak RSS
2.2 GB (1.78 GB CPU-only).

### Diagnostics

| env | effect |
|---|---|
| `NNTR_TRAIN_PROF=<first LoRA layer>` | per-layer fwd / deriv / grad averages every 10 iterations, `[PERF]` step / wait lines, `[PERF-MHA]`, `[PERF-TRUNK]` |
| `NNTR_TRAIN_DX_CHECK=1` | dX error vs the FP32 reference (first 64 calls) |
| `NNTR_LORA_SKINNY_CHECK=1` | LoRA skinny kernels vs the BLAS GEMMs |
| `NNTR_TRAIN_DX_Q8=0/1`, `NNTR_TRAIN_DX_KAI=0` | select the dX path (FP32 / Q8_0 / KleidiAI) |
| `NNTR_TRAIN_FWD_FP32=1` | QS4CX base FCs train-forward as FP32 x dequantized W instead of the KleidiAI int8 kernel |
| `NNTR_TRAIN_GRAD_DUMP=<dir>` (`NNTR_TRAIN_GRAD_DUMP_ITERS`, default 3) | dump LoRA A / B / dA / dB, sample ids and the trunk hidden state for `tools/lora_gradcheck` (see below) |

### Caveats

- **QS4CX quality.** Per-output-channel int4 without K groups is coarse for
  a 0.6B model: simulated on the HF weights (1k-token ppl), FP32 37.4,
  int8 per-channel 37.8, Q4_0 39.8, QS4CX 63.0 (56.9 with an MSE-clipped
  scale); the on-device QS4CX model answers "The capital of France is" with
  "a square", Q4_0 with "Paris". Qwen3-1.7B is less affected (group-wise
  int4 recovers) but per-channel int4 still fails that prompt. The speed
  path assumes a base re-trained / re-quantized for this format.
- QS4CX on the GPU engine only: Q4_0 GPU GEMM is disabled on Android
  (the CPU repacks Q4_0 to q4_0x4, the device kernel unpacks q4_0x8), and
  the FP16-activation Q4_0 GPU entry is not implemented.
- `--hybrid_trunk` needs `--lora_last_n` < number of layers.

## Batch training (`--batch_size`)

`--batch_size N` (app: BATCH SIZE, API JSON: `batch_size`) compiles the
training graph at batch N. The loss is the mean over all scored tokens of
the batch, so one step's gradient is the token-weighted mean of the N
per-sample gradients. The data generator still
writes one sample per call: nntrainer's iteration queue hands it one slot at
a time and builds the batch itself.

Two things had to be fixed before a batch above 1 trained correctly:

- **Quantized FCs computed only the first batch item.** `FloatTensor::dotQnK`
  (Q4_0 / Q4_K / Q6_K), `dotQInteger` and `dotQs4cx`, and the matching
  `HalfTensor` arms, took the GEMM row count from `height()` instead of
  `batch * channel * height` as `dotFloat` does. Items 1..N-1 kept whatever
  their output buffer already held. With lr = 0 a batch of 2 scored up to
  0.05 nats off the mean of its two samples. The QS4CX KAI dX goes through
  the same `dotQs4cx`, so the backward pass was affected too.
- **Hybrid read rows went out of step.** nntrainer skips a trailing
  iteration it cannot fill. In hybrid mode each produced sample has already
  queued its lm_head read row, and nobody pops the rows of a skipped
  iteration, so every later sample read its neighbour's row. The generator
  now ends each epoch after `floor(samples / N) * N` samples. The per-epoch
  reshuffle changes which samples sit out. (The read-row queue itself went
  away with the every-position loss below; the whole-batch epoch stays.)

Checked with lr = 0 (weights fixed), batch 2 against batch 1 on the same
samples, on SM-S948N with `/sdcard/q_qs4` (QS4CX, rank 8, last 7 layers,
QAT, seq 128):

| run | loss | dA / dB (49 LoRA FCs, `NNTR_TRAIN_GRAD_DUMP`) |
|-----|------|------------------------------------------------|
| CPU, 8 samples | every batch-2 step = mean of its two batch-1 steps (6 digits) | worst rel 5.9e-7 |
| hybrid, 7 samples, 2 epochs | same, including epoch 2 after a skipped sample | worst rel 7.0e-7 |

In the app, batch 4 for 2 epochs on 50 samples ran 12 steps per epoch, at
about 420 ms per step (hybrid). An epoch took 8.3 to 9.6 s, of which 5.8 to
6.7 s was training and the rest validation. Valid loss went 3.03 then 1.57. The first
step's loss matched the mean of the same four samples run at batch 1.

The lr schedule (`--lr_decay_steps`) and the QAT EMA count steps, not
samples. At batch N an epoch is N times fewer steps, so scale
`lr_decay_steps` down to keep the same decay per epoch. On the CPU a batch
does not make training faster per sample. Activation memory grows with N,
for example the attention weights take `N * heads * seq^2 * 4` bytes per
layer.

## Loss on every position (`lm_head_loss`)

Training used to score only the last token of a sample: input `ids[:-1]`,
one label `ids[-1]`, and the lm head projected a single "read row" per batch
item. A sample now trains on every position, like any causal LM: input
`ids[:-1]` right-padded to `seq_len`, label `ids[1:]` (the next token of each
position), `-1` on the padding. The loss is the mean of
`-log softmax(h W^T)[target]` over the scored positions of the batch, which is
what PyTorch's `cross_entropy(logits, ids[1:])` computes for one sample.

At vocab 151936 the logits of one sample are 78 MB at `seq_len` 128, so the
head and the loss are one layer, `lm_head_loss`
(`layers/lm_head_loss_layer.cpp`), and that tensor never exists.
`forwarding()` takes the scored rows 32 at a time: logits for the chunk (FP32
GEMM split over the vocabulary across the compute threads), log-softmax and
loss, and in training `dL/dlogits`, multiplied straight back through `W`
into `dL/dx` (GEMM split over hidden). `calcDerivative()` hands that `dL/dx`
on. `W` is the tied embedding (`shared_from` embedding0; in a hybrid head
graph the head's own copy) and stays frozen, so there is no `dW`. The label
carries which positions count, so the read-row globals and the hybrid
read-row queue are no longer used for training. Tied embeddings only.

Checked:

- `unittest_lora_backward_gradcheck`
  `LmHeadLossMatchesReferenceAndFiniteDifference`: 2 x 24 positions with
  padding and a masked middle position, vocab 4101 (crosses both the 32-row
  chunk and the 4096-column split). Loss and per-position loss match a direct
  double-precision computation to 1e-4, and `dL/dx` matches central
  differences for every input element.
- Against PyTorch on SM-S948N (`/sdcard/q_qs4`, plain-text lines, FP32
  forward and dX, `tools/lora_gradcheck`): the step loss matches the
  reference to 4e-4 relative (5.54168 / 5.54213, 4.48337 / 4.48360, 5.25659 /
  5.25435). dA / dB of the 49 LoRA FCs are within 1% of the reference for most
  layers, 2.6% at worst. `check_update.py` finds 0 / 49 wrong updates.
- Batch 2 against batch 1 with lr = 0: every batch-2 step loss equals the
  token-weighted mean of its two batch-1 steps (CPU and hybrid). The gradients
  agree to 2.8e-6 with FP32 dX. With the default KAI dX they agree to 1e-7
  when the two samples have the same length, but only to a few percent when
  they do not. KAI quantizes each `dY` row to int8. The batch rescales `dY`
  from 1/n to 1/(n1 + n2), and when that is not a power of two, values that sit
  on a rounding boundary flip, so the 1 to 5% quantization error of KAI dX
  (`NNTR_TRAIN_DX_CHECK`) comes out differently. The GEMM itself gives
  bit-identical rows batched or not.

Cost: the head now runs for every scored position, not one per sample. On
SM-S948N, hybrid, batch 4, plain-text lines of 10 to 20 tokens,
`lm_head_loss` takes 260 to 300 ms of a 550 to 680 ms step
(`NNTR_TRAIN_PROF`). In the app, 2 epochs on 50 such lines took 11.2 s and
12.1 s per epoch (7.4 s / 8.0 s training). Valid loss went 4.61 then 3.87.

## Decision models (RLCD): `nntr_decide`, `--task decision`

[anthonym21/qwen3-0.6b-rlcd-decision](https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision)
is a Qwen3-0.6B body that answers questions instead of generating text.
It reads a context and a question with 2 to 26 options and returns a
probability per option: the softmax over the logits of the letter tokens
` A`..` Z` at the last prompt token, cut to the first K letters. The prompt is
fixed (`decision.h`):

```
User: Context:
<context>

Question: <question>
Options:
A) <option>
B) <option>
Answer with the letter only.
Assistant: The answer is
```

The model ties its embeddings, and its `decision_head.safetensors` is exactly
the 26 letter rows of the input embedding (max |diff| 0). So a converted
model reads its decision head from its own tied lm head, and the body is a
plain Qwen3 graph.

### Converting and quantizing

```
# host: HF -> nntrainer FP32 (the checkpoint is a bare Qwen3Model)
python3 res/qwen3/qwen3-0.6b/weight_converter.py --model_path <hf dir> \
  --output_name <out>/nntr_qwen3_0.6b_rlcd_fp32.bin
# <out>: config.json with "architectures": ["Qwen3ForCausalLM"] (the
# converted graph has the tied head), tokenizer.json, decision.json, a
# generation_config.json with bos/eos 151643 (the repo ships none), and an
# FP32 nntr_config.json (init_seq_len 512)
# device: nntr_quantize (NNTR_ENGINE=cpu)
nntr_quantize <out> --fc_dtype QS4CX -o /sdcard/q_rlcd_qs4        # CPU head
nntr_quantize <stage FP32-FP16> --fc_dtype QS4CX --embd_dtype Q6_K \
  --lmhead_dtype Q6_K -o /sdcard/q_rlcd_qs4_gpu                    # GPU trunk
```

The FP32-FP16 stage for the trunk comes from
`weight_converter_layergraph.py --norm_fp16`, which now also reads the bare
`Qwen3Model` tensor names. Copy `decision.json` next to each quantized model.

### Inference

`nntr_decide <model_dir> <questions.jsonl>` runs one prefill per question
(`CausalLM::promptLogits()`) and prints value, probs, confidence,
entropy_confidence, plus score for score questions or p_true for noul
questions. A question is an eve-rlcd JSONL row:
`{"primitive": "choice" | "score" | "noul", "context", "question",
"choices", "answer"}`. The whole prompt is cut from the left to the token
budget, as the model's training did. eve-rlcd's `Decider` cuts the context
and the question separately; the two agree whenever the prompt fits.

Checked against PyTorch (`tools/decision/ref_decide.py`, which reproduces the
model card: INFRASTRUCTURE 0.86970, P0_CRITICAL 0.99255, p_true 0.88949) on
the card's three questions and 60 synthetic triage tickets, on SM-S948N, CPU:

| model | argmax = reference | max abs prob diff | mean TV distance | ms / question |
|-------|--------------------|-------------------|------------------|---------------|
| FP32 | 63 / 63 | 0.0024 | 0.0001 | 760 |
| QS4CX-FP32 | 60 / 63 | 0.58 | 0.082 | 127 |
| Q4_0-FP32 | 59 / 63 | 0.67 | 0.047 | 184 |

The 4-bit models agree with the reference on most answers but move the
probabilities, which is what this model is about (calibration). For
calibrated output use FP32, or measure on your own data first.

### Training

`nntr_lora_train <model> <questions.jsonl> --task decision --objective
sft|soft|rlcd|rlvr [--group 4]` (app: TASK / OBJECTIVE; `trainLora()` JSON:
`task`, `objective`, `group`). The training graph ends in
`decision_head_loss` (`layers/decision_head_loss_layer.cpp`) instead of
`lm_head_loss`. The layer reads the hidden state at the last prompt token,
multiplies it by the 26 letter rows of the tied embedding, and takes a
softmax over the question's K letters. Its label carries, per letter, 1 for
the answer, 0 for another option and -1 beyond K, plus the readout row. The
objectives follow eve-rlcd:

- `sft`: `-log p(answer)`.
- `soft`: the label's first K entries carry a target distribution (summing
  to 1) instead of a one-hot; the loss is `-sum_i t_i log p_i` and the
  gradient `p - t` (sft is its one-hot special case). When the questions
  file's first line carries an `"input_ids"` array (the personalization
  package's pre-tokenized `device_io` JSONL with `target_probs`), the rows
  are consumed verbatim -- no re-tokenization, so the package's tokenizer
  cannot drift from the device model's.
- `rlcd`: per question, draw `group` actions from p and reward each with
  `1[correct] - p(action)`. The advantage is the reward minus the
  leave-one-out mean of the group's rewards, and the loss is
  `-mean(advantage * log p(action))`. The label only supplies the
  environment's outcome. In expectation this is half the Brier-score
  gradient, which is what keeps the model calibrated.
- `rlvr`: the same with reward `1[correct]`.

The reported loss is always the mean `-log p(answer)`, so the curves of the
objectives read the same. Only the head's 26 rows are touched, so the head
costs next to nothing, unlike `lm_head_loss`.

A separate validation file selects the checkpoint instead of the training
loss: `--valid_data <file>` (app: VALIDATION DATA; JSON `valid_data`) runs
a validation pass after every epoch and keeps the adapter with the best
validation loss (implies `--validate`). `--patience <n>` (app: PATIENCE;
JSON `patience`) stops the run early after n epochs in a row without a
validation improvement.

Checked:

- `unittest_lora_backward_gradcheck`: sft probabilities, loss and `dL/dx`
  against a direct computation and central differences. The soft objective
  gets its own test: the target-distribution loss `-sum t log p` and the
  gradient `(p - t) / B . W_letters` against central differences. For
  rlcd / rlvr, the
  mean `dL/dx` over 40000 sampled forwards matches the exact expected
  REINFORCE gradient `-sum_a p_a r(a) (e_a - p)` to 3%.
- x86, FP32 model, lr 0: the epoch loss 1.21058 equals the PyTorch mean
  `-log p(answer)` 1.21058.
- SM-S948N, `/sdcard/q_rlcd_qs4`, sft with FP32 forward / dX against
  `ref_grad.py --decision`: step loss 4.3996 / 4.39953, and dA / dB of all 49
  LoRA FCs within 0.07%.
- 60 synthetic triage questions, batch 4, 3 epochs, lr 5e-5, CPU: valid
  `-log p(answer)` goes 0.699 / 0.652 / 0.592 with sft and 0.723 / 0.721 /
  0.705 with rlcd. In the app (hybrid, `/sdcard/q_rlcd_qs4_gpu` as the
  trunk, rlcd, QAT), the same run goes 0.723 / 0.717 / 0.700 at 7.5 to 9.4 s
  per epoch.
- The trained adapters, loaded by `nntr_decide` through `lora_file_name`
  (it loads an adapter the way `nntr_causallm` does), on the same 60
  questions:

  | adapter | accuracy | -log p(answer) | Brier |
  |---------|----------|----------------|-------|
  | none | 0.767 | 0.720 | 0.363 |
  | rlcd, app (hybrid) | 0.783 | 0.710 | 0.327 |
  | sft, CLI | 0.783 | 0.583 | 0.331 |
  | rlcd, CLI | 0.800 | 0.699 | 0.323 |

  Each objective improves the score it optimizes: sft the log-loss, rlcd
  the Brier score. (A QAT run's `_q4` adapter loads through
  `lora_q4_file_name` too since the qint4 adapter format below.)

## Two silent training bugs: clipped gradients sharing memory, frozen inputs overwritten

Found while chasing a "QS4CX LoRA training collapses" report on SM-S948N
(Qwen3-0.6B, LoRA rank 8 on the last 7 layers, seq 256, `--clip_grad 1.0`).
Both bugs are in the framework, not in a kernel or a quantization format, and
both silently corrupt training without NaN, crash or warning.

### Symptom

50 samples, 6 epochs, lr 5e-5 decayed x0.7 every 50 steps, hybrid trunk:
train loss 2.28 / 1.88 / 2.31 / 2.42 / 3.46 / 4.71. By the last epoch a third
of the samples sat below 0.1 and the rest above 4: the model collapsed onto
a few answers. Lower lr, decay, QAT on or off and FP32 forward only moved
the onset. A Q4_0 base looked healthy (2.15 / 1.87 / 1.70 / 1.68 / 1.66 /
1.66), but 1.66 is roughly the entropy of the label distribution (1.47 for
these 50 samples): it was not learning from the input either, just failing
more quietly.

### Method: compare numbers against a reference, not loss curves

1. A device run with `NNTR_TRAIN_GRAD_DUMP=<dir>` writes, for the first
   steps, every LoRA layer's pre-update A / B and its dA / dB, the sample
   ids, and (hybrid) the trunk's hidden state.
2. `tools/lora_gradcheck/ref_grad.py` rebuilds those steps in PyTorch on the
   HF checkpoint (QS4CX quantize-dequantize applied to the base FCs, the
   dumped A / B patched in, the dumped trunk hidden state injected into the
   first LoRA layer) and compares the gradients. With
   `NNTR_TRAIN_FWD_FP32=1 NNTR_TRAIN_DX_Q8=0` the device computes the same
   FP32 math, so the match must be exact up to float rounding.
3. `tools/lora_gradcheck/check_update.py` checks that the weight change
   between dumped steps is the Adam + global-norm-clip update of the dumped
   gradient.

Starting with only the last layer trainable narrowed things fast. Suspects
that were measured and ruled out:

| suspect | measurement | verdict |
|---|---|---|
| KleidiAI QS4CX forward (per-row int8 activations) | 1.6-5.7% rel L2 per FC vs FP32 (activation rows have max/rms 15-28); same-weights loss 6.76 vs 5.12 | real noise, but the collapse persists with `NNTR_TRAIN_FWD_FP32=1` |
| KleidiAI dX | 1-6% gradient rel L2 vs exact | direction kept; not the cause |
| GPU trunk | same sample recomputed: rel L2 0; lm_head read-row FIFO aligned | deterministic |
| frozen base weights | hashed every forward | never modified |

### Bug A: frozen layers' inputs released before their `calcDerivative`

`Manager::requestInputs()` gave the input of every non-trainable layer
`FORWARD_FUNC_LIFESPAN`, assuming a frozen layer does not need its input
after forward. With LoRA every norm (`attention_norm`, `ffn_norm`, `q_norm`,
`k_norm`, `output_norm`) is `trainable=false`, yet an RMS norm's
`calcDerivative` reads `x`. The planner handed that memory to later tensors;
by backward time the input had been overwritten (L2 of the input at forward
vs at backward: `ffn_norm` 11913 vs 0.15, `q_norm` 9664 vs 0.18, `k_norm`
7657 vs 1464). The `ITERATION_LIFESPAN` `inv_rms` cache (see "Backward math
for six layers") protected only `inv_rms`; `x` was still read from the
reused buffer. Effect on the last layer's LoRA gradients: `wk` dB 31-41%
rel L2 (cosine 0.92), `wq` 7-23%, `wv` 5-7%; the FFN side was fine because
`swiglu` carries no `trainable=false` and kept its inputs.

**Fix** (`nntrainer/tensor/manager.cpp`): in training, a non-trainable
layer's input gets `FORWARD_DERIV_LIFESPAN`, i.e. it lives until that
layer's own `calcDerivative`. For a frozen layer that is never backwarded
(the trunk below the first LoRA layer) that order lies past the backward
end, which `TensorPool::finalize()` ignores, so those inputs still die after
forward: peak RSS 2165 MB hybrid / 1741 MB CPU-only, unchanged. A frozen
layer inside the backward range that does not need its input (e.g. a frozen
FC) now keeps it a little longer; the exe-order goldens
(`fc3_mse_nt{1,2,3}`) gain exactly that order.

### Bug B: gradients clipped by global norm shared memory after a re-plan

With `clip_grad_by_norm > 0` a gradient is applied only after the whole
backward pass (the global norm needs all of them), so the manager appends
`TensorPool::PERSIST_END_ORDER` to its exec order. `TensorPool::finalize()`
replaced that entry with the concrete end order. The pool is planned again
whenever its tensors are deallocated and reallocated (here that had already
happened before the first training step), and the next plan saw a plain
order past the end, dropped it, and ended the gradient's life at its own
layer's `applyGradient`. The V1 planner then
reused same-sized gradient buffers across layers: 55 overlapping pairs among
the 98 LoRA gradients, for example `layer27_ffn_up:loraA` and
`layer27_ffn_gate:loraA` at the same address. The backward of the next FC
sharing an input overwrote the previous one's gradient (`wk` -> `wv`, `wq` ->
`wk`/`wv`, `ffn_gate` -> `ffn_up`, `ffn_up` -> `layer27_ffn_down`), so 20 of 49
LoRA layers were updated with someone else's gradient (first-step sign
agreement with their own gradient 0.5, 0.33 or 0.0).

The gradients themselves were correct at the moment they were computed,
which is why gradient checks alone did not show it. Clipping is what turned
it on: without `--clip_grad` each gradient is applied right after its layer
and is unaffected. (The clipping added in "Was it gradient clipping?" above
was meant purely as a stability aid.) Mixed precision takes the same
deferred path.

**Fix** (`nntrainer/tensor/tensor_pool.{h,cpp}`): the pool remembers which
exec-order entry was `PERSIST_END_ORDER` (`SourceDetails::persist_idx`) and
keeps treating it as persist-to-end on every later plan. Regression test:
`TensorPool.persist_end_order_survives_refinalize_p` (fails on the old code:
the second plan needs 64 instead of 128 bytes because the two gradients
share memory).

### After both fixes

Exact path (`NNTR_TRAIN_FWD_FP32=1 NNTR_TRAIN_DX_Q8=0`), layers 21-27, all 7
LoRA FCs: dA / dB within 0.01-0.15% of PyTorch in hybrid mode (0.2-1% CPU-only,
where the reference also re-simulates layers 0-20); 0 of 49 layers with a
wrong update direction; the Adam + clip update reproduced to rel L2 1e-7
(first step) / 3e-4 (second).

Same data and schedule as the symptom (50 samples, lr 5e-5, decay 0.7 / 50
steps, clip 1.0), default integer kernels:

| run | train loss, epochs 1-6 | best valid |
|---|---|---|
| hybrid, QAT off, before | 2.28 / 1.88 / 2.31 / 2.42 / 3.46 / 4.71 | 1.663 |
| hybrid, QAT on, before | 2.39 / 1.90 / 2.16 / 2.12 / 2.68 / 3.64 | 1.648 |
| hybrid, Bug A fixed only | 2.31 / 1.91 / 2.10 / 2.11 / 2.61 / 3.53 | 1.641 |
| CPU-only, Bug A fixed only | 2.35 / 1.82 / 1.91 / 1.80 / 1.91 / 2.08 | 1.651 |
| **hybrid, QAT off, both fixed** | **2.16 / 1.74 / 1.57 / 1.45 / 1.39 / 1.37** | **1.303** |
| **hybrid, QAT on, both fixed** | **2.21 / 1.73 / 1.56 / 1.44 / 1.38 / 1.36** | **1.312** |
| **CPU-only, QAT off, both fixed** | **2.14 / 1.75 / 1.55 / 1.43 / 1.43 / 1.40** | **1.387** |

Monotonic and below the label-entropy floor; samples with loss above 4
fall from 7 to 0 by epoch 3 (1 left on CPU-only). Step time is unchanged (hybrid 167 ms/iter, iterations 4-13). The
"valid" set is the training set here (see the caveats above), so these
show that the optimizer now works, not generalization. Learning-rate and
schedule conclusions drawn before these fixes (e.g. "it is an lr issue")
were made with both bugs active and should be re-checked.

### Verifying training numerics

```bash
# device: FP32 forward / dX so the reference comparison is exact
adb shell 'rm -rf /data/local/tmp/gd && mkdir -p /data/local/tmp/gd'
adb shell "cd <dir> && NNTR_TRAIN_FWD_FP32=1 NNTR_TRAIN_DX_Q8=0 \
  NNTR_TRAIN_GRAD_DUMP=/data/local/tmp/gd ./nntr_lora_train <qs4cx model> \
  <data> --epochs 1 --max_samples 4 --lora_rank 8 --lora_alpha 16 \
  --lora_last_n 7 --seq_len 256 --lr 5e-5 --clip_grad 1.0 \
  [--hybrid_trunk <trunk model>]"
adb pull /data/local/tmp/gd ./gd
# host
python3 tools/lora_gradcheck/ref_grad.py --hf <HF Qwen3-0.6B> --dump gd \
  --layers 21-27                  # expect well below 1% everywhere
python3 tools/lora_gradcheck/check_update.py --dump gd --lr 5e-5 --clip 1.0
                                  # expect "wrong: 0 / 49", rel ~1e-7
```

Drop the two FP32 switches to measure the production kernels instead
(expect a few percent). `ref_grad.py --base fp32` is for an FP32 base model.
Leave out `--lr_decay_rate` for `check_update.py` runs.

### Pre-existing issues found on the way (not fixed here)

- `unittest_compiler` (TFLite export) fails on this branch:
  `Exporter::saveTflResult` is specialized for an FC props tuple without the
  `LoraQAT` / `LoraQatEmaMomentum` / `LoraGPU` members in `fc_props`, so the
  FC node no longer matches ("given node cannot be converted to tfnode").
- `unittest_qwen3_lora_training_smoke` is flaky under `meson test`, which
  sets a random `MALLOC_PERTURB_`: it fails for 1, 33, 128 and 165, with or
  without these fixes. The "base checkpoint" it saves comes from a model
  that is `initialize()`d but never given weights, i.e. uninitialized memory
  (zeros normally: the loss is exactly ln 32 and every activation is 0;
  garbage under perturbation, which then makes every LoRA gradient NaN).

## Commit structure

The work above is organized into ten reviewable, individually-buildable
commits. Each commit leaves the tree in a working state (compiles, and any
test it introduces passes at that point in history):

1. **`[fc_layer] fix incremental_forwarding crash for inference-only LoRA`**
   `nntrainer/layers/fc_layer.cpp` (the crash-fix hunk only)
2. **`[fc_layer] use standard LoRA init ordering (A=random, B=zero)`**
   `nntrainer/layers/fc_layer.cpp` (the initializer-swap hunk)
3. **`[CausalLM] add backprop (derivative + gradient) for rms_norm and reshaped_rms_norm`**
   `Applications/CausalLM/layers/{rms_norm,reshaped_rms_norm}.{cpp,h}` +
   the corresponding gradient-check tests + meson wiring
4. **`[CausalLM] add backprop for swiglu`**
   `Applications/CausalLM/layers/swiglu.{cpp,h}` + its tests (including the
   aliasing regression test) + meson wiring
5. **`[CausalLM] add backprop for lm_head and tie_word_embedding`**
   `Applications/CausalLM/layers/{lm_head,tie_word_embedding}.{cpp,h}` +
   their tests + meson wiring
6. **`[CausalLM] add LoRA training forward/backward for mha_core`**
   `Applications/CausalLM/layers/mha_core.{cpp,h}` + its tests (including
   the training-forward-vs-inference-prefill check) + meson wiring
7. **`[CausalLM] thread LoRA config and add a training entrypoint to Transformer`**
   `Applications/CausalLM/models/{transformer.{cpp,h},qwen3/qwen3_causallm.cpp,causal_lm.cpp}`
   + `unittest_qwen3_lora_training_smoke.cpp` + meson wiring
8. **`[CausalLM] add a LoRA training data pipeline and CLI driver`**
   `Applications/CausalLM/{lora_train.{cpp,h},train_qwen3_lora.cpp}` +
   meson wiring for `nntr_lora_train`
9. **`[CausalLM] load LoRA adapters at inference time`**
   `Applications/CausalLM/main.cpp`
10. **`[CausalLM] document LoRA training support and its commit structure`**
    this file

Follow-up commits for on-device speed and training correctness:

11. **`[utils] Add an env-gated per-layer training profiler`**
12. **`[layers] Speed up LoRA fully-connected training on quantized bases`**
    (int8 dX, skinny LoRA kernels)
13. **`[layers] Parallelize the host SwiGLU forward and backward`**
14. **`[CausalLM] Vectorize and parallelize mha_core training attention`**
15. **`[CausalLM] Fix QS4CX base weights in load_weight_lora`**
16. **`[CausalLM] Add GPU-trunk / CPU-head hybrid LoRA training`**
17. **`[CausalLM] Expose hybrid LoRA training in the API and SampleTestAPP`**
18. **`[tensor] Keep gradients clipped by global norm alive on every re-plan`**
    Bug B + regression test
19. **`[tensor] Keep a frozen layer's input until its calcDerivative`**
    Bug A + exe-order goldens
20. **`[CausalLM] Add LoRA gradient / update verification against PyTorch`**
    `NNTR_TRAIN_GRAD_DUMP`, `NNTR_TRAIN_FWD_FP32`, `tools/lora_gradcheck`
21. **`[CausalLM] Document fast on-device training and the two training bugs`**
    this file

## Known limitations / follow-ups

- `mha_core`'s training path is plain scalar C++ (chosen for verifiability
  over speed while the math was unproven); it is single-threaded and not
  SIMD, unlike the inference path. Training throughput on a dataset the
  size of SST-2 (67k samples) would take days, not hours, until this is
  optimized.
- `calcGradient` for the norm layers is unit-tested but was only exercised
  in a real run once (`--train_norms` / `lora_train_norms`); under the
  default LoRA-only recipe it is never called (those layers are frozen).
- Quantization-aware LoRA (Q4_0/Q6_K/W4A8) is out of scope for this change.
- `mha_core`'s training forward explicitly rejects sliding-window,
  attention-sink, and logit-softcapping configurations (throws rather than
  silently computing them wrong) — correct for Qwen3-dense, but any of the
  other CausalLM architectures using those features would need the training
  path extended first.
- The tensor-aliasing hazard documented above is specific to nntrainer's
  memory-reuse design and is not something the compiler or existing test
  harness catches automatically; any *future* layer that implements
  `calcDerivative` and reads an input after writing a different input's
  outgoing derivative should be checked for the same class of bug.
- `unittest_compiler`'s TFLite export of FC and the uninitialized "base
  checkpoint" in `unittest_qwen3_lora_training_smoke` are broken on this
  branch independently of the fixes above; see "Pre-existing issues found on
  the way".

## qint4 adapter format: the int4 values QAT trained against

A QAT run (`--lora_qat`) also saves `<output>_q4.bin`, the adapter as
per-channel int4 -- the same scheme the QS4CX base weights use -- so what
ships to the device is the exact quantization training saw.

The file is one record per loraA / loraB, in graph order, behind an `NQI4`
header:

    u32 magic "NQI4", u32 version 1
    record: u32 K, u32 N
            N * ceil(K/2) nibble bytes: row n of the (K x N) weight, even k
            in the low nibble, odd k in the high
            N fp32 scales
    value = (code - 8) * scale[n], with scale[n] = amax of channel n / 7

The record is the trimmed QS4CX plain layout, so an adapter is byte-valid
as a base weight of the same shape. `load_weight_lora_qint4` loads the base
checkpoint like `load_weight_lora` and dequantizes the record into the
graph's FP32 LoRA tensors (rank-sized LoRA GEMMs cost nothing next to the
base weights); the graph is never built with int4 LoRA tensors. A file
without the header (the Q4_0 format saved before the qint4 unification) is
refused -- re-train and re-save. The app's AFTER test loads the `_q4`
adapter of a QAT run.

The fake-quantizer matches the saved format exactly:

- the scale is per output channel (the weight's width): d = amax of the
  channel / 7, so both sign extremes land inside the code range and a
  channel's largest magnitude sits on code 7;
- the EMA tracks each channel's d across steps, and scales stay FP32 --
  the precision the file stores (the Q4_0 format rounded them to FP16).

So dequantizing the file reproduces, bit for bit, the fake-quantized weight
the saved per-channel scales give
(`QatQInt4AdapterRoundTripsTheTrainedInt4Values`).

