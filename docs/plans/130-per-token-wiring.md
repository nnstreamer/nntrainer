# 130 — Wire #82's small ops and #81's m=1 attention into `hexkl_graph.c`'s resident table (⑨ wiring, ㉓ step 1)

Issue: dlwlzzero/nntrainer#130 (LEDGER §3 ⑨ / ㉓; tracker #76). Read against
`htp_moe` @ `dbaf24b5` (PRs #119 / #125 / #126 / #127 / #128 merged). Builds on
`docs/plans/85-per-token-entry-skeleton.md` (the op table, `forward`, the
layer-granular resume), `82-m1-small-ops.md` (the four `_det` kernels, §3.4 the
RoPE table), `81-m1-attention-kv-cache.md` (the cache object, §3.6 the accuracy
decision) and `84-host-e2e-inproc.md` (the host E2E harness this issue gates on).

**What this buys, plainly: a *higher* call count, and the proof that the resident
set works end to end.** `forward` stops at the first non-resident op. In the LFM2.5
list (`htp_graph_lfm2_build`, `htp_graph_desc.h:388-426`) every one of the five new
kinds is separated from the MoE op by an op that stays on the ARM: `FC` (in_proj /
qkv / out_proj / o_proj), `ADD` (its second operand, the residual, lives on the
ARM) and `ROUTER_TOPK`. So no resident stretch merges with the MoE call. With all
five kinds on, the per-layer stretches are `[RMSNORM] FC [CONV1D_GATE] FC ADD
[RMSNORM] ROUTER [MOE] ADD` (conv, MoE) = 4 calls, `[RMSNORM] FC [QK_NORM ROPE
ATTN_M1] FC ADD [RMSNORM] ROUTER [MOE] ADD` (attention) = 4, `[RMSNORM] FC
[CONV1D_GATE] FC ADD [RMSNORM] DENSE_FFN ADD` (layers 0–1) = 3, plus the tail
`[RMSNORM] LM_HEAD` = 1: **6 × 4 + 16 × 4 + 2 × 3 + 1 = 95 FastRPC calls per
token** (22 today). Each extra call is one transport (≈ 88–92 µs since PR #103 /
#118, LEDGER rule 35) plus two small staging copies, so on the device this variant
is expected to read **≈ +6–8 ms/token, i.e. ≈ −20 % decode at G=512**, not a gain.
That is why the switch stays off by default and why §3.2 adds a per-kind mask so
the handoff can also run "MoE only" (22 calls, plan 85's never-measured B). The
count falls below 22 only with ㉓'s follow-up (M=1 FCs, router + top-k, lm_head),
which is explicitly not this issue (§3.6).

## 1. Goal and gate

Host-only gate (issue text; contract §9 rungs 0–1 on the Mac, 2–3 on the workstation
before merge). The issue ends in a PR into `htp_moe`, `state:review`. No device
number is claimed; the device ride-along (§4 step 7) is read under the ⑨ user
decision.

1. **`graph_host_check`** (`test/htp/host/run_host_checks.sh:184-189`), extended:
   * `resident_ok` = `MOE | RMSNORM | QK_NORM | ROPE | CONV1D_GATE | ATTN_M1` and
     `hexkl_graph_resident_kinds()` returns exactly that mask; the LFM2.5 list
     (228 ops, 22 MoE) validates with all six resident; the existing 10 mutations
     fail as before (the "resident bit on a kind with no kernel" mutation moves
     from op 0, a RMSNORM, to op 1, an FC, `graph_host_check.c:156-159`);
   * **one new negative case per kind**, each with a definite code, printed in the
     mutation table and never `AEE_EBADPARM` (rule 3): RMSNORM with `eps_bits` 0 →
     `AEE_EINVALIDFORMAT`; ROPE before QK_NORM in one layer → `AEE_EINVALIDITEM`;
     ROPE resident with `head_dim != 64` → `AEE_ESCHEMENOTSUPPORTED` (the kernel's
     shape rule; the same code refuses QK_NORM / ATTN_M1 at `head_dim % 32 != 0`
     or `max_seq % 32 != 0`, and RMSNORM at a `K` that is not a power of two);
     ATTN_M1 resident while its layer's ROPE is not → `AEE_ENOTALLOWED` (§3.1, the
     RoPE-inside-attention dependency); CONV1D_GATE: a `forward` at a CONV1D_GATE
     op with no `conv_w` bound → `AEE_EBADSTATE` (forward half). Also in the
     forward half: an ATTN_M1 op with **no registered cache** (`env->attn_m1 ==
     NULL`) → `AEE_EBADSTATE`; a RMSNORM op with no gamma bound → `AEE_EBADSTATE`;
     `pos > kv_len` (a hole) → `AEE_EBADSTATE` from the kernel, passed through.
   * **Bit identity through the graph loop.** The check now links the **real**
     `hvx_m1_ops_f32.c`, `hvx_conv_gate_f32.c`, `hvx_attn_m1_f32.c` and the pthread
     pool on `hvx_emu/` (plans 82 §3.2 / 81 §3.3), and `memcmp`s each resident
     stretch's `act_out` against the scalar specs: `[RMSNORM]` vs `rmsnorm_det`,
     `[CONV1D_GATE]` (chain of 8 tokens from a seeded state) vs `conv_gate_m1_det`,
     `[QK_NORM ROPE ATTN_M1]` at pos 0, 31, 32, 63 after a 32-row `kv_append` vs
     `rmsnorm_det` → `rope64_det` → `attn_m1_det`, on the hd64 shape of §3.4
     (hidden 128, 2 q heads, 1 kv head, head_dim 64, max_seq 64). It prints
     `GRAPH STRETCH BIT-IDENTICAL: RMSNORM CONV1D_GATE QK_NORM+ROPE+ATTN_M1` and
     asserts `resume_at` = the first non-resident op after the start (`start + 1`,
     `+ 1`, `+ 3`), that slot routing is the builder's (QK_NORM / ROPE in place in
     slot 2, ATTN_M1 slot 2 → 1) and that `op_pcycles` is non-zero exactly for the
     ops run. The MoE stand-in and its bit case (plan 85) are unchanged.
2. **`bash test/htp/host/run_inproc_e2e.sh`** prints `INPROC E2E PASS` with the
   present five gate lines **unchanged** (switch off: `E2E eval golden …
   bit_identical=1`, `hmx-loop … bit_identical=1`, `E2E tokens htp==cpu 8/8`,
   `cpu … min_snr_db ≥ 60`, `self-test ok` — the default path must not move by a
   byte, and the golden is not regenerated), plus the new lines of §3.5:
   * `E2E fwd tiny kinds=MOE,RMSNORM,CONV1D_GATE calls/token=11` and
     `E2E eval fwd-tiny … min_snr_db=<x>` with **x ≥ 40** against the switch-off
     run of the same fixture, every file listed (the 8 decode MoE calls' `_in` /
     `_out`, the prefill call, the 9 logits files);
   * `E2E fwd tiny-all-kinds refused: AEE_ESCHEMENOTSUPPORTED` — the hd8 fixture
     with all five kinds requested must be refused by the validator (head_dim 8),
     end to end, through the real throw;
   * on the new hd64 fixture (§3.4): `E2E eval golden-hd64 … bit_identical=1`
     (switch off, its own committed golden), `E2E fwd hd64
     kinds=MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1 calls/token=12`,
     `E2E eval fwd-hd64 … min_snr_db ≥ 40` vs its switch-off run, and
     `E2E tokens fwd==off 8/8 expected_mismatch=0` under the policy of §3.5;
   * the E2E log's `[HTP] graph: init n_ops=<n> resident=MOE|RMSNORM|QK_NORM|ROPE|
     CONV1D_GATE|ATTN_M1 …` line and `[HTP] attn_m1: registered layers=1 kv=1
     gqa=2 head_dim=64 max_seq=32 cache=… KiB`.

   **Why an SNR floor and not a bit gate here** (a deliberate deviation from the
   issue text, which asked for bit identity). The switch-on run computes the
   resident ops with the `_det` kernels (bit-equal to the specs, gate 1); the
   switch-off run computes the same ops on nntrainer's CPU path
   (`rms_norm_wrt_width_fp32_intrinsic`, `compute_kcaches_fp32_reference`, the
   NEON / x86 conv), which is not `_det` (plans 82 §3.1, 81 §3.2). The two differ
   by f32 rounding, so the MoE `_in` files cannot be byte-equal and the u8
   activation quantizer in front of the MoE GEMM can flip a level. The bit gate
   for the kernels is gate 1; the E2E gate proves the *wiring* (right op, right
   gamma / conv_w / state / pos, right slot, right stretch). Calibration: a wiring
   fault reads 0–20 dB (plan 84 §1, the harness's own note), f32 rounding alone
   reads ≥ 100 dB, one u8 level flip in a 64–128-wide row reads ≈ 50–60 dB; **40
   dB** sits between the fault band and the rounding band with margin on these
   shapes. The observed `min_snr_db` of the first passing run is recorded in the
   PR so a later drop is visible.
3. **Standing rung-1 set**: `ninja -C build`; `unittest_causallm_models
   --gtest_filter='*Lfm2Moe*'` 6/6 (fixture generated, none skipped); the
   `*qs4cx*` cpu-backend gtests; `run_host_checks.sh` → `ALL CHECKS PASS`, `WORKER
   POOL LANES OK`, `M1 OPS BIT-IDENTICAL`, `ATTN M1 BIT-IDENTICAL`, `GRAPH CHECKS
   PASS`; `tools/htp_syntax_check.sh`; `clang-format-14` on changed lines;
   `generate_stub.sh` (the IDL gains one method).
4. **Rungs 2–3, workstation, before merge**: `test/htp/build.sh` (`-Wall -Werror`,
   `UNDEFINED SYMBOLS OK`), `build_android.sh --htp`, `readelf -d libnntrainer.so`
   lists `libsdkl.so` + `libcdsprpc.so`, `ndk-build unittest_hvx_softmax
   unittest_hvx_attn` — **md5s in the PR** (LEDGER §3a: none were recorded for the
   cycle-16 merges). A PR opened from the Mac says so and waits for them.
5. **Standing E2E gates, carried to the device ride-along**: prefill ≥ −5 % of
   variant A (the hooks fire only at `rows == 1`; the conv block's prefill runs
   the `conv_block` layer's CPU path, §3.3, so the handoff must read it), text per
   the ⑨ user decision (a) / (b) / (c) — it binds the handoff, not this PR
   (§3.5, §4 step 7).

## 2. Where it lives

DSP side and the shared header (`path:line` at `dbaf24b5`):

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/htp_graph_desc.h` | **wire version 2** (`:39`). The op record (`:83-99`) gains four words after `rsv`: `n_kv`, `gqa`, `head_dim` (QK_NORM / ROPE / ATTN_M1; 0 elsewhere) and `eps_bits` (RMSNORM / QK_NORM: the f32 bits of the norm epsilon) → `HTP_GRAPH_OP_WORDS = 16 + 64 = 80`, 320 B, 80 KiB table at 256 ops. `htp_graph_lfm2_shape` (`:311-324`) gains `eps` (`Transformer::NORM_EPS`, `transformer.cpp:198-199`). The validator (`:195-308`) gains: the kernel shape rules per kind (RMSNORM `K` a power of two and multiple of 32; QK_NORM / ATTN_M1 `head_dim % 32 == 0`, `K == (gqa + 2) · n_kv · head_dim`, ATTN_M1 `N == gqa · n_kv · head_dim == hidden`, `max_seq % 32 == 0`; ROPE `head_dim == 64`) → `HTP_GRAPH_E_SCHEMENOTSUPPORTED` (`AEE_EOFFSET + 0x00F`); RMSNORM / QK_NORM `eps_bits` a positive finite float → else `INVALIDFORMAT`; op order inside an attention layer QK_NORM → ROPE → ATTN_M1 → `INVALIDITEM`; a resident ATTN_M1 whose layer's ROPE is not resident → `HTP_GRAPH_E_NOTALLOWED` (`+ 0x02B`). `htp_graph_err_name` (`:125-152`) names the two codes. The builder (`:358-438`) fills the new fields |
| `nntrainer/tensor/htp_backend/hmx/hexkl_graph.h` | `hexkl_graph_env` (`:31-39`) gains `hvx_attn_m1_ctx *attn_m1` (the session's cache; NULL = none). `hexkl_graph` (`:52-59`) gains per-op parameter and state pointers (`float *param[HTP_GRAPH_MAX_OPS]`, `float *state[HTP_GRAPH_MAX_OPS]`, DSP heap, calloc'd at `set_param`, freed in `hexkl_graph_free`) and `float *rope_cs` (`max_seq × 64`). New `int hexkl_graph_set_param(hexkl_graph *g, uint32_t op, uint32_t which, const float *data, uint32_t n)` with `which ∈ {HEXKL_GRAPH_PARAM_GAMMA, CONV_W, CONV_STATE, ROPE_TABLE}` (§3.1). `hexkl_graph_forward`'s doc (`:83-99`) gains the `AEE_EBADSTATE` cases (missing param / state / cache) |
| `nntrainer/tensor/htp_backend/hmx/hexkl_graph.c` | five wrappers in the kernel table (`:92-104`): `graph_op_rmsnorm` → `hvx_rmsnorm_f32(in, gamma, out, K, K, eps, NULL)`; `graph_op_qk_norm` → two `hvx_rmsnorm_f32` calls (q heads with `gamma[0..head_dim)`, k heads with `gamma[head_dim..2·head_dim)`, chunk `head_dim`), v copied when `in != out`; `graph_op_rope` → `hvx_rope64_f32(q, gqa·n_kv, k, n_kv, rope_cs + pos·64)` in place (copy first if `in_slot != out_slot`); `graph_op_conv1d_gate` → `hvx_conv_gate_m1_f32(in, state3[op], conv_w[op], out, N)`; `graph_op_attn_m1` → `hvx_attn_m1_forward(env->attn_m1, ordinal, pos, 1.0f / (float)sqrt(head_dim) — exact 0.125f at 64 —, q, k, v, out, NULL)`, `ordinal` = the count of ATTN_M1 ops before this one (computed at init into a per-op field). `hexkl_graph_init` (`:125-183`) also sizes the slots from the widest resident op (already `:153-158`; QK_NORM makes it 3072 words). `pos` reaches the kernels through `graph_call` (`:42-45`). Address-space comment (`:12-16`) updated: table 80 KiB + slots 36 KiB + params ≈ 1.8 MiB at LFM2.5 (plan 82 §3.4) + the 48 MiB cache the session owns |
| `test/htp/nntr_hvx.idl:592` | one method after `attn_m1_forward`: `AEEResult graph_set_param(in uint32 op, in uint32 which, in sequence<float> data);` (§3.1). Additive; earlier indices unchanged |
| `test/htp/nntr_hvx_graph.c` | `nntr_hvx_graph_set_param` entry (graph live → else `AEE_EBADSTATE`; lengths → `AEE_EINVALIDFORMAT`); `graph_env_of` (`:82-90`) sets `env->attn_m1 = s->attn_m1`; the init FARF (`:43-47`) prints the mask as names (`resident=MOE|RMSNORM|…`); the per-call FARF (`:94-108`) adds the per-kind pcycle sums |
| `test/htp/nntr_hvx_attn_m1.c`, `test/htp/nntr_hvx_small_ops.c`, `hvx/*.c` | **unchanged** (the kernels and their test entries are #81 / #82's; this issue only calls them) |
| `test/htp/build.sh:77-89` `SRCS`, `htp_backend/meson.build:74-99` inproc list | unchanged: every file already listed (`hexkl_graph.c`, the three kernel files, the pool). `hexkl_graph.c` now includes `hvx_m1_ops_f32.h` / `hvx_attn_m1_f32.h`, both on the include path |
| `test/htp/host/stub/AEEStdErr.h` | `AEE_ESCHEMENOTSUPPORTED`, `AEE_ENOTALLOWED` with the SDK's values if absent (the `#error` block at `hexkl_graph.c:29-39` gains both) |
| `test/htp/host/graph_host_check.c`, `run_host_checks.sh:184-189` | §1 gate 1. The compile line gains `-I hvx_emu -pthread -include malloc.h -ffp-contract=off`, the three kernel sources, `hvx_worker_pool.c`, `-lm` |

ARM side (core files outside the supervision scope named per contract §6):

| file | change |
|---|---|
| `nntrainer/tensor/cpu_backend/compute_ops.h:312` (**core**) | two virtuals beside `set_decode_graph_desc`: `virtual int decode_op_fp32(unsigned kind, unsigned pos, const float *in, unsigned in_len, float *out, unsigned out_len, const float *param, unsigned param_len, const float *state, unsigned state_len, float eps) { return 0; }` (0 = not resident, run the CPU path; 1 = done; 2 = the attention cache needs seeding first) and `virtual bool decode_kv_seed_fp32(unsigned n_rows, const float *k_rows, const float *v_rows) { return false; }` (rows `[0, n_rows)` of the layer whose hook just returned 2, `[n_rows][n_kv][head_dim]` f32). Plain scalars and pointers, no accelerator type in core (plan 85's rule) |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | the stretch machinery of §3.2: `set_decode_graph_desc` (`:1237-1266`) validates with the ARM-known mask, builds `stretch_end[i]` (first non-resident op after `i`) and the per-kind op index lists, reads `NNTR_HTP_FORWARD_KINDS`; `ensureGraphInit` (`:1301-1316`) adds `attn_m1_register` when any ATTN_M1 op is resident and prints the names; `decode_op_fp32` / `decode_kv_seed_fp32` overrides (per-kind counters keyed by `pos`, param binding through `graph_set_param`, conv-state and KV seeding, the pending stretch input); `invokeForward` (`:2023-2103`) generalised to `(start_op, expected_resume, pos, act, out)` and used by both the MoE call (`:1189-1195`) and the hooks; `dumpMoeCall` (`:1985`) unchanged for the MoE stretch, not called for the others; the profile `graph:` line (`:771-779`) prints the mask names and per-kind pcycles; a one-line summary at close `[HTP] graph: forward calls=<n> tokens=<t> calls/token=<n/t>` (the harness greps it); `graph_short_warned_`'s message (`:1197-1204`) covers the new kinds |
| `Applications/CausalLM/layers/rms_norm.cpp:47-118` | at `to - from == 1`, FP32: `decode_op_fp32(HTP_OP_RMSNORM, from, in, W, out, W, gamma, W, nullptr, 0, epsilon)` → return on 1. The kind constants come from `htp_graph_desc.h` under `#ifdef ENABLE_HEXKL` (as `lfm2_moe_causallm.cpp:20-23` includes it); the hook reaches the backend through `nntrainer::get_htp_ops()` (`compute_ops.h:639-643`) |
| `Applications/CausalLM/layers/qkv_layer.cpp:197-249` | after `input_step.dot(Weights, Outputs)` (`:242`), when `feature_size` and `to - from == 1`: concatenate `q_raw | k_raw | v` (three tensors, `:206-212`) into a 12 KiB thread-local row and call `decode_op_fp32(HTP_OP_QK_NORM, from, row, qkv, nullptr, 0, [q_gamma | k_gamma], 2·feature_size, …, epsilon)` → on 1 skip both `headNorm` calls (the Q / K outputs are then not written: mha_core's hook consumes the DSP stretch, §3.2) |
| `Applications/CausalLM/layers/mha_core.cpp:406-565` | at `step_size == 1`, not `use_external_cache`, FP32 or FP16 cache: `decode_op_fp32(HTP_OP_ATTN_M1, _from, nullptr, 0, output_row, num_heads_Q · head_dim, rope_table, max_timestep · 64, nullptr, 0, 0)`; on 2 build `k_rows` / `v_rows` from `cache_key` / `cache_value` rows `[0, _from)` (`:716-721` layout `[pos][n_kv · head_dim]`, fp16 → f32 on Android, `:203-238`), call `decode_kv_seed_fp32(_from, k, v)`, call the hook again; on 1 skip `one_batch_incremental_forwarding` and do **not** advance the CPU cache (the DSP owns it, §3.3). `rope_table` is built once per layer instance from `freqs_fp32->cos[i][0..31] | sin[i][0..31]`, `i < max_timestep` (`precompute_freqs`, `:871-905`; `calc_trigonometric_vals_dup` is the CPU RoPE's own source, plan 82 §3.4) |
| `Applications/CausalLM/layers/conv_block_layer.cpp:155-300` | at `rows == 1` after the in_proj `dot` (`:230`): `decode_op_fp32(HTP_OP_CONV1D_GATE, from, proj, 3C, y, C, conv_w, 3C, state, 2C, 0)` → on 1 skip `gate_pre` / `causal_depthwise_conv1d_k3_decode` / `gate_post` (`:253-256`) and go to the out_proj `dot`. `proj` is `a | b | c` contiguous, `conv_w` is `w0 | w1 | w2` (`causal_conv1d_layer.cpp:52-58` layout, `[1,1,3,C]`), `state` is `x_{t-2} | x_{t-1}` — exactly the kernel's contract (`hvx_m1_ops_f32.h:57-72`) |
| `Applications/CausalLM/models/lfm2/lfm2_causallm.cpp:136-156` | `createConvBlock` builds the one-layer `conv_block` form also when `NNTR_HTP_FORWARD` is set (engine stays `block_eng`, "cpu" by default, so prefill runs its CPU path: `use_htp` is false at `:192-194`). Same three weights in the file's order (`:139`), so the model file loads unchanged. The four-layer form (`:159-206`, `split` → `custom_multiply` → `causal_conv1d` → `custom_multiply`) has no single hook point for a `K = 3C` op — that is why the switch takes the folded layer (§3.3) |
| `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:60-97` | `shape.eps = NORM_EPS`; `resident_mask` from the backend's `NNTR_HTP_FORWARD_KINDS` parse (default: all six) instead of the hard-coded `HTP_GRAPH_KIND_BIT(HTP_OP_MOE)` (`:88`) |

Harness and fixture:

| file | change |
|---|---|
| `test/unittest/models/causallm_reference/generators/generate_lfm2_moe_reference.py:38-68` | the constants become `argparse` options with today's defaults (`--dim --n-heads --n-kv-heads --head-dim --max-pos --out`), so `--dim 128 --n-heads 2 --n-kv-heads 1 --head-dim 64 --max-pos 32 --out …/lfm2_moe_tiny_hd64` writes the second fixture (§3.4). Default output byte-identical to today's (the `*Lfm2Moe*` reference logits must not change) |
| `test/unittest/models/causallm_reference/lfm2_moe_tiny_hd64/` (new) | `config.json`, `nntr_config.json`, `tokenizer.json`, `meta.json`, `reference_*.json`; the `.bin` gitignored like the tiny one |
| `test/htp/host/golden/lfm2_moe_tiny_hd64/` (new, ≈ 200 KiB) | the switch-off dump set, `README.md` as the tiny one's |
| `tools/htp/htp_dump_eval.py` | `--snr-floor <dB>` (exit 1 below it, even when `bit_identical=0` is otherwise tolerated with `--allow-diff`), `--tokens-policy <ref_log> <got_log>` (§3.5's expected-mismatch rule from the logits files) |
| `test/htp/host/run_inproc_e2e.sh` | the runs and lines of §1 gate 2 (`NNTR_HTP_FORWARD=1`, `NNTR_HTP_FORWARD_KINDS`, `--max-seq 32` for hd64, the negative case, the `calls/token` grep). `NNTR_INPROC_GOLDEN=update` also writes the hd64 golden |
| `test/htp/host/htp_e2e_test.cpp` | prints `margin=<top1 − top2>` per step (for the token policy); a `--prompt-seed` option is **not** added (the policy does not need it) |

Consumers of a changed contract:

* **IDL + both stubs.** One additive method. `generate_stub.sh` (ARM) and
  `test/htp/build.sh` (skel) together (rule 3); the inproc build regenerates the
  header and links the same skel sources (no list change).
* **`HtpComputeOps`**: as above. `tools/htp_syntax_check.sh` declares every entry
  variadic, so it stays green and proves only the C++ side.
* **Quantizer format tag (`nntr_quantize_stream`), loader check: unchanged.** No
  weight bytes or dtypes change; gammas, conv weights and the RoPE table travel as
  f32 parameters bound after `graph_init`. The hd64 fixture is quantized with the
  existing flags.
* **`NNTR_HTP_PROFILE`.** The M==1 MoE row keeps receiving only the MoE stretch
  (`addInvokeMoeLayer`, `:1779-1782` of `invokeForward` today), so `host / dsp /
  transport` stay comparable to A. The other stretches feed `addInvokeForward`
  and a new per-kind table under the `graph:` line: `graph: calls=<n> ops/call=<x>
  resident=<names> dsp=<us>/call pcyc: RMSNORM=<p> QK_NORM=<p> ROPE=<p>
  CONV1D_GATE=<p> ATTN_M1=<p> MOE=<p>` (from `forward_debug`'s `op_pcycles`,
  level ≥ 2; level 3 keeps the 5× repeat and `min`, rule 2). `tools/htp_fc_report.py`
  parses `FC_FIELD` / `FC_STAGE` from `unittest_hvx_fc` and is unaffected.
* **Wire version 2** is a contract change between `lfm2_moe_causallm.cpp`,
  `htp_compute_ops.cpp`, `graph_host_check.c` and the skel — all compile the one
  header, and a stale skel refuses a v2 list with `AEE_EUNSUPPORTED` (named by
  `graphErr`, `:1224-1235`), not `AEE_EBADPARM`.

## 3. Design

### 3.1 DSP: parameters bound once, kernels called as they are

*Chosen.* Doc 45 §3.1 (activation handles, everything else bound at init) applied
to the small ops: gammas, conv weights, the conv state seed and the RoPE table go
to the DSP heap through `graph_set_param(op, which, data)` after `graph_init` and
before the first `forward`; `forward` refuses an op whose parameter is missing
(`AEE_EBADSTATE`). Lengths are validated against the op record (`GAMMA`: `K` for
RMSNORM, `2·head_dim` for QK_NORM; `CONV_W`: `3·N`; `CONV_STATE`: `2·N`, written
into rows 0–1 of the op's 3-row state buffer, re-sendable; `ROPE_TABLE`: `op ==
HTP_GRAPH_NO_OP`, `max_seq · 64`). The KV cache stays #81's session object
(`attn_m1_register / kv_append / release`, `nntr_hvx_attn_m1.c`): the graph borrows
it through `env->attn_m1` and never owns it, so `graph_release` and
`attn_m1_release` stay independent and `nntr_hvx_close`'s order (`hvx_add_f32.c:184-201`)
is unchanged. The ATTN_M1 op wrapper passes `ordinal` = attention-layer index
(count of ATTN_M1 ops before it), `pos` from the call, scale `1/√head_dim`
(exact at 64), `stats = NULL` (plan 81 §3.4: production pays nothing).

Why ATTN_M1 requires ROPE resident: on the CPU, RoPE is applied *inside*
`mha_core` (`mha_core.cpp:725`, `:747`), so once the ATTN_M1 hook takes over
`mha_core` no CPU RoPE runs; the DSP stretch must contain the ROPE op. QK_NORM is
independent (a non-resident QK_NORM runs in `qkv_layer` and the stretch starts at
ROPE with normed q / k as `act_in`). The validator enforces the one dependency
(`AEE_ENOTALLOWED`), so a `NNTR_HTP_FORWARD_KINDS=ATTN_M1` typo fails at
`graph_init`, not as wrong text.

Kernel shape rules move into the validator (`AEE_ESCHEMENOTSUPPORTED`) because
the kernels cannot return a status through the op table (`hvx_m1_ops_f32.h:21-25`'s
"SHAPE CONTRACT": they return without writing). This is what refuses the hd8
fixture with the attention kinds resident.

*Rejected: parameters as per-call side inputs of `forward`* (gamma travelling with
`act_in`, as `row_index` does for the routing). It doubles the per-call bytes for
RMSNORM (8 KiB gamma per 8 KiB activation), puts 24 KiB of conv weights on every
conv call, and contradicts plan 83 §2 / doc 45 §3.1 (handles out of the per-token
call). The routing stays per-call only because the router is on the ARM (plan 85).

*Rejected: the graph owning the KV cache* (`graph_init` allocating it). It would
tie the 48 MiB allocation to every `set_decode_graph_desc` re-init and make #81's
entries and gtests a second path to the same memory.

### 3.2 ARM: stretches run from the CPU layers' hooks; the graph executor is unchanged

*Chosen.* As plan 85 §3.1: "the ARM runs the remainder" is nntrainer's normal
graph executor, and no call crosses a layer boundary in the executor's sense —
every resident stretch begins and ends inside CPU layers this branch already
owns. A **stretch** is a maximal run of resident ops `[s, e)`; `stretch_end[s] = e`
is computed on the ARM from the words at `set_decode_graph_desc`. A hook
`decode_op_fp32(kind, pos, in, …)` arrives from the CPU layer of op `i`
(`i` = the `counter[kind]`-th op of that kind in list order; the counters reset
whenever `pos` changes, so every token re-synchronises and prefill hooks never
run — hooks fire only at `rows == 1`):

* `i` is a one-op stretch (RMSNORM, CONV1D_GATE): bind params if unbound, run
  `invokeForward(i, e, pos, in → out)`, return 1;
* `i` is the first op of a longer stretch (QK_NORM): bind its params, **record**
  `in` as the pending stretch input (12 KiB), return 1 — the layer's outputs are
  not written; nothing on the CPU reads them once the last op's hook consumes the
  DSP result;
* `i` is the last op of the stretch (ATTN_M1): bind the RoPE table if unbound,
  seed the cache if `kv_len_[ordinal] != pos` (return 2 first; the layer converts
  the fp16 rows and calls `decode_kv_seed_fp32`), run `invokeForward(s, e, pos,
  pending → out)`, return 1;
* a hook whose op is in the middle of a stretch (none today; ROPE has no CPU
  layer of its own) returns 1 with nothing to do;
* the MoE op keeps its path (`gemm_qs4cx_moe_layer_fp32:1189-1195`), now through
  the generalised `invokeForward(op, stretch_end[op] = op + 1, …)`.

`invokeForward` checks `resume_at == stretch_end[start]` and throws otherwise
(plan 85's `op + 1` rule, generalised). A pending input that is not consumed by
the next hook of its stretch, or `pos` moving backwards while a stretch is
pending, throws: the failure policy of contract §2 / plan 85 §3.1 (no silent
fallback). `pos` is the layer's `from` (`incremental_forwarding(context, from,
to, …)` in every hooked layer), which is the absolute token position in both the
app's decode loop and the harness.

**The mask.** `NNTR_HTP_FORWARD=1` turns the path on; `NNTR_HTP_FORWARD_KINDS`
(comma-separated kind names, default `MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1`)
chooses which kinds get the resident bit. The bit is the one source of truth for
both sides: the DSP runs what carries it, the ARM hooks return 0 for a kind
without it (the CPU path runs), and `stretch_end[]` follows. `KINDS=MOE`
reproduces plan 85's B (22 calls); the default gives 95 (§0). The DSP validator
remains the authority on what this skel can run (`AEE_ECLASSNOTSUPPORT`).

*Rejected: a model-level decode loop* (the ARM walks the op list and calls
`forward` per stretch, bypassing the executor — plan 85 §3.3's deferred end
state). It is still premature: every stretch here is bounded by CPU FCs, so the
executor must run between stretches anyway, and the loop would have to
re-implement the layer stack's data flow for no fewer calls. It becomes the right
shape when ㉓ makes the FCs resident and a stretch spans a whole layer.

*Rejected: layer-name or property based op binding* (a `htp_graph_op` property on
four layer classes, or parsing `layer<N>_…` in the backend). Call order per kind
with a per-`pos` reset is what plan 85 already does for the MoE handles, needs no
new layer property, and is checked end to end by gate 2 (a misbound gamma reads
≈ 20 dB).

### 3.3 State ownership: the DSP owns the conv state and the KV cache after the first routed token

After the CPU prefill (`rows > 1`: `conv_block_layer.cpp:257-262` saves the conv
state, `mha_core` fills `cache_key / cache_value` rows `[0, prompt)`), the first
decode hook per op seeds the DSP: conv state `x_{t−2} | x_{t−1}` (`CONV_STATE`,
16 KiB per layer) and the KV rows (`attn_m1_kv_append`, 2 MiB per layer at prompt
512, once). From then on the DSP advances both and the CPU copies are **stale**:
the CPU conv state is not updated (its decode kernel is skipped) and the CPU cache
does not grow (`cache_index` is not advanced by the hook path). Re-seeding rules,
all on the ARM: the conv state is re-sent when `pos != last_pos_ + 1` (the first
token after any prefill, whose CPU path refreshed it); the KV cache is re-seeded
when `kv_len_[ordinal] != pos` (`kv_len_` mirrors the DSP's `kv_len`, which the
kernel enforces — a hole is `AEE_EBADSTATE`, a rewind is allowed by the kernel but
the ARM throws on `pos < last_pos_` while a generation is running, because the CPU
state it would re-seed from is stale). A second prompt in one process runs a
prefill on the CPU (fresh state), then its first decode token re-seeds. On Android
the seeded KV rows are the CPU's fp16-rounded post-RoPE k and the fp16 v
(`mha_core.cpp:203-238`); the decode rows are f32 — LEDGER ⑨'s note, one more
reason option (a) of the accuracy decision cannot hold. On the host the cache is
f32 and the seed is exact.

The conv block under the switch is the `conv_block` layer (doc 51), whose CPU path
"folds the split and the two custom_multiply layers byte for byte"
(`conv_block_layer.cpp:203-207`) and whose weights are the four-layer form's in
the file's order (`lfm2_causallm.cpp:139`). With the switch off the model is
byte-for-byte today's (the golden gate proves it); with it on, the switch-on
prefill runs this layer's CPU path — the device handoff's prefill gate reads that.

### 3.4 The second fixture, `lfm2_moe_tiny_hd64`

The tiny fixture has `head_dim 8` (`lfm2_moe_tiny/config.json`), which the
RoPE kernel (head_dim 64 only), the per-head norm (chunk a multiple of 32) and
the attention cache (head_dim a multiple of 32) cannot run; RMSNORM (chunk 64)
and CONV1D_GATE (C = 64) can. So gate 2 uses **both**: the existing fixture for
RMSNORM + CONV1D_GATE and for the refusal of the attention kinds, and a second
fixture from the same generator with `hidden 128, 2 q heads, 1 kv head (gqa 2),
head_dim 64, layers conv / attention / conv, num_dense_layers 1, 4 experts top-2,
moe_intermediate 32, vocab 32, max_position_embeddings 32`. `n_heads · head_dim ==
hidden` holds (the builder's ATTN_M1 record has `N == hidden`, as LFM2.5), gqa 2
exercises the GQA fusion, and `--max-seq 32` is a multiple of 32 for the cache.
The `*Lfm2Moe*` gtests keep the hd8 fixture and are untouched.

### 3.5 The E2E gate's numbers: SNR floor and token policy

The floor is 40 dB (§1 gate 2, with the calibration). Tokens: `htp_dump_eval.py
--tokens-policy` compares the two runs' `logits_<step>.f32`: a top-1 mismatch at
step `s` is *expected* iff the switch-off run's margin `top1 − top2` at `s` is
below `2 · rms(logits_on − logits_off)` at that step (the rounding noise is
larger than the decision gap), otherwise it is a failure. It prints `E2E tokens
fwd==off k/8 expected_mismatch=m` and exits 1 on an unexpected mismatch. On these
fixtures every margin is ≫ the ≈ 1e-6 noise, so the expectation is `8/8
expected_mismatch=0`; the rule exists so a legitimate near-tie does not force a
fixture change and a real fault (a wrong-gamma run flips many tokens with large
margins) still fails.

The `calls/token` line is the count of `nntr_hvx_forward*` calls divided by the
decode steps, from the backend's close-time summary; the harness greps 11 (hd8,
three kinds) and 12 (hd64, six kinds) — the stretch arithmetic of §0 on the
fixtures' layer lists (`conv dense: 3, attention MoE: 4, conv MoE: 4, tail: 1`;
without the attention kinds the attention layer is 3).

### 3.6 Explicitly not done

The M=1 FCs (conv in_proj / out_proj, q/k/v/o, the dense FFN of layers 0–1;
Q4_0 on the NPU model), the router FC + top-k, LM_HEAD, ADD (its residual operand
is on the ARM until the FCs move), the weight-format decision they need (LEDGER
㉓ — the supervisor files it as the follow-up); the NEON `_det` twin on the CPU
(option (c)); the fp16 KV cache and the `l2fetch` lead for the KV read (plan 81
§3.1 upgrades); the rpcmem activation mapping (plan 85 §3.3); any prefill-shape
residency.

## 4. Steps

Rungs from `.claude/skills/hexagon-gates`. The Mac runs rungs 0–1 (`tools/docker/run.sh
bash -c 'source tools/htp/env.sh && <command>'`); the generator needs torch in the
container (the skill's pip line). Rungs 2–3 are the workstation's.

1. **Header v2 + validator negatives** (`htp_graph_desc.h`, `graph_host_check.c`
   validator half: the five negatives, the moved no-kernel mutation, the two new
   codes in `stub/AEEStdErr.h`, the LFM2.5 list with all six resident, `kLfm25` /
   `kTiny` gaining `eps`). Gate: rung 1 `run_host_checks.sh` → `GRAPH CHECKS PASS`
   with the extended table. Mac.
2. **`hexkl_graph.{c,h}`**: env `attn_m1`, per-op params / state, `set_param`, the
   five wrappers, the mask; `graph_host_check.c` forward half on the real kernels
   (`GRAPH STRETCH BIT-IDENTICAL …`, the forward negatives, `resume_at`, slots,
   pcycles); `run_host_checks.sh` compile line. Gate: rung 1 `run_host_checks.sh`
   → `ALL CHECKS PASS`; `clang-format-14`. Mac.
3. **IDL + skel glue** (`graph_set_param` entry, `graph_env_of`, the FARF names).
   Gate: `generate_stub.sh`; `qaic` + `hexagon-clang -c` of `nntr_hvx_graph.c` and
   `hexkl_graph.c` in the container (as PRs #125 / #126 did); the inproc build
   (`ninja -C build_htp_host`) links with `--no-undefined`. Rung 2 proper
   (`test/htp/build.sh`, `UNDEFINED SYMBOLS OK`, md5) on the workstation in step 6.
4. **ARM**: `compute_ops.h` (two virtuals), `htp_compute_ops.cpp` (§3.2, §3.3, the
   mask, `attn_m1_register` at init, profile lines, the close summary), the four
   layer hooks, `createConvBlock` under the switch, the builder's `eps` and mask.
   Gate: rung 1 `ninja -C build`, `*Lfm2Moe*` 6/6, `*qs4cx*`,
   `tools/htp_syntax_check.sh`, `clang-format-14`. Mac.
5. **Harness**: generator options + hd64 fixture + its golden (`NNTR_INPROC_GOLDEN=update`,
   deliberate, said in the commit), `htp_dump_eval.py --snr-floor / --tokens-policy`,
   `htp_e2e_test.cpp` `margin=`, `run_inproc_e2e.sh` runs and lines. Gate: rung 1
   `run_inproc_e2e.sh` → `INPROC E2E PASS` with every line of §1 gate 2; the old
   five lines unchanged; the observed `min_snr_db` values and the two `calls/token`
   numbers copied into the PR. Mac.
6. **Rungs 2–3, workstation**: `test/htp/build.sh`, `build_android.sh --htp`
   (`--cache` after the first), `readelf -d`, `ndk-build unittest_hvx_softmax
   unittest_hvx_attn`, md5s of skel, app libs and gtests in the PR. PR into
   `htp_moe`, `state:review`. **The issue ends here.** A PR opened from the Mac
   states that rungs 2–3 were not run there and is not merged before they are.
7. **Device ride-along (a later sitting's handoff, not this PR's gate; a device
   measurement is unavoidable for any speed or text verdict).** One sitting, full
   E2E, prompt 512, gen 64 / 512 / 1024, `NNTR_NUM_THREADS=8`, one binary set,
   ≤ 4 variants, A first:
   * **A**: `htp_moe` head, switch off (the reference);
   * **B**: `NNTR_HTP_FORWARD=1` (all six kinds, **95 calls/token**);
   * **C**: `NNTR_HTP_FORWARD=1 NNTR_HTP_FORWARD_KINDS=MOE` (22 calls; plan 85's
     B, never measured — the plumbing's own cost);
   * **B-prof**: B at `NNTR_HTP_PROFILE=2`, G=64 once: the init line
     (`resident=MOE|RMSNORM|…`, `attn_m1: … cache=49152 KiB`), the `graph:` line's
     per-kind pcycles, and `ATTN_M1`'s cost at pos 512–575 against plan 81 §0's
     0.5 ms estimate.

   Plus, on that sitting's skel: `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'`
   and `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` (`bad=0`; `0x8000040e` =
   stale skel), and the `ATTN_M1_FIELD pos=` read. Expected: B ≈ −20 % decode vs A
   at every G (§0), C within A's drift, prefill of B and C ≥ −5 % of A, M==1
   `dsp=` of C ≈ A. **The accuracy column is the ⑨ user decision** — (a) text B ≡ A,
   expected to fail on the attention layers alone; (b) `NNTR_PPL` (#110 (d)) with
   the CPU `q40` PPL as the reference, the supervisor's recommendation; (c) a NEON
   `_det` twin — it binds this handoff only; the PR's host gate does not depend
   on it.

## 5. Risks (host-vs-device gaps, and how the handoff shows them)

* **The call count rises by construction** (§0: 22 → 95). Not a host-visible
  cost. The handoff's B vs C vs A isolates it: C − A is the plumbing, B − C is
  73 round trips plus the five kinds' DSP time; B-prof's per-kind pcycles say how
  much is compute. A B *faster* than C would contradict rule 35's transport
  figure and is the finding to look for.
* **Heap and address space** (rule 8). The DSP side adds ≈ 48 MiB (KV cache at
  `max_seq` 2048, 26 % of the ≈ 182 MiB heap) + ≈ 1.8 MiB (params) + 116 KiB
  (table, slots) on top of the 3840 MiB arena; the host is 64-bit and cannot see
  the limit. `attn_m1_register`'s FARF line and `[HTP] attn_m1: registered …
  cache=… KiB` on the first B token prove the allocation; `AEE_ENOMEMORY` throws
  and voids B. A `max_seq_len` of 4096 would take 96 MiB — the plan allows it but
  the handoff notes the figure; the kernel refuses > 2 GiB.
* **Conv state and KV ownership across calls** (§3.3). The CPU copies go stale
  after the first routed token. A second prompt re-seeds through the `pos` jump;
  a rewind throws. Host gate 2's 8-step decode plus the graph check's 8-token conv
  chain and 4-position attention chain cover the bookkeeping; the device adds only
  the fp16 seed rows, whose effect is the accuracy column's business.
* **`pos` bookkeeping.** Three places must agree: the layers' `from`, the ARM's
  `kv_len_[ordinal]` / `last_pos_`, the DSP's `kv_len`. The kernel's hole check
  (`AEE_EBADSTATE`) makes a disagreement a throw, never a silent wrong row.
* **The non-resident FC returns to the ARM mid-layer.** Four `invoke_mutex_`
  sections per layer, each with a 8–12 KiB staging copy in and 8 KiB out; the
  poll window of rule 23's corollary is hit 95 times. Visible in B-prof's
  `transport` and in B − C.
* **Stale skel / stale stub.** One new method; a stale skel returns `0x8000040E`
  on the first `graph_set_param`, a stale ARM stub fails to compile. `graphErr`
  names it; the first B token is the check.
* **DVFS / thermal drift between sittings.** B's regression is read as a
  same-sitting A/B/C (rule 9); the per-kind pcycles, not tok/s, carry the op cost
  (rules 20, 23, 27).
* **The emulation premise** (plans 81 §5 / 82 §5) still stands between the host
  gates and silicon: `HvxM1Ops.*` / `HvxAttnM1.*` ride along.
* **Prefill.** The switch-on prefill runs `conv_block`'s CPU path instead of the
  four layers; byte-equal by that layer's claim, speed within noise expected —
  the −5 % gate on B and C is the check. Switch-off prefill is untouched.
* **Baseline.** `run_host_checks.sh` is green in the Mac container at
  `dbaf24b5` (this planning session: `ALL CHECKS PASS`, `WORKER POOL LANES OK`,
  `GRAPH CHECKS PASS`, `M1 OPS BIT-IDENTICAL`, `ATTN M1 BIT-IDENTICAL`, the two
  GEMV mutants caught, rc 0); `build/` and `build_htp_host/` are current for
  `dbaf24b5`. `run_inproc_e2e.sh` was not re-run here; the implementer runs both
  harnesses first.

## 6. Docs to update

* **BENCHMARK.md**: nothing in this PR (no number). The ride-along adds, under its
  sitting and serial, rows "#130 B (six kinds resident, 95 calls/token)" and
  "#130 C (MOE only, 22 calls/token)" beside A, with the accuracy column per the
  ⑨ decision, and one `HvxM1Ops` / `HvxAttnM1` gtest line each.
* **LEDGER.md**: ⑨ — "wired (PR #…): `hexkl_graph.c` table = MOE | RMSNORM |
  QK_NORM | ROPE | CONV1D_GATE | ATTN_M1, host-gated on both fixtures; 95
  calls/token with all on, 22 with `KINDS=MOE`; parameters bound through
  `graph_set_param`, KV seeded from the CPU prefill once, conv state per layer on
  the DSP; device cost pending the ride-along". ㉓ — becomes the only path below
  22 calls; the follow-up issue (FCs, router, lm_head, ADD, weight format) is the
  supervisor's to file. §3a — the PR's skel / app md5s. A design verdict for §2:
  "per-op residency without adjacency to the MoE op raises the call count; a
  stretch pays only when it absorbs a round trip". The open decision (a)/(b)/(c)
  stays open; this PR does not need it.
