# 85 — Per-token entry skeleton: validated op table, `forward` / `forward_debug`

Issue: dlwlzzero/nntrainer#85 (LEDGER §4 lift 2 → §3 ⑨; tracker #76).
Read against `htp_moe` @ `9fcbbc20`. Builds on `docs/plans/83-transport-options.md`
§2 (the prebind table) and §4 (dspqueue deferred until one call per token).

## 0. What this buys at G=512 — plainly

LEDGER §2 budget row (#88 B, G=512, estimate): token **42.0 ms** = MoE dsp
**30.4** (72 %) + transport 22 × 0.088 = **1.9** + outside the MoE call
**≈ 9.6** (byte floor ≈ 6.5).

**The skeleton alone removes 0 ms and does not move tok/s.** In this issue
the only resident kind is the MoE FFN. Between two MoE ops of LFM2.5 sit
ops that stay on the ARM: RMSNorm, the conv or attention block and its FC
projections, the residual add, and the router FC + top-k. So `forward` runs
one op and returns: **22 calls/token, as today**. Both the 1.9 ms and the
9.6 ms stay. The only per-call change is that the 32 + 32 handles move from
the call into the table bound at init. Plan 83 §2 put that at ≤ 5–10 µs/call,
so ≤ 0.1–0.2 ms/token. That is under one sitting's drift (rule 9: ±5 %,
about ±2 ms at 42 ms), and this plan does not claim it.

What it enables: the call count falls only when **every** op between two
MoE ops is resident. At 1 call/token the transport goes 1.9 → ≈ 0.1 ms
(−1.8), and the outside-the-call time can approach its floor (−≈ 3 ms of
excess). That is **≈ −5 ms at G=512**: 42.0 → ≈ 37 ms (≈ 27 tok/s) on
today's MoE column. At the ladder's MoE-at-floor point (≈ 24 ms) it is the
difference between ≈ 41 and ≈ 50 tok/s. #81 (m=1 attention) and #82
(norm / RoPE / conv1d + gating) do not cover that set. Nothing yet covers
the **FC projections** (conv in_proj / out_proj, q/k/v/o, dense FFN), the
**router FC + top-k** or **lm_head** at M=1. Those are Q4_0 on the NPU model
today (contract §11). See §7.

## 1. Goal and gate

Host gates only (issue: "No device step before #77 is measured"; #77 is done,
but the issue ends in a PR with contract §9 gates 0–3; the device handoff is
a later issue):

1. `test/htp/host/graph_host_check.c` (new), run from `run_host_checks.sh`:
   * the LFM2.5-8B-A1B op list (24 layers: 18 conv + 6 attention in the
     `config.json` `layer_types` order, layers 0–1 dense FFN, 22 MoE, tied
     lm_head), built by the header's builder, validates → `AEE_SUCCESS`;
   * five mutations fail, each with its **own** code (a table in the test,
     printed): wrong shape → `AEE_EINVALIDFORMAT`, missing / unregistered
     weight handle → `AEE_EINVHANDLE`, unknown kind → `AEE_ENOTYPE`,
     attention op in a conv layer → `AEE_EINVALIDITEM`, `next_mm[i] <= i`
     → `AEE_EBADITEM`. Also: a resident bit on a kind with no kernel in this
     build → `AEE_ECLASSNOTSUPPORT`, and a truncated list →
     `AEE_EINCOMPLETEITEM`. **No validator path returns `AEE_EBADPARM`.**
     That code stays the stale-skel symptom (rule 3, rule 17's table).
   * `forward` with every op non-resident is the identity: it returns at
     once, `resume_at == start_op`, the output is untouched, and the per-op
     pcycles are all 0;
   * `forward(start_op = MoE op of layer L)` with MoE resident is
     **bit-identical** (memcmp) to a direct `hexkl_mm_u8i4_moe_layer_run`
     on the tiny fixture's shapes (hidden 64, inter 64, 4 experts, top-2,
     `test/unittest/models/unittest_causallm_lfm2_moe.cpp:96-106`), with the
     scalar stand-ins, for `flags` 0 **and** `HEXKL_MOE_FLAG_M1_GEMV`, and
     `resume_at == start_op + 1`.
2. `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6, and
   `*qs4cx*` cpu-backend gtests unchanged. **Limit, stated plainly:** the host
   build has no HTP, so `NNTR_HTP_FORWARD=1` changes nothing on the host. This
   gate proves only that the switch plumbing does not perturb the CPU path.
   Text identity with the switch on is decided by the device handoff (§4
   step 6), not here.
3. `clang-format-14` on changed lines; `ninja -C build`;
   `tools/htp_syntax_check.sh`; stub regenerated (`generate_stub.sh`);
   `test/htp/build.sh` (`HEXKL_SDK_VER=6.4.0.1`, `-Wall -Werror`,
   undefined-symbol guard of rule 17 passes); `build_android.sh --htp`,
   `readelf -d libnntrainer.so` lists `libsdkl.so` + `libcdsprpc.so`; md5s in
   the PR.
4. Standing gates, carried to the later handoff: prefill ≥ −5 % of variant
   A (prefill keeps the per-op entries; the table shares the worker pool,
   the DMA ring and `moe_scratch`), and text identical to A and to the CPU
   run. A's text already differs from the CPU `q40` run (LEDGER ⑱ side
   reading), so the text gate is **B ≡ A**.

## 2. Where it lives

DSP side:

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/htp_graph_desc.h` (new, C99, header-only, same pattern as #108's `htp_moe_opts.h`) | wire format: header `{magic, version, n_layers, n_ops, hidden, vocab}`, `layer_kind[]` (CONV/ATTN), `ffn_kind[]` (DENSE/MOE), fixed 76-word op record `{kind, layer, resident, K, N, N_out, n_experts, top_k, in_slot, out_slot, next_mm, rsv, h_gu[32], h_dn[32]}`; kind enum (RMSNORM, FC, CONV1D_GATE, QK_NORM, ROPE, ATTN_M1, ADD, ROUTER_TOPK, MOE, DENSE_FFN, LM_HEAD); the error codes of §1; `static inline` validator and the LFM2 builder. It is compiled by the skel, the host check and `htp_compute_ops.cpp` from one source |
| `nntrainer/tensor/htp_backend/hmx/hexkl_graph.{c,h}` (new) | `hexkl_graph_init` (validator + handle check against `hexkl_weight_u8i4_table` + table copy), the `forward` loop `for i in [start, n): if !resident break; table[kind]()`, per-op pcycles into a fixed `uint64 op_pcycles[HEXKL_GRAPH_MAX_OPS]`, and the MoE op wrapper that calls `hexkl_mm_u8i4_moe_layer_run` (`hexkl_mm_u8i4_moe.h:115`) unchanged |
| `test/htp/nntr_hvx_graph.c` (new) | skel entries `graph_init / graph_release / forward / forward_debug`; one FARF line per call |
| `test/htp/nntr_hvx.idl:449` | four methods after `moe_set_opts` (§3.2) |
| `test/htp/nntr_hvx_session.h:62-64` | `hexkl_graph *graph;` (NULL = none; the session is calloc'd) |
| `test/htp/hvx_add_f32.c:184` `nntr_hvx_close` | free the graph **before** the weight tables are released |
| `test/htp/nntr_hvx_mm_u8i4.c:138` `weight_release_u8i4` | return `AEE_EBADSTATE` for a handle bound in the live graph (the `arena_detach` rule, IDL `:56-59`) |
| `test/htp/nntr_hvx_mm_u8i4.c:982-1041` | move the probe → `stage_us` block of `moe_layer_timed` into one static helper that `forward_debug` also calls, so the two cannot drift |
| `test/htp/build.sh:77-89` `SRCS` | add `nntr_hvx_graph.c` and `$BACKEND/hmx/hexkl_graph.c` (rule 17) |
| `test/htp/host/stub/AEEStdErr.h`, `stub/HAP_perf.h` | add the codes of §1 with the SDK's values (`AEE_EOFFSET 0x80000400` + n) and `HAP_perf_get_pcycles` |
| `test/htp/host/graph_host_check.c` (new) + `run_host_checks.sh` | §1 gate 1 |

ARM side:

| file | change |
|---|---|
| `nntrainer/tensor/cpu_backend/compute_ops.h:294` (**core, outside the supervision scope, named per contract §6**) | one virtual: `virtual bool set_decode_graph_desc(const std::vector<uint32_t> &words) { return false; }`. It uses words, not a new type, so core gains no HTP struct |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` | override that stores the words. In `gemm_qs4cx_moe_layer_fp32` (`:1044-1084`), when `NNTR_HTP_FORWARD=1` and `M == 1`: fill each MoE op's handles from the first pass's call order, call `graph_init` once, then `invokeForward` (a sibling of `invokeMoeLayer` `:1714`, the same staging pools and `stagedMemcpy`). `sendMoeOptsOnce` (`:1094`) runs before `graph_init`, so both paths use one `moe_flags` |
| `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp` (after setup, `:38` `NUM_DENSE_LAYERS`; `layer_types_` parsed at `lfm2/lfm2_causallm.cpp:295`) | when the switch is set, build the words with the header's builder and hand them to the HTP context's ComputeOps. Guarded `#ifdef ENABLE_HEXKL` |

Consumers of a changed contract:

* **IDL + both stubs.** `generate_stub.sh` (ARM, `htp_backend/generated/`,
  gitignored) and `test/htp/build.sh` (skel). The local
  `generated/nntr_hvx.h` on this workstation is **already stale** (it
  names `mm_u8i4_conv_block` and `fwht_rows_f32`, which `htp_moe`'s IDL
  does not have). Regenerate before the first build.
* **`HtpComputeOps`**, as above.
* **Quantizer format tag (`nntr_quantize_stream`) and the loader check:
  unchanged.** No weight bytes or dtypes change.
* **`NNTR_HTP_PROFILE`.** `invokeForward` feeds the **same** bucket through
  `addInvokeMoeLayer` (`:303`, called at `:1807`), so the M==1 row
  (host / dsp / transport, stages) stays comparable across A/B. At level
  ≥ 2 it calls `forward_debug` and gets the MoE op's `stage_us` from the
  shared helper. It adds one line under the M==1 row:
  `graph: calls=<n> ops/call=<x> resident=MOE loop_us=<forward dsp − moe op>`.
  Level 3's 5× repeat is kept.
* **`tools/htp_fc_report.py`: unaffected.** It parses `FC_FIELD` /
  `FC_STAGE` markers from `unittest_hvx_fc`, not `[HTP-PROFILE]`.
  `tools/htp_attn_report.py` is likewise untouched.

## 3. Design

### 3.1 Chosen: layer-granular resume, DSP owns the table

* **Real kinds in the list, residency as a bit.** The ARM sends the whole
  LFM2 decode sequence with real kinds, so the validator can check layer
  consistency (attention in a conv layer, MoE in a dense layer, lm_head
  last). Each op carries `resident`. `hexkl_graph`'s kernel table has a
  non-NULL slot only for `MOE`. `ATTN_M1` is a defined kind with a NULL slot
  until #81; the other kinds wait for #82 and later issues. This is
  hvx_impl's `htp_op_table` NULL-slot rule (`htp_graph.c`), with a
  distinct code.
* **`next_mm[]` is supplied and validated, not built.** The issue's gate
  needs a "points backwards" failure. The builder emits it: the next
  weight-streaming op (`FC | MOE | DENSE_FFN | LM_HEAD`), or `UINT32_MAX`.
  The validator requires `next_mm[i] > i` and that it names a
  weight-streaming op. Nothing consumes it yet (the prefetch hook is a later
  issue).
* **Handles bound at init (plan 83 §2).** The MoE op record holds its 32 +
  32 handles. `graph_init` checks each one against `weights_u8i4` (in use,
  K/N match gate_up = K × 2·inter, down = inter × N_out). The 64-word
  per-layer handle sequences leave the per-token call.
* **Activation slots, not host pointers (doc 45 §3.1).** Ops read and
  write session-owned DSP-heap slots named by `in_slot` / `out_slot`
  (3 × `hidden` f32 = 24 KiB). `forward`'s `act_in` / `act_out` sequences
  are the **f32 wrapper** §3.1 prescribes: the entry copies `act_in` into
  the start op's input slot, and the exit copies the last run op's output
  slot out. The kernels already take pointers. When the ARM stops touching
  activations, the wrapper goes away and the kernels stay.
* **Routing is the per-call side input of the start op.** The router FC +
  top-k stays on the ARM in this issue, so `forward` carries `row_index /
  row_count / row_weight`, validated against the start op's `n_experts`
  (the same checks as `check_moe_layer_args` / `check_moe_row_totals`,
  `nntr_hvx_mm_u8i4.c:906-953`). They become empty once `ROUTER_TOPK` is
  resident.
* **The ARM "runs the remainder" = nntrainer's normal graph executor.** In
  this issue a call never crosses a layer boundary (`resume_at == start+1`
  always; a mismatch throws). So no graph-executor change is needed. The
  core files that **will** change when the resident set spans an OP_HOST-free
  stretch (skip layers already run on the DSP, keyed on `resume_at`) are:
  `nntrainer/models/neuralnet.cpp:504-563` (`incremental_forwarding`),
  `nntrainer/graph/network_graph.cpp:418`, or, preferably and in scope,
  the decode loop in `Applications/CausalLM/models/causal_lm.cpp:583-634`.
  They are named here and not touched.
* **Failure policy (contract §2).** With the switch on, a `graph_init` or
  `forward` error **throws**, carrying the AEE code. It does not silently
  fall back to `mm_u8i4_moe_layer`: a fallback would report per-layer
  numbers as the graph's (the same reason `sendMoeOptsOnce` throws,
  `htp_compute_ops.cpp:1086-1094`). With the switch off (the default) the
  path is byte-for-byte today's.
* **Profiling.** Per-op pcycles (`HAP_perf_get_pcycles`) go into the fixed
  table. **One FARF line per call** (`RUNTIME_HIGH`, so it is silent unless
  the FARF mask enables it): `[graph] start=<i> resume=<j> ops=<n>
  pcyc=<total> moe=<p>`. `forward_debug` returns
  `op_pcycles[n_ops_limit]`, the MoE `stage_us`, and the output slot of the
  last op run (the per-op output, for bisecting, hvx_impl
  `find_divergence.py` style).
* **Budgets.** The table is ≤ 128 ops × 304 B ≈ 38 KiB of DSP heap, plus
  24 KiB of slots. That is noise against the ≈ 182 MiB heap and needs no
  arena. No new VTCM, no new DMA, no arithmetic before a quantizer (§3.3
  `_det` untouched). Bit-identity holds by construction (same kernel, same
  flags, same scratch); the host check proves the plumbing.

### 3.2 IDL (additive)

```
AEEResult graph_init(in sequence<uint32> desc, rout uint32 n_ops);
AEEResult graph_release();
AEEResult forward(in uint32 start_op, in uint32 pos,
                  in sequence<uint32> row_index, in sequence<uint32> row_count,
                  in sequence<float> row_weight,
                  in sequence<float> act_in, rout sequence<float> act_out,
                  rout uint32 resume_at);
AEEResult forward_debug(in uint32 start_op, in uint32 n_ops_limit, in uint32 pos,
                  in sequence<uint32> row_index, in sequence<uint32> row_count,
                  in sequence<float> row_weight,
                  in sequence<float> act_in, rout sequence<float> act_out,
                  rout uint32 resume_at, rout sequence<uint32> op_pcycles,
                  rout sequence<uint32> stage_us);
```

`pos` is carried now (the attention and RoPE kinds need it) and checked
`< max_seq`. `tokens → logits` arrives with the embedding and lm_head kinds.
It is not in this signature, because nothing would fill it.

### 3.3 Rejected: a model-level decode loop now (ARM walks the op list and calls `forward` repeatedly, replacing the graph executor for decode)

That is the end state, but with only MoE resident it replaces nntrainer's
executor for the whole decode step to save nothing (22 calls either way).
It also puts `neuralnet.cpp` / `network_graph.cpp` in the diff before any
second resident kind exists to test it against. Deferred to the first issue
where two adjacent ops are resident.

Also rejected: mapping an rpcmem activation fd now (`fastrpc_mmap`, the
hvx_impl `executor.c` + `AEE_EALREADY` lift). While the ARM reads and writes
every activation, a mapped cached buffer moves the cache maintenance into
our code, and an uncached one is doc 46 §10.1's trap. It pays only once the
ARM stops touching activations.

## 4. Steps

1. **Header + validator + builder** (`htp_graph_desc.h`), and the host
   check's validator half (the LFM2.5 list plus 7 mutations). Gate:
   `run_host_checks.sh`.
2. **`hexkl_graph.{c,h}`** (init, forward loop, MoE wrapper, pcycles),
   plus the stub additions. Host check's forward half: all-non-resident
   identity; MoE bit-identity vs direct `hexkl_mm_u8i4_moe_layer_run` for
   flags 0 and M1_GEMV; `resume_at`. Gate: `run_host_checks.sh`.
3. **IDL + skel glue** (`nntr_hvx_graph.c`, session field, close order,
   `weight_release` refusal, the shared `stage_us` helper, `build.sh` SRCS).
   Gate: `generate_stub.sh`, then `test/htp/build.sh` (`-Wall -Werror` + the
   undefined-symbol guard), md5.
4. **ARM** (`compute_ops.h` virtual, `htp_compute_ops.cpp` override +
   `invokeForward` + `graph:` profile line, `lfm2_moe_causallm.cpp` builder
   call). Gate: `ninja -C build`, `*Lfm2Moe*` 6/6, `*qs4cx*`,
   `tools/htp_syntax_check.sh`, `clang-format-14`.
5. **App**: `build_android.sh --htp` then `--htp --cache`, `readelf -d`,
   md5s. PR into `htp_moe`, `state:review`. **The issue ends here.**
6. **Device (a later issue; unavoidable for any speed or text verdict).**
   One sitting, full E2E, prompt 512, gen 64 / 512 / 1024, ×2 runs,
   `NNTR_NUM_THREADS=8`, one binary set:
   * **A**: `htp_moe` head, switch off (the reference, first);
   * **B**: the same binaries, `NNTR_HTP_FORWARD=1`;
   * **B-prof**: B at `NNTR_HTP_PROFILE=2`, G=64 once, to read the `graph:`
     line and the M==1 transport against A's level-2 run;
   * optional **A-prof**: A at level 2, G=64.

   Expected: B within A's drift on tok/s (§0: 0 ± 0.2 ms/token), M==1
   `dsp=` +≤ 10 µs/call (loop + 2 × 8 KiB copies), transport −0..−10 µs/call,
   text B ≡ A at every G, prefill within −5 % (untouched path). A B that is
   *slower* than A by more than drift is the finding to look for. The
   verdict is the plumbing's cost, not a gain. Per rule 22 the handoff names
   commit + build lines, not staging paths.

## 5. Dependencies on open PRs

* **#108** (#101/#102, GEMV default on; `htp_moe_opts.h`, edits in
  `sendMoeOptsOnce` and the M==1 row) and **#107** (#105, GEMV kernel
  internals, `run_host_checks.sh` loop over `HVX_GEMV_PF_LEAD_KB`). There is
  **no functional dependency**: the MoE op calls `hexkl_mm_u8i4_moe_layer_run`
  with the session's `moe_flags`, whatever the default, and the host check
  covers both flag values. The dependency is **textual**: both PRs edit
  `htp_compute_ops.cpp` and `run_host_checks.sh`. Implement after both
  merge, or rebase onto whichever lands first. Do not start in parallel on a
  branch off `9fcbbc20`. After #108 the handoff's A carries the GEMV path,
  which is what we want.
* **#81 / #82**: not required. `ATTN_M1` and the small-op kinds are defined
  and non-resident. Each of those issues then adds a table slot and a
  builder `resident` bit, with no IDL change.
* **#110** (`QS4CX_WH_HAD`): the HAD opts bit travels in `moe_flags`, so the
  graph path inherits it through `sendMoeOptsOnce`. No interaction beyond
  that.

## 6. Risks (host-vs-device gaps)

* **Stale skel / stale stub.** This is a new IDL method set. An old skel
  gives `0x8000040E` (rule 3), and the local ARM stub is already stale (§2).
  The handoff's first B command is one decode token with the switch on. The
  `[HTP] graph: init n_ops=<n>` stderr line proves the skel knows the
  methods, and B voids if it is missing.
* **DVFS / thermal drift.** The effect is ≤ 0.5 %, well inside rule 9 / 20
  drift, so the B verdict is read from the level-2 `dsp=` and `transport`
  columns of B-prof vs A-prof in the same sitting (rule 23), not from tok/s.
* **Poll window.** The call pattern is unchanged, so the 5 ms poll (rule 23
  corollary) behaves as in A.
* **Address space.** ≈ 62 KiB of heap, no mapping, so rule 8 is not
  approached.
* **Call-order binding of handles.** MoE ops get their handles from the
  first pass's MoE call order. A model variant whose MoE layers run out of
  order, or a slim/cached MoE layer (`lfm2_moe_layer_{fsu,cached}.cpp`, which
  never call the layer kernel), would bind wrongly or not at all.
  `graph_init` validates the shapes per op, and the ARM refuses to init
  unless the observed MoE count equals the list's. Both are covered by a
  host mutation (swapped layers → `AEE_EINVALIDFORMAT` when inter differs;
  when shapes match, the ARM count check).
* **Baseline host checks** were not re-run in this planning session: the
  run was killed (exit 137, resource limit) before finishing. `ninja -C
  build` is green at `9fcbbc20`. The implementer runs `run_host_checks.sh`
  first to confirm the baseline.

## 7. Docs to update

* **BENCHMARK.md**: nothing in this issue. The later handoff adds an
  "#85 B (per-token entry, MoE only)" row next to its A, tagged with the
  serial.
* **LEDGER.md**: ⑨ records "skeleton landed (PR #…), resident = MOE, 22
  calls/token unchanged; 0 ms by construction". It also gets a new open item
  (supervisor's call): **the resident set that actually removes calls is
  larger than #81 + #82.** Every op between two MoE layers must be resident:
  the M=1 FCs (conv in_proj/out_proj, q/k/v/o, and the dense FFN for layers
  0–1), router FC + top-k, and, for the tail, final norm + lm_head. The FC
  weights are Q4_0 on the NPU model, so either a Q4_0 HVX GEMV on the DSP or
  a re-quantization of the FCs to `QS4CX_WH`. The latter changes the weights
  vs the CPU control and needs the user's say on the accuracy gate. Without
  that, ⑨'s −5 ms cannot be reached.
