# 81 — Decode attention at m=1 with a DSP-resident KV cache: `_det` spec, HVX kernel, host bit check, IDL entries

Issue: dlwlzzero/nntrainer#81 (LEDGER §4 lift 1 → §3 ⑨; tracker #76).
Read against `htp_moe` @ `63c10d5e`. Aligned with `docs/plans/85-per-token-entry-skeleton.md`
(the `ATTN_M1` kind, `pos` in `forward`) and `docs/plans/82-m1-small-ops.md` (the `_det` spec
pattern, the `hvx_emu/` host bit check, q/k norm and RoPE as separate ops that run *before* this one).
This issue fills #85's `ATTN_M1` kernel slot. It does not wire it: the slot stays NULL here.

**What this buys, plainly: 0 ms/token by itself.** `mha_core` costs the ARM 2.30 ms/token on the
NPU `--profile` run (LEDGER §2 ①, inflated by the profile build; in TPS terms the whole
outside-the-call time is ≈ 9.6–9.9 ms at G=512). Its value is residency: attention with its cache
is the largest single op between two MoE calls in the 6 attention layers, and plan 85 §0 says the
call count falls only when every op in that stretch is resident. What the DSP will pay once wired,
bytes only (the arithmetic is small): 6 layers × 4 KiB per position of f32 K+V read once per token
= **12.6 / 25.2 / 37.7 MB/token at pos 512 / 1024 / 1536**, i.e. ≈ 0.5 / 1.0 / 1.5 ms at the
21–27 GB/s direct HVX read (rules 26, 27) — about the CPU's own cost, not a gain. fp16 halves it
(§3.1, the upgrade path).

## 1. Goal and gate

Host gates only. The issue ends in a PR into `htp_moe` with contract §9 gates 0–3. No device number
is claimed; the device check of gate (a) is a ride-along in a later sitting (§4 step 6).

1. **Bit identity, host.** A new `test/htp/host/attn_m1_host_check.c`, run from
   `run_host_checks.sh`, compiles the **real HVX source** `hvx_attn_m1_f32.c` on the `hvx_emu/`
   intrinsic emulation (§3.3) and the **real pthread worker pool** (`stub/qurt.h`), and `memcmp`s
   the kernel's output against the scalar spec `attn_m1_det.h` for context lengths
   **L = 1, 63, 64, 65, 512, 1024** (pos = L − 1), at the LFM2.5 shape (32 q heads, 8 kv heads,
   gqa 4, head_dim 64, `max_seq` 1024 for the check; 2048 in one case), random f32 q/k/v from a
   fixed seed in [−4, 4], plus three fixed rows (all zero, all subnormal ±1e-39 — rule 24, no FTZ
   on either side — and large |x| ≈ 1e3 so the softmax hits its clamp). Each length runs with the
   pool at **0, 3 and 7 workers** and the three outputs must be byte-equal (the kv-head split is
   deterministic by construction; this proves it). It prints `ATTN M1 BIT-IDENTICAL` with `bad=0`
   per case and the per-head `(max, sum)` stats of the first divergence otherwise.
   Two structural cases: (i) **append-chain ≡ bulk**: L calls of `forward` from an empty cache at
   pos 0..L−1 must leave the cache byte-equal to one `kv_append` of L rows, and the last output
   byte-equal to a `forward` on that cache; (ii) **L = 1 is the identity**: `out == v` bit for
   bit for every head (e = [1.0], l = 1.0, `recip_det(1.0) == 1.0f`, §3.2).
2. **Tolerance, host.** The same check compares the spec with a **double reference** (plain
   softmax(q·Kᵀ·scale)·V in double) and prints `max_abs_err / max|V|` per length. It asserts
   ≤ 2⁻¹³ (1.2e-4) at L ≤ 1024: the f32 sums over ≤ 1024 terms and the `_det` exp/recip are each
   under 1e-6 relative, so a miss here is a spec bug, not rounding. It is a sanity bound; the bit
   identity is the contract (rule 25).
3. **Standing host gates.** `clang-format-14` on changed lines; `ninja -C build`; `*qs4cx*`
   cpu-backend gtests; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6 (fixture
   generated, none skipped); `tools/htp_syntax_check.sh`; `run_host_checks.sh` → `ALL CHECKS
   PASS`, `WORKER POOL LANES OK`, `ATTN M1 BIT-IDENTICAL`.
4. **Skel and app.** `generate_stub.sh`; `test/htp/build.sh` (`-Wall -Werror`, `UNDEFINED SYMBOLS
   OK`), md5; `build_android.sh --htp`, `readelf -d libnntrainer.so` lists `libsdkl.so` +
   `libcdsprpc.so`; `ndk-build unittest_hvx_attn` (it now holds `HvxAttnM1.*`); md5s in the PR.
5. **Prefill gate: nothing on the prefill path changes, provable from the diff.**
   `git diff --stat origin/htp_moe...` lists no file under `Applications/`, no
   `htp_compute_ops.cpp` / `htp_backend.cpp`, no `hmx/*`, no `cpu_backend/*`, and **not**
   `test/htp/nntr_hvx_attn.c` or `hexkl_attn_u8.*` (the prefill entries stay as they are). The IDL
   change is additive: methods appended after the last existing one, earlier indices unchanged.
   The E2E gates (prefill ≥ −5 % of variant A, text B ≡ A per plan 85 §1.4, text vs the CPU run
   — see §3.6) are carried to the wiring issue's handoff. No model path calls these entries here.

## 2. Where it lives

| file | change |
|---|---|
| `nntrainer/tensor/attn_m1_det.h` (new, C99, header-only, C++-includable) | the scalar spec of §3.2. Same place and pattern as `swiglu_det.h` / plan 82's `m1_ops_det.h`; it reuses `swiglu_det_exp` and `swiglu_det_recip` (`nntrainer/tensor/swiglu_det.h:113`, `:170`), which are the normative scalar twins of `hvx_exp_det_sf` / `hvx_recip_det_sf` (`htp_backend/hvx/hvx_swiglu_det.h`). It sits outside `htp_backend/` so the ARM side can compile it later (doc 45 §3.3) and the device gtest can include it (`test/jni/Android.mk:1017` already has `nntrainer/tensor` on the path) |
| `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.{c,h}` (new) | the cache object `hvx_attn_m1_ctx` (`create / free / kv_append / forward`) and the HVX kernel: per-kv-head worker on `hvx_worker_pool_run` (`hvx_worker_pool.h:52`), `HVX_UVector` for the FastRPC-side q/k/v/out, aligned loads for the `memalign(128)` cache. The `_det` arithmetic is `hvx_exp_det_sf` / `hvx_recip_det_sf`, the lane reductions are `reduce_max_sf` / `reduce_sum_sf` (`hvx_softmax_util.h:67-75`) taken at lane 0 |
| `test/htp/nntr_hvx_attn_m1.c` (new) | skel entries of §3.4; one FARF line per register |
| `test/htp/nntr_hvx.idl:449-450` | four methods after `moe_set_opts` (after #85's and #82's methods if they landed first) |
| `test/htp/nntr_hvx_session.h:52-64` | `hvx_attn_m1_ctx *attn_m1;` (NULL = none; the session is calloc'd), like #85's `graph` |
| `test/htp/hvx_add_f32.c:184-201` `nntr_hvx_close` | free the cache before `hvx_worker_pool_destroy` (the ctx holds the pool pointer, as `g_attn[i].pool` does at `nntr_hvx_attn.c:78`) |
| `test/htp/build.sh:77-89` `SRCS` | add `nntr_hvx_attn_m1.c` and `$BACKEND/hvx/hvx_attn_m1_f32.c` (rule 17) |
| `test/htp/host/stub/AEEStdErr.h` | `AEE_EBADSTATE` (`0x80000400 + 0x0D`), `AEE_EINVALIDFORMAT` (`+ 0x11`) with the SDK's values if #85 has not added them |
| `test/htp/host/hvx_emu/` | plan 82's emulation, extended by the ≈ 8 intrinsics of §3.3 that attention adds; created here if #81 lands first |
| `test/htp/host/attn_m1_host_check.c` (new) + `run_host_checks.sh` (append after the `hvx_emu` block, or after `:77` if first) | gates 1–2 |
| `test/unittest/unittest_hvx_attn.cpp` (after `HvxAttnScores`, fixture `HtpSession` `:54-78`) | `HvxAttnM1.{RejectsBadShapes, MatchesDetSpecBitExact, AppendChainEqualsBulk, PerLayerCost}` (§4 step 5). `Android.mk:1007-1030` gains nothing but the include is already there |

Consumers of a changed contract:

* **IDL + both stubs.** `generate_stub.sh` (ARM) and `test/htp/build.sh` (skel) together (rule 3).
  `HtpComputeOps` does not call the new methods in this issue. `tools/htp_syntax_check.sh` builds
  its declarations from `htp_compute_ops.cpp`, so it is unaffected.
* **Quantizer format tag (`nntr_quantize_stream`), loader check: unchanged.** No weight bytes or
  dtypes change; the cache holds activations.
* **`NNTR_HTP_PROFILE` stage tables, `tools/htp_fc_report.py`: unchanged.** No model path runs the
  op. Its per-op pcycles arrive with #85's `forward_debug` once the slot is filled.
* **#85 (functional, one way).** The `ATTN_M1` op wrapper of the wiring issue calls
  `hvx_attn_m1_forward(s->attn_m1, ordinal, pos, 0.125f, q, k, v, out)` with q/k/v sliced from the
  op's input slot (3072 f32 = q 2048 | k 512 | v 512, the fused qkv layer's row, upstream
  `1f538c20`) and `out` 2048; op record K = 3072, N = 2048; `ordinal` = the count of `ATTN_M1` ops
  before it (the validator can compute it from `layer_kind[]`). `pos` is #85's `forward` argument,
  checked `< max_seq` on both sides. `scale` is a constant of the model (1/√64 = 0.125, exact in
  f32), so #85's signature needs no new argument. **Textual:** both append to the IDL,
  `build.sh` `SRCS`, `nntr_hvx_session.h`, `nntr_hvx_close`, `stub/AEEStdErr.h` and
  `run_host_checks.sh`. Rebase onto whichever lands first.
* **#82 (textual).** `hvx_emu/` and the IDL. No functional dependency: q/k norm and RoPE run
  before this op and are not inside it (the CPU applies RoPE *inside* `mha_core`,
  `mha_core.cpp:725`, `:747`; on the DSP they are #82's `ROPE` op, so this kernel takes post-RoPE
  q and k).

## 3. Design

### 3.1 KV cache: f32, DSP heap, fixed at register, own object

**Layout** (hvx_impl's, `hvx-attn.c` header comment): per `(layer, kv head)`,
`Kt[head_dim][max_seq]` f32 — one 128-B vector is 32 consecutive positions of one head dim, so
the score loop is 64 `vmpy + vadd` per 32 positions with a splat of `q[d]` — and
`V[max_seq][head_dim]` f32, one position = two vectors. `layer` is the attention-layer ordinal
0..5, `max_seq` must be a multiple of 32 (`AEE_EINVALIDFORMAT` otherwise). Per-worker score
scratch `gqa × max_seq` f32 per lane, from the ctx (`n_lanes = pool workers + 1`).

**Dtype: f32.** Decided by the gate, not by bytes. A `_det` spec is a sequence of IEEE f32
operations; an fp16 cache adds a narrowing whose rounding on v79 (`Q6_Vhf_equals_Vsf` under
`-mhvx-ieee-fp`) has no host-confirmed spec yet (rule 24 confirmed add/sub/mul only), and the
CPU's own cache is fp16 anyway (`mha_core.cpp:232-238`, `:346-354`), so no dtype makes the DSP
match the CPU bit for bit. LEDGER's silicon rule "compute in fp32 inside an op, narrow once"
narrows *nothing* here. **Bytes**, 6 layers × 8 heads × 64 × 2 (K, V) × `max_seq`:

| `max_seq` | f32 | fp16 | u8 (prefill cache dtype) |
|---|---|---|---|
| 1536 (prompt 512 + gen 1024, the issue's floor) | 36 MiB | 18 MiB | 9 MiB |
| **2048 (`nntr_config.json` `max_seq_len`, the value used)** | **48 MiB** | 24 MiB | 12 MiB |
| 4096 | 96 MiB | 48 MiB | 24 MiB |

48 MiB is 26 % of the ≈ 182 MiB DSP heap (rule 8; #82 adds 1.8 MiB, #85 62 KiB). It fits with
room; 4096 would not be comfortable. `ponytail:` f32 caps `max_seq` at ≈ 2048–3072 in this
address space and reads 2× the bytes of fp16 per token (§0). Upgrade: fp16 K/V with a
device-confirmed `sf → hf` RNE spec (one narrowing at append, the score loop widens on load,
as hvx_impl's `hvx_vec_mpyacc_f32_f16`), 24 MiB and half the read; the spec changes by one
operation per stored element.

**Residency: heap, not VTCM.** Each K/V byte is read once per token (the four q heads of a kv
head share one Kᵀ/V stream), so staging into VTCM by DMA buys nothing a direct read does not
(rule 27: arena and cached heap read alike from HVX), and VTCM (8 MB) cannot hold 48 MiB. If the
wiring issue's per-op pcycles show the read is DDR-latency-bound like the GEMV (rule 26), the
lever is an `l2fetch` lead (rule 31; hvx/58's variant C, never measured) — not this issue.

**Growth policy: none.** Allocated at `attn_m1_register` for `max_seq` (calloc, so lanes past the
context are finite zeros — they are masked anyway), freed at release/close. `pos ≥ max_seq` →
`AEE_EINVALIDFORMAT`; `pos > kv_len[layer]` (a hole) → `AEE_EBADSTATE`; `pos ≤ kv_len` writes
position `pos` and sets `kv_len = pos + 1` (a rewind is allowed, matching `cache_index` resets).
The Kᵀ stride is `max_seq`, so growth would be a re-layout; the number is known at model load.

**Own object, not the prefill cache.** The existing `attn_register / attn_kv_append /
attn_forward` (`nntr_hvx_attn.c`, `hexkl_attn_u8.h`) hold u8 per-block-quantised Kt/V registered
as HMX weights with fp16 shadows, re-quantise the tail block on every append, and are called by no
app path (only `unittest_hvx_attn.cpp`). Sharing would put a u8 quantiser in front of every decode
score (no `_det` bit gate possible) and pay a block registration per layer per token. They stay
untouched (gate 5). Seeding this cache from the CPU prefill (or, later, from a DSP prefill) is the
wiring issue's concern: one `attn_m1_kv_append` per layer after the prompt, 512 rows = 2 MiB per
layer (§6 open item: the CPU cache is fp16 on Android, so the seeded rows are fp16-rounded while
decode rows are f32).

### 3.2 The `_det` spec (`attn_m1_det.h`)

Every step is one IEEE f32 operation, RNE, no FTZ (rule 24), no FMA, no qf32, no divide, no libm.
Order is part of the contract. Per call `(layer, pos, scale, q[32×64], k[8×64], v[8×64])`,
L = pos + 1, and per q head `hq` with kv head `h = hq / 4`:

1. **Append**: `Kt[layer][h][d][pos] = k[h·64 + d]`, `V[layer][h][pos][d] = v[h·64 + d]` (copies).
2. **Scores**, per position block of 32 lanes, lane j = position p = 32b + j:
   `acc = 0; for d = 0..63 ascending: acc = acc + (q[hq·64 + d] * Kt[d][p])` — a multiply then an
   add, each rounded. Then `s[p] = acc * scale`. Lanes p ≥ L in the last block are computed and
   discarded (they never enter a reduction).
3. **Max**: `m = max over p < L of s[p]`. Exact and order-free (IEEE max on finite values), so the
   HVX lane-wise `vmax` over blocks (masked lanes −inf) + `reduce_max_sf` tree is the spec.
4. **Exp**: `e[p] = exp_det(s[p] − m)` — one subtract, then `swiglu_det_exp` (clamps at
   [−88, 85]; the argument is ≤ 0, `exp_det(0) == 1.0f` exactly, so `e[argmax] = 1`).
5. **Sum**: `acc[j] = 0; for b ascending: acc[j] = acc[j] + e[32b + j]` (masked lanes add 0),
   then the pair tree `(j, j+16), (j, j+8), (j, j+4), (j, j+2), (j, j+1)` **taking lane 0** —
   what `reduce_sum_sf` (`vror` 64/32/16/8/4 bytes + `vadd`) leaves in lane 0. Same rule as plan 82
   §3.1 step 2; IEEE add is commutative, so only the association matters.
6. **Reciprocal**: `r = recip_det(l)`. Domain: `l ≥ 1` because `e[argmax] = 1`, and
   `l ≤ L ≤ max_seq`, far below the seed's 1.6e38 edge. `recip_det(1.0f) == 1.0f` (three
   Newton steps from the seed 0.949 land within 5e-11, which rounds to 1).
7. **PV**: per lane d (two vectors of 32 per head): `o[d] = 0; for p = 0..L−1 ascending:
   o[d] = o[d] + (e[p] * V[p][d])`. Then `out[hq·64 + d] = o[d] * r`.

**Domain.** Scores are finite for |q|, |k| < ~1e17; a row outside that is a broken residual
stream. `scale = 0.125` makes step 2's last multiply exact, but the spec keeps it as an operation
so a non-power-of-two head_dim would still be specified. The code comment cites LEDGER §4's
clamp rule (inherited from `exp_det`) and "fp32 inside, narrow once" (nothing narrows).

**Not bit-identical to the CPU path, and known.** On Android the CPU runs this layer in fp16
(`mha_core.cpp:346-354` copies Q/K/V steps to FP16; `compute_kcaches<uint16_t>`,
`softmax_triangle`, `compute_fp16vcache_transposed`, `:774-779`) with its own exp; on the host
FP32 build it is `compute_kcaches_fp32_reference` with a different accumulation. No DSP dtype
choice changes that. See §3.6.

### 3.3 Host bit check: real HVX source on the intrinsic emulation, real pool

As plan 82 §3.2: the only host check that tests the *kernel* rather than a second copy of the
spec compiles `hvx_attn_m1_f32.c` unmodified on `hvx_emu/` (`HVX_Vector` = 32 × int32 struct,
one `volatile` f32 op per lane) with `-std=c99 -O2 -ffp-contract=off`, no `-ffast-math`. Beyond
plan 82's list it needs `Q6_Vsf_vmin/vmax_VsfVsf`, `Q6_Vsf_equals_Vw` (int → f32 numeric convert,
exact for |k| ≤ 2²⁴), `Q6_Vw_vaslacc_VwVwR`, `Q6_Vw_vasl_VwR`, `Q6_Q_vcmp_gt_VwVw` (+ the
`HVX_VectorPred` type), `Q6_V_vmux_QVV`, `Q6_Q_vsetq_R` or an equivalent lane mask for the tail
block, and `Q6_R_vextract_VR` — all bit-exact integer/select ops except `equals_Vw`, which is a
single exactly-representable conversion. The premise is unchanged: one `Vsf` op = one IEEE RNE op
without FTZ, device-confirmed for add/sub/mul by `HvxSwigluDet.MatchesScalarBitExact` and rule 24;
`vmin`/`vmax` on finite values are exact by definition. The device gtest re-checks the premise for
this code (§4 step 5–6).

The pool is the real `hvx_worker_pool.c` on `stub/qurt.h` (as `worker_pool_host_check`), so the
kv-head split, the per-lane scratch and the "index 0 runs inline on the caller" convention
(opposite to hvx_impl's `wp_run`, LEDGER §4 last paragraph) are exercised at 0 / 3 / 7 workers.

**Rejected: hvx_impl's arithmetic as it stands** (`hvx-attn.c` on `hvx_impl` and the hvx/58
branch). Its scores are `hvx_vec_mpyacc_f32_f16` widening fp16 products, its exp is the
llama.cpp qf32 `hvx-exp.h`, its probabilities are narrowed to fp16 before PV, its 1/sum is a
scalar `1.0f / sum` (libm-free but the qf32 `vmpy` after it is not), and the hvx/58 position-block
merge re-associates the softmax. None of that can meet a `_det` bit gate, the same reason plan 82
rejected `hvx-rmsnorm.c` / `hvx-rope.c`. **What is reused:** the Kᵀ `[head_dim][max_seq]` layout
and the "one vector = 64 (here 32, f32) positions" score loop shape, the per-layer/per-head cache
indexing, the split by kv head with per-worker score scratch, the fp32 vector-pair accumulation of
PV (hvx_impl's own finding that chained qf32 adds break on v79), and the GQA fusion of hvx/58
(one Kᵀ/V stream serves the 4 q heads of a kv head — here as four accumulator sets in one pass,
which changes no arithmetic). **Not reused:** the token loop (m = 1 only), the hvx/58
position-block split + merge (`ponytail:` 8 kv-head units over ≤ 6–7 lanes leave 1–2 lanes with
two heads; the upgrade is hvx/58's flash-decoding merge, which needs its own `_det` for the merge
and a second spec case), the `HTP_ATTN_L2FETCH` knob (a later lever, §3.1), and `hvx_op_attn`'s
exec-ctx/oplist plumbing (#85 owns the op table).

Also rejected: the Hexagon tools' `libnative` emulation (used by `gemv_native_check` for integer
`vrmpy`) as the gate here — its `Vsf` rounding and FTZ behaviour are not documented, and the gate
must rest on the stated premise explicitly.

### 3.4 IDL (additive)

```
// [#81] Decode attention at m=1 against attn_m1_det.h, with a session-owned
// f32 KV cache (Kt [layer][kv][head_dim][max_seq], V [layer][kv][max_seq][head_dim]).
// layer = attention-layer ordinal. Not wired into the model here (#85 slot).
AEEResult attn_m1_register(in uint32 n_layers, in uint32 n_kv, in uint32 gqa,
                           in uint32 head_dim, in uint32 max_seq);
AEEResult attn_m1_release();
AEEResult attn_m1_kv_append(in uint32 layer, in uint32 kv_from, in uint32 n_rows,
                            in sequence<float> k_rows, in sequence<float> v_rows);
AEEResult attn_m1_forward(in uint32 layer, in uint32 pos, in float scale,
                          in sequence<float> q, in sequence<float> k,
                          in sequence<float> v, rout sequence<float> out,
                          rout sequence<float> stats);
```

One cache per session (like #85's `graph`): a second `register` returns `AEE_EBADSTATE`. `k_rows`
/ `v_rows` are `[n_rows][n_kv][head_dim]` f32 (the CPU cache layout, `hexkl_kv_quant.h` comment).
`stats` is empty or `2 × n_q` floats `(m, l)` per q head, the localising intermediate
(`swiglu_det_f32`'s reason for returning `exp_out` / `recip_out`); production passes it empty and
pays nothing. Bad shapes → `AEE_EINVALIDFORMAT`, holes / no cache → `AEE_EBADSTATE`, never
`AEE_EBADPARM` (the stale-skel symptom, rule 3, plan 85 §1). The entries use the caller's FastRPC
buffers for q/k/v/out (`HVX_UVector`) and the session heap for the cache.

### 3.5 Rejected alternative: per-token append-only entry without `kv_append`

The issue text names only "q, k and v of the new position". Without a bulk append the host check
and the device gtest would need 1024 `forward` calls to reach L = 1024, and the wiring issue would
have no way to seed the prompt's 512 positions from the CPU prefill except 512 calls per layer.
One extra method, 20 lines, and gate 1's append-chain ≡ bulk case proves the two agree.

### 3.6 The accuracy gate: a user decision to surface, not decide here

Contract §1 gate (c) is "text identical to the CPU run"; plan 85 §1.4 already reads it as
**B ≡ A** because A's text differs from the CPU `q40` run. For attention the gap is structural:
the CPU decodes this layer in fp16 with a different exp and accumulation (§3.2), so once #81 is
wired, B's tokens will diverge from A's at some G for that reason alone, exactly as plan 82 §5
warns for the small ops. Three ways to read the wiring handoff, for the user (via the supervisor):

* (i) keep **text B ≡ A** — expect it to fail on attention layers; a fail then says nothing about
  the kernel (gate (a) passed);
* (ii) a **PPL column** (`NNTR_PPL`, upstream `32b46e32`, LEDGER ⑱ item (d)) with a band, as
  hvx_impl's #58 used (± 0.3 %); rule 25 says accuracy verdicts come from full-model columns;
* (iii) switch the **CPU path to the same `_det`** (doc 45 §3.3: one spec, scalar/NEON/HVX) — a
  `mha_core` change that moves the CPU control and the prefill path.

This plan builds (a) only; the choice binds the wiring issue's handoff, not this PR.

## 4. Steps

1. **Spec + tolerance half.** `attn_m1_det.h` and the spec-only part of `attn_m1_host_check.c`
   (double reference, the L = 1 identity, `recip_det(1.0f) == 1.0f`). Gate: `run_host_checks.sh`.
2. **Cache object + HVX kernel.** `hvx_attn_m1_f32.{c,h}` with the §3.1 layout, the §3.2 order
   and the rule citations in comments; `hvx_emu/` extended (or created, if #82 has not landed).
   The check's bit half: six lengths × three fixed rows × pool 0/3/7, append-chain ≡ bulk. Gate:
   `run_host_checks.sh` prints `ATTN M1 BIT-IDENTICAL`; `clang-format-14`.
3. **IDL + skel.** `nntr_hvx_attn_m1.c`, the session field, the close order, `build.sh` `SRCS`,
   stub codes. Gate: `generate_stub.sh`, then `test/htp/build.sh` (`-Wall -Werror`, `UNDEFINED
   SYMBOLS OK`), md5.
4. **Host build gates.** `ninja -C build`, `*qs4cx*`, `*Lfm2Moe*` 6/6, `tools/htp_syntax_check.sh`.
5. **Device gtest + app.** `HvxAttnM1.*` in `unittest_hvx_attn.cpp`: `RejectsBadShapes`
   (`AEE_EINVALIDFORMAT + kDspOffset`, never `0x8000040E`), `MatchesDetSpecBitExact` (the six
   lengths, `memcmp` against `attn_m1_det.h` compiled into the gtest, `stats` on mismatch, and
   the double-reference error printed as `ATTN_M1_FIELD` — printed, not asserted, rule 25),
   `AppendChainEqualsBulk`, `PerLayerCost` (`ATTN_M1_FIELD pos=<L> us=<t>` per length, printed).
   Gates: `build_android.sh --htp`, `readelf -d`, `ndk-build unittest_hvx_attn`, md5s, the gate-5
   diff check. PR into `htp_moe`, `state:review`. **The issue ends here.** No E2E sitting is
   needed for this issue alone: nothing it adds runs in the model.
6. **Device, later (unavoidable for gate (a) on silicon; not this issue).**
   * **Ride-along in the next sitting that has room** (e.g. #117's): no E2E variant (A's cells
     cannot change). `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` once on that sitting's
     binaries. Expected `bad=0` at all six lengths; a `bad > 0` with the host at 0 is the
     emulation premise failing for `vmin`/`vmax`/`equals_Vw`/the tail mask, and `stats` says
     whether it is the softmax or the PV half. `PerLayerCost` at L = 512 / 1024 gives the first
     read of §0's 0.5 / 1.0 ms estimate (DDR-latency-bound or not, rule 26).
   * **The wiring issue's handoff** (full E2E, prompt 512, gen 64 / 512 / 1024, ×2,
     `NNTR_NUM_THREADS=8`, ≤ 4 variants): **A** = `htp_moe` head unchanged (first);
     **B** = the same binaries with #85's switch and the attention kind resident;
     **B-prof** = B at `NNTR_HTP_PROFILE=2`, G=64, for the per-op pcycles; optional **A-prof**.
     Read prefill ≥ −5 % of A, text per §3.6's decision, `ATTN_M1` pcycles vs the ≈ 0.5–1.5 ms
     estimate.

## 5. Risks (host-vs-device gaps)

* **The emulation premise** now also covers `vmin`/`vmax` (exact by definition), the tail-lane mask
  and `Q6_Vsf_equals_Vw` (one exact conversion). The overflow/inf encodings of rule 24's open case
  are kept out by the domain; the ride-along gtest is the check.
* **Compiler contraction.** Host: `-ffp-contract=off` and `volatile` per op; skel: `-O3` without
  fast-math, HVX has no sf FMA. The device gtest is the check for the latter.
* **Alignment.** q/k/v/out are FastRPC buffers with no vector alignment (`HVX_UVector`); the
  cache and scratch are `memalign(128)` and may use aligned loads. The emulation cannot catch a
  misaligned aligned-load, so this is a review item (plan 82 §5 has the same one).
* **Uninitialised lanes.** Positions past the context in the last 32-lane block are read; calloc
  keeps them finite and the mask keeps them out of the max/sum. A NaN anywhere in a live lane
  would propagate through `vmax` — the domain excludes it and the subnormal/large fixed rows
  probe the edges.
* **Address space.** 48 MiB of heap at `max_seq` 2048 (26 % of ≈ 182 MiB, rule 8). No mapping,
  no VTCM, no DMA. The device gtest registers at 2048 once to prove the allocation.
* **Lane count and HVX contexts.** The device pool has `n_hvx − 1` workers (`hvx_add_f32.c:113-118`;
  hvx_impl's v79 unit ran 6 lanes); 8 kv-head units over 6 lanes leave two lanes with two heads.
  Bit identity does not depend on the count (gate 1's 0/3/7 proves it); only the cost does.
* **DMA rate, DVFS, thermal drift.** Not exposed here: no timing is claimed. They bind the wiring
  handoff, which reads per-op pcycles (rules 20, 23) and a same-sitting A/B, not tok/s alone. The
  KV read at 21–27 GB/s is a latency-bound direct read (rule 26); if `PerLayerCost` lands well
  above §0's estimate, the `l2fetch` lead is the next lever, not a redesign.
* **Stale skel/stub.** Four new methods; `RejectsBadShapes` runs first and expects
  `AEE_EINVALIDFORMAT`; `0x8000040E` is recorded as a stale skel (rule 3), not a kernel failure.
* **Text identity when wired.** §3.6. Until the user decides, the wiring handoff's text gate can
  fail for the attention path's rounding alone.
* **Baseline.** `run_host_checks.sh` was not re-run in this planning session (read-only
  exploration on `htp_moe_cycle`); plan 82 recorded it green at `485435ce` in 38 s. The
  implementer runs it first.

## 6. Docs to update

* **BENCHMARK.md:** nothing in this issue. The ride-along records one `HvxAttnM1` gtest line
  (bad counts, `ATTN_M1_FIELD` cost at 512 / 1024, serial) under that sitting.
* **LEDGER.md:**
  * §4 lift 1: mark done. Record what was lifted (Kᵀ layout, kv-head split, per-worker scratch,
    fp32 PV accumulation, hvx/58's GQA fusion) and what was not (fp16 widening scores, qf32 exp,
    fp16 probabilities, scalar `1/sum`, the position-block merge, `HTP_ATTN_L2FETCH`), and why
    (the `_det` bit gate).
  * ⑨: the m=1 attention kernel and its cache exist (PR #…), unwired, 0 ms; the wiring issue
    needs `attn_m1_register` at graph init and one `attn_m1_kv_append` per layer after the CPU
    prefill.
  * **New open items (supervisor's call):** (1) the accuracy-gate reading for the wiring handoff,
    §3.6, a user decision; (2) KV seeding from the CPU prefill: the Android CPU cache is fp16 after
    RoPE inside `mha_core`, so the seeded 512 rows are fp16-rounded and the decode rows f32 — fine
    for a PPL gate, one more reason (i) of §3.6 cannot hold; (3) fp16 cache as the upgrade once an
    `sf → hf` RNE spec is device-confirmed (halves 37.7 MB/token at pos 1536, and `max_seq` 4096
    fits).
