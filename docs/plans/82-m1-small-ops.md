# 82 — M=1 small ops: RMSNorm, q/k norm, RoPE (head_dim 64), conv1d + gating, each `_det` + HVX + host bit check

Issue: dlwlzzero/nntrainer#82 (LEDGER §4 lift 5 → §3 ⑨; tracker #76).
Read against `htp_moe` @ `485435ce`. Aligned with `docs/plans/85-per-token-entry-skeleton.md`
(`efcd7f32`). This issue fills the kernel side of #85's kinds `RMSNORM`, `QK_NORM`, `ROPE` and
`CONV1D_GATE`. It does not wire them: #85's table slots stay NULL here.

**What this buys, plainly: 0 ms/token by itself.** These ops cost the ARM
≈ 0.46 ms/token today (BENCHMARK `--profile` decode per type, NPU run: `rms_norm` 0.19,
`custom_multiply` 0.15, `reshaped_rms_norm` 0.06, `causal_conv1d` 0.06). Their value is residency.
Plan 85 §0 says the call count falls only when every op between two MoE calls is resident. Conv
layers are 18 of the 24, so the conv block's norm → conv1d + gate path is the most common stretch
between two MoE calls. It still needs the M=1 FCs (plan 85 §7).

## 1. Goal and gate

Host gates only. The issue ends in a PR into `htp_moe` with contract §9 gates 0–3. The device check
of gate (a) is a ride-along in a later sitting (§4 step 6).

1. **Bit identity, host.** A new `test/htp/host/m1_ops_host_check.c`, run from
   `run_host_checks.sh`, compiles the **real HVX sources** against an intrinsic emulation header
   (§3.2). It compares them with `memcmp` against the scalar `_det` spec and prints
   `M1 OPS BIT-IDENTICAL` with `bad=0` per case:
   * RMSNorm, n = 2048, eps = 1e-5;
   * q/k norm, 32 × 64 and 8 × 64 (one shared 64-float gamma, as `reshaped_rms_norm`);
   * RoPE, 32 q + 8 k heads × 64, rows for positions 0, 1, 511, 1023 and 4095, theta 5e6
     (`config.json` `rope_parameters.rope_theta`). Position 0 must also be the identity bit for bit;
   * conv1d + gate at M=1 (C = 2048), a chain of 8 tokens from a zero state and from a state that
     a 7-row prefill-shape call left behind. The chain must be bit-identical to one prefill-shape
     `hvx_conv_gate_f32` call over the same 8 rows.

   Inputs are random f32 from a fixed seed, plus three fixed rows: all zeros, all subnormal
   (±1e-39, rule 24: HVX keeps subnormals, so neither side may flush), and large values (|x| ≈ 1e4).
2. **Tolerance, host.** The same check compares each spec output with a **plain fp32 reference**
   (straight C: `1.0f / sqrtf(sum / n + eps)`, `a*c - b*s`, CPU-order conv) and with a double
   reference. It prints `max_ulp`. It asserts RMSNorm ≤ 4 ulp of |y|, RoPE ≤ 2⁻²² · (|a| + |b|),
   and conv ≤ 2⁻²¹ · |b| · Σ|wᵢ·gᵢ|.
3. **Standing host gates.** `clang-format-14` on changed lines; `ninja -C build`;
   `*qs4cx*` cpu-backend gtests; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6 (fixture
   generated, none skipped); `tools/htp_syntax_check.sh`; `run_host_checks.sh` →
   `ALL CHECKS PASS`, `WORKER POOL LANES OK`, `M1 OPS BIT-IDENTICAL`.
4. **Skel and app** (the IDL gains three test methods, §3.3). `generate_stub.sh`;
   `test/htp/build.sh` (`-Wall -Werror`, `UNDEFINED SYMBOLS OK`), md5; `build_android.sh --htp`,
   `readelf -d libnntrainer.so` lists `libsdkl.so` + `libcdsprpc.so`;
   `ndk-build unittest_hvx_softmax` (it now holds `HvxM1Ops.*`); md5s in the PR.
5. **Prefill gate: nothing on the prefill path changes, provable from the diff.**
   `git diff --stat origin/htp_moe...` must list no file under `Applications/`, no
   `htp_compute_ops.cpp` / `htp_backend.cpp`, no `hmx/*` and no `cpu_backend/*`. The IDL change is
   additive: three methods appended after `moe_set_opts`, earlier method indices unchanged. The
   standing E2E gates (prefill ≥ −5 % of variant A, text identical to the CPU run, B ≡ A per
   plan 85 §1.4) are carried to the wiring issue's handoff. No model path calls these kernels in
   this issue.

## 2. Where it lives

| file | change |
|---|---|
| `nntrainer/tensor/m1_ops_det.h` (new, C99, header-only, also includable from C++) | The scalar spec, §3.1. Same place and pattern as `nntrainer/tensor/swiglu_det.h`: one `volatile` store per f32 operation so no compiler can contract an FMA, and no FTZ. It sits outside `htp_backend/` so the ARM side can compile the same spec later (doc 45 §3.3) |
| `nntrainer/tensor/htp_backend/hvx/hvx_conv_gate_f32.{c,h}` | **Taken unchanged from upstream `7f81560b`** (case (b), below). Verified identical at upstream head `a996b4bf`. Its includes (`hvx_convert.h`, `hvx_worker_pool.h`) are byte-identical on `htp_moe` |
| `nntrainer/tensor/htp_backend/hvx/hvx_m1_ops_f32.{c,h}` (new) | `hvx_rmsnorm_f32(x, gamma, y, n, chunk, eps, row_scale_out)`, `hvx_rope64_f32(q, n_q, k, n_k, cs)` (in place), `hvx_conv_gate_m1_f32(abc, state, conv_w, out, C)`. All run on the caller: no pool, VTCM, DMA or heap. `HVX_UVector` loads and stores throughout |
| `test/htp/nntr_hvx_small_ops.c` (new) | the three skel test entries of §3.3 |
| `test/htp/nntr_hvx.idl:449` | three methods after `moe_set_opts`, before `};` at `:450` |
| `test/htp/build.sh:77-89` `SRCS` | add `nntr_hvx_small_ops.c`, `$BACKEND/hvx/hvx_m1_ops_f32.c` and `$BACKEND/hvx/hvx_conv_gate_f32.c` (rule 17) |
| `test/htp/host/hvx_emu/{hexagon_types.h,hvx_hexagon_protos.h}` (new, ≈ 80 lines) | the intrinsic emulation of §3.2. It lives in its own directory, **not** `stub/`, so no existing host check starts seeing an `hexagon_types.h` |
| `test/htp/host/m1_ops_host_check.c` (new) + `run_host_checks.sh` (append after `:77`) | gates 1–2. Built with `-std=c99 -O2 -ffp-contract=off` and without `-ffast-math`. It links a 5-line inline `hvx_worker_pool_run` (NULL pool path) instead of the pthread pool |
| `test/unittest/unittest_hvx_softmax.cpp` (after `HvxSwigluDet`, `:176-245`) | `HvxM1Ops.{RmsnormMatchesDetBitExact, QkNormMatchesDetBitExact, Rope64MatchesDetBitExact, ConvGateM1MatchesDetBitExact, RejectsBadShapes}`, compared against `m1_ops_det.h`. It is already built by `test/jni/Android.mk:979`, whose include path has `nntrainer/tensor` |

Consumers of a changed contract:

* **IDL + both stubs.** Regenerate the ARM stub (`generate_stub.sh`) and rebuild the skel
  (`test/htp/build.sh`) together (rule 3). `HtpComputeOps` does not call the new methods.
  `tools/htp_syntax_check.sh` builds its declarations from `htp_compute_ops.cpp`, so it is
  unaffected.
* **Quantizer format tag (`nntr_quantize_stream`), loader check:** unchanged. No weight bytes or
  dtypes change. The norm gammas and conv weights stay the f32 tensors the CPU layers hold.
* **`NNTR_HTP_PROFILE` stage tables, `tools/htp_fc_report.py`:** unchanged. No model path runs the
  ops. The per-op pcycles arrive with #85's `forward_debug` when the slots are filled.
* **#85 (textual only).** Both issues append to `nntr_hvx.idl`, `build.sh` `SRCS` and
  `run_host_checks.sh`. There is no functional dependency. Rebase onto whichever lands first.

**Upstream case (a) or (b).** Upstream is not merged today (LEDGER "Upstream": only three
transport cherry-picks are in). So the default is **(b)**: one commit
`git checkout 7f81560b -- nntrainer/tensor/htp_backend/hvx/hvx_conv_gate_f32.{c,h}`. Its message
names the sha and the original author (SeungHui Lee), and the user signs off. **One deviation from
the issue text:** `hvx_scalar_stubs.{c,h}` are **not** taken. They are the MoE check's stand-ins,
moved out when `hexkl_conv_block` arrived. They include `hexkl_conv_block`'s `hvx_dq_mul_job`
types, which `htp_moe` lacks. Taking them also means rewriting `htp_moe`'s diverged
`moe_layer_host_check.c` (#80/#87/#105 edits), with nothing gained. This check needs only the
stand-in's formula (`hvx_scalar_stubs.c:369-388` at `7f81560b`), and that formula *is* the conv
spec of §3.1. If the user merges #4327 before implementation (case **(a)**), skip the cherry-pick
commit. Everything else is identical.

## 3. Design

### 3.1 The `_det` specs (`m1_ops_det.h`)

Every step is one IEEE f32 operation with round-to-nearest-even and no FTZ (rule 24). There is no
FMA, qf32, divide, sqrt or libm call. Order is part of the contract.

* **`rmsnorm_det(x[n], gamma[chunk], chunk, eps)`**, per chunk of `chunk` floats. `chunk` must be
  a power of two, a multiple of 32, and divide n: 2048 for the hidden norm, 64 for per-head (this
  is hvx_impl's `FLAG_PER_HEAD`, made a parameter):
  1. `acc[j] = 0`; for i ascending: `acc[j] = acc[j] + x[32i+j]*x[32i+j]`, for 32 lanes j.
  2. Reduce by pairs `(j, j+16)`, `(j, j+8)`, `(j, j+4)`, `(j, j+2)`, `(j, j+1)`, taking the
     lane-0 result. That is exactly what `vror` by 64/32/16/8/4 bytes plus `vadd` leaves in
     lane 0. IEEE add is commutative bit for bit, so operand order inside a pair does not matter.
  3. `d = sum * (1/chunk) + eps`. The multiply is exact because `chunk` is a power of two.
  4. `r = rsqrt_det(d)`: seed `bits(0x5F3759DF − (bits(d) >> 1))` (logical shift), `h = d * 0.5f`,
     then three times `t = h*y; t = t*y; t = 1.5f − t; y = y*t`.
  5. `y[i] = (x[i] * r) * gamma[i % chunk]`. This is the CPU's order: `rms_norm_wrt_width` scales,
     then `multiply_i(gamma)` (`Applications/CausalLM/layers/rms_norm.cpp:93`, `:116`).
  * **Domain**, the analogue of LEDGER §4's SiLU/exp clamp. `d ≥ eps = 1e-5` keeps the seed in
    the normal range. The upper end is |x| < 1.8e19 (sumsq finite). A row outside that range is
    already a broken residual stream. The code comment cites the clamp rule and says why no
    `exp` clamp applies (there is no `exp` in these ops). It also cites "fp32 inside, narrow once":
    nothing here narrows, and qf32 is not used because v75 and v79 disagree on qf32 → sf.
* **`rope64_det(x[64], cs[64])`** per head. `cs = cos[0..31] | sin[0..31]`, and for i < 32:
  `a = x[i]; b = x[i+32]; out0 = (a*c) − (b*s); out1 = (a*s) + (b*c)`. This is the CPU's formula
  and operand order (`neon_impl.cpp:2504-2505`), rounded per operation. q (32 heads) and k (8)
  are contiguous `[heads × 64]` rows. Each half-head is exactly one HVX vector (32 f32 lanes).
* **`conv_gate_m1_det(abc[3C], state[2C], w[3C], out[C])`**. Chunk order `a | b | c` is the in_proj
  split, and `state = x_{t−2} | x_{t−1}` is the layout of `CausalConv1DLayer`'s state
  (`causal_conv1d_layer.cpp:115-122`). Steps:
  1. `g = a*c`.
  2. `y = (w0*g + w1*s1) + w2*s0`, with the upstream kernel's five operations in its order
     (`hvx_conv_gate_f32.c`).
  3. `out = b*y`.
  4. `state ← s1 | g`.

  `w0` multiplies the current row, as on the CPU (`arm_compute_backend.cpp:787-790`). The
  pre-gate `a*c` is inside the op on purpose. At M=1 there is no FC epilogue to hold it: the #80
  GEMV writes f32 rows, whereas upstream fuses it into `hvx_dq_mul` at prefill. That also makes
  the op's in/out match #85's `CONV1D_GATE` record: K = 6144 in, N = 2048 out. The issue's
  exclusion ("gating multiply … are FC calls") is read as the FC epilogues and the in/out
  projections, which stay FC calls.

  The HVX wrapper keeps a 3-row buffer `[s0 | s1 | g]` per layer. It writes `g` into row 2,
  copies `b` into `out`, and calls upstream `hvx_conv_gate_f32(out, C, buf, C, t0=2, m=1, C, w,
  NULL)`. Then it moves rows 1–2 down to 0–1 (16 KiB).
  `ponytail:` two-row memmove per token, ≈ 1–2 µs × 18 layers. Upgrade: a ring index with three
  row pointers, if #85's per-op pcycles show it.

**None of these is bit-identical to today's CPU path.** The NEON RMSNorm uses FMA and 16
accumulators, and NEON RoPE and conv may contract under `-ffast-math` (swiglu_det.h's finding).
Doc 45 §3.3 says the CPU reference must use the same `_det` (scalar/NEON/HVX). Switching the ARM
layers is a CPU-path and prefill-path change that moves the CPU control. It is **not** in this
issue (gate 5). It is filed as a decision for the wiring issue (§6).

### 3.2 Host bit check: real HVX source on an intrinsic emulation

With no simulator (contract §4.1), the only host check that tests the *kernel* rather than a
second copy of the spec compiles the HVX `.c` on the host. `hvx_emu/` defines `HVX_Vector` as a
32 × int32 struct. The intrinsics the four kernels use are emulated lane by lane, with one
`volatile` f32 operation per lane:

* `Q6_V_vzero`, `Q6_V_vsplat_R`;
* `Q6_Vsf_vadd/vsub/vmpy_VsfVsf`;
* `Q6_Vw_vadd/vsub_VwVw`, `Q6_Vuw_vlsr_VuwR`;
* `Q6_V_vror_VR`, `Q6_R_vextract_VR`.

It proves loop bounds, lane and head indexing, the reduction tree, operation order, zero-state
handling and the state shift against the spec. The premise it rests on is that each `Vsf` op is
one IEEE RNE op without FTZ. That is device-confirmed for add, sub and mul by
`HvxSwigluDet.MatchesScalarBitExact` and rule 24. `vror`, `vextract` and the integer ops move bits
exactly. The device gtest re-checks the premise for this code (§4 step 6).

**Prototyped during planning** (scratch only, nothing committed): upstream `hvx_conv_gate_f32.c`
compiled unmodified with gcc `-Wall -Wextra -ffp-contract=off` on a 12-line emulation, and matched
the stand-in formula on 6 × 2048 random elements, `bad=0`.

**Rejected: hvx_impl's kernels as they are** (`hvx-rmsnorm.c`, `hvx-rope.c`). They are fp16. They
use qf32 products and `Vhf_equals_Wqf32`, which v75 and v79 convert differently. They also call a
scalar `1.0f / sqrtf(...)` per row, whose libm bits no host spec can pin. They cannot satisfy a
`_det` bit gate. **What is reused:** the per-head chunking (`chunk = head_dim`), the rotate-half
structure with one vector per half, and the cos/sin row formula (`nntr_htp_rope.h`) as the host
check's input generator. `hvx-quant.h`'s rounding recipe is not lifted, because these ops end
before the quantizer and `hvx_quant_u8.c` is untouched.

Also rejected: a host check that swaps the HVX kernel for a scalar stand-in (upstream's
`conv_block_host_check` style). For an op with no surrounding loop, it only compares the spec
with itself.

### 3.3 IDL (additive, test-only)

```
// [#82] M=1 small ops against m1_ops_det.h. Test entries: no model path calls them.
AEEResult rmsnorm_det_f32(in uint32 chunk, in float eps, in sequence<float> x,
                          in sequence<float> gamma, rout sequence<float> y,
                          rout sequence<float> row_scale);
AEEResult rope64_det_f32(in uint32 n_q, in sequence<float> cs,
                         in sequence<float> qk, rout sequence<float> y);
AEEResult conv_gate_m1_f32(in sequence<float> abc, in sequence<float> conv_w,
                           in sequence<float> state_in, rout sequence<float> out,
                           rout sequence<float> state_out);
```

`row_scale` returns r per chunk, so a failure says whether the reduction and rsqrt or the scaling
diverged (the reason `swiglu_det_f32` returns intermediates). A bad shape returns
`AEE_EINVALIDFORMAT`, never `AEE_EBADPARM`, which stays the stale-skel symptom (rule 3, plan 85
§1). The entries use the caller's FastRPC buffers: no session state, heap or VTCM.

### 3.4 RoPE cos/sin source and memory (the issue asks the plan to decide)

The kernel takes one 64-float row pointer and does not know the position. For the wiring issue:

* **Chosen:** the ARM generates the rows with the function the CPU RoPE already uses
  (`calc_trigonometric_vals_dup`, `mha_core.cpp:897-900`; the NEON `cos_ps`/`sin_ps`). It uploads
  `max_seq_len` rows once at `graph_init`, and the DSP indexes by #85's `pos` (already
  `< max_seq`, plan 85 §3.2). Memory is **256 B/position: 512 KiB at `max_seq_len` 2048** (the
  NPU model's `nntr_config.json`), 1 MiB at 4096. The cos/sin bits equal the CPU's, and #85's
  `forward` signature needs no new argument.
* **Rejected:** a table for `max_position_embeddings` 128000: 31.25 MiB of the 32-bit DSP
  address space for ≤ 1536 positions used (rule 8).
* **Rejected:** lazy per position on the DSP. It is 256 B, but DSP libm `cosf`/`sinf` bits
  differ from the CPU's `cos_ps` polynomial. That moves every attention score against the CPU
  control for no gain.

The whole DSP memory these ops need once wired, all DSP heap: rope table 0.5 MiB, conv state
18 × 3 × 2048 × 4 = 432 KiB, conv weights 18 × 3 × 2048 × 4 = 432 KiB, norm gammas
(2 × 24 + 1) × 8 KiB + q/k 6 × 2 × 256 B ≈ 395 KiB. That is **≈ 1.8 MiB** against ≈ 182 MiB,
with no VTCM and no DMA. In this issue the number is 0.

## 4. Steps

1. **Spec + tolerance half.** `m1_ops_det.h` and the spec-only part of `m1_ops_host_check.c`
   (gate 2, the pos-0 identity). Gate: `run_host_checks.sh`.
2. **Upstream kernel.** The case-(b) commit of the two `hvx_conv_gate_f32` files (skip under
   (a)), plus `hvx_emu/`. The check compiles upstream's kernel on the emulation and matches the
   conv spec at the prefill shape (t0 = 0, m = 8). Gate: `run_host_checks.sh`.
3. **HVX kernels.** `hvx_m1_ops_f32.{c,h}`, with the §3.1 domain and rule citations in comments.
   The check's bit half runs every case of gate 1, including M=1 chain ≡ prefill-shape call.
   Gate: `run_host_checks.sh` prints `M1 OPS BIT-IDENTICAL`, and `clang-format-14`.
4. **IDL + skel.** `nntr_hvx_small_ops.c`, the IDL methods and `build.sh` `SRCS`. Gate:
   `generate_stub.sh`, then `test/htp/build.sh` (`-Wall -Werror`, `UNDEFINED SYMBOLS OK`), md5.
5. **Device gtest + app.** `HvxM1Ops.*` in `unittest_hvx_softmax.cpp`. Gates: `ninja -C build`,
   `*qs4cx*`, `*Lfm2Moe*` 6/6, `tools/htp_syntax_check.sh`, `build_android.sh --htp`, `readelf -d`,
   `ndk-build unittest_hvx_softmax`, md5s, then the gate-5 diff check. PR into `htp_moe`, then
   `state:review`. **The issue ends here.**
6. **Device, later (unavoidable for gate (a) on silicon; not this issue).** Two parts:
   * **Next sitting that has room (e.g. #100's):** a ride-along with no E2E variant, because the
     ops are unwired and A's cells cannot change. Run `unittest_hvx_softmax
     --gtest_filter='HvxM1Ops.*'` once on that sitting's binaries. Expected: `bad=0` on every
     case, with the printed `max_ulp` vs fp32. Any `bad > 0` names the stage through `row_scale`.
     That gtest also prints, but does not gate, one overflow row (rule 24's unexplained ±3e38
     case).
   * **The wiring issue's handoff** (full E2E, prompt 512, gen 64 / 512 / 1024, ×2,
     `NNTR_NUM_THREADS=8`, ≤ 4 variants):
     * **A** = `htp_moe` head unchanged (first);
     * **B** = the same binaries with #85's switch and the small-op kinds resident;
     * **B-prof** = B at `NNTR_HTP_PROFILE=2`, G=64;
     * optional **A-prof**.

     Read text B ≡ A and ≡ CPU, prefill ≥ −5 % of A, and per-op pcycles.

## 5. Risks (host-vs-device gaps)

* **The emulation premise.** The host check can only be as right as "one `Vsf` op = one IEEE
  op". It is confirmed for add, sub and mul including subnormals, but not for inf/NaN encodings
  (rule 24's overflow row). The domain in §3.1 keeps real inputs away from both, and the
  ride-along gtest prints the overflow row. A `bad > 0` there with the host at 0 is the finding
  to look for, and `row_scale` localises it.
* **Compiler contraction or reassociation.** Host: `-ffp-contract=off`, `volatile` per operation,
  no `-ffast-math` (rule 24: it can set FTZ). Skel: `build.sh` compiles `-O3` without fast-math,
  and HVX has no sf FMA to contract into. The device gtest is the check for the latter.
* **Stale skel/stub.** There are three new methods. The gtest's first case is `RejectsBadShapes`,
  which expects `AEE_EINVALIDFORMAT`. `0x8000040E` means a stale skel (rule 3), and the sitting
  records it as such rather than as a kernel failure.
* **Alignment.** FastRPC buffers carry no vector alignment (`nntr_hvx_softmax.c:87-88`). The
  kernels use `HVX_UVector` only. The emulation cannot catch an aligned-load truncation, so this
  is a review item.
* **DMA rate, DVFS, thermal drift, address space.** Not exposed in this issue: no timing is
  claimed and no memory is allocated. They bind the wiring handoff: per-op pcycles, not tok/s,
  carry the op cost (rules 20, 23, 27), and §3.4's ≈ 1.8 MiB is noted against rule 8.
* **Text identity when wired.** The `_det` ops round differently from the ARM NEON ops that A
  runs (§3.1). Any u8 level they flip in the next FC's quantizer can change B's text vs A. Until
  the ARM side runs the same spec (doc 45 §3.3), the wiring handoff's text gate may fail for that
  reason alone. See §6.
* **Baseline.** `run_host_checks.sh` passes at `485435ce` in 38 s. `ninja -C build` was not run
  during planning: this worktree has no `build/`, and no host-built source changes before step 5.

## 6. Docs to update

* **BENCHMARK.md:** nothing in this issue. The ride-along records one `HvxM1Ops` gtest line
  (bad counts, `max_ulp`, serial) under that sitting.
* **LEDGER.md:**
  * §4 lift 5: mark as done. Record what was lifted (structure and row formula) and what was not
    (the fp16/qf32 arithmetic, the scalar `sqrtf`, `hvx-quant.h`).
  * "Upstream": `hvx_conv_gate_f32.{c,h}` are cherry-picked from `7f81560b` (case (b)), and
    `hvx_scalar_stubs` is not.
  * ⑨: the small-op kernels exist (PR #…), unwired, 0 ms.
  * **New open item (user decision via the supervisor): does the CPU path switch to
    `m1_ops_det.h`** (a NEON twin, as `swiglu_det.h` has) when the small ops become resident?
    Doc 45 §3.3 requires it for text identity, but it changes the CPU control and the prefill
    path. Without it, the wiring handoff's B ≡ A text gate is at risk.
