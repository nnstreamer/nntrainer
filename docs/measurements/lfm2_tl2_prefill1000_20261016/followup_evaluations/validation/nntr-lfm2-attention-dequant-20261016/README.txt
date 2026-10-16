# LFM attention integer score correction candidate

Production patch: `dequant-only.patch` (one function in `hexkl_attn_q.c`, 21 additions / 6 removals). Portable validation patch: `dequant-fixture.patch` (a C fixture and a Python runner). Base: e58b8f8dc37d3eecc8b082757e1ac004e21126dd. Both patches pass `git apply --check` against the parent's current combined Qprep/store tree. Source and fixture were formatted with clang-format 14. No production-parent files or device were changed.

## Change and exactness argument

For the existing full LFM dynamic-Q8 geometry only (Q32/KV8/hd64, n_q >=128 and multiple128, from0/to=n_q, no window/softcap), replace the floating-point accumulator/zero-point correction with:

```
Q6_Vsf_equals_Vw(Q6_Vw_vmpyieacc_VwVwVh(acc_i32, colsum_i32,
                                      Q6_Vh_vsplat_R(-q_zp)))
```

The subsequent multiplication by q_scale, then multiplication by k_scale and the FP16 tile conversion retain their order. All other shapes and fixed-Q8 caches take the previous FP correction. No cache capacity, quantization, layout, persistent allocation, DMA or worker scheduling changes.

The Q quantizer clamps zero points to [0,255]. The cache stores signed int8 K; using the conservative [-128,127] bound at hd64 gives |colsum|<=8192 and |acc|, |zp*colsum|<=2,088,960. The conservative difference bound is 4,177,920. All are less than 2^24 and exactly representable in FP32, so the old multiply/subtract of integer-valued floats and the new integer correction agree. The halfword multiplier fits [-255,0], with no int32 overflow. Cancellation returns +0 in the actual tested HVX operations; signed-zero scale products therefore remain identical. The existing scale multiplications are not reassociated.

## Validation

- Raw arithmetic: Hexagon libnative ASan/UBSan, 1,048,576 lanes exact for FP32 correction and final FP16 bits. All qz values, positive/negative/extreme colsums, extreme/cancelling/random accumulators, signed-zero and small/large finite scales covered. `math_check.c` is the additional audit harness.
- Raw arithmetic on Hexagon v81 simulator: 131,072 lanes exact; `math-sim.log`.
- Production-function fixture: actual `dequant_s`, context and address helpers are extracted from the candidate. The independent reference uses the previous FP operation order and independent row-pair-to-FP16-tile addressing. **384 cases exact** under both libnative ASan/UBSan (`native.log`) and Hexagon v81 simulator (`sim.log`). Covers 128/1024/2048 rows; all 256 zero points; real/zero/cancelling/boundary/random score patterns; misaligned registry metadata; accumulator base32 and stride32/64; last KV head; qi1/i1/col1/sbuf1; out-of-registry zero path; Q4, fixed-Q8, shifted cache, short rows, window, softcap and other-head geometry fallbacks. Full backing-buffer comparison catches writes outside the two target score tiles and changes to accumulator data.
- Mutations: wrong correction sign and allowing fixed-Q8 through the new path both fail assertions as expected.
- Complete production attention kernel compiles for v81/HVX128/HMX, O3, Wall/Werror. `hexagon-compile.log` is empty on success. Disassembly shows the intended `vmpyie` sequence and loop unswitching, rather than a branch per scalar row.
- Compiler stack usage: phase_qk 512 -> 640 B (+128 B), top prefill stays1408 B. This is stack/register-spill usage, not new heap/weight/activation storage. Function-level static frames are not a measured entire call-chain peak.

## Reproduction after applying both patches

From the repository root, set `DEFAULT_HEXAGON_TOOLS_ROOT` to the SDK's HEXAGON_Tools/19.0.04 directory, then:

```
python3 tools/lfm2_attn_dequant_check.py --output-dir /tmp/attn-dq-native --sanitize --mutations
python3 tools/lfm2_attn_dequant_check.py --output-dir /tmp/attn-dq-sim --sim
```

`--repo` and `--hexagon-tools` can override the default repository and SDK paths. Simulator execution requires the SDK's normal runtime dependencies; this host uses already-extracted libncurses5/libtinfo5 via LD_LIBRARY_PATH, with no system installation. The runner itself contains no user-local path or baseline commit dependency.

This is an exactness/dispatch candidate, not a measured end-to-end speed result. It still needs parent-controlled matched DSP replay and LFM A/B timing. Simulator total cycles include fixture construction, allocation, reference, and canary comparisons and must not be presented as kernel speed. HMX, FastRPC, QuRT and real memory effects are outside this host/simulator fixture. Gemma fixed-Q8 is statically guarded and covered by the host dispatch fixture; no Gemma benchmark was run.
