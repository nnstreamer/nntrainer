# LFM normal fused MoE stage diagnostic (test build only)

Base: `9d8706182fcf56fb41de3c9ec5af17c7ee80e8a4`.
Patch: `lfm2-fused-stage-trace.patch` (SHA-256 `774584e7059733b73a79555c9d5a5c58e39921f53794e1cb42ab23a2b7bb81cf`).

This adds 32 guarded lines to `test/htp/nntr_hvx_mm_u8i4.c`. No IDL, RPC signature, ARM, weight, cache, reader, dispatch, or arithmetic change. The guard is `M > 1, K = N = 2048, inter = 1536, act = SwiGLU, pre_gammaLen = 2048`. It selects the LFM2 fused prefill shape, rather than inspecting weight format; run with the already validated native TL2 model. Gemma and M=1 are excluded.

The ordinary fused helper still sets `pre_gamma/pre_eps/out_heap` and calls exactly the same kernel with exactly the same parameters. Both norm and norm_add callers share the helper. The diagnostic resets existing probes immediately before that call and disables them immediately after it, including failure returns. **kernel elapsed includes the fused activation RMSNorm and the core FFN, but excludes the outer post-norm and residual epilogue.** It is not total RPC duration. The normal helper cleanup and epilogue remain unchanged.

Apply the patch in the parent-controlled tree, build the existing DSP target with:

```sh
HEX_EXTRA_CFLAGS='-DNNTR_LFM2_MOE_STAGE_TRACE=1' HEX_ARCH=v81 ./test/htp/build.sh
```

Use the normal ARM binary and normal flags, with `NNTR_HTP_PROFILE=1` if parent profiling is needed. **Do not set PROFILE=2, HOST_NORMS, or MOE_HOST**: PROFILE=2 deliberately moves norm/epilogue to ARM and changes this workload. Keep the authorized C16 / preload128 / PAGE0 / six-reader setup. This is a diagnostic run and is ineligible for normal TPS claims. The existing HAP timer and DMA trace instrumentation perturb elapsed time. The DSP helper's static stack is 128 bytes with the flag off and 272 bytes with it on; no heap/weight-cache allocation is introduced.

The prefix is `[LFM2-MOE-STAGES] v=1 M=... E=... rc=... kernel=...`. CSV has all 31 existing `MOE_T_*` fields in their enum order. `parse.py` documents their units, rejects incomplete records and nonzero kernel status, and emits raw per-call data plus clearly labeled sums:

```sh
python3 parse.py /path/to/logcat.txt --expected-calls 38 --out /path/to/stages.json
```

Check that the log has the expected 38 real LFM MoE calls and the normal 65 generated IDs before interpreting it. A different call count should be investigated for warmups or multiple prompt blocks, rather than silently dropped. `FARF(ERROR)` is used **only for this diagnostic** because the saved device log enables adspmsgd mask 8 (ERROR); `FARF(ALWAYS)` maps to HIGH and may not be collected under that mask. The tagged ERROR-level line does not report an execution failure. The SDK fallback formatter is 256 bytes; the parser refuses truncated lines. On-device receipt and 65-ID parity remain parent-owned validation gates.

Stage interpretation:

- `swiglu_worker_us` is overlapped aggregate worker time, not a serial contribution.
- `dequant_wait_us`, `requant_wait_us`, and `scatter_wait_us` are exposed waits, not full worker busy time.
- `drain_gu_us` and `drain_dn_us` include native expansion-worker waits as well as DMA waits.
- Descriptor trace stores at most 512 pushes; calls above that count have incomplete busy/first-ready/last-issue information. Do not derive bandwidth from those fields or an average that mixes valid and truncated calls.
- Arrays of times are not disjoint; summing every time field double counts work.

Validation completed on host:

- Real entire DSP C translation unit, v81 `-O3 -Wall -Werror`, flag off and on: PASS (`compile.json`, `.o`, `.su`).
- Actual baseline and modified norm/norm_add source extracted into one ASan+UBSan harness: 176 conditions each with flag off and on (352 conditions total), including M=1,2,63,64,65,1024,2048; norm/add; pre/post norm presence; kernel success/failure; Gemma, dense, activation and shape fallbacks. Kernel call arguments, output bytes, norm/add execution, and cleared scratch flags match. 48 conditions exercise trace-on and validate all 31 CSV entries. This mocks the numerical kernel; it does not replace device output parity.
- Parser accepts a complete line and rejects truncated, failed, decode, absent, and unknown-version records.
- clang-format 14 on changed lines and `git diff --check`: PASS.

Reproduce host validation with explicit paths:

```sh
python3 check.py /path/to/patched/repo /path/to/host-output
```

`check.py` uses the given repository's HEAD as the unmodified reference and GCC ASan+UBSan. `compile.py` records local SDK paths in `compile.json` and is a local build aid, not proposed repository infrastructure.
