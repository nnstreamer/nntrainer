# LFM actual plain MoE stage diagnostic v2 (test build only)

Base: `9d8706182fcf56fb41de3c9ec5af17c7ee80e8a4`.
Patch: `lfm2-plain-stage-trace-v2.patch` (SHA-256 `79c19f6d54dce0b2d8f41b697b8ea4dfcbf09e8caa195d11044dae7aa1355203`). Supersedes v1, which instrumented an unused norm-helper path on this LFM builder; parent device run confirmed 65-ID parity but zero trace records. The previous README and patch are retained as explicitly named v1 artifacts.

Current LFM dispatch is **plain**, not the Gemma fused norm-add path: the builder sets in_norm=true but no out_add, and the app only folds the expert norm when out_add && in_norm. Normal LFM therefore computes its shared router/expert input norm on ARM and passes pre_gamma=nullptr, post_gamma=nullptr, add_x2=nullptr. The earlier assumption that this model already used fused pre-norm/direct input was incorrect.

V2 wraps the existing core call inside `nntr_hvx_mm_u8i4_moe_layer`, which is reached by the real plain dispatch and by any norm-helper callers. No IDL, RPC signature, ARM, weight, cache, reader, dispatch, or arithmetic change. Guard: M>1, K=N=2048, inter=1536, act=SwiGLU. It selects the LFM prefill geometry without asserting the quant format. Gemma and M=1 are excluded. The kernel arguments, scratch flags, and return value are unchanged. Probes are disabled on both success and failure after the core returns.

**kernel elapsed includes the core FFN, but excludes the ARM input norm, outer post-norm/residual if used, host staging, and RPC overhead.** V2 prints `f` bits: bit0 is pre_gamma present, bit1 is out_heap; current LFM should be f=0. The load-time all-zero warmup also matches the shape and is expected as the first record. Its E64, one64-row block/expert, zero row weights are constructed in `gemma4_htp_weights.cpp:474–491`; it is separate from the38 actual prefill layers.

Apply the patch in the parent-controlled tree, build the existing DSP target with:

```sh
HEX_EXTRA_CFLAGS='-DNNTR_LFM2_MOE_STAGE_TRACE=1' HEX_ARCH=v81 ./test/htp/build.sh
```

Use the normal ARM binary and normal flags, with `NNTR_HTP_PROFILE=1` if parent profiling is needed. **Do not set PROFILE=2, HOST_NORMS, or MOE_HOST**: whole-model PROFILE=2 has known output differences; folded dense/norm paths can change dispatch. LFM MoE itself already has null pre/post gamma on the normal plain path, so the prior claim that PROFILE=2 newly removes LFM MoE pre-gamma was incorrect. Keep the authorized C16 / preload128 / PAGE0 / six-reader setup. This is a diagnostic run and is ineligible for normal TPS claims. The existing HAP timer and DMA trace instrumentation perturb elapsed time. The DSP plain entry's static stack is 88 bytes with the flag off and 232 bytes with it on; no heap/weight-cache allocation is introduced.

The prefix is `[LFM2-MOE-STAGES] v=2 M=... E=... rc=... f=... kernel=...`. CSV has all 31 existing `MOE_T_*` fields in their enum order. `parse.py` documents their units, rejects incomplete records and nonzero kernel status, and emits raw per-call data plus clearly labeled sums:

```sh
python3 parse.py /path/to/logcat.txt --expected-calls 39 --skip-initial 1 --out /path/to/stages.json
```

Check that the log has39 calls (1 known load warmup +38 real LFM MoE calls) and normal65 generated IDs before interpreting it. `--skip-initial 1` explicitly preserves that first record in skipped_initial_records and summarizes only the38 real calls. A different count must be investigated for multiple prompt blocks or missing logging. Do not infer warmup solely from a block count on an arbitrary run. `FARF(ERROR)` is used **only for this diagnostic** because the saved device log enables adspmsgd mask 8 (ERROR); `FARF(ALWAYS)` maps to HIGH and may not be collected under that mask. The tagged ERROR-level line does not report an execution failure. The SDK fallback formatter is 256 bytes; the parser refuses truncated lines. On-device receipt and 65-ID parity remain parent-owned validation gates.

Stage interpretation:

- `swiglu_worker_us` is overlapped aggregate worker time, not a serial contribution.
- `dequant_wait_us`, `requant_wait_us`, and `scatter_wait_us` are exposed waits, not full worker busy time.
- `drain_gu_us` and `drain_dn_us` include native expansion-worker waits as well as DMA waits.
- Descriptor trace stores at most 512 pushes; calls above that count have incomplete busy/first-ready/last-issue information. Do not derive bandwidth from those fields or an average that mixes valid and truncated calls.
- Arrays of times are not disjoint; summing every time field double counts work.

Validation completed on host:

- Real entire DSP C translation unit, v81 `-O3 -Wall -Werror`, flag off and on: PASS (`compile.json`, `.o`, `.su`).
- Actual baseline and modified plain/norm/norm_add source extracted into one ASan+UBSan harness: 264 conditions each with flag off and on (528 conditions total), including M=1,2,63,64,65,1024,2048; norm/add; pre/post norm presence; kernel success/failure; Gemma, dense, activation and shape fallbacks. Kernel call arguments, output bytes, norm/add execution, and cleared scratch flags match. 144 conditions exercise trace-on and validate all 31 CSV entries. This mocks the numerical kernel; it does not replace device output parity.
- Parser accepts a complete v2 line and rejects truncated, failed, decode, absent, and unknown-version records. 39-record fixture validates explicit separation of the first warmup from38 real records.
- clang-format 14 on changed lines and `git diff --check`: PASS.

Reproduce host validation with explicit paths:

```sh
python3 check.py /path/to/patched/repo /path/to/host-output
```

`check.py` uses the given repository's HEAD as the unmodified reference and GCC ASan+UBSan. `compile.py` records local SDK paths in `compile.json` and is a local build aid, not proposed repository infrastructure.
