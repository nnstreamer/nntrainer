LFM2 S26 Ultra 1024-token prefill optimization (2026-10-13)

The selected measured runtime reaches 825.806 TPS at the median of three
uninstrumented runs: 825.806, 827.809 and 775.758 TPS. One run remains below
800 TPS. The user accepted this result and requested a commit.
The previous recovered-device baseline measured 619.667 TPS at median
latency. This comparison includes the larger expert cache and fewer readers,
not only kernel changes. Expert 2-bit / other 4-bit weights are unchanged.

Implementation
  Attention: fused softmax/quantization, guarded 64-row query tiles, faster
  dynamic Q8 layout and cache append, overlapping QK dequantization, parallel
  stores, and opt-in Q/K RoPE plus V cast through one NPU RPC.
  Conv: inline HMX issue, accumulator-plane dequantization and four-row gate.
  FFN/norm: shared NPU output slices avoid CPU concatenation and unused
  staging allocations; residual add/RMSNorm and per-head FC epilogues are
  fused. Generation-only final FFN/norm work can be limited to the last 64
  rows, retaining the entire terminal Conv and its persistent state.

Selection and use
  selected-environment.json contains every measured environment setting.
  The key settings are cache 22, prefetch/miss readers 4, tail rows 64,
  NNTR_HTP_PREFILL_QKV_PREPARE=1, NNTR_HTP_PREFILL_ROPE_FUSION=0 and
  NNTR_HTP_PREFILL_SHARED_IO=7. Use the same 1024-token model configuration
  in nntr_config.json, adapting its model/tokenizer paths to the deployment.
  The tail and new QKV preparation paths are opt-in; unsupported shapes or
  other output consumers retain the original behavior.
  Tokenization, embedding, sampling, expert reads and some staging remain
  on the CPU. This change does not claim that all CPU work moved to the NPU.

Rebuild the ARM FastRPC stub and DSP skel together from test/htp/nntr_hvx.idl.
The measured working tree also contained unrelated, uncommitted TL2 methods.
Those are excluded from this commit, so its appended RPC ordinal numbers
are different. Never combine a regenerated half with an old half.
The hashes in result.json identify the exact historical measured runtime.
The standalone commit's build checks are in commit-validation.json; its
fresh binaries were not used to replace the recorded device measurements.

Validation
  bash test/htp/host/run_host_checks.sh
  python3 test/htp/host/run_prefill_host_checks.py --sanitize
  The latter reproduces 576 RoPE dispatch cases, 10368 rejected transactions
  and 572 generation-tail scope/offset/diagnostic/thread checks.
  Real-device block validation and full-model parity are separate from
  timing runs. The untrimmed graph matches 286 outputs, and the selected
  trimmed runtime matches all 64 generated tokens and 65536-way FP32 logit
  vectors at each step. See the accompanying JSON evidence.
  This is one fixed prompt. The original quantized model repeats "cred";
  the equivalence check does not establish general language quality.
  Full CI and the cross-platform build matrix have not been run.

Memory/startup tradeoff
  Cache22 adds 1221.9375 MiB of logical expert slots versus cache 8 before
  arena allocation granularity and other buffers. More experts are loaded
  before prefill. Prefill TPS excludes startup; total peak RAM and startup
  deltas were not measured before the requested stop. Registry 4096/cache 28
  and cache 32 were left as experiments and are not included here.
