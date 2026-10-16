Normal LFM native-TL2 MoE stage audit
===================================

This is diagnostic evidence, not a normal TPS comparison. C16, PAGE_CACHE=0
Direct I/O, CACHE_GROW=0, preload 128, reader counts and native model values
remain fixed. The normal ARM core/app binaries are unchanged. The temporary
DSP-only compile flag NNTR_LFM2_MOE_STAGE_TRACE=1 enabled existing per-stage
probes and one FARF record per matching call. The patch has been removed
from production source and a production DSP build completed afterwards.

Dispatch correction
-------------------
LFM has in_norm=true, out_add=false and router_norm=false. It computes one
shared router/expert RMSNorm input on ARM, then calls the plain DSP MoE
entry. Gemma's folded norm-add/direct-input path is not active for this
LFM call. An earlier audit incorrectly stated otherwise; the corrected
audit is retained here. Turning on out_add would change LFM's mathematics.
Skipping ARM norm would also leave its router without the normalized input.

The first v1 probe was placed in the unused norm helper, so its device run
preserved all 65 IDs but produced no stage records. V2 instruments the actual
plain entry. The benchmark's old text filter also missed the new tag; its
39 full FARF lines were recovered from logcat using the recorded run start.
The raw recovered lines and parser output are retained. bench.py now keeps
LFM2-MOE-STAGES lines explicitly. ERROR severity is only a diagnostic logging
choice under the device mask, not an execution failure; all rc values are 0.

Normal path measurement
-----------------------
One 1024-token profile-level1 run: 39 calls = one known load warmup +38 actual
prefill MoE layers. The parser preserves the warmup separately and sums
only those 38 layers, all with f=0 (no pre_gamma or cached output path).
The generated 65 IDs match the normal reference exactly:
b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e
No competing known inference/checker process was observed.

  DSP core elapsed                       382.527 ms
  HMX issue                              104.761 ms
  GU weight/expansion drain                54.343 ms
  DN weight/expansion drain                12.916 ms
  DSP activation/output staging           25.678 ms
  Exposed scatter wait                    24.615 ms
  Accumulator read                        22.340 ms
  Source activation quantization          17.309 ms
  Exposed dequant wait                    12.161 ms
  Gather                                   1.649 ms
  Exposed intermediate requant wait        0.625 ms
  Scratch allocation/free                  0.024 ms
  HMX 64-row blocks                         3776

Fields overlap and must not all be added together. The 1168.875ms SwiGLU
worker sum is overlapping worker busy time, not serial elapsed. The 67.259ms
GU+DN drains include expansion-worker joins, not just physical DMA stalls.
Descriptor traces truncate beyond 512 pushes, so their incomplete busy-time
fields cannot establish memory bandwidth or hardware saturation. Core time
excludes ARM norm/staging, router, file reads and RPC transport overhead.

Decode uses native TL2 LUT GEMV for both Gate/Up and Down. Prefill losslessly
expands native TL2 codes into HMX int4 panels without requantization; there
is no per-expert decode-style activation LUT on that HMX path. Its source
activation quantization is already shared across experts, expansion is
already tracked per panel, and GU/SwiGLU plus DN/scatter are already fused.
The 3776-block count agrees with the independently parsed normal top4 routes.
Their 35.59% padded-row capacity is not a whole-FFN speedup opportunity: weight
reads, expansion, valid-row epilogues and scheduling remain necessary.

Candidate priorities, not measured gains
---------------------------------------
The router's cached input is copied again by plain MoE: 304 MiB over 38 layers
at 1024 rows. Removing this could save part of the 25.678ms staging total, but
requires exact buffer lifetime/generation checks; pointer/shape equality or
sampled payload comparisons alone do not establish correctness. No input
reuse production change is included in this evidence commit.

Stable native metadata refresh currently drains GU and DN separately and
zeros already initialized native bias again. A paired descriptor/drain
change is under separate evaluation. Its cost is mixed into the residual
and has no isolated measured saving here. Neither candidate justifies
claiming a 15% FFN gain or 1000 TPS.

The apparent ~293ms registration counter is load-time lifetime accounting,
not prefill work. Its native "convert" field includes file reads. Likewise,
the all-zero warmup's discarded GEMM is load-time work. Removing it safely
would need to preserve scratch/ION preparation, first-page touches and the
initial accumulator-layout calibration; moving those into the first prefill
would not improve the user's requested prefill metric. See the audit.

FP32 router spill experiment: rejected
-------------------------------------
The existing v81 router loop has one accumulator stack store/reload pair.
Removing it changed 27 to 33 packets and slowed a strict-FP32 simulator case
by 13.26% (1,002,510 -> 1,135,482 cycles). All 36 numerical cases were bit-exact.
Other alignment/unroll/live-range variants did not produce a better schedule.
No router production edit was adopted and no device speedup is claimed.

Validation and reproduction
---------------------------
The v2 diagnostic builds with the flag both on and off for v81. Its extracted
plain/norm/norm-add wrapper checker passes 528 ASan/UBSan conditions, including
failure cleanup and 31-field recording. Numerical kernels are mocked in this
wrapper test; the device 65-ID comparison is recorded separately. The strict
parser requires 39 complete records and skips exactly the known first warmup.
Use diagnostic/parse.py with --expected-calls 39 --skip-initial 1. Exact build flag, raw commands,
binary/source hashes, diagnostic patch and compiler commands are retained.
The measured model/normal memory metric is the same as the preceding report.
These instrumented runs do not enter normal TPS or peak-memory comparisons.
