Rejected follow-up candidates after the convolution ION change

None of the three candidate changes below is in the accepted production
source. Preserve both their positive local measurements and the full-model
results that did not justify adoption. Reproduction patches are compressed
in validation/; exact deployed binary hashes and source diffs are in artifacts/.
All measurements use the same S26 Ultra/native TL2 model, C16, grow0,
Direct I/O/page-cache0, preload128 and existing reader pools documented in
the parent README. No additional Gemma benchmark was run.

1. Attention integer zero-point correction
For dynamic Q8/HD64, bounded integer subtraction before conversion to float
is exact and replaces FP32 correction arithmetic. Actual-function host
sanitizers, v81 simulation, the21-case replay and generated65 IDs pass.
The first comparison reduced dequantization44.061 to43.478ms over ten layers.
A repeated comparison measured43.706 to43.458ms; total attention150.601 to
150.233ms. These are instrumented local stages, not headline TPS. The tiny
benefit did not establish a full-model improvement and adds a guard/128B
phase frame, so the patch was removed. No normal three-run speed claim.

2. Native TL2 decode paired panels
Two output panels share activation LUT loads; integer sums remain bounded
and independent. Existing per-expert acc_dn scratch is reused (no allocation
growth, but up to32KiB of previously untouched accumulator pages across top4).
Host/sanitizer/v81 checks and independent multi-expert boundary checks pass.
Device normal and profile runs preserve every generated ID.
Three-run decode medians: control17.6698, DN17.6942, DN+GU17.5294TPS.
Profile MoE compute excluding miss wait:9490,9523,9471us/token respectively.
The simulator's ~13% DN/~9% GU isolated kernel improvements correspond to
only ~0.065ms/~0.087ms per token after eight-way concurrency, not those
percentages of the entire9.5ms MoE path. This is an estimate; actual device
gain was not established. All code and fixture changes were removed.
DN build uses the default paired-DN guard; DN+GU additionally uses
HEX_EXTRA_CFLAGS=-DNNTR_LFM2_TL2_PAIR_GU=1. Prefill math is unchanged, so
prefill differences between these variants are not credited as an improvement.

3. Dynamic Q8 attention output in the existing ION pool
Reuse an existing disjoint output allocation, copy the full output to host,
then use the existing stageInput swap to avoid the next FC input copy.
No pool allocation/capacity increase;1024 limited,2048 and Gemma fall back.
29 focused sanitizer cases and an independent extended31-case review pass.
All normal/profile65 IDs, the2048 fallback and same-handle2x1024+64 pass.
The first profile removed80MiB/10 FC input copies but added80MiB/10 explicit
attention output copies. Attention wall-minus-DSP fell26.036 to17.935ms;
the repeated trace fell32.846 to18.143ms. These are full non-DSP call costs,
not a direct measurement of the output-only transport component.
Three-run normal prefill medians nevertheless fell845.582 to831.169TPS;
all three paired observations were slower. Decode17.6357 to17.6211TPS.
The change was removed. The profile shows other stage/I/O variation; no
specific causal explanation for the total regression has been established.
Only libnntrainer.so differs in this A/B; app and DSP hashes match. Rebuilding
the original core after removing the patch reproduces its previous SHA256.

Measurement and build notes
- results.json contains every run, including incomplete peak scans where
  present. Raw memory reports retain all samples and scan error fields.
- Memory is the sampled same-time RssAnon + deduplicated fdinfo allocation
  metric from child exec through exit, not total physical device memory.
  Private DSP/driver/VTCM and between-sample transients may be absent; do not
  add VmHWM/RSS/arena size. See parent README for the full method.
- All normal runs and profile-level1 runs have matching65 IDs and no known
  competing process observed. Instrumented runs are excluded from TPS medians.
- postpush-convion-profile-level2 deliberately enables the existing diagnostic
  mode, which changes materialization and host execution paths. It exits0 but
  differs from the normal generated IDs, as already observed on the earlier
  baseline. The harness correctly marks it invalid for normal comparison.
  Retain it for internal stage diagnosis only; do not use it for correctness,
  headline TPS, or an estimate of costs in the normal fused execution path.
- One core build initially stopped because the generated ARM stub header's
  timestamp preceded the IDL. Regeneration reproduced the exact same stub
  and header hashes, and the subsequent Android build passed. Both logs are
  retained. Full Windows/Tizen builds and remote CI were not run.
- No change to cache capacity, preload count, weight quantization, reader
  count or the benchmark timer was adopted. The1000/25TPS targets remain unmet.

The parent followup_conv_ion/ records the accepted9d8706182 change and its
performance/correctness/memory evidence. manifest.json hashes every file in
this directory except itself.
