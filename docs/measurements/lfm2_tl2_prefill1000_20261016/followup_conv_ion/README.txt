LFM2 prefill convolution: consume the existing ION projection

Change
The LFM FP32/affine convolution path consumes the projection's existing ION
view before out_proj, avoiding one full host projection copy per conv layer.
It uses the same FC arithmetic and weights. Shapes outside rows16..1920,
noncontiguous or overlapping tensors, unsupported backends and unusable views
retain the original/materialized path. In particular2048 rows still use the
existing1920+128 projection calls without increasing staging capacity.
No expert slots, reader counts, preload, weights or activation storage added.

Fixed conditions
S26 Ultra R3CY70LV96T; native LFM2-24BA2B TL2 experts/affine dense model.
Model SHA256 ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d.
C16, grow0, page-cache0 (Direct I/O), preload128, 608 expert slots unchanged.
Commands and environment are in every raw result JSON. Binary hashes and
the exact diff over222336577 are in artifacts/. The control is the already
published attn-qprep-dn4 variant. No new Gemma benchmark was run.

Results, normal1024 input +64 decode, three runs each
                         control       retained ION
prefill median TPS       821.171       833.876 (+1.55%)
prefill range TPS         792.570..822.490 813.344..853.333
decode median TPS        17.6114       17.6211 (no material gain claimed)
max complete sample KiB  2738620       2733860
The candidate is faster in all three paired prefill observations, with an
overlapping overall range. This does not establish a hardware-wide guarantee.
All65 generated IDs are byte-identical. The single2048 fallback run is also
exact (747.445 prefill,16.6147 decode TPS); it is a correctness check, not a
performance comparison. Same-handle2x1024+64 inference also stays identical.
The1000 prefill/25 decode targets remain unmet.

Copy evidence
Instrumented profiles, excluded from the normal TPS medians, show the FC
output-copy call site changing from895.5MiB/90 copies/40.9ms to
175.5MiB/60 copies/7.0ms: exactly720MiB and30 copies removed. Total measured
ARM staging copy time changes107.1 to72.9ms. This local stage measurement
does not imply that the entire difference becomes end-to-end speedup.
The profiler labels bytes as MB but divides by1024*1024 (MiB).

Peak memory method and limitations
The external monitor starts before child exec and samples through exit,
including loading. Requested period20ms; actual sample timing is preserved.
The metric is same-sample RssAnon plus fdinfo allocation sizes deduplicated
by(mnt_id,ino), not total device physical memory. Private DSP/driver/VTCM
and brief between-sample spikes can be absent; allocated size need not be
resident. Do not add VmRSS/VmHWM or the mapped arena to this value.
postpush-ion1-r3 has two unreadable fds in its selected peak scan; retain that
raw report and flag it in results.json. The table uses the largest complete
scan across each variant's runs. Its candidate maximum comes from r1 with
all readable fd identities present, not from the incomplete r3 sample.

Validation
- Android NDK r30 application build, x86 NDK Clang and Linux GCC syntax pass.
- Actual dispatch extraction:48 ASan/UBSan cases, including overlap, failed
  view, unsupported type/shape and materialization fallbacks.
- Actual DSP affine2048x6144 projection:15 cases (rows16/17/64/1024/1920,
  three patterns) preserve projection, gated values, output and conv state.
- Device normal runs,2048 fallback and same-handle repeat described above.
The portable dispatch fixture uses backend stand-ins; real numeric/DMA
behavior is checked separately by the device projection fixture. Full
Windows/Tizen builds and remote CI were not run for this change.

Reproduce from repository root
python3 test/htp/host/lfm2_conv_ion_route_check.py \
  --source Applications/CausalLM/layers/conv_block_layer.cpp --output /tmp/ion-check
The device fixture is tools/lfm2_conv_ion_projection_check.cpp. Its exact NDK
compile command and runtime log are in validation/. Run it with the tested
runtime's LD_LIBRARY_PATH/ADSP_LIBRARY_PATH, NNTR_NUM_THREADS=8,
NNTR_HTP_PROFILE=0 and NNTR_HTP_QKV_TO_HOST=0.
Use the parent bench.py with --memory and the raw JSON environment for the
full model; repro/ in the parent measurement directory documents deployment.

manifest.json hashes every file here except itself; source-sha256.json hashes
the production source and focused fixture sources committed with this result.
