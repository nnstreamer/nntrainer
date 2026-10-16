LFM dynamic-Q8 attention: overlap score conversion with current PV
=================================================================

The final change reuses the existing gathered-Q scratch after its complete
chunk has been quantized. Next-block raw QK readouts occupy that dead region;
workers convert them into the other score bank while the caller computes
current P.V and updates O. Every worker joins before the next softmax, Q
preparation or error return. Arithmetic and per-row accumulation order are
unchanged. No new weight cache, activation cache, ION, heap or VTCM allocation.

Scope: full dynamic-Q8 LFM attention, >=2048 rows divisible by 128, 32/8 heads,
head dimension 64, no window/softcap/sinks, existing 128/256/512 tile geometry,
real workers and checked scratch extents. Fixed-Q8 Gemma and 1024-row LFM retain
their serial paths. No new Gemma benchmark was run for this LFM-only change.

Fixed comparison settings
-------------------------
S26 Ultra R3CY70LV96T, native TL2CX_WH expert weights, affine QS4CX_WH dense
weights, FP32 router. Model SHA256:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d
C16, 608 slots, preload 128, PAGE_CACHE=0 Direct I/O, CACHE_GROW=0, six prefill
readers and existing decode readers/spares. The predictor/handoff settings
are identical in both variants and recorded in every raw run JSON.
Only the DSP library changes. ARM core/app artifacts remain those of the
committed conv-ION implementation 9d8706182. Sources and binaries have hashes.

Normal whole-model results
--------------------------
Final >=2048 guard, three interleaved pairs per prompt length:

  input   baseline PF median   candidate PF median   decode baseline/candidate
  1024       810.768 TPS            844.884 TPS          17.6796 /17.6454 TPS
  2048       730.906 TPS            784.374 TPS          16.6363 /16.6884 TPS

The 1024 overlap branch is disabled: its observed difference is run variance,
not a credited improvement. The 2048 median improves 7.32%, with two of three
pairs faster. Ranges overlap: baseline 726.757..849.440, candidate 774.584..810.447.
Decode is unchanged by this optimization; no decode speedup is claimed.
These results do not reach 1000 prefill TPS or 25 decode TPS.

The earlier candidate enabled 1024 too. It regressed all three 1024 pairs,
median 844.188 -> 827.141 TPS (-2.02%), despite shorter attention in a profile.
Its 2048 pairs were all faster, median 729.605 -> 759.644 (+4.12%), with a high
895.496 outlier. Those results selected the narrower final scope. All initial
and final runs are retained; the outlier is not a steady-performance claim.

Normal generated outputs match all 65 IDs (prefill 1 +decode 64) in every run:
1024 SHA b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e
2048 SHA 2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33
The harness saw no competing known inference/checker process in these runs.
Device thermal snapshots and per-run commands are retained. This is a small
serial device sample, not a controlled clock/thermal significance study.

Peak memory
-----------
Maximum complete sampled process RssAnon + deduplicated fdinfo allocation
sizes, from child exec/loading through exit (KiB):
 1024 baseline 2733776, candidate 2734240 (candidate 2.608 GiB)
 2048 baseline 3053396, candidate 3052524 (candidate 2.911 GiB)
All selected final normal-run peak scans have zero unreadable fdinfo files,
known identities and complete deduplication for their readable size fields.
This is not total physical-device memory: DSP private heap, driver memory
and VTCM may be absent, allocated sizes may exceed resident pages, and the
requested 20ms/observed~21ms sampler can miss transients. The independent
in-app 100ms estimate is also retained and can differ. Neither is added to
VmHWM/RSS/arena totals. See memory_monitor.cpp and the first-phase README.

The raw-score bank uses 524288 bytes of existing Qx. At 2048 rows Qx already
has 1179648 bytes and has been written by the preceding Q preparation. No
new allocation or previously untouched scratch pages are introduced.
v81 compiler frames: caller +128B, new worker 384B, below the existing largest
softmax worker frame 1920B and 32KiB pool stack. These are not full stack peaks.

Diagnostic evidence
-------------------
One final 2048 profile-level 1 pair, all 65 IDs exact:
  ten-layer attention elapsed 472.913 -> 362.996ms (-109.917ms)
  score-conversion field 157.015 -> 36.386ms
  exposed softmax 31.833 -> 44.891ms
  QK 85.030 -> 84.363ms, PV 24.890 -> 24.786ms.
The score-conversion field changes meaning with overlap: first-block serial
conversion plus submission and exposed final waits, not all worker busy
time. Do not sum it with an independent worker-duration estimate. Whole
attention elapsed and separate normal-run TPS are the relevant comparisons.
Profiles include instrumentation and do not enter the normal-run medians.

Correctness/build checks
------------------------
- Final actual-HVX/libnative + real worker-pool host check: 460 cases under
  ASan/UBSan, 18 shape/extent guards repeatedly checked, three intentionally
  broken addressing/extent/drain mutations rejected. HMX/DMA/softmax/PV are
  stand-ins in this fixture; it is not a device performance test.
- Actual DSP replay: 21 attention/KV/router cases, including 2048/4096 rows,
  Q4, windows, softcap, sinks and other head sizes. All output bytes match
  baseline, initial candidate and final candidate; hashes are retained.
- Same-handle two complete 2048+64 runs match each other and the normal
  reference. Initial candidate same-handle 2x1024+64 also matched.
- The first 2048 repeat attempt used the old 1024-only checker and failed its
  explicit input-shape precondition before inference. Its logs are retained
  under excluded/. A separately built 2048 checker then passed; production
  model code was not changed to accommodate the checker.
- Complete v81 DSP build passes; v79 production-source compile and independent
  lifetime/numerical review pass. Full platform matrix/CI was not run here.

Source/binary provenance details
-------------------------------
conv-ion-rebuilt was compiled from clean 9d8706182. Its deployment helper
captured git diff after the later candidate edit, although the already-built
binary hash 83170a05... was recorded and verified before inference. The
manifest now states the empty compiled-source patch and preserves the later
worktree snapshot separately as *.after-build-edits.patch.gz. This is a
metadata correction, not a replacement of benchmark binaries or results.

One earlier default adb push produced a mismatching destination hash; the
verification stopped deployment before inference. Retrying with adb push -Z
matched. Every measured runtime was SHA-verified; a compression bug was not
established. Subsequent deployment helpers use -Z and retain verification.

After a separate temporary MoE diagnostic, its source changes were reversed
and the production DSP was rebuilt successfully. The rebuilt ELF has a
different hash from the tested ee66d16b... library; section hashes are saved
without assuming byte-reproducible linking. All results here refer to the
SHA-verified attn-overlap2048 runtime, which remains intact on the device.

Reproduction
------------
Normal commands and env are in raw/*.json; model/runtime/fixture directories
refer to this device workspace. Rebuild the DSP with the SDK/HexKL settings
from repro/dsp-build.sh. The host checker is tools/lfm2_attn_overlap_check.py;
its usage and simulated-vs-real boundaries are in test/htp/host/
attn_overlap_check.txt. The repeat2048 build helper generates only a temporary
standalone test CLI from Applications/CausalLM/main.cpp. Replay uses tracked
test/htp/lfm2_prefill_replay.cpp. The manifest hashes every retained file;
large text logs and memory samples are gzip-compressed without modification.
