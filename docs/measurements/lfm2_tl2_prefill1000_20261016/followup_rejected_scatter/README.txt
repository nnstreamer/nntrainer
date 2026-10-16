Rejected LFM compact-i24 scatter partition
========================================

Production was restored to 03d652fc0. The candidate gave each worker an aligned
four-row bulk and a balanced tail, retaining exactly its original row count.
It reduces repeated loading/unpacking of the three i24 byte planes at worker
boundaries. Its guard excludes Gemma/generic/int32 layouts. Heap, ION, VTCM,
expert cache, preload and worker counts did not grow; v81 worker stack +8 B.

Actual HVX/libnative ASan/UBSan checks passed 24,576 serial and 240 actual-pool
cases, each with three repeated expert contributions. Row coverage and each
lane's original row count matched in 4,160 partition conditions. A deliberately
missing tail row was detected. Full production v81 compilation passed.

The final candidate's simulator used the normal 1024 route histogram and the
actual 32-tile scatter. Weighted maximum-lane cycles fell 2.19-9.84% across
3-8 workers; seven workers improved 5.15%. This measured each lane separately
and did not simulate concurrent HMX, expansion, DMA or preceding requantization.
The first naive split was rejected before phone testing: its seven-worker
maximum-lane cycles increased 3.70% despite reducing total lane work.

S26 Ultra, native TL2 model, C16/PAGE_CACHE=0/Direct I/O, growth=0, preload128:

  Three runs per variant        Control       Candidate
  1024 median prefill TPS        825.141        846.281
  1024 median decode TPS          17.4244        17.7040
  1024 max complete peak KiB   2,733,236      2,733,124
  2048 median prefill TPS        824.809        782.277
  2048 median decode TPS          16.7320        16.7145
  2048 max complete peak KiB   3,052,332      3,052,584

Each model used predeclared A1 B1 B2 A2 A3 B3 order. The 1024 median improved
2.56%, with substantial run variation; 2048 regressed 5.16%, slower in 2/3
pairs. A separate single 2048 correctness run is retained outside these
medians. All runs generated the reference 65 IDs, with no known concurrent
benchmark. Model SHA256 is unchanged:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d

One diagnostic pair used the real plain MoE dispatch, PROFILE=1, and a
temporary stage timer. Excluding the load warmup leaves 38 layers and 3,776
blocks, identical between variants:
  Whole DSP MoE:        377.807 -> 378.811 ms.
  Exposed scatter wait:  24.267 ->  26.747 ms.
The simulator saving therefore was not reproduced in exposed device waiting.
Concurrent cache/worker scheduling effects are possible but unproven. This
candidate is not adopted; the positive 1024 result alone is insufficient.

Memory covers child exec/load through exit, requested every 20 ms, as the
maximum same-sample RssAnon plus fdinfo allocation sizes deduplicated by
(mnt_id, ino). Every retained peak has complete fd/read identity coverage.
These allocated sizes are not total physical device memory; unreported DSP
heap/driver/VTCM memory and between-sample spikes may be absent. Do not add
VmHWM, RSS or arena size. Peaks are about 2.607 GiB/2.911 GiB at 1024/2048.

results.json contains individual runs and medians; raw/ contains full logs,
memory traces and IDs. artifacts/ identifies deployed binary/source hashes
and the diagnostic-only build delta (NNTR_LFM2_MOE_STAGE_TRACE=1). validation/
contains frozen candidate/reference sources, fixture, commands, simulator
and sanitizer results. Sources compressed as .c.gz can be unpacked before
using run_check.py. The final patch is 91a74ff10408ccd4a07304eeb73ba34bcf88a1e2e14477d4fff24f77c6be0390.
No new Gemma phone test was run. These local checks are not the full CI matrix.
manifest.json hashes every retained file except itself.
