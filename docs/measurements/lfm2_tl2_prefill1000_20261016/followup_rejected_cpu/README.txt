Rejected LFM CPU selection and reader-barrier trials
==================================================

Neither candidate is enabled in production. Both were tested against
03d652fc0 on S26 Ultra with the existing native TL2 model. Expert weights,
C16, cache growth=0, PAGE_CACHE=0/Direct I/O, preload128, 608 expert slots,
32 decode spares and reader ceilings were unchanged. Every raw run JSON
records the full command and environment. Gemma was not benchmarked again.

Exact top-k insertion
--------------------
The LFM sigmoid router selects a sorted prefix of 4 or 9 from 64 scores.
The candidate replaced partial_sort with allocation-free insertion, retaining
score arithmetic, equal-score ID order and fallback for unsupported inputs.
Host ASan/UBSan and Android fast-math checks passed 16,384 exact selections
and 335 unchanged-input rejection cases. A captured real-score replay also
matched all 38,912 rows and was faster in isolation. The 19,922,944-byte
score capture is not published; its SHA256 is in validation/decision.json.

Predeclared normal order A1 B1 B2 A2 A3 B3, 1024 input + 64 decode:
  Median prefill: 828.479 -> 819.856 TPS (candidate slower in all 3 pairs).
  Median decode: 17.5921 -> 17.5728 TPS.
  Maximum complete sampled peak: 2,733,492 -> 2,732,240 KiB.
One separate diagnostic pair measured router top-k 24.839 -> 27.658 ms.
These model results do not reproduce the standalone microbenchmark gain.
No specific code-generation cause has been established. Source and tests
were reverted. The single 2048 run is correctness evidence, not a speedup.

Bounded reader barrier
---------------------
The candidate wakes and joins only the workers needed by the current batch,
instead of waiting for all pool workers when the caller and one worker can
cover two jobs. It adds no readers or buffers (pool bookkeeping +8 bytes on
the tested host ABI). Late worker, exception, concurrent caller and actual
Direct I/O pair checks passed ASan/UBSan. The normal A/B uses one binary and
changes only NNTR_LFM2_DECODE_READ_BARRIER=0/1.

Normal order A1 B1 B2 A2 A3 B3, three runs each:
  Median decode: 17.6066 -> 17.6747 TPS (+0.39%, 2 of 3 pairs faster).
  Maximum complete sampled peak: 2,733,600 -> 2,732,576 KiB.
One separate PROFILE=1 pair measured exposed miss wait 30.7374 -> 30.8614
ms/token and ARM service 1.056 -> 1.058 ms/round. Thus the small normal TPS
change does not establish reduced I/O waiting. The source/tests were reverted.
The flag affects decode only; prefill differences are not its speedup.

Correctness, memory and reproduction
-----------------------------------
All retained full-model runs matched the reference generated 65 token IDs,
with no known concurrent benchmark. Normal 1024 IDs SHA256:
b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e
The native TL2 model SHA256 remains:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d

Memory is sampled over child exec/load through exit, requested every 20 ms:
maximum same-sample RssAnon plus fdinfo allocation sizes deduplicated by
(mnt_id, ino). Retained peak scans have complete readable fd identities.
This is approximately 2.607 GiB, not total physical device memory: sizes may
not be resident, unreported DSP/driver/VTCM storage and brief spikes may be
absent. Do not add RSS, VmHWM or arena size to this metric.

results.json preserves every run and group medians. raw/ within each trial
contains IDs, run JSON, compressed full logs and memory traces. validation/
contains rejected patches, checks, runner inputs and decision records.
artifacts/ ties deployed binaries to exact source deltas. manifest.json
hashes all retained files except itself. Build/test excerpts are local
checks, not the full cross-platform CI matrix. No 1000/25 TPS claim is made.
