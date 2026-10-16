Rejected native LFM decode partial-ready trial
=============================================

The candidate is not retained in production. It let each completed expert
read publish an immutable mailbox record, rebind on DSP, and compute into
its original output row while other reads continued. Final accumulation
order and the final owner join were preserved, including error exits.

Tests used the same native TL2 model and production base 03d652fc0 on S26
Ultra. C16, cache growth=0, PAGE_CACHE=0/Direct I/O, preload128, 608 expert
slots, 32 decode spare slots and all reader counts were unchanged. The
candidate reused 768 bytes of the existing mailbox and existing output
scratch; it added no weight/activation cache. Normal DSP token stack grew
by 224 bytes, graph-miss stack by 8 bytes, and ARM bookkeeping by 8 bytes.
Gemma was not benchmarked again.

Normal same-binary comparison
-----------------------------
NNTR_LFM2_DECODE_PARTIAL_READY=0/1; 1024 input + 64 decode; A1 B1 B2 A2 A3 B3.
  OFF decode: 17.4577, 17.6649, 17.5824 TPS.
  ON decode:  17.1720, 17.1169, 17.1628 TPS.
  Medians: 17.5824 -> 17.1628 TPS (-2.39%); all three pairs slower.
  Maximum complete sampled peak: 2,733,428 -> 2,733,652 KiB (~2.607 GiB).
All generated 65 token IDs matched the reference. Prefill is outside this
feature's selection guard, so its measured differences are not a feature
speedup. The clear decode regression did not justify more normal runs or
2048/repeated-prompt adoption checks. Source and tests were reverted.

Separate diagnostics
--------------------
The first diagnostic pair used FARF(ALWAYS), which maps to HIGH in this
SDK; no per-token trace reached logcat. Those raw runs remain archived and
are not evidence of zero early work. Version 2 changes only that diagnostic
logging level to FARF(ERROR), separately from the normal benchmark binary.

Version 2 has exactly 64 successful token records per run and unchanged
65 output IDs. ARM published 1,291 partial rounds; DSP saw the same count.
Only 29 rounds received the final answer first. DSP actually computed
2,250 expert rows in 2,219 early batches, about 1.014 rows per batch, with
early work in every token. Thus the optimization did execute.

In that single diagnostic pair:
  MoE calls/token: 67.89 -> 97.95 (+44.28%).
  MoE time excluding miss wait: 9.558 -> 10.472 ms/token.
  Exposed miss wait: 30.6235 -> 31.0810 ms/token.
  ON ready wait + final wait: 24.9507 + 6.1277 ms/token.
The extra split calls increased non-waiting MoE time without reducing total
wait. Net MoE also includes rebind, polling and grouping; the profile does
not isolate each repeated LUT/quantization cost. Diagnostic
TPS is not used as a normal performance result; the three normal pairs
independently establish the regression.

Validation and evidence
-----------------------
Actual graph/token lifecycle 103 cases and production read scheduler with
real O_DIRECT 84 cases passed ASan/UBSan, including failure drains and
prediction adoption. Independent review repeated these checks and tested
unequal expert weights and whole-token paths. Three intentional mutants
(output-row alias, skipped error drain, duplicate rebind) were rejected.
Android ARM and full v81 DSP builds passed. These are local checks, not the
full cross-platform CI matrix. Host arithmetic stand-ins are distinguished
from actual device output checks in the retained reports.

Model SHA256:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d
1024 reference generated-ID SHA256:
b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e

Memory covers child exec/load through exit, sampled about every 21 ms
(20 ms requested): maximum same-sample RssAnon plus fdinfo allocation
sizes deduplicated by (mnt_id, ino). Peak scans have complete readable fd
identities. Allocated bytes need not be resident; unreported DSP/driver/VTCM
memory and brief spikes may be absent. Do not add RSS, VmHWM or arenas to
this metric. This is not total device physical memory.

raw/ contains full commands, output IDs, compressed application/logcat logs
and memory traces. artifacts/ binds binaries to exact source patches.
validation/ contains rejected patches, build logs, checks and strict trace
analysis. results.json records every measured run. manifest.json hashes
every retained file except itself. No 1000 prefill / 25 decode TPS claim is
made.
