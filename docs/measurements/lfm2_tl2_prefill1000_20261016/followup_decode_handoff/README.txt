Queued LFM decode predictions handed to the existing demand pool

The same candidate runtime is compared with
NNTR_LFM2_DECODE_DEMAND_HANDOFF=0 and =1. Other flags match the optimized
LFM configuration in ../README.txt. C16, preload128, PAGE0, GROW0, six
prefill readers, two speculative decode readers, four existing demand
readers and 32 spare slots are unchanged. No new payload buffers or reader
threads are added. A Queued prediction is removed under the queue mutex;
the caller fills the already staged demand slot through readMisses.
Reading/Ready predictions retain their existing claim/join/adopt path.

Three alternating comparisons per input size (A1 B1 B2 A2 A3 B3):
  1024 input / 64 decode: 17.1306 -> 17.5969 TPS (+2.72%)
  2048 input / 64 decode: 16.1047 -> 16.4991 TPS (+2.45%)
All 12 runs preserve the complete 65-ID output. No prefill speedup is
claimed for this decode-only change. Handoff counts are 321-338 per
1024-input run; see results.json for all counters and timing ranges.
The 25 decode TPS target remains unmet.

Largest sampled full-process footprint across each group:
  1024: 2,738,988 -> 2,738,900 KiB
  2048: 3,051,928 -> 3,051,736 KiB
These differences are not evidence of a memory reduction. The monitor
and accounting limitations are the same as ../README.txt. The third
1024/on run has one fd read error at its selected peak, likely a close
race; it is retained and flagged. The maximum over that group comes from
the second run with a complete peak scan. Private DSP/driver/VTCM memory
is not fully accounted for by this metric.

The handoff option defaults off and is guarded to native TL2
K2048/inter1536/N2048 with PAGE0. Generic queue behavior is unchanged
when the option is off. Gemma was not rerun for this LFM-only followup,
per the user's request. Host fixtures retain Gemma geometry and ordinary
queue coverage, including cancellation, EOF/error, expiry and ownership.
validation.txt describes the sanitizer checks. artifacts.json records
the exact measured binary hashes and source patch identity.

Use ../bench.py with the parent's LFM flags and additionally
--env NNTR_LFM2_DECODE_DEMAND_HANDOFF=0 or =1. Keep NNTR_MOE_PREFETCH_JOIN
unset or 1. Profiled timing is excluded from these comparisons. Raw logs,
65 output IDs and complete compressed memory samples are retained here.
