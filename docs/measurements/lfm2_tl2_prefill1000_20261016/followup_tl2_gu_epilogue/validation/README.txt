Native LFM TL2 decode GU epilogue: bounded redundant-work removal

Frozen production: production.patch
  SHA256 d86b15ae408261f17728b3928dc29dd72fb375d5a772213234338c5c16f94d74
Frozen tests: tests.patch
  SHA256 4792923d1e0fa16f8f6bcc35981652a25ea56589bf3ed8d2030f192b63c3db88
Base: 7b3db6d189fa8086fa0daaa19e61eaca78a5c87c (207d299c adds evidence only).
Changed production source: nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c
  SHA256 a19d1ee954f294721d09d2eaf46286d0ee5f3682150938c22169b98f9e7013ea

The original native M1 GU epilogue copied each gate/up 32-column pair through
an aligned stack tile, then invoked the generic serial-pool SwiGLU wrapper.
The candidate uses the existing hvx_dq_swiglu_job and hvx_dq_swiglu_worker to
consume the already contiguous gate and up accumulator planes directly.
FP dequant/SwiGLU code, requantization, LUT construction, weighting, worker
partition, scheduling, joins, scratch sizes and allocation remain unchanged.
There is no new API, buffer, persistent cache, reader or production global.

The helper accepts only native TL2, K=2048/inter=1536/N=2048, both palettes
magnitude one, one K split, SILU and one row. Nonfused calls must span exactly
one aligned 48-tile expert; each lane keeps floor(48*id/threads) through
floor(48*(id+1)/threads). Fused calls retain their existing expert-major lane
assignment and process that expert serially. Magnitude7, split2/4, Gemma shape,
GELU and generic/misaligned ranges retain the old epilogue. GU accumulators
live in per-expert scratch independently of the reused DMA weight slabs.
GU jobs join before this epilogue; it returns before requantization/LUT use.

Author verification (checks/host-san-final/validation.json):
- 544 actual-helper/libnative byte-parity cases; 264 include actual production
  requantization and DN activation LUT, 88 epilogue fallback cases.
- 2664 intermediate lane comparisons, lanes1..8/reverse order/expert0..3.
- 13 exact guard/unchanged-output cases, poisoned padding and input immutability.
- Native3+QS2/unfused and native4+QS2/fused, each also mixing magnitude1/7.
- 4 mutants rejected: wrong up plane, all-lanes overwrite, split acceptance,
  magnitude7 acceptance. Every mutant was restored before the final rerun.
- Fused active1..3 tests intentionally force a private worker boundary;
  production threshold coverage is separately represented by native4 and the
  mixed native3/native4 cases.
- The actual dequant, quant and LUT C files are used. The NULL worker-pool
  branch is a serial scheduling stand-in; private structs are extracted from
  production. SDK libnative is host-width, with its missing unaligned HVX type
  supplied as an aligned(1) alias. HexKL host constants use INNER32/COL32/ROW64.
  DMA, FastRPC and device timing are outside this fixture's scope.

Independent review: /tmp/nntr-lfm2-tl2-gu-epilogue-review-20261016
  Reviewer reproduced the frozen suite and added 176 concurrent pthread-lane
  pipeline cases with actual helpers/HVX math, mixed experts and fallback.
  No unresolved production blocker was reported.

Full v81 MoE baseline/candidate compilation passed (checks/codegen/commands.json).
The epilogue worker frame remains640B and requant worker remains896B. Both
contain fallback vector stack operations, unchanged in count; the accepted
branch bypasses those tile copies. Whole-function packet counts include both
branches: epilogue144->182, requant231->261. They are NOT execution-cycle counts.
The fast branch adds bounded shape/palette checks and one existing worker job;
there is no new vector spill on that branch. Generic FP code is unchanged.

At38 layers/top4 per generated token the original epilogue makes7296 generic
wrapper and SwiGLU worker calls. It also performs14592 stack vector stores and
14592 stack vector reloads:3735552 logical bytes/token, not DDR traffic.
Actual sequence-validated trace-v2 records show 4,345 kernel calls over the
64 decode tokens and exactly 9,728 active experts / 466,944 tile pairs. All
reported pool widths are eight. There are 2,076 fused worker calls/99,648 tiles
and 61,216 unfused worker calls/367,296 tiles. Thus the new worker count is
988.9375 per token versus 7,296 originally (6,307.0625 fewer); the generic
wrapper is bypassed for all 7,296 tile pairs per token. These are observed
invocation/logical-access counts, not measured DDR traffic or cycle savings.
This is a small compute cleanup, not evidence of25TPS. Parent's normal paired TPS observations were variable; profile netMoE
9516->9471us/token is a45us difference, with no resolved throughput gain.

Diagnostics are separate and must not be used for normal TPS comparisons:
- trace-plain-rpc-archived/: superseded prototype; graph decode bypasses it.
- trace/: original kernel trace v1, patch86cd3276... . Device capture had40246
  records over the actual3.65s decode interval, only4649 distinct full lines.
  Millisecond+shape equality cannot identify calls, so v1 counts are INVALID.
  Its source, patch, host evidence and raw device log are preserved unchanged.
- trace-v2/: frozen diagnostic patch7c3f1148... adds a per-success sequence.
  Only identical sequence+payload transport repetitions may be deduplicated;
  duplicates remain reported. Unique1..N, per-call counters, complete workload
  and sentinel are required. Parent's fresh-process device run passed: raw
  log SHA0577164bedd57ffc66e8cdd32bbb0739aa57ec905af2829ded386faf0f242a59,
  21,510 raw deliveries = 4,345 unique seq1..4345 plus17,165 exact duplicate
  deliveries. One PID8595/TID8634; no missing/conflicting/malformed record;
  maximum payload100bytes with terminal sentinel. Independent token-field
  readback agrees with the strict parser. See trace-v2/device-validation.json
  and raw-readback.json. Root separately verified all65 generated IDs exact.

Reproduce author fixture with explicit local paths:
  python3 candidate/test/htp/host/run_tl2_gu_epilogue_checks.py \
    --repo <full-repo> --source <candidate-overlay> --reference <baseline-overlay> \
    --output <new-output-dir> --hexagon-tools <SDK/tools/HEXAGON_Tools/version> \
    --sanitize --mutations
The committed tests have no author-specific absolute paths. commands.json and
source-hashes.json record the exact inputs used for this author run.
