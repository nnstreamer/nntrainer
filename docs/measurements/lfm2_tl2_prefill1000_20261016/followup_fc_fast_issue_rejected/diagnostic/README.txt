FC fast-issue regression isolation: AFCv2 timing plus actual branch/arena

This directory is diagnostic only. It changes no production math, dispatch,
allocation, weight/cache/reader configuration or shared MoE probes. Apply ONE
patch with -DNNTR_LFM2_AFFINE_FC_TRACE=1:

control-diagnostic.patch
  SHA256 37cfb4885ea4e9ddd47fb296e5cf548f84003ce3230edfdf7831347bac1e7d91
  Apply to normal generic FC (base7b3db6d / evidence207d, GU d86 does not touch FC).
candidate-diagnostic.patch
  SHA256 6338ebac45f01a85a6167f7417ea3ad89f2c437c7e6465a5dec67600c837da06
  Apply on frozen FC fast-issue e1e45c09574d0b131fea13d2de3822b413effb47b550d22e81c718dfe93e9e0e.
Do not apply both or stack these on the older AFCv2 diagnostic.

Exact resulting hexkl_mm_u8i4_dma.c hashes:
  control 1ffa4a59b81b4b562699cae184899105cad8d01057bff530c6a894aac3d5e34a
  candidate bda3a495e03e0eef4ca0485cad01b15e1175d9a3dbe0f81e71fc50da0083ba16

Existing AFCv2 G/T packets/timer brackets are retained. A third bounded packet:
  [LFM-AFC2B] s=hex_start f=eligible,fast_calls,generic_calls a=arena_bytes !
uses the SAME start tick as G/T. eligible is the actual lfm_hmx_inline guard
result (control0). fast_calls increments only after lfm2_fc_issue_aligned returns;
generic_calls increments after the original K loop succeeds. Each side adds
one call-local integer increment per completed tile branch. There is no global,
atomic, per-worker log or additional timer read. All three FARF calls occur
after body timing and calibration. Preserve all LFM-AFC2G/T/B raw lines.
On success, expected candidate fast_calls==reads and generic_calls==0; control
is the reverse. This verifies actual branch execution, not only enabled status.
Failure packets cannot pass the parser. Actual full-width formatter maxima:
G188/T167/B80bytes; each fits256 with64prefix bytes+NUL. Require final! as well.

The original80-main-call2048 assumption was WRONG and is not accepted here.
Profile call counts are ARM RPC counts, distinct from DSP kernel invocations:
- htp_compute_ops.cpp fcMaxRows(2048)=1920 (7.75MiB budget minus4MiB weights).
  invokeFc calls conv and ordinary N2048 FC at1920+128 rows.
- invokeLayer recognizes contiguous_conv K2048/N6144 and uses layer_norm_ld.
  It is internally chunked AGAIN by nntr_hvx_mm_u8i4.c fc_rows_a_call.
- That helper reserves512KiB for two accumulator banks. Vendor v81
  hexkl_micro_hmx_config_size disassembles to0x4000 (16KiB).
  At actual8MiB VTCM: arena=8372224; DSP step=1728.
  The B packet MUST confirm this arena; the parser rejects a different value.
- Therefore conv logical2048 -> ARM1920+128 -> DSP1728+192+128.
  QKV's one2048-row ARM call -> DSP1728+320. Ordinary N2048 stays1920+128.

2048 source-derived MAIN kernel cohort (190 calls):
  N6144: M1728 x30, M192 x30, M128 x30.
  N2048: M1920 x40, M128 x40.
  N3072: M1728 x10, M320 x10.
Separate LOAD warmup cohort (100 calls), same for1024/2048:
  M512/N6144 x1; M64/N6144 x29; M64/N2048 x50; M64/N512 x20.
1024 MAIN remains M1024/N6144 x30, M1024/N2048 x40, M1024/N3072 x10 (80).
Thus2048 requires290 distinct G/T/B identities,870 unique packets;1024 needs180/540.
There is no blanket count relaxation or cohort inferred from observed success.

M1920/N2048 has only15 accumulator tiles/bank at this arena (not32).
All other expected shapes have32. The parser derives this from the existing
activation + 2*widest_weight layout. It derives reads, joins and brackets per
call, including the extra batches at bank15. Main totals:
  1024:148480 reads,4880 joins,153840 brackets,312800 qtimer reads with calibration.
  2048:296960 reads,13390 joins,311520 brackets,635200 qtimer reads with calibration.
The existing post-head grouping leaves32 unchanged for the QKV cohort.

Parser:
  python3 parse_fc_regression.py RAW.logcat.txt --prompt 2048 --expected-fast 1 --output candidate.json
For control use --expected-fast 0;1024 is supported explicitly.
It requires complete G/T/B pairing, exact geometry/layout/branches, total
source-derived cohort, positive nonoverlapping serial call bodies, and known
clock fields. A repeated start-ID/packet is accepted only when every parsed
field matches; its duplicate count is reported. Conflicts or missing parts
reject. Multiple logcat PIDs reject. Host fixture plain lines omit logcat
prefixes; real logs preserve them. Start ticks identify serialized calls;
timestamps/shape equality never identify calls. Separate model exit/65 IDs,
PAGE0/C16/preload/readers/weight hashes remain root's validation.

Validation (checks/ and v81/):
- Actual whole kernel TU, scalar scheduling/HMX/DMA and deterministic clock
  stand-ins: each control/candidate passes50 topology/error cases, including
  M128/192/320/1728/1920, pack failure and accumulator-read failure.
- Each also runs complete180-call1024 and290-call2048 cohorts. Actual formatter
  output passes the strict parser; every HMX packet count and branch tile count
  is checked. This is diagnostic/control-flow proof, not hardware arithmetic
  or timing proof. The host arena is10MiB only for isolated M2048 unit cases;
  all real cohort calls use8372224bytes and exercise bank15 where required.
- Actual diagnostic helper:25 guard cases, unchanged options/registry/shared
  probes and no timer work for rejected guards; full-width G/T/B formatting
  passes ASan/UBSan. Branch counters do not change timer topology.
- Four actual-output cohorts, reverse packet ordering and reported identical
  replay PASS.21 malformed/branch/count/layout/error cases and174 prefix
  truncations of G/T/B all reject (checks/parser-negative.json).
- Full v81 -O3 -Wall -Werror control/candidate normal/off/on compile PASS.
  Diagnostic OFF objects are byte-identical to their respective normal inputs.
  Frames: control656->808B, candidate672->784B when enabled; all diagnostic.
  Added fixed fields are16B of call-local trace state, no persistent allocation.

Interpretation limits:
body begins after validation/allocations and before initial weight push; final
frees included, calibration/logging excluded. pack overlaps initial DMA.
read includes HMX completion and accumulator readout; it is not a pure store
cost. join/drain are exposed foreground waits. Never sum overlapped worker
service time into serial cost. Timing probes add substantial qtimer/scalar
work and can change compilation/overlap; these runs isolate phases only.
No trace TPS can establish or refute normal throughput. Normal FC12-run matrix
regressed (parent), while the later profile1 pair's FC aggregate decreased;
that aggregate includes load warmups and cannot explain main-only behavior.

Deferred feasibility note, NOT implemented; previous capacity statement corrected:
The nested conv split is real call/setup work, but a size-class key is NOT an
allocated capacity. fit() allocates roundup4096(requested_bytes)+4096. Existing
1920x6144 output needs45MiB+one page in the64MiB class; whole2048 needs48MiB+one
page in the SAME class, a real3MiB increase. The earlier wording that the class
already accommodated full2048 without growth was incorrect. Root explicitly
excludes the simple whole2048 retained-projection guard expansion under the
no-capacity-growth constraint. Aligning ARMstep1728 also changes the tail320
size-class demand and is not an approved capacity-preserving implementation.
No such production change is included. A separate read-only design audit may
consider consuming each existing1920+128 projection view before the next
backend call, using existing gated/output tensors and unchanged RPC chunking.
That requires explicit lifetime, alias, global-row/history and fallback proof;
no allocation or arithmetic equivalence is assumed from class names.

Device diagnostic audit (root/phase-summary.json and .txt):
Both actual2048 logs pass strict290-call/870-packet topology with100 load then
190 main calls. No duplicate packets. Main actual branches are296960 generic
control versus296960 fast candidate. Instrumented body267.168->216.760ms;
read56.320->73.727ms,join21.244->28.028ms,other124.436->49.597ms. This is phase
redistribution with a lower instrumented body, not an explanation of the normal
end-to-end regression. Trace TPS and empty-timer subtraction remain excluded.
Independent raw-field readback imports neither AFC2 parser and agrees exactly.
