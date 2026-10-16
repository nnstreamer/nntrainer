LFM conv pre-gate + conv/post-gate Tile8 experiment

Status and scope
Production is opt-in only (NNTR_LFM2_CONV_GATE_TILE=1). The environment value is
read only after the existing native LFM shape/type/layout guard. This patch is
relative to ccea0d476 (conv chunk v2, source SHA846921ea...), and changes only
Applications/CausalLM/layers/conv_block_layer.cpp. No core, DSP, quantization,
cache size, read lanes, preload, model dimensions, or buffer capacities change.
Root owns integration and all device work. This is not yet an end-to-end gain.

Production files
production.patch SHA256 ba054f78c730430438408231fbb465841b9acdebb5bf06e7f4b3f2e66f5427cc
source SHA256 afb5021ff5f95ed4f1287357b5dce77bc8032c9ec07045c5f5944f1e3c305e8c
frozen.json lists the fixture, patch, native executable, and ION micro hashes.
The sibling parent directory holds the frozen Tile16 experiment; it is not
silently replaced. variant8/ is the selected Tile8 candidate. No other tile
sizes were implemented.

Ownership, arithmetic and scheduling
Only native LFM FP32 M1024/C2048 full projection and M2048/C2048's existing
1920+128 projection chunks qualify. Helper geometry accepts only full1024,
[0,1920) of2048, or[1920,2048) of2048. Existing app type/contiguous/from/state
ownership and exact weight shape guards remain. Gemma, decode, compare/shadow,
unsupported shapes and non-AArch64 use their prior paths.

Each independent8-row task retains two previous gated vectors in registers.
It recomputes the two-row halo from the current projection, so neighboring jobs
need not execute in order. At the beginning of the second RPC chunk, it reads
only gated1918/1919 left by the first completed dispatch. The next RPC can
invalidate the entire first ION projection. Only gated1918/1919 and2046/2047 are
stored in2048 mode;1024 stores1022/1023. The final state copies run after all
projection RPCs and CPU dispatches complete. Failed RPC/sync leaves state
unchanged; a retry recomputes the required history. Data outside those rows in
the existing gated tensor is not part of the output contract.

The operation order is rounded gate multiply(C*A), rounded current*weight0,
FMA(previous1,weight1), FMA(previous2,weight2), then rounded post multiply.
The order C*A matches the actual AArch64 gate_pre instruction operand order
of the unchanged scalar-source A*C loop, including NaN selection priority.
Precise intrinsics and float-control scopes match the existing full helper.
Source arithmetic order by itself is not proof for NaNs: the full ARM object,
native special-value fixture and actual ION micro all form the evidence.

The new helper rejects overlap between its output/gated/projection/weights
before any write. Outer chunk guards additionally retain input/state disjoint
checks and view validation. There is no new ION view lifetime and no second
projection staged ahead of consumption.

Capacity and bounded temporary overhead
2048 RPC capacities remain exactly:
 act 15,732,736 +1,052,672 bytes;
 out 47,190,016 +3,149,824 bytes.
The existing host projection/gated/output/state tensors remain allocated at
exactly the prior sizes. Tile8 neither shrinks nor grows them. It reduces
written/read gated data and dispatches:1024 uses128 tile jobs/one dispatch;
2048 uses240+16 jobs/two dispatches. The first and second chunks use the same
completion barrier as before.

Whole Android -O3/-ffast-math object codegen:
 app entry1760B before/after; chunk helper464B before/after;
 tile helper128B; tile parallel_for wrapper96B; tile arithmetic callback0B.
The newly outlined nested call frames add224B on the caller's active stack;
the arithmetic worker callback spills no vector or scalar data to its stack.
The7 old gate/post callbacks have identical instruction words. That does not
prove unchanged whole-program timing for unsupported paths.

The new closure holds7references=56B. Android libc++ requests64B including its
function wrapper, versus48B for the old full-post callback and56B for the old
range-post callback. This uses the pre-existing ThreadManager task_ slot;
assignment briefly overlaps old/new closure allocations and releases the old
closure, as before. The latest task remains until the next dispatch or pool
teardown. Thus there is a small+16/+8-byte callback payload request, not zero
temporary metadata. There is no new persistent weight/activation cache. Actual
allocator rounding and full-model RssAnon+accelerator peak require device data.

Logical traffic/cost (not measured DRAM traffic)
Tile8 recomputes254 halo rows for1024 and508 for2048, +24.8047% gated multiply
work versus one product per row. It reads approximately27.969/55.938MiB of
projection values instead of24/48MiB, while avoiding almost all8/16MiB gated
writes and the following gated reads. Final gated writes are16/32KiB. Weights
are loaded once per8-row task's channel vector. The power-of-two24576B row
stride and8192B gate offsets may cause cache conflicts; observed micro results
are evidence of behavior, not proof of a cache-miss mechanism.

Validation and limits
Final actual-source fixture:112cases ASan+UBSan, opt-in ON and OFF; native
AArch64 executable cross-build. It checks full1024 and two-chunk2048, one/eight
workers with ascending/reverse/shuffled work stealing, two generations,
finite/qNaN/sNaN/Inf/zero/subnormal/ULP and distinct halo values, exact full
output and final state bytes, guard/fallback dispatch, rejected overlap,
oldION0xff overwrite, ASan-poisoned oldION and unused gated rows, repeated
buffer reuse, first/second RPC/sync failure and retry, and unchanged capacities.
The reference uses the actual older gate_pre and full post helper. DSP writes
are deterministic stand-ins. ASan is not a race detector; independent schedule
proof and real ThreadManager native execution are separate evidence.

Eight deliberately broken sources are rejected at runtime: missing boundary
halo save, reading dead gated rows, writing dead gated rows, omitted post gate,
reversed FMA order, global instead of chunk-local projection row, missing
output overlap guard, and partial-tile acceptance. Compile failures are not
counted as mutation detection. The existing committed chunk checker remains
unaltered and passes82cases with Tile8 source and option unset/OFF. Non-ARM
actual helper compilation with g++ -Wall -Wextra -Werror passes8fallback cases.

Reproduction from a checkout R and this artifact directory V:
  python3 V/test/htp/host/run_lfm2_conv_gate_tile_check.py \
    --repo R --source V/candidate --baseline B --simde-include SIMDE \
    --output OUT --sanitize --tile-on
  NNTR_LFM2_CONV_GATE_TILE=0 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1 OUT/check
  python3 V/test/htp/host/run_lfm2_conv_gate_tile_mutations.py \
    --repo R --source V/candidate --baseline B --simde-include SIMDE \
    --output MUTANT_OUT
B is the old full-helper source overlay (parent baseline/, HEAD065bbd308f).
SIMDE is external SIMDe0.8.2; no vendored repository dependency is modified.
The runner can cross-build with --cxx NDK_AARCH64_CLANGXX --native-arm
--build-only (no SIMDe include). Native execution requires the option=1.
The exact whole Android TU command is codegen/command.json; all output stays
in this artifact directory, not in the checkout. The original chunk checker
command is preserved in checks/existing-chunk-san/commands.json and its log.

Actual ION CPU-only micro (root-run S26 Ultra,8ThreadManager threads)
variant8/root-ion-micro.log/json: 12exact cases PASS, including both tile sizes
and first ION overwrite.31samples per path, six-way rotating ordering, same
ION allocations and same host/gated/output/state arrays. Baseline,Tile16,Tile8
medians in microseconds:
  1024 warm       863.490 /768.594 /619.010
  1024 host sweep1020.312 /925.157 /790.729
  2048 warm      2238.646 /2144.271 /1770.833
  2048 host sweep2502.552 /2451.719 /2066.615
This region includes sampled host projection writes and verification, CPU gate,
conv/post, ThreadManager completion, and final state copies. It excludes DSP
projection work, model execution, expert I/O contention and accelerator cache
coherency. The fixture CPU fills cached ION. Host sweep touches only the
existing host projection; it is not a guaranteed cold-cache experiment.
The correctness snapshot reuses that existing host array. Timing vectors are
fixture bookkeeping only. No extra eviction or activation buffer is allocated.

An arithmetic extrapolation of this isolated baseline-to-Tile8 delta across
30conv layers is roughly7ms at1024 and13-14ms at2048. This is neither an actual
model gain nor a guarantee under concurrent work. Root will decide from the
same-binary option0/1 normal full-model measurements, exact65dumped IDs,
same-handle repeated prompts, and RssAnon+accelerator peak. No Gemma device
benchmark or Gemma unchanged-performance claim is part of this candidate.
