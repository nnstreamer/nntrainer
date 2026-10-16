LFM native TL2 expansion: compiler-visible disjoint source/destination
===================================================================

Frozen production: production.patch
SHA256 59306fe9542a22cd278f0ed08fbee5a7cbcafbee2d2515da88a045bbe7af971e
Frozen tests: tests.patch
SHA256 6c345f4fcfe7bf7fda059a049857297c5d8b72544725c0e1036ad42dfc40f6e7

The baseline/ and candidate/ overlays were made from commit
83b3235830771ed2313135393efb48d7efaf549b. The parent applied this same patch on
8288e3ba3dff986e0c72f37c8db4f3adec4c628c, which additionally contains the separate
M1 setup removal. That change does not alter moe_expand_unit. Exact overlay
source hashes are in source-hashes.json. The parent's integrated MoE source has
SHA256 816647df631873ebb5dfef9ceb12c368063736313e612cfe0c33bed1af6e09bc.
This report does not attribute any device result to the helper simulator.

Scope and ownership
-------------------
The existing TL2 expansion body is shared by the original public entry and a
new __restrict-qualified entry. The compiler may retain packed vectors across
the existing stores in the new entry. Arithmetic, lookup tables, WH layout,
allocation sizes, panels, jobs and worker sequencing are unchanged.

Only existing disjoint native slabs with these LFM dimensions select it:
  GU: native_K=2048, native_npanels=24, row stride=96*512 (N=3072)
  DN: native_K=1536, native_npanels=16, row stride=64*512 (N=2048)
Both also require native_base and native_packed. These packed slabs lie after
the WH destination region in the existing VTCM layout. The eligibility test is
computed once per moe_expand_unit job, outside the panel/group loops.

The generic entry is retained. Other shapes, Gemma shapes and the existing
in-place staging path select that entry; in-place staging still snapshots its
source before expansion. The GU final 32-row group retains its 1280-byte source
bound; the DN has full 96-row groups. There is no new weight cache, activation
allocation, reader, preload, expert capacity or context field.

Evidence
--------
1. checks/hoisted-asan/run.log and results.json:
   Actual SDK libnative HVX arithmetic passed 1440 primitive cases against the
   original entry and an independent nibble/WH oracle, and 72 cases executing
   the actual extracted moe_expand_unit and context type. Cases cover 0/32/64/96
   rows, 0/32/64/96/128 columns, aligned/unaligned input and output, magnitudes
   1/7 plus scalar fallback, packed palette patterns, source bounds, destination
   canaries, LFM/Gemma/near-shape selection, sparse panel masks, in-place snapshot
   and the eight actual job partitions. ASan and UBSan were enabled.
   All three deliberate selection mutations were rejected: bypassing the packed
   slab requirement, omitting the GU stride guard, and omitting the DN guard.

2. codegen/summary.json and codegen/{baseline,candidate}:
   Four full production translation units compiled for v81, -O3 -fPIC, with
   stack-usage output. After normalizing comments/local symbols/label numbers,
   the original generic function's instruction sequence is unchanged.
   For a full 96-row panel:
     packed vector loads: 36 -> 20
     lookup table vector loads: 3 -> 3
     output vector stores: 48 -> 48
     VLUT instructions: 96 -> 96
   The generic helper frame remains 176 bytes; the new disjoint helper has a
   192-byte scalar frame (+16 bytes). moe_expand_unit remains 2816 bytes. No
   direct SP/FP vector memory accesses or vector spills were observed. The
   native shape checks are outside the panel/group loops; each expansion keeps
   only the branch that selects the generic or disjoint function.

3. checks/cxx/commands.json and logs:
   Existing g++17 commands compiling hvx_gemv_u8tl2.c as C++ passed for native
   packing, prepared TL2 (legacy and unit fixtures), prepare_tl2 utility, and
   model conversion. __restrict is confined to the .c definition; the header
   remains C/C++ compatible. always_inline is guarded for GNU/Clang compilers.

4. checks/sim/run.log and summary.json:
   Standalone v81 --timing, 12 cases, 500 repetitions plus 10 warmups each,
   comparing original and disjoint helpers. All output bytes match.
     32 rows: approximately 202.12 -> 180.11 cycles/call (-10.89%)
     64 rows: approximately 384.12 -> 346.11 cycles/call (-9.90%)
     96 rows: approximately 566.12 -> 512.11 cycles/call (-9.54%)
   This uses ordinary simulator memory, not device VTCM. It excludes caller
   selection, pool scheduling, DMA, HMX overlap and model I/O. No device TPS or
   exposed-wait improvement follows from these helper-only results.

5. Parent integration, reported separately:
   root-dsp-build.log: full DSP build passed on 8288 plus this patch.
   root-core-build.log: Android core had no work because these HVX C sources are
   excluded from that build. root-integrated-asan.log reports the same 1440+72
   exact cases and three expected mutant rejections against the integrated
   source. The parent controls device execution and performance decisions.

Historical work estimate, not a bandwidth claim
-----------------------------------------------
The earlier 1024-token route record contained 2283 active expert instances.
For those dimensions there are 760 full 96-row panel/groups and 24 GU 32-row
tails per active expert. Removing 16 packed vector loads per full group and
four per tail corresponds to 3,581,497,344 fewer logical vector-load bytes in
that historical workload. This is not DDR traffic, measured bandwidth, serial
time, or a prediction of end-to-end speed. Expansion overlaps DMA/HMX and the
older probe cannot isolate its exposed cost. Parallel SwiGLU worker time must
not be added to serial kernel time.

Reproduce the portable fixture
-----------------------------
Set the following paths for the local checkout/toolchain/output directory:
  REPO=<full checkout with test/htp/host/stub>
  SOURCE=<candidate overlay or patched checkout>
  REFERENCE=<pre-change overlay or checkout>
  HEXAGON_TOOLS=<SDK/tools/HEXAGON_Tools/version>
  OUTPUT=<new output directory>
Then run:
  python3 "$SOURCE/test/htp/host/run_tl2_expand_disjoint_checks.py" \
    --repo "$REPO" --source "$SOURCE" --reference "$REFERENCE" \
    --hexagon-tools "$HEXAGON_TOOLS" --output "$OUTPUT" \
    --sanitize --mutations

compile-commands.json records exact production v81 commands; checks/cxx and
checks/sim contain their exact commands. All tests operate in their output
directories and do not edit the supplied source trees.

Limits and archive
------------------
The host test uses SDK libnative and a correctly unaligned HVX type; it does not
simulate DMA. Eight actual caller units execute in a permuted serial order, so
the fixture makes no new worker-race proof. Existing ownership and submit/wait
ordering are untouched. Exact device output, scratch peak and model timing
remain the parent's verification responsibility.

production-v1-c99-restrict.patch and production-v2-inner-guard.patch are retained
only as superseded drafts. Do not apply them. Independent review is recorded in
the sibling's separate review directory and is not replaced by author checks.

Final independent review
------------------------
/root/gemma_prefill_port completed frozen59306/tests6c345 review with no
unresolved blocker. It independently reran 1440+72 ASan/UBSan cases, all three
mutants, existing g++ consumers, C11/C++17 header checks, four v81 translation
units, and ownership/range checks. See:
  /tmp/nntr-lfm2-tl2-expand-review-20261016/review.txt
  /tmp/nntr-lfm2-tl2-expand-review-20261016/validation.json
Its verdict permits root device parity/performance testing and does not claim
a device gain or use the author-only simulator cycle measurement.
