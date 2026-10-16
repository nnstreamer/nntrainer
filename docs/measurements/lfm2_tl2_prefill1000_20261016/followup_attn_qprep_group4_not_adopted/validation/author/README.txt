Cached LFM2048 Q preparation without the intermediate Qx gather

Production: production.patch a07332c9043b6760a42392cae06a237319e6a1e383829edc0bee5d838b435df8
Tests: tests.patch f962d48fabdb73ce2702333f67907b8c5f5ddd86e17c01c38d37048ea66db23f
Exact changed source and fixture hashes: source-hashes.json / tests-hashes.json.
Base source is root742735f3, with no production root/device/commit changes by this author.
This author bundle contains host/codegen evidence. Root maintains device dispatch,
accuracy, peak memory and normal performance separately. No TPS gain or bandwidth
claim is made from the host results.

Scope and operation
- Only q_step's successfully staged cached Q, dynamic Q8, n_q/cache_to2048,
  cache_from0, 32Q/8KV heads, hd64, full unwindowed/no-softcap/no-sinks attention,
  G4/Br128/Bc256, original stride2048 and complete query chunks can use it.
- q_step sets one call-local q_cached capability only after existing K/V/Q DMA
  copies finish. Allocation failure, direct prefill/decode,1024, fixed-Q8/Gemma,
  unsupported shape/tiling, alias or missing capability retain the old path.
- The new group4 worker calls the original quant_row_params_one and
  pack_group_rows bodies with the same four head rows, in the same order, using
  the same floor(groups*lane/threads) ownership and synchronous pool run.
  Scale1/zp0 initialization and final scale multiplication are unchanged.
  Generic quantization and padded-gather code remain available without edits.
- A query row's four consecutive hd64 heads form one group; successive groups
  are2048 floats apart in cached Q. New code reads those addresses directly.
  It removes the serial Q→Qx gather; it does not add another FP calculation.

Ownership and capacity
The existing RPC owns a completed cached Q slice for the synchronous HMX call.
It is disjoint from output, VTCM, scales and zero points; range guards verify
those relationships without comparing content or retaining pointers. No state
survives the call. Existing HMX/session serialization and DMA drain are used.
The last source endpoint for either9+7 chunk is within the16MiB Q allocation.

Qx remains in the original plan and allocation:1,179,648B. The existing2048
pipeline's deferred raw QK readouts use524,288B there. First QK uses L.acc;
subsequent QK readouts overwrite every complete raw tile before the dequant
worker reads it. Existing worker joins, PV ownership and failure drains remain
unchanged. The tests poison Qx and exercise the existing overlap/readout path.

No activation/cache/reader/pool/scratch capacity or allocation is added. The
existing cached K/V/Q/O staging remains36MiB for2048. The private DSP IO record
is24→28B; all original offsets stay0/4/8/12/16/20. This is4B call-local metadata,
not a persistent cache. v81 prefill1536B, phase_qprep512B and q_step336B frames
are unchanged; new group4 worker640B/helper32B fit the existing worker stacks.

At2048,10 layers ×8 heads ×2 chunks =160 group4 helper calls. Per head the9+7
chunks contain1152/896 groups, covering65,536 head rows/layer. The logical
eliminated gather payload is16MiB/layer,160MiB/prompt. This is not DDR traffic:
the quantizer now rereads cached source rows instead of VTCM Qx; cache locality
and dispatch/guard overhead can offset the copy saving. Prior49.639ms Q-prep
covered gather plus quantization/packing; it is not the saving estimate.

Build boundary
The appended private IO field changes an internal C layout, so do not mix old
producer/consumer objects. Actual original compiler dependencies identify:
  test/htp/nntr_hvx_attn_q.c                  (all3 IO producers)
  nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c
  nntrainer/tensor/htp_backend/hvx/hvx_attn_decode_q.c
The implementation adds the fourth recompiled TU, hvx/hvx_quant_u8.c. A separate
hvx_quant_u8_group4.h avoids modifying the widely consumed hvx_quant_u8.h.
No IDL/ARM change. Root's fixed-object pair retains59 identical common objects,
recompiles these4 candidate TUs and uses the same single-thread linker. This
bundle compiles affected TUs only, and does not claim to reproduce old DSP3b.

Validation
- checks/frozen-strict: actual phase_qprep + SDK libnative quantizers,576
  parity cases (AH bytes, scales, zero points, source/canaries, Qx poison),
 31 capability/shape/alias guards,5 intentional mutants. Includes both real
 9+7 chunks, all8 KV heads, NULL/3-worker pool, ±zero/constant/ties/mixed finite
 patterns; unsupported/padded/fixed-Q8 cases exercise the original branch.
- checks/rpc-final:15 actual RPC-body capability/lifecycle checks and3 mutants.
 Reserve, DMA, append and attention kernels are explicit stand-ins; this proves
 propagation/order/fallback, not hardware cache coherence or full attention.
- checks/overlap: existing actual QK/readout/dequant/block-loop test,460 cases
 and3 mutants; Qx starts poisoned and PV clobbers the old accumulator. HMX/DMA/
 softmax/PV are stand-ins; the HVX conversion and pool implementation are real.
- compile-frozen.log, codegen/: baseline/candidate4 TU v81 object+assembly PASS.
 Existing11 quantization function bodies have identical object bytes; the
 decode kernel object is byte-identical. Prefill controller instruction text
 differs only in worker-symbol relocation offsets; unchanged widen helper's
 normalized instructions match. The new helper code is separately dispatched
 and uses the original row/pack implementations without changing their math.
- All5 production files passed changed-lines clang-format14.0.6 no-op recheck.

Test-harness correction (retained evidence)
The initial tests-v1-ffb26.patch mapped HVX_UVector to the libnative host vector
with8B alignment, while deliberately testing q_alloc+1 at4B alignment. Author
checks/final had8 recoverable UBSan alignment warnings and still exited0.
Its initial sanitizer-PASS description is superseded. Independent strict halt
caught this. Original logs and patch remain preserved. The final test-only
attn_qprep_native_unaligned.h typedef matches SDK HEXAGON_UVect1024's aligned(4)
contract, asserts size128/alignment4 and leaves all math unchanged. No sanitizer
suppression is used. Final runner uses -fno-sanitize-recover=all and explicit
ASan/UBSan halt settings. SDK prebuilt libnative itself is not instrumented.
An intermediate attempt to add may_alias after a struct definition failed a
compiler warning; only aligned(4) remains in the final fixture.

Reproduction (R=complete repo; D=this bundle; T=Hexagon tools19.0.04 root)
python3 D/candidate/tools/lfm2_attn_qprep_check.py --repo R --source D/candidate --hexagon-tools T --output-dir OUT/quant --sanitize --mutations
python3 D/candidate/tools/lfm2_attn_qprep_rpc_check.py --repo R --source D/candidate --output OUT/rpc --sanitize --mutations
UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 python3 R/tools/lfm2_attn_overlap_check.py --repo R --source D/candidate/nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c --hexagon-tools T --output OUT/overlap --sanitize --mutations
After applying both patches to a complete repo, omit --source and use that repo
for the overlap command so all tests see the updated IO header.
The source-only audit is separate: /tmp/nntr-lfm2-prefill2048-waste-audit-20261017.
Diagnostic-only actual-entry evidence instructions: diagnostic/README.txt.
