Native LFM prefill tail gather: frozen bounded candidate

Production base ccea0d476daf43ddd5dbe201178755e6ff978dfd.
Production patch: production.patch
SHA256 4008a7108a632d218fbb081fb16eb06a884db4a9c70bc4da1dbf45a3aadf7824
Final kernel source SHA256
725b006d02ebfedd442279e54691c499ad1ebc7270959bf6dafa4b2c6f33e3ba
Tests patch: tests.patch
SHA256 75e264f53c041bc9ce91a1b9417ed7fcd04667a97d4660b2de5c03654fef147a
All exact files are listed in source-hashes.json.

Change and scope
- Only nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c changes.
- Native LFM prefill (M>1, !use_m1, K2048/I1536/N2048, 1<=E<=64,
  all active experts native TL2, no affine offsets, SILU) gets a private
  gather worker. The public generic worker and its nonzero-skip contract
  are unchanged.
- Existing pack_skip bytes become row counts 0/4/8/12/16 ONLY for this
  worker. Existing 16-row queue units/done bytes/worker partition/drains
  are unchanged. Metadata caps the last nonempty unit at ceil(rows/4)*4.
- The app compresses virtual experts into active/group vectors before
  stage(active); invokeMoeLayer passes h_gu.size() to DSP. E is therefore
  compact active experts, not the model's fixed64. The original draft
  E==64 guard was corrected before freeze; E1/63/64/65/0 are covered.
- No activation/weight buffers, allocation, persistent content cache,
  reader, C16 pool, VTCM arena, scratch sizing, HMX issue, accumulator
  read/convert, quantization, row mapping or float operation is changed.

Why unused rows are safe
- Original scratch slot construction already maps padding rows to row0
  and skips completely padded16-row units. The content of such HMX rows
  is already irrelevant; m_blk bounds every consuming epilogue.
- hvx_gather_u8_ah_rows operates in4-row groups; each group writes one
 128-byte vector per K/32. The new end is always4-aligned, lies inside
  the existing64-padded slot block, and never excludes a real row.
- Slots sharing an identical row list keep the existing prior metadata;
  only unique slot ranges are initialized and enqueued as before.
- The whole64-row AH block still goes through unchanged DMA and HMX.
  Unwritten padding can hold old bytes. UInt8 dot bounds remain valid
  even for arbitrary dead-row bytes; there is no cross-row reduction.
- Current native signed24 GU/DN epilogues consume only m_blk rows.
  Actual libnative checks poison every unused accumulator row differently
  and retain byte-exact valid FP32 outputs, including unaligned tail-row
  reads and compact6144-byte vs legacy8192-byte tile strides.
- Existing error out drains foreground, last background pack/down, and
  live expansions. The new worker has the same context lifetime and no
  new asynchronous work or failure path.

Validation (author)
- final-san-v2/: 198 actual-HVX gather cases, rows0..65, arbitrary byte
  palettes/maps, slot offsets0/64, 1/8 concurrent pthread lanes,
  complete output/gap canaries;16 source-extracted guard checks.
- final-san-v2/:512 actual-HVX signed24 GU/DN dead-row-poison cases,
  m1..64, lanes1/8, tile strides6144/8192. Full output and seen bytes exact.
- final-san-v2/:21 whole-production-TU comparisons, baseline and candidate
  linked together. Real-sized LFM GU/DN, compact E1/63/64, shared slots,
  mixed native/int4 and E65 fallback;5 full successful calls and16
  first-HMX injected failures, including M1024/2048. Real pthread pool
  and real HVX gather; equal return values, valid FP32 bytes, scratch cap,
  canaries and drained gather work. Immediate DMA and scalar HMX/other
  math use the existing host stand-ins. Physical DMA/HMX remains device
  validation, not a property claimed by this fixture.
- ASan/leak and UBSan halt-on-error passed. Only the existing scalar
  stand-in TU has -fno-sanitize=shift-base because swiglu_det_exp shifts
  a negative exponent at swiglu_det.h:152. This exclusion is NOT used
  for either production TU or actual native gather/dequant TU.
- mutants-final/:7/7 wrong source variants rejected: floor4, restore16,
  wrong slot offset, generic skip contract change, E65 acceptance,
  mixed-format guard acceptance, wrong job-function dispatch. The last
  one executes the complete production TU; others use actual extracted
  functions/metadata and strict byte/count assertions.
- Whole v81 TU compiles baseline and candidate with -O3 -Wall -Werror.
  codegen/commands.json retains actual commands and source/object evidence.

Code generation and limits
- Public generic gather worker:124 object bytes, SHA
  400925ef73e0a9e9c1947d5ca22eb5543287abb73f04898bbd2d8df50efa2d70,
  identical before/after;13 static packets,8-byte frame. This is pre-link
  object identity; link-address relocation is not a runtime-byte claim.
- New native worker:48 bytes,7 static packets,0-byte frame, no vector or
  stack spill. It tail-jumps to the unchanged actual gather primitive.
- Main kernel frame17152 unchanged; no new vector stack accesses.
  Existing workers' normalized instruction streams are unchanged. Shared
  main body grows21392->21840 bytes and2376->2406 static packets. Its
  static scalar SP/FP memory-operation count is1655->1673; these counts
  include all branches and are NOT executed-operation or cycle counts.
- Generic/Gemma source branch and workers remain the old behavior, but
  shared entry/metadata code is not byte-identical. No unmeasured Gemma
  performance identity is claimed.

Bound, not performance evidence
Observed1024 normal route:2283 active expert slices,3776 M64 blocks.
Original rounded16 gather rows174112 -> rounded4 rows159228.
Omitted14884 rows /3721 four-row groups /8.5485% of gather rows.
Logical stores removed:30482432 bytes (29.07MiB).
Logical source vector-load bytes removed:121929728 (116.28MiB), because
four128-byte source loads form one128-byte output vector. These are
instruction-level logical counts, NOT DDR traffic or measured bandwidth.
The HMX padded computation, DMA block bytes and acc readout are unchanged.
The old isolated GATHER exposed wait was only1.649ms/38layers; other worker
aggregates overlap HMX and cannot be added as serial savings. Extra scalar
metadata work can offset the saving. No TPS gain or1000TPS bound is claimed.
No2048 expert-route histogram was available, so no synthetic saving is given.

Reproduction
Use explicit checkout/overlay paths; tests contain no machine-local paths:
  python3 <source>/test/htp/host/run_lfm_tail_gather_checks.py \
    --repo <complete-checkout> --source <candidate-overlay> \
    --reference <baseline-overlay> --output <new-output-directory> \
    --tools-root <hexagon-release-containing-Tools> --sanitize --whole
  python3 <source>/test/htp/host/run_lfm_tail_gather_mutants.py \
    --source <candidate-overlay> --checks <completed-output-directory> \
    --output <new-mutants-directory>
The reference overlay needs just the baseline kernel; other headers/helpers
come from --repo. This run used the unchanged ccea0d source context; hashes
of important reused dependencies are in dependency-hashes.json.

No root repository edits, device access, commit or push were performed.
Independent review and root-only real-model A/B are required before adoption.
The prior padding/re-expansion feasibility audit is preserved separately at
/tmp/nntr-lfm2-m-padding-audit-20261016/review.txt (SHA0700507942bd8b4be821d1dcc20878e27abf649b4f4519fd05cb32bdc4e742c0).
