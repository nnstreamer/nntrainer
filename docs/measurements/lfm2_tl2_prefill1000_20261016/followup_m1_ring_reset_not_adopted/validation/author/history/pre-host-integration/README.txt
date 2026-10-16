Native LFM M1 completed-descriptor reset candidate (2026-10-17)

Status: production and portable tests frozen; no device run or speed claim by
this author. Baseline is ccea0d476daf43ddd5dbe201178755e6ff978dfd. Root tree,
phone, commits and pushes were not touched. Production delta changes only the
three DSP C/header files listed in production-hashes.json. tests.patch adds two
Linux host-test files and depends on the retained lfm2_m1_setup_check fixture.

Scope and actual work removed
The current native M1 MoE path clears the entire static descriptor ring once
per expert-group invocation: 256 descriptors, 128-byte stride, 32,768 bytes.
The v81 reset function explicitly calls memset with 32768. Each push already
rewrites all eight 32-bit hardware words before clean/invalidate and submit.
The new try helper reads only the current last descriptor and, if complete,
resets g_push/g_pop/g_tail/g_started without changing any descriptor byte.

The sole production caller is inside the existing lfm_m1_setup gate: actual
M1/native TL2, K2048/inter1536/N2048, expert table64, <=4 active experts,
widest row1, all active experts native, no affine, SiLU. Existing math, output
clear/0+weighted sum, quant/LUT, scratch layout, lanes, cache and reader counts
are unchanged. Prefill/Gemma/nonqualifying calls retain the exact old reset.
No heap, persistent storage, activation/weight capacity, or new runtime knob.
The helper's implementation uses only existing fixed-width types and ring
API, with no new platform API/header. Tests explicitly reject non-Linux hosts.

Profile context, not a speed estimate
continue-conv-isolated-profile2048.log (hash in baseline.json) has 64 tokens,
38 MoE graph nodes, q4 M1 feed and 68.58 MoE kernel calls/token (4,389 calls is
the integer count consistent with two-decimal printing). Logical old reset
stores are at most 4,389*32,768/64 = 2,247,168 bytes/token. This is cached DSP
slab traffic, NOT measured DRAM traffic or measured savings: first/reset-
sentinel and incomplete/unknown cases still use memset, and no hit counter
or on-device reset time was measured. ACC_COPY includes reset and copies.
The graph's calibrated us/token are miss wait32,924.2; net MoE9,463;
FC6,321; attention3,382; router1,564; dense1,452; LM head1,293;
RMS641; conv348; QK norm186; add143; RoPE47. Graph wall58.008ms/token.
This identifies a definite repeated store operation, not a solution to the
remaining ~33ms/token I/O wait. No defensible time/TPS bound is available.

Completion and lifetime contract
1. First BSS tailNULL, original reset's type-zero sentinel, mismatched tail,
   size/type mismatch, or last.done==0 return false WITHOUT state mutation
   or waiting. The original reset remains the caller's fallback.
2. g_tail must identify (g_push-1)&255 and be a size1/type9 descriptor; the
   done check uses existing volatile hexkl_dma_ring_is_done. Earlier linked
   descriptors retire before the last, as the existing ring wait assumes.
3. Every push clears next/done/reserved0/reserved1/offset and sets all other
   fields. Actual v81 code stores words0,4,8,12,16,20,24,28, then dccleaninva,
   release(desc):at and dmstart/dmlink. The helper emits a fresh volatile
   done-word load, not reuse of the earlier size load. An old done1 is
   overwritten with done0 before submission.
4. Normal q4 native M1 input is copied by moe_dma_copy -> push -> drain before
   math. Weight q4 lanes use separate g_m1q_desc, with existing pool/lane
   joins. rows_out then returns after those joins; non-rows_out additionally
   copies the output via the drained ring. q1 feed waits each used ring
   transfer. Changed metadata also uses the existing ring drain before this
   call. These successful paths leave a completed tail for the next reset.
   Relevant candidate MoE lines: 1677, 1832-1853, 2196, 2450, 2466,
   2795-2848, and out at3558.
5. No error path or cleanup was added. Existing raw ring drain is a void,
   bounded 50,000,000-poll wait and can advance software indices on timeout;
   MoE out joins workers but does not repair an outstanding raw transfer.
   The fallback retains that pre-existing limitation. This candidate does
   NOT fix DMA timeout retirement or prove re-entry safe after such timeout.
   A checked-error alternative was intentionally not introduced because it
   would need a wider cleanup/engine-retirement contract.

Validation (checks/final-san-dispatch is final)
- Actual ring source included byte-for-byte, delayed ordered hardware shim:
  175 lifecycle/reset checks, 5,957 fills, 50,011,639 polls; ASan/UBSan PASS.
  First BSS/sentinel/unknown tail, delayed completion, stale done1 -> fill0,
  chained wrap up to769 pushes, wait vs drain, old fields/unused padding
  poison, exact 2D payload/strides and full generic reset were checked.
- The production 50,000,000-poll timeout is exercised with last index255:
  g_push==g_pop==0 and started0 do NOT fool the helper when tail.done==0.
  The TEST engine then retires the pending transfer before cleanup. It does
  not invoke the unsafe old fallback while a simulated DMA is active.
- 8 mutants rejected by intended assertions/ASan: reserved bit omission,
  done-clear omission, completion/type/size/tail guards, full slab memset,
  and missing g_started reset.
- Both actual whole-MoE TUs plus actual SDK HVX quantizer: 31 numerical,
  quant, padding/alias and guard cases PASS; 1..4 experts, rows_out on/off,
  queues1/4, lanes1/8, Gemma/E32/GeGLU fallback, zero experts, bad row,
  reuse 4->fallback->1->4, pre_gamma and lane timeout. Try helper is mocked
  here to alternate successful and failed eligibility; call counts prove
  the exact shape guard and original fallback. Two caller mutants (removed
  shape guard, bypassed helper) rejected by reset-call assertions.
- This caller fixture retains -fno-sanitize=shift-base ONLY for its pre-
  existing scalar HVX stand-in (negative left shift in swiglu_det.h). The
  production ring/MoE and actual quantizer have full ASan/UBSan. It does
  not model real DMA, HMX/epilogue timings or device cache coherency.
- Host ring tests model ordered completion and check data/lifetime; they
  cannot prove Qualcomm device cache visibility. Device exact-ID testing
  remains necessary. No integrated real DMA+whole-kernel claim is made.

v81 code generation
Both complete ring and MoE TUs compile with -O3 -mv81 -mhvx -mhmx -Wall
-Werror. Command JSON, asm/object/stack files are in codegen/. Target static
assertions verify 128-byte descriptor size/alignment and src8/dst12 for
32-bit pointers. The helper is144 code bytes, stack0. MoE frame remains
17152B; layer_run code +32B. Generic reset raw function bytes are identical.
Math worker raw bytes are identical. Ring push/drain raw object sizes differ
by +8/+4 alignment bytes but normalized instruction packets are identical;
two other MoE helper raw differences are relocation offsets with identical
normalized packets. See comparison.json and normalized-comparison.json;
do not describe entire objects as identical.

Portable reproduction (run from a tree containing the retained M1 fixture)
python3 test/htp/host/run_lfm2_m1_ring_reset_checks.py \
  --repo REPO --source CANDIDATE_OVERLAY --reference BASELINE_OVERLAY \
  --output NEW_OUTPUT --sanitize --mutations --hexagon-tools HEXAGON_TOOLS_ROOT
HEXAGON_TOOLS_ROOT is the directory containing Tools/bin and Tools/libnative.
The optional SDK argument adds whole-M1 checks; actual ring checks need gcc
or CC with ASan/UBSan only. The test source has no fixed local workspace path.
