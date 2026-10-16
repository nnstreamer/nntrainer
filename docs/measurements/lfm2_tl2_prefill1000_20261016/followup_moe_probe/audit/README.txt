# LFM2 native TL2 prefill FFN residual-work audit

Audited tree: `9d8706182fcf56fb41de3c9ec5af17c7ee80e8a4` at `/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000`. Read-only inspection; no device access. C16 / preload128 / PAGE0 / six readers / native TL2 weight values are fixed. No weight-cache growth, reader-scratch growth, or work moved outside the prefill timer is proposed.

## Finding

Per-expert duplicate activation quantization, repeated native expansion, and unfused GU/DN epilogues are already eliminated on the current LFM HMX path. **Correction after actual-dispatch tracing: LFM input RMSNorm is still computed on ARM, and its plain MoE core still stages activation and output. The earlier claim that the Gemma folded pre-norm/direct-input path was already active for LFM was incorrect.** I did not establish a remaining compute-only saving of at least 15%. A measured normal-path stage breakdown is needed before proposing that scope. PROFILE=2 is not the normal workload and cannot provide that proof.

A **test-only DSP diagnostic**, without IDL/RPC/ARM changes, is available at `/tmp/nntr-lfm2-moe-stage-trace-20261016/lfm2-plain-stage-trace-v2.patch`. Its README, strict CSV parser, v81 compile and ASan/UBSan fixtures are in the same directory. Keep PROFILE=1 and verify normal 65 generated IDs. It probes the actual existing plain call. Its f bits explicitly expose pre_gamma/out_heap state; current LFM is expected to have f=0. The v1 norm-helper location produced no records because LFM does not call that helper.

## One-by-one review

Source paths below are relative to the audited tree.

| Potential duplicate work | Current behavior | Remaining opportunity / limit |
|---|---|---|
| Source activation quantization per expert | `hmx/hexkl_mm_u8i4_moe.c:2454` calls `hvx_quant_rows_u8_params_rm` once for all M rows. The primitive supports folded RMSNorm when pre-gamma is supplied, but current LFM passes no pre-gamma and computes its norm on ARM. | Top-4 does not cause four requantizations of the original activation. A direct Gemma port would duplicate an optimization already present. |
| Per-expert gather / AH packing | `hexkl_moe_pack_bg_worker` at line1745 gathers the shared U8 rows directly into AH in 16-row units. Fully padded units are skipped; exact repeated row layouts can share AH slots. | The routed gather is needed because experts see distinct row lists. Avoiding it needs a different HMX consumption layout, not deletion of an extra copy. |
| TL2 lookup-table preparation | `hvx_tl2_prepare` at lines1400/2516 belongs to M1 HVX GEMV. | There is no per-expert TL2 GEMV LUT in the prefill HMX path to eliminate. Prefill losslessly expands native TL2 into HMX int4 panels. |
| Native weight DMA and expansion per row block | `tl2_pushed`/`tl2_expanded` masks at lines204–412 prevent re-push/re-expansion of panels across chunk boundaries and 64-row blocks of an expert. The compact native matrices are retained in the existing VTCM budget. | Reading/expanding once for an active expert is necessary. Repeated scalar/index work inside expansion would need actual instruction/cycle proof; do not count each source expression as a separate memory load. |
| GU dequant + activation | Fused GU dequant/SwiGLU epilogue, including interleaved independent row work; the up FP32 matrix is not separately materialized. | Existing worker parallelism and overlap matter. Whole worker busy sums are not exposed elapsed cost. |
| Intermediate quantization / packing | Per-expert SwiGLU output legitimately differs. Native DN work carries the next block's requantization, avoiding a separate queue/barrier (`hexkl_mm_u8i4_moe.c:601–806,3332`). | Min/max needs to precede quantization. Current local row quant/pack already combines its useful work; this is unlike the old attention Qprep extra pass. |
| DN dequant + routing/scatter | Fused dequant and weighted scatter; first writer takes a row as zero, avoiding a full output clear. Per-expert accumulation order is retained. | Reordering expert accumulation risks changing generated IDs. No unneeded full DN FP32 output array remains. |
| Accumulator planes / staging | Native bounded accumulators use exact 24-bit planes; GU/DN compact panels fit the existing VTCM layout. Gemma's fused norm path can consume input directly and accumulate into the session's cached output, but current LFM uses the plain path and keeps the activation-in/output-out staging copies. | These direct paths are not active for current LFM, independently of PROFILE=2. Normal-path staging remains a real candidate. Keep output parity and routing fixed before using instrumented costs. |
| Scratch allocation | Session scratch is reserved using the shape bound and reused after warmup. | No per-layer bulk allocation remains to remove from a normal steady prefill. |
| 64-row HMX padding | Padding remains substantial; see recorded-route counts below. | A small-tail HVX path exists for 4-bit weights but excludes native 2-bit (`allow_tail = !n_affine && all_4bit`, line1994). Existing comments document worker contention negating saved HMX time. Native support requires new exact kernels/scheduling, not a flag flip. |
| Stable-handle metadata refresh | Changed arena generations refresh scales/column sums before compute (`hexkl_mm_u8i4_moe.c:1977`, `hexkl_mm_u8i4_dma.c:192`). | This is required after slot overwrite; there is no colsum recomputation. Current refresh does two small tail drains per pair and re-zeroes already zero native biases. Pairwise batching could remove control/barrier work, but no measured >=15% FFN saving is established. |

All `hmx` paths above mean `nntrainer/tensor/htp_backend/hmx`. The quant/expand primitives are in the sibling `hvx` directory.

## Route padding, measured structurally

`routes.json` and `analyze_routes.py` reproduce these figures from the saved **normal** `lfm2-prefill.trace` (38 layers, M1024, top4 actual routes; trace contains top9 and only its first4 per token are consumed). The parser asserts that actual-route union matches the recorded expert union.

| Quantity across 38 MoE layers | Count |
|---|---:|
| Active expert/layer instances | 2,283 |
| Actual routed rows | 155,648 |
| Issued HMX 64-row blocks | 3,776 |
| HMX row capacity | 241,664 |
| Padded row positions | 86,016 (35.59% of capacity) |
| Nonempty partial blocks | 2,264 |
| Partial blocks with <=4 / <=8 / <=16 useful rows | 308 / 517 / 863 |
| Padded GU+DN MACs | 2,280,627,634,176 |

35.59% padded row capacity is **not** a 35.59% whole-FFN time saving: native weights still have to be read and expanded, valid-row epilogues already skip padding, queue/control work remains, and an HVX tail competes with those epilogues and expansion. PROFILE=2 produces different routing and blocks, so do not multiply its MM time by this fraction to claim normal-path speedup.

Reproduce:

```sh
python3 analyze_routes.py /path/to/lfm2-prefill.trace --top-k 4 --out /path/to/routes.json
```

## Why the level-2 breakdown is only structural evidence

The fresh saved `postpush-convion-profile-level2.log` reports 39 MoE calls and 411.1ms host, with 3,853 blocks (includes a warmup). The matching JSON marks the run invalid for output/performance comparison; generated IDs differ from the normal reference, as already observed on baseline.

`htp_compute_ops.cpp:7170` sets `host_bits = level >= 2 ? 7 : host_mask` for folded callers. However, the LFM builder does not set out_add; `lfm2_moe_layer.cpp:1843` consequently keeps pre_gamma=nullptr, post_gamma=nullptr, add_x2=nullptr in both normal and PROFILE=2 MoE calls. The previous explanation that PROFILE=2 removes a normally active LFM MoE pre-gamma was incorrect. The normal LFM MoE already performs activation-in/output-out staging. The larger whole-model PROFILE=2 run is still not a valid normal-output comparison: other folded paths (including dense FFN before MoE layers) can change dispatch, and this run has different IDs and routes. Its671.7us/call staging is suggestive of real plain-path work, but requires v2 normal-path confirmation rather than automatic subtraction from normal wall time.

The printed per-call values are: quant451.5us; gather37.8; requant17.1; swiglu(hidden)30542.6; dequant325.1; accumulator558.9; GU drain1433.7; DN drain336.1; push134.3; scatter636.7; allocation13.4; stage671.7; MM2762.2; residual2625.8. These are **not a normal-run cost model**.

Interpretation traps:

- Hidden SwiGLU is a sum of overlapping worker busy times. Multiplying it by calls and calling the result serial time is wrong.
- Dequant, requant and scatter timers measure exposed waits. GU/DN drain includes expansion-worker joins as well as DMA waits.
- Initial synchronous native expansion, stable metadata refresh before the main loop, and scheduling/setup can remain in residual time.
- DMA trace capacity is 512 pushes. This row averages599 descriptors/call; dropped traces return zero busy/last-issue while totals still count all bytes. Mixing truncated calls with valid ones produces the bogus printed 1026–2502GB/s figures. No physical throughput conclusion follows from them.

## Registration: the apparent 292.6ms is load-time accounting

The latest level-2 log prints 384 registered weights, "convert"149.0ms, FastRPC47.0ms, and "alloc+other"96.6ms (total292.6ms). The profiler heading says all registration lands in first forward, but that heading is stale for the current eager load path.

Exact call order:

1. `Applications/CausalLM/main.cpp:349` calls `model->load_weight` and then `repack_weight` before any `run`.
2. `lfm2_htp_moe_causallm.cpp:219–228` already prepares/re-packs the decode graph inside load. This invokes shared Gemma HTP weight preparation, including preload128 and FC registration.
3. The saved log records preload128 at line23; the graph reports256 WH handles at line854; `model->load_weight` completes at line856 (1029.0ms). The later repack line857 is already complete.
4. `CausalLM::run` starts the prefill timer at `causal_lm.cpp:1694`, and the early read kickoff is later (saved log line885).
5. The HTP registration counters are lifetime accumulators (`addRegister`, `htp_compute_ops.cpp:352`), not reset at prefill start.

The 384 count is **128 preloaded expert pairs plus256 FC WH handle registrations**, not384 separate on-demand experts. `register_qs4cx_wh_expert_file(at_load=true)`, lines4738–4755, explicitly puts `stageExpert + readExpert` elapsed into the old `convert_us` bucket. On this native model that bucket largely means file reading, not dequantization or requantization. FC arena registration sets convert=0; its preparation/copy work is lumped into alloc+other.

Thus 292.6ms is not a prefill-timer saving available by removing a newly discovered quantization pass, and no timer relocation is proposed.

There is one real load-only materialization candidate: dense-DN row chunks in `registerFcWh` (`htp_compute_ops.cpp:7887–7898`) lack file metadata for their partial K sums, so they unpack packed WH into a temporary int8 matrix and scan it for the partial colsum. Computing those integer sums directly from the packed WH could remove that temporary and one pass. The full-K metadata cannot simply be reused for a partial-K slice. Standard whole-K native/affine FC registration already consumes file metadata and copies packed data into ION without requantizing. This dense load-only candidate does not address the user's prefill TPS target and should be lower priority.

## Narrow next candidate, if normal-stage data supports it

Stable native metadata refresh can be audited independently of HMX arithmetic. In the exact LFM geometry, scale+colsum payload is40KiB per refreshed expert pair. The general helper issues two DMA descriptors and drains each GU/DN separately, then zeroes20KiB of bias which native experts keep zero. A specialized pair refresh could enqueue both matrices, drain once, invalidate both destinations, and preserve already initialized zero bias. Source/destination metadata arrays are contiguous within each matrix. GU24KiB exceeds one16KiB DMA row and must be described with valid multi-row geometry; DN16KiB fits. Validation must cover first registration, changed/unchanged generations, wraparound policy, reused arena slots, and stale L1/L2 lines. No shared/Gemma behavior should change.

This removes barriers and repeated zero writes; it does **not** remove required metadata refresh or justify skipping a changed generation. Without measured normal cost, its upper bound is unknown and it is not claimed as a >=15% FFN optimization. All production edits remain deferred to the parent after the diagnostic run.

## Correction record

The initial report inferred LFM pre-norm folding from the shared Gemma-capable backend without checking the LFM builder's out_add property. The v1 diagnostic returned no records and prompted a full caller audit, which exposed this error. The superseded initial report is retained separately for provenance; this corrected report is authoritative. Registration/load-time findings, existing GU/DN fusion, native panel reuse, and recorded-route padding counts are unchanged.
