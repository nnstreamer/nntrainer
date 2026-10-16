# Independent review: stable native TL2 metadata pair refresh

**Result: no new correctness blocker found in the supplied production patch.** This is a source/ownership/geometry review; no device access and no root production edits. Device parity and speed are the parent's separate validation. It does not establish a performance gain.

Base: `b464fbf6f167b8e5ac2e135201c1316cc6f141f8`.
Patch SHA256: `323a6e2b94b52de3796ca8e4f658f7db3d30f496129106112d3bd490177dfe57`.
Only `hexkl_weight_tl2_refresh_pair` changes. No allocation, weight representation, arithmetic, cache capacity, reader or preload setting changes.

| Review item | Finding |
|---|---|
| Exact DMA extent | GU: row/strides12,288 bytes ×2 =24,576 bytes; DN:8,192 ×2 =16,384. These are exactly the unions of the old separate scale and colsum transfers. Neither descriptor reaches bias. |
| Width/height/stride | Row width and both strides equal the previously used values. Height changes1→2; descriptor height has16 bits, width/strides24 bits. All fields fit, rows are contiguous, and per-row widths are below16 KiB. This review relies on the existing supported row widths; no larger-width hardware claim is needed. |
| Alignment and cache lines | Native metadata offsets, array bases, row widths and copied range ends are128-byte aligned. Both source invalidations exclude the generation padding; both destination invalidations end exactly before bias. No shared boundary cache line is invalidated accidentally. See `geometry.json`. |
| Invalidate order | Invalidate GU and DN source ranges; enqueue GU and DN; drain the chain; invalidate both destination ranges; then publish `generation_seen`. This preserves the old source-before-read and destination-after-completion ordering. Bypass flags remain0/0. |
| Source ownership | `prefetch_qs4cx_wh_experts_end` joins all readers before `registerStagedBatch`; that function publishes the generation only for successful reads under `handle_mutex_`. Active expert slots remain held/pinned by the existing pool/LRU protocol. The patch does not extend the live interval outside the compute call. |
| Destination ownership | Metadata arrays belong to separate registered weight handles. GU/DN have different validated geometries, so the same handle cannot be a valid pair. The core refresh occurs before starting this call's worker/DMA pipeline; previous call cleanup retires its workers. |
| Generation semantics | Missing generation remains an immediate no-op. Same generation still performs only the generation invalidate/read. A changed generation becomes seen only after both transfers and both destination invalidations. No additional publication or skipped generation was introduced. |
| Native bias immutability | Compact registration passes no caller bias and `fill_slot` uses the unchanged tail helper, which writes positive-zero bits. Every compact rebind uses the same helper. Legacy rebind clears compact and generation state and also initializes bias; transitioning back to compact again initializes it. Kernels only read registered bias. Removing refresh's repeated memset therefore preserves its exact bits on every supported lifecycle. |
| Rejection and error paths | All existing compact/borrowed/in-use/shape checks and early returns are before the new transfers and unchanged. No new failing operation/allocation exists. General register/rebind keeps the old helper and its behavior. |
| Gemma/fallback scope | The changed helper is gated by native stable generation and exact GU2048×3072/DN1536×2048. Non-stable handles return before the specialized path; Gemma2816/704 shapes are outside the caller and helper guards. No repeated Gemma benchmark is requested. |

## Existing DMA failure limitation

`hexkl_dma_ring_wait_idx_` has a bounded polling loop, but its public wait/drain APIs return void and do not report a timeout. Both old and new refresh publish `generation_seen` after such a return. This patch neither fixes nor newly introduces that failure-reporting limitation. Consequently the review does **not** claim transactional recovery from actual hardware DMA failure. The ordinary argument-validation failures remain non-mutating.

The ring still drains all pending descriptors. Combining the two metadata matrices removes one drain/restart boundary; it does not replace completion with a speculative ready check. The caller's existing assumption that no unrelated kernel is concurrently using the same registry/ring remains required.

## Validation examined

- Supplied patch SHA confirmed exactly.
- Independent arithmetic/range checks in `geometry.json`: exact baseline versus candidate byte intervals, native tail offsets, descriptor bit-width bounds, metadata/bias cache boundaries.
- Reviewed the actual deferred-DMA fixture. It runs production compact registration/rebind/refresh bodies, delays copies until drain, checks source invalidation before submission and destination invalidation after completion, and rejects early `generation_seen` publication. It compares scales, colsum and bias bytes against the old serial helper, asserts no allocation/free during refresh, and checks generation/no-op/rejection behavior.
- Existing host ASan/UBSan and real RPC swap logs in the candidate's `checks/` report PASS. The author is extending/re-running the fixture for the two additional g.K/d.N rejection branches, legacy nonzero bias→compact rebind, and moving to another arena/generation binding; source additions were reviewed. These additions do not change the production patch.
- v81 `-O3 -Wall -Werror` object compile log is clean. `.su` reports refresh's own frame48→64 bytes; the unchanged tail helper has32 bytes and is no longer called by refresh. No heap/scratch capacity increase.

Host stubs prove sequencing and byte extents, not physical DSP cache coherence or device timing. The parent's actual device checks should include unchanged65 IDs, two-prompt reuse, unchanged complete-run peak and uninstrumented timing. The same generation/source ownership protocol is used for1024 and2048; metadata geometry is independent of M.
