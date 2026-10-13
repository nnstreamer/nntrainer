# Expert GU/DN reader-order audit

## Verdict

Do not implement an all-GU then all-DN phase for the measured `weights.bin`. LFM2 writes and requests `GU0, DN0, GU1, DN1`; the BIN loader assigns consecutive offsets in that order.

For K=2048, intermediate=1536, N=2048, GU is 1,597,444 bytes and DN is 802,820 bytes. `off_dn(e) = off_gu(e) + 1,597,444`; the next expert begins immediately after DN. The GPU/HTP arena's page padding is unrelated to file layout. A 64-expert GU-first pass would skip DN ranges and then move backwards by 151,216,632 logical bytes (144.21 MiB). This is file-offset arithmetic, not a physical-flash seek model.

## Current scheduling

Seven persistent reader threads each claim a whole expert and synchronously read GU then DN. Batches and experts are queued in forward model order. Each expert remains incomplete until both validated halves finish; registration is performed only when the layer collects the batch between DSP calls. Initial host staging also claims whole experts, then drains before ordinary arena jobs.

The existing `NNTR_MOE_MISS_READERS=1..4` knob splits synchronous demand misses into interleaved GU/DN tasks. It does not change the seven prefetch readers. There is no current GU-first prefetch knob.

## A different candidate, only if the pending timeline warrants it

Offset-ordered GU/DN part jobs could redistribute the tail of a batch. It does not increase maximum simultaneous blocking reads beyond seven. The smallest pilot would cover direct compact-QS2 prefill reads only; host-bank reads/promotions and decode stay whole-expert. Keep layer FIFO priority.

Two parts require two result slots and one expert-level completion, stable batch/slot ownership, transactional enqueue rollback, deterministic error precedence, and no recycle or lease release until both halves finish. Existing decode cancellation assumes one queue job per expert; using an encoded part index without changing that logic would be unsafe.

First use the forthcoming initial/last-read timeline. If a part-job pilot is justified, compare actual descriptor regions with seven identical readers and the original validator, require exact output hashes and unchanged bytes/requests, then run full-model raw parity and clean repeated TPS. A byte-only scratch benchmark cannot establish full-model benefit.

No code, device, root, or PUB change was made. Exact references and validation design are in the adjacent JSON.
