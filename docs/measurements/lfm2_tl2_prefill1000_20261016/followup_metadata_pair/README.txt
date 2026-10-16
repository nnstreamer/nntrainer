LFM2 native TL2: batch stable-pair metadata refresh
=================================================

The stable LFM expert refresh now submits Gate/Up scales+sums and Down
scales+sums with two DMA descriptors and one drain. Previously it used four
descriptors and two drains. Both copies and destination invalidations finish
before publishing generation_seen. The existing exact LFM shape guard remains.

The transfer still carries 40,960 bytes per changed pair. Compact register and
rebind already initialize immutable positive-zero bias, so refresh no longer
rewrites its 20,480 bytes. There is no new allocation, cache, runtime switch,
weight precision change, reader, preload, IDL or ABI change. The helper's own
v81 stack frame changes from 48 to 64 bytes; it no longer calls the existing
32-byte-frame tail helper. This is not a whole-program stack measurement.

Base: b464fbf6f167b8e5ac2e135201c1316cc6f141f8.
The deployed production source delta is recorded in artifacts/metadata-pair.*.
Its git patch and the agent's production.patch encode the same change with
different diff headers. source-sha256.json identifies the final source/tests.

Uninstrumented S26 Ultra measurements
------------------------------------

LFM2-24BA2B, native TL2CX_WH experts and QS4CX_WH_AFFINE dense weights.
Model SHA256:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d
Model size: 5,912,746,496 bytes. This metadata change does not requantize it.

Fixed settings: C16, NNTR_MOE_CACHE_GROW=0, NNTR_HTP_PAGE_CACHE=0 (Direct I/O),
NNTR_LOAD_PREFETCH=0, NNTR_MOE_PRELOAD_EXPERTS=128. Existing 608 expert slots,
32 decode spares, 6 prefill readers, 2 speculative decode readers and 4 demand
readers are unchanged. Every raw JSON records the full command/environment.
No new Gemma device benchmark was run; its non-stable path remains a no-op.

Normal 1024-input + 64-decode comparison, three runs per variant:

                              Control       Candidate
  Median prefill TPS           820.513        841.413
  Prefill TPS range        807.571-852.623  817.891-843.493
  Median decode TPS             17.4768        17.5872
  Maximum complete peak KiB   2,733,436      2,733,316

The predeclared order was A1 B1 B2 A2 A3 B3 (see repro/). Two of three pairs
improved prefill. Median differences are +2.55% prefill and +0.63% decode,
but the ranges overlap and these differences include run variance. They must
not all be attributed to the isolated metadata saving below. The earlier
control sanity run postpush-meta0-r1 is retained but excluded: its long build
and deployment gap was documented before the first candidate inference.

The candidate's single 2048-input correctness/memory run measured 850.145
prefill TPS, 16.5161 decode TPS and 3,050,280 KiB peak. It is not a comparative
performance estimate. Neither the 1000 prefill nor 25 decode TPS goal is met.

All normal, pilot and diagnostic runs returned the same 65 generated IDs as
their established reference, with no known concurrent benchmark process.
The same model/handles also passed two consecutive 1024+64 inferences.
Expected IDs SHA256:
  1024: b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e
  2048: 2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33

Peak measurement covers child execution/model loading through process exit.
It is the maximum same-sample RssAnon + deduplicated fdinfo allocation sizes
(identity: mnt_id, ino), requested every 20 ms, approximately 21 ms observed.
All retained peak scans have complete fd identity/read coverage. The normal
candidate maximum is approximately 2.607 GiB. Allocation sizes are not physical
resident bytes; unreported DSP/driver/VTCM memory and between-sample spikes
can be absent. Do not sum this metric with VmHWM, RSS or the arena again.

Isolated metadata timing
------------------------

A temporary DSP-only timer encloses the refresh loop and emits one line per
M>1 call. Each variant records 39 calls: the first load-time warmup is retained
separately and excluded from the 38 actual prefill layers. Both variants have
2,231 changed pairs, with identical per-layer expert and changed-pair counts.

  Control:   16,033 us
  Candidate: 13,461 us
  Saving:     2,572 us (16.0% of this measured stage)

This is one diagnostic pair, separate from the normal TPS comparison. It is
also separate from the previously measured approximately 67 ms of GU/DN
weight/expansion drains. The temporary probe has been removed from production.
Build flags and the diagnostic delta are retained under artifacts/ and repro/.

Artifact identity
-----------------

Control DSP SHA256:
573ef628be76610d18a63b23d4cb53b6847f4cedecf7fd290e66a20c70e5d8c4
Candidate DSP SHA256:
bd680249270078a1046349c6f40c1d3a35a6ea5124c49eadaead8ad7f7cfe6ca
The final local DSP binary was restored to the exact tested, uninstrumented
candidate. All deployed files were verified by SHA256; source patch hashes
are stored with each runtime variant. The control rebuild's binary differs
from the earlier attention experiment despite matching source, as documented
in followup_attention_overlap; this new comparison uses the recorded rebuild.

Validation and reproduction
---------------------------

The complete integrated command `bash test/htp/host/run_host_checks.sh` passed
with exit 0. Its full log and result are under validation/. Existing compiler
warnings remain. The optional ARM NEON abs_max cross-check was skipped because
the aarch64 compiler or qemu-aarch64 is unavailable. This is not the full CI
build matrix. The new fixture is guarded for Linux ELF hosts; production adds
no OS-specific API. Check Review counts reviewer approvals, so this standalone
host fixture requires no unit-test-count file change.

The production registry fixture checks deferred completion, exact byte ranges,
cache invalidation and generation order, zero allocations, six generations
including wraparound, unchanged-generation no-ops, ten state/shape rejections,
Gemma no-op, legacy nonzero-bias transitions and different-arena rebinding.
ASan, UBSan and leak checks passed; six deliberate mutations were caught.
Existing real-RPC swap sanitizer checks and v81 -O3 -Wall -Werror object
compiles passed. The full candidate DSP build and device ID/reuse checks also
passed. clang-format 14 was applied to the changed C files.

For the focused sanitizer/mutation checks, use the retained runner:
  python3 validation/agent/checks/run_checks.py --repo /path/to/checkout \
    --output /tmp/tl2-refresh-check --sanitize --mutations

Independent review and geometry checks found no new blocker. The existing
void DMA wait/drain API does not report its internal timeout; this patch does
not add transactional recovery from hardware DMA failure. Host stubs validate
software ordering; actual-device parity provides separate hardware evidence.

raw/ contains run JSON, generated IDs, compressed complete memory traces and
logs. results.json retains individual runs, medians and per-layer timer data.
validation/ retains tests, commands, build logs and independent review.
deferred-input-reuse/ documents a separate router-input reuse idea; it is not
implemented by this change. manifest.json hashes every retained evidence file
except itself. Test diagnostics and failed mutants are not performance data.
