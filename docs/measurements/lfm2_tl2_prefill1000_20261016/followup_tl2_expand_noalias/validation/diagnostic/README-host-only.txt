Count-only diagnostic for frozen TL2 noalias candidate 59306fe
============================================================

Patch: diagnostic.patch
SHA256: 371efbd1be5f69a7fd8dd49e895b6e316e2b33f870f439f50d12c567c99a8c0e
This is a separate diagnostic overlay. It must not replace the normal candidate
or contribute its instrumented timing to a performance comparison.

Build every affected DSP translation unit with:
  HEX_EXTRA_CFLAGS='-DNNTR_LFM2_TL2_EXPAND_TRACE=1'
Use the existing full DSP build and its existing other required flags. Both
hvx_gemv_u8tl2.c and nntr_hvx_mm_u8i4.c must receive this define. No IDL, RPC ABI,
ARM code, model math, layout, allocations, expert count, job scheduling or
reader/cache configuration changes. Source snapshots and exact hashes are in
baseline/, candidate/ and source-hashes.json. The MoE kernel itself is untouched.

How the measurement works
-------------------------
An aligned four-byte uint32_t global counts actual entry into
hvx_tl2_expand_wh96_disjoint using __atomic_fetch_add(..., __ATOMIC_RELAXED).
The existing generic function does not increment it. The RPC reads the same
counter with __atomic_load_n before the existing kernel invocation and subtracts
that snapshot after the successful kernel returns. Unsigned subtraction handles
counter wrap; a single RPC has far fewer than 2^32 invocations.

Only successful plain RPCs with M>1, K2048, I1536 and N2048 log one ERROR line:
  [nntr-tl2-disjoint] M=1024 K=2048 I=1536 N=2048 E=52 count=40768
E is the count of strictly positive row_count entries. count is measured from
real helper entries; it is NOT populated from E or an expected formula. Failed
kernel/argument calls, decode M1 and other shapes do not produce this tag.
The tag begins with nntr so the parent's existing logcat filter retains it.

This cumulative diagnostic assumes serialized RPCs in one model/session, as in
the current benchmark. Atomic increments are race-safe across expansion workers.
It does not add synchronization between independent simultaneous RPCs; another
session calling the helper during the interval would contaminate the delta.
The existing benchmark's single-session ownership is the measurement scope.

Worker drain proof
------------------
hexkl_mm_u8i4_moe.c, the existing out: cleanup around lines3519-3534, first calls
hvx_worker_pool_wait, then waits last_job if present, then waits every exp_live
job in MOE_EXP_SLOTS before returning rc. On success each expansion was already
joined before its WH chunk was used by HMX. The new after snapshot is in the RPC
only after hexkl_mm_u8i4_moe_layer_run has returned. No new drain or dispatch was
added, and no per-worker logging occurs.

Expected count
--------------
With native disjoint LFM slabs, each positive-row expert expands every GU and
DN panel once. The existing tl2_expanded mask suppresses repeated panel work
across chunks and row blocks:
  GU: 24 panels * ceil(2048/96)=22 groups = 528 entries
  DN: 16 panels * ceil(1536/96)=16 groups = 256 entries
  Total: 784 entries per active expert, including each GU final32-row group.
Every successful line should therefore satisfy count == 784 * E.
A warmup with E64 should have count50176. A fresh model load followed by one
normal prompt is expected to produce39 lines: one load-time warmup plus38 real
prefill MoE layers. Same-handle subsequent prompts do not necessarily repeat
the load-time warmup; validate their lifecycle separately. Prompt1024 and2048
share the per-expert formula, while their active-expert counts can differ.
Do not require the historical aggregate expert count for a new prompt.

Verification executed
---------------------
v81/results.json and commands.json:
- Both affected full TUs compiled for baseline, macroOFF and macroON: six
  -mv81 -O3 -fPIC -Wall -Werror builds, plus assembly/stack output.
- The counter uses inline memw_locked load/store retry instructions.
- No __atomic or __sync runtime import is introduced. The new RPC getter symbol
  is defined by the modified HVX TU. Existing HVX __hexagon_udivsi3/memset
  imports are unchanged.
- MacroOFF HVX object is byte-identical to the frozen production object.
  RPC objects include different existing FARF __FILE__/__LINE__ metadata because
  source lines moved. Two additional verification-only builds normalized just
  those built-in metadata macros; their complete objects are byte-identical.
- Trace counter storage is exactly4 bytes, aligned4. Trace getter stack0.
  Disjoint helper stack remains192 bytes. RPC static frame120->144 bytes (+24)
  under this diagnostic. No activation/cache/VTCM allocation was introduced.

host/counter-run.log, commands.json:
- Actual helper compiled with SDK libnative and ASan/UBSan.
- Eight concurrent host threads produce32792 exact increments:32768 zero-row
  entry calls plus24 real HVX output comparisons across rows32/64/96.
- Generic outputs and traced outputs match; generic calls do not increment.
  This checks counter concurrency and output parity, not DSP cache timing.

host/rpc-run.log, rpc-commands.json:
- 112 comparisons using the actual original and diagnostic plain RPC function
  bodies, actual argument-check functions and real handoff inline helpers.
- 16 success-only exact delta/log cases, with M1024/M2048/ticketed calls and
  active counts0..3; failure/invalid-argument/M1/Gemma/near-shape cases excluded.
- Return values, output markers and kernel arguments remain identical. Kernel
  execution is a deterministic stub invoking the real counted helper; this
  checks RPC control flow and logging, not model arithmetic or real pool drains.
  Minimal test session fields stand in for the full runtime ABI. Full real DSP
  TUs were separately compiled as described above.

Overhead and restoration
------------------------
The trace introduces a contended atomic RMW at every helper entry, two counter
reads and one active-expert metadata scan/log per logged RPC (the before read
also occurs in other valid plain RPCs),4 bytes of global state and24 bytes of
RPC stack. Timing under this diagnostic is not comparable to normal timing.
After collecting count evidence, reverse diagnostic.patch and rebuild/restore
the exact frozen normal sources/binary, verifying its source/binary hashes.
No root tree or device was modified by the author of this diagnostic.
