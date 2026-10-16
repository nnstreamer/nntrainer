Native LFM M1: remove unused initialization
==========================================

Retained as a bounded removal of dead stores, with no established end-to-end
speedup. Native TL2 decode consumes one row, but inherited HMX setup filled
64-row padding and cleared an unused output accumulator in rows_out mode.
The candidate initializes only the consumed scale/zero-point entries and
skips that unused accumulator clear. The actual output still uses the same
zero + weight * result operation, in the same expert order.

The guard requires native M1, K2048/I1536/N2048, 64 experts, 1..4 active
experts, one row per expert, all-native TL2, no affine offsets and SwiGLU.
Prefill, Gemma, mixed formats and other shapes retain their original setup.
Ordinary accumulation and zero-expert behavior are preserved. Scratch sizes,
64-slot strides, VTCM layout, readers, cache slots and LUT math are unchanged.
There is no new persistent state, allocation, runtime flag or RPC change.
The author's before/candidate v81 TUs both use a 17,152-byte layer stack
frame; this is not a whole-program stack measurement.

Normal device comparison
------------------------
S26 Ultra, native TL2 model SHA256:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d
1024 input + 64 decode tokens, order A1 B1 B2 A2 A3 B3, profiling off.
Only the DSP binary differs. Both sides enable the retained router handoff.
C16, CACHE_GROW=0, PAGE_CACHE=0/Direct I/O, preload128, reader counts and
expert/spare capacity are fixed. Complete environments are in raw/*.json.

                              Control        Candidate
Median decode TPS             17.5583         17.6503
Decode range                  17.4625-17.7433 17.6406-17.7040
Median prefill TPS            825.806         840.722
Maximum sampled peak KiB      2733648         2734172

Decode improves in two of three pairs; the median difference is +0.52% with
overlapping ranges. This is insufficient to establish a throughput gain.
Prefill does not select the new setup guard, so its +1.81% median difference
is not attributed to the removed decode stores. The earlier pre-build control
run is retained separately: the comparison was restarted after the build and
validation gap, before seeing any candidate inference result.

A separate profile pair reports net MoE 9613 -> 9606 us/token, a 7 us change.
It is not an isolated setup timer and provides no resolved compute gain.
Both make 67.89 MoE calls/token. Total MoE is 41444 -> 40129 us/token; almost
all of that difference is exposed expert wait (31831 -> 30523 us/token),
which this candidate does not optimize. Instrumented TPS is not mixed into
the normal comparison.

One candidate 2048-input correctness/memory run gives 869.270 prefill TPS,
16.9312 decode TPS and 3050936 KiB peak. It is not a comparative speed result.
The maximum normal 1024 peak is about 2.608 GiB; the 2048 peak is 2.910 GiB.
All ten retained runs match their 65 reference output IDs. Two consecutive
1024 prompts on the same model/handles also match all IDs. No Gemma phone
benchmark was repeated. Neither 1000 prefill nor 25 decode TPS is reached.

Validation and limits
---------------------
The complete before/candidate MoE translation units pass 31 optimized-host
cases and 31 ASan/UBSan cases. Root independently runs the exact integrated
handoff baseline and candidate, and a second agent repeats the author suite.
The production quantizer runs through Qualcomm SDK libnative. Fixtures cover
untouched-byte snapshots/poisoning, all 1..4 active-expert counts, both output
modes, real scratch reuse without refill, shape/activation fallbacks, norms,
zero experts, bad rows and injected timeout with drain. Three mutations are
rejected: a padding read, removal of actual output zeroing and removal of the
fallback accumulator clear. The independent reviewer also passes 96 cases
using the actual HVX epilogue and catches a poisoned-padding-read mutation.

DMA/pool and some arithmetic are host stand-ins, not a DSP timing/coherence
simulation. The existing scalar swiglu_det negative signed shift requires
disabling only shift-base UBSan in hvx_scalar.c; both production MoE TUs and
the actual quantizer retain ASan/UBSan. The original diagnostic is archived.
Full integrated Hexagon v81 DSP build, changed-lines clang-format14 and diff
checks pass. The prior full host suite passed for the handoff commit; this
small follow-up uses the targeted whole-kernel checks above, not a repeated
full suite or the cross-platform CI matrix.

The source-store estimate is approximately 508416 bytes/token of unused
output clear, 115520 bytes/token of slot metadata and 34216.875 bytes/token
of quantizer padding at the earlier recorded call counts. These are logical
source stores, not measured DDR traffic, bandwidth, elapsed time or energy.

Memory covers child exec/load through exit, requested every 20 ms: maximum
same-sample RssAnon + fdinfo sizes deduplicated by (mnt_id, ino). Every peak
scan has complete readable fd identities. Allocations need not be resident;
unreported DSP/driver/VTCM memory and between-sample spikes may be absent.
This is not total device physical memory and must not be added to VmHWM/RSS.

Reproduction
------------
artifacts/ identifies exact deployed binaries and source patches; control
is router-handoff-remapped and candidate is m1-dead-setup. The candidate
patch applies to 83b3235830771ed2313135393efb48d7efaf549b. The author's patch
is based on 03d652fc0; root-frozen/ retains the remapped patch/reference.
Only libnntr_hvx_skel.so changes. The normal DSP has no extra trace.
validation/ retains commands, hashes, logs, mutants, independent review and
repeat checks. test/htp/host/lfm2_m1_setup_check.txt documents the standalone
SDK-dependent fixture. manifest.json hashes every retained file except itself.
