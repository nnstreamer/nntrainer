LFM native TL2 prefill: remove redundant packed-vector loads
==========================================================

Retained after two prompt lengths each passed three normal A/B pairs. LFM's
existing native packed slabs and expanded WH destinations are disjoint. A
separate expansion entry makes that fact visible to the compiler, allowing
packed vectors to remain live across stores. This removes repeated loads
without changing arithmetic, layout, scheduling, pool jobs or buffer sizes.

Only native GU K2048/N3072/stride49152 and DN K1536/N2048/stride32768 select
the new entry. The check is outside the worker's panel/group loops. The
original generic entry retains the same normalized v81 instruction stream;
in-place staging retains its source snapshot, and other shapes including
Gemma select the generic entry. There is no added heap/VTCM allocation,
activation/weight cache, reader, preload or expert slot. The new wrapper's
scalar frame is 192 rather than 176 bytes; the worker frame stays 2816 bytes.

Normal device comparison
------------------------
S26 Ultra, native TL2 model SHA256:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d
Only the DSP binary differs. Control is m1-dead-setup on 8288e3ba3, candidate
is tl2-expand-noalias. Both enable router input handoff. C16, CACHE_GROW=0,
PAGE_CACHE=0/Direct I/O, preload128, reader counts and expert/spare capacity
are fixed. Each prompt generates 64 decode tokens, with 65 output IDs.
Both comparisons use A1 B1 B2 A2 A3 B3 order, profiling disabled.

                            1024 control   1024 candidate
Median prefill TPS          830.495        842.798 (+1.48%)
Prefill range               811.410-849.793 836.601-855.472
Median decode TPS           17.5631        17.6991
Maximum sampled peak KiB    2733392        2732956

                            2048 control   2048 candidate
Median prefill TPS          797.818        828.479 (+3.84%)
Prefill range               793.798-805.982 792.263-838.657
Median decode TPS           16.7408        16.7979
Maximum sampled peak KiB    3053760        3052128

Prefill improves in two of three pairs at each length, with overlapping
ranges. These small samples show a positive median result, not a guaranteed
gain on every run. The decode kernel is unchanged; its small timing changes
are not claimed as decode improvements. Candidate sampled peaks are about
2.606 and 2.911 GiB. No memory reduction is claimed from this variation.
All 12 normal runs match every reference output ID. Two consecutive prompts
on the same model/handles at each length also match all IDs. Gemma phone
benchmarks were not repeated. Neither 1000 prefill nor 25 decode TPS is met.

Actual execution check, separate from performance
------------------------------------------------
A diagnostic-only build counts actual disjoint-helper invocations atomically
and snapshots the difference around each successful plain MoE RPC after all
workers drain. Each prompt produces one load warmup plus 38 real MoE calls.
The shared loader warms at max(512, INIT_SEQ_LEN), so M matches the prompt.
Every call reports exactly active_experts * (24*22 + 16*16) = E*784 entries.
Real 1024 prefill: 2283 active expert instances, 1,789,872 helper entries.
Real 2048 prefill: 2328 active expert instances, 1,825,152 helper entries.
Both diagnostic generations also match all 65 IDs. Trace TPS is excluded
from the normal results. This proves execution, not isolated helper latency.

The trace adds a 4-byte global, one relaxed atomic increment per helper and
24 bytes to the RPC scalar frame. It has no new atomic runtime dependency.
It is archived as a patch, not retained in production. Root reversed that
patch and restored the exact frozen normal DSP SHA256:
b30be327d815af865452bb6f4a18c1d80c785a4c6abbdc0c1f06840345219273

Validation and limits
---------------------
Root integrated ASan/UBSan checks pass 1440 actual HVX primitive parity,
independent-oracle and canary cases plus 72 actual caller cases, using SDK
libnative. All three deliberately broken selection guards are rejected.
Cases cover tails, unaligned addresses, scalar fallback, invalid palette
codes, packed/snapshot inputs, Gemma/near shapes, sparse/full masks and eight
worker partitions. The independent reviewer reran these checks, existing
g++ direct-.c packing/preparation/conversion consumers, header C11/C++17
checks, four full v81 translation units and 784 ownership/partition extents.
Root also passed the existing LUT and expansion tests. Full v81 DSP build,
changed-line clang-format14 and diff checks pass. Android core reports no
work because these HVX C files are excluded from that build configuration.
The cross-platform CI matrix was not run for this branch push.

The full 96x128 helper has 36 -> 20 packed vector loads with the same three
LUT loads, 48 stores and 96 VLUT operations, with no vector spills. A separate
12-case v81 timing simulation reports about 566.12 -> 512.11 cycles per full
helper and 202.12 -> 180.11 per GU tail, with exact output bytes. It uses
ordinary simulator memory and excludes VTCM/DMA/HMX/caller scheduling; these
numbers are not converted to model TPS or DDR bandwidth. Host workers are
permuted serially, not a hardware race test. Existing lifetime/drain ordering
was audited and remains unchanged.

Diagnostic validation separately passes 112 actual RPC fixture cases and an
eight-thread actual-helper counter test (32,792 increments, 24 HVX parity
checks), under ASan/UBSan. ON/OFF v81 compiles verify inline atomic operations.

Memory is sampled from child exec/load through exit at a requested 20 ms
period (about 21 ms observed): maximum same-sample RssAnon plus fdinfo
allocation sizes, deduplicated by (mnt_id, ino). All peak scans have complete
readable-size fd identities. Allocation sizes need not be resident; unreported
DSP/driver/VTCM memory and between-sample spikes may be absent. This is not
total physical device memory and is not added to RSS/VmHWM.

Reproduction
------------
raw/ retains all normal and diagnostic run metadata, raw text, 65 IDs and
memory traces. artifacts/ identifies deployed binary and source hashes.
validation/ retains commands, full build logs, independent review, static
code generation, simulation, tests, repeat runs and the diagnostic patch.
The author/reviewer overlay starts at 83b323583; root applied the same change
on 8288e3ba3 with the independent M1 cleanup. root-frozen/source.json records
the exact integrated source hashes. The production helper and tests have
hashes 59306fe... and 6c345f4... in their respective patch artifacts.
The standalone test is test/htp/host/run_tl2_expand_disjoint_checks.py; its
commands.json records SDK libnative paths and flags. manifest.json hashes
every retained evidence file except itself. No model or runtime binary is
committed to this evidence directory.
