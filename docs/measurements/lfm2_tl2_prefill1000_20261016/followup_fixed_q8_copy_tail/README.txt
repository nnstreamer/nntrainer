Fixed-Q8 master-row copy boundary repair — correctness evidence

The original fixed-Q8 HD64 control failed a legal A/B/A rewrite on the S26 Ultra.
The unchanged fixture passes after the narrow xor80_copy tail fix. This archive
supports that correctness repair independently of the proposed raw-plane
attention optimization. It contains no model throughput claim and no measured
Gemma model performance result.
The current dynamic LFM path does not use the repaired fixed-Q8 row-copy branch;
this is a separate correctness commit, not an LFM speed optimization.

Problem and fix

The KV API accepts head dimensions that are multiples of 32 up to 512 and
explicitly permits row rewrites. Fixed Q8 uses plain masters with exactly hd
bytes per row. The old vector loop copied 128 bytes whenever i < n, including
hd=64. Its final master-row store exceeded the allocation by 64 bytes, and
adjacent row workers could overwrite each other's row bytes. The actual fixed
quantizer already writes exactly hd bytes; the following copy was incorrect.

The fix preserves the existing full 128-byte load/XOR/store loop, bounds it to
n & ~127u, and XOR-copies only the remaining bytes. It changes no FP operation,
quantization, row layout, worker count, synchronization or buffer capacity. No
cache, reader or allocation is added. Existing full-vector HD128/HD256 paths
retain their byte behavior; the runtime-variable helper adds prefix/tail checks.
Constant-size wrapper assembly identity does not establish generic caller or
Gemma model performance identity.

Source and test identity

Production patch: c86726cb86782f33b820dafa9af5120c6831506ff991b8195d1f17b0a9cbf6ac
Corrected C: fa750e41854303b716990c5d722e8b14a0c85e1776e26096ff880acaf007c291
Baseline C: 6f20a5a4cbd0c4614c24b91cbd0c0c16f15f640e1d53014b66014b5039c4a46d
Final canonical test C: 1099fec0182ed53d6aa13592f06980e0ba0c77bf74894f589ff2fbde492175de
Final runner: e02c79d5ec51bea65f8ecd33bf82f6bcdfba0976bd2103ac7e34b188bbfcdc32

The root applied clang-format 14 to the test C, changing indentation only from
author b0e57b26. source/test-format-only.diff.gz preserves that difference, and
source/tests-canonical.patch matches the final canonical files. Root reran the
strict fixture on this final source; production stayed fa750e41. Author and
independent frozen reports are preserved verbatim: their earlier "device
pending" statements are superseded by the corrected device records here.

Validation

Author, independent reviewer and root canonical runs each pass 3,094 actual
copy cases and 384 actual fixed-quantization worker A/B/A cases. The original
baseline separately fails with an ASan heap overflow and a deterministic reverse
worker-order master-byte mismatch. Corrected dedicated regressions also pass.
ASan/UBSan are fatal without sanitizer suppression. The tests exercise all 16
legal head dimensions, tails and offsets, partial last-row rewrites, reverse
completion, and 1/2/4/8 pthread workers. They use the actual extracted copy and
worker plus actual SDK libnative HVX quantization. HMX and device cache behavior
are not simulated by these host tests.

Independent v81 whole-TU builds passed. All function stack frames remain equal,
including quant_rows_worker 896 B and append_fixed_i8 1,408 B. Constant full-vector
copy wrappers have identical assembly; the non-Hex whole object is byte-identical.
Only the latest meaningful host run from each participant is retained.

Original and repaired device outcome

The original verified control DSP 3855767890... returned 1 during fixed-Q8 HD64
case 3, repetition 2 (the final A). Dynamic-Q8 1024/2048 and Q4 A/B/A cases had
passed before that failure. No raw-plane candidate fixture had run at that point.
The partial dump was 117,440,512 bytes with SHA
776062a1a0b33b33f70821c21091c4bf0f2a867c80dda81399dad294b6857fb3.

The same fixture binary 351ca73b... then passed on corrected control, raw-plane
candidate and diagnostic. Each completed 36 attention calls plus 4 signed
accumulator probes, returned 0, and produced exactly 231,908,352 bytes. All three
full dumps have SHA
e64b0a2342d8132bd09e23763ca8958270c62a8e2a99dd720e1cb27f4478c394.
The corrected control's fixed-HD64 A/B/A replay is exact. The independent
readback report also verifies the complete cross-arm bytes, changed middle B,
and the original successful output prefix. These are correctness fixtures,
not tokens-per-second measurements. Dump files themselves are deliberately
omitted; device/dump-hashes.json records independently streamed hashes and sizes.

All three repaired arms use the same corrected KV object. The common-object
builder first reproduced the original KV object through a no-op overlay, then
compiled only corrected KV and relinked each arm twice identically. Each arm has
63 active objects, with 62 common between normal arms: 61 prior common objects
plus corrected KV. Only the attention TU differs. The raw-plane production patch
d8d182 was not folded into this correctness patch. Full seven-runtime and fixture
hashes are preserved in device execution records; all six non-DSP runtime files
are unchanged between the original failed control and corrected control.

Memory and interpretation limits

Original and corrected control memory-monitor samples are preserved verbatim,
compressed. Monitoring spans each short fixture from before exec through exit,
at a requested 20 ms period. The original run terminated early, so these runs
do not provide comparable model peak-memory or timing measurements. The sampled
composite is RssAnon plus deduplicated fdinfo size accounting, not total RSS or
a proven complete accelerator-memory census. dma_buffer_identity_verified is
false. Corrected control/candidate/diagnostic have respectively 5/1/5 samples
with transient fdinfo read errors; each also has an unavailable terminal sample.
The peak-selected samples had no read errors, but other scan errors are retained
and must not be reclassified as full-memory certification. All monitor limits
remain in the raw records. No memory reduction or throughput result is claimed.

Contents and reproduction

source/ and tests/ contain the exact patch, gzip-wrapped before/after C,
canonical portable fixture, formatting-only delta and frozen host dependency
hashes. host/ contains
one final strict run per participant. independent/ and author/ preserve reviews;
build/ records the common-KV object provenance and exact compile/link commands.
device/original and device/corrected retain the failed/repaired control logs,
memory and execution metadata; device/independent links the complete cross-arm
readback. Large ELFs, objects, vendor binaries and dumps are excluded.

Run archive verification without SDK, compilation or device access:
  python3 -B verify.py

Re-run the packaged Linux host fixture with a matching repository and SDK:
  python3 -B reproduce_host.py --repo /path/to/nntrainer \
    --hexagon-tools /path/to/Hexagon_SDK/tools/HEXAGON_Tools/19.0.04 \
    --output /tmp/kv-fixed-tail-reproduction

The dependency check prevents accidentally testing against changed quantizers
or stubs. The reproduction script decompresses the two historical C sources into
a temporary directory, keeps them alive until the synchronous fixture completes,
then removes only those temporary copies. Build provenance scripts retain their
original environment paths and are not automatic device actions. Reconstructing
full DSP artifacts additionally
requires the frozen parent object set and external SDK inputs identified in the
build records. verify.py checks archived hashes and recorded relationships;
because dumps are omitted, it cannot independently recompare their bytes.

provenance.json maps each copied payload to its frozen source and original SHA.
manifest.json hashes every archive payload, including raw contents of gzip files.
The production patch and original whitespace-bearing logs are gzip-wrapped
without modifying their contents; decompress before applying or inspecting them.
Every frozen historical C snapshot is likewise gzip-wrapped so repository CI
does not reformat evidence. Only tests/kv_fixed_tail_check.c remains plain C;
its exact canonical bytes already pass repository clang-format 14.
No shared repository, accepted runtime, SDK or phone was modified by curation.
