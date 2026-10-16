Native LFM prefill recency compaction — frozen opt-in candidate

Base: accepted ARM-router source 103b1e1df18efe8de8085f01167f29f2a4502178ac22e5c1aaab68dc38c036a8
(root commits bb23520b3 / evidence 742735f3 carry the same source).
production.patch: 1f130610bee92f91b5d8ca125c2dc661e42a89c4a5c6312e49c7748bba560f49
Candidate CPP: ac743dacb89bae9b3961452aedf7fdebe1b3f97a7e6c7d712ec23b6d7a2abbfa
Candidate header: b6fb960d6dff9615e367e18460288f8ea44a62d754978a5ebe0bfdd4b6a0b0ad
tests.patch: 30397390363ff7dc1e6fea28de969fbf9a3bf4081c172d83dbc7609628c4610d

Enable only with NNTR_LFM2_COMPACT_PREFILL_RECENCY=1 before the first call.
Default, 0, unset and other strings retain the full hint. No DSP/core/IDL,
ExpertLru, reader, slot, cache, preload or buffer-capacity changes are made.
No throughput benefit is claimed before the parent's actual-model comparison.

Scope and live placement
The gate is after the existing input.reshape({total_tokens,1,1,hidden_size}).
It therefore validates flattened [1024|2048,1,1,2048], not original prompt shape.
The original outer input is single batch/channel, from=0; virtual experts,
input norm only, one input, sigmoid/SiLU, E64/top4/I1536 are required. All 64 GU
and 64 DN tensors must be native TL2 with exact [1,1,2048,3072] and
[1,1,1536,2048] metadata, and input must be contiguous FP32/NCHW on HTP.
M1, Gemma geometry/softmax, mixed types, other engines and pre-existing nonempty
extra_top_k vectors retain the old path. A configured nonempty NNTR_MOE_TRACE
path (even failed fopen) OR a previously opened trace file disables compaction.

The existing sigmoid, bias, std::partial_sort(9), top4 weights, accumulation
order and per-expert token append are byte-identical source. The complete
sorted prefix is still computed for every row, even after all 64 IDs are seen.
Only extra_top_k append is compacted; a 64-bit seen mask retains FIRST ID
occurrences in original token/rank order. Generic ExpertLru is byte-identical.

Order/alias/concurrency proof
Let S be the original 9*M hint and U its first-occurrence subsequence. The
existing consumer constructs reverse(S), then calls refresh under one mutex.
For any resident key, its last touch in reverse(S) is its first occurrence in
S. reverse(U) preserves that last-touch ordering. This remains true when
multiple IDs map to the same Weight/Tensor object: its first position is the
earliest first position of any aliased ID, which is retained. Distinct Tensor
objects sharing data storage remain distinct keys; no storage-pointer dedup
was introduced. Absent keys remain absent and untouched.
refresh holds its mutex across the whole list, and touch only splices the list
and updates its iterator. It does not modify score_, clock_, held_ or capacity.
All readers/mutators of order_ and pos_ use that mutex; order() returns a copied
snapshot. capacity() and setLrfu() do not read list state (their existing external
synchronization contract is unchanged). Thus intermediate duplicate touches
are not observable through the list API. Existing future acquire/evict and LRFU
tie ordering are preserved, including held slots and aliases.

Work and memory, not a timing claim
The old retained 1024-token route contains 350,208 = 38*9*1024 entries. Its
first-seen subsequences contain 2,405 entries (59..64/layer), avoiding 347,803
integer appends plus the corresponding Weight lookups, key-pointer writes and
LRU lookups/splices. This is 9*M per layer, not nine million entries. That old
trace is not a measurement of current scores. At 2048 the mathematical bound
is 700,416 -> at most 2,432 entries across 38 layers; actual values need the
parent's diagnostic. The new gate itself reads 128 existing weight metadata
records once per eligible layer. No isolated elapsed-time bound is established.
The existing extra vector now reserves 64 ints (256 logical payload bytes) on
the fast path; later existing recency reserves at most 64 pointers (512 bytes
on AArch64). These replace larger versions of the same two vectors, not new
activation/weight buffers. sig/scored vectors and expert assignments are intact.
New per-call mask is 8 bytes, plus bool metadata; native compiler frame for
buildExpertAssignments is 528 -> 544 bytes and incremental_forwarding remains
3008 bytes. The inline gate adds a 1-byte static option plus its 8-byte C++
initialization guard. No task or new heap object outside existing vectors.

Validation
candidate/test/htp/host/run_lfm2_prefill_recency_check.py extracts the real
before/after complete assignment methods, constants, actual reshape, actual
caller gate, trace helper and native metadata helper, and links unchanged real
ExpertLru. It asserts byte-identical FP/selection body and unchanged LRU source.
Tensor metadata/getWeight/reshape are stand-ins; reshape checks element counts.
No numerical router/method or cache policy oracle is substituted for the real
implementations. The first-ID expected list is a small independent oracle.
Each process tests 49 numeric cases, 162 guards, 72 real LRU future-transition
cases and 2 concurrent-observer cases. Includes ties, signed zero, subnormal,
Inf/NaN logits or bias, bit63, early/late full masks, empty prompt, generic
E32/65/128 and softmax, nonempty/null hint, alias maps and absent keys. No new
guarantee is claimed for previously undefined/nonfinite comparator ordering;
both actual methods perform precisely the same selection on the same bytes.

Own checks/final-san: ASan/UBSan halt + leak detection, six environments PASS.
Own checks/fast: -O3 -ffast-math, same six environments PASS.
Own checks/generic: non-AArch64/HTP extraction, six environments PASS; no gate.
Whole TU: before/candidate Android AArch64, candidate Android x86 and Linux GCC
PASS (codegen/*/command.json). Production and fixture clang-format 14 exact.
checks/native/check is the same actual method/LRU fixture compiled by NDK at
-O3 -ffast-math; SHA abf446746fe3a2d795d4057a75d6aaf4009d2e8eacc027310b61e5777c5286b6.
Parent independently ran all six environments on S26: each 49/162/72/2 PASS,
recorded under /tmp/nntr-lfm2-prefill-recency-root-20261017/root-native-results.json.
No agent device access occurred. Model results belong to the parent separately.

Reproduce (no local SDK path needed for host)
  python3 candidate/test/htp/host/run_lfm2_prefill_recency_check.py \
    --source candidate --reference before --output /tmp/recency-own --sanitize
Add --fast-math instead of --sanitize for O3/fast-math, or --generic for fallback.
For native build add --native-build --cxx /path/to/aarch64-linux-android29-clang++.
The script never runs adb. It records exact extraction/source/command hashes.

Preserved draft failures and limits
checks/host-san failed because clang-format sorted separate caller.inc before
reshape.inc; prepared-caller.inc now contains both in actual source order.
This test exposed the wrong fixture execution order, not a production failure.
Initial source draft used unflattened metadata; reviewer caught it before any
source freeze or model deployment. Final gate explicitly tests flattened input.
Observer start handshake now follows its first snapshot to avoid a zero-sample
scheduling false failure. Native/fast command now has one effective -O3, not
an earlier -O3 overridden by later -O2. Old logs remain, final paths are above.
Fixtures prove method/metadata/LRU equivalence; they do not model DMA, file I/O,
DSP output, complete-model state or throughput. AArch64 native parity and model
65-ID/peak/dispatch checks are separately managed by the parent.

Diagnostic-only overlay
diagnostic/diagnostic.patch adds one guarded stderr record AFTER the actual
assignment call. Compile only that app TU with NNTR_LFM2_RECENCY_TRACE=1.
No production default or route-trace semantics changes. Macro0 whole-TU
disassembly is identical to production (diagnostic/codegen.json); macro1 whole
TU compiles. Actual formatter with stand-in metadata produces four complete
cohorts, rejects M1/M64/M512, and max unsigned/size values yield 82-byte lines.
Strict parser requires exactly ordered layers 0..37, requested M, raw=9*M,
expected fast flag and valid kept extent. It rejects duplicate/missing/reordered
or truncated lines; 65 invalid cases PASS. It requires one process's stderr and
does not encode/validate PID. Counts prove dispatch/materialization, not timing.
