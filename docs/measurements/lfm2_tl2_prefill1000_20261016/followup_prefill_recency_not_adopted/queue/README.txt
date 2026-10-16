LFM prefill queue and recency audit, 2026-10-17

Scope and decision
------------------
Read-only source/evidence audit. No production edits, builds, existing-fixture
reruns, device work, commits, cache/slot/reader changes or new kernel were made.
The current matched 1024 OFF/ON trace has no reader starvation to recover by
moving the existing read submission ahead of router/top-k. The bounded source
candidate is native-LFM-only first-occurrence recency materialization, retaining
the original selection/weight arithmetic and final cache order. No TPS gain or
candidate-only time estimate is established. Implementation awaits root approval.

The source was read at root HEAD 80a8b754e86b7594ee7400e1ca2ef841d22f2920
with the independently reviewed ARM-router candidate applied. Source and raw
input hashes are in existing-evidence.json and matched-{0,1}-summary.json.
The ring source also present during initial reads is outside this audit's files.

Matched normal-app diagnostic evidence
-------------------------------------
Input: continue-router-arm-io1024-{0,1}.log under the root measurement directory.
Root reports the same normal app/core/original DSP, router option 0/1 only, exact
65 output IDs and fixed C16 / preload128 / PAGE0 / GROW0 / six prefill readers.
Both runs enable existing PROFILE1, PREFILL_IO_TRACE, M0 and layer profiling.
These instrumented runs are mechanism evidence, not the normal performance matrix.

                                       OFF             ON
  Joined batches                        65              65
  Unique expert reads                  2304            2304
  Direct-read calls                    2304            2304
  Requested read bytes           4656726016      4656726016
  Read window, ms                  1178.197        1179.362
  Sum of six read-body times, ms    7063.616        7064.392
  Sum of pread time, ms             6752.897        6730.627
  Sum of copy time, ms               268.326         288.874
  Remaining read-body work, ms        42.393          44.891
  Exposed batch wait, ms              28.807          42.162
  Registration bracket, ms             1.652           1.938
  Read-body capacity utilization     99.921%         99.834%
  Definite empty-queue idle, us              0               0
  M0 top-k, ms                       29.088          30.346
  M0 FFN, ms                       445.572         488.755
  M0 other, ms                     157.881          80.223
  Backend MoE brackets, ms          378.964         409.125

The ordered (batch, expert, offset, bytes, calls) stream is identical. Strict
checks require exactly 65 unique batch IDs, 2304 distinct offsets and complete
per-batch expert indices; successful rc, one syscall per expert and monotonic
timestamps; 102 load/warmup compute calls excluded and 120 main compute calls,
including exactly 38 MoE calls. Warmup can have M=1024 and is excluded using the
load/repack boundary, not merely by geometry. M0 requires exactly 38 ordered
1024-token records with conserving named timers. No malformed tag is ignored.

The service/(6*window) value measures occupied read-function bodies, including
blocking I/O and copies, NOT ARM CPU utilization or a hardware bandwidth limit.
The observed window minus service/6 is 0.928 / 1.963 ms. This is a conditional
lower-bound gap at those exact observed per-expert service times, not a speedup
claim; it includes start/finish imbalance and scheduler/dispatch gaps.

Queued-but-not-started jobs exceed six for 99.513 / 99.440% of each read window,
which proves at least one job remained in the FIFO even allowing all six workers
to have popped a job before recording its start timestamp. There is no interval
inside the recorded read window with zero read bodies and zero submitted jobs.

The substantial waits occur at MoE ordinals 26, 27, 31, 34 and 35 (zero based),
plus ordinal 30 in ON. Their batches were queued 228--270 ms before demand, with
417--477 earlier jobs in the actual FIFO at enqueue. Thus those waits do not
come from submitting those reads after routing, a reader batch barrier or missing
registration concurrency. The reader queue was already occupied. The extra ON
wait is not itself evidence of a router regression: this is one profiled pair,
MoE/DSP timings also differ, and normal performance comes from root's matrix.

The M0 FFN timer includes stage/queue preparation, batch wait/register, backend
MoE and recency. Subtracting the backend MoE bracket plus explicit wait/register
leaves 36.149 / 35.530 ms. This broad remainder includes TRACE formatting and other
host work, so it does not measure recency alone. The previous standalone router
diagnostics had top-k 50.746 ms at 1024 and 122.214 ms at 2048; those are separate
runs and are not substituted for this matched pair's 29.088 / 30.346 ms.

Source and ownership findings
-----------------------------
lfm2_moe_layer.cpp:244 beginHtpExpertPrefill starts seven full 64-expert batches
in the existing C608 pool before the first layer, retaining the 128 preloaded
experts and preserving later partial batches. No capacity is left unused merely
because a later batch is partial once the ordinary per-layer path executes.

At lines 1570--1665, take_batch joins/registers batches due for the current layer,
stage(active) acquires the selected current experts, then the caller reclaims
finished-layer slots and submits future batches before the current compute RPC.
held_ excludes in-flight slots from room/acquire. The current exact active set is
only known after top-k. Before top-k, conservatively pinning all 64 current experts
would change partial-batch sizes and potentially the sequence of LRFU acquires;
moving submission is not a trivial text reorder that preserves all bookkeeping.

htp_compute_ops.cpp:4834 begin takes actual arena slots before enqueuing jobs.
At 9404 each reader pops one FIFO job; after the individual read it increments
done and immediately pops another job, including from the following batch. There
is no reader-side whole-batch barrier to remove. end at 4875 waits the oldest
batch only because the current compute requires its experts.

The stable native-TL2 path at 9275 already avoids reused-slot registration RPCs:
publishStableTl2 publishes a new generation after payload completion, and
fileRegistered updates host maps/handles. The existing arena's code bytes are
not copied a second time by that registration. The shared DSP metadata DMA path
cannot safely be invoked arbitrarily during an unrelated compute RPC; immediate
registration is neither a missing generic overlap nor a large measured budget.

The PAGE0 read path already coalesces GU/DN into one direct read. It scatters from
the existing aligned scratch into the arena and converts metadata tails. Existing
fadviseKnob returns zero under explicit page-cache policy, so there is no hidden
per-expert fadvise syscall in this configuration. No new redundant full-payload
copy was found. Prior rejected scatter/read/barrier candidates are not re-proposed.

The future-resident pinned list is rebuilt inside the submit loop even though
makeRoom only evicts outside that list and no batch is registered inside that
loop. Hoisting that existing construction is a possible small bookkeeping cleanup,
but it has no separate time measurement and cannot recover absent FIFO starvation.
It is lower priority than the directly counted recency materialization below.

Native-LFM recency candidate design (not implemented)
---------------------------------------------------
buildExpertAssignments at 601 allocates sig/scored once per layer and reuses them
per row. It does NOT allocate those two vectors once per row. The sigmoid and
partial_sort extended prefix 9 execute once per row; assignment vectors grow as
top-4 pairs are appended, and extra_top_k grows to 9 * tokens entries.

At lines 1684--1693 every extra ID is converted into a Weight key in globally
reversed order, then ExpertLru::refresh performs one lookup/splice per entry.
refresh holds its mutex for the complete vector. Its touch at expert_lru.h:299
only splices the list and updates pos_'s iterator. It does not change score_,
clock_, held_ or capacity_. Public order()/resident()/size()/acquire()/makeRoom()/
reset()/refresh() use the same mutex, and no private list iterator escapes. The
unlocked capacity()/setLrfu() methods do not observe/change the list or intermediate
touches. Thus no supported observer sees intermediate redundant splices.

Let S be the original token-major/rank-major extra ID sequence and R=reverse(S)
the current refresh sequence. The last touch in R for an expert corresponds to
its FIRST occurrence in S. If U keeps only the first occurrence of each ID in S,
refresh(reverse(U)) yields the same final touched-key order; untouched resident
keys retain their prior order. Skipping nonresident keys commutes with this
reduction. For a non-injective ID->Weight-key mapping, each key's first position is
the minimum first position of its aliased IDs, which remains present in U. Thus
aliased expert weights preserve the same final key order too; at most some
redundant touches remain among the <=64 IDs. No cache policy correction is made:
the globally reversed sequence's existing FIRST-occurrence policy is preserved,
even though a nearby comment suggests last-token recency.

Proposed narrow scope:
  * Explicit opt-in, AArch64+HTP, virtual native-TL2 LFM K2048/I1536/N2048,
    E64/top4/SILU/sigmoid, normal 1024/2048 prefill. All GU/DN types verified.
  * Routing trace configured => original full materialization, preserving every
    token's nine IDs and the existing trace format. Decode/Gemma/generic paths
    retain the original behavior; generic ExpertLru remains unchanged.
  * An internal compact-recency argument defaults false. In the qualified call,
    the extra output uses one call-local uint64 seen mask and the existing vector
    for first-seen IDs. No new persistent state/weight/activation buffer. Existing
    reverse conversion and refresh remain ordered as before.
  * Keep every sigmoid, score, partial_sort(9), top-4 ID/weight, weight summation
    and expert/token assignment append unchanged. Once all 64 IDs have appeared,
    the extra-output loop alone may be skipped; selection still computes top9.

Older saved 1024 routes (SHA406e9a3a829af344765a7a1012a45f64fab949e887bfb5a71576bf8a03e7dcb0)
contain 38 * 1024 * 9 = 350,208 entries. First-seen output totals 2,405 (59--64 per
layer), removing 347,803 extra integer writes and the same number of later key
materializations and refresh lookups. Logical bytes avoided are 1,391,212 integer
bytes plus 2,782,424 pointer bytes on AArch64, excluding vector growth/allocator
traffic. These are a prior saved-route work count, not measured timing or the new
matched run's actual routing capture. For 2048 the input count is 700,416 with an
upper bound of 2,432 compact IDs, not nine million and not a measured 2048 count.

This is separate from the rejected top-k insertion trial. That trial replaced
partial_sort, passed standalone exact replay, but regressed all three normal
1024 prefill pairs (828.479 -> 819.856 median) and top-k profiling 24.839 ->
27.658 ms. Its evidence is followup_rejected_cpu/README.txt; do not repeat it.

Possible later prefix-4 reduction must remain separate
----------------------------------------------------
Once all 64 first-seen IDs are recorded, future extra IDs cannot affect recency.
For a row whose selection scores contain no NaNs, the (score,ID) comparator is a
total order (including +/-zero ties and infinities), so partial_sort's top4 is
identical for prefix4 and prefix9. But a full seen mask says nothing about NaNs in
later rows. A finite ARM dot output is insufficient if expert bias contains NaN.
The existing comparator is not a strict weak order over NaNs; different prefix
lengths can then behave differently. Preserving existing abnormal-input behavior
requires a per-row bitwise score qualification and prefix9 fallback, adding new
work/selection code and codegen risk. Keep prefix9 in the first materialization
candidate rather than bundle this unmeasured change.

Validation required if implementation is approved
------------------------------------------------
Extract/execute the actual production assignment method and actual ExpertLru,
not a mirror alone. Compare full row IDs and FP32 weight bytes, unchanged top4
sum/append order, and final LRU order followed by future acquires/evictions under
LRU and LRFU. Include duplicates, absent keys, arbitrary initial orders, aliased
Weight keys, first64-seen early/late/never, zero/full masks, trace on/off, unsupported
model guards and special FP inputs including signed zero, ties and nonfinite
scores. Reuse the existing native captured-score check only if root approves;
host ASan/UBSan and actual Android compiler/native parity precede model trials.
No separate performance budget follows from the current broad top-k/FFN timers.

Trace interpretation and reproducibility
---------------------------------------
Use analyze_existing.py --repo R --output existing-evidence.json and
analyze_matched.py --repo R --output-dir THIS_DIRECTORY. The latter additionally
uses the retained validation/analyze_prefill_io.py for standard summaries after
its own strict record/cohort checks. All generated files stay outside root.

queued_us/jobs_ahead are recorded under the enqueue mutex before push. A READ
start is recorded after pop/unlock/cpuBit, so submitted-minus-started is not the
exact deque size: it can include at most six already-claimed jobs. The analysis
uses >6 for a definite remaining queued job. Read-end precedes the done counter
increment; small ready-count/time discrepancies cannot be interpreted as missing
work. Batch records are emitted after join by the caller, so trace formatting can
delay subsequent caller submission/compute even though readers never print.
Existing fields lack explicit free/held slot timestamps. Additional slot markers
would be needed to distinguish such causes if a future trace actually showed
starvation. This matched pair does not justify adding that instrumentation now.
