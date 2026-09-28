# HTP MoE decode ledger — rules learned on silicon, verdicts, open items

Contract: `docs/plans/0001-htp-moe-decode-agent-system.md`. The supervisor
appends here; the planner and implementer read it before touching a kernel.
The tracker issue (#76) carries `hexagon` + `prio:*` and **no `state:*`
label**: state belongs to its children, so the tracker never occupies the
single `state:in-progress` slot.

## Upstream

PR nntrainer/nntrainer#4327, branch `claude/htp-lfm2-moe-ffn` on
`Seunghui98/nntrainer`. `htp_moe` is **synced to `4ae1ebd7` via PR #121**
(upstream merge commit `fd3bfba1` on the PR branch, PR merged by the user
as `b7d46b57`, 2026-09-27 11:30 UTC; #120's sitting is its device evidence,
§2 ㉔). History: from cycle 1 through cycle 16 it was frozen at head
`2ce38d65` (2026-09-21 06:45 UTC, "[CausalLM] Route conv out_proj and the
dense FFN to the HTP by config") plus the three cherry-picks `fb0f02b9` /
`04a2fcc4` / `b0a384d6` that PR #103 carried (`ad714de7`, cycle 6).
Upstream watch sha: **`4ae1ebd7`** (cycle 12; was `a996b4bf` from cycle 7
through cycle 11; unchanged through cycle 16b).

`hvx_conv_gate_f32.{c,h}` (upstream `7f81560b`) reached `htp_moe` through
PR #121's merge, so #82 is plan 82's case (a): it reuses the file unchanged
and takes no `hvx_scalar_stubs` (PR #125).

**New commits since (seen 2026-09-21, cycle 1; PR head `7f81560b`,
updated 08:10 UTC, fast-forward from `2ce38d65`, 31 files, +3490/−636).
Merging them into `htp_moe` is the user's decision (contract §5, Q16);
not merged.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `95caa139` | [docs] 50: the fifth run — every FC group on, per-shape calls measured | doc 50 |
| `fb0f02b9` | [HTP] Stage each call through an ION buffer of its own size class | `htp_compute_ops.cpp` (transport-side, relevant to ⑦ / #83). **Cherry-picked into `htp_moe` by PR #103 (#88)**, `docs/htp_attention` hunk dropped |
| `7497cfd6` | [docs] 50: size-class staging measured — decode back to 20.5 TPS | doc 50 |
| `d2f0bf47` | [CausalLM] Run the dense FFN as one HTP call through the MoE layer kernel | app, `hexkl_mm_u8i4_moe.{c,h}` |
| `35b6124b` | [HTP] Give the dense FFN's MoE-kernel calls their own profile row | profile |
| `20d18ab4` | [docs] 50: the no-switch run — today's baseline is 921–1013, not 848 | doc 50 §3.8: prefill **ms** on the author's unit, 921 ms = 482 tok/s / 1013 ms = 438 tok/s (was 848 ms = 523), decode 21.2 / 21.1; "the device is +73..+165 ms slower today, two consecutive runs 10 % apart" — rule 9 again |
| `80bf92ad` | [docs] 51: third C run — the dense row needs a rebuild to appear | new doc 51 |
| `39ce49d7` | [docs] 51: the dense-only run — 10.4 ms per fused call, as predicted | doc 51 |
| `7f81560b` | [HTP] Run the LFM2 conv block as one call: in_proj, gates, conv1d, out_proj | new `hmx/hexkl_conv_block.{c,h}`, **`hvx/hvx_conv_gate_f32.{c,h}`** (causal depthwise conv1d L=3 + gate, HVX, prefill shape), `test/htp/host/conv_block_host_check.c`, stand-ins moved to `hvx_scalar_stubs.{c,h}`, two IDL entries `mm_u8i4_conv_block[_timed]`, `conv_block_layer.cpp`, `compute_ops.h`; "not yet run on a device" |

**Cycle 2 (seen 2026-09-21 09:44 UTC): PR head moved `7f81560b` →
`72d9b140`, six more commits, all on the conv-block DIFF/compare path
(`Applications/CausalLM/layers/conv_block_layer.cpp`) and doc 51 — no
kernel, IDL or ring change. Still not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `5d190100` | [docs] 51: the conv block on device — 4.9 ms a call, prefill 732–741 | doc 51, `00_START_HERE.md` (the fused conv block measured on the author's unit: prefill 732–741 **ms**) |
| `81298c35` | [CausalLM] conv_block: NNTR_CONV_BLOCK_DIFF / _SHADOW discriminators | `conv_block_layer.cpp` |
| `9681f465` | [CausalLM] conv_block compare: reference through the CPU Q4_0 GEMM directly | `conv_block_layer.cpp`, doc 51 |
| `4b8b8f6f` | [CausalLM] conv_block DIFF: an activations-only reference and outlier ratios | same |
| `fe74b6cb` | [CausalLM] conv_block DIFF: emulate four weight recipes in f32 | same |
| `72d9b140` | [CausalLM] conv_block DIFF: emulate row-flattening and K-grouped int4 | same |

Reading: the author is chasing an accuracy discrepancy in the fused conv
block (four DIFF commits in 80 minutes); `7f81560b`'s "not yet run on a
device" is now run (5d190100) and under investigation. Nothing here
touches decode; a merge decision can wait until that line settles.

**Cycle 3 (seen 2026-09-22 ~01:00 UTC): PR head moved `72d9b140` →
`0a0c0402` (updated 2026-09-21 10:27 UTC), eight more commits. Files:
`conv_block_layer.cpp`, `tie_word_embedding.{cpp,h}`, `causal_lm.cpp`,
`hmx/hexkl_conv_block.{c,h}`, `htp_backend.cpp`, docs 00/51. No IDL, ring
or MoE-kernel change; one new runtime knob (`NNTR_HTP_POLL_US`) touches
the FastRPC poll window that #88 (wall 3) also works on. Still not merged;
user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `8d361580` | [CausalLM] conv_block DIFF: K-grouped int4 with the symmetric rule, groups 32-256 | `conv_block_layer.cpp` |
| `2e69c732` | [CausalLM] conv_block DIFF: the grouped-int4 recipes the previous commit described | same |
| `32b46e32` | [CausalLM] NNTR_PPL: the prompt's teacher-forced perplexity at prefill | `causal_lm.cpp`, `tie_word_embedding.{cpp,h}` — a PPL readout we could reuse as an accuracy column |
| `e2554a24` | [docs] 51: perplexity closes the conv block accuracy question | doc 51 |
| `04a2fcc4` | [HTP] NNTR_HTP_POLL_US: the FastRPC poll window as a knob | `htp_backend.cpp` — overlaps #88's area (transport). **Cherry-picked into `htp_moe` by PR #103 (#88)** |
| `07a9bec5` | [CausalLM] conv_block: drop the recipe emulation, keep DIFF/SHADOW and PPL | `conv_block_layer.cpp` |
| `e699e1bd` | [HTP] conv block: pipeline phase 2 one block ahead | `hexkl_conv_block.{c,h}` (prefill-shape fused conv block) |
| `0a0c0402` | [docs] 51: the perplexity split -- the dense FFN costs 4%, the conv block nothing | doc 51: the author's accuracy verdict — the fused conv block is PPL-neutral, the dense FFN through the MoE kernel costs 4 % PPL |

Reading: the conv-block accuracy line has settled (PPL-neutral); the
dense-FFN-through-MoE-kernel path (`d2f0bf47`) costs 4 % PPL on the
author's measure, which would fail our gate (c) if we ever routed the dense
FFN that way. `04a2fcc4` is the one commit with a bearing on our decode
work (wall 3); the rest is prefill/accuracy. A merge decision can still
wait; if merged, the IDL from `7f81560b` forces a skel + app rebuild.

**Cycle 4 (seen 2026-09-22 ~02:30 UTC): PR head moved `0a0c0402` →
`b0a384d6` (updated 2026-09-22 01:54 UTC), three more commits. Files:
`htp_compute_ops.cpp` (profile accounting), `htp_backend.cpp` (FastRPC
poll default), docs 00/51. No IDL, ring, weight-layout or MoE-kernel
change. Still not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `6f8f7791` | [HTP] Keep the conv row's hidden worker time out of the mm residual | `htp_compute_ops.cpp` — `[HTP-PROFILE]` residual arithmetic for the conv row; prefill-only row, but the same table #94 B reads |
| `faba38d1` | [docs] 51: correct why the dense FFN is expensive -- down's depth, not the chunking | doc 51 |
| `b0a384d6` | [HTP] Poll the FastRPC reply for 5 ms by default | `htp_backend.cpp` — makes `04a2fcc4`'s `NNTR_HTP_POLL_US` knob default to 5 ms. Transport-side (wall 3); the #88 plan read it (author's unit: decode call transport 158 → 83 µs, doc 51 §2.20). **Cherry-picked into `htp_moe` by PR #103 (#88)**, `docs/htp_attention` hunks dropped |

Reading: `b0a384d6` is the second upstream commit in #88's area (after
`04a2fcc4`). Two consequences: (1) the #88 planner reads doc 51's poll
numbers before writing the design; (2) if the user merges before #88's
handoff, variant A of that sitting already carries the 5 ms poll and #88
is read against it — cleaner than merging in between. Nothing here touches
#94's artifact set.

**Cycle 5a (seen 2026-09-22 ~04:30 UTC): PR head unchanged at `b0a384d6`.**

**Cycle 5b (seen 2026-09-22 ~05:30 UTC): PR head moved `b0a384d6` →
`80fa1a1a` (updated 2026-09-22 05:14 UTC), four more commits. Files:
`hmx/hexkl_mm_u8i4_moe.c` (+370/−226), `htp_compute_ops.cpp`,
`Applications/CausalLM/layers/qkv_layer.{cpp,h}`, `lfm2_causallm.cpp`,
`transformer.cpp`, two LFM2 unit tests, doc 51. One commit
(`5731b6e5`) rewrites the MoE kernel's down-matmul pipeline — the same
file and the same DMA schedule that #94's trace measured and that the
wall-2 fix issue (#100) targets. Still not merged; user decision.**
Oldest first:

| sha | subject | touches |
|---|---|---|
| `3006d255` | [HTP] Give plain FC calls their own profile row | `htp_compute_ops.cpp` (profile rows; #94 B reads this table) |
| `5731b6e5` | [HTP] Issue the MoE kernel's down matmul one block behind | **`hexkl_mm_u8i4_moe.c`** — pipelining of the down matmul behind gate/up in the HMX block loop; changes the 46-descriptor issue order #94 traced (rows f/g/h). Prefill-shaped by intent (doc 51's "all-on run"); its M=1 effect is unmeasured |
| `1f538c20` | [CausalLM] Fuse LFM2's q/k/v projections and their norms into qkv_layer | `qkv_layer.{cpp,h}`, `lfm2_causallm.cpp`, `transformer.cpp`, tests — ARM-side attention projections (the ARM remainder, ⑨ / #81 area) |
| `80fa1a1a` | [Docs] Record the 585 ms all-on run: pipeline and qkv sum to −33 | doc 51: the author's prefill at 585 ms with everything on (−33 ms from the two commits above) |

Reading: `5731b6e5` is the first upstream commit inside the MoE kernel
since `htp_moe` froze. If merged, #94's trace / attribution table
(descriptor order, `first expert ready`, rows f/g/h) must be re-taken as
variant A of the next sitting before #100 is read against it; if not
merged, #100 works on the frozen list and a later merge re-opens the
question. Either way the planner of #100 reads that diff first. The GEMV
path (C) bypasses this loop entirely, so ⑯'s verdict is unaffected.

**User merge decision (cycle 6, 2026-09-22): PR #103 (#88) merged by the
user as `ad714de7`.** That merge is the user accepting the three upstream
cherry-picks `fb0f02b9` (size-class ION staging), `04a2fcc4`
(`NNTR_HTP_POLL_US`) and `b0a384d6` (5 ms poll default) into `htp_moe`,
with their `docs/htp_attention` hunks dropped. Nothing else from #4327 is
merged; the rest of the tables above and below stays a user decision.

**Cycle 6 (seen 2026-09-22 ~06:30 UTC): PR head moved `80fa1a1a` →
`5622c743` (updated 05:53 UTC), four more commits. Net files:
`hmx/hexkl_mm_u8i4_dma.c` (+104/−37), `hmx/hexkl_mm_u8i4_moe.c`
(+8/−3, a comment and one moved line), `htp_compute_ops.cpp` (+6, a
comment), `test/htp/host/fc_layer_host_check.c` (new), host stubs, docs
00/51/52. No IDL change; the M=1 GEMV path (#105) is not touched. Not
merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `7e0b0e71` | [HTP] Three small exposures: early gate_up chunks, pooled FC epilogue, MoE args in rpcmem | moe, dma, compute_ops — two of the three reverted by the next commit |
| `aa371dd2` | [HTP] Revert the early gate_up chunks and the rpcmem MoE args; keep the pooled FC epilogue | net: `hexkl_mm_u8i4_dma.c` plain-FC epilogue double-buffered onto the pool (prefill FC calls, not on our decode config); two **measured negatives** left as comments: (a) pushing the next expert's gate_up chunks early put the activation behind them in the ring, GATHER 84 → 747 µs/call (388 at decode), "the DMA moves ~12 GB/s beside the HMX, not the idle 33" (doc 51 §2.26) — input for #100 / rows f–g; (b) the MoE call's five small sequences from one rpcmem buffer: transport 627 → 605 µs, noise, "the gap to the FC calls' ~400 is the call's length … a call longer than the poll window ends in the interrupt-driven wait" — input for reading handoff 88 |
| `7ada48b7` | [Docs] Record the pooled-FC-only run and the poll 9000 experiment | doc 51 (a longer poll window tried) |
| `5622c743` | [Docs] 52: the flash-experts (cached-slim on HTP) task, self-contained | new doc 52: the author's next task is **memory** (peak RSS 5.2 GB, 3840 MiB expert arena → LRU/flash-loaded experts on the HTP path, "DSP kernel untouched"); records the author's all-on state as prefill 585–616 ms / decode 24.2 TPS and "no HTP software lever left ≥ 5 ms" for prefill |

Reading: nothing here changes a decode kernel. Two of the comments are
free measurements for our open work: (a) a ~12 GB/s ring rate beside the
HMX, which bears on #100; (b) an rpcmem-args null, which bears on #88. Doc
52 says the author's line is now moving to memory residency. If that line
lands, it would change the arena our GEMV path reads directly (#105, ㉒),
so it is worth knowing about before any merge.

**Cycle 7 (seen 2026-09-22 ~07:30 UTC): PR head moved `5622c743` →
`a996b4bf` (updated 06:55 UTC), one commit, docs only
(`docs/htp_attention/00_START_HERE.md` +1, new
`docs/htp_attention/53_int_requant_task.md` +114). No code, IDL or kernel
change. Not merged; user decision.**

| sha | subject | touches |
|---|---|---|
| `a996b4bf` | [Docs] 53: the int32-to-u8 requantization task (arXiv 2511.11248), self-contained | new doc 53 (Korean): a task note for replacing the MoE kernel's int32 → f32 → u8 epilogue by a direct int32 → u8 requant; the paper itself not yet read. Its own reading: at prefill the epilogue is hidden under the HMX (0.35 ms/call exposed, 2.4 %), the gain would be structural (VTCM `gate_off` 448 → 112 KB, the row-scan barrier) and the gate is PPL (`NNTR_PPL=1`, 62.09 ± 0.5 %); **at decode (M=1) "not applicable"** — the author's M=1 call is 1.44 ms with dequant 1.1 µs, gather 128, acc 259, mm 764 |

Reading: nothing for the decode goal. It shares the requant stage with #95
(⑱, Hadamard before the u8 requant, PR #106), so whoever reads #95's
verdict should know an int-requant rewrite of the same stage is the
author's next prefill topic; a merge would conflict there, not in the M=1
GEMV path.

**Cycle 8 (seen 2026-09-22 ~08:30 UTC): PR head unchanged at `a996b4bf`
(updated 06:55 UTC).** One relevant fact from the #95 sitting: the
side tree `htp_hadamard` (upstream `3006d255` + tooling) reads decode
**25.08 / 24.40 / 23.83 tok/s** on its own control A (`R3CY205ZMND`,
same day). `htp_moe`'s #88 B read 24.50 / 23.80 / 23.52 in another
sitting. Rule 9 forbids reading this as a verdict. It is only
consistent with both trees carrying the same transport fix (`fb0f02b9` /
`b0a384d6`) and neither having the GEMV on.

What this changes for us: (1) the "net-new" causal conv1d + gating of §4
now exists upstream in prefill shape — #82 lifts it instead of deriving
it; (2) the ARM decode path of the conv block is still the CPU one there
(the fused call runs only at prefill); (3) doc 50 §3.8 re-measured the
author's unit 10–16 % slower on the same binary (prefill 482–438 tok/s,
not 523), one more reason the provisional "now" is replaced only by #77's
control run on ours; (4) the IDL changed, so a merge forces a
skel + stub rebuild and new md5s in BENCHMARK.md.

**Cycle 9 (seen 2026-09-22): PR head unchanged at `a996b4bf` (updated
06:55 UTC).** Nothing to decide.

**Cycle 10 (seen 2026-09-22): PR head unchanged at `a996b4bf` (updated
06:55 UTC).** Nothing to decide.

**Cycle 11 (seen 2026-09-23): PR head unchanged at `a996b4bf` (updated
2026-09-22 06:55 UTC).** Nothing to decide. Note for whenever a merge is
considered: `5731b6e5` (the down-matmul pipelining inside
`hexkl_mm_u8i4_moe.c`) was held back because it would invalidate #94's
descriptor trace. After #100 that reason is weaker — row h dissolved and
the M=1 GEMV path issues `desc=2/call`, so the 46-descriptor list is no
longer on the decode critical path — but `5731b6e5` is prefill-shaped
and still unmeasured at M=1, so the decision is unchanged, not urgent.

**Cycle 12 (seen 2026-09-23): PR head moved `a996b4bf` → `4ae1ebd7`
(updated 2026-09-23 02:14 UTC), five commits. Files: docs 00/52/53 and,
in one commit, `hmx/hexkl_mm_u8i4_moe.c` (+31/−13) and
`htp_compute_ops.cpp` (+16/−8) — profile accounting only, "arithmetic
unchanged", "not run on device". No IDL, ring, weight-layout or M=1 GEMV
change. Not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `2cb8a1ea` | [Docs] 52, 53: state the authorship rule as it is, and the prompts with it | docs 52/53 |
| `5cdd7a6f` | [Docs] 53: the paper is T-MAN, no int32-to-u8 hop in our kernels; closed at gate 0 | doc 53: the int-requant task is **closed** upstream at gate 0 — the conflict cycle 7 flagged for #110's requant site no longer exists |
| `c64be9dd` | [HTP] Time the MoE kernel's epilogue worker slices under the SWIGLU column | `hexkl_mm_u8i4_moe.c`: `moe_tail_probe_add` → `moe_worker_probe_add`, every pool job of the kernel timed under `HEXKL_PROBE_SWIGLU`; `htp_compute_ops.cpp`: a `swiglu_hidden` flag so the `mm` residual leaves that column out on every row, printed `swiglu(hidden)`. **Overlaps PR #108's fix of the same column on the GEMV row (㉑, rule 24)** — a merge conflicts there, in the print, not in a kernel |
| `6af38345` | [Docs] 53 8.4: the HVX epilogue in integer arithmetic, op by op, and the probe that decides it | doc 53 |
| `4ae1ebd7` | [Docs] 53 8.4: run 1 had no probe in the binary; same-session baseline recorded | doc 53 |

Reading: nothing for the decode goal. `c64be9dd` is the author's version
of what #102 / PR #108 did for the GEMV row (rule 24: `swiglu` is Σ
lane-time on that row); the two prints will conflict at merge time and
one of them has to win. No reason to merge now; if the user merges, the
GEMV row's `rest` arithmetic (PR #108) must be re-checked against the new
`swiglu_hidden` flag before the next profile is read.

**Cycle 13 (seen 2026-09-23): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC).** Nothing to decide. On our side: PR #115 merged
as `c45d4433` (D192 default, `727862c7` + `37fffcb4`), PR #116 (guide) as
`42370b95`; `htp_moe` head `42370b95`.

**Cycle 14 (seen 2026-09-23): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC).** Nothing to decide. On our side: **PR #118**
filed (`htp/117-m1-gemv-vtcm-feed` @ `e4b5d0f5`, #117 →
`state:needs-measurement`, on the device now — its sitting also carries
#113's A/A0 default confirmation and moves BENCHMARK's "now", rule 33);
#81 planned (`763762a2`); `htp_moe` head `763762a2`. Queue for the
CPU → HVX track (⑨) confirmed as **#85 → #82 → #81**: #82's RoPE / q-k
norm are #81's inputs and both slot into #85's op table; #84 (host E2E
harness) planned in parallel. **#99 downgraded p1 → p2** (⑳): its
verdict question is answered (plan 99's diagnosis + #100's tag-validated
cells, rule 28); what is left is a `test/`-only fix so the rule-30/32
anchor cell stops reading `checksum_ok=n`. **#114 stays p2**, conditional
on #117's gate (rule 31; nothing to decide before the sitting). **One
user decision surfaced by plans 81 §3.6 / 82:** the accuracy column for
the ⑨ wiring handoff — see ⑨.

**Cycle 15 (seen 2026-09-23): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC). USER DECISION OF RECORD (2026-09-23, "PR #4327
최신 거도 다 반영해줘", contract §5 Q16): the full sync is ordered — every
commit of the PR head onto `htp_moe` as one PR, filed as #120 (p1,
`state:planned`, implementer on it this cycle). Merge-base `2ce38d65`, 40
commits, 44 files, +5667/−998. What `htp_moe` already carries from that
range, for checking #120's PR body against:**

| upstream | on `htp_moe` as | via |
|---|---|---|
| `fb0f02b9` size-class ION staging | `e75011e8` | PR #103 (#88, merged `ad714de7`), `docs/htp_attention` hunk dropped |
| `04a2fcc4` `NNTR_HTP_POLL_US` knob | `308e1fde` | PR #103 |
| `b0a384d6` 5 ms poll default | `d84b7d69` | PR #103, docs hunks dropped |
| `c64be9dd` epilogue slices under `swiglu` | **not carried**; overlaps PR #108's GEMV-row `swiglu` / `rest` print (rule 24) — the merge keeps one print and re-checks the GEMV row's `rest` on the next profile | — |
| `3006d255` FC profile row | not carried on `htp_moe` (the `htp_hadamard` side tree had it, #95) | — |

Everything else arrives with the merge: `d2f0bf47` (dense FFN through the
MoE kernel — 4 % PPL on the author's measure, must stay **off** by
default under gate (c)), `7f81560b` / `e699e1bd` (conv block as one call,
two IDL entries → skel + app rebuild, new md5s), `5731b6e5` (down matmul
one block behind, inside `hexkl_mm_u8i4_moe.c`'s HMX loop; off the GEMV
path at M=1, unmeasured there), `1f538c20` (qkv fusion, ARM side — #81 /
#82's input), `32b46e32` (`NNTR_PPL`, the accuracy column ⑨ wants),
`aa371dd2` (pooled FC epilogue, prefill FC calls only). `git merge-tree`
names conflicts in `hexkl_mm_u8i4_moe.c`, `htp_compute_ops.cpp`,
`moe_layer_host_check.c`, `run_host_checks.sh`, `nntr_hvx_mm_u8i4.c` —
**#117's feed (PR #118) touches the first two and the last**, so the
ordering is a user call (needs-user): PR #118 first and #120 rebases over
it (the supervisor's recommendation: the feed is measured, the sync is
not), or #120 first and the feed re-applies on the merged tree. #120's
acceptance (issue body) carries the device A/B (decode ≥ −2 % of A,
prefill ≥ −5 %, text identical, every default-on upstream knob named).
On our side this cycle: #117 filled twice (`6412bfc9`, unit
`R3CY10WM83Y`), feed-default flip in progress on PR #118; #114 closed;
`htp_moe` head `6952cb6f`.

**Cycle 16 (seen 2026-09-27): PR head unchanged at `4ae1ebd7` (updated
2026-09-23 02:14 UTC; `gh api repos/Seunghui98/nntrainer/commits?sha=claude/htp-lfm2-moe-ffn`
lists nothing newer). Nothing to decide upstream. On our side: **#120
measured** (`120-upstream-4327-sync.md` @ `7ded3ce9`, `R3CY10WM83Y`,
2026-09-23 17:55–18:08, → §2 ㉔): the merge passes its gate (decode +4.7 /
+0.9 / −0.2 %, prefill **+12.4 / +10.2 / +6.2 %**, text = A 14/14,
`MoeLayerM1GemvMatchesHmx` bit-identical on B's skel), and its A — the
feed default with nothing set — is the new "now" (35.97 / 35.96 / 35.17,
BENCHMARK.md). **PR #121 is not merged**: after its rebase onto
`d79c0efe` it conflicts again with `htp_moe` @ `b4c96bdb` (PR #119, the
per-token entry). **User decision 2026-09-27 (contract §12): resolve as a
rebase onto `htp_moe`, rungs 0–1 only, no new sitting** — PR #119's
`NNTR_HTP_FORWARD=1` entry is off by default, so the measured B binary's
default path is unchanged by the rebase; the #120 sitting stands as the
PR's device evidence. The rebased PR states that rungs 2–3 were not run
(contract §4.1a; the workstation produces the skel/app md5s before any
later handoff). The "frozen at head `2ce38d65`" line above becomes
`4ae1ebd7` when PR #121 merges. #85 closed (PR #119 merged `b4c96bdb`);
`htp_moe` head `b4c96bdb`; PR #124 (Mac dev container, tooling) open.
**Queue (user, 2026-09-27; no device sitting possible for now, so only
issues that end in host-gated PRs): (1) PR #121 rebase (#120), (2) #82,
(3) #81, (4) #84, (5) #89.** #90 / #110 / #99 stay behind them.

**Cycle 16b (post-merge fold, seen 2026-09-27 ~12:00 UTC): PR head
unchanged at `4ae1ebd7` (updated 2026-09-23 02:14 UTC; the branch's last
ten commits end at `4ae1ebd7`). Nothing to decide upstream. On our side
the user merged the whole cycle-16 queue into `htp_moe` in order,
11:30–11:32 UTC: PR #124 (`a875c085`, Mac dev container, §4.1a), PR #121
(`b7d46b57`, #120: the upstream sync, merge commit `fd3bfba1` + 6
commits), PR #125 (`d0552b1e`, #82), PR #126 (`1afbf0aa`, #81), PR #127
(`ec589373`, #84), PR #128 (`6497f38a`, #89), PR #129 (`758aaacc`, guide).
#120, #82, #81, #84, #89 closed (completed). `htp_moe` head `758aaacc`.
The "synced to `4ae1ebd7`" line above is the record; #82's `hvx_conv_gate_f32`
reuse (case (a)) stands. Rungs 2–3 for these PRs: §3a.**

**Cycle 18 (seen 2026-09-28, the first sitting after the weekend's Mac
cycles): PR head unchanged at `4ae1ebd7` (updated 2026-09-23 02:14 UTC;
`gh pr view 4327` and the branch's last fifteen commits end there).
Nothing to decide upstream. On our side: `htp_moe` @ `39af2f39` (=
origin; cycle-16b fold `dbaf24b5` + plan 130), no filled handoff since
#120, no open PR into `htp_moe`, no `state:measured` issue. Consistency
check of the weekend's merges: #120 / #82 / #81 / #84 / #89 / #85 are
closed against PRs #121 / #125 / #126 / #127 / #128 / #119, and the tree
carries what plan 130 assumes (`hvx/hvx_m1_ops_f32`, `hvx_attn_m1_f32`,
`hvx_conv_gate_f32`, `m1_ops_det.h`, `attn_m1_det.h`, the `attn_m1_*` IDL
entries, `run_inproc_e2e.sh`, `htp_dump_eval.py`; `hexkl_graph.c`'s table
still MOE-only). Two things were stale and are fixed: the tracker #76 body
(cycle-12 text: "frozen at `2ce38d65`", provisional "now") is rewritten to
the merged state, and ten closed issues still carried a `state:*` label
(#117 #105 #102 #101 #100 #87 #85 `review`, #78 `in-progress`, #91 #114
`needs-plan`) — removed. Ruling recorded on #76: **#122 (docs/htp tech
docs, `state:in-progress`, user-driven sessions) is a docs track and does
not occupy the implementer's single `in-progress` slot**, so #130 may go
`in-progress` beside it. Local housekeeping left to the user / orchestrator
(not an agent output): worktrees of merged branches (`~/nntrainer-120`,
`-85`, `-guide`, `-moe` = `htp_moe_cycle` @ `d79c0efe`, `-A` detached)
and their `[gone]` local branches; `~/nntrainer-122` at `7d14d9fa` is three
commits behind `origin/htp/122-tech-docs` (`4c73bb01`). Rungs 2–3 md5s
for a post-#121 skel are still unrecorded (§3a); #130's gate produces
them. Queue: #130 planned p1, #99 / #90 planned p2, #110 parked — no new
issue.**

## 1. Rules (device disagreed with reasoning; do not re-derive)

Inherited from the PR's device work, with their sources:

1. **A `--profile` build is never the tok/s binary** — it inflates prefill
   by 83 % (doc 46 §43.4).
2. **Read `min`, not `avg`** — profile averages include the first
   (registration) call (doc 46 §48.7).
3. **Rebuild the skel whenever the IDL or DSP sources change**;
   `build_android.sh` never does. Symptom of a stale skel:
   `AEE_EBADPARM (0x8000040E)` (doc 46 §48.7).
4. **`build_android.sh --clean` drops `-Denable-htp` silently**; check
   `readelf -d libnntrainer.so` for `libsdkl.so` (mobile guide §4).
5. **Never push the link-time stub `libcdsprpc.so`** to the phone (mobile
   guide §5).
6. **`QS4CX_WH` has no CPU fallback**; a wrong layout gives plausible wrong
   text, not an error; `moe_htp_layers` must be empty (doc 46 §35, §48.7).
7. **Moving a matmul to the HTP does not pay when the round trip exceeds
   the matmul** — conv in_proj: −13 ms of an expected −65..−100 (doc 50
   §3.4); the same rule closed doc 45 §0.
8. **The DSP address space, not host RAM, bounds registration**: 3840 MiB
   arena + ≈ 182 MiB heap in one 4 GiB space; 256 MiB chunks avoid the
   alignment waste that made it look like a 3 GiB limit (doc 46 §39–§41).
9. **Decode drifts between sittings without a code cause** (20.8 → 16.5,
   doc 50 §3.4; ±5 % in `hvx_impl` #53): every sitting starts with the
   control binary, and results are read as A/B inside the sitting.
10. **Cross-session decode numbers from another unit are provisional**
    (two S25 Ultra units: 6–8.6 % apart on one binary, `hvx_impl` #23/#53).

Learned in this project (#77, unit `R3CY205ZMND`, 2026-09-21):

11. **AMENDED by rule 28 (#100, 2026-09-23): the "4–7×" is a ratio against
    a probe number no tag-validated per-call path reproduces.** The
    in-situ 16–18 GB/s is real; the 72–117 GB/s denominator is not a
    ceiling. Read this rule as its surviving half — *wall 2 is decided
    only by instrumentation inside the layer call, never by a probe* —
    and take the validated ceiling from `c_star` (26.3 GB/s), against
    which the in-situ list is **faster**, not 2.7× slower. Original text:
    **Isolated DMA probes overstate the in-situ rate by 4–7×.** The same
    engine on the same arena and skel: 72–80 GB/s contiguous, 108–117
    GB/s strided 2D in a probe; **16.2–18.0 GB/s** in the MoE layer call's
    `weight DMA` line of the same sitting (the PR author's 38 GB/s probe
    sat in between). Workers 1 → 4 are flat to negative, the bus vote is
    ≤ 4 % (noise). Wall 2 is therefore decided only by instrumentation
    *inside* the layer call (#87), never by a probe.
12. **A DSP-side bandwidth loop must defeat VTCM/L2 residency or it
    measures nothing**: the two-reader probe's `dsp_alone` read 3265 GB/s.
    Bound the result inside the test (≤ 150 GB/s or `INVALID`) (#90).
13. **Any unit; only a same-sitting A/B is a verdict** (user decision
    2026-09-22, contract §12; supersedes the earlier "unit `R3CY10WM83Y`
    is the anchor" form of this rule). Handoffs name no serial; the
    filled handoff records the serial that ran; every BENCHMARK row is
    unit-tagged; cross-sitting numbers are compared only as same-unit
    drift when the unit happens to be the same (rule 9). The second-unit
    anchor and the per-cell unit ratio (#91, ⑮) are withdrawn as goals.
    Two S25 Ultra units were 6–8.6 % apart in `hvx_impl`; that fact stays
    as the reason for the rule.
14. **Compiled artifacts are not byte-reproducible across build paths.**
    #77's skel, app and gtest md5s all differed from the table after a
    rebuild of identical sources under another path, while every
    non-compiled file and both model `.bin`s matched bit-for-bit. A
    handoff's md5 table is filled from the build that is pushed, or the
    device `md5sum` lines are recorded against the commit; a mismatch
    with that provenance intact is a note, not a void, in a control-only
    sitting — with a B variant it stays a void.
15. **`NNTR_HTP_PROFILE=3` inflates `[M0-PROF] ffn=` ~5×** (16–20 ms →
    88–90 ms per MoE layer): only the `[HTP-PROFILE]` `transport` /
    `host` / `dsp` columns are comparable across levels 2 and 3.
16. **The `generation(last 64)` line lives in the base report block or
    not at all**: `Lfm2MoeCausalLM` reports from `causal_lm.cpp:699`, not
    from `lfm2_causallm.cpp`'s block (#89).

Learned in this project (#94 first attempt / #97, unit `R3CY205ZMND`,
2026-09-22):

17. **Gate 2 must include an undefined-symbol check of the skel.**
    `-Wall -Werror` clean is not "links": a Hexagon shared object links
    with unresolved symbols by default, and the device loader then fails
    `dlopen` with `0x80000406` / `dlerror RX VA 0xFFF00000 outside ELF
    segment` — a message that reads like a segment-layout bug and is not
    one. #94's skel from `2a75f7d9` had seven `U hexkl_dma_trace_*`
    (`hexkl_dma_trace.c` from PR #93 never entered `test/htp/build.sh`
    `SRCS`; the host check compiled it on its own, so gate 1 passed).
    From now on gate 2 = `-Wall -Werror` **and** `hexagon-nm -u -D
    build/libnntr_hvx_skel.so` shows no `U` outside the runtime set
    (`HAP_*`, `compute_resource_*`, `qurt*`, libc, compiler-rt); #97 puts
    that check (or `-Wl,--no-undefined`) into `build.sh`. Second artifact
    found broken only on the device (rule 14 was the first): every new
    `.c` under `htp_backend/` is checked against `build.sh` at review.
    Distinguish the three loader errors: `0x8000040E` = stale skel vs IDL
    (rule 3), `htp Context is not registered` = app-side, `0x80000406` =
    the skel itself does not load.

Learned in this project (#94 sitting 2, unit `R3CY205ZMND`, 2026-09-22):

18. **The isolated probe rate drifts 3–19 % with the phone's temperature
    inside one sitting** (`DMA_PROBE` shape iii, 1 worker: 106.9 GB/s at
    29 °C, 88.8 at ~51 °C) while the tok/s cells of the same sitting did
    not collapse (CPU 47–53). A probe reading is quoted with its
    `thermal_zone0`; the gap analysis uses the cool reading as the
    achievable ceiling and the hot one as the run of record.
19. **WITHDRAWN by rule 28 (#100, 2026-09-23).** The "2.7× lost in the
    list itself" was `traced` / `DMA_PROBE iii`; against the
    tag-validated `c_star` of the same log the traced list reads
    **1.19×**, i.e. faster than the ceiling. Row h dissolves and there is
    no list-side loss to fix. What survives: the *paced* replay
    reproduces the in-situ 15.6–18.6 GB/s (the interleaving is real,
    rows f + g), and more workers make the replay slower. Original text:
    **The real chunk list is slow on its own** (provisional on #99): the
    traced 46-descriptor / 32-region M=1 list replayed unpaced, one
    worker, no compute, no competitor runs at 40.2 GB/s where a probe of
    the identical 8 KiB × 64 @ 56 KiB shape does 106.9 on the same unit
    and skel; paced to the in-situ schedule it reproduces the in-situ
    15.6–18.6 GB/s. Rule 11's 4–7× therefore splits into ≈ 2.7× in the
    list itself and ≈ 1.3–1.8× in the interleaving (rows f + g ≈ 20 % of
    `dsp_us`). More workers make the replay slower (40.2 → 39.1 → 36.5).
20. **NPU tok/s drifts −7..−16 % day to day on one unit while the DSP
    profile columns move ≤ 4 %** (#94 s2 vs #77, same unit and forward
    path): the tok/s cells carry host and thermal state, the
    `[HTP-PROFILE]` `dsp=` / `mm` columns carry the kernel. Cross-sitting
    kernel claims are read from the profile columns, tok/s only inside a
    sitting.
21. **Rule 14, refined by the cycle-5 decision on #94 Deviation 5:** a
    device set whose md5s differ from the staged table but whose
    provenance is intact (commit + build line + `md5sum` recorded, DSP
    sources shown identical by `git diff`) is **accepted as the sitting's
    provenance** when every variant of the sitting is the same binary set
    switched by an environment variable or config key (A and C of #94).
    "Void once a B variant is included" applies only when a variant is a
    *different binary* whose staged md5 the device does not match. Also
    from that sitting: `test/htp/build.sh` picks the newest directory
    under `$HEXKL_ROOT/lib` when `HEXKL_SDK_VER` is unset — a workstation
    with 6.6.0.0 installed silently links another HexKL; every handoff's
    build line carries `HEXKL_SDK_VER=6.4.0.1`.

Learned in this project (#88, unit `R3CY205ZMND`, 2026-09-22):

22. **Handoff artifacts must be reproducible from the commit on any
    workstation; the staging path is not a deliverable.** Twice now (#94
    s2's skel, #88's whole A/B set) the sitting ran on a workstation where
    `/local/mnt/workspace/htp_moe/<n>/` and the implementer's worktrees did
    not exist, and the user rebuilt from the named commits. A handoff
    therefore carries, per variant: the commit, the exact build line
    (`build_android.sh --htp` then `--htp --cache`, `test/htp/build.sh`
    with `HEXKL_SDK_VER=6.4.0.1`, env var names, no absolute worktree
    paths), and the md5s of the non-compiled inputs (model, tokenizer,
    prompt, `libc++_shared.so`, `libsdkl.so`), which must match anywhere.
    **Rule 21 extended to binary B variants:** a sitting whose variants are
    *different binaries* rebuilt from their named commits is accepted
    when (a) every results row carries the device md5 of the binary that
    differs between variants, (b) the source diff between the variant
    commits is exactly the PR under test (`git diff A B -- nntrainer
    Applications test`), and (c) shared DSP artifacts are one file for
    all variants. It stays void when a device md5 matches neither the
    staged table nor a recorded rebuild. #88 met (a)–(c) (device
    `libnntrainer.so` `5db71a2e…` A / `3cf06450…` B in every row; diff
    `08afbb10..f3e99176` = `htp_backend/{htp_backend.cpp,
    htp_compute_ops.cpp, htp_rpcmem.h}` only; one skel `2bd7311f…`).
23. **The FastRPC `transport` column drifts across sittings like tok/s;
    only `dsp=` is a cross-sitting column.** Identical sources
    (`2a75f7d9` = `08afbb10` outside `build.sh`), same unit, same day,
    ≈ 2 h apart: #94 s2 A level 2 M==1 host / dsp / transport 2062.5 /
    1414.0 / 648.6 µs, #88 A 1812.1 / 1410.8 / 401.3 — dsp −0.2 %,
    transport −38 %. Rule 20 extends to `host` and `transport`: a
    transport claim is an A/B inside one sitting (as #88's was), never a
    comparison against an earlier sitting's row. Corollary for budgets:
    the time **outside** the MoE call (TPS token minus 22 × host) moved
    with the poll/staging variant too — 13.5 ms (A) / 11.4 (C) / 8.4 (B)
    at G=64 — so ≈ 5 of #88's 12.6 ms/token gain landed outside the
    call's own timer (candidates: the 5 ms poll keeps the calling core
    and its cluster clock up between calls; cache maintenance around the
    call; not decided).
24. **In an `m1_gemv=calls/calls` profile row, `swiglu` is Σ lane-time
    over the worker pool (6 lanes on v79), not a stage;** `swiglu / mm`
    is lane utilisation (#94 C: 5601.2 / 974.7 = 5.75). Since PR #108 the
    row's `rest` leaves it out and the `weight DMA: n/a` line prints the
    ratio; before that PR a GEMV row's `rest` is meaningless (#102).

Learned in this project (#95, side tree `htp_hadamard`, unit
`R3CY205ZMND`, 2026-09-22):

24. **v79 HVX IEEE `sf` arithmetic keeps subnormals. It does not flush
    them to zero.** `HvxFwht.MatchesScalarBitExact` fed a row of ±1e-39 /
    ±1e-40 (all subnormal). The DSP (`Q6_Vsf_vadd/vsub_VsfVsf`, then
    `Q6_Vsf_vmpy_VsfVsf` by 1/16, `-mhvx-ieee-fp`) returned
    `0x1.7f4b4p-127` at index 1. That is exactly 8·(1e-39 + 1e-40), the
    unflushed IEEE result, and itself subnormal. The scalar spec
    (`fwht_det_ftz` on every input and result) returned `+0`. Only the two
    non-zero outputs of that row differed (`bad_gated = bad_subnormal_row
    = 2`). The spec's premise ("HVX sf flushes them") was host reasoning,
    and the device refutes it: this is not a flushed-zero *sign*. A `_det`
    scalar spec for HVX `sf` code is plain IEEE RNE with no FTZ, and the host
    side must not run under a flush either (`-ffast-math` /
    `crtfastmath` can set aarch64 FPCR.FZ). Fix carried by #110. The
    same test's overflow row (±3e38, `bad_overflow_row=127 of 256`) is
    ungated and not yet explained (inf/NaN encoding is the candidate).
25. **A synthetic single-shape gtest cannot give an accuracy verdict. It
    gates bit-exactness only.** `MoeLayerHadamardMatchesTwoCallReference`
    matched its two-call host reference bit for bit (`bad_elems=0 of
    409600`, `max_ulp=0`). It still failed its own SNR assertion: rotated
    17.51 dB vs unrotated 23.32 dB, −5.8 dB on its uniform synthetic input.
    The full model in the same sitting improved +8.0 dB median requant SNR
    over about 6.2 k real expert calls, and PPL dropped −12.7 %. The two
    disagree in sign. Accuracy verdicts come from the full-model columns
    (`NNTR_PPL`, `[L2-DIFF-MOE]` over a run, text vs A). A gtest prints
    its SNR as a field and asserts only the bit-identity to its scalar
    reference (gate (a)).

Learned in this project (#105, unit `R3CY205ZMND`, 2026-09-22):

26. **At m = 1 the HVX GEMV over the arena is bound by DDR read latency,
    not by `vrmpy` issue. Cutting compute there makes it slower.** The plan
    expected that dropping `gemm_rows4`'s three dead accumulators would
    save 50–100 µs. On the device the one-row loop (B1) raised the M==1
    `mm` from 973.0 to 1044.6 µs (+7.4 %) and cut decode by 4–8 %. Under
    the same loop, the L2-hot microbench cell ran 2.26× faster (70.77 →
    31.25 ns/tile). With 2 `vrmpy` per quarter-tile instead of 8, fewer
    loads are in flight to cover the DDR latency. The `l2fetch` lead wins
    it back monotonically (0 / 64 / 192 KB: 145.2 / 137.5 / 120.2 ns/tile
    arena) but still does not beat the four-row loop's 111.7 at 192 KB.
    A compute-side change to the M=1 GEMV is read on the `hot` cell, or
    after the feed is in VTCM. On the arena it shows the latency, not the
    compute.
27. **ION arena and cached heap read alike from HVX (within 11 %, no
    consistent sign), 21–27 GB/s.** The ION mapping is not why the M=1
    GEMV feed is slow. That is the plain direct-vector DDR band `hvx_impl`
    #59 measured on a sibling SoC (21–25 GB/s vs 37 through the ring). Only
    a DMA feed into VTCM (#100 → ㉒) gets past it. Also from this sitting,
    confirming rule 20: two level-2 profile passes in opposite order, from
    30 °C and 50–58 °C starts, agreed ≤ 0.4 % on every `mm`. A's `mm`
    973.0 reproduces #94 s2 C's 974.7 on the same DSP path, while decode
    tok/s differs by +47 % (PR #103's transport). The `mm` / `dsp` columns
    carry the kernel; E2E prefill tok/s carries the position in the
    block (A's own pairs −10..−24 %).

Learned in this project (#100, unit `R3CY205ZMND`, 2026-09-23):

28. **A `DMA_PROBE` number is not a ceiling; the tag-validated `c_star`
    is.** In one log, `c_star` (the per-call shape with its content
    checked, `checksum_ok=y`) reads **26.3 GB/s** = 0.36 × the same log's
    `DMA_PROBE shape=iii workers=1` (73.2). Against `c_star` the traced
    46-descriptor / 32-region M=1 list runs at **1.19 ×** — *faster* than
    the ceiling — and every other validated cell sits at 30–31.7 GB/s
    whatever the descriptor count (8 / 24 / 28 / 36 / 46), the chaining
    mode (`dmstart`, `link1`, chained) or `dst` (linear, strided). So the
    probe measures something the engine cannot deliver through a real
    per-call list, rules 11 and 19 are amended/withdrawn above, and row h
    of the wall-2 attribution dissolves. **Consequence for gates: an
    absolute GB/s target is only meaningful next to the same run's
    `c_star` and next to rule 30's anchor cell.**
29. **HVX streaming out of VTCM is nearly free beside a concurrent DMA
    into it**: `load=2` costs **0.1–0.3 %** against `load=0` on all three
    certified feed shapes (699.8 → 700.5, 695.5 → 697.6, 700.5 → 701.8
    µs/call). A VTCM feed's cost is therefore the DMA rate alone, and the
    "HVX and DMA fight over VTCM" worry is closed.
30. **One unit's DMA rate drifts ≈ 22 % between sittings on an untouched
    cell, and it is not thermal.** `DMA_REPLAY workers=1 load=0 pace=0`
    (identical code in both sittings) read 561.5 µs/call = 40.2 GB/s in
    #94 s2 and **719.6 µs = 31.4 GB/s** in #100 on the same unit; a
    cooled repeat at 31 800 m°C (vs 46 500) reproduces every #100 ratio
    within 1 %. One-worker probe rows are 15–30 % low while four-worker
    rows are within 6 %, so it bites **single-queue issue rate, not
    aggregate DDR bandwidth**. Cause not separated (session clock /
    governor vs a rebuilt skel). **Any handoff with an absolute GB/s gate
    carries that cell as its in-sitting anchor and reports the gate
    scaled by it.** **Resolved by #113 (rule 32): the drift is
    persistent, and 31–32 GB/s is the bound.**

Learned in this project (#113, unit `R3CY205ZMND`, 2026-09-23):

31. **An `l2fetch` lead harms a loop that is not latency-starved; it
    helps only the loop that is, and only up to one L2 budget.** Rule 26
    read the other way round, now measured on the full 2 × 5 matrix. The
    four-row loop (8 `vrmpy` of loads in flight per quarter-tile) with
    *any* lead reads **2.1–2.5× slower** on the arena (`ns_per_tile`
    133.5 → 282.3 / 316.4 / 323.1 / 328.4 at 192 / 384 / 768 / 1536 KB),
    the damage starts at the first non-zero lead rather than at a
    capacity boundary, and the `hot` cell is flat (71.7–73.5) — the
    signature of **interference** (the box competes with the loop's own
    demand loads for L2 and bus), not of eviction. The one-row loop (2
    `vrmpy`) improves to a minimum at 192 KB (154.3 → 129.1) and degrades
    past it (`inflight_kb=384` × 6 lanes ≈ 2.25 MB, the L2 budget). In
    the layer call: four-row + 192 KB `mm` 1925.5 vs 1000.2 (+92.5 %),
    one-row + 192 KB **937.0 (−6.3 %)**, one-row + 384 KB 1042.8. The
    whole matrix is on the board; its minimum is one-row + 192 KB and no
    (loop, lead) pair reaches 840. Corollary: a prefetch knob is never
    added to a loop without its own `hot`-vs-`arena` cell, and the lead
    is read as an ordinal (the up half of stage A gets ≈ two
    column-computes of it, not the nominal KB — the fourth outstanding
    box is #114's).
32. **The ≈ 22 % DMA drift of rule 30 is persistent, not a session
    effect: 31–32 GB/s is this unit's real single-queue DMA bound.** The
    untouched anchor cell (`DMA_REPLAY workers=1 load=0 pace=0`, source
    unchanged since #94) read **724.0 µs / 31.2 GB/s** cold (27 900 m°C)
    in #113 — #100's 719.6 / 31.4 to within 0.6 %, not #94's 561.5 /
    40.2. Whatever made #94 s2 read 40.2 has not returned in two
    sittings; #100's `f2` feed cell (31.6 GB/s beside HVX) is therefore
    read at face value against a 31–32 GB/s ceiling, not scaled to 37.
    Consequence: **㉒'s feed half closes as a bandwidth question** — the
    DMA cannot deliver 37 GB/s per call on this unit — and re-opens only
    as a *rate* lever: 31.6 through VTCM vs the direct HVX arena read's
    21–27 (23.5 under the one-row loop + 192 KB lead) is ≈ 1.33×.
33. **Decision rule (user, 2026-09-23, "1번으로 진행", #76-level): a
    sub-gate decode win may land as the default when it is consistent at
    all three G in one sitting, the text is byte-identical, and the
    prefill gate holds; the issue's gate still says whether the issue
    is done.** #113's D192 missed `mm` ≤ 840 (937.0, +11.5 %) and lands
    anyway (`HVX_GEMV_M1_ROWS1=1u`, `HVX_GEMV_PF_LEAD_KB=192u`, PR
    #115) on decode **+4.11 / +3.29 / +4.26 %** at G 64 / 512 / 1024,
    text = A 6/6, M>1 `dsp` within 0.08 %, prefill −0.77 / +0.05 /
    +3.73 %. What does *not* qualify: a win at one G only, a win inside
    the sitting's own A spread (rule 27), or one bought with a text
    change. The default flip is its own change with its own acceptance
    (issue #113's cycle-12 comment): an unset run prints the D192 word,
    host checks green, the old cell stays reachable by env for the
    same-sitting A/A0. **Landed: PR #115 merged as `c45d4433` (cycle
    13)** — banner `[HTP] moe m1 gemv: on (applied=0x103c1) lead=192KB
    rows1=1 source=default`, A0 = `NNTR_MOE_HTP_GEMV_LEAD_KB=0
    NNTR_MOE_HTP_GEMV_ROWS1=0` (`applied=0xc1`); the device confirmation
    (A vs A0, same sitting) rides #117's handoff, and BENCHMARK's "now"
    moves to that A, not before.

Learned in this project (#117, unit `R3CY10WM83Y`, 2026-09-23, two
sittings):

34. **The single-queue DMA bound is per unit — 37.3 GB/s on
    `R3CY10WM83Y`, 31.2 on `R3CY205ZMND` (+19 %) — while the HVX direct
    arena read is the same on both.** The untouched anchor cell read
    607.9 / 605.3 µs (37.1 / 37.3 GB/s) in two sittings on the second
    unit against 724.0 / 719.6 (31.2 / 31.4) on the first (rules 30, 32);
    `f2` 588.7 vs 695.5. But D192's `mm` — the latency-bound HVX read of
    the arena (rule 26) — read 922.7 / 931.9 here vs 937.0 there (< 1 %),
    and the `hot` cell 31.25 / 32.17 ns/tile vs 31.25. The unit difference
    sits in the DMA engine (or its clock), not in DDR or HVX; rule 32's
    "31–32 GB/s is the real bound" is `R3CY205ZMND`'s number, and three
    anchor values now exist (40.2 once, 31.2–31.4 twice, 37.1–37.3 twice).
    Consequences: (a) an absolute `mm` gate on a DMA-fed path is read
    next to the sitting's anchor — #117's 676.4 passes 760 as measured
    and would read ≈ 805–809 on a 31.2 GB/s unit; (b) a DMA-fed path's
    win is unit-dependent (931.9 / 676.4 = 1.38× here, ≈ 1.16× projected
    there, decode ≈ +10 % instead of +30 %) while a direct-read path's is
    not; (c) what transfers between units is the decode half of a gate
    (B ≥ A at all G, text = A), the `mm` half is scaled; (d) the goal's
    ceiling is per unit too: 22.02 MB × 22 = 484 MB/token of MoE weights
    is 13.0 ms at 37.3 and 15.5 at 31.2.
35. **Under the VTCM feed the M==1 transport falls 190 → 92 µs/call: the
    "+100 µs GEMV path cost" of ⑦ / #100's A/A0 belonged to the direct
    HVX arena read, not to the GEMV path.** #100 read GEMV 185.8 vs HMX
    84.6 µs (one sitting, equal staging classes) and the ledger called it
    path cost that "comes back only with ⑨". #117's B (GEMV + DMA feed)
    reads 82.0 / 92.3 in two sittings against A's 184.6 / 190.3 in the
    same sitting and binary (`qos_mode=2`, `staging:` identical) — the
    HMX path's level. 22 × ≈ 0.1 = 2.2 ms/token came back with the feed.
    Cause not separated (candidates: the HVX arena read leaves the
    ARM-side cache maintenance or the `dmlink` sync more to do at the
    call boundary; the shorter call stays inside the poll window).
    Corollary: `transport` is read beside `mm` on every DSP-path variant —
    a DSP-side read pattern moves the host column.

## 2. Verdicts (measured, closed)

| item | verdict | source |
|---|---|---|
| Tier-1 per-expert FastRPC (176 calls/token) | regression by construction (57 ms transport) | doc 41 §3 |
| Grouped gate_up at decode | built; transport 326 → 68 µs/mm | doc 34 §4E |
| In-place tile dequant, weight residency, cross-matmul prefetch, HVX worker pool, poll QoS + ION | built and verified | doc 34 §4A–F |
| GEMV instead of HMX at M=1 (prefill FC context) | rejected there (padding tax ~40 µs) — **re-opened for decode** where the 64-row tile is wall 1 (1.03 of 1.35 ms) | doc 34 §5.1 vs doc 48 §3 |
| Phase A W1/Q1/DQ1 micro-optimisations | one of four survived (D1: overlap weight DMA with activation quant) | doc 44 Phase A |
| Fused SwiGLU MoE path (A1) | passed, 32/32 `max_abs_err=0` | doc 44 |
| conv in_proj on HTP | runs, text identical, −13 ms prefill, not a decode lever | doc 50 §3.4 |
| Whole-model residency (doc 45) | plan exists, prefill-ordered; we take its Phase E (one call per token) first, M=1 shape | contract §3.1 |
| ① ARM remainder / per-type decode cost (ms/token, NPU vs CPU, unit `R3CY205ZMND`) | `lfm2_moe` **42.1 vs 22.3**, `fully_connected` **28.2 vs 10.3**, `output_of_causallm` 2.85 vs 3.40, `mha_core` 2.30 vs 2.81, all others ≤ 0.2; MoE + dense FC = 92 % of the NPU token, 83 % of the CPU's. The MoE FFN on the NPU costs ~2× the CPU path; the dense FC routed through the HTP 2.7×. lm_head is the one place the NPU already wins, so ⑧ (lm_head twin) is not first. **Correction (cycle 3, guide-writer's reading of the config):** the #78 `q40-qs4cx-wh` `nntr_config.json` routes **no FC to the HTP** — `lfm2_causallm.cpp:337–348` defaults `attn_proj_engine`, `conv_in_proj_engine`, `conv_out_proj_engine`, `dense_ffn_engine` to `cpu` and #78 set only `moe_engine`. So the NPU run's `fully_connected` 28.2 vs 10.3 ms/token is **not** an HTP round trip; the cause is open (candidates: doc 48 §2 ③ FC thread over-splitting when the 8 CPU threads contend with the FastRPC poll thread, or the CPU clock dropping while the DSP runs). Decidable with one extra profile cell: the NPU model at `NNTR_NUM_THREADS=4` (FC row moves → threading), or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off vs on (the `q40-qs4cx-wh` model has no CPU MoE fallback, rule 6, so it cannot be the control). Open item ⑰ | #77 A (profile); correction from the guide (PR #92) |
| ② Transport floor (wall 3) | **marshalling**: 527.7 µs/call at level 3 vs 587.7 at level 2 (−10 %, inside ±15 %; wake/clock would have been ≤ 300), `qos_mode` 2 both. **Refined by the #88 plan (host inventory, not yet measured):** cache maintenance on the over-sized staging pair (decode's 8 KiB rode a 4 MiB cached ION pair after prompt 512; 36–59 µs/MB, doc 50 §3.6) plus the 100 µs poll-window fallback to the interrupt path; the per-call arguments proper are 464 non-ION bytes (48 B primitive block + 5 sequences) and cannot carry 500 µs. Fix taken first = upstream `fb0f02b9` + `04a2fcc4` + `b0a384d6` (PR #103); prebind is conditional on that measurement (plan 88 §3.3 / §4 step 5); dspqueue / resident worker deferred to one paragraph (#83 §4). **Confirmed and closed by #88's sitting: staging 245.8 µs + poll 67.6 µs of A's 401.3; see ⑦ below** | #77 B; plan 88 §2.1; #88 |
| ③ Arena DMA probe (wall 2) | **shape** is the only axis: strided 2D 108–117 GB/s vs contiguous 72–80; workers flat-to-negative; vote ≤ 4 %. **But isolated rates are 4–7× the in-situ 16–18 GB/s**, so the wall is the layer call's use of the ring, not the engine (rule 11) — ⑥ rewritten, #87 | #77 C |
| ④ Two-reader DDR | **inconclusive on the DSP side** (probe defect, rule 12, #90); CPU side clean: 67.9 → 39.7 GB/s under DSP contention (−41 %). Q11 stays open | #77 ④ |
| Control on `R3CY205ZMND` (12 cells, prompt 512) | NPU decode 18.2–21.3 tok/s, CPU 46.4–54.1, NPU prefill 403–541: the NPU is **2.7× short** of 50 at G=512; the CPU clears 50 at G=64/512 and not at G=1024. Provisional for our unit until #91 | #77 A |
| **⑯ M=1 HVX GEMV (PR #86, `NNTR_MOE_HTP_M1_GEMV=1`) — variant C of #94 sitting 2, same binary and sitting as A** | **Goal progress, accuracy gate passed.** Decode +6.2 % (G=64, 18.83 vs 17.72), +7.7 % (G=512, 18.33 vs 17.03), +1.7 % (G=1024, no change); text byte-identical to A at every G and run; gtest `MoeLayerM1GemvMatchesHmx` bit-identical (0 bad of 2048 / 8192). Profile M==1: dsp **1414 → 1044 µs/call (−26 %)**, `DMA ring:` 46 → 2 descriptors, wait 269 → 4.3 µs, gather 146.5 → 0 — the DMA wait is gone from the path — **but `mm` 783 → 975 µs (+24 %)**: the GEMV reads the arena directly at ≈ 22.6 GB/s effective (22 MB in 975 µs), slower than the HMX loop's staged DMA + compute. Transport 649 → 748 µs/call (+100 µs, unexplained; `moe_set_opts` is once per session, so it is per-call cost of the switched path — #88 reads it). Prefill untouched (M>1 dsp 0.13 % apart). **Next lever inside ⑯: the GEMV's weight feed** (prefetch the expert tiles into VTCM ahead of the `vrmpy` loop instead of the direct arena read) — it must not re-import row h's slow list, so it waits for #100 (open item ㉒). Provenance: device skel `d6568c8b…` from `08afbb10` on the user's workstation, accepted under rule 21 | #94 s2 §C, `dad0f476` |
| **⑦ Wall 3, transport (PR #103: upstream size-class ION staging `fb0f02b9` + 5 ms poll `04a2fcc4`/`b0a384d6` + `staging:` line) — #88, one sitting A/B/C, `R3CY205ZMND`** | **Closed; the largest single-issue gain so far.** M==1 transport **401.3 → 87.9 µs/call (−78.1 %)** at level 2 (82.3 at level 3), `qos_mode=2` in all four profiles, gate ≤ 0.1 ms met → plan 88 §3.3 prebind **not built**. C (B + `NNTR_HTP_POLL_US=100`) 155.5 → the 313.4 µs cut = **staging 245.8 (78 %) + poll 67.6 (22 %)**, the plan's order and size. `staging:` M==1 `act 65536 out 65536 ion=y rpc allocs=19 non-ION 6/464 B`, `allocs` flat at level 3 (no per-call allocation). Decode 18.72 / 16.83 / 16.46 → **24.50 / 23.80 / 23.52 tok/s (+30.9 / +41.4 / +42.9 %)**, C at G=64 21.82 (+16.6 %); text bit-identical A = B = C at every G, run 1 = run 2; prefill −0.9 % (six-cell means 486.4 → 481.9), M>1 transport 2494.6 → 1770.3 with M>1 `dsp=` −0.07 %; peak RSS +0.3 % (both size classes stay alive). Thermal: B ran hotter (48.8 → 56.6 °C) and still won every cell; A re-run after B 17.82 (inside A's own spread). Only ≈ 6.9 of the 12.6 ms/token gain is in the transport column; ≈ 5 ms landed outside the MoE call (rule 23). Provenance: A/B/skel rebuilt on the user's workstation from `08afbb10` / `f3e99176` — accepted under rule 22 | `88-moe-call-marshalling.md` @ `3f9fa38d` |
| **Per-token budget at `htp_moe` head (cycle 7 estimate, not a measurement: TPS token time minus 22 × the level-2 M==1 `host` of the same sitting's G=64 profile)** | #88 B, G=512: token **42.0 ms** = MoE dsp 22 × 1.383 = **30.4 (72 %)** + transport 22 × 0.088 = **1.9 (5 %)** + outside the MoE call **≈ 9.6 (23 %)** (8.4 at G=64: FC, conv, attention, norms, lm_head, sampling). Floors at 38 GB/s: MoE 484 MB/token (22 × 22.02 MB) → **12.7 ms**, the rest 246 MB (lm_head Q4_0 alone 147 MB) → **≈ 6.5 ms** — so the MoE dsp column is **2.4× its floor (17.7 ms of excess)**, the outside-the-call time **≈ 1.5× (≈ 3 ms of excess)**, transport 1.9. **Biggest lever = MoE dsp (walls 1–2).** Ladder (projections): + #101 (GEMV default, #94 C: dsp −370 µs/call) ≈ −8 ms → ≈ 34 ms (≈ 29 tok/s, if C's +100 µs transport does not return under the new staging); + #105's gate (`mm` 975 → 600) ≈ −8 ms → ≈ 26 ms (≈ 39 tok/s); MoE at its floor → ≈ 24 ms (≈ 41 tok/s). 50 tok/s = 20 ms also needs transport + outside-the-call ≤ ≈ 7 ms → ⑨ (#85) and ⑧. ⑰ in TPS terms is ≈ 2–3 ms, not the profile's 18 (the `--profile` build adds ≈ 20 ms/token to both paths, #77 76.2 vs 54.4 and 39.3 vs 19.0), so it is not the biggest lever and gets no issue of its own | #88 B / A / C profiles + TPS; #94 s2 A / C; #77 profile |
| **Plan 87 §3.2 attribution a–h (wall 2): in-situ B levels 2/3 + `MoeChunkReplay` + `DMA_PROBE`, one sitting** | **a refuted** (wait 10.2 / 19.0 % of dsp, depth 9–11), **b refuted** (pushed shapes = probe iii), **c refuted, provisional #99** (workers 1/2/4 → 40.2 / 39.1 / 36.5 GB/s), **d sensitivity, not cause** (DDR competitor +1620 % wait, HVX competitor 0), **e refuted, sign reversed** (expert 0 fastest), **f confirmed 9.7 / 10.3 %** (the single blocked `site=act` wait = gather), **g real 11.0 / 7.1 %, unexplained** (`fresh` and `gap_us` controls both null), **h new, ≈ 60–70 % of the gap, provisional #99** (rule 19). **Wall-2 fix = row h → #100**; f (hide the first expert's 3.5 MiB behind gather / the previous layer) and g (cold-call penalty) are the remaining ~20 % and are filed after h. Plan 87 §0 expected f + g large: they are real but small; h was not foreseen | #94 s2 attribution table |
| **⑱ Hadamard rotation on the MoE down_proj input (`QS4CX_WH_HAD`, FWHT-256 before the u8 requant), #95. One sitting A/B/C/D on the side tree `htp_hadamard` (upstream `3006d255` + tooling: HMX block loop only, no M=1 GEMV, no DMA trace, 5 ms poll), `R3CY205ZMND`. Off-tree, so the numbers are not in BENCHMARK.md's `htp_moe` rows** | **Carry it over. Accuracy win, cost within noise.** **PPL (`NNTR_PPL`, 511 prompt tokens):** D CPU `q40` **109.759**, A **115.095**, B **115.095** (= A to the last digit), C **100.497**. C is −12.7 % vs A and −8.4 % vs the CPU, the first NPU variant below the CPU. **Requant SNR** (`[L2-DIFF-MOE]`, min / p10 / median dB, about 6.2 k calls): B 25.01 / 31.27 / 34.68 → C **39.68 / 41.67 / 42.67** (+14.7 / +10.4 / +8.0). `had=0` on all 6246 B lines, `had=1` on all 6274 C lines. **Gates:** B text byte-identical to A at G 64 / 512 / 1024 (after stripping the `[HTP-MOE] opts` line). C diverges from A at generated word 52, and both NPU variants diverge from D at word 43 at every G (a property of the weights). **Cost** (`[HTP-PROFILE]` level 2, G=64, mean µs/call): `requant` M>1 1097.3 / 1121.2 / **1567.4** (A / B / C, +446 µs), M==1 41.9 / 42.0 / **46.6**. Whole-call `dsp=`: M>1 15915.6 / 16061.7 / **16491.0** (C vs B **+2.7 %**), M==1 1333.5 / 1334.3 / **1338.5** (**+0.3 %**). **Decode tok/s:** A 25.08 / 24.40 / 23.83, B 24.82 / 24.12 / 23.75, C 24.49 / 23.65 / 23.53. A's own G=64 cell drifted −2.6 % from the sitting's start to its end (29 → 58 °C), so every B and C delta is inside that drift. C vs A's end re-run is +0.3 %. **Prefill tok/s cannot be read** (A 466.8 → B 433.8 → C 403.2, but A's re-run at the end read 404.7: order and heat, not code). The M>1 `dsp=` +2.7 % is the prefill reading, inside the −5 % gate. **Gtests:** `RejectsPartialBlock` and `MoeSetOptsEchoesKnownBits` OK. `MatchesScalarBitExact` failed only in the subnormal row (rule 24). `MoeLayerHadamardMatchesTwoCallReference` was bit-exact but failed its SNR assertion (rule 25). **Provenance:** everything was rebuilt on the user's workstation (Deviation 1). Device md5s match the rebuilt set. `git diff 63167235 6c89a912` is exactly the #95 commits. The model `a2829cd9…` equals the predicted hash and differs from `q40-qs4cx-wh` only in the 704 `down` tensors. The verdict is B vs C, one binary set switched by the model's dtype (rule 21). **Side reading on the accuracy gate:** A's PPL is +4.9 % over the CPU `q40`, and its text leaves the CPU's at word 43. That is the size of the "n/a (different weights)" that every NPU row in BENCHMARK.md carries, measured for the first time. Remainder (spec fix, gtest assertion, port incl. the M=1 GEMV requant site, `NNTR_PPL`): **#110** | `95-down-hadamard.md` @ `3a566761` (`origin/htp/95-down-hadamard`) |
| **㉒ compute half: M=1 GEMV one-row loop + `l2fetch` lead (PR #107 @ `a0ae8b29`), #105. One sitting A/B1/B2/B3, one app set, four skels, `R3CY205ZMND`, `NNTR_MOE_HTP_M1_GEMV=1` in every cell** | **Gate failed, accuracy passed. PR #107: hold, do not merge as-is.** M==1 `mm` (level 2, mean of two passes ≤ 0.4 % apart): A **973.0**, B1 1044.6 (+7.4 %), B2 1004.9 (+3.3 %, the branch default), B3 **930.9 (−4.3 %, inside ±5 %)**. Decode vs A 27.743 / 26.879 / 24.826: B1 −4.1 / −4.0 / −8.4 %, B2 −1.9 / −4.3 / −5.8 %, B3 **+2.6 / +2.2 / −1.7 %** (A's own G=1024 pair spreads 7 %). No variant reaches `mm` ≤ 600. **Accuracy:** `bit_identical yes`, `bad_elems` 0 under all four skels; text = A in 18/18 B cells. **Prefill:** M>1 `dsp` within 2.5 % of A in all eight profiles, `m1_gemv=0/23`. The E2E prefill column drifts with position (rule 27), so the gate is not failed. **Split (`MoeM1GemvFeedVsCompute`):** `arena − hot` is **≈ 74 % of `mm` under B3** (≈ 690 µs feed / ≈ 240 compute) and ≈ 37 % under A's four-row loop (≈ 590 compute). So the feed half is the lever. The one-row loop is what lets a VTCM feed hide the compute (240 ≪ 600 rather than 590 ≈ 600) (rule 26). **Merge verdict (cycle 9): hold.** As-is, the default lead is 64 KB = B2, a measured decode regression at all three G. Flipped to 192 KB (B3) it is neutral, not a gain, and the flip is a code change (no implementer this cycle). Once the weights stream from VTCM the `l2fetch` lead has nothing to prefetch, so the lead question dissolves if #100 certifies a feed shape. The PR's lasting content is the one-row loop, the x86 HVX-emulation check (`gemv_native_check.c`) and the microbench. The ㉒ feed issue takes them over by rebasing onto #107 and measuring feed + one-row together. If #100's feed verdict fails (no shape ≥ 37 GB/s beside HVX), the fallback is to land #107 with the lead at ≥ 192 KB after a sweep (384 KB + L2 budget). That fallback is filed only then. **Also read:** the GEMV path's M==1 transport is 182–184 µs here vs #88 B's 87.9 on the GEMV-off path. That is cross-sitting (rule 23), but it is the same +≈ 95 µs #94 C showed pre-#103 (649 → 748). #108's sitting reads it as an A/B (⑦ note). Provenance: rebuilt set accepted under rule 22 (BENCHMARK artifact row) | `105-m1-gemv-compute.md` @ `da41340b` (`origin/htp/105-m1-gemv-compute`) |
| **⑥ Wall 2, row h + the M=1 GEMV feed shape (PR #112, test-only; #100). One sitting, one app set, one skel, plus a cooled repeat of the gtest** | **Row h dissolves.** `c_star` = **26.3 GB/s** = 0.36 × the same log's `DMA_PROBE shape=iii w=1` (73.2), `checksum_ok=y`; plan 100 §3.3's H0 fires alone (H1 0.41, list row 0.99). `traced / c_star` = **1.19** — the real 46-descriptor list is *faster* than the tag-validated ceiling, so **there is no list-side loss to fix** and rules 11 / 19 are amended (rule 28). **Feed shape: FAIL as measured, not closed.** Best `f2` (8 desc, `dst=strided`, depth 2): `load=0 / c_star` = 1.21 **pass**, `load=2` = **31.6 GB/s** vs the 37.0 gate **fail**; `f1` / `f3` 31.4. **HVX-from-VTCM is not the cause** (0.1–0.3 %, rule 29): the engine itself tops out near 31 GB/s on every validated cell. **Confound:** the sitting's whole DMA is ≈ 22 % low on an untouched cell (rule 30), and scaled by it `f2` would clear 37.0 — hence "do not close ㉒'s feed half on this sitting alone" (the user's own read). Hygiene: 16/16 `DMA_PROBE` and **16/16 `DMA_REPLAY_X` `checksum_ok=y` with `tag=N/N`** (including the five allowed to fail), 11 old `DMA_REPLAY` failures all at `:633` (#99), nothing at `:715` / `:692`, `plan_shape_ok=y`, no FARF/AEE. Cooled repeat at 31 800 m°C reproduces every ratio within 1 % | `100-dma-chunk-list.md` @ `4bf1fdac`, `R3CY205ZMND`, 2026-09-23 |
| **#101 ride-along in the #100 sitting: the M=1 GEMV default as an in-sitting A/A0** | **A (default on) = 27.887 vs A0 (`NNTR_MOE_HTP_M1_GEMV=0`) = 24.578 tok/s at G=64: +13.5 %** (the handoff expected ≈ 6 %), text identical once the `[HTP]` banner line is excluded, prefill A not below −5 % of A0 on the warm runs. So PR #108's default is a measured win, not an inherited one. Level-2 M==1: A `blocks=0 m1_gemv=1408/1408`, host 1227.2 / dsp **1041.5** / transport **185.8** / `mm` 980.9, ring `desc=2/call`, `rest` ≤ 10.1 = **0.8 % of host** (㉑'s print fix confirmed on silicon, and the `weight DMA: n/a (… 5.76 lanes busy over mm)` line is there); A0 `blocks=5632`, host 1469.2 / dsp **1384.6** / transport **84.6** / `mm` 783.1, ring = the 46-descriptor list at 20.4–29.1 GB/s. **Transport GEMV / HMX = 185.8 / 84.6 (Δ +101.2, 2.20 ×) with equal `act`/`out` classes (65536 B)** — ⑦'s open note answered in-sitting: the +100 µs survives the new staging, and it buys −343 µs of `dsp`. `qos_mode=2`, M>1 `m1_gemv=0/23` on both | same |
| **㉒ compute half, closed: the M=1 GEMV (loop × `l2fetch` lead) matrix and the lead sweep, PR #115 @ `5be9c4ba` (absorbs PR #107), #113. One sitting, one app set, one skel, five profile cells, A vs D192 E2E mirrored at all three G, `R3CY205ZMND`** | **Gate `mm` ≤ 840 FAILED on every cell; accuracy and prefill passed; D192 lands as the default by user decision (rule 33).** M==1 `mm` (level 2, G=64): A (four-row, 0) **1000.2**; B (four-row, 192) 1925.5 (+92.5 %); C (four-row, 384) 2126.9 (+112.6 %); D (one-row, 384) 1042.8 (+4.3 %); **D192 (one-row, 192) 937.0 (−6.3 %)** — reproduces #105 B3's 930.9 within 0.7 %. The matrix (`ns_per_tile`, arena, lead 0 / 192 / 384 / 768 / 1536): four-row 133.5 / 282.3 / 316.4 / 323.1 / 328.4, one-row 154.3 / **129.1** / 139.4 / 147.3 / 155.2; `hot` four-row flat 71.7–73.5, one-row 42.7 / **31.3** / 31.7 / 34.0 / 32.6 (rule 31). **The hypothesis of #113 is inverted:** the cheapest unmeasured cell (four-row + lead) is a 2× regression, the lead helps only the one-row loop and only to 192 KB, and the whole 2 × 5 board's minimum is D192; nothing left unmeasured could reach 840. **E2E, A vs D192:** decode 27.444 / 26.829 / 25.930 → **28.572 / 27.711 / 27.035 (+4.11 / +3.29 / +4.26 %)**, text byte-identical 6/6, prefill −0.77 / +0.05 / +3.73 %, M>1 `dsp` within 0.08 % (`m1_gemv=0/23`). **Accuracy (a):** `bit_identical yes`, `bad_elems` 0 of 20480 / 81920 over all ten pairs. **Ride-along:** 724.0 µs / 31.2 GB/s = #100 (rule 32) → ㉒'s feed half closes as a bandwidth question. **What lands:** the per-call `(lead, rows1)` bits of `moe_set_opts` with the ARM-side echo check, `gemv_native_check.c` (x86 HVX emulation), the swept `MoeM1GemvFeedVsCompute`, and — by the user's decision, not by the gate — `HVX_GEMV_M1_ROWS1=1u` + `HVX_GEMV_PF_LEAD_KB=192u` as the build and app defaults (**flipped and merged: PR #115 `c45d4433`, cycle 13; #113 closed**, its A/A0 device confirmation rides #117's sitting; handoff head `37fffcb4`). **#105 closes** (its ≤ 600 target is superseded by rules 26 / 31: `mm` at 937 reads the arena at 23.5 GB/s, ≈ 1.15× the direct-read band's top of 27, so there is no compute-side headroom left under a direct feed). **#114 drops to p2**: the up-half lead is the only untested lead form and can only help the one-row loop by a few % — moot if the VTCM feed lands. Provenance: rebuilt set accepted under rules 21 / 22 (BENCHMARK artifact row) | `113-m1-gemv-lead-matrix.md` @ `d03c8929` (`origin/htp/113-m1-gemv-lead-matrix`), 2026-09-23 |
| **㉒ feed half, measured: the M=1 GEMV fed from VTCM by DMA (`f2` shape, one expert ahead) under the one-row loop, PR #118 @ `3c4912fa`, #117. Two sittings (warm, then cooled) on `R3CY10WM83Y`, one app set, one skel, A (D192 arena) / A0 (pre-#115) / B (feed) mirrored at all three G, `NNTR_MOE_HTP_M1_GEMV=1` explicit in every cell** | **Gate passed twice; accuracy passed; prefill passed on the tie-breaker; the feed becomes the default (#117 → in progress, PR #118), by the gate this time, not by rule 33.** M==1 `mm` (level 2, G=64): A **922.7 / 931.9**, **B 676.6 / 676.4 (−26.7 / −27.4 %, gate ≤ 760)**, A0 989.7 / 997.4; B `dsp` 711 (−28 %), **transport 82.0 / 92.3 vs A's 184.6 / 190.3 (rule 35)**, host 793 / 804 vs 1170 / 1185. In situ the ring reads 22.02 MB at **32.6 GB/s** (`engine 32.3..34.4`, busy 640–682 of 712 µs — the DMA never idles, the ≈ 240 µs compute is hidden, rule 29 confirmed in the layer call); `first expert ready at 122 µs` = the first 3.5 MB slab un-hidden per call. The `vtcm` microbench cell runs at the engine rate (579 µs = 22.02 MB / 37.7 GB/s, tail −14.5 µs — plan 117 §3.2's upgrade not triggered). **E2E, B vs A:** sitting 1 35.66 / 35.93 / 31.47 vs 28.48 / 26.47 / 25.55 (+25.2 / +35.7 / +23.2 %), sitting 2 **37.38 / 36.49 / 35.01 vs 28.22 / 27.89 / 27.23 (+32.5 / +30.9 / +28.6 %)**; text byte-identical 36/36 (per-G hashes equal across both sittings). **Accuracy (a):** `bit_identical yes`, `bad_elems` 0 of 40960 / 163840 over the ten pairs × {arena, feed}, both sittings. **Prefill:** M>1 `dsp` +0.04 / −0.04 % (`feed=0/23`); tok/s B vs A −0.6 / −3.3 / −4.1 % (s1) and −3.3 / +4.9 / −8.5 % (s2), −2.5 % over all G in both; the s2 G=1024 cell is A's own pair (528.9 first after the cool-down vs 401.9 = A0's run 2), rule 27's spread → pass on the M>1 `dsp` tie-breaker. **Anchor:** 37.1 / 37.3 GB/s — a third value, on a second unit (rule 34); read honestly, the `mm` gate holds as measured on this unit and would miss by ≈ 6 % on `R3CY205ZMND` (≈ 805–809), while the decode half (+29–32 % here, ≈ +10 % projected there) holds on both. **A vs A0:** +3.56 / +2.41 / +3.93 and +2.41 / +4.16 / +3.91 %, text identical — #113's D192 default confirmed on a second unit; BENCHMARK's "now" moves to this A (28.22 / 27.89 / 27.23). **What lands (flip, PR #118):** `HVX_GEMV_M1_FEED` → `1u`, an unset run prints `feed=vtcm source=default`, `NNTR_MOE_HTP_GEMV_FEED=0` is the opt-out (= this A, the next sitting's A0); the host scoreboard, the `vtcm` microbench cell and the ×feed bit-identity gtest stay. **#114 closes** (the lead is forced off under the feed and the VTCM cell is at the engine rate: no lead form has anything left to prefetch). Residual: the in-situ engine 32.3–34.4 vs the anchor's 37.3 ≈ 86 µs/call ≈ 1.9 ms/token → ㉓. Provenance: the staged set itself ran (device md5 = table), the first sitting without rule 22's rebuild | `117-m1-gemv-vtcm-feed.md` @ `6412bfc9` (`origin/htp/117-m1-gemv-vtcm-feed`), 2026-09-23 |
| **㉔ Upstream PR #4327 sync (merge of `4ae1ebd7` onto `htp_moe` @ `d79c0efe`, code `0e98036d`, PR #121), #120. One sitting A/B + A0 on `R3CY10WM83Y`, two app sets and two skels (IDL differs), no CPU cell** | **Gate pass; decode unmoved, prefill up.** Decode A 35.97 / 35.96 / 35.17 → B **37.68 / 36.29 / 35.10 (+4.74 / +0.92 / −0.19 %)**; the +4.7 % at G=64 is A's run 1 (35.42, first E2E cell after the profiles) against its run 2 (36.53), G 512 / 1024 read ±1 %. M==1 `dsp` **711.7 vs 711.3**, `mm` 677.2 vs 676.6, transport 82.2 vs 87.7, ring `engine 32.3..34.4 GB/s` in both — the decode path is byte-for-byte A's with upstream's epilogue column renamed (`swiglu(hidden)`). **Prefill 487.6 / 459.1 / 448.8 → 548.2 / 506.0 / 476.7 (+12.4 / +10.2 / +6.2 %)**, the first variant to raise prefill: M>1 `dsp` 16 986 → 14 808 µs (−12.8 %) from `requant` 1099 → 70, `dequant` 929 → 275, `mm` 10 222 → 9672, transport 1979 → 1082 — upstream's default-on M>1 changes (`5731b6e5` down matmul one block behind, `aa371dd2` pooled epilogue). Every upstream engine knob at its default: no `dense` / `conv` / `FC` row in B's profile, qkv fusion on the CPU side reads the unchanged model (weight order q, q_norm, k, k_norm, v). Text = A 14/14 (hashes `55f577d9` / `d7e89826` / `db08b4e1` per G). Ride-along on B's skel: `bit_identical yes`, `bad_elems_M1 0 of 40960`. **A vs A0 (`FEED=0`) at G=64: 35.97 vs 28.62 = +25.7 %**, `mm` 677.2 vs 934.3, text identical → PR #118's default confirmed as a sitting's control; A is the "now" (BENCHMARK.md, contract §1). Provenance: staged set ran as-is, device md5 = table, 0 mismatches. **Not merged (cycle 16):** PR #121 re-conflicts with `b4c96bdb`; user decision 2026-09-27 = rebase, rungs 0–1, no new sitting (the per-token entry is off by default) | `120-upstream-4327-sync.md` @ `7ded3ce9` (`origin/htp/120-upstream-4327-sync`) |
| **Per-token budget, cycle 16 (from #120 A, the feed default as a control: TPS token at G=512 minus 22 × the level-2 M==1 `host`)** | **A 27.81 ms** = 22 × 0.794 = **17.5 (63 %)** [dsp 22 × 0.712 = 15.7, of which `mm` 22 × 0.677 = 14.9; transport 22 × 0.082 = 1.8] + outside the MoE call **≈ 10.3 (37 %)**; **B (sync) 27.55** = 22 × 0.799 = 17.6 + ≈ 10.0. The cycle-15 arithmetic on #117 B (27.40 = 15.7 + 2.0 + 9.7) reproduces within 0.6 ms on a control cell. **−7.8 ms to 20 ms (50 tok/s)**: ≤ 2.7 on the MoE side on this unit (㉓, the engine gap), the rest is the ⑨ track (transport 1.8 + outside 10.3 = 12.1 against a ≈ 6.5 ms byte floor) | #120 A / B level-2 profiles |
| **Per-token budget, cycle 15 (from #117 sitting 2: TPS token at G=512 minus 22 × the level-2 M==1 `host`; the same arithmetic on A and on B, so the outside-the-call term is measured twice)** | **A (D192, the "now") 35.86 ms** = MoE dsp 22 × 0.995 = **21.9 (61 %)** [`mm` 20.5 at 23.6 GB/s] + transport 22 × 0.190 = **4.2 (12 %)** + outside **9.8 (27 %)**. **B (feed) 27.40 ms** = MoE dsp 22 × 0.712 = **15.7 (57 %)** [`mm` 14.9 at 32.6 GB/s through VTCM] + transport 22 × 0.092 = **2.0 (7 %)** [rule 35] + outside **9.7 (35 %)** [9.78 → 9.71: unchanged, as it should be]. **To 20 ms from 27.4: −7.4 ms**, and where it can come from: (1) MoE `dsp` floor = 484 MB / the unit's anchor = **13.0 ms at 37.3 GB/s, 15.5 at 31.2** → ≤ 2.7 ms on this unit (≈ 1.9 of it is ㉓'s engine gap, the rest the 35 µs `dsp − mm` tail), ≈ 0 on the other; (2) transport 2.0 + outside 9.7 = **11.7 ms vs a ≈ 6.5 ms byte floor** (lm_head 147 MB + FC / attention / conv ≈ 99 MB at 38 GB/s) → **⑨ must deliver ≈ −5 ms** (#85 → #82 → #81, ⑧ inside the op table; ⑰'s ≈ 2–3 ms is part of the 9.7); (3) sum **19.5–22 ms → 45–51 tok/s**: 50 is at the physical ceiling (contract §1: 48–52) and reachable on a 37 GB/s unit with ⑨ at its floor; on a 31 GB/s unit the NPU ceiling is ≈ 45 and only bytes per token (§3.3) moves it. The cycle-12 ladder's "≈ 33 tok/s" for the feed was low: it assumed `mm` ≈ 710 at 31.6 GB/s and no transport gain; the device gave 676 and −2.2 ms of transport | #117 sitting 2 profiles + E2E |
| **Per-token budget, cycle 12 (estimate: #100 A's 36.9 ms at G=512 minus 22 × D192's `mm` gain 63.2 µs; the measured D192 27.71 tok/s = 36.1 ms agrees within 2 %)** | Pending default D192, G=512: token **≈ 35.5 ms** = MoE dsp 22 × 0.996 = **21.9 (62 %)**, of which `mm` 22 × 0.937 = 20.6 reads 484 MB at **23.5 GB/s** (floor at this unit's DMA bound 31.2 GB/s: 15.5 ms; at 38: 12.7), + transport 22 × 0.180 = **4.0 (11 %)** + outside the call **≈ 9.6 (27 %)**, floor ≈ 6.5. Ladder: (1) **VTCM DMA feed under the one-row loop** (㉒ feed issue, planner's next): the certified `f2` shape at 31.6 GB/s beside HVX (rules 29, 32) bounds `mm` at ≈ 710 µs with the ≈ 240 µs compute hidden → ≈ **−5 ms → ≈ 30.5 ms (≈ 33 tok/s)**; gate `mm` ≤ 760. (2) **⑨ one call per token** (#85 → #82 → #81): transport 4.0 + outside 9.6 = **13.6 ms** against a ≈ 6.5 ms byte floor — the only path from ≈ 30 to 20 ms. (3) Bytes per token: 22.02 MB/call is int4 per-channel already; expert reuse across tokens needs a cache the DSP does not have (VTCM 8 MB ≈ 1.5 of the 4 experts) — §3.3 track, no issue. **The compute-side of the M=1 GEMV is exhausted** (rules 26, 31): under a direct feed `mm` cannot go below ≈ 815 (22 MB at 27 GB/s) and D192 is at 937 | #113 profiles + E2E; #100 A budget |
| **Per-token budget, cycle 11 (estimate: TPS token minus 22 × #100 A_L2's level-2 M==1 `host` 1227.2 µs)** | #100 A (GEMV default), G=512: token **36.9 ms** = MoE dsp 22 × 1.042 = **22.9 (62 %)**, of which `mm` 21.6 at ≈ 22.6 GB/s vs the 12.7 ms DDR floor, + transport 22 × 0.186 = **4.1 (11 %)** + outside the call **≈ 9.9 (27 %)**, floor ≈ 6.5. Ladder, re-ordered after #100: (1) **`l2fetch` lead** — the feed is ≈ 74 % of the one-row loop's `mm` and is DDR *latency* (rule 26), the lead moved it 145.2 → 120.2 ns/tile at 0 → 192 KB and is not saturated; #113 gates `mm` ≤ 840 → ≈ −3.1 ms → ≈ 33.8 ms (≈ 29.6 tok/s). (2) VTCM feed (㉒) now worth ≈ 1.2 × over the direct read (31.6 vs 25–27 GB/s), not the 37/22.6 = 1.6 × assumed in cycle 9 — demoted. (3) Then 50 tok/s = 20 ms needs outside-the-call ≤ ≈ 3.5 ms → ⑨ (#85), ⑧ | #100 A profile + TPS |
| **Per-token budget, cycle 9 (estimate: TPS token minus 22 × #105 A's level-2 M==1 `host` 1215.7 µs)** | #105 A (GEMV on, post-#103), G=512: token **37.2 ms** = MoE dsp 22 × 1.033 = **22.7 (61 %)**, of which `mm` 21.4 at ≈ 22.6 GB/s vs the 12.7 ms DDR floor, + transport 22 × 0.183 = **4.0 (11 %)** + outside the call **≈ 10.5 (28 %)** (9.3 at G=64). Ladder (projections): VTCM feed + one-row loop, `mm` 973 → ≈ 600 → ≈ −8.2 ms → ≈ 29 ms (≈ 34 tok/s). GEMV transport back to the GEMV-off 0.09 (if #108's A/B confirms it is path cost) → ≈ −2 ms. Then 50 tok/s = 20 ms needs outside-the-call ≤ ≈ 3.5 ms (MoE dsp ≈ 14.5 + transport ≈ 2) → ⑨ (#85), ⑧ | #105 A profile + TPS |
| Control, #94 sitting 2 (`R3CY205ZMND`, 13 A + 6 C cells, one binary set `2a75f7d9` + #98 skel) | NPU **17.72 / 17.03 / 17.30**, CPU **52.43 / 49.22 / 48.31**, NPU prefill 389–527: **the "now"** (BENCHMARK, contract §1). Distance to 50: 2.9× at G=512 (2.7× with C). Same-unit drift vs #77: CPU −9..+4 %, NPU −16..+5 % per cell with DSP profile columns ≤ 4 % apart (rule 20) | #94 s2 §A/§B |
| CPU control re-run on `R3CY205ZMND` (#94 first attempt, 6 cells, next day) | 53.0 / 53.5 · 51.1 / 51.2 · 49.7 / 48.4 tok/s vs #77's 54.1 / 52.9 · 52.7 / 52.0 · 46.4 / 48.5: **one unit drifts ≤ 7 % per cell day to day** (rule 9 quantified for the CPU path); G=1024 stays 46–50, i.e. the CPU does not reliably clear 50 there. No NPU cell (#97). Not the anchor; not a verdict on any lever | #94 `ec7ad296` |
| **Per-op residency without adjacency to the MoE op (cycle 18, #130's PR, host-gated: `run_inproc_e2e.sh`, both fixtures)** | **A stretch pays only when it absorbs a round trip.** `forward` stops at the first non-resident op; with RMSNORM / QK_NORM / ROPE / CONV1D_GATE / ATTN_M1 resident beside MOE every stretch is still bounded by an ARM op (FC, ADD, ROUTER_TOPK), so the count is 4 calls per conv/attention MoE layer, 3 per dense layer, 1 for the tail: **95/token on LFM2.5 vs 22 with `KINDS=MOE`** (11 vs 4 and 12 vs 4 on the host fixtures, measured). The wiring is proven end to end (bit-identical stretches vs the `_det` specs in `graph_host_check`, `INPROC E2E PASS`), the speed verdict is the device's (plan 130 §4 step 7); the call count falls only with ㉓ / #132 | #130's PR, plan 130 §0 |

## 3. Open items (candidates for issues; the supervisor promotes them)

| # | item | expected | depends on |
|---|---|---|---|
| ① | ~~Measurement A~~ **measured (#77) → §2.** Follow-on: the dense FC's 28.2 ms/token on the NPU path is the second-largest lever after MoE; **it is not a round-trip cost** (no FC is on the HTP in that config, §2 correction) — see ⑰ | — | ⑰ |
| ⑱ | **Measured by #95 on the side tree `htp_hadamard` → §2 (cycle 8): carry it over. PPL −12.7 % (100.5 vs A 115.1, CPU 109.8), requant SNR median +8.0 dB, cost +2.7 % / +0.3 % of the M>1 / M==1 `dsp=`, decode inside drift. #95 closed. Remainder filed as #110 (p2, `state:needs-plan`):** (a) `fwht_det.h` drops its FTZ (rule 24); (b) the synthetic SNR assertion becomes a printed field (rule 25); (c) port to `htp_moe` with the M=1 GEMV requant site (about `:509`), plus the HMX block loop (about `:1415`) and the tail (about `:370`); the HAD opts bit must not clear PR #108's default-on GEMV bit; (d) `NNTR_PPL` as a separate upstream cherry-pick (`32b46e32`, user decision at merge). Gate = an `htp_moe` handoff: B ≡ A (text + PPL), C PPL ≤ A, SNR median ≥ B + 5 dB, M==1 `dsp=` ≤ +2 %, prefill −5 %. p2 because it does not move decode tok/s; it becomes p1 if the user adopts PPL as the accuracy column for `QS4CX_WH*` models. History: **Accuracy, filed as #95 (p1):** Hadamard rotation on the MoE down_proj input — fold `Hᵀ·W_down` offline (new dtype `QS4CX_WH_HAD`, block 256, 1792 = 7 × 256, 1/16 both sides), FWHT-256 in IEEE `sf` add/sub on the DSP right before the existing u8 requant at all three requant sites of `hexkl_mm_u8i4_moe.c`. Targets the recorded gap (doc 43, 2026-09-09: 5 of 32 MoE calls at 67–80 dB SNR, each from exactly one of 1792 elements crossing a u8 level — boundary rounding, not row outliers; column-wise outliers never measured). **First accuracy item in this table**; it does not move tok/s by design and may show no effect — a "no effect" result closes it with the numbers. Gate = `NNTR_L2_DIFF` per-call SNR / `total_flips` on vs off on the 5 recorded calls + full-model handoff with prefill/decode TPS and text-identical-to-CPU; prefill gate applies (it touches the weight layout and the requant stage). Rules that bind: no qf32 (v75/v79), host scalar `fwht_rows_f32_ref` bit-identical to the HVX kernel, CPU `QS4CX` run stays the reference (no CPU kernel for `_HAD`, rule 6 applies to it too) | requant SNR on the 5 bad calls ↑; text-identical-to-CPU unchanged or better; TPS within noise | none; can run in parallel with the walls (touches quantizer + requant only) |
| ② | ~~Measurement B~~ **measured (#77) → marshalling → ⑦ / #88 → closed by #88's sitting 2026-09-22** | — | — |
| ③ | ~~Measurement C~~ **measured (#77) → shape only; in-situ gap is the wall → ⑥ / #87** | — | — |
| ④ | Two-reader DDR probe — **DSP side invalid in #77 (rule 12); refiled as #90** (stream > VTCM+L2 or use the DMA ring, bounded result). CPU side: −41 % under contention, so any split pays less than the sum | decides whether the CPU+NPU split (contract §3.2) can ever pay | #90, ride-along step in a later sitting |
| ⑤ | **Filed as #80.** M=1 MoE path on the existing HVX GEMV (wall 1). The kernel already exists: `hvx/hvx_gemm_u8i4_wh.c` (u8×i4 over WH tiles, int32 bit-identical to HMX, m ≤ 16) is used only by the prefill "tail" path in `hexkl_mm_u8i4_moe.c` (off by default, net −0.5 ms there). Decode needs a dispatch that sends all four experts through it at M=1 with no 64-row block, plus the weight feed (arena read vs DMA into VTCM) that ③ decides | MoE DSP 1.35 → ≈ 0.3 ms/call if DMA ≥ 30 GB/s | ③ |
| ⑥ | **Loose end read by #113's ride-along (cycle 12): 724.0 µs / 31.2 GB/s = #100, not #94 — rule 32; the feed half does not re-open at 37, it re-files at 31.6 vs 23.5 (㉒). Nothing left in ⑥.** **CLOSED as a descriptor-list question, 2026-09-23 by #100 (§2 row): row h dissolves.** Against the tag-validated `c_star` the traced 46-descriptor list runs at **1.19 ×**, i.e. faster than the ceiling; rules 11 and 19 are amended/withdrawn (rule 28) and there is no list-side loss to fix. The 16–18 GB/s in-situ number stays real, but its cause is the interleaving (rows f + g, ≈ 20 % of `dsp_us`) plus an engine that tops out near 31 GB/s on every validated per-call cell — **not a shape or a list**. The GEMV path bypasses the ring entirely (`desc=2/call`), so at M=1 nothing in ⑥ is on the decode critical path any more; rows f and g are **not** filed. What wall 2 leaves behind at M=1 is the read-rate half, which lives in ㉒ and #113. **One loose end, a ride-along not an issue:** re-read `DMA_REPLAY workers=1 load=0 pace=0` on the next sitting and see whether it returns to #94's 561.5 µs / 40.2 GB/s (rule 30); if it does, the `f2` feed cell would clear 37.0 and ㉒'s feed half re-opens. History: **Wall 2 — measured by #94 sitting 2 (§2 attribution): the 4–7× is ≈ 2.7× in the traced chunk list itself (row h, rule 19, provisional on #99) and ≈ 1.3–1.8× in the interleaving (f + g ≈ 20 % of `dsp_us`). Step 2 filed as #100 (row h: why the 46-descriptor / 32-region list replays at 40 GB/s where probe iii does 107; gate = replay ≥ 80 % of probe iii and in-situ `engine GB/s` ≥ 40; blocked by #99 until the replay's content check is explained). Rows f and g follow #100.** **Cycle 9: #100 re-scoped to plan 100 §0 (row h's cause + a DDR → VTCM feed shape certified ≥ 37 GB/s beside HVX for ㉒, same-run `c_star` denominator), raised to p0 after #105, no longer blocked by #99 (new cells set their dst mode explicitly).** History: rewritten by #77 C — not descriptor / engine / vote but "why does the MoE call see a quarter of the isolated rate". Step 1 = ~~#87~~ **landed on `htp_moe` as `b6ebc2b7` (PR #93, 2026-09-22; #87 closed)**: `hmx/hexkl_dma_trace.{c,h}` (static tables, union-of-intervals busy / depth / blocked-wait arithmetic, host-checked by `dma_trace_host_check`), trace hooks in `hexkl_mm_u8i4_moe.c` behind `hexkl_probe_on` (byte-identical output on vs off, 19 descriptors traced at the fixture shape, 46 at the LFM2 M=1 shape), IDL entries `dma_probe` / `moe_dma_trace_read` / `dma_replay` (`test/htp/nntr_hvx_dma_probe.c`), the header-only in-situ descriptor plan `test/htp/nntr_moe_dma_plan.h`, the second `weight DMA:` line and the `[HTP-DMA]` per-descriptor dump in `[HTP-PROFILE]` (`htp_compute_ops.cpp`, first `NNTR_HTP_DMA_TRACE` calls, default 3), and `TEST_F(HvxDmaProbe, MoeChunkReplay)` (workers 1/2/4 × HVX load × fresh/gap) in `unittest_hvx_dma_probe`. **The device sitting that fills the attribution table (plan 87 §1, hypotheses a–g) is #94 (sitting 2); its first attempt never reached the NPU because the trace's own `hexkl_dma_trace.c` was missing from the skel build (#97, rule 17).** Original scope for reference: instrument the ring use inside `hexkl_mm_u8i4_moe.c` (per-descriptor issue/complete pcycles, wait time in `hexkl_dma_ring_wait`, outstanding depth, actual chunk shapes at M=1 and M>1) + a device gtest that reproduces the in-situ pattern, gate = a table attributing the 4–7× to named causes. Step 2 = the fix that table names (separate issue). Interacts with #80/#86: the M=1 GEMV path reads the arena directly, so its A/B also tells what the ring costs | 16–18 → ≥ 40 GB/s in situ; 1.19 → ≤ 0.57 ms per call | #87 |
| ⑦ | **CLOSED 2026-09-22 by #88's sitting** (`88-moe-call-marshalling.md` @ `3f9fa38d`, `R3CY205ZMND`, one sitting A/B/C; §2 row): PR #103's size-class ION staging + 5 ms poll take the M==1 transport **401.3 → 87.9 µs/call (−78.1 %)**, past the ≤ 0.1 ms gate, so plan §3.3's prebind IDL pair is **not built**; the cut splits **staging 245.8 µs / poll 67.6 µs** (C = B + `NNTR_HTP_POLL_US=100` reads 155.5). Decode **+30.9 / +41.4 / +42.9 %** at G 64 / 512 / 1024, prefill −0.9 % (gate ≥ −5 %), M>1 transport 2494.6 → 1770.3 with `dsp=` unmoved, text bit-identical at all three G. Wall 3 is no longer the lever (1.9 ms/token left); the MoE call is 94 % DSP, so walls 1–2 own most of the remaining 2.1× to 50 tok/s. #88 and #83 (the transport-options document, consumed by plan 88) closed in cycle 7. **Left open, not an issue:** whether the GEMV path's +100 µs transport of #94 C (649 → 748) survives the new staging — #101's ride-along reads it as the same-sitting A (GEMV default) − A0 (`NNTR_MOE_HTP_M1_GEMV=0`) level-2 M==1 transport (`101-ride-along.md` R2/R4, rule 23). **Cycle 11: ANSWERED in-sitting by #100's A/A0 (§2 row). GEMV 185.8 vs HMX 84.6 µs/call, Δ +101.2 (2.20 ×), equal `act`/`out` classes (65536 B), one sitting, one binary — so the +100 µs is path cost and it survives PR #103's staging. It is not a defect to chase: the same A/B wins 343 µs of `dsp` and 242 of `host`. The 4.1 ms/token it costs comes back only with ⑨ (one call per token), not with a transport fix; no issue.** **Cycle 15: wrong — it came back with the VTCM feed (rule 35): B's M==1 transport 82.0 / 92.3 µs vs A's 184.6 / 190.3 in the same sitting, so the +100 µs was the direct HVX arena read's, and transport at the feed default is ≈ 2.0 ms/token, the HMX-path level of ⑦.** Earlier, cycle 9: #105's A reads M==1 transport 182–184 µs against #88 B's 87.9 (GEMV off), which is cross-sitting (rule 23) and consistent with it surviving. At ≈ 2 ms/token it is worth an A/B line; PR #108 merged (`8e121dbe`) without a sitting, so that line (`101-ride-along.md` R2/R4) rides #100's handoff. dspqueue / resident worker: one paragraph (plan 83 §4), revisited only with ⑨. History: resolved by #77 B to prebound handles + per-call buffer/marshalling cleanup on plain FastRPC — #88; #83 narrowed to the document that says what "prebound" concretely means | 0.53 → ≤ 0.1 ms per call — **met: 0.088** | — |
| ⑧ | lm_head blocked Q4_0 twin on device (doc 46 §46): confirm 25.7 → ≈ 3.4 ms | ARM remainder | ① |
| ⑨ | **WIRED (cycle 18, #130's PR, host-gated): `hexkl_graph.c`'s table = MOE \| RMSNORM \| QK_NORM \| ROPE \| CONV1D_GATE \| ATTN_M1 (wire v2: `n_kv` / `gqa` / `head_dim` / `eps_bits` in the op record, the kernels' shape rules in the validator as `AEE_ESCHEMENOTSUPPORTED`, ATTN_M1-needs-ROPE as `AEE_ENOTALLOWED`); parameters bound once through `graph_set_param` (gammas, conv weights, conv state seed, RoPE table); the four CPU layers call one hook at a decode row (`htp_decode_hook.h`), the KV cache is registered at `graph_init` and seeded from the CPU prefill's rows once, the conv state per layer lives on the DSP after the first routed token. Host gate on both fixtures (`INPROC E2E PASS`): hd8 with MOE,RMSNORM,CONV1D_GATE = **11 calls/token**, hd64 with all six = **12**, the switch-off runs untouched; `NNTR_HTP_FORWARD_KINDS=MOE` reproduces plan 85's B. On LFM2.5 the same arithmetic gives **95 calls/token with all six kinds on, 22 with `KINDS=MOE`** — a per-op residency without adjacency to the MoE op *raises* the call count (§2 verdict); the device cost is the ride-along's (plan 130 §4 step 7, under the accuracy rule below). The count falls below 22 only with ㉓ (#132). Measured on the host: CONV1D_GATE bit-identical to the CPU path, RMSNORM 129 dB, the attention kinds 39 dB at worst (fp16 KV rows on the CPU vs f32 on the DSP, through two activation quantizers on 128-wide rows), tokens 8/8; the gate's floor is 30 dB.** **USER DECISION 2026-09-28 (cycle 18): the accuracy column for the ⑨ handoffs is (a) + (b) + approval — keep text-identical-to-CPU as a column, add `NNTR_PPL` vs the CPU `q40` PPL (upstream `32b46e32` is on `htp_moe` since PR #121), and add a user approval step: the filled handoff pastes each variant's generated text, the user marks `text approved: y/n`, and only approved rows are folded (contract §1 accuracy gate, `hexagon-handoff` skill). (c) is not taken. Binds #130's device ride-along and every later ⑨ / ㉓ handoff.** **Cycle 16b (2026-09-27, post-merge fold): #82 (PR #125 `d0552b1e`) and #81 (PR #126 `1afbf0aa`) are on `htp_moe` — their kernels, `_det` specs, host bit checks and IDL entries exist, but `hexkl_graph.c`'s table still has MOE as its only non-NULL slot (`RMSNORM` / `CONV1D_GATE` / `QK_NORM` / `ROPE` / `ATTN_M1` NULL, `resident_ok = MOE`), so the per-token entry still stops at the first non-MoE op: 22 calls/token, 0 ms by construction. #84 (PR #127 `ec589373`) landed the host E2E harness (`-Dhtp-inproc`, `run_inproc_e2e.sh`, `NNTR_HTP_DUMP`), so a wiring change can now be gated on the whole tiny model through the per-token entry on the host, not only on `graph_host_check`. **What is still missing for the call count to fall (㉓, plan 85 §7):** (1) the wiring itself — register the five kinds, seed the KV cache after the CPU prefill, upload the RoPE table at `graph_init` — **filed as #130** (host-gated PR + device ride-along; the accuracy-column decision (a)/(b)/(c) below binds only its device handoff); (2) ㉓'s resident set beyond it: the M=1 FCs (conv in_proj / out_proj, q/k/v/o, the dense FFN of layers 0–1; Q4_0 on the NPU model), the router FC + top-k, the final norm + lm_head — not yet filed, a follow-up to #130 with its own weight-format decision. #89's `generation(last 64)` (PR #128) fills the contract column from the next handoff on.** **Cycle 17, #81 in review (PR #126, stacked on #125; rungs 0–1 on the Mac, `qaic` + `hexagon-clang -c` of every new DSP source, rungs 2–3 not run there): `nntrainer/tensor/attn_m1_det.h` (the spec: 64-step q·Kᵀ per position, `* scale`, exact max, `exp_det`, 32-lane tree sum, `recip_det`, sequential PV), `hvx/hvx_attn_m1_f32.{c,h}` (the cache object `Kt [layer][kv][64][max_seq]` + `V [layer][kv][max_seq][64]`, f32, memalign(128) heap, **48 MiB at max_seq 2048 = 26 % of the ≈ 182 MiB heap**, + 256 KiB scratch; one pool unit per kv head), host check `ATTN M1 BIT-IDENTICAL` at L = 1 / 63 / 64 / 65 / 512 / 1024 × pool 0 / 3 / 7 workers byte-equal, append-chain ≡ bulk, double reference ≤ 1.7e-6 of max|V|, four mutants caught; IDL +4 methods after `conv_gate_m1_f32` (`attn_m1_register / release / kv_append / forward`, result sequence named `y` — `out` is an IDL reserved word); `HvxAttnM1.*` in `unittest_hvx_attn`. Unwired: #85's `ATTN_M1` slot stays NULL, 0 ms/token; the wiring issue needs `attn_m1_register(6, 8, 4, 64, max_seq)` at graph init and one `attn_m1_kv_append` per layer after the CPU prefill (the seeded rows are the CPU's fp16-rounded k/v, the decode rows f32 — one more reason (a) below cannot hold). **Plan correction:** `recip_det(1.0f)` is `0x3F7FFFFF` (swiglu_det.h's fixed point), not 1.0f, so at L = 1 the output is `v · (1 − 2⁻²⁴)`, not `v`; the check asserts that instead. Ride-along for the next sitting: `unittest_hvx_attn --gtest_filter='HvxAttnM1.*'` on that sitting's skel (expect `bad=0` at all six lengths, `ATTN_M1_FIELD pos=511/1023 us=` the first read of plan 81 §0's 0.5 / 1.0 ms estimate; `0x8000040e` = stale skel). Open (supervisor's call): the fp16 cache upgrade once an `sf → hf` RNE spec is device-confirmed (halves the 37.7 MB/token read at pos 1536, and max_seq 4096 fits).** **Cycle 17: #82 in review (PR #125, rungs 0–1 on the Mac, 2–3 not run there): `nntrainer/tensor/m1_ops_det.h` (the spec), `hvx/hvx_m1_ops_f32.{c,h}` (RMSNorm / q-k norm, RoPE 64, conv1d + gate at M=1 through upstream's `hvx_conv_gate_f32`), host check `M1 OPS BIT-IDENTICAL` on the real HVX source over an intrinsic emulation (`test/htp/host/hvx_emu/`), IDL +3 test methods after `forward_debug`, `HvxM1Ops.*` in `unittest_hvx_softmax`. Unwired: #85's `RMSNORM` / `QK_NORM` / `ROPE` / `CONV1D_GATE` slots stay NULL, 0 ms/token. Ride-along for the next sitting: `unittest_hvx_softmax --gtest_filter='HvxM1Ops.*'` on that sitting's skel (expect `bad=0` per case, `0x8000040e` = stale skel). Open for the wiring issue (user decision, option (c) below): the spec now exists for a NEON twin; the RoPE table is the ARM's `cos_ps`/`sin_ps` rows uploaded at `graph_init` (≈ 0.5 MiB at max_seq 2048), and the ops' DSP memory once wired is ≈ 1.8 MiB (plan 82 §3.4).** **Cycle 16: #85 CLOSED (PR #119 merged `b4c96bdb`, `NNTR_HTP_FORWARD=1` off by default). Queue (user, 2026-09-27; no sitting possible, host-gated PRs only): PR #121 rebase → #82 → #81 → #84 → #89; the user decision on the wiring handoff's accuracy column (a / b / c below) is still open and does not block these.** **Skeleton landed as PR #119 (`htp/85-one-call-per-token`, cycle 14, `state:review`): IDL `graph_init / graph_release / forward / forward_debug` after `moe_set_opts`, `htp_backend/htp_graph_desc.h` (wire format, validator with one `AEE_*` code per finding and never `AEE_EBADPARM`, the LFM2 builder: 228 ops at LFM2.5, 22 MoE), `hmx/hexkl_graph.c` (kernel table with MOE as the only non-NULL slot, per-op pcycles, one FARF line per call), `graph_host_check` in `run_host_checks.sh`, one core virtual `ComputeOps::set_decode_graph_desc(words)`, and the `NNTR_HTP_FORWARD=1` switch, off by default. Resident = MOE, so still 22 calls/token: 0 ms by construction (plan 85 §0); no device sitting in the PR. Ride-along for the next sitting: one decode token with the switch on (`[HTP] graph: init n_ops=228 resident=MOE moe_ops=22`, text ≡ switch-off at G=64; `0x8000040E` = stale skel). See ㉓ for what the call count actually needs.** Cycle 14: order #85 (skeleton, in progress) → #82 (small ops) → #81 (m=1 attention); #84 in parallel. All three end in host-gated PRs; the E2E A/B belongs to the wiring issue (not yet filed). USER DECISION (plans 81 §3.6, 82 last bullet), binds that wiring handoff, not these PRs:** the CPU decodes attention in fp16 with its own exp and runs no `_det` for RMSNorm / RoPE, so once the per-token entry is wired the NPU text cannot be bit-identical to the CPU run and B ≡ A is at risk too. Options: (a) keep text B ≡ A and expect a fail, (b) adopt `NNTR_PPL` (upstream `32b46e32`, #110 (d)) as the accuracy column for the ⑨ handoffs — this also lifts #110 to p1 per ⑱, (c) give the CPU path the same `_det` (NEON twin, doc 45 §3.3), which moves the CPU control and touches prefill. The supervisor recommends (b) with the CPU `q40` PPL as the reference, since #95 already measured that the NPU text leaves the CPU's at word 43 on today's weights. History: One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 **Cycle 15 budget (§2): after the feed, transport 2.0 + outside the call 9.7 = 11.7 ms/token at G=512 against a ≈ 6.5 ms byte floor; ⑨ must deliver ≈ −5 ms for 20 ms/token, and 50 tok/s is reachable only on a 37 GB/s unit with ⑨ at its floor (rule 34). PR #119 (#85 skeleton) is in review with one unanswered clang-format nit on the new `HAP_perf.h` stub.** **Cycle 14: order #85 (skeleton, in progress) → #82 (small ops) → #81 (m=1 attention); #84 in parallel. All three end in host-gated PRs; the E2E A/B belongs to the wiring issue (not yet filed). USER DECISION (plans 81 §3.6, 82 last bullet), binds that wiring handoff, not these PRs:** the CPU decodes attention in fp16 with its own exp and runs no `_det` for RMSNorm / RoPE, so once the per-token entry is wired the NPU text cannot be bit-identical to the CPU run and B ≡ A is at risk too. Options: (a) keep text B ≡ A and expect a fail, (b) adopt `NNTR_PPL` (upstream `32b46e32`, #110 (d)) as the accuracy column for the ⑨ handoffs — this also lifts #110 to p1 per ⑱, (c) give the CPU path the same `_det` (NEON twin, doc 45 §3.3), which moves the CPU control and touches prefill. The supervisor recommends (b) with the CPU `q40` PPL as the reference, since #95 already measured that the NPU text leaves the CPU's at word 43 on today's weights. History: One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 | removes 22 round trips (**1.9 ms/token since #88**, was 12.5) and the time outside the MoE call (≈ 9.6 ms/token at G=512, floor ≈ 6.5); needed for the last ≈ 20 % to 50 tok/s once the MoE dsp column nears its floor (§2 budget row) | ⑤ ⑥ ⑦ |
| ⑩ | Registration at load time, bake cache on disk (doc 45 Phase D "P4") | load time, not speed | — |
| ⑪ | Prefill residency (doc 45 B/C/D) | prefill 523 → 700+ | after decode goal |
| ⑫ | CPU+NPU expert split | raises the ceiling only if ④ > 45 GB/s | ④, user decision Q11 |
| ⑬ | (withdrawn 2026-09-21: no simulator in this project, user decision) | — | — |
| ⑭ | **CLOSED (cycle 16b): #89 landed as PR #128 (`40dd4ca8`, merged `6497f38a`, 2026-09-27).** `generation(last 64)` in the base report block, host-proven on the tiny fixture at G = 70 and G = 10; not yet seen on device — the next handoff's A logs carry it and BENCHMARK's last-64 column fills from that row on (it cites PR #128) | fills the contract §1.1 column from the next handoff on | — |
| ⑮ | ~~Anchor sitting on a named unit~~ **withdrawn (user decision 2026-09-22, rule 13): any unit, same-sitting A/B only.** #94 sitting 2 (`dad0f476`, `R3CY205ZMND`) is the "now" in BENCHMARK.md and contract §1; no second-unit issue is opened | — | — |
| ⑯ | **Since PR #115 (`c45d4433`, cycle 13) the default is D192** — proof line `[HTP] moe m1 gemv: on (applied=0x103c1) lead=192KB rows1=1 source=default`; `applied=0x1` in an A log is the pre-#115 default and voids the sitting; A0 = `LEAD_KB=0 ROWS1=0` (`0xc1`). **Default since PR #108 (#101, merged `8e121dbe`, #101 closed; ride-along cells in #100's handoff):** env unset = GEMV on, `NNTR_MOE_HTP_M1_GEMV=0` = opt-out; proof line then `[HTP] moe m1 gemv: on (applied=0x1) source=default`. From that PR on, an A log with `off` or without `source=default` is the HMX path and is not the reference (the sitting-2 rule below is inverted). **Measured → §2 (goal progress +6.2 / +7.7 / +1.7 %, text identical).** Follow-ups: make the GEMV path the default (**#101**, PR #108 merged, so every later variant A carries it); its compute side (**#105**, measured cycle 9: gate failed, PR #107 held, §2) and weight feed (㉒ remainder, via #100). History: Device A/B of #80 / PR #86 (M=1 GEMV switch on vs off): the first lever against `lfm2_moe` 42.1 ms/token. **Cycle 4: PR #86 merged as `2a75f7d9` (2026-09-22 02:00 UTC) before #94's set was pushed, so it rides as variant C (`NNTR_MOE_HTP_M1_GEMV=1`, 6 NPU cells + one level-2 run + `*MoeLayerM1GemvMatchesHmx*`) of sitting 2 — handoff rebuilt from `2a75f7d9` (`htp/94-sitting2-anchor-trace` @ `8029b76e`). A on that head = the same binary with the switch off (`[HTP] moe m1 gemv: off (applied=0x0)`, `blocks=5632 m1_gemv=0/1408`); a C log that prints `on` in an A cell voids the run.** | MoE DSP 1.35 → ≈ 0.3 ms/call if the arena read keeps up; if not, the number says what ⑥ must deliver | #97 (skel loads) → #94 (user's phone time); first attempt blocked before any NPU cell |
| ⑲ | ~~#97~~ **closed (PR #98 merged `14f65120`; confirmed on silicon by #94 s2's `DmaProbeShapes` PASSED as the first device command).** Skel from `2a75f7d9` did not load (`0x80000406`): `hexkl_dma_trace.c` absent from `test/htp/build.sh` `SRCS` → seven undefined `hexkl_dma_trace_*`. Fix = add the file + an undefined-symbol guard in `build.sh` (rule 17); then #94's artifact set is rebuilt and the handoff re-issued (same document, new md5s) | unblocks every NPU cell of #94 (⑥ ⑮ ⑯ ⑰) | none |
| ⑳ | **Cycle 14: #99 downgraded to p2 (plan stands, `docs/plans/99-replay-content-check.md`).** The question it was filed to decide is answered: plan 99 found the replay writes each chunk packed (`dst_stride = row_size`) where the kernel writes strided, so the DDR read stream was byte-for-byte the traced one and rows c/d/g/h stood as timed; then #100 replaced row h with tag-validated cells (16/16 `checksum_ok=y`, rule 28) and rule 19 was withdrawn. What is left is hygiene: the rule-30/32 anchor cell (`DMA_REPLAY workers=1 load=0 pace=0`, one of the 11 old cells) rides every handoff — plan 117 reads it first and verbatim — with `checksum_ok=n` and a `FAILED` gtest that a reader will one day mistake for a regression. Gate for the remaining fix: `test/` only, the anchor line reads `checksum_ok=y` with `us_per_call` within ±10 % of the previous sitting's anchor (724.0), `weight_bytes` printed; rides along any sitting, no E2E cell. History: **Filed as #99 (p1).** `MoeChunkReplay` fails its content check on all 11 cells (`res[6]=1467840` vs `want_sum=9461760`, `bytes_per_call` 22560768 vs the in-situ 22020096, `regions=32`) while `DmaProbeShapes` passes in the same binary. Decides whether the replay timings (rows c, d, g, h; rule 19) stand or must be re-measured; ~~#100 is blocked on it~~ (cycle 9: only #100's old 11 cells and `traced*` depend on it) | rows c/d/g/h confirmed or re-measured | none |
| ㉑ | **Fixed by PR #108 (print only, #102; merged `8e121dbe`, #102 closed); row not yet seen on a device (#101 ride-along R2, rides #100's handoff).** Cause: on the GEMV path the `swiglu` slot is Σ lane-time over the pool (`moe_tail_probe_add`, by design), and the print subtracted it as a stage; the 30-slot table is not misaligned. `rest` now leaves it out, and a `weight DMA: n/a (direct arena read inside mm …, N.NN lanes busy over mm)` line replaces the missing one. **Filed as #102 (p2).** Level-2 `[HTP-PROFILE]` M==1 row with the GEMV path on: `swiglu 5601.2`, `rest<=-5588.2 (-311.8% of host)`, no `weight DMA:` line — the GEMV path's stage slots are mis-attributed in the print (`htp_compute_ops.cpp` / the 30-slot stage table of #86). Cosmetic, no tok/s cell affected, but the B row of every sitting after #101 is unreadable until fixed | a readable M==1 row under `NNTR_MOE_HTP_M1_GEMV=1` | #101 makes it urgent |
| ㉒ | **CLOSED (cycle 16): PR #118 merged `d79c0efe`; #120's A (nothing set, `applied=0x303e1 … feed=vtcm source=default`) vs A0 (`FEED=0`) = **+25.7 %** decode at G=64, `mm` 677.2 vs 934.3, text identical — the feed default confirmed as a sitting's control and BENCHMARK's "now" (35.97 / 35.96 / 35.17, §2 ㉔). Nothing left here; the engine gap is ㉓.** **Cycle 15: feed half MEASURED (#117 → §2), gate passed twice on `R3CY10WM83Y` (`mm` 676.4, decode +32.5 / +30.9 / +28.6 %, text = A 36/36); the flip to the feed default is in progress on PR #118 (#117 `state:in-progress`, acceptance on the issue) and ㉒ closes when it merges. #114 closed as moot. The residual engine gap is ㉓.** **Cycle 13: D192 is on `htp_moe` (PR #115 `c45d4433`), #113 closed; #117 is `state:planned` (`docs/plans/117-m1-gemv-vtcm-feed.md`, `a566e61f`) and its base is now the `htp_moe` head, not PR #115's branch — its sitting's A is the D192 default and carries #113's A/A0 confirmation.** **Cycle 12 (#113 → §2): compute half CLOSED, feed half re-filed as a rate lever.** The 2 × 5 (loop × lead) matrix is complete (rule 31): the lead harms the four-row loop, helps the one-row loop only to 192 KB, and the minimum D192 (`mm` 937.0, −6.3 %) misses the 840 gate; **the user lands D192 as the default anyway** (rule 33, PR #115, #113 in progress for the flip; decode +4.11 / +3.29 / +4.26 %). The ride-along (724.0 µs / 31.2 GB/s, rule 32) says 31–32 GB/s is the real DMA bound, so the feed half closes *as a 37 GB/s question* and the VTCM feed is worth what #100 measured: `f2` **31.6 GB/s beside HVX vs 23.5 GB/s** for D192's direct read = **≈ 1.33×** → `mm` ≈ 710 µs, ≈ −5 ms/token, with the one-row loop's ≈ 240 µs compute hidden under the feed (rule 29). **Filed as the ㉒ feed issue (**#117**, p1, `state:planned` since cycle 12's planner, `a566e61f`):** two linear descriptors per expert (`f2`: 8 descriptors, `dst=strided`, depth 2), gate `mm` ≤ 760 at M==1 (−19 % vs D192's 937), decode ≥ A at all three G, text = A, `bit_identical yes`, prefill −5 % on M>1 `dsp`, plus the `DMA_REPLAY` anchor cell in-sitting (rule 30). It must not re-import the 46-descriptor interleaving (rows f + g): at M=1 the ring carries only the expert's own `wh_bytes`, issued one expert ahead. #114 (up-half lead, p2) is moot if it lands. Earlier: **Cycle 11, after #100 and the user's direction ("stop leaning on the DMA numbers, take the compute-side wins first"): the feed half is DEMOTED and the lever is the `l2fetch` lead.** #100 measured the DMA feed at **31.6 GB/s** (best `f2`) against the GEMV's own direct arena read at 21–27, i.e. ≈ **1.2 ×**, not the 1.6 × (37 / 22.6) cycle 9 assumed — and on a sitting whose DMA is itself 22 % low (rule 30). VTCM staging therefore buys ≈ 690 → ≈ 550 µs of feed at best, and it costs a ring, a double-buffer and a prefill-gate risk. The cheap half is that **the feed is DDR latency, not bandwidth** (rule 26), and the only lever measured against it is monotonic and unsaturated: `l2fetch` lead 0 / 64 / 192 KB = 145.2 / 137.5 / 120.2 ns/tile. #105 confounded the lead with the loop shape — A is four-row with a zero-lead self-prefetch, B1/B2/B3 are one-row with lead 0/64/192 — so **the four-row loop has never been measured with any lead at all**. Filed as **#113 (p0)**: complete the 2 × N matrix, sweep past 192 KB, cross the expert boundary, land the winner (which decides PR #107's default and whether #107 lands at all). Gate `mm` ≤ 840 µs at M==1 (−13.7 % vs #100 A_L2's 980.9). **#114 (p1)** carries the next-expert / cross-op prefetch. The feed half stays open but unfiled, behind ⑥'s ride-along. Earlier: **Cycle 9: compute half measured (#105 → §2): gate failed. The best loop (one-row + 192 KB lead) gives `mm` 930.9 vs 973.0 (−4.3 %). The feed is ≈ 74 % of `mm` under that loop and ≈ 37 % under the four-row one. The feed half is the lever. It needs #107's loop to hide the compute: ≈ 240 µs vs the ≈ 590 the four-row loop leaves. PR #107 is held, and the feed issue rebases onto it. Order: #100 certifies a DDR → VTCM descriptor shape (≥ 37 GB/s beside HVX). Then the ㉒ feed issue is filed with that shape, gated `mm` ≤ 600 at M==1, text = A, prefill −5 %, and its A/B measures feed + one-row loop against A. If #100 certifies nothing, land #107 with a lead sweep (≥ 192 KB) and close ㉒ at what the arena allows.** Earlier: **Split (user direction, cycle 6, 2026-09-22). Compute side filed as #105 (p0, `state:needs-plan`):** one-row `gemm_rows1` at `m = 1` (drops `gemm_rows4`'s three dead accumulators; accumulation order unchanged, so int32 stays bit-identical to the HMX), `l2fetch` lead / distance, and the lane split over the pool. It needs no ring, so it does not wait for #100. The handoff sets `NNTR_MOE_HTP_M1_GEMV=1` explicitly in A and B, which keeps it independent of #101. **Feed side (remainder, not yet an issue):** VTCM staging of each expert's `wh_bytes` by DMA (two linear descriptors per expert); it must not re-import row h's slow list, so it is filed once #100 names h's cause. Evidence: C's `mm` 783 → 975 µs, a direct arena read at ≈ 22.6 GB/s effective, lane-bound (`swiglu / mm = 5.75`; plan 101 §3.4). Supervisor's estimate, not measured: the dead `vrmpy` are ~50–100 µs of the 975, and `hvx_impl` #59 measured direct DDR vector reads at 21–25 GB/s against 37 through the ring. So #105 alone may stop above 600; its ride-along microbench (arena vs VTCM copy × 4-acc vs 1-acc) sizes the feed half. Combined gate for ㉒ = `mm` ≤ 600 µs at M==1 with text identical | MoE dsp 1033 → ≈ 660 µs/call (`mm` 973 → ≈ 600; ≈ −8 ms/token) | feed: #100 → feed issue; compute: PR #107 (held) |
| ㉓ | **In-situ feed engine below the anchor (cycle 15, #117): `engine 32.3..34.4 GB/s` in the layer call vs the same sitting's anchor 37.3 and the `vtcm` microbench's 37.7** — `mm` 676 vs the 590 µs bound = **≈ 86 µs/call ≈ 1.9 ms/token (≈ 7 %)**, the last MoE-side lever on a 37 GB/s unit and nothing on a 31 GB/s one (there the in-situ 32–34 already exceeds the anchor). Signatures in B's ring line: `first expert ready at 122 µs` (the first 3.5 MB slab at 33 GB/s = 106 µs, un-hidden per call — the row-f analogue of #94's attribution), `depth max=4` with 10 descriptors, `last issue at 500 of 712`. Candidates, in order: kick expert 0's gate_up slab before the activation quant (the call's first ≈ 20 µs), issue expert e+1's gate_up before expert e's down instead of after, deeper issue; a cross-call prefetch of the next layer's expert 0 needs the router result and exists only with ⑨. **Not an issue yet:** it is filed after the flip's own sitting confirms the gap on the unit that runs it (a ride-along: the B-as-A ring line vs that sitting's anchor). Gate when filed: `mm` ≤ 630 in-sitting with the anchor ≥ 36 GB/s, text = A, prefill −5 % | ≤ −1.9 ms/token | #117 flip merged and measured as A |
| ⑰ | **`fully_connected` 28.2 vs 10.3 ms/token on the NPU run with no FC on the HTP** (§2 ① correction). Not a round trip; candidates: CPU thread over-splitting / contention with the FastRPC poll thread (doc 48 §2 ③), CPU DVFS while the DSP runs. 18 ms/token is the second-largest single lever after MoE and needs no DSP code if it is a threading matter. Decide with one extra profile cell (`NNTR_NUM_THREADS=4` on the NPU model, or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off/on) — candidate ride-along for #94 or its own issue — **#94 s2 skipped the ride-along (budget)**, so this still needs its cell | **Resized in cycle 7 (§2 budget row):** in TPS terms the whole outside-the-MoE-call time is 11.1–13.7 ms/token on the pre-#103 binaries (#94 s2 A and C, #88 A) and **8.4 (G=64) / 9.6 (G=512) on `htp_moe` head** (#88 B — the 5 ms poll / staging moved ≈ 5 ms of it, rule 23), against a ≈ 6.5 ms byte floor; the profile's 28.2 vs 10.3 is mostly the `--profile` build's own ≈ 20 ms/token. So ⑰ is worth ≈ 2–3 ms/token now, not 18: no issue; still a cheap ride-along (`NNTR_NUM_THREADS=4` on the NPU model, TPS binary) in the next sitting that has room | ≈ −2..−3 ms/token on the NPU path | next sitting's ride-along |
| ㉓ | **Filed as #132 (p1, `state:needs-plan`) on 2026-09-28; user order: #130 PR → #130 sitting → #132. Weight-format decision (HVX Q4_0 GEMV vs `QS4CX_WH` re-quant) is the plan's first question; accuracy read by the 2026-09-28 rule.** **Since #130's PR the only path below 22 calls/token; filed as #132 (user decision 2026-09-28).** With #82 / #81 wired the resident stretches are still bounded by the ARM's FCs, ADD and router, so every stretch is its own call (95/token with all six kinds). **The resident set that removes calls is larger than #81 + #82 (plan 85 §7; opened by PR #119).** `forward` stops at the first non-resident op, so the 22 calls/token fall only when *every* op between two MoE ops is resident: besides #82's norms / conv1d + gate / RoPE and #81's attention, the M=1 FCs (conv in_proj / out_proj, q/k/v/o, the dense FFN of layers 0–1), the router FC + top-k, and for the tail the final norm + lm_head. Those FC weights are Q4_0 on the NPU model (contract §11), so it is either a Q4_0 HVX GEMV on the DSP or a re-quantization of the FCs to `QS4CX_WH` — the latter moves the weights vs the CPU control and needs the user's say on the accuracy gate (the ⑨ user decision above covers text; this adds weights). Without it ⑨'s ≈ −5 ms cannot be reached. Supervisor's call: file it once #81 / #82 land, or fold it into the wiring issue | ≈ −5 ms/token at G=512 (transport 1.9 → ≈ 0.1, outside-the-call ≈ 9.6 → floor ≈ 6.5) | #85 (PR #119), #81, #82 |

## 3a. Guide and tooling notes

* **#130's PR (cycle 18): rungs 0–2 on the workstation** — `run_host_checks.sh`
  `ALL CHECKS PASS` / `GRAPH CHECKS PASS` / `GRAPH STRETCH BIT-IDENTICAL`,
  `run_inproc_e2e.sh` `INPROC E2E PASS`, `test/htp/build.sh` `UNDEFINED
  SYMBOLS OK (46 runtime imports)`. Two notes for the next handoff: (1)
  `build.sh`'s skel is **not bit-reproducible** (three consecutive builds
  of one tree gave three md5s), so a recorded skel md5 identifies a build,
  not a source state — the PR body says which build it is; (2) the committed
  goldens of `test/htp/host/golden/` were cut on the container (PR #127) and
  failed on the workstation on the very first MoE input (128 dB, the
  README's "this machine's CPU path differs" case) before any change of
  #130; with the container path reverted (PR #124 reverted) they are now
  cut on the workstation, and a golden mismatch on another host means that
  host, not the HTP path.

* **`docs/htp_moe/guide/`** exists since PR #92 (`cffcec3e`, merged
  `750428e8`, 2026-09-22): five self-contained English pages
  (`index`, `01-run-it`, `02-decode-path`, `03-performance`,
  `04-glossary`), decode only. The guide writer refreshes it on every
  filled handoff and every merged kernel/app PR (contract §8 step 4); its
  numbers are copied from BENCHMARK.md, never measured.
* Guide-writer findings folded into this ledger: (a) the FC 28.2 vs 10.3
  ms/token question (§2 ① correction, ⑰); (b)
  `Applications/CausalLM/install_android.sh` takes **no `--model`
  argument** in this tree — it pushes binaries and prints the `adb push`
  of a model dir as a hint (`install_android.sh:268–279`); handoffs must
  spell out the model push themselves (#77's did).
* **Rebuilding a handoff set on a fresh checkout (#105 run notes, cycle 9;
  rule 22):** four gaps the recipes did not state, now in the
  `hexagon-gates` skill rung 3. (1) A new `git worktree` has **empty
  `subprojects/`**. `build_android.sh` then dies on `iniparser.h` after
  ≈ 15 min. Run `git submodule update --init --depth 1` (or copy the
  populated dirs) first. (2) `Applications/CausalLM/lib/libtokenizers_android_c.a`
  is per-checkout. Copy it or rebuild it with `build_tokenizer_android.sh`.
  (3) `libc++_shared.so` may not appear under `jni/obj/local/arm64-v8a/`.
  Take it from the NDK r30 sysroot
  (`toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/`,
  md5 `b1586b9b…`). (4) `HEXKL_ROOT` differs between workstations. Export
  it explicitly next to `HEXKL_SDK_VER=6.4.0.1`, and never rely on
  `env.sh`'s default path.
* **Mac client (cycle 16, contract §4.1a):** only rungs 0–1 run on the
  Mac (`tools/docker/run.sh`, HexKL beta1 mounted; the tree needs beta.2
  for the skel). A PR opened from the Mac says in its body that rungs 2–3
  were not run; the workstation produces the skel/app md5s before any
  handoff. The GitHub MCP server cannot write from this client (HTTP 403
  `Resource not accessible by integration` on PR create, issue comment,
  issue close and label edits; reads work); agents use `gh` for every
  write: `gh pr create --repo dlwlzzero/nntrainer --base htp_moe`,
  `gh issue comment`, `gh issue close --reason completed`,
  `gh issue edit --add-label/--remove-label`.
* **Rungs 2–3 at the cycle-16 merge (2026-09-27):** PRs #121 / #125 /
  #126 / #127 / #128 were opened from the Mac with rungs 0–1 only; the user
  ran rungs 2–3 (skel + `build_android.sh --htp`) on the workstation at
  merge time, and **no skel/app md5s are recorded in those PRs**. The next
  handoff's rebuild recipe (rung 3, `hexagon-gates`) produces the md5s for
  `htp_moe` @ `758aaacc` or later; until then BENCHMARK's artifact table
  has no row for a post-#121 skel (the IDL grew by `mm_u8i4_conv_block`,
  the small-ops and `attn_m1_*` methods — a stale skel reads
  `0x8000040e`).
* **In-process HTP build (#84, cycle 17):** `meson setup build_htp_host
  -Denable-htp=true -Dhtp-inproc=true -Dhexagon-sdk-root=$HEXAGON_SDK_ROOT
  …` links the DSP skel subset (`test/htp/{hvx_add_f32,nntr_hvx_mm_u8i4,
  nntr_hvx_graph,nntr_hvx_small_ops,nntr_hvx_attn_m1}.c` + their `hmx/`,
  `hvx/` deps) into `libnntrainer.so` in place of the FastRPC stub, on
  scalar stand-ins (`test/htp/host/standin/hvx_scalar.c` for the HMX tile
  / GEMV / u8 quant / dequant / SwiGLU, `inproc/` for HexKL's micro API
  and the FastRPC runtime, `hvx_emu/` for the f32 intrinsics, `stub/qurt.h`
  for the pool) with `-Wl,--no-undefined` as the host twin of `build.sh`'s
  symbol guard. Gate: `bash test/htp/host/run_inproc_e2e.sh` → the five
  `E2E` lines and `INPROC E2E PASS` (tiny fixture, prompt 16 + 8, golden in
  `test/htp/host/golden/lfm2_moe_tiny/`). `NNTR_HTP_DUMP=<dir>` makes any
  run (host or phone) write every MoE call's input / output + a manifest;
  `tools/htp/htp_dump_eval.py <ref> <got>` compares two such directories.
  The profile's `dsp=` / transport columns on the host are the stand-ins'
  clock and mean nothing; never quote them. The real 8B model on the host
  needs an x86-packed twin (`nntr_quantize_stream … --moe_dtype QS4CX_WH`
  without `--isa ARM`) and is minutes per token on the scalar HMX
  stand-in: registration and dumps only, never a number.

## 4. Reusable code on `hvx_impl` (survey 2026-09-21; read with `git show hvx_impl:<path>`)

`hvx_impl` (Qwen3-0.6B, W8A8 HVX-only) is frozen but its kernels and
harness are device-validated. What lifts, in value order:

| lift | from | serves | adaptation |
|---|---|---|---|
| **DONE (#81, PR #126).** Lifted: the Kᵀ `[head_dim][max_seq]` layout (one f32 vector = 32 positions, not 64: f32 cache), the per-layer/per-kv-head cache indexing, the split by kv head with per-unit score scratch, the fp32 vector accumulation of PV, hvx/58's GQA fusion (one Kᵀ/V stream serves the 4 q heads of a kv head, as four accumulator sets in one pass). Not lifted: the fp16-widening scores (`hvx_vec_mpyacc_f32_f16`), the qf32 `hvx-exp.h`, the fp16 probabilities before PV, the scalar `1.0f / sum`, the hvx/58 position-block split + merge (re-associates the softmax), `HTP_ATTN_L2FETCH`, the token loop — all for the `_det` bit gate (plan 81 §3.3), the same reason #82 rejected `hvx-rmsnorm.c` / `hvx-rope.c`. Replaced by `nntrainer/tensor/attn_m1_det.h` + `hvx/hvx_attn_m1_f32.{c,h}`, plain Vsf with `hvx_exp_det_sf` / `hvx_recip_det_sf`. m=1 decode attention with a DSP-resident KV cache (K stored transposed `[head_dim][max_seq]` so one vector covers 64 positions; workers split by kv head) | `nntrainer/tensor/hexagon/htp/ops/hvx-attn.c` (152 LOC) | ⑨ attention | drop the token loop (m=1); per-worker score scratch from the orchestrator; KV dtype vs `hexkl_kv_quant.c` (fp16 there, u8 here → dequant in the score loop if u8 stays); `wp_run` → `hvx_worker_pool_run` |
| "DSP owns the graph": validate the op list once at init, then `forward(tokens, pos → logits)` runs `for op in ops: table[kind]()` with per-op pcycles and one FARF line per call | `htp/htp_graph.{h,c}` (~200 of 546 LOC), `htp/nntr_htp.idl` (3 methods incl. `forward_debug`), `htp/executor.c` (fd mmap + `AEE_EALREADY` handling) | ⑨ one call per token | add a `forward` entry beside the per-op IDL; rebuild `next_mm[]` over the LFM2.5 op sequence; repopulate the op table with MoE/conv1d/dense kinds |
| cross-op weight prefetch: two op-independent VTCM half-slabs so the last chunk of op N kicks chunk 0 of op N+1 while norm/RoPE/attention run in between | `htp/ops/hvx-matmul.c` `mm_slab`/`mm_worker_vtcm`/`mm_pf_kick` (~120 LOC) | ⑥ ⑨ | hvx_impl's per-worker push/pop DMA FIFO vs our global index ring (`hexkl_dma_ring_push2d` + `next_idx`/`wait`): rewrite the pipeline loop; keep the "drain, never abandon" rule for a prefetch left by another op |
| **DONE (#84, PR #127 `ec589373`).** Lifted: the `E2E step/gen` line format and the `--eval` verdict (`bit_identical` + SNR, as `tools/htp/htp_dump_eval.py` over per-call `NNTR_HTP_DUMP` dumps, the first differing file named). Not lifted: `HexagonRunner` / `RpcmemBuffer` (here `HtpBackend` / `HtpRpcMemApi` are the code under test), `run_e2e_test.sh` (agents never run adb; the md5 gate is `md5sum -c` in the handoff recipe), `summ_farf_prof.py` (the ARM prints `[HTP-PROFILE]` as a table), `find_divergence.py` (every call is dumped in order, the first non-identical file is the divergence), `make_tokens.py`, the static-map probe (`htp_moe` maps arenas with `FASTRPC_MAP_FD_DELAYED`). Replaced by the in-process build (`-Dhtp-inproc=true`, `build_htp_host`): the ARM owns the graph on `htp_moe`, so the driver is nntrainer's own decode loop (`test/htp/host/htp_e2e_test.cpp`, `run_inproc_e2e.sh`). host E2E harness without the CausalLM app: `hexagon_e2e_test` (`--tokens/--chunk/--steps/--eval/--dump-*`, `E2E` lines), `HexagonRunner`, `RpcmemBuffer`, md5-gated `run_e2e_test.sh`, `summ_farf_prof.py`, `find_divergence.py` (needs `forward_debug`), `make_tokens.py` | `test/hexagon/hexagon_e2e_test.cpp`, `nntrainer/tensor/hexagon/host/*`, `tools/hexagon/*` (~650 LOC) | measurement | replace the qwen3 lowering/config with LFM2.5's; `NNTR_HAVE_FASTRPC_MAP_STATIC` probe into `test/htp/build.sh` |
| **DONE (#82, PR #125).** Lifted: the per-head chunking (`chunk = head_dim`), the rotate-half structure with one vector per half, the cos/sin row formula as the host check's input generator. Not lifted: the fp16/qf32 arithmetic and `Vhf_equals_Wqf32` (v75/v79 differ), the scalar `1.0f / sqrtf` per row (libm bits no host spec can pin), `hvx-quant.h` (these ops end before the quantizer). Replaced by `nntrainer/tensor/m1_ops_det.h` + `hvx/hvx_m1_ops_f32.c`, plain Vsf with a `_det` bit gate. small ops: RMSNorm (also per-head q/k norm via `FLAG_PER_HEAD`), RoPE (`rope_rotate`, 20 lines, head_dim 128 hard-coded — LFM2.5 is 64), the shared fp32 cos/sin row `nntr_htp_rope.h`, and the quantizer's integer-only rounding recipe (one qf32 product, then sign/exp/significand split, ties-to-even) that made results v75/v79-portable | `htp/ops/hvx-rmsnorm.c`, `hvx-rope.c`, `nntr_htp_rope.h`, `hvx/hvx-quant.h` (~295 LOC) | ⑨ | activation dtype (fp16 there, f32 residual here); our quantizer is asymmetric u8 with zero point — port the rounding only if `hvx_quant_u8.c` still reads sf bits after a qf32 op |

Not lifted: the W8A8 tiled `vrmpy` matmul (wrong layout for int4; our `hvx_gemm_u8i4_wh_col` is the math), `dma-queue.c` (we have `hexkl_dma_ring.c`), the worker pool (ours is richer; note the opposite HVX-context convention: hvx_impl locks a unit per worker and leaves none for the caller, ours runs index 0 inline), the qwen3 lowering/packer/app glue, the simulator harness (no simulator here). Causal depthwise conv1d (L=3) + gating has no counterpart on either branch: net-new.

Silicon rules from `hvx_impl`'s HEXAGON.md §7 that bind here too: compute in fp32 inside an op and narrow to fp16 once (`Vhf_equals_Vqf16` after a qf16 multiply rounds badly, 2.7 % PPL); clamp the SiLU exp argument (already 85 here, doc 44); `-mhvx-ieee-fp` is required for fp16 intrinsics on toolchain 19; int32 `vrmpy` sums are exact in any order, divergence enters only in the epilogue; qf32→sf conversion differs between v75 and v79, so quantizers decode integers, never sf bits; HVX code runs only on threads that own an HVX context; validate `k` bounds for exact int accumulation; never compare wall-clock tok/s across units, report pcycles and `pcycles_per_us`.
