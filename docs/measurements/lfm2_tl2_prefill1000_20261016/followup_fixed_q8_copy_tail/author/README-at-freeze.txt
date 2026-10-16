Fixed Q8 master-row copy tail: separate correctness repair
========================================================

Status: production/source/tests frozen; host and v81 checks passed. The original
device failure is preserved, and the unchanged 36-call/4-probe device fixture
must be rerun by the parent with this SAME KV object in all compared variants.
This is separate from the raw-QK readout optimization. It makes no TPS claim.

Origin and contract
-------------------
The first original CONTROL device run failed fixed_q8 case 3 repetition 2:
"FAIL: A/B/A output reuse mismatch". The raw-plane candidate had not run.
The preceding full1024, full2048 and Q4 A/B/A groups passed. Original records:
  /tmp/nntr-lfm2-attn-raw-planes-root-20261017/device-fixture/control.log
  /tmp/nntr-lfm2-attn-raw-planes-root-20261017/device-fixture/control.execution.json
These records and the original device fixture are unchanged. Failed fixed A2
was checked before its output dump/CASE record and therefore was not retained
as a complete output. Do not infer its byte differences from the partial dump.

hexkl_kv_q_register accepts head_dim multiples of 32 through 512. Its masters
are exactly n_head_kv * rounded_max_rows * head_dim bytes, without tail padding.
hexkl_kv_q.h explicitly says rows may be rewritten; the fixture sets its fixed
scales once before the first append, then overwrites the same rows. This obeys
the public contract. Changing this fixture to avoid A/B/A would hide a bug.

On Hexagon, fixed Q8 sets plain_masters and enters append_fixed_i8. Its actual
quant_rows_worker partitions row ranges and quantizes exactly head_dim bytes.
The old xor80_copy instead loaded/stored 128 bytes for every nonempty remainder.
At head_dim=64 this overwrites the next row's 64 bytes, overlaps distinct workers'
row ownership, and writes 64 bytes beyond the final head's final master row.
The untouched part of the local quantization array is not valid row data.
The separate bake_hvx condition head_dim % 128 == 0 does not guard this copy.

This baseline memory-safety/ownership defect is independently reproducible and
consistent with the device mismatch. Successful rerun on the corrected common
baseline remains required to resolve the original device failure; host evidence
alone is not a hardware cache-coherency or complete attention correctness claim.

The original baseline KV source matches the fixed-object control inventory:
  6f20a5a4cbd0c4614c24b91cbd0c0c16f15f640e1d53014b66014b5039c4a46d
Source base: 73a4ab7baf30aa96fcdf7dbca3d7c86f899bca3a.

Narrow change
-------------
production.patch changes only xor80_copy in hexkl_kv_q.c, inside its existing
__hexagon__ guard. The vector loop handles n & ~127 bytes with its old load/XOR/
store body. A scalar byte XOR handles only the remaining bytes. There is no new
allocation, buffer, cache, worker, API, quantization or attention math change.
All three callers have disjoint source/destination ranges: two master-row copies
of head_dim bytes, and stage_v of 32*head_dim bytes. The stage size is always a
multiple of 128 for registered shapes. HD128/256/512 have no scalar remainder.
The dynamic-Q8 LFM path does not call this fixed-Q8 master-copy helper.

The raw candidate remains exactly d8d1827242aab3e698e7c69eefb73711305d37b843ba3798ba2b2e251e75fbb3
(attention C 8c54a3c86509c17b0c5cda01c0a2de6abf3de335fca6e268cb74b7dfed26d4e5).
Do not fold this repair into that patch or compare corrected and uncorrected KV
objects as if only raw attention differed. The parent plans one corrected KV TU
object shared by normal control, raw candidate and raw diagnostic, retaining all
other original objects and the original failed runtime under separate names.

Validation
----------
Portable test files are candidate/test/htp/host/kv_fixed_tail_check.c and
run_kv_fixed_tail_check.py; tests.patch adds these optional Linux host checks.
No vendor SDK files are copied or modified. The runner takes explicit paths.

Run with D=this bundle, R=the repository and HEXAGON_TOOLS=SDK tools root:
  python3 "$D/candidate/test/htp/host/run_kv_fixed_tail_check.py" \
    --repo "$R" \
    --source "$D/candidate/nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c" \
    --reference "$D/baseline/nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c" \
    --hexagon-tools "$HEXAGON_TOOLS" --output /tmp/kv-tail-independent

The runner verbatim-extracts each actual xor80_copy and quant_rows_worker plus
their context. It links actual hvx_kv_quant.c via SDK libnative, the unchanged
host production registry/fixed-scales TU, and the production worker pool with
the existing pthread QuRT stand-in. This is not a scalar mirror of the worker.
It deliberately does not emulate HMX baking, DSP DMA or hardware timing.

ASan + UBSan halt-on-error, no disabled checks or sanitizer suppressions:
  - Original actual HD64 copy: expected ASan heap-buffer-overflow, WRITE 128.
  - Original worker in deterministic reverse-lane order: expected master byte
    assertion failure. Later rows finish before the prior worker's last row.
    This exposes boundary corruption without relying on probabilistic racing.
  - Corrected copy: 3,094 cases: lengths 0..512 at six byte offsets, plus all
    legal full-stage sizes. Source and destination tails are ASan-poisoned;
    prefix/suffix canaries and every valid XOR byte are checked.
  - Corrected actual worker: 384 A/B/A cases (1,152 worker jobs), all legal HD
    multiples of 32, 2/3 heads, 1/2/4/8 lanes, reverse order and real pthread
    pool, whole 64 rows, last-row partial appends and interior partial appends.
    Full master allocations, untouched rows, K sums, K/V scale metadata and A/B
    difference/A replay are checked. K/V include signs, zeros, finite clamp
    inputs and fp16 subnormals. Actual quantizers are both executor/reference;
    this checks copy ownership and quantizer result preservation, not a new
    independently derived quantization formula.
  - Corrected versions of the two original regression reproductions pass.

checks/portable-final is the final packaged-runner execution. Earlier host-san
and final-san runs are retained. Both pre/post vector-copy source and expected
baseline failures are preserved rather than reclassified as ignored failures.

Compilation / scope
-------------------
The changed helper is a clang-format 14 no-op using the repository style;
the test C is formatted with that same version. Full before/after v81 KV TU
object and assembly compilation pass -O3/-Wall/-Werror using the frozen control
cc1 flags, with output paths confined to this directory. commands.json records
all exact commands; no parent ELF/object is modified.

All per-function stack frames are unchanged: quant_rows_worker 896 B,
append_fixed_i8 1,408 B, hexkl_kv_q_append_pool 128 B, bake_tiles_worker 32 B.
Constant-size 128/256/512 and full-stage 1024/4096/8192/16384 copy wrapper assembly
is identical before/after. Runtime-variable HD calls gain the full-bound/tail
checks; identical constants do not prove every generic caller instruction is
unchanged or prove unchanged hardware timing. Non-Hexagon whole-TU objects are
byte-identical. Windows/non-HTP production behavior remains compiled out; the
optional host runner explicitly rejects non-Linux hosts for its test stand-in.

Independent review is in /tmp/nntr-fixed-q8-copy-tail-review-20261017. Its author
independently reproduced both original failures and passed the corrected actual
helpers, actual worker/libnative, and whole-v81-TU compilation.

No source in the shared root worktree and no phone was changed by this author.
No new performance trial, device fixture relaxation or cache capacity change
was performed. Device outcome remains the parent's separate evidence.
