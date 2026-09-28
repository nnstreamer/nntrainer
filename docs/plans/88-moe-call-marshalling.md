# 88 — Wall 3: the MoE call's per-call transport on plain FastRPC

Issue: dlwlzzero/nntrainer#88 (LEDGER §3 ⑦, wall 3; tracker #76; design
input #83 → `docs/plans/83-transport-options.md`, written in the same
pass). Contract `docs/plans/0001-htp-moe-decode-agent-system.md`. Branch
`htp/88-moe-call-marshalling` from `htp_moe` @ `883024d4`; every
`path:line` below was read against that tree, the upstream commits against
PR nnstreamer/nntrainer#4327's head `b0a384d6` (fetched, not merged).

**Baseline assumption.** The wall-3 numbers this plan is read against are
#77 B on unit `R3CY205ZMND`: M==1 transport **527.7 µs/call at
`NNTR_HTP_PROFILE=3`, 587.7 at level 2**, `qos_mode` 2 (BENCHMARK.md side
table). Sitting 2 (#94) has not run, so there is no `R3CY10WM83Y` anchor
yet; the handoff in §4 step 4 measures its own A first and reads B against
it (LEDGER rule 9), never against #77.

**What the 528 µs are (the finding this plan rests on).** Two upstream
commits in this exact area, both measured on the author's unit, both
absent from `htp_moe`:

| upstream sha | what it removes | author's numbers (doc 50 §3.6–3.7, doc 51 §2.20) |
|---|---|---|
| `fb0f02b9` "Stage each call through an ION buffer of its own size class" | The driver's per-call cache clean/invalidate on a **cached ION buffer covers the whole dma-buf, not the bytes passed**. On `htp_moe` every call stages through one `act_buf_`/`out_buf_` pair grown to the largest shape seen (`ensureCapacity`, `htp_compute_ops.cpp:1425`), so decode's 8 KiB + 8 KiB rides a pair sized by the prompt: prompt 512 × 2048 × 4 B = **4 MiB + 4 MiB per call**, at the measured 36–59 µs/MB ≈ 290–470 µs | MoE-only config: 553 → ≈ 290 expected; all-on: 1626 → **161 µs** measured |
| `04a2fcc4` + `b0a384d6` `NNTR_HTP_POLL_US`, default 100 → **5000 µs** | `RPC_POLL_QOS` polls for `latency` µs then falls back to the glink interrupt wait (SDK `incs/remote.h:322-326`). `htp_backend.cpp:78` sets 100 µs; the MoE decode call lasts ≈ 1.2–1.35 ms, so every call pays the interrupt tail | decode call transport **158 → 83 µs**, MoE prefill call 2190 → 1131, decode 20.6 → 23.7 tok/s; 10000 refused by the driver (falls to `qos_mode=1`) |

Sum of the two ≈ 365–545 µs against #77 B's 528–588: the "marshalling"
verdict of LEDGER ② is, more precisely, **cache maintenance on an
over-sized staging pair plus the poll-window fallback**. The per-call
argument marshalling proper (§2 inventory) is ≤ 0.5 KiB of non-ION
buffers and cannot be where 500 µs go. The plan therefore takes the two
upstream fixes first and makes the issue's "prebind" a measured,
conditional third step.

**If the user merges upstream (`b0a384d6`) into `htp_moe` first**, step 1
of §4 disappears: variant A of the sitting already carries both fixes and
its M==1 transport should read ≈ 80–160 µs. #88 then measures only its
own additions (instrumentation, conditional prebind) and the issue's
"≥ −30 % is progress" clause is void — the only gate left is the absolute
one, transport ≤ 0.1 ms. The plan is written so that either order yields
the same tree.

## 1. Goal and gate

Acceptance criterion (issue), made measurable:

| criterion | what counts as done |
|---|---|
| Host check: per-call argument size and per-call allocations printed by a `NNTR_HTP_PROFILE=2` host line, at the inventory floor | New line under the M==1 row: `[HTP-PROFILE]     staging: act 65536 B out 65536 B ion=y  rpc allocs=<n> (session)  non-ION in-args=6/416 B` (numbers from §2's inventory: two 64 KiB class buffers, `n` ≤ 4 after a prefill+decode run = 64 KiB and 4 MiB class × act/out; 6 non-ION `in` buffers totalling 416 B at M=1, or 4/160 B once prebind lands). Without a device the line is exercised by the syntax check only; its arithmetic is a `static_assert`-free constant table, so the check is the printed value on device |
| "x86 stub round-trip shows the same arguments" | No x86 FastRPC round-trip harness exists in this tree (`tools/htp_syntax_check.sh:15-18` declares every `nntr_hvx_*` variadic on purpose) and building one is speculative. The inventory is read from the generated stub instead: `generated/nntr_hvx_stub.c` `_stub_method_22` packs `REMOTE_SCALARS_MAKEX(0, mid, 7, 1, 0, 0)` — 7 `in` (1 primitive block of 48 B + 6 sequences) and 1 `rout`. The PR body pastes that line before and (if prebind lands) after |
| `tools/htp_syntax_check.sh`, `ninja -C build`, `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6 | unchanged gates (the tiny fixture never reaches the HTP path) |
| Skel + app build, md5s in the PR | rung 2 only if the IDL changes (step 5); rung 3 always (the stub and `htp_compute_ops.cpp` change) |
| Bit-identity | The DSP arithmetic is untouched in steps 1–4 (host staging and QoS only; the kernel reads `act_f32` once into `act_c` and writes `out_f32` once, `hexkl_mm_u8i4_moe.c:1000-1005`, wherever the bytes live). Text at G = 64 / 512 / 1024 identical to variant A's in the same sitting; run1 = run2. `NNTR_L2_DIFF` n/a for `QS4CX_WH` (LEDGER rule 6) |
| Handoff verdict (LEDGER ⑦ target) | M==1 row `transport` at `NNTR_HTP_PROFILE=2`, B vs A in one sitting: **≤ 0.1 ms/call** = done; ≥ −30 % = progress (issue clause); < −15 % = no change → back to #83 §4's remaining options |
| Prefill gate | prompt-512 prefill tok/s of B ≥ −5 % of A; the M>1 row's transport must not rise (expected to **fall**: the M>1 call keeps its 4 MiB class but gains the 5 ms poll; author: 2190 → 1131) |

Standing gates: text identical to the CPU run n/a for `QS4CX_WH` (clause
(c) is replaced by A = B text); `NNTR_NUM_THREADS=8`; no `--profile`
binary for tok/s (rule 1).

## 2. Where it lives

### 2.1 Inventory: one M=1 `mm_u8i4_moe_layer` call at `htp_moe` head

From `test/htp/nntr_hvx.idl:318-329`, the stub `_stub_method_22`, and
`invokeMoeLayer` (`htp_compute_ops.cpp:1658-1773`). Shapes: M=1, K=2048,
inter=1792, N_out=2048, 32 experts, top-4.

| remote_arg | IDL argument | bytes at M=1 | memory | who pays |
|---|---|---:|---|---|
| in 0 | `M K inter N_out` + 6 sequence lengths + `out_f32Len` | 48 | stub stack | copied into the invoke message |
| in 1 | `h_gate_up` `sequence<uint32>` | 128 | `std::vector` heap, rebuilt per call (`:1035`) | driver copy (non-ION) — **per-layer constant** |
| in 2 | `h_down` | 128 | same | same — **per-layer constant** |
| in 3 | `row_index` | 16 (4 rows) | `std::vector` from `lfm2_moe_layer.cpp:524-538`, per token | driver copy |
| in 4 | `row_count` | 128 | same, per token | driver copy |
| in 5 | `row_weight` | 16 | same, per token | driver copy |
| in 6 | `act_f32` | 8 192 | **`act_buf_` cached ION, size = largest shape seen = 4 MiB after prompt 512** (`ensureCapacity` `:1425`, `HtpRpcBuffer` `htp_rpcmem.h:99` with `RPCMEM_DEFAULT_FLAGS` = cached) | driver cache maintenance over the **whole 4 MiB** |
| rout 0 | `out_f32` | 8 192 | `out_buf_`, same 4 MiB | same |

Per-call host work outside the stub: `stagedMemcpy` in of 8 KiB and out
of 8 KiB (`:1683`, `:1741`; counted in "arm staging memcpy", not in
`transport`), `std::vector` builds of `h_gu`/`h_dn` (`:1035-1046`), and
the routing vectors in `tryMoeLayerOnAccelerator`. **No `rpcmem` alloc,
`fastrpc_mmap`, or registration happens per call**: `ensureCapacity`
allocates only when growing (twice in a run: first decode call, first
prefill call), and the arena is attached at registration. The per-call
ION cost is therefore not allocation but cache maintenance proportional
to the buffer, which is exactly what `fb0f02b9` measured.

### 2.2 Files that change

* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp`
  * `ensureCapacity` `:1425-1431` and `act_buf_`/`out_buf_` `:2554-2555`
    → upstream's `StagingPool`/`stage()` (size classes from 64 KiB,
    powers of two), applied at the six call sites `fb0f02b9` touches:
    `invokeLayer` `:1489`, `invokeLayerU8In`, `invokeFused`,
    `invokeMoeLayer` `:1679-1682`, `invokeGateUpSwiglu`,
    `invokeLayerU8InRaw`. `act_ah_buf_` keeps `ensureCapacity` as upstream
    does.
  * `HtpProfile` (`:193-700`): new per-bucket fields `stage_act_bytes`,
    `stage_out_bytes`, `stage_ion`, and a session counter of
    `HtpRpcBuffer` constructions; printed as the `staging:` line after the
    M==1 / M>1 row in `dump()` `:545-660`. `addInvokeMoeLayer` `:303`
    gains the two byte arguments. The `HTP_MOE_T_*` table `:146-186`
    is **not** touched (no DSP stage changes).
  * Step 5 only: `sendMoeOptsOnce`-style once-per-layer bind cache and
    the `moe_layer_run` call (see §3.3).
* `nntrainer/tensor/htp_backend/htp_backend.cpp:74-90` → upstream
  `04a2fcc4` + `b0a384d6` verbatim: `lat.latency = 5000`, `NNTR_HTP_POLL_US`
  override, the `ml_logi` line. `htp_backend.h` unchanged.
* `nntrainer/tensor/htp_backend/htp_rpcmem.h:99-108`: one static counter
  in the `HtpRpcBuffer` constructor for the `rpc allocs=` field (2 lines).
* Not touched unless step 5 runs: `test/htp/nntr_hvx.idl`,
  `generate_stub.sh` / `test/htp/build.sh:72-75` (regenerate from the
  IDL), `test/htp/nntr_hvx_mm_u8i4.c`, `test/htp/nntr_hvx_session.h:51-64`.
* Never touched: the quantizer's format tag (`nntr_quantize_stream`, no
  layout change), the loader check (`float_tensor.cpp:765-775`),
  `tools/htp_fc_report.py` (parses the FC `FC_STAGE` records only; the
  MoE table is printed by `dump()` directly), `hexkl_mm_u8i4_moe.c`,
  `lfm2_moe_layer.cpp`.

## 3. Design

### 3.1 Chosen: upstream's two fixes verbatim, then instrument, then decide prebind on the number

1. **Size-class staging = cherry-pick `fb0f02b9` with `-x`.** It is the
   PR author's own fix for the mechanism, it is what `htp_moe` will
   contain once the user merges upstream (contract §5 Q16, decision Q14
   "upstream-shaped commits"), and a different shape here would conflict
   at that merge. Expect offset conflicts in `invokeMoeLayer` (the `reps`
   loop and the `[#87]` trace block sit below the hunk) — resolve by
   keeping upstream's text. Decode stays on the 64 KiB pair, the MoE
   prefill call at prompt 512 on the 4 MiB class (4 194 304 B fits
   exactly). ION total ≈ 8.25 MiB instead of 8; no DSP address-space cost
   (the staging buffers are host-side, never `fastrpc_mmap`'d).
2. **Poll window 5000 µs = cherry-pick `04a2fcc4` then `b0a384d6`.** Same
   provenance argument. The knob is also this plan's free variant: variant
   C = variant B's binary with `NNTR_HTP_POLL_US=100`, which separates the
   two contributions in one sitting without a third build.
3. **Host-side inventory line** (§2.2) so the handoff's proof is printed,
   not inferred: buffer class sizes, ION yes/no, allocation count,
   non-ION in-arg count and bytes.
4. **Prebind** (`moe_layer_bind` / `moe_layer_run`, IDL shape in
   `83-transport-options.md` §2) is **step 5, conditional**: built only if
   the step-4 handoff's B transport is still > 100 µs **and** C (poll 100)
   vs B shows the poll change is saturated. At M=1 prebind removes two
   128-byte non-ION `in` buffers (`h_gate_up`, `h_down`); `row_index`,
   `row_count`, `row_weight` are per-token routing and stay. Expected gain
   ≤ 5–10 µs of a ≈ 80 µs floor; cost = an IDL change (skel + app
   rebuild, the `AEE_EBADPARM` trap, LEDGER rule 3) and a DSP-side table.
   It returns with #85 (one call per token) where per-layer state on the
   DSP is needed anyway, so the IDL shape is specified now and built then
   unless step 4 says otherwise.

Respecting contract §2: no CPU fallback is added for `QS4CX_WH`; the
arena budget is untouched; wall 3 is attacked on plain FastRPC as #77 B
decided (dspqueue / resident worker stay one paragraph, #83 §4). Doc 45
§3: no new kernel, so no `_det` question; activation handles (§3.1 there)
are what step 5's `layer` handle starts.

### 3.2 Rejected: a dedicated uncached decode pair (or prebind first)

Allocating a separate `RPCMEM_FLAG_UNCACHED` 64 KiB pair for M ≤ 4 would
zero the cache maintenance rather than shrink it to ≈ 128 KiB (≈ 5 µs).
Rejected: the host then reads 8 KiB of output from uncached DDR per call
(doc 46 §10.1 is the DSP-side version of that lesson), it diverges from
upstream's shape for a gain below the sitting's noise, and it adds a
second code path to the one the prefill call uses. Prebind-first (the
issue's ordering) is rejected by §2.1's arithmetic: 256 B of the 416
non-ION bytes cannot carry 500 µs, and building the IDL change before the
staging fix would measure prebind against a 4 MiB flush.

### 3.3 Step 5 shape, for the record (built only if step 4 asks)

IDL, additive at the end of the interface (after `moe_set_opts`):
`moe_layer_bind(in sequence<uint32> h_gate_up, in sequence<uint32> h_down, rout uint32 layer)`
and `moe_layer_run(in uint32 layer, in uint32 M, in uint32 K, in uint32 inter, in uint32 N_out, in sequence<uint32> row_index, in sequence<uint32> row_count, in sequence<float> row_weight, in sequence<float> act_f32, rout sequence<float> out_f32)`
plus `moe_layer_unbind(in uint32 layer)` and a `_timed` twin. Session:
`nntr_hvx_session` gains `struct { uint32_t n; uint32_t h_gu[32], h_dn[32]; } moe_layers[32]`
(8 KiB, static, no heap). Host: `std::unordered_map<const void *, uint32_t>`
keyed on `gate_up_data[0]` next to `handle_cache_` `:2527`; the stub's
scalar becomes `MAKEX(0, mid, 5, 1, 0, 0)`. The `bind` refuses a handle
not in `weights_u8i4`; `weight_release_u8i4` of a bound handle returns
`AEE_EBADSTATE` (same rule as `arena_detach`, IDL `:56-59`).

## 4. Steps

Each ends in a rung of `.claude/skills/hexagon-gates`.

1. **Upstream fixes** (skip if `htp_moe` already contains `b0a384d6`):
   `git cherry-pick -x fb0f02b9 04a2fcc4 b0a384d6` onto the branch;
   resolve `htp_compute_ops.cpp` offsets; do **not** take `6f8f7791`
   (conv row, not in this tree). Gate: rung 0 + rung 1 (`ninja -C build`,
   `tools/htp_syntax_check.sh`, `*Lfm2Moe*` 6/6, `run_host_checks.sh`
   unchanged because no DSP source moved).
2. **Inventory line**: `HtpProfile` fields + `staging:` line +
   `HtpRpcBuffer` counter; `addInvokeMoeLayer` signature. Gate: rung 1;
   PR body carries §2.1's table and the `_stub_method_22` scalar line.
3. **App build**: rung 3 (`build_android.sh --htp`, `readelf` NEEDED
   lines, md5s). Skel unchanged → rung 2 not needed; record that the
   device skel of sitting 2 (`libnntr_hvx_skel.A.so` `20fb9801…`) is
   still valid for this app.
4. **Device measurement (unavoidable — the transport column exists only
   on silicon).** Handoff `docs/measurements/88-moe-call-marshalling.md`,
   `state:needs-measurement`, ≈ 45 min:
   * **A** = the sitting-2 reference set (`htp_moe` @ `2a75f7d9` app +
     skel A), `NNTR_HTP_PROFILE` unset; NPU model, prompt 512, G = 64 /
     512 / 1024 × 2. If #94 already ran in the same sitting, its A cells
     are this A.
   * **B** = this branch's app, same skel, env unset: the same 6 cells.
     Full E2E, text diffed against A per cell.
   * **C** = B's binary with `NNTR_HTP_POLL_US=100`: G = 64 × 1 (isolates
     staging vs poll).
   * Profile block, G = 64, one run each, `NNTR_HTP_PROFILE=2`: A, B, C.
     Paste the header (`qos_mode` must be 2 in all three — a `1` means
     the poll value was refused), the M==1 and M>1 rows, and B/C's
     `staging:` line. Also `NNTR_HTP_PROFILE=3` for B only (the #77 B
     twin).
   * Verdict table: transport A / B / C, decode tok/s per cell, prefill
     gate, text = A. A fourth variant (D = prebind on) is added only if
     step 5 was built before the sitting.
5. **Conditional prebind** (§3.3), only if step 4 reads B > 100 µs with
   C − B saturated: IDL + skel + `HtpComputeOps` + gtest
   `TEST_F(HmxMmU8I4Layer, MoeLayerBoundMatchesUnbound)` (`memcmp == 0`
   between the two entries). Gates: rung 1, **rung 2** (skel md5), rung 3;
   then a second handoff with D = B + prebind.

## 5. Risks

* **Unit and sitting drift** (rules 9, 13): the #77 B floor is from the
  other unit; the handoff reads B only against its own A, and the C cell
  tells whether the poll part reproduces on `R3CY10WM83Y`.
* **Poll spin vs the CPU FC row** (LEDGER ⑰): 5 ms of polling per call is
  on the calling thread while the DSP works, so it should not contend
  with the 8 CPU threads — but ⑰'s `fully_connected` 28 vs 10 ms/token is
  unexplained. The level-2 run's ARM-side time and the decode tok/s of B
  vs A make a regression visible; if B's decode falls while transport
  drops, `NNTR_HTP_POLL_US=1000` is the next cell.
* **Driver refusing the value** (doc 51 §2.16: 10000 → `qos_mode=1`):
  the profile header prints `qos_mode`; a `1` voids the cell.
* **Thermal drift between A and B blocks**: A at G=64 is re-run after B
  (the #94 pattern, block 4) to show the sitting did not move.
* **Stale skel**: none of steps 1–3 touch the skel; step 5 does, and its
  handoff must ship skel + app from one commit (rule 3).
* **Cherry-pick conflicts silently dropping a call site**: after step 1,
  `grep -n "ensureCapacity(act_buf_\|ensureCapacity(out_buf_"` must
  return nothing (`act_ah_buf_` may remain).
* **Host-vs-device gap**: nothing in steps 1–3 is testable for effect on
  the host; the rungs prove only that the code builds and the fixture's
  CPU path is unchanged. Stated plainly in the PR body.

## 6. Docs to update

* BENCHMARK.md: 6 B rows + 1 C row in Results (unit-tagged); the side
  table "Transport floor" gains a B/C line next to #77 B; Artifacts row
  for the #88 set (app md5s; skel unchanged unless step 5).
* LEDGER.md: §2 ② verdict refined ("cache maintenance on the over-sized
  staging pair + poll fallback; per-call arguments ≤ 0.5 KiB"); §3 ⑦
  status and the measured number; Upstream table: `fb0f02b9`, `04a2fcc4`,
  `b0a384d6` marked "cherry-picked into `htp_moe` by PR #<n>" (or "merged
  by the user" if that happened first); a new rule if the poll value
  behaves differently on our unit.
* `83-transport-options.md` §1 inventory gets the measured B/C numbers.
