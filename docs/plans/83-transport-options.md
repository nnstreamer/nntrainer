# 83 — Transport options for the MoE FastRPC call (design input for #88)

Issue: dlwlzzero/nntrainer#83 (LEDGER §3 ⑦, wall 3; tracker #76).
Planner-only document; no source change, no device step. Written in the
same pass as `docs/plans/88-moe-call-marshalling.md`, which is the plan
that consumes it. Read against `htp_moe` @ `883024d4` and upstream PR
nnstreamer/nntrainer#4327 @ `b0a384d6` (fetched, not merged). SDK headers
cited from `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/incs/`.

Measured input: #77 B, unit `R3CY205ZMND`: M==1 `transport` 587.7 µs/call
at `NNTR_HTP_PROFILE=2`, 527.7 at level 3, `qos_mode` 2 → the DSP is hot
and the cost stays, so wake/clock is not it (LEDGER ②).

## 1. Inventory of the per-call cost

One `mm_u8i4_moe_layer` call at M=1 (K=2048, inter=1792, N_out=2048, 32
experts, top-4), from `test/htp/nntr_hvx.idl:318-329`, the qaic stub
`nntrainer/tensor/htp_backend/generated/nntr_hvx_stub.c` (`_stub_method_22`,
scalars `MAKEX(0, mid, 7, 1, 0, 0)` = 7 `in`, 1 `rout`) and
`htp_compute_ops.cpp` `invokeMoeLayer` `:1658-1773`:

| arg | bytes at M=1 | where the bytes live | per call the driver… | constant across calls? |
|---|---:|---|---|---|
| primitives (`M K inter N_out`, 6 lengths, `out_f32Len`) | 48 | stub stack | copies into the message | shape yes, lengths no |
| `h_gate_up` | 128 | `std::vector` rebuilt per call (`:1035-1046`) | copies (non-ION) | **yes, per layer** |
| `h_down` | 128 | same | copies | **yes, per layer** |
| `row_index` | 16 | `std::vector` (`lfm2_moe_layer.cpp:524-538`) | copies | no (routing) |
| `row_count` | 128 | same | copies | no |
| `row_weight` | 16 | same | copies | no |
| `act_f32` | 8 192 | `act_buf_`, **cached ION** (`HtpRpcBuffer`, `htp_rpcmem.h:85,99`; `RPCMEM_DEFAULT_FLAGS` = `ION_FLAG_CACHED`, `rpcmem.h:50-52`) sized by `ensureCapacity` `:1425` to the **largest shape seen = 4 MiB** at prompt 512 | cache clean over the **whole dma-buf** | buffer yes, content no |
| `out_f32` | 8 192 | `out_buf_`, same, 4 MiB | cache invalidate over the whole dma-buf | same |

Between the stub's return and the next call the host does: `stagedMemcpy`
of 8 KiB out (`:1741`), the rest of the layer (norm, residual add, the
next layer's ARM ops, router, `buildExpertAssignments`,
`tryMoeLayerOnAccelerator` `:455-545`), `stagedMemcpy` of 8 KiB in
(`:1683`). No `rpcmem_alloc`, `fastrpc_mmap` or registration per call:
`ensureCapacity` grows twice per process (first decode-shaped call, first
prefill-shaped call); the arena is attached at load. The DSP side is
tiny and already measured (`stage` 4.6 µs, `alloc` 0.2 µs, #77 B).

**Measured on device:** the B/C columns of `docs/measurements/88-moe-call-marshalling.md` (PR #103) fill this inventory's cost side; the host side prints it as the `staging:` line under the M==1 row (`act <class> B out <class> B ion=y rpc allocs=<n> non-ION in-args=6/464 B`).

**Reading.** Non-ION bytes crossing the stub per call: 464 (48 + 416).
The FastRPC driver copies small non-ION `in` buffers into its per-call
message; that is microseconds, not hundreds. The two ION buffers are 16
KiB of payload on 8 MiB of dma-buf; the author measured the driver's
cache maintenance at **36–59 µs/MB of buffer** (doc 50 §3.6, three
buffer sizes, decode call transport 553 → 814 → 1633 µs with DSP time
flat), so on `htp_moe` at prompt 512 the two 4 MiB buffers cost ≈
290–470 µs per decode call. The second term is the QoS window:
`htp_backend.cpp:77-78` sets `RPC_POLL_QOS` with `latency = 100`; per
`incs/remote.h:322-326` the CPU polls for that long and then "falls back
to waiting for a glink response". The decode call runs ≈ 1.2–1.35 ms, so
every call takes the interrupt path. The author measured 100 → 5000 µs as
decode call transport 158 → 83 µs (doc 51 §2.20). Sum ≈ 365–545 against
#77 B's 528–588. That is the whole of "marshalling": **buffer hygiene
(§3) and the poll window, not the arguments (§2)**.

Where an `NNTR_HTP_PROFILE=2` host-side line goes: after the M==1 row in
`HtpProfile::dump()` (`:545-660`), fed from `invokeMoeLayer` with the
staging buffers' actual sizes and a static count of `HtpRpcBuffer`
constructions; format in plan 88 §1. The existing `transport = host − dsp`
column already isolates the FastRPC cost from the memcpys (`stagedMemcpy`
sits outside the `host_us` window, `:1693-1710`).

## 2. Prebound handles

Per-layer constants in today's call: `h_gate_up`, `h_down` (32 + 32
`uint32`, 256 B). Everything else is per token (`row_*`) or per shape
(`M K inter N_out`). Scales, arena offsets and shapes are **already
DSP-resident** behind the weight handles (`weight_register_u8i4_arena`,
IDL `:66-73`; `hexkl_weight_u8i4_table` in `nntr_hvx_session.h:56`), so
there is nothing else to bind. `remote_handle64` already gives one
session with `moe_flags` (`nntr_hvx_session.h:62`, set once by
`moe_set_opts`) — the bind table is the same pattern, one level down.

IDL pair (additive, after `moe_set_opts` `:449`):

```
AEEResult moe_layer_bind(in sequence<uint32> h_gate_up,
                         in sequence<uint32> h_down, rout uint32 layer);
AEEResult moe_layer_unbind(in uint32 layer);
AEEResult moe_layer_run(in uint32 layer, in uint32 M, in uint32 K,
                        in uint32 inter, in uint32 N_out,
                        in sequence<uint32> row_index,
                        in sequence<uint32> row_count,
                        in sequence<float> row_weight,
                        in sequence<float> act_f32,
                        rout sequence<float> out_f32);
// + moe_layer_run_timed(..., rout sequence<uint32> stage_us)
```

Session: `moe_layers[32]` of `{n, h_gu[32], h_dn[32]}` (8 KiB static; no
DSP heap, no address-space note needed). `bind` validates every handle
against `weights_u8i4`; `weight_release_u8i4` of a bound handle returns
`AEE_EBADSTATE` (the `arena_detach` rule, IDL `:56-59`). Host: a map
keyed on `gate_up_data[0]` next to `handle_cache_` (`:2527`), bound on
the layer's first call. Stub scalars go from (7 in, 1 rout) to (5 in, 1
rout). Expected saving ≤ 5–10 µs per call against a ≈ 80 µs floor: two
fewer `remote_arg`s of 128 B. Worth building **with #85** (one call per
token needs per-layer tables on the DSP anyway), not on its own — plan 88
makes it a conditional step behind the §3 numbers.

## 3. Buffer hygiene

* **Size-class staging** — upstream `fb0f02b9` (`StagingPool`/`stage()`:
  one cached ION buffer per power-of-two class from 64 KiB, chosen per
  call). What it removes: the whole-dma-buf cache maintenance scaling
  with the largest shape; decode stays on 64 KiB × 2 (≈ 5 µs at the
  measured rate), the MoE prefill call at prompt 512 on the 4 MiB class.
  What it does **not** remove: the maintenance itself (a cached buffer is
  still cleaned/invalidated per call), the poll fallback, the two
  memcpys, the argument copies. Author's measurement: decode call
  transport 1626 → 161 µs in the all-FC-on config (doc 50 §3.7).
* **Persistent mapping (`fastrpc_mmap` / `remote_register_buf`).** The
  staging buffers are already persistent (allocated once, reused); the
  driver recognises rpcmem buffers and keeps their SMMU mapping across
  calls (`htp_rpcmem.h:16-22`). `fastrpc_mmap(FASTRPC_MAP_FD)` is what
  the **arena** uses so the DSP reads weights in place (`tryChunk`
  `:2350-2405`); for a 64 KiB activation buffer it would save the per-call
  map lookup but not the cache maintenance, and the kernel would then
  DMA from a mapped address instead of receiving a `remote_buf` — that is
  doc 45 §3.1's "activation handle" design and belongs to #85, not here.
  `remote_register_buf` (`remote.h:1605-1625`) is the same idea without
  the DSP-side address.
* **Uncached staging** (`RPCMEM_FLAG_UNCACHED`, `rpcmem.h:95`): removes
  the maintenance entirely, but the host reads 8 KiB of output from
  uncached DDR per call and the path diverges from upstream. Doc 46 §10.1
  is the DSP-side version of this trap (uncached `out_f32` scatter, 104
  ms). Rejected in plan 88 §3.2.
* **Poll window**: `NNTR_HTP_POLL_US` (upstream `04a2fcc4`), default 5000
  (`b0a384d6`); 10000 is refused by the driver and the session falls to
  PM QoS (`qos_mode=1`, doc 51 §2.16). Cost: the calling core spins up to
  5 ms while the DSP works (it is otherwise idle in a synchronous call).
  Not a buffer matter, but it is the second half of the 528 µs and the
  same two commits carry both, so plan 88 takes them together.

Smallest host-testable first step for #88: cherry-pick the three upstream
commits, add the `staging:` line, rung 1 + rung 3. What cannot be known
without silicon: the transport column itself (every number above is the
author's unit or #77's), whether 5000 µs is accepted on `R3CY10WM83Y`,
and whether prebind's ≤ 10 µs is even visible.

## 4. dspqueue and the resident worker — one paragraph

`incs/dspqueue.h` is an asynchronous packet queue: the host writes
packets (message + buffer references, `DSPQUEUE_PACKET_FLAG_BUFFERS`,
with explicit `FLUSH_SENDER`/`INVALIDATE_RECIPIENT` flags per buffer) and
a DSP thread that never returns consumes them; signalling is tuned with
early-wakeup packets (`DSPQUEUE_STAT_EARLY_WAKEUP_*`). It removes the
per-call invoke/return and lets the host choose exactly which bytes get
cache maintenance — but #77 B showed the DSP hot and the cost unchanged,
and §1 attributes that cost to buffer size and the poll window, both
fixable on the synchronous path in two upstream commits. A resident
worker on plain FastRPC has the same shape (one long call, a shared
ring) and the same argument. Both return when the call pattern changes:
**one call per token (#85)** makes the per-call constant term the whole
transport budget, and a queue that pipelines token *t+1*'s dispatch
behind token *t*'s DSP work is then the only way below the synchronous
floor (~50–80 µs on this SDK, to be measured by plan 88's B cell).
Condition to reopen: #85 landed and the M==1 transport ≥ 10 % of the
per-token DSP time.
