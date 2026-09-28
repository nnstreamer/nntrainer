# 84 — Host E2E harness without the phone: the DSP skel linked into the process, driven by the real ARM path

Issue: dlwlzzero/nntrainer#84 (LEDGER §4 lift 4; tracker #76). Read against
`htp_moe_cycle` @ `763762a2`. Aligned with `docs/plans/85-per-token-entry-skeleton.md`
(the op table this harness must exercise), `82-m1-small-ops.md` §3.2 (`hvx_emu/`)
and `81-m1-attention-kv-cache.md` §3.3.

**What this buys, plainly: 0 tok/s.** It makes the ARM-side HTP path
(`HtpComputeOps`, `HtpBackend`, the MoE call marshalling, #85's `graph_init` /
`forward` once it lands) runnable on the workstation against the DSP sources
compiled for x86, so #85 / #82 / #81 get a whole-model, per-token host gate
instead of a phone sitting for every wiring question. Today the only E2E driver
is `nntrainer_causallm` on the phone; `run_host_checks.sh` stops at the kernel
loop with hand-built inputs.

## 0. What `hvx_impl` had, what lifts, what does not

`hvx_impl`'s harness drove a **DSP-owned graph**: `hexagon_e2e_test` loaded a
packed image into rpcmem, called `HexagonRunner::forward(tokens, pos → logits)`
once per token, and `--eval` was teacher-forced PPL from those logits. That
shape does not exist on `htp_moe`: the ARM owns the graph and the DSP runs one
MoE op per layer (22 calls/token), and will keep owning it through #85 → #81
(plan 85 §3.1: "the ARM runs the remainder = nntrainer's normal executor").
A standalone token driver would have to re-implement the model.

| hvx_impl piece | verdict | why |
|---|---|---|
| `hexagon_e2e_test.cpp` (arg parsing, `E2E step … top1= logprob=` / `E2E gen` line format, medians) | **line format lifted**, binary not | the per-token driver here is nntrainer's own decode loop (`Applications/CausalLM/models/causal_lm.cpp:625-640`), reached through the tiny-fixture test runner; the harness prints the same `E2E` lines so log parsers carry over |
| `--eval` (identical → flag, else SNR) | **lifted as a dump comparator** (`tools/htp/htp_dump_eval.py`) | the issue's spec; hvx_impl's PPL meaning is replaced by SNR + `bit_identical` over per-call activation dumps, which is what doc 45 §3.4 (a)/(b) need |
| `HexagonRunner`, `RpcmemBuffer` | **not lifted** | `HtpBackend` (`htp_backend.cpp:33-110`) and `HtpRpcMemApi` (`htp_rpcmem.h:44-75`) are the `htp_moe` equivalents and are the code under test, not a dependency |
| `run_e2e_test.sh` (adb push/run/logcat; its md5 only skips an unchanged push, it never refuses) | **not lifted** | agents never run adb (contract §4.2); the user's bridges differ per sitting (BENCHMARK rows: "remote adb bridge", "ADF SSH bridge"). The md5 *gate* the issue asks for is `md5sum -c staged/md5.txt` (coreutils, exits 1 on mismatch): one line in the `hexagon-handoff` recipe, no script |
| `summ_farf_prof.py` | **not lifted** | the ARM prints `[HTP-PROFILE]` as a table already (`htp_compute_ops.cpp:523`); the FARF lines are silent at `RUNTIME_HIGH`; per-op pcycles arrive with #85's `forward_debug` and are read from that table |
| `make_tokens.py` | not needed | the app tokenizes; the tiny fixture ships `input_ids.json` |
| `find_divergence.py` (binary search over `forward_debug` dumps) | **its job is absorbed** | every MoE call's input and output is dumped in call order, so the first non-identical file *is* the divergence; no search |
| `NNTR_HAVE_FASTRPC_MAP_STATIC` probe | not lifted | `htp_moe` maps arenas with `FASTRPC_MAP_FD_DELAYED` + `HAP_mmap_get` (`htp_compute_ops.cpp:2449`, `nntr_hvx_mm_u8i4.c:361`); no static-map variant exists here |

## 1. Goal and gate

Host gates only; no device step; no IDL change. The issue ends in a PR into
`htp_moe` with contract §9 gates 0–3 (rung 3 once, because `htp_compute_ops.cpp`
gains the dump hook).

1. **In-process HTP build links.** `meson setup build_htp_host -Denable-htp=true
   -Dhtp-inproc=true …` then `ninja -C build_htp_host` produces `libnntrainer.so`
   with `-Wl,--no-undefined`, containing the real `htp_compute_ops.cpp` /
   `htp_backend.cpp` / `htp_context.cpp` **and** the skel entries the ARM path
   calls (`nm -D` lists `nntr_hvx_open`, `nntr_hvx_mm_u8i4_moe_layer`,
   `nntr_hvx_moe_set_opts`, `nntr_hvx_arena_attach`, …, all `T`). A skel entry
   the ARM calls but the inproc list lacks is a **link error**, the host twin of
   rule 17's undefined-symbol guard.
2. **Whole-token run on the tiny fixture, bit-identical to a committed golden.**
   `test/htp/host/run_inproc_e2e.sh` builds the tiny LFM2-MoE fixture
   (`test/unittest/models/causallm_reference/lfm2_moe_tiny`, hidden 64, inter 64,
   4 experts, top-2, 1 attention + 1 conv layer), quantizes it twice
   (`--moe_dtype QS4CX` = CPU control, `--moe_dtype QS4CX_WH` = HTP model; FC
   `Q4_0`, host ISA), runs prompt 16 + 8 greedy tokens through both, and
   prints per step `E2E step k pos=p n=m top1=t logprob=l`. Gate lines:
   * `E2E eval golden files=<n> bit_identical=1` — every MoE call's dumped
     input and output (prefill M = 16 through the HMX loop stand-in, the 8
     decode calls through the M=1 GEMV stand-in) equals
     `test/htp/host/golden/lfm2_moe_tiny/` byte for byte;
   * the same with `NNTR_MOE_HTP_M1_GEMV=0` (HMX loop at M = 1): `bit_identical=1`
     against the **same** golden — plan 85 §1's "flags 0 and M1_GEMV agree",
     now at model level;
   * `E2E tokens htp==cpu 8/8` and `E2E eval cpu min_snr_db=<x>` with x ≥ 60
     (a wiring fault reads 0–20 dB; the QS4CX-CPU vs WH-inproc difference is
     the activation quantizer's rounding, doc 46 §35). The SNR is printed with
     the run, never the verdict (rule 25);
   * `E2E eval self-test ok`: `htp_dump_eval.py` on an identical copy reports
     `bit_identical=1`, on a copy with one byte flipped `bit_identical=0
     snr_db=<finite>`, and on a length mismatch exits 2 — the issue's
     "perturbed dump" test, as a shell check inside the script.
   Runtime target ≤ 60 s after the first build (the tiny fixture's MoE is
   64 × 128 int4; the scalar stand-ins are microseconds per call).
3. **Standing host gates unchanged.** The default `build/` is untouched by the
   new option: `ninja -C build`, `*qs4cx*` cpu-backend gtests,
   `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` 6/6,
   `run_host_checks.sh` → `ALL CHECKS PASS` (its MoE check now links the
   lifted stand-in file, same results), `tools/htp_syntax_check.sh`,
   `clang-format-14`.
4. **Skel and app.** `test/htp/build.sh` (no DSP source changes; run for the
   `UNDEFINED SYMBOLS OK` line), `build_android.sh --htp`, `readelf -d`, md5s.
5. **Prefill gate: provable from the diff.** The only ARM-path change is the
   `NNTR_HTP_DUMP` hook in `invokeMoeLayer`, a `static const bool` env read
   that is false by default; no kernel, ring, pool or layout change. The E2E
   gates (prefill ≥ −5 % of A, text identical) are not exercised by this issue
   and stay with the next sitting.

**What this harness proves and what it cannot.** It serves rung 1 (host),
as a second target that needs the SDK's headers and `qaic` but no device and
no simulator. It proves: the op-table / session / opts wiring on the real ARM
code (validator errors surface as the real throws, `sendMoeOptsOnce`'s echo
check, handle binding and call order, `resume_at`, the close order), the
ARM-side marshalling (row_index / row_count / row_weight, staging pools,
`stagedMemcpy`), whole-token bit-identity against a golden (regression) and
against a device dump of the same token (later, the same `--eval`), and that
the M > 1 prefill path still produces the golden bytes. It cannot prove tok/s,
transport, DMA rate, DVFS, HVX/HMX arithmetic bits (the stand-ins are scalar,
`run_host_checks.sh`'s own caveat), or the 32-bit DSP address budget (rule 8;
the host is 64-bit).

## 2. Where it lives

Build variant (core files named per contract §6):

| file | change |
|---|---|
| `meson_options.txt:44` | new `option('htp-inproc', type: 'boolean', value: false)` next to `enable-htp` |
| `meson.build:280-316` | under `enable-htp`, when `htp-inproc`: skip `cc.find_library('sdkl')` (nothing in `htp_backend/` calls libsdkl — `htp_backend.h:20-28` records its removal); `hexkl_inc` = `test/htp/host/stub`, `test/htp/host/inproc`, `test/htp/host/replay_stub`, then the SDK `incs` / `incs/stddef` (`remote.h`, `AEEStdErr.h`, `HAP_*.h` are the real headers; the stand-ins define their functions). `extra_defines += -DNNTR_HTP_INPROC=1` beside `-DENABLE_HEXKL=1` |
| `nntrainer/tensor/htp_backend/meson.build:25-58` | when `htp-inproc`: instead of `generated/nntr_hvx_stub.c`, add the skel subset and the stand-ins to `nntrainer_sources` (§3.1 list); keep the `generate_stub.sh` requirement (the header `generated/nntr_hvx.h` is still the qaic one, so the prototypes are the IDL's, not a copy); add `link_args: -Wl,--no-undefined` for this variant |
| `nntrainer/tensor/meson.build:101-110` | unchanged (`hexagon_sdk_root/incs` already added) |

Stand-ins and the lift out of the MoE check:

| file | change |
|---|---|
| `test/htp/host/standin/hvx_scalar.{c,h}` (new; **moved**, not written) | the scalar stand-ins now inside `test/htp/host/moe_layer_host_check.c:27-490`: `hexkl_acc_layout_get`, `hexkl_micro_hmx_acc_clear_int32 / mm_u8i4 / acc_read_int32` with the `g_acc[64][32]` tile and `wh_value`, `hvx_gemm_u8i4_wh_col[_nopf] / _prefetch`, `hvx_quant_rows_u8_params`, `hvx_quant_pack_u8_ah[_mapped/_rows]`, `hvx_dequant_acc_tile_to_f32`, `hvx_dequant_swiglu_acc_tiles_to_f32`, `hvx_dq_*_worker`, `hvx_scale_add_rows_f32`, `hvx_copy_ah_block`, `hexkl_probe_on`. The check's instrumentation (`gemv_log_*`, the `pf_box` ring, `g_rows1_seen`) stays in the check behind two hook pointers the shared file calls when non-NULL (`hvx_scalar_hooks`), so the check's `HVX GEMV MUTANT` / lead assertions are unchanged. The `hexkl_dma_ring_*` stubs at `:219-243` are **not** moved: the inproc build uses the real `hexkl_dma_ring.c` on `replay_stub/hexagon_protos.h` (a descriptor lands whole when started), as `dma_replay_host_check` does |
| `test/htp/host/moe_layer_host_check.c`, `run_host_checks.sh:31-38` | include `standin/hvx_scalar.h`, link `standin/hvx_scalar.c` |
| `test/htp/host/stub/hexkl_micro.h` | grows from 3 prototypes to the set the skel subset calls (`hexkl_micro.h` of HexKL: `hw_init` (3-arg), `hmx_lock/unlock`, `hmx_config_size`, `hmx_setup_acc_read_int32`, `hmx_rm_to_wh_i4/i8`, `hmx_copy_32b_to_submatrix`, `hmx_mm_u8i8`) with the `HEXKL_HMX_*` constants it has now |
| `test/htp/host/inproc/hexkl_micro_standin.c` (new, ≈ 80 lines) | `hw_init` → one static 128-B-aligned **8 MiB** buffer (v79's VTCM size, so `hexkl_mm_u8i4_moe_layout` and the DMA plan compute the device's numbers); `hmx_config_size` → 2048; lock/unlock/setup → 0; `rm_to_wh_i4` → `whPack` from `nntrainer/tensor/htp_wh_layout.h:82` (one layout source, the same the quantizer uses); `rm_to_wh_i8`, `copy_32b_to_submatrix`, `mm_u8i8` → `AEE_EUNSUPPORTED` (no model path reaches them on the NPU config, LEDGER ⑰) |
| `test/htp/host/inproc/rpc_standin.c` (new, ≈ 120 lines) | `rpcmem_init/alloc/free/to_fd` (aligned malloc + an fd table), `fastrpc_mmap/munmap` (record the fd → va), `HAP_mmap_get/put` (look it up), `remote_session_control`, `remote_handle64_control` → `AEE_SUCCESS`, `HAP_power_set` → 0, `HAP_debug` (FARF sink: stderr when `NNTR_HTP_FARF=1`), `HAP_perf_get_pcycles` / `HAP_perf_get_qtimer_count` (`clock_gettime`), `qurt_hvx_get_units` → `0x0400` (4 units → 3 pool workers, the size class of `hvx_add_f32.c:113-118`). All `__attribute__((visibility("default")))`: `HtpRpcMemApi::get()` finds them with `dlsym(RTLD_DEFAULT, …)` (`htp_rpcmem.h:61-69`) |
| `test/htp/host/inproc/hvx_hexagon_protos.h` (new, 20 lines) | includes `replay_stub/hexagon_protos.h` and adds `Q6_Vsf_vadd_VsfVsf` and `HVX_UVector` for `hvx_add_f32.c:229` and `hvx_scale_add_f32.c`; replaced by plan 82's `hvx_emu/` when that lands (the same premise, rule 24) |
| `test/htp/host/inproc/hvx_swiglu_standin.c` (new, 15 lines) | `hvx_swiglu_inplace_f32` as a loop over `swiglu_det` (`nntrainer/tensor/swiglu_det.h`), the scalar spec of `hvx_swiglu_det.h`; needed because `hvx_swiglu_f32.c:40-44` includes the intrinsic header |

ARM-side hook, driver, comparator:

| file | change |
|---|---|
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1748` `invokeMoeLayer` | `NNTR_HTP_DUMP=<dir>`: after the call returns, write `<dir>/moe_<call#>_in.f32` (M × K) and `_out.f32` (M × N_out) and append one line to `<dir>/manifest.txt` (`call M K inter N_out flags n_experts row_count…`). `static const bool` env read, false by default. Plan 85's `invokeForward` calls the same helper, so #85's whole-token dumps come for free |
| `test/htp/host/htp_e2e_test.cpp` (new, ≈ 200 lines; the issue's `test/htp/htp_e2e_test`, under `host/` because it never runs on the phone) | built only in the inproc variant (`test/htp/host/meson.build`, new, included from `Applications/CausalLM/meson.build` after `causallm_dep`, since it needs `Lfm2MoeCausalLM` and `causallm_test_utils`). Args: `--fixture <dir> --prompt N --steps N --dump <dir> [--moe-engine cpu|htp]`. Builds the tiny config as `unittest_causallm_lfm2_moe.cpp:80-110` does with `max_position_embeddings` / `max_seq_len` raised to 32, loads the quantized fixture, prefill + greedy loop through `TinyCausalLMRunner` (`causallm_test_utils.h:293-350`), prints `E2E` lines, writes `logits_<step>.f32` into the dump dir |
| `tools/htp/htp_dump_eval.py` (new, ≈ 80 lines, numpy) | `htp_dump_eval.py <ref_dir> <got_dir>`: per file name in the manifest order, byte compare then SNR (`10·log10(Σref² / Σ(ref−got)²)`, inf when identical); prints one line per file and `E2E eval files=n bit_identical=0|1 min_snr_db=x first_diff=<file>`; exit 0 / 1 (differ) / 2 (shape or manifest mismatch) |
| `test/htp/host/run_inproc_e2e.sh` (new) | configure-if-missing (`meson setup build_htp_host -Denable-htp=true -Dhtp-inproc=true -Dhexkl-sdk-root=$HEXKL_ROOT -Dhexagon-sdk-root=$HEXAGON_SDK_ROOT -Denable-transformer=true …` the same extra flags as `build/`), `ninja`, fixture generation if missing (the gates skill's python line), the two quantizations, the four runs of §1.2, the eval self-test, the gate lines |
| `test/htp/host/golden/lfm2_moe_tiny/` (new, ≈ 40 KiB) | the dump set of the reference run, plus `manifest.txt` and the sha of the fixture generator / quantizer flags that produced it. Regenerated only deliberately, like `reference_logits.json` |

Consumers of a changed contract:

* **IDL and stubs: unchanged.** No method is added. The inproc build runs
  `generate_stub.sh` for `generated/nntr_hvx.h` only and links the skel `.c`
  files instead of `nntr_hvx_stub.c` (qaic emits the skel-side entry with the
  same C signature as the client stub, which is why the replacement is a link
  choice and not a shim). When #85 / #82 / #81 add methods, the header follows
  the IDL; the inproc list must gain `nntr_hvx_graph.c` + `hexkl_graph.c`
  (#85) or the link fails naming the symbol — no silent drift.
* **`HtpComputeOps`:** the dump hook only.
* **Quantizer format tag, loader check: unchanged.** The tiny fixture is
  quantized with the existing `--moe_dtype QS4CX_WH` (`quantize.cpp:677`).
* **`NNTR_HTP_PROFILE`, `tools/htp_fc_report.py`: unchanged.** The profile
  prints on the host too, but its `dsp=` / `transport` columns are the
  stand-ins' clock and mean nothing; the driver does not print them and the
  plan says so here so nobody reads them.
* **`run_host_checks.sh`:** one link line changes (the moved stand-ins);
  results identical. Textual overlap with #82 (`hvx_emu/`, `stub/AEEStdErr.h`)
  and #85 (`stub/AEEStdErr.h`, `run_host_checks.sh`): rebase onto whichever
  lands first.

## 3. Design

### 3.1 Chosen: the skel compiled into the process, the real ARM path as the driver

The skel entries are plain C functions with the IDL's signatures; FastRPC adds
only marshalling. Linking `hvx_add_f32.c` (open/close, session, pool) and
`nntr_hvx_mm_u8i4.c` (weights, arena, MoE layer, opts) plus their `hmx/` and
`hvx/` dependencies into the host `libnntrainer.so` makes every
`nntr_hvx_*` call in `htp_compute_ops.cpp` a direct call. Everything the DSP
needs from its platform is small and already half-built: the pthread
`stub/qurt.h` (worker pool), `replay_stub/` (DMA descriptors land at once),
`stub/hexkl_micro.h` (HMX as a 64 × 32 int32 tile), the scalar HVX stand-ins
of the MoE check. The inproc source list is:

```
test/htp/hvx_add_f32.c  test/htp/nntr_hvx_mm_u8i4.c
hmx/hexkl_mm_u8i4.c  hmx/hexkl_mm_u8i4_dma.c  hmx/hexkl_mm_u8i4_moe.c
hmx/hexkl_mm_u8i8_dma.c  hmx/hexkl_dma_ring.c  hmx/hexkl_dma_trace.c
hmx/hexkl_probe.c  hmx/hexkl_acc_tile.c
hvx/hvx_worker_pool.c  hvx/hvx_gather_ah_u8.c  hvx/hvx_scale_add_f32.c
test/htp/host/standin/hvx_scalar.c  test/htp/host/inproc/*.c
```

(`hvx_quant_u8.c`, `hvx_dequant_i32.c`, `hvx_gemm_u8i4_wh.c`, `hvx_swiglu_f32.c`
are the intrinsic-heavy files and are replaced by the stand-ins; `nntr_hvx_softmax.c`,
`nntr_hvx_attn.c`, `nntr_hvx_mm_u8i8.c`, `nntr_hvx_dma_probe.c` are gtest-only
entries the ARM never calls and stay out; `--no-undefined` is the proof.)

Why this and not a new binary: the driver the orchestrator asked for — "runs
the real ARM-side code paths (op table, session, opts)" — is, by definition,
the code in `htp_compute_ops.cpp` and the model's layers. Any harness that
does not go through them tests a copy. With the in-process build, the drivers
already exist: the tiny-fixture runner for the gate, and `nntr_causallm` (built
for the host, `Applications/CausalLM/meson.build:176`) for an on-demand run of
the real 8B model (§3.3). `htp_e2e_test.cpp` is thin: config, two runs, `E2E`
lines, dumps.

The three contract rules this respects: (i) no CPU fallback for `QS4CX_WH`
(contract §2): the inproc `HtpBackend` either opens or the model throws, as on
the device — `nntr_hvx_open` is the real one; (ii) the arena budget and DMA
hiding are not modelled and not claimed; (iii) `_det` before every quantizer
(doc 45 §3): the stand-ins are the scalar specs where a spec exists
(`swiglu_det`, `whPack`), and where none exists yet (the u8 activation
quantizer's rounding, the int32 → f32 epilogue) the stand-in is the MoE check's
current formula, so the harness's bit gate is a regression gate (golden) until
those get their `_det` — the honest scope, stated in the golden's README.

**Bit-identity levels, so the claim is exact:** (a) run-to-run and
GEMV-vs-HMX-loop on the host: bit-identical, gated; (b) host vs the committed
golden: bit-identical, gated; (c) host vs CPU control (QS4CX on the CPU path,
`lfm2_moe_layer.cpp:667`): SNR printed, top-1 gated on the tiny fixture;
(d) host vs device dump of the same token: `--eval` is ready for it, the
device dump is a later ride-along (§4 step 6), and any mismatch names the
first call.

### 3.2 Rejected: a standalone `htp_e2e_test` that opens the session and feeds an activation file (the issue's literal shape, hvx_impl's shape)

It would drive only `mm_u8i4_moe_layer` per call — `unittest_hvx_mm_u8i4`
already does that on the device, and the MoE check does it on the host — and
none of the ARM-side code the wiring issues change. It becomes the right shape
only when `forward(tokens → logits)` exists with every kind resident (plan 85
§3.3's end state); at that point it is ≈ 150 lines on top of this plan's
inproc build and the same `E2E` / `--eval` tooling, and can be filed then.
Also rejected: an arm64 build of the harness beside `unittest_hvx_mm_u8i4`
(issue text). On the phone the real driver is the app, every device number
comes from a filled handoff, and `NNTR_HTP_DUMP` gives the app the one thing
the harness would add there (dumps for `--eval`). This is a deliberate
deviation from the issue's acceptance list, and the reason is in the PR.

Also rejected: compiling `htp_compute_ops.cpp` outside meson against the
default `build/` (the `htp_syntax_check.sh` style, with `Engine::registerContext`
at runtime). It links only by ODR luck across `#ifdef ENABLE_HEXKL`, and a
second real meson variant costs one option and forty lines.

### 3.3 The real model on the host (on demand, not a gate)

`nntr_causallm` from `build_htp_host` with `nntr_config.json` `moe_engine: htp`
runs LFM2.5-8B-A1B through the same in-process path. Two facts bound it:
* it needs an **x86-packed** twin of the NPU model
  (`nntr_quantize_stream fp32/ -o q40-qs4cx-wh-x86/ --fc_dtype Q4_0 --embd_dtype Q4_0
  --lmhead_dtype Q4_0 --moe_dtype QS4CX_WH`, **no** `--isa ARM`): the `_ARM.bin`
  Q4_0 packing is wrong on x86 (contract §11's trap, inverted), while the WH
  bytes are ISA-free. 4.1 GB more under `/local/mnt/workspace/models`;
* the HMX stand-in is scalar over 64 × 32 × 32 per tile: ≈ 3 G MAC per prefill
  MoE call, ≈ 44 M per decode call on the GEMV stand-in. Prompt 16 + gen 8 is
  minutes; prompt 512 is hours. So its use is registration of the real 3.7 GiB
  arena through the real `ensureArena` / `newChunk` (`htp_compute_ops.cpp:2340`,
  `:2528`, 256 MiB chunks as malloc) and a few real-activation dumps for a
  later device `--eval`, not a benchmark. Peak RSS ≈ 5–6 GB against 30 GB.

## 4. Steps

1. **Lift the stand-ins.** `standin/hvx_scalar.{c,h}` + hooks; `moe_layer_host_check.c`
   and `run_host_checks.sh` follow. Gate: `run_host_checks.sh` → `ALL CHECKS PASS`,
   `HVX GEMV MUTANT CAUGHT` ×2, same `bad=0` lines as before.
2. **Inproc build variant.** `meson_options.txt`, `meson.build`,
   `htp_backend/meson.build`, `inproc/*.c`, `stub/hexkl_micro.h`. Gate:
   `ninja -C build_htp_host` links with `--no-undefined`; `nm -D
   build_htp_host/nntrainer/libnntrainer.so | grep ' T nntr_hvx_'` lists the
   entries; `ninja -C build` unchanged (the option is off there).
3. **Dump hook + driver + comparator + golden.** `invokeMoeLayer` hook,
   `htp_e2e_test.cpp`, `htp_dump_eval.py`, `run_inproc_e2e.sh`, first golden
   from the `QS4CX_WH` run, `golden/README`. Gate: the five `E2E` gate lines of
   §1.2; `tools/htp_syntax_check.sh`; `clang-format-14`.
4. **Host gates.** `ninja -C build`, `*qs4cx*`, `*Lfm2Moe*` 6/6, `run_host_checks.sh`.
5. **Skel + app.** `test/htp/build.sh` (`UNDEFINED SYMBOLS OK`),
   `build_android.sh --htp`, `readelf -d`, md5s; the prefill-gate diff check
   (§1.5). PR into `htp_moe`, `state:review`. **The issue ends here. No
   device measurement is needed or claimed; there is no handoff.**
6. **Later, ride-along (not this issue):** in the next sitting that has room,
   one decode run of the app with `NNTR_HTP_DUMP=/data/local/tmp/dump`, prompt
   16, G = 4, on the `q40-qs4cx-wh` model; pull the dir; on the workstation
   `htp_dump_eval.py <host dump of the same prompt on q40-qs4cx-wh-x86> <device dump>`.
   Never in a tok/s cell (16 KiB of file I/O per call). The result is level
   (d) of §3.1 and the first whole-token device-vs-spec reading. Any
   `bit_identical=0` names the first call and its M, which is where a `_det`
   is missing (rule 24's premise or the u8 quantizer).

## 5. Risks

* **Stand-in arithmetic is not silicon.** Level (b) can pass while the device
  differs. Stated in every gate line's name (`golden`, not `device`), and the
  ride-along of step 6 is the check. The two places without a `_det` today
  (u8 activation quantizer rounding in `hvx_quant_u8.c`, the int32 → f32
  epilogue order in `hvx_dequant_i32.c`) are named in `golden/README` so a
  device mismatch there is expected, not a surprise.
* **`dlsym(RTLD_DEFAULT)` and symbol visibility.** If `libnntrainer.so` is
  built with hidden visibility, `HtpRpcMemApi::get()` finds no `rpcmem_alloc`
  and the arena path silently stays off (`htp_rpcmem.h:140-141`). The
  stand-ins carry default visibility and step 2's gate greps `nm -D` for
  `rpcmem_alloc` too.
* **The 32-bit DSP address budget is invisible on the host** (rule 8). A
  registration that would fail on the device succeeds here. Nothing in this
  harness may be read as a residency result.
* **Scheduler, not QuRT.** The pool runs on pthreads; a race that depends on
  QuRT's priorities can pass here (the MoE check's caveat). The DMA stand-in
  completes at issue, which surfaces a read-before-landed bug as a wrong
  result rather than hiding it — the useful direction.
* **Tiny shapes.** K = 64, N = 128 is one tile row; the multi-block, tail and
  empty-expert cases stay with `moe_layer_host_check` (M = 37, five experts).
  The prompt of 16 with top-2 of 4 gives up to 16 rows per expert, above
  `HVX_GEMM_U8I4_MAX_ROWS`, so the HMX loop runs at prefill.
* **x86 QS4CX CPU path.** The CPU control assumes the QS4CX MoE path runs on
  x86 (`*qs4cx*` gtests do). If it does not, the control falls back to the
  FP32 fixture and level (c) becomes a looser SNR; step 3 records which.
* **Textual overlap** with #85 / #82 / #81 (`run_host_checks.sh`,
  `stub/AEEStdErr.h`, `hvx_emu/`); and #85's PR must add its two files to the
  inproc list — the link error is the reminder. Sequencing: land this before
  plan 85 step 4 (its ARM side) so `NNTR_HTP_FORWARD=1` has its host gate.
* **Build time and disk.** A second meson dir, minutes on first build, 1–2 GB.
  Not part of `run_host_checks.sh` (38 s, SDK-free); a separate target with
  its own line in the gates skill.
* **Baseline.** No host build was run in this planning session (read-only
  worktree, no `build/`); plan 82 recorded `run_host_checks.sh` green at
  `485435ce`. The implementer runs it first.

## 6. Docs to update

* **BENCHMARK.md:** nothing (no number). The step-6 ride-along later adds one
  line under its sitting (`htp_dump_eval` verdict, first differing call if any).
* **LEDGER.md:** §4 lift 4 → done: lifted the `E2E` line format and the
  `--eval` verdict; not lifted `HexagonRunner` / `RpcmemBuffer` /
  `run_e2e_test.sh` / `summ_farf_prof.py` / the static-map probe, and why
  (§0). §3a tooling note: `build_htp_host` + `run_inproc_e2e.sh`,
  `NNTR_HTP_DUMP`, `htp_dump_eval.py`, the x86-packed model twin, "profile
  columns on the host are meaningless". ⑨: the wiring issues get a host E2E
  gate; #85's ARM step should follow this PR.
* **`.claude/skills/hexagon-gates`** (agent-system commit, separate): rung 1
  gains `bash test/htp/host/run_inproc_e2e.sh` with its pass lines and the
  "needs the SDK sourced, no device" note. **`hexagon-handoff`:** the md5 gate
  line (`md5sum -c staged/md5.txt` before staging, `adb shell md5sum` compared
  by the user) and the `NNTR_HTP_DUMP` ride-along recipe.
* **Priority:** stays p2 (moves no row). One line for p1 would be: it is the
  only whole-model gate #85's `NNTR_HTP_FORWARD=1` can get before a sitting.
