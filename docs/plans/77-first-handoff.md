# 77 — First handoff: control runs on our unit + doc 48 A/B/C + two-reader DDR probe

Issue: dlwlzzero/nntrainer#77 (part of #76, contract
`docs/plans/0001-htp-moe-decode-agent-system.md` §4.3). Branch
`htp/77-first-handoff` from `htp_moe` @ `8fd03100`. Nothing in the model
path changes; this plan adds one report line to the app, one DMA probe
entry to the skel, one device gtest, one host check, and the handoff.
Blocked on #78 (the `q40` and `q40-qs4cx-wh` model dirs) only for the
final artifact table; everything else can be built and gated now.

## 1. Goal and gate

Acceptance criterion (issue): "the handoff runs top to bottom in one
sitting (≤ 90 min); BENCHMARK.md's 'now' cells are replaced with unit
`R3CY10WM83Y` numbers; LEDGER.md items ①–④ get their verdicts; the planner
can order ⑤ ⑥ ⑦ from them."

Made measurable:

| deliverable | what counts as done |
|---|---|
| `docs/measurements/77-first-handoff.md` | every cell of the tables in §4 step 8 filled from one sitting; estimated minutes at the top ≤ 90; every artifact row has an md5 |
| BENCHMARK.md Goals "now" | NPU decode tok/s at gen 64 / 512 / 1024 (all, and last 64) and CPU decode tok/s at the same lengths, prompt 512, from the `generation:` and `generation(last 64):` lines of the A runs; NPU prefill tok/s at prompt 512 |
| LEDGER ① | the `[PROFILE]` per-type decode ms/token table for `output_of_causallm`, `fully_connected`, `mha_core`, `lfm2_moe`, and the rest, computed as (sum at G=512 − sum at G=64) / 448 |
| LEDGER ② | transport µs/call for the `M==1` row at `NNTR_HTP_PROFILE=2` and `=3`; verdict "wake/clock" if the =3 number is ≤ 300 µs, "marshalling" if it stays within ±15 % of the =2 number |
| LEDGER ③ | a 3 shapes × 4 worker counts × 2 skels table of GB/s from the DMA probe; verdict names which axis (shape / workers / vote) moves the rate from ≈ 18 toward ≈ 38 GB/s |
| LEDGER ④ | CPU-alone, DSP-alone, and concurrent GB/s (aggregate = sum of the two concurrent numbers); verdict "split can pay" if aggregate > 45 GB/s (contract §3.2 Q11 threshold) |

Prefill gate (contract §1) does not apply to this handoff (no variant other
than A), but every A run records prefill so the next handoff has its
denominator. Accuracy gate: within a model, run 1 and run 2 texts are
identical (greedy decoding); across the two models the texts differ by
construction (different expert quantisation) and both are recorded.
`NNTR_L2_DIFF` is not applicable to `QS4CX_WH` (no CPU path, LEDGER rule 6).

## 2. Where it lives

All `path:line` references are against `htp_moe` @ `8fd03100`.

### 2.1 Prompt, generation length, printed lines (no change except one line)

* Prompt input: `Applications/CausalLM/main.cpp:424-425` takes the prompt
  from `argv[2]` when present, else `nntr_config.json`'s `chat_input`
  (through the chat template, `:428-431`) or `sample_input` (`:438`).
  There is no file argument. LFM2 encodes
  `system_prompt + prompt + tail_prompt`
  (`Applications/CausalLM/models/lfm2/lfm2_causallm.cpp:487-491`); the
  authored `nntr_config.json` has no `system_prompt` key, and the LFM2
  tokenizer adds no BOS in either `add_special_tokens` mode (checked on
  the workstation with `tokenizers` 0.22.2 against `hf/tokenizer.json`:
  `"Hello world."` → 3 ids both ways). The token count is clamped to
  `INIT_SEQ_LEN` (`:493-496`; `init_seq_len` = 512 in
  `/local/mnt/workspace/models/lfm2.5-8b-a1b/fp32/nntr_config.json`), and
  `input_len + 1 + NUM_TO_GENERATE ≤ MAX_SEQ_LEN` is enforced at
  `:555-563` (`max_seq_len` = 2048 ≥ 512 + 1 + 1024).
* Config keys: `num_to_generate`, `init_seq_len`, `max_seq_len` are read
  once at construction (`Applications/CausalLM/models/transformer.cpp:135-137`),
  so G is set per run by editing the model dir's `nntr_config.json` on the
  device before launch.
* Printed lines (`lfm2_causallm.cpp:757-767`):
  `prefill: <input_len> tokens, <ms> ms, <tps> TPS`,
  `generation: <generation_cnt> tokens, <ms> ms, <tps> TPS`,
  `total: <ms> ms`, `peak memory: <KB> KB`. `generation_cnt` is the number
  actually generated; the loop breaks on EOS (`:731-741`).
* Determinism: `main.cpp:458` reads `do_sample` from
  `<model>/generation_config.json` (loaded at `:374-377` if the file
  exists); `nntr_quantize_stream` copies HF's `generation_config.json`
  into the output dir (`Applications/CausalLM/quantize_stream.cpp:1245-1251`),
  and HF's file says `"do_sample": true`. Deleting the file is not an
  option: `Applications/CausalLM/models/causal_lm.cpp:112-119` reads
  `generation_cfg["eos_token_id"]` unguarded and `.get<unsigned int>()` on
  null throws. The file must be edited to `"do_sample": false`.
* Exact generation length: EOS ends the loop early. `bad_word_ids` in
  `nntr_config.json` (`causal_lm.cpp:86-87`) sets those logits to `-INFINITY`
  before argmax (`Applications/CausalLM/llm_util.cpp:68-72`, applied at
  `causal_lm.cpp:316-317`), so `"bad_word_ids": [124900]` (LFM2's
  `eos_token_id`, `hf/generation_config.json`) guarantees `generation_cnt
  == G` in both models.
* "Last 64 tokens" decode tok/s: **not derivable from existing output.**
  The decode loop keeps one start/finish pair
  (`lfm2_causallm.cpp:683` / `:744`) and prints tokens as text only
  (`causal_lm.cpp:268-270`). One report-only change is needed (§3.1).

### 2.2 Measurement A (`--profile` build)

* `Applications/CausalLM/build_android.sh:14-17` sets `USE_PROFILE`;
  `:159-162` passes `-Denable-profile=true` to `tools/package_android.sh`
  (→ `-DPROFILE=1`, `meson.build:528-529`); `:265` passes
  `CAUSALLM_PROFILE=$USE_PROFILE` to ndk-build. **There is no separate
  builddir**: `:181-186` removes `$NNTRAINER_ROOT/builddir` unless
  `--cache`, and `--cache` would reuse the non-profile one. The profile
  build is therefore a second full build (≈ 10 min) whose outputs are
  moved aside by hand (§4 step 5).
* Output: `main.cpp:460-483` wraps `model->run` in
  `PROFILE_BEGIN/END(GenericProfileListener)` and prints
  `[PROFILE] per-layer-type totals over the whole run ...` followed by
  the listener's table (`nntrainer/utils/profiler.cpp:142-183`: columns
  `key avg min max sum pct`, µs, one row per layer type — the keys are
  layer type names because `nntrainer/graph/network_graph.cpp:1192-1195`
  registers `profile_keys[lnode->getType()]`). The lm_head type is
  `output_of_causallm` (`Applications/CausalLM/quantize.cpp:491, 590`).
  Per-token decode cost of a type = (sum at G=512 − sum at G=64) / 448:
  the prefill contribution cancels exactly, which doc 49 §6's
  "sum − max" only approximates for types with many nodes.

### 2.3 Measurement B (`NNTR_HTP_PROFILE=3`)

* `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:380-381` reads the
  level; `:1479-1500` runs the MoE layer call `reps = 5` times at level ≥ 3
  and keeps the fastest (`elapsed` and its `stage_us` together), so the
  `host` column is min-of-5 and `transport = host − dsp` of that same
  call. Level ≥ 2 routes through `nntr_hvx_mm_u8i4_moe_layer_timed`
  (`:1487`), identical for =2 and =3.
* Report (`:395-517`, stderr at exit): header
  `[HTP-PROFILE] level=<n> qos_mode=<q> ...` (`:397`, `qos_mode` must match
  between the two runs), the per-shape row
  `[HTP-PROFILE]   K=2048 N=2048 M==1 calls=… rows=… host=… ms (… us/call)  dsp=… us/call (…%) transport=… us/call  [quant … gather … requant … swiglu … dequant … acc … drain …+… push … scatter … alloc … stage … mm … | rest<=… blocks=…]`
  (`:436-476`), and the DMA line
  `[HTP-PROFILE]     weight DMA: … KB/call, first … KB took … us = … GB/s; averaged over the call … GB/s`
  (`:499-506`). The fields that answer B are `transport` and `host` of the
  `M==1` row at level 2 vs level 3; `first … GB/s` is C's in-situ
  reference number.

### 2.4 Measurement C (DMA probe) — what exists, what is added

* Existing: `arena_probe` (`test/htp/nntr_hvx.idl:283-284`,
  `test/htp/nntr_hvx_mm_u8i4.c:198-326`) maps an fd and pushes **one** 2D
  transfer of `dma_bytes` with `row_size` = largest power of two ≤ 4096
  dividing it (`:288-296`) through the shared ring, then drains. The
  gtest `ArenaMapAndDma` (`test/unittest/unittest_hvx_mm_u8i4.cpp:1637-1767`)
  shows the whole host recipe: `rpcmem_alloc(25, 1, bytes)` via dlsym
  (`:1642-1650`), pattern fill, `rpcmem_to_fd`, `fastrpc_mmap(CDSP_DOMAIN_ID,
  fd, buf, 0, bytes, FASTRPC_MAP_FD)` (`:1686-1700`), and prints
  `U8I4_FIELD path=arena field=<k> value=<v>` lines. `MoeLayerFromArenaMatchesHeap`
  (`:2117`) uses `nntr_hvx_arena_attach` (`nntr_hvx_mm_u8i4.c:336-374`) the
  way the model does. This probe cannot answer C: one shape, one engine,
  and the vote is compiled in.
* The ring is one global chain: `nntrainer/tensor/htp_backend/hmx/hexkl_dma_ring.c:62-65`
  (`g_ring`, `g_tail`, `g_started`), `:117-122` (first push `dmstart`,
  later pushes `dmlink`). The descriptor struct and the two asm helpers
  (`:24-56`) are file-private. A per-worker `dmstart` needs its own
  descriptors, so the struct and helpers move to `hexkl_dma_ring.h`
  (no behaviour change; the `.c` keeps using them).
* The weight-chunk geometry the model uses, for shape (ii):
  `hexkl_mm_u8i4_moe.c:133-141` — `row = cn*512`, `stride = n_col*512`,
  `nrows = k_tiles`; with `cn = 32`, `n_col = 112` (gate_up N = 3584),
  `k_tiles = 64` (K = 2048) that is 16 KB × 64 rows @ 56 KB stride, 1 MiB
  per descriptor. Shape (iii) is `cn = 16`: 8 KB × 64 @ 56 KB.
* The bus vote is unconditional in `nntr_hvx_open`
  (`test/htp/hvx_add_f32.c:126-173`; the mips_bw block is `:164-172`,
  40 GB/s, 100 %). `test/htp/build.sh:91-107` has no hook for extra
  flags; one is added (`${HEX_EXTRA_CFLAGS:-}` on the hexagon-clang line)
  and the mips_bw block gets `#ifndef NNTR_HVX_NO_BUS_VOTE`.
* Worker threads: the session pool `s->quant_pool` is created with
  `n_hvx − 1` workers (`hvx_add_f32.c:113-120`);
  `hvx_worker_pool_run(pool, func, ctx, n_units)` runs `func(n, i, ctx)`
  for `i < n = min(n_units, workers+1)`, `i == 0` on the caller
  (`nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.h:44-54`). VTCM
  (`s->vtcm_base`, `s->vtcm_size`, `test/htp/nntr_hvx_session.h:52-53`)
  is split among workers as destination.
* Device gtest build: `test/jni/Android.mk:928-950` is the
  `unittest_hvx_mm_u8i4` module (sources: the `.cpp` and
  `../htp/generated/nntr_hvx_stub.c`, links `-lcdsprpc`); a second module
  is a copy of that block. Device dir convention:
  `/data/local/tmp/htp_u8i4_layer_test` (`test/htp/run_u8i4_layer_on_device.sh:26`).
* Host check wiring: `test/htp/host/run_host_checks.sh:27-46` compiles
  each check against `test/htp/host/stub/` (`AEEStdErr.h HAP_perf.h
  hexkl_micro.h qurt.h`) and runs it; a third block is appended.
* IDL: new entries go at the **end** of `interface nntr_hvx`
  (`test/htp/nntr_hvx.idl`, after `mem_probe_touch`, `:303-304`) so the
  existing method indices are untouched. Both the skel and the stub are
  regenerated anyway (`test/htp/build.sh:72-75`,
  `nntrainer/tensor/htp_backend/generate_stub.sh` via `build_android.sh:167`).

### 2.5 Two-reader DDR probe

Same gtest binary, one more test (§3.5). CPU side is plain `std::thread`
+ NEON in the gtest; DSP side is the C probe entry streaming
`kArenaChunkMax`-sized rpcmem chunks (256 MiB,
`htp_compute_ops.cpp:2092`) — the same mapping path the model uses.

### 2.6 ABI note

`test/htp/nntr_hvx.idl` changes (two appended entries), so the skel and
every host consumer of the stub are rebuilt from the branch head:
`libnntr_hvx_skel.so`, `nntrainer_causallm`/`libnntrainer.so` (stub
compiled into `libnntrainer.so`), and all four device gtests
(`unittest_hvx_{mm_u8i4,softmax,attn,fc}` compile the stub). The
`nntr_hvx_session` struct and `HTP_MOE_N_STAGES` do not change. There is
no `nntr_htp_common.h` ABI version in this tree.

## 3. Design

### 3.1 The one app change: `generation(last 64)` line

In `Lfm2CausalLM::run_with_embeddings`, keep a
`std::vector<std::chrono::high_resolution_clock::time_point>` of one
timestamp per generated token (push after `++generation_cnt`,
`lfm2_causallm.cpp:710`; ≤ 1024 entries) and, in the `log_output` block
(`:756-768`), print
`generation(last 64): 64 tokens, <ms> ms, <tps> TPS` when
`generation_cnt ≥ 64`, from `ts[n−1] − ts[n−65]`. Same clock and unit as
the existing lines; nothing in the forward path is touched.

Rejected: reading token timestamps off the adb pipe on the workstation
(`registerOutputs` does `std::cout.flush()` per token,
`causal_lm.cpp:268-270`). It works in principle but adds USB/pty latency
of the same order as one decode step (20–50 ms), holds tokens back on
punctuation (`:262-266`), and makes the number depend on the host. A line
printed by the app is what BENCHMARK.md can cite.

Consequence for "variant A = PR head unchanged": A's `nntrainer_causallm`
is the PR head plus this report-only commit (and the regenerated stub).
The handoff states this and lists both the PR-head md5
(`53814a39…`, BENCHMARK.md) and A's md5; the forward path is identical, and
the 6/6 `*Lfm2Moe*` gtests prove the outputs are.

### 3.2 Determinism and length: config edits, not code

Both model dirs get, on the device: `"do_sample": false` in
`generation_config.json` (edited with `sed`, never deleted — §2.1) and
`"bad_word_ids": [124900]` in `nntr_config.json`. G is set with `sed` on
`num_to_generate` before each run. The expected lines are then exactly
`prefill: 512 tokens, ...` and `generation: <G> tokens, ...`; a different
count is a failed step, not a data point. Rejected: three config copies
per model (more state on the device to get wrong) and a `--max_new_tokens`
flag (the app has none, `main.cpp:356-359`).

### 3.3 The 512-token prompt

A committed file `docs/measurements/77-prompt512.txt`: plain ASCII prose,
no `"`, `` ` ``, `$` or `\` (it is passed through `adb shell "..."` as
`"$(cat prompt512.txt)"`), a single paragraph ending in an instruction to
continue the text (so greedy decoding does not stop early even without
the EOS ban). Its length is tuned on the workstation with the HF
tokenizer until `len(Tokenizer.from_file(hf/tokenizer.json).encode(text).ids)
== 512`; the device prints `prefill: 512 tokens` because the app
tokenizes the same string with the same `tokenizer.json` and no BOS
(§2.1). Rejected: `sample_input` in `nntr_config.json` (would have to be
edited into both model dirs and differs from what the PR used) and
`chat_input` (adds template tokens the count would have to compensate).

### 3.4 Measurement C: a self-contained DMA probe entry

New skel source `test/htp/nntr_hvx_dma_probe.c` (added to `SRCS`,
`build.sh:77`) implementing

```
AEEResult dma_probe(in uint32 arena, in uint32 src_off, in uint32 bytes,
                    in uint32 row_size, in uint32 nrows, in uint32 src_stride,
                    in uint32 workers, in uint32 passes,
                    rout sequence<uint32> res);
```

`res = [us, bytes_lo, bytes_hi, checksum, workers_used, vote_state,
n_descriptors, max_desc_bytes]`. The pure planning function
`nntr_dma_probe_plan(bytes, row_size, nrows, src_stride, workers, vtcm_per_worker, out[])`
splits the source range into descriptors of `row_size × nrows` at
`src_stride` (contiguous when `src_stride == row_size`), assigns them
round-robin to workers, and bounds each worker's destination to its VTCM
slice (`vtcm_size / workers`, descriptors ≤ 1 MiB). Each worker, in
`hvx_worker_pool_run(s->quant_pool, …, workers)`, walks its list:
`dmstart` on its own 128-byte-aligned descriptor, spin on `done` with
`dmpoll` (the same loop as `hexkl_dma_ring.c:67-75`), next; repeated
`passes` times. The caller times the whole pool run with the qtimer
(`hexkl_probe_now()`), sums a 64-byte-strided checksum of its own VTCM
slice (so the transfer cannot be elided, as `arena_probe` does at
`nntr_hvx_mm_u8i4.c:298-306`), and reports `vote_state` as the compile-time
constant (1 with the vote, 0 under `-DNNTR_HVX_NO_BUS_VOTE`) so every log
line names the skel it came from.

Shapes the gtest drives (source = a 256 MiB rpcmem chunk attached with
`nntr_hvx_arena_attach`, filled with a pattern):

| shape | row_size | nrows | src_stride | note |
|---|---|---|---|---|
| (i) linear 1 MiB | 4096 | 256 | 4096 | contiguous, same descriptor rule as `arena_probe` |
| (i') linear 1 MiB, one row | 1048576 | 1 | 1048576 | tests whether a single long row differs; skipped if the engine rejects it (24-bit `row_size`) |
| (ii) weight chunk | 16384 | 64 | 57344 | `hexkl_mm_u8i4_moe.c:133-141` with cn=32, n_col=112 |
| (iii) half chunk | 8192 | 64 | 57344 | cn=16 |

× workers 1, 2, 3, 4 × skel {vote on, vote off}. `passes` chosen so each
cell moves ≥ 512 MiB (≈ 15–30 ms at 18–38 GB/s); each cell is run 3 times
and the gtest prints the max. One grep-able line per cell:
`DMA_PROBE shape=<i|i1|ii|iii> workers=<n> vote=<0|1> gbs=<x.x> us=<n> bytes=<n> checksum_ok=<y|n>`.

Rejected: extending `arena_probe` with more parameters. It runs on the
shared ring (one `dmstart`, dmlinked chain), so it can never show whether
per-thread engines add up, and its 10-slot `res` contract is already
consumed by `ArenaMapAndDma`. Also rejected: a runtime vote toggle inside
the probe (`bwBytePerSec = 0`) — HAP votes are per-PD and aggregate, so an
"off" reading would still sit on the session's open-time vote; a second
skel is the only clean off (doc 48 §5 C says the same).

Bit-identity / kernel rules (gates skill "kernel review list"): the probe
adds no arithmetic and no DSP heap allocation (descriptors live in a
static array sized for the worst-case plan; VTCM is the session's own).
The `hexkl_dma_ring.h` refactor is a pure move; `run_host_checks.sh`
already compiles the MoE kernel against the stubbed ring, which catches a
broken move on the host.

### 3.5 Two-reader DDR probe: a mode of the same gtest

`TEST_F(HvxDmaProbe, TwoReaderDdr)` in the new
`test/unittest/unittest_hvx_dma_probe.cpp`:

1. CPU alone: 8 `std::thread`s, each streams its 64 MiB slice of a
   512 MiB `malloc`ed, pre-faulted, pattern-filled buffer with
   `vld1q_u8` ×4 unrolled + `veorq_u8` accumulation, looping until ≥ 2 s
   have elapsed; GB/s = bytes / wall (`steady_clock`). The XOR result is
   printed so the loads are not dead.
2. DSP alone: `dma_probe` on two 256 MiB rpcmem chunks (attached as two
   arenas), shape (i), workers = the best count from §3.4 (the gtest
   picks it from its own DMA_PROBE lines run just before), `passes`
   sized for ≥ 2 s; GB/s from `res[0..2]`.
3. Both: start the CPU threads spinning on an atomic `go`, set
   `t0 = now`, set `go`, issue the (blocking) DSP call, on return set
   `stop`, join; CPU GB/s over `[t0, stop]`, DSP GB/s from its own `res`,
   aggregate = sum. The DSP side's bytes are known exactly; the CPU side
   counts completed 64 MiB passes only inside the window.

Line: `DDR_TWO_READER cpu_alone=<gbs> dsp_alone=<gbs> cpu_with=<gbs> dsp_with=<gbs> aggregate=<gbs> cpu_threads=8 dsp_workers=<n>`.

Rejected: a standalone ARM binary plus a skel entry. It would need its
own Android.mk module, its own rpcmem/fastrpc_mmap boilerplate, and its
own push/run lines in the handoff; the gtest already has the session and
the mapping recipe, and one binary keeps the handoff to one directory.

### 3.6 Measurements A and B need no code

A is the second build (§2.2) run at G=64 and G=512 on both models, read as
differences. B is two runs of A's binary with `NNTR_HTP_PROFILE=2` (plus
`NNTR_M0_PROFILE=1`, `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:320, 432`)
and `NNTR_HTP_PROFILE=3` at G=64 on the NPU model.

## 4. Steps

Each step ends in a gate from `.claude/skills/hexagon-gates` (rungs 0–3;
there is no simulator rung in this tree). `source tools/htp/env.sh` first
in every shell. Branch `htp/77-first-handoff`.

1. **Report line** (commit 1, `[CausalLM] Print decode tok/s over the
   last 64 generated tokens`): §3.1 in `lfm2_causallm.cpp`. Gate: rung 0;
   rung 1 (`ninja -C build`, `unittest_causallm_models --gtest_filter='*Lfm2Moe*'`
   6/6 passed, none skipped; `run_host_checks.sh` and the syntax check
   unchanged). `git diff --stat` shows only that file.
2. **Skel flag hook + vote-off variant** (commit 2, `[htp] Let build.sh
   take HEX_EXTRA_CFLAGS and make the bus vote a compile-time option`):
   `build.sh:91-107` gets `${HEX_EXTRA_CFLAGS:-}`; `hvx_add_f32.c:164-172`
   gets `#ifndef NNTR_HVX_NO_BUS_VOTE`. Gate: rung 2 twice —
   `./test/htp/build.sh` → copy to `libnntr_hvx_skel.A.so`;
   `HEX_EXTRA_CFLAGS=-DNNTR_HVX_NO_BUS_VOTE ./test/htp/build.sh` → copy to
   `libnntr_hvx_skel.novote.so`; both `-Wall -Werror` clean, md5s differ
   and are recorded.
3. **DMA probe: DSP side** (commit 3, `[htp] Add a per-worker dmstart
   DMA probe entry for the arena bandwidth question`): move the
   descriptor struct and asm helpers to `hexkl_dma_ring.h`; append
   `dma_probe` to the IDL; `nntr_hvx_dma_probe.c` with
   `nntr_dma_probe_plan()` in a small header so the host check can
   compile it. Host check `test/htp/host/dma_probe_host_check.c`
   (disjoint coverage, byte totals, descriptor bounds, worker balance
   for every shape × workers in §3.4; prints `DMA PROBE PLAN OK`),
   appended to `run_host_checks.sh`. Gate: rung 1 (`ALL CHECKS PASS`,
   `WORKER POOL LANES OK`, `DMA PROBE PLAN OK`), rung 2 (both skel
   variants rebuilt; record the new md5s — these replace step 2's).
4. **DMA probe: gtest** (commit 4, `[test] Device gtest for the DMA probe
   shapes and the two-reader DDR probe`): `unittest_hvx_dma_probe.cpp`
   with the fixture (session open, rpcmem chunk ×2, `arena_attach`),
   `DmaProbeShapes` (§3.4 table, prints `DMA_PROBE` lines) and
   `TwoReaderDdr` (§3.5); Android.mk module `unittest_hvx_dma_probe`
   copied from `:928-950`. Gate: rung 3 — `generate_stub.sh`, then
   `build_android.sh --htp` (fresh builddir → the gates skill's `prefix`
   workaround, then `--cache`), `readelf -d` shows `libsdkl.so` and
   `libcdsprpc.so`, and ndk-build of all five gtests (the four existing
   ones must relink against the regenerated stub). md5s of
   `nntrainer_causallm`, `libcausallm_core.so`, `obj/local/.../libnntrainer.so`,
   `libccapi-nntrainer.so`, `unittest_hvx_dma_probe` recorded. Then
   **move the TPS outputs aside**: `mv builddir builddir.tps`, copy
   `Applications/CausalLM/jni/libs/arm64-v8a` and `jni/obj/local/arm64-v8a`
   to `/local/mnt/workspace/htp_moe/77/tps/`.
5. **Profile build** (no commit): `./build_android.sh --htp --profile`
   (fresh builddir, same `prefix` workaround), copy its four app files to
   `/local/mnt/workspace/htp_moe/77/profile/`, `mv builddir builddir.profile`,
   `mv builddir.tps builddir`, restore `jni/libs` from `tps/`. Gate: the
   profile `nntrainer_causallm` md5 differs from A's; `readelf -d` on the
   profile `libnntrainer.so` still lists `libsdkl.so`/`libcdsprpc.so`;
   `strings profile/nntrainer_causallm | grep -c 'per-layer-type totals'`
   is 1 and on A's binary is 0.
6. **Prompt file** (commit 5, `[docs] Add the 512-token handoff prompt
   and the first-handoff measurement document`): `77-prompt512.txt`
   tuned to 512 ids (§3.3; the check snippet goes into the handoff so the
   user can re-run it); the handoff document (step 8). Gate: the python
   check prints 512; `grep -c '["`$\\]' 77-prompt512.txt` is 0.
7. **Artifacts from #78**: once `q40/` and `q40-qs4cx-wh/` exist, fill
   their `*_ARM.bin` md5s and confirm each dir has
   `generation_config.json` (to be edited on device) and that the WH
   dir's `nntr_config.json` has `"moe_engine": "htp"`,
   `"moe_htp_layers": ""`, and the CPU dir has neither key. This is the
   only step that waits on #78; the handoff is written with those cells
   marked "from #78" until then.
8. **The handoff** `docs/measurements/77-first-handoff.md`
   (`hexagon-handoff` template), set the issue to `state:needs-measurement`.
   Its content, in run order, with the phone plugged in, screen off,
   battery ≥ 60 %, noted as warm/cool:

   *Install (≈ 10 min):* `install_android.sh --model=` is not used for
   models (it pushes binaries only); the handoff gives explicit
   `adb push` lines: `tps/` files into `/data/local/tmp/nntrainer/causallm/`,
   `profile/` files into `/data/local/tmp/nntrainer/causallm_profile/`,
   both model dirs into `/data/local/tmp/nntrainer/causallm/models/`,
   `libnntr_hvx_skel.A.so` → `causallm/libnntr_hvx_skel.so` and
   `causallm_profile/libnntr_hvx_skel.so`; the gtest, `libc++_shared.so`,
   and both skels (A as `libnntr_hvx_skel.so`, novote under its own name)
   into `/data/local/tmp/htp_u8i4_layer_test/`. Never `builddir/.../libcdsprpc.so`.
   Then the config edits (§3.2) with `sed -i` and a `grep` echo of the
   three keys per model, and `md5sum` on the device of the skel and
   `nntrainer_causallm` in each directory.

   *Control runs, A (≈ 25 min):* for model in `q40`, `q40-qs4cx-wh`; for
   G in 64, 512, 1024; twice:
   `adb shell "cd /data/local/tmp/nntrainer/causallm && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": G/' models/<m>/nntr_config.json && NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/<m> \"\$(cat prompt512.txt)\"" 2>&1 | tee logs/A_<m>_G<G>_r<n>.log`.
   Expected lines: `prefill: 512 tokens, … TPS`, `generation: G tokens, …
   TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory: … KB`.
   CPU first (no registration, thermal gate: CPU decode < 30 tok/s means
   throttling, doc 49 §6 — cool down and redo), then NPU. 60 s pause
   before each NPU G=1024 run.

   *Profiles B (≈ 4 min):* NPU model, G=64:
   `NNTR_HTP_PROFILE=2 NNTR_M0_PROFILE=1 …` then `NNTR_HTP_PROFILE=3 …`;
   paste the header line, the `M==1` and `M>1` rows, the `weight DMA`
   lines, and 3 `[M0-PROF]` lines.

   *Profiles A (≈ 6 min):* from `causallm_profile/` with
   `./models` → `../causallm/models` (symlink or `../causallm/models/<m>`
   path), NPU G=64 and G=512, CPU G=64 and G=512; paste the whole
   `[PROFILE]` table each time (four tables). Never read TPS from these.

   *Probe C (≈ 5 min):* in `htp_u8i4_layer_test/`:
   `LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*'`
   with the A skel; `cp libnntr_hvx_skel.novote.so libnntr_hvx_skel.so`
   and repeat; restore the A skel. Paste all `DMA_PROBE` lines (32 + 32).

   *Probe ④ (≈ 2 min):* `--gtest_filter='*TwoReaderDdr*'` with the A skel;
   paste the `DDR_TWO_READER` line and the three timing lines above it.

   Estimated total ≈ 55 min; budget 90.

   *Result tables (empty, reference numbers in):*

   | variant | model | G | run | prefill ms / tok/s | decode tok/s (all) | decode tok/s (last 64) | gen tokens printed | peak RSS KB | text run1 = run2 | skel md5 (device) |
   |---|---|---|---|---|---|---|---|---|---|---|
   | A | q40 | 64 | 1 | | | | | | | |
   | … 12 rows (2 models × 3 G × 2 runs); reference: NPU 20.8 / CPU 48 decode, NPU prefill 523 (PR), goal ≥ 50 |

   | B | level | qos_mode | M==1 calls | host µs/call | dsp µs/call | transport µs/call | first KB GB/s | avg GB/s |
   |---|---|---|---|---|---|---|---|---|
   | | 2 | | | | | (ref 565) | (ref 16–18) | |
   | | 3 | | | | | | | |

   | A (profile) | type | sum G=64 NPU | sum G=512 NPU | ms/token NPU | ms/token CPU | ref (doc 48 §4) |
   |---|---|---|---|---|---|---|
   | | output_of_causallm | | | | | 3.4 expected after twin |
   | | fully_connected / mha_core / lfm2_moe / rms_norm / causal_conv1d / other | | | | | |

   | C | shape | workers | vote=1 GB/s | vote=0 GB/s |
   |---|---|---|---|---|
   | | (i) (i') (ii) (iii) × 1 2 3 4 | | (ref 18 in situ, 38.8 isolated) | |

   | ④ | cpu alone | dsp alone | cpu with | dsp with | aggregate | verdict (> 45?) |
   |---|---|---|---|---|---|---|

   Texts: full generated text of each run in an appendix, one per
   (model, G, run); "text run1 = run2" is filled by `diff` on the
   workstation.

9. **PR** into `htp_moe` with commits 1–5 (kernel/app commits separate
   from the docs commit per contract §5), `state:review` only after the
   handoff is filled and read (the supervisor updates BENCHMARK/LEDGER
   from it; §6).

The device measurement is unavoidable at step 8; there is exactly one
variant (A) and two additional artifacts (the profile binary set and the
gtest with two skels). Gen lengths 64 / 512 / 1024 are the goal-check
lengths; there is no 512-token sweep beyond the A cells.

## 5. Risks

* **Sampling left on.** If `generation_config.json` is not edited, both
  runs of a cell produce different text and the "text run1 = run2"
  column fails; the handoff's `grep do_sample` echo before the first run
  makes it visible, and a `false` in the log is required for the cell to
  count.
* **Prompt tokenises to ≠ 512 on device.** The `prefill: N tokens` line
  is the check; N ≠ 512 voids the sitting's prefill column but not
  decode. Cause would be a tokenizer.json mismatch between `hf/` and the
  model dir (the quantizer copies it, `quantize_stream.cpp:1245`); the
  handoff records `md5sum tokenizer.json` in both places.
* **Stale skel / stub.** The IDL changed. All five artifacts (skel, app
  libs, gtests) come from one branch head; the device `md5sum` lines in
  the handoff prove the skel in each of the three directories is the A
  skel (or, for C's second half, the novote one), and `AEE_EBADPARM
  (0x8000040E)` on the first MoE call is the symptom of a mix.
* **Profile binary in the TPS directory.** Two device directories and
  the `strings … per-layer-type` check on both binaries keep them apart;
  a `[PROFILE]` table in an A log voids that run.
* **Thermal drift and unit drift** (LEDGER rules 9, 10). Two runs per
  cell, CPU cells first, the 60 s pauses, and the battery/warm note at
  the top make drift visible; the supervisor reads NPU vs CPU inside this
  sitting only.
* **DMA probe measures the wrong thing.** A 1 MiB single-row descriptor
  may be rejected (shape (i') is optional and skipped with a printed
  reason); a per-worker `dmstart` that actually serialises on one engine
  shows as flat GB/s across workers — that is a valid answer, not a
  failure. The checksum column guards against a mapping that reads
  zeros.
* **Two-reader CPU side lands on little cores.** Android does not let
  the gtest pin threads reliably; the CPU-alone number is reported next
  to the concurrent one so a low CPU-alone rate reads as a probe
  limitation, not as bus contention. If CPU-alone < 25 GB/s the ④
  verdict is "inconclusive", not "no".
* **rpcmem 2 × 256 MiB fails on the gtest** (`AEE_ENOMEMORY` or null):
  fall back to 2 × 128 MiB; the line reports `bytes=` so the size in use
  is on record.
* **HF `generation_config.json` absent from a model dir** (if #78's
  quantiser input lacked it): the app throws at `causal_lm.cpp:119`; the
  install step copies `hf/generation_config.json` into the dir before
  the `do_sample` edit.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md` (supervisor, from the filled handoff):
  Goals "now" cells for NPU decode (64 / 512 / 1024, all and last 64),
  CPU decode, NPU prefill at prompt 512, status "provisional" → "measured
  on R3CY10WM83Y"; Results: 12 A rows; Artifacts: A skel, novote skel,
  A app set, profile app set, `unittest_hvx_dma_probe`, both model
  `*_ARM.bin` md5s, `77-prompt512.txt` md5; Method: add the
  `generation(last 64)` line and the `bad_word_ids`/`do_sample` edits as
  part of the standard run.
* `docs/htp_moe/LEDGER.md` (supervisor): verdicts for ①–④ in §2; a new
  rule if the device disagrees with the plan's expectations (candidates:
  "per-thread `dmstart` does / does not add bandwidth", "the bus vote is
  worth X GB/s", "two readers reach Y GB/s aggregate"); §3 items ⑤ ⑥ ⑦
  reordered from ②③, ⑫ resolved or kept from ④.
* `docs/measurements/77-first-handoff.md`: written in step 8, filled by
  the user.
* `.claude/skills/hexagon-gates/SKILL.md` (agent-system commit, separate
  from the code commits): note that a `--profile` build overwrites
  `builddir` and `jni/libs` and must be moved aside, and that
  `unittest_hvx_dma_probe` joins the four device gtests in rung 3.
* `docs/htp_attention/*` stay untouched (read-only history).
