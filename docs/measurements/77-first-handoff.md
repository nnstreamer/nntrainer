# Measurement 77: first handoff — control runs on our unit, doc 48 A/B/C, two-reader DDR probe

Branch `htp/77-first-handoff` @ `3c6e8397` — estimated device time: 55 min (budget 90)

Plan: `docs/plans/77-first-handoff.md`. Issue #77 (part of #76).

## Why

Every number in `docs/htp_moe/BENCHMARK.md` is still from the PR author's
device; this sitting replaces them with unit `R3CY10WM83Y` numbers (CPU
`q40` and NPU `q40-qs4cx-wh`, prompt 512, gen 64 / 512 / 1024) and takes
the three measurements LEDGER ①–④ hang on: the `--profile` breakdown of
the ARM remainder (A), the transport floor at `NNTR_HTP_PROFILE=3` (B),
the arena DMA rate by shape / workers / bus vote (C), and CPU+DSP
concurrent DDR bandwidth (④). The planner orders LEDGER ⑤ ⑥ ⑦ from them.

There is exactly one model-path variant, **A**: the PR head (`htp_moe` @
`856b4a64`) plus one report-only commit (`generation(last 64)` line,
`64a50bcf`) and the regenerated FastRPC stub (one appended IDL entry).
The forward path is unchanged; `nntrainer_causallm` itself is
byte-identical to the BENCHMARK.md artifact (`53814a39…`), the change
is in `libcausallm_core.so`. A second app build (`--profile`) exists only
for measurement A and is never read for tok/s.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

All from `htp/77-first-handoff` @ `3c6e8397`. Staged copies (what the
push lines below use) are under `/local/mnt/workspace/htp_moe/77/`.

| file | md5 | built with |
|---|---|---|
| `test/htp/build/libnntr_hvx_skel.A.so` (→ `tps/`, `profile/`, `gtest/` as `libnntr_hvx_skel.so`) | `0dd2c303f0436a0daa5b25f8654e4190` | `./test/htp/build.sh` |
| `test/htp/build/libnntr_hvx_skel.novote.so` (→ `gtest/`) | `400075032c0c3c15ed5b0cb45a52c0b2` | `HEX_EXTRA_CFLAGS=-DNNTR_HVX_NO_BUS_VOTE ./test/htp/build.sh` |
| `tps/nntrainer_causallm` | `53814a39abad75c045dc6d8ae16e67df` | `build_android.sh --htp --cache` (== PR head `53814a39…`, see Why) |
| `tps/libcausallm_core.so` | `83955c809d3cd6cdc2d3c020a462ebf9` | same |
| `tps/libnntrainer.so` (from `jni/obj/local/arm64-v8a/`) | `326094f4388a3b8928b81d0dfb74f090` | same; NEEDED lists `libsdkl.so`, `libcdsprpc.so` |
| `tps/libccapi-nntrainer.so` | `65c9034c6341384443166b89de66191d` | same |
| `profile/nntrainer_causallm` | `ed4e73fc18dc85971ec2bca8d45eefc3` | `build_android.sh --htp --profile` (fresh builddir) |
| `profile/libcausallm_core.so` | `83955c809d3cd6cdc2d3c020a462ebf9` | same |
| `profile/libnntrainer.so` | `a7d7036c136a0b297173867b2c76e022` | same; NEEDED lists `libsdkl.so`, `libcdsprpc.so` |
| `profile/libccapi-nntrainer.so` | `65c9034c6341384443166b89de66191d` | same |
| `gtest/unittest_hvx_dma_probe` (`test/jni/obj/local/arm64-v8a/`) | `cf18de2cf4ab76d8bc98b06024bc7c3c` | `ndk-build … unittest_hvx_dma_probe` |
| `libc++_shared.so` (NDK r30 sysroot, in `tps/`, `profile/`, `gtest/`) | `b1586b9b512712800fd36a24abac1c0a` | NDK |
| `libsdkl.so` (HexKL `lib/6.4.0.1/armv8_android26`, in `tps/`, `profile/`, `gtest/`) | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL drop |
| `docs/measurements/77-prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | 512 ids with `hf/tokenizer.json` |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40/nntr_lfm2_8b_a1b_q40_arm.bin` (4768855808 B) | `d28f55c5bd7adeb8bf73b02de582eb88` | #78, `nntr_quantize_stream fp32 --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 --isa ARM` (CPU control) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, same + `--moe_dtype QS4CX_WH` (NPU model) |
| `tokenizer.json` (identical in `hf/`, `q40/`, `q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Sanity on the workstation before pushing (both must hold):

```
strings /local/mnt/workspace/htp_moe/77/tps/nntrainer_causallm     | grep -c 'per-layer-type totals'   # 0
strings /local/mnt/workspace/htp_moe/77/profile/nntrainer_causallm | grep -c 'per-layer-type totals'   # 1
```

The four other device gtests (`unittest_hvx_{mm_u8i4,softmax,attn,fc}`)
were relinked against the new stub in the same build and are not part of
this handoff.

## Steps (workstation, phone on USB)

Shell setup once: `cd /home/j2z0-lee/nntrainer && git fetch && git checkout
htp/77-first-handoff && source tools/htp/env.sh`, then
`W=/local/mnt/workspace/htp_moe/77` and `M=/local/mnt/workspace/models/lfm2.5-8b-a1b`.
`md5sum $W/tps/* $W/profile/* $W/gtest/*` must match the table.
Log directory on the workstation: `mkdir -p $W/logs`.

### 0. Device state (1 min)

`adb devices` shows `R3CY10WM83Y device`. Note battery % and warm/cool
in "Notes from the run". Screen off, charger in.

### 1. Install (≈ 10 min, mostly the two model pushes)

```
adb shell mkdir -p /data/local/tmp/nntrainer/causallm/models /data/local/tmp/nntrainer/causallm_profile /data/local/tmp/htp_u8i4_layer_test
adb push $W/tps/nntrainer_causallm $W/tps/libcausallm_core.so $W/tps/libnntrainer.so $W/tps/libccapi-nntrainer.so $W/tps/libc++_shared.so $W/tps/libsdkl.so /data/local/tmp/nntrainer/causallm/
adb push test/htp/build/libnntr_hvx_skel.A.so /data/local/tmp/nntrainer/causallm/libnntr_hvx_skel.so
adb push docs/measurements/77-prompt512.txt /data/local/tmp/nntrainer/causallm/prompt512.txt
adb push $W/profile/nntrainer_causallm $W/profile/libcausallm_core.so $W/profile/libnntrainer.so $W/profile/libccapi-nntrainer.so $W/profile/libc++_shared.so $W/profile/libsdkl.so /data/local/tmp/nntrainer/causallm_profile/
adb push test/htp/build/libnntr_hvx_skel.A.so /data/local/tmp/nntrainer/causallm_profile/libnntr_hvx_skel.so
adb push $W/gtest/unittest_hvx_dma_probe $W/gtest/libc++_shared.so $W/gtest/libsdkl.so /data/local/tmp/htp_u8i4_layer_test/
adb push test/htp/build/libnntr_hvx_skel.A.so /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.so
adb push test/htp/build/libnntr_hvx_skel.novote.so /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.novote.so
adb shell "chmod 755 /data/local/tmp/nntrainer/causallm/nntrainer_causallm /data/local/tmp/nntrainer/causallm_profile/nntrainer_causallm /data/local/tmp/htp_u8i4_layer_test/unittest_hvx_dma_probe"
adb push $M/q40 /data/local/tmp/nntrainer/causallm/models/q40
adb push $M/q40-qs4cx-wh /data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh
adb shell "ln -sfn /data/local/tmp/nntrainer/causallm/models /data/local/tmp/nntrainer/causallm_profile/models"
```

Never push `builddir/.../libcdsprpc.so`; the device's own is the right one.
If a model dir lacks `generation_config.json`, `adb push $M/hf/generation_config.json /data/local/tmp/nntrainer/causallm/models/<m>/` first (the app throws without it, plan §5).

Config edits — greedy decoding and an exact generation count (plan §3.2):

```
for m in q40 q40-qs4cx-wh; do
  adb shell "cd /data/local/tmp/nntrainer/causallm/models/$m && \
    sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
    sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
    grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|moe_engine|moe_htp_layers' nntr_config.json"
done
# Guard (a no-op on the #78 output, which carries both keys): the app
# defaults moe_engine to "cpu" and QS4CX_WH has no CPU path, so a WH dir
# without the key must get it before the first NPU run.
adb shell "cd /data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh && grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json; grep -H -E 'moe_engine|bad_word_ids' nntr_config.json"
```

Expected echo per model: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`; the WH dir shows `moe_engine": "htp"` and
`moe_htp_layers": ""`, the `q40` dir shows neither key. Anything else:
stop, fix, re-echo. Both `generation_config.json` files exist in the #78
output (`"do_sample": true` before the edit), and both `nntr_config.json`
files already point `tokenizer_file` at the device model path.

Provenance lines to paste under "Notes":

```
adb shell "md5sum /data/local/tmp/nntrainer/causallm/libnntr_hvx_skel.so /data/local/tmp/nntrainer/causallm/nntrainer_causallm /data/local/tmp/nntrainer/causallm_profile/libnntr_hvx_skel.so /data/local/tmp/nntrainer/causallm_profile/nntrainer_causallm /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.so /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.novote.so /data/local/tmp/htp_u8i4_layer_test/unittest_hvx_dma_probe /data/local/tmp/nntrainer/causallm/models/q40/tokenizer.json /data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/tokenizer.json"
md5sum $M/hf/tokenizer.json
```

### 2. Control runs, A (≈ 25 min)

One helper, defined in the workstation shell:

```
runA() { # runA <model> <G> <run#>
  adb shell "cd /data/local/tmp/nntrainer/causallm && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/$1/nntr_config.json && \
    grep num_to_generate models/$1/nntr_config.json && \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$1 \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/A_$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)'
}
```

Order — CPU first (no registration, and its decode rate is the thermal
gate: CPU decode < 30 tok/s means throttling, doc 49 §6 → cool down and
redo that cell), then NPU; 60 s pause before each NPU G=1024 run:

```
runA q40 64 1;  runA q40 64 2
runA q40 512 1; runA q40 512 2
runA q40 1024 1; runA q40 1024 2
runA q40-qs4cx-wh 64 1;  runA q40-qs4cx-wh 64 2
runA q40-qs4cx-wh 512 1; runA q40-qs4cx-wh 512 2
sleep 60; runA q40-qs4cx-wh 1024 1; sleep 60; runA q40-qs4cx-wh 1024 2
```

Expected lines in every log, exactly: `prefill: 512 tokens, … TPS`,
`generation: <G> tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`,
`total: … ms`, `peak memory: … KB`. `prefill: N` with N ≠ 512 voids the
prefill column of the sitting (tokenizer mismatch — compare the
`tokenizer.json` md5s above); `generation: N` with N ≠ G is a failed
step (config edit did not take), not a data point. A `[PROFILE]` table in
any of these logs means the profile binary was run from the wrong
directory: void the run.

Text comparison on the workstation (fills "text run1 = run2"):

```
for m in q40 q40-qs4cx-wh; do for g in 64 512 1024; do
  printf '%s G=%s: ' $m $g; diff <(sed -n '/^=====/q;p' $W/logs/A_${m}_G${g}_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_${m}_G${g}_r2.log) >/dev/null && echo same || echo DIFFERENT
done; done
```

### 3. Profiles B (≈ 4 min) — NPU model, G=64, TPS binary

```
adb shell "cd /data/local/tmp/nntrainer/causallm && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json"
adb shell "cd /data/local/tmp/nntrainer/causallm && NNTR_HTP_PROFILE=2 NNTR_M0_PROFILE=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/B_level2.log | grep -E 'HTP-PROFILE|M0-PROF' | head -40
adb shell "cd /data/local/tmp/nntrainer/causallm && NNTR_HTP_PROFILE=3 NNTR_M0_PROFILE=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/B_level3.log | grep -E 'HTP-PROFILE|M0-PROF' | head -40
```

Paste, for each level: the `[HTP-PROFILE] level=… qos_mode=…` header, the
`M==1` and `M>1` rows, the `weight DMA:` line(s), and three `[M0-PROF]`
lines. `qos_mode` must be the same in both.

### 4. Profiles A (≈ 6 min) — profile binary, never read for tok/s

```
runP() { # runP <model> <G>
  adb shell "cd /data/local/tmp/nntrainer/causallm_profile && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/$1/nntr_config.json && \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$1 \"\$(cat ../causallm/prompt512.txt)\"" 2>&1 | tee $W/logs/P_$1_G$2.log | sed -n '/per-layer-type totals/,$p'
}
runP q40-qs4cx-wh 64; runP q40-qs4cx-wh 512
runP q40 64; runP q40 512
```

Paste all four `[PROFILE]` tables whole (they are small). Per-token
decode cost of a type = (sum at G=512 − sum at G=64) / 448.

### 5. Probe C (≈ 5 min) — `htp_u8i4_layer_test`, both skels

```
adb shell "cd /data/local/tmp/htp_u8i4_layer_test && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*'" 2>&1 | tee $W/logs/C_vote1.log | grep -E 'DMA_PROBE|md5|PASSED|FAILED'
adb shell "cd /data/local/tmp/htp_u8i4_layer_test && cp libnntr_hvx_skel.novote.so libnntr_hvx_skel.so && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*'" 2>&1 | tee $W/logs/C_vote0.log | grep -E 'DMA_PROBE|md5|PASSED|FAILED'
adb push test/htp/build/libnntr_hvx_skel.A.so /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.so
```

Expect 16 `DMA_PROBE` lines per run (`vote=1` then `vote=0`), each with
`checksum_ok=y`; shape `i1` may print `skipped err=…` (a 1 MiB single-row
descriptor the engine refuses — a fact, not a failure). Paste all lines.
A DMA process crash or `0x8000040e` on the first call means a stale skel
(the md5 line tells which).

### 6. Probe ④ (≈ 2 min) — A skel

```
adb shell "cd /data/local/tmp/htp_u8i4_layer_test && md5sum libnntr_hvx_skel.so && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*TwoReaderDdr*'" 2>&1 | tee $W/logs/D_two_reader.log | grep -E 'DDR_|md5|PASSED|FAILED'
```

Paste the `DDR_TWO_READER` line and the `DDR_CPU` / `DDR_DSP` lines above
it (three timings plus the concurrent pair).

### 7. Finish

Fill the tables below, copy the generated texts into the appendix,
`git add docs/measurements/77-first-handoff.md && git commit -s`, push the
branch, set the issue to `state:measured`.

## Results

> **Unit deviation — read first.** This sitting ran on **`R3CY205ZMND`**
> (SM-S938N), not on `R3CY10WM83Y` as §0 and issue #79 specify: the only
> device reachable from this workstation was the second S25 unit, over an
> SSH bridge. The two units are known to differ by ~7 % in DSP clock, so
> **these numbers must not be read as "our unit" numbers** and must not
> replace the provisional values in `docs/htp_moe/BENCHMARK.md` without a
> `R3CY10WM83Y` sitting to anchor them. Everything else in the protocol
> was followed as written. See "Deviations" below.

Battery at start: 100 % (USB powered)   phone: cool (battery 27.3 °C,
`thermal_zone0` 28.7 °C)   date/time: 2026-09-21 18:20–18:35 KST
Battery at end: 100 %, `thermal_zone0` 41.8 °C.

### A — control runs (prompt 512, `NNTR_NUM_THREADS=8`, TPS binary)

| variant | model | G | run | prefill ms / tok/s | decode tok/s (all) | decode tok/s (last 64) | gen tokens printed | peak RSS KB | text run1 = run2 | skel md5 (device) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | q40 | 64 | 1 | 1522 ms / 336.399 | 54.0997 | n/a ⑴ | 64 | 5910448 | same | n/a (CPU) |
| A | q40 | 64 | 2 | 2023 ms / 253.089 | 52.9363 | n/a ⑴ | 64 | 5302900 | same | n/a |
| A | q40 | 512 | 1 | 1721 ms / 297.501 | 52.6749 | n/a ⑴ | 512 | 5612836 | same | n/a |
| A | q40 | 512 | 2 | 2212 ms / 231.465 | 51.9691 | n/a ⑴ | 512 | 5659096 | same | n/a |
| A | q40 | 1024 | 1 | 1782 ms / 287.318 | 46.4378 | n/a ⑴ | 1024 | 5396472 | same | n/a |
| A | q40 | 1024 | 2 | 1778 ms / 287.964 | 48.5285 | n/a ⑴ | 1024 | 5620196 | same | n/a |
| A | q40-qs4cx-wh | 64 | 1 | 1269 ms / 403.467 | 19.6802 | n/a ⑴ | 64 | 5319016 | same | `47c14253…` |
| A | q40-qs4cx-wh | 64 | 2 | 1077 ms / 475.395 | 21.2695 | n/a ⑴ | 64 | 5329596 | same | `47c14253…` |
| A | q40-qs4cx-wh | 512 | 1 | 1054 ms / 485.769 | 18.3756 | n/a ⑴ | 512 | 5323824 | same | `47c14253…` |
| A | q40-qs4cx-wh | 512 | 2 | 1035 ms / 494.686 | 18.2434 | n/a ⑴ | 512 | 5312408 | same | `47c14253…` |
| A | q40-qs4cx-wh | 1024 | 1 | 946 ms / 541.226 | 19.0448 | n/a ⑴ | 1024 | 5440104 | same | `47c14253…` |
| A | q40-qs4cx-wh | 1024 | 2 | 1125 ms / 455.111 | 16.9587 | n/a ⑴ | 1024 | 4983988 | same | `47c14253…` |

⑴ **The `generation(last 64)` line never prints for this model.** The
step-2 expectation ("Expected lines in every log, exactly: … `generation(last
64): 64 tokens, … TPS` …") cannot hold on this tree. `64a50bcf` added that
line to `Applications/CausalLM/models/lfm2/lfm2_causallm.cpp:778`, inside
the report block at `lfm2_causallm.cpp:764`. LFM2-MoE
(`Lfm2MoeCausalLM : public Lfm2CausalLM`) does not reach that block; it
reports from the base block at `Applications/CausalLM/models/causal_lm.cpp:699`,
which prints only `prefill` / `generation` / `total` / `peak memory`. This
is a defect in the handoff's expectation, not a failed step: all four other
required lines are present in all 12 logs. The column is therefore
unfillable until the last-64 timing is moved into the base report block.

All 12 cells printed `prefill: 512 tokens` and `generation: <G> tokens`
exactly, so no cell is a failed step and no tokenizer mismatch occurred
(device `tokenizer.json` md5 `7b8067a5…` == workstation `hf/` value).
CPU decode stayed at 46.4–54.1 tok/s, above the 30 tok/s thermal gate in
every cell, so no cell was redone for heat.

Text comparison (step 2 `diff` loop) — all six cells `same`:

```
q40            G=64    same
q40            G=512   same
q40            G=1024  same
q40-qs4cx-wh   G=64    same
q40-qs4cx-wh   G=512   same
q40-qs4cx-wh   G=1024  same
```

Reference (PR author's device, prompt 444, gen 512): NPU decode 20.8,
CPU decode 48, NPU prefill 523 tok/s. On this unit at G=512: NPU decode
18.3, CPU decode 52.3, NPU prefill 490. **The ≥ 50 decode goal is met by
the CPU control only; the NPU path is at ~18 tok/s, ~2.7× short.** Prefill
is the NPU's win: 455–541 tok/s against the CPU's 231–336.

### B — transport floor (NPU model, G=64)

| level | qos_mode | M==1 calls | host µs/call | dsp µs/call | transport µs/call | first KB → GB/s | avg GB/s |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 1408 | 1948.6 | 1360.9 | **587.7** (ref 565) | 1024 KB → 18004.8 | 16.2 (ref 16–18) |
| 3 | 2 | 1408 | 1750.4 | 1222.7 | **527.7** | 1024 KB → 19685.3 | 18.0 |

`qos_mode` is 2 at both levels, as required.

**Verdict: marshalling.** Level-3 transport is 527.7 µs/call, far above the
≤ 300 µs "wake/clock" threshold, and −10.2 % from level 2, inside the ±15 %
band. Level 2's 587.7 µs also lands close to the 565 µs reference, which
supports the reading.

Pasted lines:

```
(level 2)
[HTP-PROFILE] level=2 qos_mode=2 (2=poll 1=PM 0=interrupt-driven -- every transport number below is only comparable to 34_fc_measured.md's 326us/call at qos_mode=2)
[HTP-PROFILE]   K=2048  N=2048  M>1     calls=23      rows=11776    host=    418.0 ms (18172.5 us/call)  dsp=15941.3 us/call (87.7%) transport= 2231.2 us/call  [quant 297.3 gather 116.0 requant 1106.7 swiglu 0.0 dequant 1007.6 acc 3133.4 drain 224.5+56.7 push 49.9 scatter 75.2 alloc 73.4 stage 428.0 mm 9175.8 | rest<=196.8 (1.1% of host) blocks=1105]
[HTP-PROFILE]     weight DMA: 154969 KB/call, first 1024 KB took 0 us = 2679.7 GB/s; averaged over the call 10.0 GB/s
[HTP-PROFILE]   K=2048  N=2048  M==1    calls=1408    rows=1408     host=   2743.7 ms ( 1948.6 us/call)  dsp= 1360.9 us/call (69.8%) transport=  587.7 us/call  [quant 14.8 gather 143.5 requant 41.9 swiglu 0.0 dequant 13.0 acc 259.1 drain 108.7+7.3 push 2.5 scatter 0.9 alloc 0.2 stage 4.6 mm 748.0 | rest<=16.4 (0.8% of host) blocks=5632]
[HTP-PROFILE]     weight DMA: 21504 KB/call, first 1024 KB took 0 us = 18004.8 GB/s; averaged over the call 16.2 GB/s
[M0-PROF] moe_layer[0] tokens=512 us=17938  setup=0 router=1478 topk=188 wksp=0 gather=0 ffn=16265 route=0 scatter=0 other=7
[M0-PROF] moe_layer[1] tokens=512 us=17861  setup=0 router=1178 topk=176 wksp=0 gather=0 ffn=16501 route=0 scatter=0 other=6
[M0-PROF] moe_layer[2] tokens=512 us=21822  setup=0 router=1186 topk=194 wksp=0 gather=0 ffn=20433 route=0 scatter=0 other=9

(level 3)
[HTP-PROFILE] level=3 qos_mode=2 (2=poll 1=PM 0=interrupt-driven -- every transport number below is only comparable to 34_fc_measured.md's 326us/call at qos_mode=2)
[HTP-PROFILE]   K=2048  N=2048  M>1     calls=23      rows=11776    host=    410.0 ms (17824.5 us/call)  dsp=15701.7 us/call (88.1%) transport= 2122.9 us/call  [quant 303.6 gather 100.4 requant 1128.3 swiglu 0.0 dequant 1036.7 acc 3124.3 drain 46.3+19.8 push 50.0 scatter 66.6 alloc 0.7 stage 442.5 mm 9184.0 | rest<=198.4 (1.1% of host) blocks=1105]
[HTP-PROFILE]     weight DMA: 154969 KB/call, first 1024 KB took 0 us = 3014.7 GB/s; averaged over the call 10.1 GB/s
[HTP-PROFILE]   K=2048  N=2048  M==1    calls=1408    rows=1408     host=   2464.6 ms ( 1750.4 us/call)  dsp= 1222.7 us/call (69.9%) transport=  527.7 us/call  [quant 21.9 gather 122.6 requant 41.9 swiglu 0.0 dequant 7.3 acc 258.8 drain 0.9+0.4 push 2.6 scatter 0.9 alloc 0.1 stage 4.4 mm 746.6 | rest<=14.1 (0.8% of host) blocks=5632]
[HTP-PROFILE]     weight DMA: 21504 KB/call, first 1024 KB took 0 us = 19685.3 GB/s; averaged over the call 18.0 GB/s
[M0-PROF] moe_layer[0] tokens=512 us=88835  setup=0 router=826 topk=129 wksp=0 gather=0 ffn=87866 route=0 scatter=0 other=14
[M0-PROF] moe_layer[1] tokens=512 us=91288  setup=0 router=767 topk=129 wksp=0 gather=0 ffn=90380 route=0 scatter=0 other=12
[M0-PROF] moe_layer[2] tokens=512 us=91411  setup=0 router=793 topk=134 wksp=0 gather=0 ffn=90471 route=0 scatter=0 other=13
```

Note the `[M0-PROF]` `ffn=` cost per MoE layer rises from ~16–20 ms at
level 2 to ~88–90 ms at level 3: level 3's own instrumentation dominates
the DSP-side FFN timing, so those `us=` figures are not comparable across
levels. The `[HTP-PROFILE]` transport figures, which the verdict uses, are.

### A (profile) — per-type decode cost

Aggregated from the four `[PROFILE]` tables by the layer type in
`…:forward(<type>)`; the printed table is one row per layer, not per type.
Per-token decode cost = (sum at G=512 − sum at G=64) / 448, in µs → ms.

| type | sum G=64 NPU (µs) | sum G=512 NPU (µs) | ms/token NPU | sum G=64 CPU | sum G=512 CPU | ms/token CPU |
|---|---|---|---|---|---|---|
| lfm2_moe | 3306202 | 22176944 | **42.122** | 6216557 | 16214179 | **22.316** |
| fully_connected | 2234100 | 14851307 | **28.163** | 1629326 | 6240177 | **10.292** |
| output_of_causallm (tie_word_embeddings) | 187162 | 1462608 | 2.847 | 256313 | 1778248 | 3.397 |
| mha_core | 223261 | 1254431 | 2.302 | 476119 | 1733085 | 2.806 |
| addition | 27518 | 122518 | 0.212 | 32332 | 96406 | 0.143 |
| rms_norm | 27961 | 111400 | 0.186 | 29547 | 81994 | 0.117 |
| custom_multiply | 14193 | 81931 | 0.151 | 11761 | 53991 | 0.094 |
| causal_conv1d | 38909 | 64931 | 0.058 | 53102 | 65289 | 0.027 |
| split | 14904 | 38925 | 0.054 | 15830 | 37318 | 0.048 |
| reshaped_rms_norm | 7502 | 32027 | 0.055 | 11130 | 24726 | 0.030 |
| swiglu | 8910 | 28509 | 0.044 | 13990 | 23349 | 0.021 |
| input | 245 | 2423 | 0.005 | 164 | 1239 | 0.002 |
| multiout | 99 | 532 | 0.001 | 1845 | 179 | −0.004 |
| **total (sum of rows)** | | | **76.200** | | | **39.291** |
| 1000 / decode tok/s (TPS binary, G=512) | | | 54.42 | | | 18.98 |

The profile binary inflates both totals (76.2 vs 54.4 ms/token NPU, 39.3 vs
19.0 ms/token CPU), as expected — this binary is never read for tok/s. The
**shape** is the result: `lfm2_moe` + `fully_connected` are 70.3 of the NPU's
76.2 ms/token (92 %) and 32.6 of the CPU's 39.3 (83 %). Moving the MoE FFN
to the NPU has made it *more* expensive per token than the CPU path
(42.1 vs 22.3 ms), and `fully_connected` — the dense, non-MoE FC work,
which the NPU path also routes through HTP — is 2.7× the CPU cost
(28.2 vs 10.3 ms). `output_of_causallm` is the one type where the NPU
already wins (2.85 vs 3.40 ms).

Raw tables: `logs/P_q40-qs4cx-wh_G64.log`, `logs/P_q40-qs4cx-wh_G512.log`,
`logs/P_q40_G64.log`, `logs/P_q40_G512.log` (one row per layer; ~150 rows
each, aggregated above rather than pasted whole).

### C — arena DMA probe (GB/s; `bytes=` from the line)

| shape | workers | vote=1 GB/s | vote=0 GB/s | checksum_ok (v1/v0) | workers_used |
|---|---|---|---|---|---|
| (i) 4 KiB × 256 | 1 | 79.6 | 75.2 | y/y | 1 |
| (i) | 2 | 75.9 | 72.2 | y/y | 2 |
| (i) | 3 | 74.2 | 73.6 | y/y | 3 |
| (i) | 4 | 72.0 | 72.2 | y/y | 4 |
| (i') 1 MiB × 1 | 1 | 78.4 | 76.6 | y/y | 1 |
| (i') | 2 | 76.0 | 76.0 | y/y | 2 |
| (i') | 3 | 74.1 | 73.5 | y/y | 3 |
| (i') | 4 | 72.4 | 72.3 | y/y | 4 |
| (ii) 16 KiB × 64 @ 56 KiB | 1 | 116.5 | 113.9 | y/y | 1 |
| (ii) | 2 | 117.0 | 114.3 | y/y | 2 |
| (ii) | 3 | 112.0 | 112.1 | y/y | 3 |
| (ii) | 4 | 109.7 | 110.2 | y/y | 4 |
| (iii) 8 KiB × 64 @ 56 KiB | 1 | 111.0 | 109.5 | y/y | 1 |
| (iii) | 2 | 113.2 | 112.4 | y/y | 2 |
| (iii) | 3 | 110.5 | 110.3 | y/y | 3 |
| (iii) | 4 | 108.6 | 108.1 | y/y | 4 |

All 32 lines have `checksum_ok=y`. **Shape `i1` did not skip** — the 1 MiB
single-row descriptor the step-5 note warned the engine might refuse ran
normally here (78.4 GB/s at 1 worker). No `0x8000040e`, no crash; device
skel md5 matched the pushed file on both runs (`47c14253…` then
`9955666382…`), and skel A was restored afterwards.

**Verdict: shape is the only axis that moves the rate.** Workers are flat to
slightly negative (79.6 → 72.0 GB/s going 1 → 4 on shape i), and the bus
vote is worth at most ~4 % and is inside run-to-run noise on shapes ii/iii
(where vote=0 is sometimes *faster*). Shape is worth ~1.5×: the strided
56 KiB-pitch shapes (ii, iii) reach 108–117 GB/s against 72–80 GB/s for the
contiguous ones (i, i').

These isolated rates are 4–7× the 16.2–18.0 GB/s the same silicon achieves
in situ (the `weight DMA` line of B), and ~3× the 38.8 GB/s
`ArenaMapAndDma` reference from the PR author's device. The gap between
isolated and in-situ DMA, not the DMA engine itself, is what LEDGER ⑤
should target.

Pasted lines:

```
47c14253ee734099f6a76ab90a54e16c  libnntr_hvx_skel.so
DMA_PROBE shape=i workers=1 vote=1 gbs=79.6 us=6743 bytes=536870912 checksum_ok=y workers_used=1 n_desc=256 passes=2
DMA_PROBE shape=i workers=2 vote=1 gbs=75.9 us=7071 bytes=536870912 checksum_ok=y workers_used=2 n_desc=256 passes=2
DMA_PROBE shape=i workers=3 vote=1 gbs=74.2 us=7237 bytes=536870912 checksum_ok=y workers_used=3 n_desc=256 passes=2
DMA_PROBE shape=i workers=4 vote=1 gbs=72.0 us=7459 bytes=536870912 checksum_ok=y workers_used=4 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=1 vote=1 gbs=78.4 us=6849 bytes=536870912 checksum_ok=y workers_used=1 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=2 vote=1 gbs=76.0 us=7066 bytes=536870912 checksum_ok=y workers_used=2 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=3 vote=1 gbs=74.1 us=7245 bytes=536870912 checksum_ok=y workers_used=3 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=4 vote=1 gbs=72.4 us=7414 bytes=536870912 checksum_ok=y workers_used=4 n_desc=256 passes=2
DMA_PROBE shape=ii workers=1 vote=1 gbs=116.5 us=5911 bytes=688914432 checksum_ok=y workers_used=1 n_desc=219 passes=3
DMA_PROBE shape=ii workers=2 vote=1 gbs=117.0 us=5886 bytes=688914432 checksum_ok=y workers_used=2 n_desc=219 passes=3
DMA_PROBE shape=ii workers=3 vote=1 gbs=112.0 us=6150 bytes=688914432 checksum_ok=y workers_used=3 n_desc=219 passes=3
DMA_PROBE shape=ii workers=4 vote=1 gbs=109.7 us=6279 bytes=688914432 checksum_ok=y workers_used=4 n_desc=219 passes=3
DMA_PROBE shape=iii workers=1 vote=1 gbs=111.0 us=7241 bytes=803733504 checksum_ok=y workers_used=1 n_desc=511 passes=3
DMA_PROBE shape=iii workers=2 vote=1 gbs=113.2 us=7101 bytes=803733504 checksum_ok=y workers_used=2 n_desc=511 passes=3
DMA_PROBE shape=iii workers=3 vote=1 gbs=110.5 us=7275 bytes=803733504 checksum_ok=y workers_used=3 n_desc=511 passes=3
DMA_PROBE shape=iii workers=4 vote=1 gbs=108.6 us=7402 bytes=803733504 checksum_ok=y workers_used=4 n_desc=511 passes=3
[  PASSED  ] 1 test.

9955666382efe86741f1ad3ab0451066  libnntr_hvx_skel.so
DMA_PROBE shape=i workers=1 vote=0 gbs=75.2 us=7140 bytes=536870912 checksum_ok=y workers_used=1 n_desc=256 passes=2
DMA_PROBE shape=i workers=2 vote=0 gbs=72.2 us=7437 bytes=536870912 checksum_ok=y workers_used=2 n_desc=256 passes=2
DMA_PROBE shape=i workers=3 vote=0 gbs=73.6 us=7293 bytes=536870912 checksum_ok=y workers_used=3 n_desc=256 passes=2
DMA_PROBE shape=i workers=4 vote=0 gbs=72.2 us=7437 bytes=536870912 checksum_ok=y workers_used=4 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=1 vote=0 gbs=76.6 us=7005 bytes=536870912 checksum_ok=y workers_used=1 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=2 vote=0 gbs=76.0 us=7067 bytes=536870912 checksum_ok=y workers_used=2 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=3 vote=0 gbs=73.5 us=7304 bytes=536870912 checksum_ok=y workers_used=3 n_desc=256 passes=2
DMA_PROBE shape=i1 workers=4 vote=0 gbs=72.3 us=7426 bytes=536870912 checksum_ok=y workers_used=4 n_desc=256 passes=2
DMA_PROBE shape=ii workers=1 vote=0 gbs=113.9 us=6047 bytes=688914432 checksum_ok=y workers_used=1 n_desc=219 passes=3
DMA_PROBE shape=ii workers=2 vote=0 gbs=114.3 us=6025 bytes=688914432 checksum_ok=y workers_used=2 n_desc=219 passes=3
DMA_PROBE shape=ii workers=3 vote=0 gbs=112.1 us=6146 bytes=688914432 checksum_ok=y workers_used=3 n_desc=219 passes=3
DMA_PROBE shape=ii workers=4 vote=0 gbs=110.2 us=6251 bytes=688914432 checksum_ok=y workers_used=4 n_desc=219 passes=3
DMA_PROBE shape=iii workers=1 vote=0 gbs=109.5 us=7343 bytes=803733504 checksum_ok=y workers_used=1 n_desc=511 passes=3
DMA_PROBE shape=iii workers=2 vote=0 gbs=112.4 us=7152 bytes=803733504 checksum_ok=y workers_used=2 n_desc=511 passes=3
DMA_PROBE shape=iii workers=3 vote=0 gbs=110.3 us=7289 bytes=803733504 checksum_ok=y workers_used=3 n_desc=511 passes=3
DMA_PROBE shape=iii workers=4 vote=0 gbs=108.1 us=7433 bytes=803733504 checksum_ok=y workers_used=4 n_desc=511 passes=3
[  PASSED  ] 1 test.
```

### ④ — two-reader DDR

| cpu alone GB/s | dsp alone GB/s | cpu with | dsp with | aggregate | dsp_workers | chunk_bytes | verdict (> 45?) |
|---|---|---|---|---|---|---|---|
| 67.90 | 3265.59 | 39.73 | 3113.49 | 3153.22 | 2 | 268435456 | see below |

`cpu alone` is 67.90 GB/s, well above the 25 GB/s floor, so the
"inconclusive (little cores)" escape in plan §5 does not apply.

**Reading the numbers, with a caveat the plan did not anticipate.** The
`dsp_alone` figure of 3265 GB/s is not a DDR rate — it is two orders of
magnitude above any plausible LPDDR5X bandwidth on this part, so the DSP
side of this probe is measuring something cache- or VTCM-resident rather
than a DDR stream. The `aggregate` column inherits that, and the
`> 45 ?` verdict as specified cannot be answered honestly from it.

What the probe *does* show cleanly is the CPU side under contention:
67.90 GB/s alone → 39.73 GB/s with the DSP running, a **41 % loss**. That
is real interference and is the useful result for LEDGER ④. The DSP-side
number needs the probe's DSP loop re-examined (working-set size vs VTCM)
before it can be quoted.

Pasted lines:

```
47c14253ee734099f6a76ab90a54e16c  libnntr_hvx_skel.so
DDR_CPU passes=2030 bytes=136230993920.00 s=2.01 gbs=67.90 xor=0
DDR_DSP workers=2 passes=159 bytes=85362475008 us=26140 gbs=3265.59
DDR_DSP workers=2 passes=159 bytes=85362475008 us=27417 gbs=3113.49
DDR_CPU passes=25 bytes=1677721600.00 s=0.04 gbs=39.73 xor=0
DDR_TWO_READER cpu_alone=67.90 dsp_alone=3265.59 cpu_with=39.73 dsp_with=3113.49 aggregate=3153.22 cpu_threads=8 dsp_workers=2 chunk_bytes=268435456
[  PASSED  ] 1 test.
```

## Deviations

1. **Unit.** Ran on `R3CY205ZMND`, not `R3CY10WM83Y` (see the banner above).
   All other protocol steps were followed as written.
2. **Compiled artifacts have different md5s than the Artifacts table.** The
   split is exact: every file taken as-is matches
   (`libc++_shared.so` `b1586b9b…`, `libsdkl.so` `0ad4e22a…`,
   `77-prompt512.txt` `fc65c158…`, `hf/tokenizer.json` `7b8067a5…`), and
   every file compiled here differs (both skels, both app binaries, both
   `libnntrainer.so`, `libcausallm_core.so`, `libccapi-nntrainer.so`,
   `unittest_hvx_dma_probe`). The sources are identical — `3c6e8397`
   (the Artifacts-table commit) to `81fd3ec9` (branch head, built here)
   touches only `hexagon-gates/SKILL.md`, this file, and `77-prompt512.txt`.
   The structural relations from the table reproduce exactly:
   `libcausallm_core.so` and `libccapi-nntrainer.so` are identical between
   the TPS and `--profile` builds while `nntrainer_causallm` and
   `libnntrainer.so` differ, and the `per-layer-type totals` string count is
   0 in the TPS binary and 1 in the profile binary. The strongest evidence
   that this is an environment difference and not a code difference is that
   `nntr_quantize_stream`, built here, produced **byte-identical** model
   files: `q40` = `d28f55c5…` / 4768855808 B and `q40-qs4cx-wh` =
   `7b7867fa…` / 4316133120 B, both matching the Artifacts table. The
   build path differs (`/home/j2z0/Project/nntrainer_htp77` vs
   `/home/j2z0-lee/nntrainer`), which is the likely cause, but this was not
   proven — no rebuild under the original path was attempted.
3. **`generation(last 64)` is unreachable for LFM2-MoE.** See note ⑴.
4. **Models were rebuilt from scratch**, not copied: #78's outputs were not
   present on this workstation. `hf download LiquidAI/LFM2.5-8B-A1B` →
   `weight_converter.py` → `nntr_quantize_stream` reproduced both `.bin`
   files bit-for-bit (deviation 2), so the chain is verified end to end.
5. **Toolchain gaps filled.** NDK r30 was absent and was installed to
   `$HOME/android-ndk-r30` (the `env.sh` default). `HEXKL_ROOT` is
   `$HOME/Downloads/hexkl_addon`, not the `env.sh` default
   `$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon`; `lib/6.4.0.1` is the same
   drop (`libsdkl.so` md5 matches).
6. **Config edits were made on the workstation and pushed**, not applied
   with device-side `sed`. The SSH bridge does not preserve shell quoting
   or `|`, so the step-1 and `runA` `sed` one-liners cannot run through it.
   The resulting on-device configs were read back and match what step 1
   specifies: both models `do_sample: false`, `bad_word_ids: [124900]`,
   `init_seq_len: 512`; only `q40-qs4cx-wh` carries `moe_engine: htp` and
   `moe_htp_layers: ""`.
7. **Per-layer-type profile tables were aggregated, not pasted whole.** The
   `[PROFILE]` output is one row per layer (~150 rows × 4 runs), not per
   type; the totals table above sums by the type in `…:forward(<type>)`.

## Notes from the run

- battery / thermal: 100 % at start and end, USB powered. Battery 27.3 °C,
  `thermal_zone0` 28.7 °C before the first run; `thermal_zone0` 41.8 °C
  after the last. No cell tripped the CPU-decode < 30 tok/s thermal gate.
- device md5sum lines (step 1):

```
47c14253ee734099f6a76ab90a54e16c  /data/local/tmp/nntrainer/causallm/libnntr_hvx_skel.so
170af7a65ed9cfa3798e7bc05e7cb58b  /data/local/tmp/nntrainer/causallm/nntrainer_causallm
47c14253ee734099f6a76ab90a54e16c  /data/local/tmp/nntrainer/causallm_profile/libnntr_hvx_skel.so
3806b9040aaa162abe7f48555efca8b1  /data/local/tmp/nntrainer/causallm_profile/nntrainer_causallm
47c14253ee734099f6a76ab90a54e16c  /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.so
9955666382efe86741f1ad3ab0451066  /data/local/tmp/htp_u8i4_layer_test/libnntr_hvx_skel.novote.so
9de5715ba230a704b98e78bb1a1257d6  /data/local/tmp/htp_u8i4_layer_test/unittest_hvx_dma_probe
7b8067a580173d3eb1697afae3b456f5  /data/local/tmp/nntrainer/causallm/models/q40/tokenizer.json
7b8067a580173d3eb1697afae3b456f5  /data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/tokenizer.json
```

- tokenizer.json md5 (device q40 / device WH / workstation `hf`): all three
  `7b8067a580173d3eb1697afae3b456f5`, matching the Artifacts table
  (checked 2026-09-21).
- model `.bin` on device: `q40` 4768855808 B, `q40-qs4cx-wh` 4316133120 B —
  both match the Artifacts table sizes.
- anything stale, FARF/AEE errors, retries: no `AEE_EBADPARM`, no
  `0x8000040e`, no `htp Context is not registered`, no DMA process crash.
  Both probes passed on first run. The only retries were transport-level:
  the first SSH bridge session expired mid-install (the 4.77 GB `q40`
  push was received by the bridge but never written to the device), and the
  whole install was re-run on a fresh session; device md5s were verified
  after the successful run.

## Appendix — generated texts

`run1` and `run2` are byte-identical in all six cells (the step-2 `diff`
loop above), so one block per (model, G) is recorded rather than per run.
Full logs: `logs/A_<model>_G<G>_r<n>.log`.

### q40, G=64 (run1 == run2)

```
The small harbour town of Ardley sits where a slow river meets a cold northern sea, and for most of its history it has lived by the tide. Fishing boats leave before dawn and return in the early afternoon, their holds filled with herring, cod and the occasional crab, and the smell of salt and diesel hangs over the quay long after the catch has been sold. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells newspapers and fishing line, and a small museum that opens only on summer weekends. Most of the houses are built from the same grey stone as the church, with slate roofs that shine when it rains, which is often. In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley are known for being reserved with strangers and generous with neighbours. When a boat is late they gather quietly at the harbour, and when it comes in they disperse without a word, as if nothing had happened. The school has forty pupils and two teachers, and the older children take a bus each morning to the secondary school in the larger town twelve miles inland, a journey that takes forty minutes along a narrow road that floods in spring. There is a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, usually to help pleasure craft that have misjudged the currents around the headland. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. Farming occupies the land behind the town, mostly sheep on the higher ground and barley in the sheltered valley, and the two communities, sea and land, have intermarried for so long that nearly everyone in Ardley can name a cousin on a boat and a cousin on a tractor. Continue this description of Ardley in the same style, adding more detail about its history, its people, its weather and the seasons, and do not stop until you are told to.
 town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself
```

### q40, G=512 (run1 == run2)

```
The small harbour town of Ardley sits where a slow river meets a cold northern sea, and for most of its history it has lived by the tide. Fishing boats leave before dawn and return in the early afternoon, their holds filled with herring, cod and the occasional crab, and the smell of salt and diesel hangs over the quay long after the catch has been sold. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells newspapers and fishing line, and a small museum that opens only on summer weekends. Most of the houses are built from the same grey stone as the church, with slate roofs that shine when it rains, which is often. In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley are known for being reserved with strangers and generous with neighbours. When a boat is late they gather quietly at the harbour, and when it comes in they disperse without a word, as if nothing had happened. The school has forty pupils and two teachers, and the older children take a bus each morning to the secondary school in the larger town twelve miles inland, a journey that takes forty minutes along a narrow road that floods in spring. There is a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, usually to help pleasure craft that have misjudged the currents around the headland. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. Farming occupies the land behind the town, mostly sheep on the higher ground and barley in the sheltered valley, and the two communities, sea and land, have intermarried for so long that nearly everyone in Ardley can name a cousin on a boat and a cousin on a tractor. Continue this description of Ardley in the same style, adding more detail about its history, its people, its weather and the seasons, and do not stop until you are told to.
 town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising
```

### q40, G=1024 (run1 == run2)

```
The small harbour town of Ardley sits where a slow river meets a cold northern sea, and for most of its history it has lived by the tide. Fishing boats leave before dawn and return in the early afternoon, their holds filled with herring, cod and the occasional crab, and the smell of salt and diesel hangs over the quay long after the catch has been sold. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells newspapers and fishing line, and a small museum that opens only on summer weekends. Most of the houses are built from the same grey stone as the church, with slate roofs that shine when it rains, which is often. In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley are known for being reserved with strangers and generous with neighbours. When a boat is late they gather quietly at the harbour, and when it comes in they disperse without a word, as if nothing had happened. The school has forty pupils and two teachers, and the older children take a bus each morning to the secondary school in the larger town twelve miles inland, a journey that takes forty minutes along a narrow road that floods in spring. There is a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, usually to help pleasure craft that have misjudged the currents around the headland. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. Farming occupies the land behind the town, mostly sheep on the higher ground and barley in the sheltered valley, and the two communities, sea and land, have intermarried for so long that nearly everyone in Ardley can name a cousin on a boat and a cousin on a tractor. Continue this description of Ardley in the same style, adding more detail about its history, its people, its weather and the seasons, and do not stop until you are told to.
 town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself. The lifeboat is the pride of the town, and the annual fundraising day in August, with its raffle, cake stall and tug of war, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum
```

### q40-qs4cx-wh, G=64 (run1 == run2)

```
The small harbour town of Ardley sits where a slow river meets a cold northern sea, and for most of its history it has lived by the tide. Fishing boats leave before dawn and return in the early afternoon, their holds filled with herring, cod and the occasional crab, and the smell of salt and diesel hangs over the quay long after the catch has been sold. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells newspapers and fishing line, and a small museum that opens only on summer weekends. Most of the houses are built from the same grey stone as the church, with slate roofs that shine when it rains, which is often. In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley are known for being reserved with strangers and generous with neighbours. When a boat is late they gather quietly at the harbour, and when it comes in they disperse without a word, as if nothing had happened. The school has forty pupils and two teachers, and the older children take a bus each morning to the secondary school in the larger town twelve miles inland, a journey that takes forty minutes along a narrow road that floods in spring. There is a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, usually to help pleasure craft that have misjudged the currents around the headland. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. Farming occupies the land behind the town, mostly sheep on the higher ground and barley in the sheltered valley, and the two communities, sea and land, have intermarried for so long that nearly everyone in Ardley can name a cousin on a boat and a cousin on a tractor. Continue this description of Ardley in the same style, adding more detail about its history, its people, its weather and the seasons, and do not stop until you are told to.
 town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station
```

### q40-qs4cx-wh, G=512 (run1 == run2)

```
The small harbour town of Ardley sits where a slow river meets a cold northern sea, and for most of its history it has lived by the tide. Fishing boats leave before dawn and return in the early afternoon, their holds filled with herring, cod and the occasional crab, and the smell of salt and diesel hangs over the quay long after the catch has been sold. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells newspapers and fishing line, and a small museum that opens only on summer weekends. Most of the houses are built from the same grey stone as the church, with slate roofs that shine when it rains, which is often. In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley are known for being reserved with strangers and generous with neighbours. When a boat is late they gather quietly at the harbour, and when it comes in they disperse without a word, as if nothing had happened. The school has forty pupils and two teachers, and the older children take a bus each morning to the secondary school in the larger town twelve miles inland, a journey that takes forty minutes along a narrow road that floods in spring. There is a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, usually to help pleasure craft that have misjudged the currents around the headland. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. Farming occupies the land behind the town, mostly sheep on the higher ground and barley in the sheltered valley, and the two communities, sea and land, have intermarried for so long that nearly everyone in Ardley can name a cousin on a boat and a cousin on a tractor. Continue this description of Ardley in the same style, adding more detail about its history, its people, its weather and the seasons, and do not stop until you are told to.
 town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind and the sun. In winter the streets are quiet and the streets are empty, but in summer the streets are alive with people, boats, and the sound of the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons
```

### q40-qs4cx-wh, G=1024 (run1 == run2)

```
The small harbour town of Ardley sits where a slow river meets a cold northern sea, and for most of its history it has lived by the tide. Fishing boats leave before dawn and return in the early afternoon, their holds filled with herring, cod and the occasional crab, and the smell of salt and diesel hangs over the quay long after the catch has been sold. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells newspapers and fishing line, and a small museum that opens only on summer weekends. Most of the houses are built from the same grey stone as the church, with slate roofs that shine when it rains, which is often. In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley are known for being reserved with strangers and generous with neighbours. When a boat is late they gather quietly at the harbour, and when it comes in they disperse without a word, as if nothing had happened. The school has forty pupils and two teachers, and the older children take a bus each morning to the secondary school in the larger town twelve miles inland, a journey that takes forty minutes along a narrow road that floods in spring. There is a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, usually to help pleasure craft that have misjudged the currents around the headland. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. Farming occupies the land behind the town, mostly sheep on the higher ground and barley in the sheltered valley, and the two communities, sea and land, have intermarried for so long that nearly everyone in Ardley can name a cousin on a boat and a cousin on a tractor. Continue this description of Ardley in the same style, adding more detail about its history, its people, its weather and the seasons, and do not stop until you are told to.
 town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind and the sun. In winter the streets are quiet and the streets are empty, but in summer the streets are alive with people, boats, and the sound of the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The lifeboat itself is the pride of the town, and the annual fundraising day in August, with its raffle, its cake stall and its tug of war between the fishermen and the farmers, raises more money than any other event on the calendar. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station at the end of the breakwater. The streets are quiet except for the occasional sound of a tractor or a boat, and the seasons change with the wind. The town has a lifeboat station at the end of the breakwater, staffed by volunteers who are called out a dozen times a year, and the streets are always quiet except for the sound of the wind. The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two
```
