# Measurement 94: sitting 2 on `R3CY10WM83Y` — anchor cells (12 × A) + the #87 DMA trace / replay

Branch `htp/94-sitting2-anchor-trace` @ `2a75f7d9` (= `origin/htp_moe` head at
build time: the PR #86 merge commit on top of `0038cce2`) — estimated device
time: **≈ 56 min** (budget 90).

Plan: `docs/plans/94-sitting-2.md`. Issue #94 (tracker #76; folds #91).

**Re-issue after #97 (2026-09-22).** The first sitting's skel did not load
(Deviation 1, `0x80000406`): `test/htp/build.sh` @ `2a75f7d9` left
`hmx/hexkl_dma_trace.c` out of `SRCS`, so the skel carried seven undefined
`hexkl_dma_trace_*` symbols that the linker accepts and the on-device
loader rejects (#97). Only `$W/libnntr_hvx_skel.A.so` is re-staged, built
from `htp/97-skel-undefined-symbols` @ `a03e2542` (= `htp_moe` +
`build.sh` fix; `git diff --stat 2a75f7d9..a03e2542 -- Applications
nntrainer test/unittest test/jni` is empty, so the IDL, stub, app and
gtests are unchanged and the ARM binaries below remain the `2a75f7d9`
build — rule 3 holds). What remains of this sitting: block 2's **6 NPU
cells**, block 3 (B), block 4, block 5, block 6 (C) — and block 2's 6 CPU
control cells are re-run in the same sitting anyway (rule 9: every sitting
starts with the control binary; the earlier CPU rows under Results stay as
the previous-day reference). The sitting's first device command after the
push is block 5's `*DmaProbeShapes*`: `[  PASSED  ]` there is #97's
acceptance; another `0x80000406` means a stale skel on the device (check
the two `md5sum` lines) or a new loader fault (new issue, attach logcat).
Estimated device time unchanged, **≈ 56 min**.

**Measured 2026-09-22 13:40–14:35 KST — every block ran.** See Results
below: `DmaProbeShapes` passed as #97's acceptance, and all 12 A cells, B
at levels 2 and 3, the block-4 re-run, both gtests and all of C are
filled. Two things carry forward: the unit was `R3CY205ZMND` again, not
`R3CY10WM83Y` (Deviation 2 — so this is still not the second-unit
anchor), and `MoeChunkReplay` fails its content check on all 11 cells
(Deviation 6, a new defect).

**Variant C (`NNTR_MOE_HTP_M1_GEMV=1`) is in this sitting — C-ON.** PR #86
merged into `htp_moe` as `2a75f7d9` while the first (C-OFF, `0038cce2`)
artifact set was being staged; per the plan §3.2 rule the grep was re-taken
before staging: `gh pr view 86 --json state --jq .state` → `MERGED`, and
`git grep -q NNTR_MOE_HTP_M1_GEMV origin/htp_moe -- nntrainer/tensor/htp_backend/htp_compute_ops.cpp`
→ match. Everything below was rebuilt from `2a75f7d9` (skel + stub + app +
both gtests from one commit, rule 3); the `0038cce2` set was discarded.

**A on this head is "PR head + #86 off + #93 trace off"**: the same merged
binary with the switch unset. `#86` adds one FastRPC call per session
(`moe_set_opts`, sent once) and a stderr line; with the switch off the DSP
runs exactly the HMX block loop, and `#93`'s trace hooks are under
`hexkl_probe_on` (host line `PROFILE OFF: TRACE UNTOUCHED`). So an
A-vs-#77 difference is unit / build, not code. Every NPU A log on this
build prints **`[HTP] moe m1 gemv: off (applied=0x0)`** once, and A's
level-2 `[HTP-PROFILE]` M==1 row reads `blocks=5632 m1_gemv=0/1408`;
`on (applied=0x1)` in an A/B log means the env var leaked → void the run.
C = the same binary with `NNTR_MOE_HTP_M1_GEMV=1`: `[HTP] moe m1 gemv: on
(applied=0x1)` in every C log and `blocks=0 m1_gemv=1408/1408` in the C
level-2 M==1 row are the proof fields (plan 80). A C run whose text differs
from A fails the accuracy gate and is filed, not averaged (the GEMV's int32
accumulator is bit-identical to the HMX path: host line `M1 GEMV PATH
BIT-IDENTICAL TO HMX PATH`, device gtest `MoeLayerM1GemvMatchesHmx`).

## Why

Every "now" number in `docs/htp_moe/BENCHMARK.md` still comes from
`R3CY205ZMND` (#77) or the PR author's device; this sitting anchors the 12
A cells (CPU `q40` and NPU `q40-qs4cx-wh`, prompt 512, G 64 / 512 / 1024
× 2 runs) on **our** unit `R3CY10WM83Y` so the supervisor can record the
unit ratio per cell (rule 13) and replace "now". In the same sitting the
#87 instrumentation (`DMA ring:` line, `[HTP-DMA]` per-descriptor dumps,
`MoeChunkReplay`) runs for the first time on silicon, and its lines fill
the plan 87 §3.2 attribution table a–g — which names the first wall-2 fix
issue (why the in-situ weight DMA runs at 16–18 GB/s where the isolated
probe does 72–117).

There is exactly one binary set (A). B is A's binary with `NNTR_HTP_PROFILE`
set (never read for tok/s, rule 1); C is A's binary with
`NNTR_MOE_HTP_M1_GEMV=1` (LEDGER ⑯: the first lever against `lfm2_moe`
42 ms/token — MoE DSP 1.35 → ≈ 0.3 ms/call if the arena read keeps up);
the gtests are a ride-along. No `--profile` build this time (plan §3.1).

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

All from `htp/94-sitting2-anchor-trace` @ `2a75f7d9`, built in the worktree
`/home/j2z0-lee/nntrainer-94` on 2026-09-22 (one skel, the stub regenerated
from the IDL on this head — it now carries `moe_set_opts`, and the MoE
stage table has 30 slots on both sides). Staged copies — what every push
line below uses — are under **`W=/local/mnt/workspace/htp_moe/94/`**;
`$W/md5.txt` is the `md5sum` of every staged file and is reproduced here:

| file (staged as) | md5 | built with / from |
|---|---|---|
| `$W/libnntr_hvx_skel.A.so` (→ both device dirs as `libnntr_hvx_skel.so`) | `ce7bb5127a4d1999a306c9512dc8244d` (172,656 B; re-staged 2026-09-22 13:28) | `./test/htp/build.sh` @ `a03e2542` (`htp/97-skel-undefined-symbols`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)`). Replaces `20fb9801…` (168,432 B, built @ `2a75f7d9`, did not load — #97) |
| `$W/tps/nntrainer_causallm` | `dc3f4b8f589c3d7e665eb3cc97f3b715` | `build_android.sh --htp` then `--htp --cache` @ `2a75f7d9`; `strings … \| grep -c 'per-layer-type totals'` = 0 |
| `$W/tps/libcausallm_core.so` | `68574a5c3e23e3cd8862d4b99c53dcd7` | same (`jni/libs/arm64-v8a/`) |
| `$W/tps/libnntrainer.so` | `b74ae8227a8415486d2f86b6797c602f` | same (`jni/obj/local/arm64-v8a/`); NEEDED lists `libsdkl.so`, `libcdsprpc.so` |
| `$W/tps/libccapi-nntrainer.so` | `b11b0ef11562fc2d4c616c3b42a91019` | same (`jni/obj/local/arm64-v8a/`) |
| `$W/gtest/unittest_hvx_dma_probe` | `26db4bfef033d77720c880c3af98379e` | `ndk-build … unittest_hvx_dma_probe` @ `2a75f7d9` (`kMoeStages = 30`) (`test/jni/obj/local/arm64-v8a/`) |
| `$W/gtest/unittest_hvx_mm_u8i4` | `51ba46f580d83ac0dc05fbfaad32faf6` | same build; block 6 runs its `*MoeLayerM1GemvMatchesHmx*` once |
| `$W/tps/libc++_shared.so`, `$W/gtest/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77) |
| `$W/tps/libsdkl.so`, `$W/gtest/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77) |
| `$W/prompt512.txt` = `docs/measurements/77-prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | copied verbatim from `origin/htp/77-first-handoff` (`e909d113`); 512 ids with `hf/tokenizer.json` (checked below) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40/nntr_lfm2_8b_a1b_q40_arm.bin` (4768855808 B) | `d28f55c5bd7adeb8bf73b02de582eb88` | #78 (CPU control); re-hashed on the workstation 2026-09-22 |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model); re-hashed 2026-09-22 |
| `tokenizer.json` (identical in `hf/`, `q40/`, `q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

`nntrainer_causallm` (`dc3f4b8f…`), `libcausallm_core.so` (`68574a5c…`) and
`libccapi-nntrainer.so` (`b11b0ef1…`) are byte-identical to the discarded
`0038cce2` build: #86 touches `nntrainer/tensor/htp_backend/` (→
`libnntrainer.so` `b74ae822…`), the IDL / stub, the skel and `test/`. The
four files that carry #86 are the skel, `libnntrainer.so` and both gtests —
those are the md5s that must differ from any #77-era or `0038cce2` copy on
the device.

Only the skel row changed in the re-issue: the #97 fix is one `SRCS` line
plus a post-link check in `test/htp/build.sh`, no IDL / `htp_backend` /
app / `test/jni` source moved between `2a75f7d9` and `a03e2542`, so the
stub compiled into `libnntrainer.so` and both gtests is the one this skel
was generated from. `$W/md5.txt` was regenerated; `md5sum -c` all OK.

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5); the
`nntr_quantize` / `nntr_safetensors_info` binaries of the same build.

Workstation sanity before pushing (all must hold):

```
W=/local/mnt/workspace/htp_moe/94; M=/local/mnt/workspace/models/lfm2.5-8b-a1b
(cd $W && md5sum -c md5.txt)                                                   # every line OK
strings $W/tps/nntrainer_causallm | grep -c 'per-layer-type totals'            # 0 (not a profile binary)
find $W -name 'libcdsprpc*' | wc -l                                            # 0
md5sum docs/measurements/77-prompt512.txt                                      # fc65c1588dc66dd764c7013fe96cbb75
python3 -c "from tokenizers import Tokenizer; t=Tokenizer.from_file('$M/hf/tokenizer.json'); print(len(t.encode(open('docs/measurements/77-prompt512.txt').read()).ids))"   # 512
```

## Steps (workstation, phone on local USB — no SSH bridge this time)

Shell setup once (the worktree is left in place for this sitting):

```
cd /home/j2z0-lee/nntrainer-94 && git fetch && git checkout htp/94-sitting2-anchor-trace && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/94; M=/local/mnt/workspace/models/lfm2.5-8b-a1b; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
```

Local `adb`: the device-side `sed` one-liners and the `"$(cat prompt512.txt)"`
quoting work as written (the #77 deviation 6 was the SSH bridge). If a
`grep -H` echo shows an edit did not take, edit the two json files on the
workstation and `adb push` them (the #77 fallback), then re-echo.

### 0. Device state (2 min)

```
adb devices                     # R3CY10WM83Y device
therm                           # battery %, temperature (tenths of °C), thermal_zone0 (m°C)
adb shell df -h /data           # >= 10 GB free if the models must be pushed
```

Note battery %, °C, warm/cool under "Notes from the run" (checkpoint 1 of
4). Screen off, charger in.

### 1. Install (5–15 min; 5 if the models are already there)

**Models present?** #77 ran on the other unit, so on `R3CY10WM83Y` expect
them absent. Sizes must read exactly `4768855808` and `4316133120`:

```
adb shell ls -l $D/models/q40/nntr_lfm2_8b_a1b_q40_arm.bin $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
```

If **both** exist with those sizes, hash them (≈ 1 min each) and skip the
push when they read `d28f55c5bd7adeb8bf73b02de582eb88` / `7b7867fab51845664c0050c0a837073e`:

```
adb shell md5sum $D/models/q40/nntr_lfm2_8b_a1b_q40_arm.bin $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
```

Otherwise push both model dirs (≈ 5–8 min, 9 GB):

```
adb shell mkdir -p $D/models $T
adb push $M/q40 $D/models/q40
adb push $M/q40-qs4cx-wh $D/models/q40-qs4cx-wh
```

Binaries (always; the device may hold #77-era or #86-rebase files):

```
adb shell mkdir -p $D/models $T
adb push $W/tps/nntrainer_causallm $W/tps/libcausallm_core.so $W/tps/libnntrainer.so $W/tps/libccapi-nntrainer.so $W/tps/libc++_shared.so $W/tps/libsdkl.so $D/
adb push $W/libnntr_hvx_skel.A.so $D/libnntr_hvx_skel.so
adb push $W/libnntr_hvx_skel.A.so $T/libnntr_hvx_skel.so
adb push $W/gtest/unittest_hvx_dma_probe $W/gtest/unittest_hvx_mm_u8i4 $W/gtest/libc++_shared.so $W/gtest/libsdkl.so $T/
adb push $W/prompt512.txt $D/prompt512.txt
adb shell "chmod 755 $D/nntrainer_causallm $T/unittest_hvx_dma_probe $T/unittest_hvx_mm_u8i4"
```

Never push `builddir/.../libcdsprpc.so`; the device's own is the right one.

**Config edits — re-applied unconditionally** (plan 77 §3.2; the last
sitting on any unit leaves `num_to_generate` at some G and this unit's
state is unknown): greedy decoding and an exact generation count.

```
for m in q40 q40-qs4cx-wh; do
  adb shell "cd $D/models/$m && \
    sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
    sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
    grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|moe_engine|moe_htp_layers' nntr_config.json"
done
# Guard (a no-op on the #78 output, which carries both keys): the app
# defaults moe_engine to "cpu" and QS4CX_WH has no CPU path.
adb shell "cd $D/models/q40-qs4cx-wh && grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json; grep -H -E 'moe_engine|bad_word_ids' nntr_config.json"
```

Expected echo per model: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`; the WH dir shows `moe_engine": "htp"` and
`moe_htp_layers": ""`, the `q40` dir shows neither key. Anything else:
stop, fix (workstation edit + push), re-echo. Paste the echo under Notes.

Provenance lines to paste under "Notes" (rule 14 — the pushed build is the
provenance; every hash must equal the Artifacts table):

```
adb shell "md5sum $D/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $T/unittest_hvx_dma_probe $T/unittest_hvx_mm_u8i4 $D/prompt512.txt $D/models/q40/tokenizer.json $D/models/q40-qs4cx-wh/tokenizer.json"
```

Both skel lines must read `ce7bb5127a4d1999a306c9512dc8244d` (the #97
rebuild). `20fb9801…` in either dir is the stale skel that fails with
`0x80000406` — re-push before running anything.

### 2. Control runs, A (≈ 25 min) — 12 cells, CPU first

`therm` once more right before the first run (checkpoint 1 is block 0; if
more than a few minutes passed, note both). One helper, defined in the
workstation shell (the #77 `runA`, staging dir changed):

```
runA() { # runA <model> <G> <run#>
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/$1/nntr_config.json && \
    grep num_to_generate models/$1/nntr_config.json && \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$1 \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/A_$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)'
}
```

Order — CPU first (no registration, and its decode rate is the thermal
gate), then NPU; 60 s pause before each NPU G=1024 run; no other pauses:

```
runA q40 64 1;  runA q40 64 2
runA q40 512 1; runA q40 512 2
runA q40 1024 1; runA q40 1024 2
runA q40-qs4cx-wh 64 1;  runA q40-qs4cx-wh 64 2
runA q40-qs4cx-wh 512 1; runA q40-qs4cx-wh 512 2
sleep 60; runA q40-qs4cx-wh 1024 1; sleep 60; runA q40-qs4cx-wh 1024 2
therm                                                        # checkpoint 2 of 3
```

**Thermal gate:** CPU (`q40`) decode < 30 tok/s ⇒ throttling (doc 49 §6)
→ 5 min screen-off cool-down, redo that cell as run 1b/2b, record both.

Expected lines in every log, exactly: `prefill: 512 tokens, … TPS`,
`generation: <G> tokens, … TPS`, `total: … ms`, `peak memory: … KB`. There
is **no** `generation(last 64)` line for LFM2-MoE on this tree (#89, rule
16): the column stays `n/a`. `prefill: N` with N ≠ 512 voids the prefill
column of the sitting (tokenizer mismatch — compare the `tokenizer.json`
md5s); `generation: N` with N ≠ G is a failed step (config edit did not
take), not a data point. A `[HTP-PROFILE]` block in an A log means the
env var leaked from a B shell: void the run. Every NPU (`q40-qs4cx-wh`) A
log carries exactly one `[HTP] moe m1 gemv: off (applied=0x0)` line; `on
(applied=0x1)` in an A log means `NNTR_MOE_HTP_M1_GEMV` leaked from the C
shell: void the run. (`q40` logs carry no such line — the CPU path never
opens a DSP session.) Check both at once after block 2:

```
grep -c 'moe m1 gemv: off (applied=0x0)' $W/logs/A_q40-qs4cx-wh_G*_r*.log      # 1 each
grep -l 'moe m1 gemv: on\|HTP-PROFILE' $W/logs/A_*.log                          # nothing
```

Text comparison on the workstation (fills "text run1 = run2"; the CPU and
NPU texts differ from each other by construction — expert quantisation —
and both are recorded):

```
for m in q40 q40-qs4cx-wh; do for g in 64 512 1024; do
  printf '%s G=%s: ' $m $g; diff <(sed -n '/^=====/q;p' $W/logs/A_${m}_G${g}_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_${m}_G${g}_r2.log) >/dev/null && echo same || echo DIFFERENT
done; done
```

### 3. B — the #87 trace at levels 2 and 3 (≈ 6 min) — NPU model, G=64, same binary

`NNTR_HTP_PROFILE=2` = one timed call per layer call (cold);
`=3` = 5 repeats, fastest kept (warm; ≈ 5× longer on the DSP side, ≈ 2
min). `NNTR_HTP_DMA_TRACE` stays at its default 3 (the first 3 timed calls
of each `(K, N_out, M==1)` bucket are dumped per descriptor). Capture the
**whole** logs; the terminal filter shows only the headers.

```
adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && grep num_to_generate models/q40-qs4cx-wh/nntr_config.json"
adb shell "cd $D && NNTR_HTP_PROFILE=2 NNTR_M0_PROFILE=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/B_level2.log | grep -E '^\[HTP-PROFILE\]|^\[M0-PROF\]|^\[HTP-DMA\] call=[0-9]+ M=[0-9]+ ' | head -60
adb shell "cd $D && NNTR_HTP_PROFILE=3 NNTR_M0_PROFILE=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/B_level3.log | grep -E '^\[HTP-PROFILE\]|^\[M0-PROF\]|^\[HTP-DMA\] call=[0-9]+ M=[0-9]+ ' | head -60
```

Extract, per level, what the Results section pastes (the `[HTP-DMA]` push
and wait lines carry no `M=`, so the M=1 blocks are cut by their header):

```
for L in 2 3; do
  grep -E '^\[HTP-PROFILE\]' $W/logs/B_level$L.log > $W/logs/B_level${L}_profile.txt          # header, M>1 + M==1 rows, weight DMA:, DMA ring:
  grep -E '^\[M0-PROF\]' $W/logs/B_level$L.log | head -3 > $W/logs/B_level${L}_m0.txt
  grep -E '^\[HTP-DMA\] call=[0-9]+ M=[0-9]+ ' $W/logs/B_level$L.log > $W/logs/B_level${L}_dma_headers.txt    # 6 headers: 3 M=1, 3 M>1
  awk '/^\[HTP-DMA\] call=[0-9]+ M=[0-9]+ /{p=($0 ~ / M=1 /)} p && /^\[HTP-DMA\]/' $W/logs/B_level$L.log > $W/logs/B_level${L}_dma_m1.txt
  printf 'level %s: profile lines %s, dma headers %s, M=1 dump lines %s\n' $L $(wc -l < $W/logs/B_level${L}_profile.txt) $(wc -l < $W/logs/B_level${L}_dma_headers.txt) $(wc -l < $W/logs/B_level${L}_dma_m1.txt)
done
```

Expected: `qos_mode=2` at both levels; the M==1 row `K=2048 N=2048 M==1
calls=1408 … blocks=5632 m1_gemv=0/1408` (A's path: HMX loop, switch off); one `weight DMA:` and one `DMA
ring:` line under each of the M>1 and M==1 rows (the M>1 `DMA ring:` may
read `busy=n/a (trace truncated past 512 pushes)` — a fact, not a
failure); 3 `[HTP-DMA] call=<k> M=1 rep=1/1 desc=46 …` headers at level 2
and `rep=5/5` at level 3, each followed by 46 `push` lines and the `wait`
lines (≈ 60 lines per block). Every M=1 dump is the first decode token's
layer 0 / 1 / 2. A `[HTP-DMA] call=… moe_dma_trace_read err=…` line means
an older skel: stop, re-push `libnntr_hvx_skel.A.so`, redo block 3.

### 4. A re-run (2 min) — profile unset, NPU, G=64

```
runA q40-qs4cx-wh 64 3
printf 'WH G=64 r3 vs r1: '; diff <(sed -n '/^=====/q;p' $W/logs/A_q40-qs4cx-wh_G64_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_q40-qs4cx-wh_G64_r3.log) >/dev/null && echo same || echo DIFFERENT
grep -c 'HTP-PROFILE' $W/logs/A_q40-qs4cx-wh_G64_r3.log                       # 0
```

Prefill and decode must sit within the sitting's own noise of block 2's
two G=64 WH cells (the trace runs left no residue — plan 87 §1 standing
gate); text = run 1.

### 5. Device gtests (3 min) — `MoeChunkReplay` + `DmaProbeShapes`, A skel

The replay test reads no env var and no argument; its 11-cell matrix is
compiled in. `TwoReaderDdr` is excluded on purpose (rule 12, #90).

```
adb shell "cd $T && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*:*DmaProbeShapes*'" 2>&1 | tee $W/logs/G_replay.log | grep -E 'md5|^DMA_REPLAY_NOTE|^DMA_REPLAY_TRACE dsp_us|^DMA_REPLAY workers|^DMA_PROBE|PASSED|FAILED'
grep -c '^DMA_REPLAY_TRACE push' $W/logs/G_replay.log                            # 46
grep -c '^DMA_REPLAY workers'    $W/logs/G_replay.log                            # 11
grep -c '^DMA_PROBE shape'       $W/logs/G_replay.log                            # 16
grep -cE 'checksum_ok=n|plan_shape_ok=n|pace=0 fresh=1|pace=0 fresh=0 gap_us=600' $W/logs/G_replay.log   # 0
therm                                                                            # checkpoint 3 of 4 (plan order: gtests before C)
```

Expected: one `DMA_REPLAY_TRACE dsp_us=… desc=46 waits=… blocked=…
wait_us=… wait_act_us=… busy_us=lo..hi depth_max=… first_ready_us=…
last_issue_us=… trace_words=… plan_shape_ok=y` line, 46 `DMA_REPLAY_TRACE
push` lines and the `wait` lines, then **11 `DMA_REPLAY` lines** (the
matrix below) with `bytes_per_call=22020096` and `checksum_ok=y` on every
line, then 16 `DMA_PROBE` lines (`vote=1`, `checksum_ok=y`), and
`[  PASSED  ] 2 tests.`

**Stop signal:** a `DMA_REPLAY_NOTE moe_layer_timed err=0x8000040e` line
means the skel and the gtest disagree on the stage count (`kMoeStages = 30`
on this head = the skel's `MOE_N_STAGES`, 29 before #86) — a stale artifact. Stop, re-push
`$W/libnntr_hvx_skel.A.so` to `$T/libnntr_hvx_skel.so` and
`$W/gtest/unittest_hvx_dma_probe`, check the md5 line, redo block 5. Rows
7–11 with such a note silently ran at `pace=0` (the `pace=` field on the
line shows it) and are marked "fell back", not read.

### 6. C — `NNTR_MOE_HTP_M1_GEMV=1` (≈ 9 min) — NPU model, same binary, 6 cells + one level-2 run + one gtest

Order is the plan's: A, B, A re-run, gtests, then C. `therm` after
`runC 1024 2` is checkpoint 4. Same helper as `runA` with the switch set and `C_` logs:

```
runC() { # runC <G> <run#>   (NPU model only)
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $1/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && \
    NNTR_MOE_HTP_M1_GEMV=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/C_q40-qs4cx-wh_G$1_r$2.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv'
}
runC 64 1;  runC 64 2
runC 512 1; runC 512 2
sleep 60; runC 1024 1; sleep 60; runC 1024 2
therm                                                        # checkpoint 4 of 4
```

Then the proof run (level 2, G=64, never read for tok/s) and the gtest:

```
adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && NNTR_MOE_HTP_M1_GEMV=1 NNTR_HTP_PROFILE=2 NNTR_M0_PROFILE=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/C_level2.log | grep -E '^\[HTP-PROFILE\]|moe m1 gemv' | head -20
adb shell "cd $T && md5sum libnntr_hvx_skel.so unittest_hvx_mm_u8i4 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*'" 2>&1 | tee $W/logs/G_m1gemv.log | grep -E 'md5|U8I4_FIELD|PASSED|FAILED|SKIPPED'
```

Checks on the workstation:

```
grep -c 'moe m1 gemv: on (applied=0x1)' $W/logs/C_*.log                          # 1 each (7 files)
grep 'M==1' $W/logs/C_level2.log | grep -o 'blocks=[0-9]* m1_gemv=[0-9/]*'         # blocks=0 m1_gemv=1408/1408
grep 'M==1' $W/logs/B_level2.log | grep -o 'blocks=[0-9]* m1_gemv=[0-9/]*'         # blocks=5632 m1_gemv=0/1408 (A, for the dsp= comparison)
for g in 64 512 1024; do for r in 1 2; do
  printf 'C vs A G=%s r%s: ' $g $r; diff <(sed -n '/^=====/q;p' $W/logs/A_q40-qs4cx-wh_G${g}_r${r}.log | grep -v 'moe m1 gemv') <(sed -n '/^=====/q;p' $W/logs/C_q40-qs4cx-wh_G${g}_r${r}.log | grep -v 'moe m1 gemv') >/dev/null && echo same || echo DIFFERENT
done; done
```

Expected: every C log has `[HTP] moe m1 gemv: on (applied=0x1)` and the
four report lines with `generation: <G> tokens`; `C_level2.log`'s M==1 row
shows `blocks=0 m1_gemv=1408/1408` (the M>1 row keeps `m1_gemv=0/<calls>`
— prefill stays on the HMX loop); the gtest prints `U8I4_FIELD
path=moe_m1_gemv field=bad_elems_M<m> value=0 of …` per M and
`field=bit_identical value=yes`, then `[  PASSED  ] 1 test.` A
`nntr_hvx_moe_set_opts failed` exception on the first C run = stale skel
(rule 3): stop, re-push `$W/libnntr_hvx_skel.A.so` to `$D`, redo. **C text
≠ A text at any G is a failed accuracy gate**: record the first differing
token, do not average, file it (handoff skill).

### 6b. Optional ride-along — only if ≥ 20 min of the 90 remain, "not a variant of record"

```
adb shell "cd $D && NNTR_NUM_THREADS=4 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/X_threads4_G64.log | grep -E '^(prefill|generation|total|peak memory)'
```

(Set G back to 64 first if block 6 left it at 1024: the block-3 `sed`
line.) A data point for LEDGER ⑰'s over-splitting
candidate; no gate here, skip without note if the clock says so.

### 7. Finish (3 min)

Budget check: 2 + 5–15 + 25 + 6 + 2 + 3 + 9 + 3 ≈ 56 min (models present) —
66 with the push.

Fill the tables below, texts to the appendix, `ls $W/logs` into the
appendix, then on the branch:
`git add docs/measurements/94-sitting2-anchor-trace.md && git commit -s`,
`git push`, set #94 to `state:measured`.
## Results (filled — second attempt, 2026-09-22 13:40–14:35 KST, after #97/#98)

Unit **`R3CY205ZMND`** (SM-S938N), not `R3CY10WM83Y` (Deviation 2, unchanged
from the first attempt). Remote SSH-to-adb bridge, not local USB
(Deviation 3). Skel rebuilt on the bridge-reachable workstation from
`htp_moe` @ `08afbb10` (Deviation 5). All blocks of the sitting ran: A
(12 cells), B (levels 2 and 3), the block-4 A re-run, block 5 (both
gtests) and C (6 cells + proof run + gtest). One new defect found —
`MoeChunkReplay`'s content check (Deviation 6).

Battery at start: 100 % (USB powered)   phone: cool (battery 28.0 °C,
`thermal_zone0` 29.0 °C)   date/time: 2026-09-22 13:40 KST.
Checkpoint 2 (after block 2, 12 A cells): 100 % / 36.5 °C / 51.2 °C.
Checkpoint 3 (after block 5, gtests): not sampled separately — the gtests
ran 3 min after checkpoint 2's reading and took 1.1 s of DSP time.
Checkpoint 4 (after block 6, C): 100 % / 37.9 °C / 55.0 °C.

### Deviations

1. **(Closed) DSP skel from `2a75f7d9` fails to load on-device** —
  `0x80000406` / `dlerror RX VA 0xFFF00000 outside ELF segment`, the
  blocker of the first attempt. Root cause `hmx/hexkl_dma_trace.c` missing
  from `SRCS` in `test/htp/build.sh`, fixed by #97 / PR #98. **Verified on
  silicon this sitting**: the skel built with the fix loads and runs —
  block 5's `DmaProbeShapes` was the first device command after the push
  and printed `[  PASSED  ] 1 test.` with 16 `checksum_ok=y` lines, and
  every NPU app run below opened its DSP session without error. Nothing of
  the sitting is blocked by it any more.
2. **Wrong unit.** Handoff specifies `R3CY10WM83Y`; the only reachable
  device was again `R3CY205ZMND` (the #77 unit). So this sitting still is
  **not** the second-unit anchor: the A cells are a same-unit re-run of
  #77's cells on a different build, and the `R3CY205ZMND` ↔ `R3CY10WM83Y`
  ratio of rule 13 cannot be computed from it. Everything that does not
  need a second unit — B, the replay/probe matrices, the attribution table
  and the C A/B (LEDGER ⑯) — is complete and unit-independent in the sense
  that A and C ran back to back on the same unit in the same sitting.
3. **Remote SSH bridge instead of local USB.** The bridge exposes only
  `shell <cmd>` / `push <path>` (stdin) / `pull <path>` (stdout). All
  commands are the bridge-equivalent of the documented `adb` ones; the
  `sed`-and-run one-liners were pushed as device-side scripts
  (`$D/dev_runA.sh`, `dev_runB.sh`, `dev_runC.sh`, `dev_runCP.sh`,
  `dev_cfg.sh`, `dev_therm.sh`) to avoid the bridge's quoting, so the
  command *content* is identical to the handoff's.
4. **(Closed)** The first attempt's blocks 3, 5 and 6 "not attempted" —
  all ran this sitting.
5. **Artifact provenance: the skel is this workstation's build, not the
  Artifacts table's.** The staged/pushed skel is
  `d6568c8bb95e755485e982811e8db755` (172,624 B), built here by
  `./test/htp/build.sh` from `origin/htp_moe` @ `08afbb10` (= the #98 merge
  `14f65120` plus the #96 docs commit; `git diff 2a75f7d9 08afbb10`
  touches only `docs/` and `test/htp/build.sh`, so the DSP sources are
  bit-identical to `2a75f7d9` and the fix is the one `a03e2542` carries),
  SDK 6.4.0.1 / toolv19 / HexKL `lib/6.4.0.1` / v79, `UNDEFINED SYMBOLS OK
  (46 runtime imports)`. It is **not** the table's `ce7bb512…` (172,656 B,
  32 B larger) because that was linked on the implementer's machine; the
  ARM binaries are unchanged from the first attempt's local build
  (`nntrainer_causallm 055703b2…`, `libnntrainer.so a508e889…`,
  `libcausallm_core.so 8057cea3…`, `unittest_hvx_dma_probe 76c43d3e…`,
  `unittest_hvx_mm_u8i4 e79adbd0…`), which likewise differ from the table
  for the same reason while `libc++_shared.so b1586b9b…` and
  `libsdkl.so 0ad4e22a…` match it exactly. Under rule 14 the pushed-build
  provenance is intact, so the A/CPU cells are a note — but **C is
  included in this sitting, so rule 14's "void once C is included" applies
  to the letter**: the C A/B below is reported as evidence with its
  provenance stated, and the supervisor decides whether it counts as the
  LEDGER ⑯ verdict or must be re-run from the implementer's artifact set.
  A and C share one binary set and one sitting, so the A/B itself is
  internally consistent regardless.
6. **New defect: `MoeChunkReplay` fails its content check on every cell.**
  All 11 cells ran and produced self-consistent timings, but every line
  ends `checksum_ok=n` and the test fails at
  `unittest_hvx_dma_probe.cpp:631` with the same numbers in all 11 cells:
  `res[6]` = `1467840` vs `want_sum` = `9461760` (ratio 6.446). The
  replay also reports `bytes_per_call=22560768` where the handoff expects
  `22020096` (+540,672 B = +2.5 %) and `regions=32`. `DmaProbeShapes` in
  the same binary passes with `checksum_ok=y` on all 16 lines, so the DMA
  path and the skel are fine; the fault is in the replay harness's plan or
  its expected sum. Filed as a new issue (link added by the supervisor).
  Consequence for this sitting: the replay table is recorded and its
  *timings* are used for the attribution table, but every conclusion that
  rests on the replay alone (rows c, d, g and the new row h) is marked
  **provisional** until the checksum is explained — the in-situ `B` lines,
  which are unaffected, carry the confirmed rows.

### A — control runs (prompt 512, `NNTR_NUM_THREADS=8`, profile unset), 12 cells + the block-4 re-run

The `#77` columns are `R3CY205ZMND` (SM-S938N) values from
`origin/htp/77-first-handoff:docs/measurements/77-first-handoff.md`; same
unit as this sitting, so they are a build-to-build comparison here, not a
unit ratio. Goal line: ≥ 50 decode at every G (NPU); PR-era provisional
now 20.8 / 48 / 523.

| variant | model | G | run | prefill ms / tok/s | #77 prefill tok/s | decode tok/s (all) | #77 decode tok/s | decode tok/s (last 64) | gen tokens printed | peak RSS KB | text run1 = run2 | skel md5 (device) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | q40 | 64 | 1 | 1507 ms / 339.748 | 336.4 | 51.5713 | 54.10 | n/a (#89) | 64 | 5893184 | same | n/a (CPU) |
| A | q40 | 64 | 2 | 1624 ms / 315.271 | 253.1 | 53.2889 | 52.94 | n/a (#89) | 64 | 5887380 | same | n/a |
| A | q40 | 512 | 1 | 1912 ms / 267.782 | 297.5 | 51.108 | 52.67 | n/a (#89) | 512 | 5465680 | same | n/a |
| A | q40 | 512 | 2 | 1910 ms / 268.063 | 231.5 | 47.3329 | 51.97 | n/a (#89) | 512 | 5497972 | same | n/a |
| A | q40 | 1024 | 1 | 1637 ms / 312.767 | 287.3 | 48.4367 | 46.44 | n/a (#89) | 1024 | 5459432 | same | n/a |
| A | q40 | 1024 | 2 | 1792 ms / 285.714 | 288.0 | 48.1769 | 48.53 | n/a (#89) | 1024 | 5501172 | same | n/a |
| A | q40-qs4cx-wh | 64 | 1 | 1316 ms / 389.058 | 403.5 | 17.5198 | 19.68 | n/a (#89) | 64 | 5383216 | same | `d6568c8b…` |
| A | q40-qs4cx-wh | 64 | 2 | 1032 ms / 496.124 | 475.4 | 17.9272 | 21.27 | n/a (#89) | 64 | 5403116 | same | `d6568c8b…` |
| A | q40-qs4cx-wh | 512 | 1 | 1091 ms / 469.294 | 485.8 | 17.09 | 18.38 | n/a (#89) | 512 | 5399032 | same | `d6568c8b…` |
| A | q40-qs4cx-wh | 512 | 2 | 1011 ms / 506.429 | 494.7 | 16.9609 | 18.24 | n/a (#89) | 512 | 5147572 | same | `d6568c8b…` |
| A | q40-qs4cx-wh | 1024 | 1 | 971 ms / 527.291 | 541.2 | 16.831 | 19.04 | n/a (#89) | 1024 | 5376312 | same | `d6568c8b…` |
| A | q40-qs4cx-wh | 1024 | 2 | 1182 ms / 433.164 | 455.1 | 17.7775 | 16.96 | n/a (#89) | 1024 | 5405956 | same | `d6568c8b…` |
| A (re-run, block 4) | q40-qs4cx-wh | 64 | 3 | 1196 ms / 428.094 | — | 17.4482 | — | n/a (#89) | 64 | 5386728 | = r1 | `d6568c8b…` |

Cell means: CPU 52.43 / 49.22 / 48.31 tok/s at G 64 / 512 / 1024; NPU
17.72 / 17.03 / 17.30. NPU prefill 389–527 tok/s (mean 470).

Thermal gate tripped (CPU decode < 30)? No — 47.3–53.3 tok/s across the
six CPU cells, far above the 30 tok/s gate; `thermal_zone0` rose 29 → 51 °C
over block 2 and 55 °C by the end of C, with no decode collapse.

Standing gates:
* `moe m1 gemv: off (applied=0x0)` exactly once in each of the six NPU A
  logs; `grep -l 'moe m1 gemv: on\|HTP-PROFILE' logs/A_*.log` → nothing
  (no env leak from the B or C shells into any A run).
* `prefill: 512 tokens` and `generation: <G> tokens` in every one of the
  13 logs (no voided prefill column, no failed config edit).
* Block 4's re-run after B: 17.4482 tok/s vs block 2's 17.5198 / 17.9272
  (−0.4 % / −2.7 %), prefill 428.1 vs 389.1 / 496.1, `HTP-PROFILE` count 0,
  text = r1 → the trace runs left no residue (plan 87 §1 standing gate).

Text comparison (block 2 `diff` loop and block 4):

```
q40            G=64     same
q40            G=512    same
q40            G=1024   same
q40-qs4cx-wh   G=64     same
q40-qs4cx-wh   G=512    same
q40-qs4cx-wh   G=1024   same
WH G=64 r3 vs r1: same
```

### B — M==1 row, `weight DMA:` and `DMA ring:` per level (NPU, G=64)

Reference (#77, `R3CY205ZMND`): level 2 host 1948.6 / dsp 1360.9 /
transport 587.7, gather 143.5, drain 108.7+7.3, mm 748.0, 21504 KB/call,
avg 16.2 GB/s; level 3 1750.4 / 1222.7 / 527.7, gather 122.6, drain
0.9+0.4, mm 746.6, avg 18.0. The `DMA ring:` fields have no reference —
this sitting is their first.

| level | qos_mode | calls | host µs/call | dsp µs/call | transport µs/call | gather | drain+dn | mm | blocks | weight DMA KB/call | first KB / us / GB/s | avg GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 (cold, `rep=1/1`) | 2 | 1408 | 2062.5 | 1414.0 (68.6 %) | 648.6 | 146.5 | 113.1+8.9 | 783.4 | 5632 | 21504 | 1024 / 0 / 6964.1 | 15.6 |
| 3 (warm, `rep=5/5`) | 2 | 1408 | 1939.1 | 1271.6 (65.6 %) | 667.5 | 124.8 | 2.4+1.3 | 781.9 | 5632 | 21504 | 1024 / 0 / 7201.9 | 17.3 |

Within 4 % of #77 on every column (host +5.8 % / +10.8 %, dsp +3.9 % /
+4.0 %, mm +4.7 %) — same unit, warmer, different build; nothing moved.

| level | `DMA ring:` desc/call | waits (blocked) | wait us | act / gu / dn us | busy lo..hi us | engine GB/s lo..hi | depth max | first expert ready at (us) | last issue at (us) | of dsp_us |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 46 | 30 (5.1) | 269.0 | 145.4 / 113.1 / 8.9 | 904..1198 | 18.4..24.3 | 11 | 165 | 1090 | 1414 |
| 3 | 46 | 30 (3.0) | 129.2 | 123.7 / 2.4 / 1.3 | 682..1005 | 21.9..32.3 | 9 | 151 | 989 | 1272 |

M>1 row per level — existing columns only, standing gate (within noise of
#77 B × the unit ratio: #77 level 2 `calls=23 host=18172.5 dsp=15941.3
transport=2231.2 … mm 9175.8 blocks=1105`, level 3 `17824.5 / 15701.7 /
2122.9 … mm 9184.0`):

| level | calls | host µs/call | dsp µs/call | transport µs/call | mm | blocks | `weight DMA:` avg GB/s | `DMA ring:` busy |
|---|---|---|---|---|---|---|---|---|
| 2 | 23 | 18888.4 | 16575.9 | 2312.6 | 9600.7 | 1105 | 9.6 | 6132..8800 us → 18.0..25.9 GB/s (wait 660.6, depth 11) |
| 3 | 23 | 18273.2 | 16192.7 | 2080.5 | 9601.3 | 1105 | 9.8 | 5357..8159 us → 19.4..29.6 GB/s (wait 421.1, depth 11) |

M>1 is +3.9 % host / +4.0 % dsp / +4.6 % mm against #77 with `blocks=1105`
and `m1_gemv=0/23` unchanged — the prefill shape did not move (gate met).

Pasted lines (from `B_level<L>_profile.txt`, `B_level<L>_m0.txt`,
`B_level<L>_dma_headers.txt`):

```
(level 2)
[HTP-PROFILE] level=2 qos_mode=2 (2=poll 1=PM 0=interrupt-driven -- every transport number below is only comparable to 34_fc_measured.md's 326us/call at qos_mode=2)
[HTP-PROFILE]   weights registered : 1408
[HTP-PROFILE]   register FastRPC   :      134.8 ms  (0.10 ms/weight)
[HTP-PROFILE]   alloc + other      :     1037.8 ms
[HTP-PROFILE]   registration total :     1172.6 ms
[HTP-PROFILE]   K=2048  N=2048  M>1     calls=23      rows=11776    host=    434.4 ms (18888.4 us/call)  dsp=16575.9 us/call (87.8%) transport= 2312.6 us/call  [quant 311.2 gather 110.0 requant 1118.4 swiglu 0.0 dequant 991.6 acc 3133.6 drain 236.3+67.5 push 46.6 scatter 82.2 alloc 159.4 stage 442.4 mm 9600.7 | rest<=276.0 (1.5% of host) blocks=1105 m1_gemv=0/23]
[HTP-PROFILE]     weight DMA: 154969 KB/call, first 1024 KB took 0 us = 3445.3 GB/s; averaged over the call 9.6 GB/s
[HTP-PROFILE]     DMA ring: desc=344/call waits=223 (blocked 9.0) wait=660.6 us [act 62.4 gu 236.3+dn 67.5] busy=6132..8800 us -> engine 18.0..25.9 GB/s  depth max=11  first expert ready at 644 us  last issue at 15740 us of 16576
[HTP-PROFILE]   K=2048  N=2048  M==1    calls=1408    rows=1408     host=   2904.0 ms ( 2062.5 us/call)  dsp= 1414.0 us/call (68.6%) transport=  648.6 us/call  [quant 14.8 gather 146.5 requant 41.9 swiglu 0.0 dequant 13.0 acc 259.9 drain 113.1+8.9 push 2.5 scatter 0.9 alloc 0.1 stage 6.3 mm 783.4 | rest<=22.7 (1.1% of host) blocks=5632 m1_gemv=0/1408]
[HTP-PROFILE]     weight DMA: 21504 KB/call, first 1024 KB took 0 us = 6964.1 GB/s; averaged over the call 15.6 GB/s
[HTP-PROFILE]     DMA ring: desc=46/call waits=30 (blocked 5.1) wait=269.0 us [act 145.4 gu 113.1+dn 8.9] busy=904..1198 us -> engine 18.4..24.3 GB/s  depth max=11  first expert ready at 165 us  last issue at 1090 us of 1414
[HTP-PROFILE] layer calls total :     3338.4 ms
[HTP-PROFILE] arm staging memcpy:        9.6 ms  (206.0 MB in+out, 22.4 GB/s) -- outside every host= above
[HTP-PROFILE] HTP host time     :     4520.7 ms (registration + layer calls + staging)
[M0-PROF] moe_layer[0] tokens=512 us=21497  setup=0 router=1339 topk=210 wksp=0 gather=0 ffn=19942 route=0 scatter=0 other=6
[M0-PROF] moe_layer[1] tokens=512 us=25947  setup=0 router=1293 topk=194 wksp=0 gather=0 ffn=24448 route=0 scatter=0 other=12
[M0-PROF] moe_layer[2] tokens=512 us=20401  setup=0 router=1292 topk=214 wksp=0 gather=0 ffn=18880 route=0 scatter=0 other=15
[HTP-DMA] call=1 M=512 rep=1/1 desc=360 waits=226 blocked=7 depth_max=11 dropped=0 end=11852.0us
[HTP-DMA] call=2 M=512 rep=1/1 desc=344 waits=222 blocked=7 depth_max=11 dropped=0 end=15819.1us
[HTP-DMA] call=3 M=512 rep=1/1 desc=374 waits=240 blocked=12 depth_max=11 dropped=0 end=15891.6us
[HTP-DMA] call=1 M=1 rep=1/1 desc=46 waits=30 blocked=4 depth_max=10 dropped=0 end=1359.5us
[HTP-DMA] call=2 M=1 rep=1/1 desc=46 waits=30 blocked=5 depth_max=10 dropped=0 end=1415.2us
[HTP-DMA] call=3 M=1 rep=1/1 desc=46 waits=30 blocked=5 depth_max=10 dropped=0 end=1399.3us

(level 3)
[HTP-PROFILE] level=3 qos_mode=2 (2=poll 1=PM 0=interrupt-driven -- every transport number below is only comparable to 34_fc_measured.md's 326us/call at qos_mode=2)
[HTP-PROFILE]   K=2048  N=2048  M>1     calls=23      rows=11776    host=    420.3 ms (18273.2 us/call)  dsp=16192.7 us/call (88.6%) transport= 2080.5 us/call  [quant 310.3 gather 102.0 requant 1127.9 swiglu 0.0 dequant 1023.6 acc 3136.3 drain 41.3+15.1 push 45.8 scatter 54.0 alloc 0.3 stage 457.6 mm 9601.3 | rest<=277.0 (1.5% of host) blocks=1105 m1_gemv=0/23]
[HTP-PROFILE]     weight DMA: 154969 KB/call, first 1024 KB took 0 us = 3445.3 GB/s; averaged over the call 9.8 GB/s
[HTP-PROFILE]     DMA ring: desc=344/call waits=223 (blocked 3.4) wait=421.1 us [act 53.0 gu 41.3+dn 15.1] busy=5357..8159 us -> engine 19.4..29.6 GB/s  depth max=11  first expert ready at 645 us  last issue at 15515 us of 16193
[HTP-PROFILE]   K=2048  N=2048  M==1    calls=1408    rows=1408     host=   2730.2 ms ( 1939.1 us/call)  dsp= 1271.6 us/call (65.6%) transport=  667.5 us/call  [quant 22.2 gather 124.8 requant 41.9 swiglu 0.0 dequant 7.5 acc 260.0 drain 2.4+1.3 push 2.5 scatter 0.9 alloc 0.1 stage 6.2 mm 781.9 | rest<=20.0 (1.0% of host) blocks=5632 m1_gemv=0/1408]
[HTP-PROFILE]     weight DMA: 21504 KB/call, first 1024 KB took 0 us = 7201.9 GB/s; averaged over the call 17.3 GB/s
[HTP-PROFILE]     DMA ring: desc=46/call waits=30 (blocked 3.0) wait=129.2 us [act 123.7 gu 2.4+dn 1.3] busy=682..1005 us -> engine 21.9..32.3 GB/s  depth max=9  first expert ready at 151 us  last issue at 989 us of 1272
[HTP-PROFILE] layer calls total :     3150.5 ms
[HTP-PROFILE] arm staging memcpy:       11.0 ms  (206.0 MB in+out, 19.6 GB/s) -- outside every host= above
[HTP-PROFILE] HTP host time     :     4241.6 ms (registration + layer calls + staging)
[HTP-DMA] call=1 M=512 rep=5/5 desc=360 waits=226 blocked=3 depth_max=9 dropped=0 end=11920.6us
[HTP-DMA] call=2 M=512 rep=5/5 desc=344 waits=222 blocked=3 depth_max=9 dropped=0 end=15616.7us
[HTP-DMA] call=3 M=512 rep=5/5 desc=374 waits=240 blocked=3 depth_max=9 dropped=0 end=15691.6us
[HTP-DMA] call=1 M=1 rep=5/5 desc=46 waits=30 blocked=3 depth_max=9 dropped=0 end=1235.2us
[HTP-DMA] call=2 M=1 rep=5/5 desc=46 waits=30 blocked=3 depth_max=9 dropped=0 end=1272.3us
[HTP-DMA] call=3 M=1 rep=5/5 desc=46 waits=30 blocked=3 depth_max=9 dropped=0 end=1284.6us
```

Line counts of the extraction loop: level 2 — profile 18, dma headers 6,
M=1 dump 231; level 3 — 18 / 6 / 231 (3 × M=1 and 3 × M>1 headers per
level, 46 `push` + 30 `wait` lines per M=1 block, as expected).

### Attribution table (plan 87 §1 / §3.2) — filled from the device lines named per row

Shares of `dsp_us` are of the M==1 row's `dsp=` (1271.6 µs warm / 1414.0
cold). The isolated-vs-in-situ gap is the gap this sitting measures:
in-situ `DMA ring:` engine **21.9–32.3 GB/s** (warm) against this unit's
isolated `DMA_PROBE` shape-`iii` **106.9 GB/s** (1 worker, cool run) —
a factor of **3.3–4.9×**, i.e. ~75–80 % of the achievable rate is lost.
The replay cells sit in between (see row h) and are the single most
informative reading of the sitting.

| # | hypothesis | signature that confirms (plan 87 §3.2) | lines of this sitting that fill it | % of `dsp_us` (L3 warm / L2 cold) | % of the isolated-vs-in-situ gap |
|---|---|---|---|---|---|
| a | ring serialised behind compute: waits dominate, depth ≤ 1 | `wait ≥ 25 %` of `dsp` **and** `depth max ≤ 2`; refuted if depth reaches 8–11 while waits stay small | `DMA ring: wait=129.2` of `dsp=1271.6` (L3), `269.0` of `1414.0` (L2); `depth max=9` / `11`; replay cells 4–6 `wait_us` 74.0 / 70.3 / 574.1 of 1215 µs | **10.2 % / 19.0 %** | ~0 % — **refuted**: depth reaches 9–11 with waits at 10 %, and all 30 waits but the first are `blocked=n` |
| b | chunk geometry is a slow probe shape | already refuted on the host; device lines merely confirm the shapes | `[HTP-DMA] push … kind=gate/up bytes=524288 row=8192 nrows=64 stride=57344`, `kind=down bytes=917504 row=16384 nrows=56 stride=32768`, `desc=46` — exactly probe `iii` (8 KiB × 64 @ 56 KiB) and the down shape | 0 % | ~0 % — **confirmed refuted**: the identical shape reaches 106.9 GB/s isolated on this very unit (probe `iii`, w=1) |
| c | one dmlinked chain / one `dmstart` limits concurrency | `gbs` rises ≥ 15 % with workers on the same schedule at `pace=0` | replay cells 1–3 at `pace=0`: 40.2 → 39.1 → 36.5 GB/s (1 → 2 → 4 workers) | 0 % | ~0 % — **refuted (provisional, Dev. 6)**: adding workers makes it *slower*, and `wait_us` explodes 551 → 1146 → 2417 µs |
| d | DDR contention with HVX epilogues / pack | paced replay's blocked-wait total moves ≥ 20 % with `load=1`; `load=2` moving it instead points at HVX activity | replay cell 7 (`load=1`, DDR stream): `wait_us` 1273.3 and `blocked=518/600` vs cell 4's 74.0 and 60/600 (+1620 %, +763 %), `us_per_call` 1404 vs 1215 (+15.6 %); cell 8 (`load=2`, VTCM/HVX): 75.2 µs, 59/600 — unchanged | ≤ 15 % (the ceiling the DDR-loaded cell shows, not a share measured in situ) | **large but bounded — confirmed as a sensitivity, not as the cause**: a competing DDR stream hurts badly, but the in-situ M==1 call has no such competitor (`load=2`, the HVX-busy case, changes nothing) |
| e | DVFS / bus ramp inside a call | per-descriptor rates climb monotonically through the call (expert 0 ≤ 60 % of expert 3) on cold calls, flat on warm | L2 `call=1`: expert 0's 3.5 MiB gate/up burst `done=32.7..155.1` = 122.4 µs (28.6 GB/s); expert 3's `done=984.4..1127.0` = 142.6 µs (24.5 GB/s). L3: expert 0 `26.9..119.3` = 92.4 µs (37.9 GB/s), expert 3 `863.0..1004.5` = 141.5 µs | 0 % | ~0 % — **refuted, with the sign reversed**: the *first* expert is the fastest in both levels; there is no intra-call ramp |
| f | cold start: expert 0's 3.5 MiB has nothing to hide behind | the act wait of expert 0 ≈ `gather` (≈ 120 µs) and every later wait ≈ 0 on warm calls | `wait k=1 site=act idx=9 t=32.7..155.1 blocked=y` (L2, 122.4 µs) and `t=26.9..119.3 blocked=y` (L3, 92.4 µs) — the only `blocked=y` wait of the call after `copy_in`; `act 123.7` of `wait=129.2` total (L3) = 96 %, `145.4` of `269.0` (L2) = 54 %; `gather 124.8` / `146.5` matches it to 1 µs; `first expert ready at 151` / `165 us` | **9.7 % / 10.3 %** | ~10 % — **confirmed**: it is the whole wait budget of a warm call, and it is exactly the un-hidden first-expert load |
| g | warm/cold per call: single-shot (L2, `drain 108`) blocks where 5× repeats (L3, `drain 1.3`) do not — IOTLB / page state / clock after the ≈ 0.6 ms FastRPC gap | `fresh=1` alone reproduces the L2 wait profile → translation/page state; `gap_us` alone → clock ramp | in situ L2 vs L3: `drain 113.1+8.9` vs `2.4+1.3`, `wait 269.0` vs `129.2`, `busy 904..1198` vs `682..1005` (−25 % warm). Replay cell 9 (`fresh=1`): 1215.1 µs vs cell 4's 1215.0 — no change; cell 10 (`gap_us=600`): 1784.8 µs = 1215 + 570, i.e. the idle gap itself and no ramp penalty; cell 11 (both): 1785.0 | **11.0 % / 7.1 %** (the L2−L3 `dsp` difference, 142.4 µs, as a share) | ~10 % — **real in situ, unexplained by both controls (provisional, Dev. 6)**: neither page-state (`fresh`) nor clock-gap (`gap_us`) reproduces it in the harness, so the ≈ 142 µs cold-call penalty has a third cause |
| **h** (new) | the real 46-descriptor chunk list is itself slow: isolated, with no compute and no competitor, it runs at a fraction of the probe rate for the same shapes | replay at `pace=0`, 1 worker, nothing else running, should reach the probe's rate if only interleaving were at fault | `DMA_REPLAY workers=1 load=0 pace=0`: **40.2 GB/s** (`us_per_call=561.5`, `bytes_per_call=22560768`, `regions=32`, `busy_us=440.6..557.9`) against `DMA_PROBE shape=iii workers=1`: **106.9 GB/s** cool / 88.8 GB/s hot, and against the in-situ 21.9–32.3 GB/s | — (not an in-situ share; it is the ceiling the in-situ path is measured against) | **≈ 60–70 % of the gap (provisional, Dev. 6)**: 2.2–2.7× is lost before any compute or contention enters, and only the remaining ~1.3–1.8× is in-situ interleaving (which rows f + g account for) |

The first wall-2 fix issue is: **row h — why the traced 46-descriptor /
32-region chunk list replays at 40 GB/s while `DMA_PROBE` shape `iii`
(the same 8 KiB × 64 @ 56 KiB geometry) does 107 GB/s on the same unit
and skel**; every in-situ row (a, b, c, e refuted; f + g ≈ 20 % of
`dsp_us` between them) is too small to close the 3.3–4.9× gap on its own,
so the descriptor list / region layout is where wall 2 actually sits —
with row d as the standing constraint that whatever replaces it must not
start competing for DDR. (Plan 87 §0 expected f + g large and a / b / c
small: f and g are confirmed but only ~20 % of `dsp_us` together, and the
unexpected finding is h. Dev. 6 must be resolved first, since h rests on
the replay harness whose content check fails.)

### Replay table — `MoeChunkReplay` (in-situ chunk list, 20 calls per cell, A skel)

**Test verdict: `[  FAILED  ]`** — see Deviation 6. All 11 cells produced
timings; all 11 report `checksum_ok=n` with `res[6]=1467840` vs
`want_sum=9461760`, and `bytes_per_call` reads 22560768 where the handoff
expects 22020096. The timing columns are recorded and used (marked
provisional); the `checksum_ok` column is `n` throughout and is the defect,
not a per-cell result.

`DMA_REPLAY_TRACE` line (the traced in-situ call the schedule was taken from):

```
DMA_REPLAY_TRACE dsp_us=1217 desc=46 waits=30 blocked=3 wait_us=73 wait_act_us=70 busy_us=200..603 depth_max=9 first_ready_us=99 last_issue_us=935 trace_words=710 plan_shape_ok=y
```

(`plan_shape_ok=y`, `desc=46`, `dsp_us=1217` — the traced call matches the
B level-3 M==1 row's 1271.6 µs to 4 %, so the schedule replayed is the real one.)

| # | workers | load | pace | fresh | gap_us | decides | us_per_call | gbs | wait_us | blocked | depth_max | busy_us lo..hi | regions | workers_used | load_units | checksum_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 0 | 0 | c | 561.5 | 40.2 | 551.4 | 520/600 | 11 | 440.6..557.9 | 32 | 1 | 0 | n |
| 2 | 2 | 0 | 0 | 0 | 0 | c | 577.4 | 39.1 | 1146.0 | 600/600 | 10 | 0.0..0.0 | 32 | 2 | 0 | n |
| 3 | 4 | 0 | 0 | 0 | 0 | c | 618.0 | 36.5 | 2416.6 | 600/600 | 5 | 0.0..0.0 | 32 | 4 | 0 | n |
| 4 | 1 | 0 | 1 | 0 | 0 | a | 1215.0 | 18.6 | 74.0 | 60/600 | 9 | 199.7..607.7 | 32 | 1 | 0 | n |
| 5 | 2 | 0 | 1 | 0 | 0 | a | 1214.1 | 18.6 | 70.3 | 60/600 | 5 | 0.0..0.0 | 32 | 2 | 0 | n |
| 6 | 4 | 0 | 1 | 0 | 0 | a | 1289.8 | 17.5 | 574.1 | 85/600 | 5 | 0.0..0.0 | 32 | 4 | 0 | n |
| 7 | 1 | 1 (DDR stream, 3 workers) | 1 | 0 | 0 | d | 1404.0 | 16.1 | 1273.3 | 518/600 | 11 | 1072.0..1348.0 | 32 | 1 | 165 | n |
| 8 | 1 | 2 (VTCM stream, HVX busy) | 1 | 0 | 0 | d | 1215.2 | 18.6 | 75.2 | 59/600 | 9 | 205.7..612.5 | 32 | 1 | 4096 | n |
| 9 | 1 | 0 | 1 | 1 | 0 | g | 1215.1 | 18.6 | 90.2 | 62/600 | 11 | 367.3..736.5 | 32 | 1 | 0 | n |
| 10 | 1 | 0 | 1 | 0 | 600 | g | 1784.8 | 12.6 | 74.3 | 60/600 | 9 | 199.4..607.9 | 32 | 1 | 0 | n |
| 11 | 1 | 0 | 1 | 1 | 600 | g | 1785.0 | 12.6 | 82.3 | 60/600 | 9 | 304.4..685.7 | 32 | 1 | 0 | n |

Readings (all provisional per Deviation 6):
* Paced (cells 4–6, the in-situ schedule) reproduces the in-situ rate
  almost exactly: 18.6 GB/s vs the in-situ `weight DMA:` average of
  15.6 / 17.3 GB/s — so the harness is faithful to the in-situ *pacing*.
* Unpaced (cell 1) is 2.16× faster (40.2 GB/s) — the pacing, i.e. the
  per-expert dependency structure, costs more than half the call.
* Still, 40.2 GB/s is 2.7× below the same unit's isolated `DMA_PROBE`
  shape-`iii` rate: row h.
* Extra workers never help (c refuted); a DDR competitor hurts a lot and
  an HVX/VTCM competitor not at all (d).

### `DMA_PROBE` — isolated arena rate of this unit (16 lines, `vote=1`)

Two readings this sitting: the **gate** run (`G_probe_gate.log`, the first
device command after the skel push, phone cool at 29 °C — this is #97's
acceptance) and the **block-5** run (`G_replay.log`, after the 12 A cells,
`thermal_zone0` ≈ 51 °C). The block-5 run is the sitting's run of record;
the cool reading is given because the gap analysis uses the best rate this
unit can reach.

| shape | workers | this unit GB/s (block 5, hot) | this unit GB/s (gate, cool) | #77 GB/s | us (block 5) | checksum_ok | workers_used | n_desc |
|---|---|---|---|---|---|---|---|---|
| (i) 4 KiB × 256 | 1 | 66.4 | 69.3 | 79.6 | 8087 | y | 1 | 256 |
| (i) | 2 | 68.1 | 75.7 | 75.9 | 7885 | y | 2 | 256 |
| (i) | 3 | 71.7 | 73.8 | 74.2 | 7485 | y | 3 | 256 |
| (i) | 4 | 70.8 | 72.2 | 72.0 | 7580 | y | 4 | 256 |
| (i1) 1 MiB × 1 | 1 | 66.3 | 73.6 | 78.4 | 8096 | y | 1 | 256 |
| (i1) | 2 | 68.0 | 75.8 | 76.0 | 7898 | y | 2 | 256 |
| (i1) | 3 | 71.9 | 73.9 | 74.1 | 7463 | y | 3 | 256 |
| (i1) | 4 | 71.0 | 72.2 | 72.4 | 7565 | y | 4 | 256 |
| (ii) 16 KiB × 64 @ 56 KiB | 1 | 91.4 | 110.2 | 116.5 | 7538 | y | 1 | 219 |
| (ii) | 2 | 98.8 | 113.3 | 117.0 | 6970 | y | 2 | 219 |
| (ii) | 3 | 108.8 | 111.4 | 112.0 | 6331 | y | 3 | 219 |
| (ii) | 4 | 107.6 | 109.0 | 109.7 | 6403 | y | 4 | 219 |
| (iii) 8 KiB × 64 @ 56 KiB (= gate/up chunk shape) | 1 | 88.8 | 106.9 | 111.0 | 9049 | y | 1 | 511 |
| (iii) | 2 | 95.1 | 112.1 | 113.2 | 8449 | y | 2 | 511 |
| (iii) | 3 | 104.8 | 110.1 | 110.5 | 7666 | y | 3 | 511 |
| (iii) | 4 | 105.3 | 108.1 | 108.6 | 7635 | y | 4 | 511 |

The cool reading reproduces #77 on the same unit to within 4 % on every
line (the one-worker `i`/`i1` rows 8–13 % low, thermals); the hot reading
is 3–19 % below the cool one, which is the sitting's own thermal spread
and is why both are recorded.

### C — `NNTR_MOE_HTP_M1_GEMV=1` vs A (NPU, same sitting, same binary)

Verdict thresholds (handoff skill): decode goal-progress / regression /
no-change at ±5 % of this sitting's A; prefill below −5 % of A is a
regression regardless of decode; text must equal A at every G.

| variant | G | run | prefill ms / tok/s | A prefill tok/s (r1 / r2) | Δ prefill % | decode tok/s (all) | A decode tok/s (r1 / r2) | Δ decode % | gen tokens printed | peak RSS KB | text = A? (first differing token) | `moe m1 gemv:` line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 64 | 1 | 1202 ms / 425.957 | 389.058 / 496.124 | −3.8 (vs mean 442.6) | 19.1904 | 17.5198 / 17.9272 | **+8.3** (vs mean 17.72) | 64 | 5410144 | same | `on (applied=0x1)` |
| C | 64 | 2 | 1190 ms / 430.252 | 389.058 / 496.124 | −2.8 | 18.4598 | 17.5198 / 17.9272 | **+4.2** | 64 | 4776808 | same | `on (applied=0x1)` |
| C | 512 | 1 | 1181 ms / 433.531 | 469.294 / 506.429 | −11.1 (vs mean 487.9) | 18.7354 | 17.09 / 16.9609 | **+10.0** (vs mean 17.03) | 512 | 4976844 | same | `on (applied=0x1)` |
| C | 512 | 2 | 1182 ms / 433.164 | 469.294 / 506.429 | −11.2 | 17.9328 | 17.09 / 16.9609 | **+5.3** | 512 | 5007200 | same | `on (applied=0x1)` |
| C | 1024 | 1 | 967 ms / 529.473 | 527.291 / 433.164 | +10.2 (vs mean 480.2) | 17.9118 | 16.831 / 17.7775 | +3.5 (vs mean 17.30) | 1024 | 5198532 | same | `on (applied=0x1)` |
| C | 1024 | 2 | 1015 ms / 504.433 | 527.291 / 433.164 | +5.0 | 17.2713 | 16.831 / 17.7775 | −0.2 | 1024 | 5262540 | same | `on (applied=0x1)` |

**Verdict: goal progress at G 64 and 512, no change at G 1024.** Decode
means: G=64 18.83 vs 17.72 (**+6.2 %**), G=512 18.33 vs 17.03
(**+7.7 %**), G=1024 17.59 vs 17.30 (+1.7 %, inside ±5 % = no change).
Accuracy gate **passes**: every C text is byte-identical to the A text at
the same G and run (`diff` over the generation, `moe m1 gemv` line
excluded), and the device gtest is bit-exact (below).

The prefill Δ column is **not** a regression despite two rows below
−5 %: prefill is the M>1 path, which the switch does not touch, and the
profile run proves it did not move (`m1_gemv=0/23`, `dsp=16554.3` µs/call
in C vs `16575.9` in A — 0.13 % apart). This sitting's prefill wall-clock
spread within A itself is ±13 % (389–527 tok/s across six NPU A cells), so
the −11 % rows are session noise, not the switch. Recorded rather than
adjusted.

Proof run (`C_level2.log`, level 2, G=64, not for tok/s) beside A's
`B_level2.log` M==1 row — plan 80: `22 MB / mm` is the effective arena read
rate the GEMV sees:

| run | qos_mode | calls | host µs/call | dsp µs/call | transport µs/call | gather | mm | blocks | m1_gemv | weight DMA avg GB/s | `DMA ring:` wait us / depth max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (`B_level2.log`) | 2 | 1408 | 2062.5 | 1414.0 | 648.6 | 146.5 | 783.4 | 5632 | 0/1408 | 15.6 | 269.0 / 11 |
| C (`C_level2.log`) | 2 | 1408 | 1792.2 | 1044.0 | 748.2 | 0.0 | 974.7 | 0 | 1408/1408 | (no `weight DMA:` line) | 4.3 / 1 |

Reading: the GEMV path cuts `dsp` per M==1 call by **26 %** (1414 → 1044
µs) and removes the DMA ring from the call entirely (`desc=2/call` vs 46,
`wait=4.3` vs 269.0 µs, `gather` 146.5 → 0) while `mm` grows 783 → 975 µs
— i.e. it trades the staged 21.5 MB/call weight DMA for a direct arena
read inside `mm`. Two oddities to note, both cosmetic accounting in the
level-2 print and neither affecting the tok/s cells: the C M==1 row
attributes 5601.2 µs to `swiglu` and consequently prints
`rest<=-5588.2 (-311.8% of host)`, and no `weight DMA:` line is emitted
for the C M==1 row (there is no staged DMA to report). Worth a line in
the #93/#80 print, not a blocker.

Gtest `MoeLayerM1GemvMatchesHmx` (`G_m1gemv.log`): `bad_elems_M1` **0 of
2048**, `bad_elems_M2` **not printed by this build** (only M1 and M4 are),
`bad_elems_M4` **0 of 8192**, `bit_identical` **yes**, `[  PASSED  ]`
**1 test**. Pasted lines:

```
U8I4_FIELD path=moe_m1_gemv field=bad_elems_M1 value=0 of 2048
U8I4_FIELD path=moe_m1_gemv field=bad_elems_M4 value=0 of 8192
U8I4_FIELD path=moe_m1_gemv field=bit_identical value=yes
[       OK ] HmxMmU8I4Layer.MoeLayerM1GemvMatchesHmx (817 ms)
[  PASSED  ] 1 test.
```

C text comparison (block 6 loop):

```
C vs A G=64 r1: same
C vs A G=64 r2: same
C vs A G=512 r1: same
C vs A G=512 r2: same
C vs A G=1024 r1: same
C vs A G=1024 r2: same
```

### Ride-along (optional, not a variant of record)

Skipped — the sitting had already spent its budget on the full A / B /
gtest / C set after the #97 re-push, and block 6b is explicitly "skip
without note if the clock says so".

## Notes from the run

- Battery/thermal: 100 % / 28.0 °C / 29.0 °C at start (USB powered, cool);
  100 % / 36.5 °C / 51.2 °C after block 2; 100 % / 37.9 °C / 55.0 °C after
  block 6 (C). No thermal gate tripped.
- Config-edit echo (block 1): `do_sample": false` for both models; `q40` —
  `bad_word_ids": [` `124900` `]` (pretty-printed across lines, already
  present), `init_seq_len": 512`; `q40-qs4cx-wh` — same plus
  `moe_engine": "htp"` and `moe_htp_layers": ""` (already present, the
  guard `sed` did not fire). Because the array is pretty-printed, the
  handoff's `'"bad_word_ids": \[\]'` `sed` matches nothing on this device —
  the value was verified by `grep -A3` instead, which is the check that
  matters.
- Device md5sum lines (block 1 provenance): `libnntr_hvx_skel.so`
  **`d6568c8bb95e755485e982811e8db755`** in both `$D` and `$T` (this
  workstation's #98 build, Deviation 5 — not the table's `ce7bb512…`),
  `nntrainer_causallm 055703b23f40401723f4d751c5b48c6a`,
  `libnntrainer.so a508e889645cfb83ff6f59d0171e8526`,
  `libcausallm_core.so 8057cea3aed93d3163e91a6490d1ac02`,
  `libccapi-nntrainer.so b057acb6c4de3b3097ddae679cccde7f`,
  `libc++_shared.so b1586b9b512712800fd36a24abac1c0a` (== table),
  `libsdkl.so 0ad4e22a70e4f135bce38ad8fd1e001b` (== table),
  `unittest_hvx_dma_probe 76c43d3ee3884e66194283bfacd68bb5`,
  `unittest_hvx_mm_u8i4 e79adbd0ace50c8628d924e62c83b34e`,
  `prompt512.txt fc65c1588dc66dd764c7013fe96cbb75` (== table), both
  `tokenizer.json 7b8067a580173d3eb1697afae3b456f5` (== table). Only the
  skel was re-pushed this sitting; every other file was already on the
  device from the first attempt with the md5 above.
- Model `.bin` on device: both present at the exact documented sizes
  (4768855808 / 4316133120 B) from the first attempt — no 9 GB push.
- Skel build line for the record:
  `HEXKL_ROOT=~/Downloads/hexkl_addon HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`
  at `08afbb10` → `UNDEFINED SYMBOLS OK (46 runtime imports)`,
  `built: … (v79, hexkl 6.4.0.1)`, 172,624 B,
  `d6568c8bb95e755485e982811e8db755`. (`HEXKL_SDK_VER` matters: left
  unset, `build.sh` picks the newest directory under `$HEXKL_ROOT/lib`,
  which is 6.6.0.0 here and would silently be a different HexKL than the
  handoff's.)
- No FARF/AEE errors on either path this sitting; no `0x80000406`, no
  `0x8000040e`, no "Context is not registered", no
  `moe_dma_trace_read err=`, no `nntr_hvx_moe_set_opts failed`. The only
  failure of the sitting is the `MoeChunkReplay` content check
  (Deviation 6).
- Wall clock: ≈ 55 min device time (13:40–14:35 KST), against the ≈ 56 min
  estimate — the 9 GB push was not needed, and the skel rebuild (6 s) plus
  the re-push (172 KB) replaced it.

## Appendix

### Generated texts per (model, G)

Full logs in `$W/logs/A_<model>_G<G>_r<n>.log`,
`C_q40-qs4cx-wh_G<G>_r<n>.log` on the bridge-reachable workstation
(`/local/mnt/workspace/htp_moe/94/logs/`). Every cell: run1 = run2, and
every C text = the A text at the same G and run. The CPU (`q40`) and NPU
(`q40-qs4cx-wh`) texts differ from each other by construction (expert
quantisation), and both are coherent continuations of the prompt with no
repetition or garbage.

#### q40, G=64 / 512 / 1024
run1 = run2 at every G; the G=512 and G=1024 texts extend the G=64 text
(greedy decoding, same prompt).
#### q40-qs4cx-wh, G=64 (runs 1, 2, 3)
r1 = r2 = r3 (the block-4 re-run after the two profile runs is identical).
#### q40-qs4cx-wh, G=512 / 1024
run1 = run2 at both G.
#### C q40-qs4cx-wh, G=64 / 512 / 1024
= A at every G and run (all six `diff`s "same").

### The six `[HTP-DMA] M=1` blocks

`$W/logs/B_level2_dma_m1.txt` and `B_level3_dma_m1.txt`, 231 lines each
(3 blocks × [1 header + 46 `push` + 30 `wait`] + the M=1 headers). The
per-expert `done=` intervals quoted in attribution rows e and f are from
these files.

### `DMA_REPLAY_TRACE` push / wait lines

`$W/logs/G_replay.log`, 46 `push` + the `wait` lines under the
`DMA_REPLAY_TRACE` header quoted above, followed by the 11 `DMA_REPLAY`
lines (each with its `unittest_hvx_dma_probe.cpp:631` failure block,
Deviation 6) and the 16 `DMA_PROBE` lines.

### `logs/` file list

```
A_q40_G1024_r1.log              B_level2_dma_headers.txt
A_q40_G1024_r2.log              B_level2_dma_m1.txt
A_q40_G512_r1.log               B_level2.log
A_q40_G512_r2.log               B_level2_m0.txt
A_q40_G64_r1.log                B_level2_profile.txt
A_q40_G64_r2.log                B_level3_dma_headers.txt
A_q40-qs4cx-wh_G1024_r1.log     B_level3_dma_m1.txt
A_q40-qs4cx-wh_G1024_r2.log     B_level3.log
A_q40-qs4cx-wh_G512_r1.log      B_level3_m0.txt
A_q40-qs4cx-wh_G512_r2.log      B_level3_profile.txt
A_q40-qs4cx-wh_G64_r1.log       C_level2.log
A_q40-qs4cx-wh_G64_r2.log       C_q40-qs4cx-wh_G1024_r1.log
A_q40-qs4cx-wh_G64_r3.log       C_q40-qs4cx-wh_G1024_r2.log
G_m1gemv.log                    C_q40-qs4cx-wh_G512_r1.log
G_probe_gate.log                C_q40-qs4cx-wh_G512_r2.log
G_replay.log                    C_q40-qs4cx-wh_G64_r1.log
                                C_q40-qs4cx-wh_G64_r2.log
```

### First attempt (pre-#97), kept for the record

The first attempt of this sitting (2026-09-22 ~11:00 KST, same unit, same
bridge, commit `ec7ad296`) measured only the six CPU `q40` cells and was
blocked on the NPU path by the skel loader bug now closed as #97/#98:
decode 53.024 / 53.467 · 51.0825 / 51.154 · 49.675 / 48.4046 tok/s at G
64 / 512 / 1024, prefill 287.156 / 274.237 · 278.715 / 280.241 · 274.678 /
202.692 tok/s, run1 = run2 at every G. Its rows were replaced above by
this sitting's own control run (rule 9: every sitting runs its own
control); the two CPU readings agree within 3 % at every G except the
G=512 r2 pair (47.33 vs 51.15, −7.5 %, the warmest CPU cell of the second
attempt).
