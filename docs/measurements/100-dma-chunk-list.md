# Measurement 100: row h's cause and a DDR→VTCM feed shape for the M=1 GEMV, with the #101 ride-along (A = GEMV default vs A0)

Branch `htp/100-dma-chunk-list` (PR into `htp_moe`; code head
`a8642bd1`, this file on the same branch). Estimated device time:
**≈ 32 min**, plus 5–8 if the model has to be pushed.

Plan: `docs/plans/100-dma-chunk-list.md` §4 step 5. Issue #100 (tracker
#76; LEDGER ⑥ / ㉒, rules 11 and 19). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. Ride-along cells from
`docs/measurements/101-ride-along.md` (PR #108, merged as `8e121dbe`).

## Why

The traced M=1 chunk list replays at 40 GB/s. DMA probe shape iii, with
the same 8 KiB × 64 @ 56 KiB geometry, reads 107, which is above the
DRAM peak. #100 must name the cause from 16 replay cells of one run, and
it must certify a descriptor shape that feeds the GEMV from VTCM at
≥ 37 GB/s while HVX reads VTCM. The denominator is the same-run `c_star`.
**What hangs on it:** if a shape passes, the ㉒ feed issue is filed with
that shape. It rebases onto PR #107, and its gate is M==1 `mm` ≤ 600 µs.
If no shape passes, ㉒'s feed half closes and #107 lands alone after a
lead sweep. The same sitting also carries PR #108's ride-along (A0 and
the fixed M==1 level-2 row).

## Variants (one app set, one skel, one gtest)

* **A**: `htp_moe` app with the GEMV default, env unset. Every A log must
  print `[HTP] moe m1 gemv: on (applied=0x1) source=default` exactly
  once. An A log without that line voids the sitting (#108's inverted
  rule). Run first.
* **A0**: A's binaries with `NNTR_MOE_HTP_M1_GEMV=0` (the HMX path), G = 64
  only. This is #101's cell R3 and R4.
* #100 adds **no variant**. Its cells are a gtest (`unittest_hvx_dma_probe`)
  run once under A's skel. No model binary changes: `git diff --stat
  db7c6eb2 a8642bd1 -- nntrainer/ Applications/ test/htp/nntr_hvx.idl` is
  empty. The skel carries the new replay entry, but no model call reaches
  that code.
* **#99's replay check is not in this sitting.** Its fix is not merged, so
  the 11 old `DMA_REPLAY` lines still fail their `res[6]` check, as they
  did in #94 (see "Reading the gtest").

Accuracy: the text of A0 must equal A at G = 64. Run 1 must equal run 2
for A at every G. `NNTR_L2_DIFF` and text-vs-CPU `q40` are n/a for
`QS4CX_WH`.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/100/`**. `$W/md5.txt`
reproduces this table. `(cd $W && md5sum -c md5.txt)` must pass on every
line. Rule 22: the staging path is a convenience, and the recipe below
rebuilds the set from the commit.

| file (staged as) | md5 | built from / with |
|---|---|---|
| `$W/libnntr_hvx_skel.so` (→ device `libnntr_hvx_skel.so`; A, A0, gtest) | `25f8518986a5821a91218fa715092fde` | `test/htp/build.sh` @ `a8642bd1`, `HEXKL_SDK_VER=6.4.0.1`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)`. Skel builds are not byte-reproducible (LEDGER rule 14), so a rebuild gives another md5; record the one that ran |
| `$W/A/nntrainer_causallm` | `9ed4e439f30dba04dfa9638e02cbc788` | `build_android.sh --htp`, then the prefix fix, then `--htp --cache`, on this branch (app sources = `htp_moe` @ `db7c6eb2` = `8e121dbe`'s code) `jni/libs/arm64-v8a/` |
| `$W/A/libcausallm_core.so` | `7c84f335615ca48c7dbf574fe9fafe75` | same, `jni/libs/arm64-v8a/` |
| `$W/A/libnntrainer.so` | `dcd57fc1b598ec420725ee1a5c6429d6` | same, `jni/obj/local/arm64-v8a/`; `readelf -d` lists `libsdkl.so` and `libcdsprpc.so`; `strings … \| grep -c 'source=%s'` = 2 |
| `$W/A/libccapi-nntrainer.so` | `c57dd47f6f835b9499dc5eb07dafb3cc` | same, `jni/obj/local/arm64-v8a/` |
| `$W/gtest/unittest_hvx_dma_probe` | `0f6d53dfa418a83b3591608c213c722e` | `ndk-build … unittest_hvx_dma_probe` @ `a8642bd1` (`test/jni/obj/local/arm64-v8a/`) |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot `toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/` (== #77/#88/#101) |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77/#88/#101) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | == #77/#88/#101 (`docs/measurements/77-prompt512.txt`) |
| model `q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model, the only model in this sitting) |
| `q40-qs4cx-wh/tokenizer.json` | `7b8067a580173d3eb1697afae3b456f5` | HF |

The app's sources are the same as #101's staged A (`git diff 8018c88a
db7c6eb2 -- nntrainer Applications` is empty). The md5s differ because
this is another build path (rule 14). Never staged or pushed:
`builddir/…/libcdsprpc.so` (rule 5).

### Rebuild recipe (rule 22; any workstation)

```
git fetch origin && git checkout a8642bd1              # or the branch head
git submodule update --init --depth 1                  # a fresh worktree has empty subprojects/ (LEDGER §3a)
cp <another checkout>/Applications/CausalLM/lib/libtokenizers_android_c.a Applications/CausalLM/lib/   # or build_tokenizer_android.sh
source tools/htp/env.sh
export HEXKL_ROOT=<your HexKL package root, holds lib/6.4.0.1/>   HEXKL_SDK_VER=6.4.0.1   ANDROID_NDK=<NDK r30>
./test/htp/build.sh                                    # must print UNDEFINED SYMBOLS OK (46 runtime imports)
(cd Applications/CausalLM && ./build_android.sh --htp) # fresh builddir: ends in a /usr/local permission error (googletest install); then
(cd builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install)
(cd Applications/CausalLM && ./build_android.sh --htp --cache)
ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT unittest_hvx_dma_probe -j8)
cp $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so <stage>/   # b1586b9b…
cp $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/libsdkl.so <stage>/                                                           # 0ad4e22a…
```

Take the four A files from `Applications/CausalLM/jni/libs/arm64-v8a/`
(`nntrainer_causallm`, `libcausallm_core.so`) and `jni/obj/local/arm64-v8a/`
(`libnntrainer.so`, `libccapi-nntrainer.so`). Env var names in this
sitting: `NNTR_MOE_HTP_M1_GEMV` (unset = on, `0` = off),
`NNTR_HTP_PROFILE`, `NNTR_NUM_THREADS=8`.

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/100; M=/local/mnt/workspace/models/lfm2.5-8b-a1b
(cd $W && md5sum -c md5.txt)                                          # every line OK
strings $W/A/nntrainer_causallm | grep -c 'per-layer-type totals'      # 0 (not a profile binary)
strings $W/A/libnntrainer.so | grep -c 'source=%s'                     # 2 (the GEMV default is in)
find $W -name 'libcdsprpc*' | wc -l                                   # 0
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd <your checkout> && git fetch && git checkout htp/100-dma-chunk-list && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/100; M=/local/mnt/workspace/models/lfm2.5-8b-a1b; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
```

### 0. Device state (2 min)

```
adb devices                     # exactly one device; write its serial under Notes (any S25 Ultra, contract §4.2)
therm                           # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
adb shell df -h /data
```

Screen off, charger in.

### 1. Install (5 min; + 5–8 if the model must be pushed)

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin   # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin  # 7b7867fab51845664c0050c0a837073e, else:
#   adb shell mkdir -p $D/models && adb push $M/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models
adb push $W/libnntr_hvx_skel.so $W/libc++_shared.so $W/libsdkl.so $W/prompt512.txt $D/
adb push $W/A/nntrainer_causallm $W/A/libcausallm_core.so $W/A/libnntrainer.so $W/A/libccapi-nntrainer.so $D/
adb push $W/gtest/unittest_hvx_dma_probe $D/
adb shell "chmod 755 $D/nntrainer_causallm $D/unittest_hvx_dma_probe"
```

Config edits, applied every time (greedy decoding, exact token count, and
the WH dir must carry `moe_engine: htp`):

```
adb shell "cd $D/models/q40-qs4cx-wh && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|moe_engine|moe_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`.

Provenance (paste under Notes; must equal the table):

```
adb shell "md5sum $D/libnntr_hvx_skel.so $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so $D/unittest_hvx_dma_probe $D/prompt512.txt $D/models/q40-qs4cx-wh/tokenizer.json"
```

### 2. A: 6 cells (≈ 14 min), env unset

```
run() { # run <variant> <G> <run#> [extra env]
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && \
    $4 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv'
}
therm                                                          # checkpoint 1
run A 64 1;  run A 64 2
run A 512 1; run A 512 2
sleep 60; run A 1024 1; sleep 60; run A 1024 2
therm                                                          # checkpoint 2
grep -c 'moe m1 gemv: on (applied=0x1) source=default' $W/logs/A_G*_r*.log      # R1: 1 per file
```

Expected in every A log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `total: … ms`, `peak memory: … KB`, exactly one
`[HTP] moe m1 gemv: on (applied=0x1) source=default`, and **no**
`[HTP-PROFILE]` block. A count of `0` for R1 voids that A as the GEMV
reference. `prefill: N` with N ≠ 512 voids the prefill column, and
`generation: N` with N ≠ G is a failed config edit.

### 3. #100 gtest ride-along (≈ 2 min), A's skel, right after checkpoint 2

```
adb shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && cat /sys/class/thermal/thermal_zone0/temp && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*:*MoeChunkReplay*'" 2>&1 \
  | tee $W/logs/G_replay.log | grep -E 'md5|^[0-9]+$|^DMA_REPLAY_NOTE|^DMA_REPLAY_TRACE dsp_us|^DMA_REPLAY |^DMA_REPLAY_X|^DMA_PROBE|OK \]|PASSED|FAILED'
therm                                                          # checkpoint 3
```

Read-out (paste the output under Results):

```
L=$W/logs/G_replay.log
grep -c '^DMA_PROBE .*checksum_ok=y' $L                          # 16
grep -c '^DMA_REPLAY_X .*checksum_ok=y' $L                       # 16
grep -c '^DMA_REPLAY workers.*checksum_ok=n' $L                  # 11 (#99, expected; see below)
grep -E 'skipped|stale_skel' $L                                  # nothing
grep -E '^DMA_PROBE shape=iii workers=1 |^DMA_REPLAY_X' $L | sed -E 's/ (bytes_per_call|regions|load_units)=[^ ]*//g'
awk '/^DMA_PROBE shape=iii workers=1 /{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="gbs")p=a[2]}}
     /^DMA_REPLAY_X/{n="";g="";for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="name")n=a[2];if(a[1]=="gbs")g=a[2]};r[n]=g}
     END{c=r["c_star"];
       printf "H0: c_star/probe_iii = %.1f/%.1f = %.2f (fires < 0.80)\n",c,p,c/p;
       printf "H1: iii_dmstart/probe_iii = %.2f (>= 0.80), iii_chain/iii_dmstart = %.2f (<= 0.60), iii_link1/iii_dmstart = %.2f\n",r["iii_dmstart"]/p,r["iii_chain"]/r["iii_dmstart"],r["iii_link1"]/r["iii_dmstart"];
       printf "list: iii_chain/iii_dmstart >= 0.80 and traced/iii_chain = %.2f (<= 0.60); traced/c_star = %.2f\n",r["traced"]/r["iii_chain"],r["traced"]/c;
       printf "  H3 traced_nowait/traced = %.2f (>= 1.25)  H2 traced_gu/traced_dn = %.2f (>= 1.5 either way)  H4 traced_f/traced = %.2f (15%%)  H5 iii_strided/iii_chain = %.2f (15%%)\n",r["traced_nowait"]/r["traced"],r["traced_gu"]/r["traced_dn"],r["traced_f"]/r["traced"],r["iii_strided"]/r["iii_chain"];
       split("f1 f2 f3",f," ");for(k=1;k<=3;k++)printf "feed %s: load0/c_star = %.2f (>= 0.80), load2 = %.1f GB/s (>= 37.0)\n",f[k],r[f[k]]/c,r[f[k]"_load"]}' $L
```

**Reading the gtest.**

* `DmaProbeShapes` must print `[       OK ]` with 16 `DMA_PROBE` lines,
  all `checksum_ok=y`. Its check is the same as before: the fill keeps
  0xA5 at every 64-byte sample.
* `MoeChunkReplay` is **expected to print `[  FAILED  ]`**. The failures
  must all come from `unittest_hvx_dma_probe.cpp:633` (`res[6]` 1467840
  vs `want_sum` 9461760), one for each of the 11 old `DMA_REPLAY` lines.
  That is #99, unchanged by this PR. A failure at `:715` (an X line's
  `res[12]`) or at `:692` (`skipped err=`) is a #100 result. Read it as
  follows:
  * `skipped err=`, or `stale_skel_or_nothing_landed` on every X line,
    means the skel on the device is not this branch's. Re-push
    `$W/libnntr_hvx_skel.so`, check the md5 line, and redo step 3.
  * `checksum_ok=n` on some X lines only means one of two things.
    * If the lines are among `traced`, `traced_f`, `traced_nowait`,
      `traced_gu` and `iii_chain`: these are the only cells whose writes
      overlap while in flight (the packed gate/up chunks of #99, or a slot
      reused 8 pushes on). Record which ones, because the result says the
      engine lets overlapping writes land out of list order. Their timing
      is still read. It is not a dropped transfer.
    * If `c_star`, `iii_dmstart`, `iii_link1`, `iii_strided` or any `f`
      line fails: those cells wait or drain before they rewrite a
      destination. File the line, and do not read its rate.
* In the same log, the step-0 line `DMA_REPLAY_TRACE dsp_us=… desc=46 …
  plan_shape_ok=y` must be present. It runs the HMX path, because the
  gtest does not call `moe_set_opts`.

### 4. A0: opt-out TPS, G = 64, one run (1.5 min)

```
run A0 64 1 NNTR_MOE_HTP_M1_GEMV=0
```

Expected: `[HTP] moe m1 gemv: off (applied=0x0) source=env`. The text
must equal `A_G64_r1`
(`diff <(sed -n '/^=====/q;p' $W/logs/A0_G64_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_G64_r1.log)`).
Decode is expected ≈ 6 % **below** A. A 0 % delta means the default did
not take, and that voids A as the GEMV reference. Prefill: A must be
within −5 % of A0.

### 5. Profiles (≈ 4 min; never read for tok/s), G = 64, one run each

```
adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json"
prof() { # prof <name> <env...>
  adb shell "cd $D && $2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/prof_$1.log | grep -E '^\[HTP-PROFILE\]' | grep -E 'level=|K=2048 N=2048|staging:|weight DMA' | cut -c1-400
}
prof A_L2  "NNTR_HTP_PROFILE=2"                                   # R2
prof A0_L2 "NNTR_HTP_PROFILE=2 NNTR_MOE_HTP_M1_GEMV=0"            # R4
```

Expected, from #101's R2 and R4:

* **A_L2** header `qos_mode=2`.
  * M==1 row: `blocks=0 m1_gemv=1408/1408`, `rest` ≥ 0 and ≤ 5 % of
    `host`, `swiglu / mm` ≈ 5.7.
  * Next line: `weight DMA: n/a (direct arena read inside mm, no ring;
    swiglu = lane-time, N.NN lanes busy over mm)` with N.NN ≈ 5.7, and a
    `staging:` line.
  * M>1 row: `m1_gemv=0/23`.
* **A0_L2** M==1 row: `blocks=5632 m1_gemv=0/1408`, followed by a normal
  `weight DMA:` line.
* **Transport question (supervisor).** Report M==1 `transport=` as
  `transport GEMV / HMX = x / y µs/call (Δ)` from A_L2 minus A0_L2, with
  both `staging:` lines. Expect equal `act`/`out` classes (65536 B).

### 6. A re-run (2 min): thermal check, then the text checks

```
run A 64 3
for g in 64 512 1024; do printf 'A G=%s r1=r2: ' $g; diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r2.log) >/dev/null && echo same || echo DIFFERENT; done
printf 'A0 vs A G=64: '; diff <(sed -n '/^=====/q;p' $W/logs/A0_G64_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_G64_r1.log) >/dev/null && echo same || echo DIFFERENT
printf 'A r3 vs r1 G=64: '; diff <(sed -n '/^=====/q;p' $W/logs/A_G64_r3.log) <(sed -n '/^=====/q;p' $W/logs/A_G64_r1.log) >/dev/null && echo same || echo DIFFERENT
grep -l 'HTP-PROFILE' $W/logs/A*_G*_r*.log                             # nothing
therm                                                                   # checkpoint 4
```

Then fill the tables below, commit this file on the branch, push, and
set #100 to `state:measured`.

## Results

Run on **2026-09-23**, unit **R3CY205ZMND** (SM-S938N, SM8750, v79),
over the ADF SSH bridge. This is the **same unit** #94 sitting 2 and
#105 ran on (#94's handoff asked for `R3CY10WM83Y` and got
`R3CY205ZMND`, its Deviation 2), so the cross-sitting DMA comparison
below is same-unit. The artifacts were rebuilt from the commit with the
§"Rebuild recipe" (rule 22) because `$W` did not exist on this
workstation; the md5 table is therefore replaced by the one under Notes.

### Verdict

* **Row h's cause: H0, and row h dissolves.** `c_star` = 26.3 GB/s is
  **0.36 ×** the same log's `DMA_PROBE shape=iii workers=1` (73.2), far
  under the 0.80 trigger, and `c_star` is `checksum_ok=y`. Exactly one
  row of plan §3.3 fires: H1 misses (`iii_dmstart`/probe iii = 0.41,
  needs ≥ 0.80) and the list row misses (`traced`/`iii_chain` = 0.99,
  needs ≤ 0.60). Against the validated ceiling the traced list is
  `traced`/`c_star` = **1.19**, i.e. the list is *faster* than the
  ceiling, so there is no list-side loss to fix. The "2.7×" of rule 11
  and the 4–7× isolated-vs-in-situ gap were measured against a number
  (probe iii) that no tag-validated per-call path reproduces.
* **Feed shape: FAIL as measured — but the absolute half of the gate is
  confounded, so do not close ㉒ on this sitting alone.** The best `f`
  cell is `f2` (8 descriptors, `dst=strided`, depth 2): `load=0` /
  `c_star` = **1.21** (≥ 0.80, **pass**) but `load=2` = **31.6 GB/s**,
  under the 37.0 gate (**fail**). `f1` and `f3` read 31.4 GB/s at
  `load=2`. Both halves are required, so the literal verdict is FAIL and
  the bound as measured is 31.6 GB/s. **The confound:** this whole
  sitting's DMA runs ≈ 22 % slower than #94 sitting 2 on the *same
  unit*. The evidence is the cell that this PR does not touch —
  `DMA_REPLAY workers=1 load=0 pace=0`, the 11 old lines' first cell —
  which read `us_per_call=561.5`, **40.2 GB/s** in #94 and reads
  `us_per_call=719.6`, **31.4 GB/s** here (see "Session drift" below).
  Scaled by that same factor, `f2`'s `load=2` would read ≈ 40 GB/s,
  i.e. **over** the gate. The ratio half of the gate (`load=0 / c_star`,
  within-run) is unaffected and passes. **Recommendation:** one repeat
  of the gtest cell alone on a sitting whose `DMA_REPLAY workers=1
  load=0 pace=0` reproduces #94's 40.2 GB/s, before ㉒'s feed half is
  closed or #107 is committed to as the sole path. If that cell cannot
  be brought back to 40.2, the FAIL stands and 31.6 GB/s is the bound.
* Note *why* it fails: HVX streaming VTCM is not the cause. `load=2`
  costs 0.1–0.3 % against `load=0` on all three shapes (699.8 → 700.5,
  695.5 → 697.6, 700.5 → 701.8 µs). The DMA engine itself tops out near
  31 GB/s on every validated per-call cell in this run, whatever the
  descriptor count (8, 24, 28, 36, 46), the chaining mode or `dst`.
* **Confirmed cooled** (`G_replay_cool.log`, `thermal_zone0` 31 800 m°C
  vs 46 500 for the protocol run): every ratio reproduces within 1 %
  (H0 0.36, `traced`/`c_star` 1.18, `f2` `load=2` 31.3 GB/s). So nothing
  here is a *within-sitting* thermal artefact. This repeat is an
  addition to the handoff.

**Session drift against #94, same unit.** Every DMA reading in this
sitting is well below #94 sitting 2's on `R3CY205ZMND`, and the cooled
repeat at 31 800 m°C does not recover it, so it is not this sitting's
thermals:

| cell (identical code in both sittings) | #94 s2 | here (protocol) | here (cooled) | ratio |
|---|---|---|---|---|
| `DMA_REPLAY workers=1 load=0 pace=0` (us_per_call) | 561.5 | 719.6 | — | 1.28 × slower |
| the same, GB/s | 40.2 | 31.4 | — | 0.78 |
| `DMA_PROBE shape=iii w=1` | 88.8 hot / 106.9 cool | 73.2 | 72.5 | 0.68–0.82 |
| `DMA_PROBE shape=i w=1` | 66.4 hot / 69.3 cool | 59.2 | — | 0.85–0.89 |
| `DMA_PROBE shape=iii w=4` | 105.3 hot / 108.1 cool | 101.4 | — | 0.94–0.96 |

The one-worker rows are 15–30 % low while the four-worker rows are
within 6 %, so whatever it is bites single-queue issue rate, not
aggregate DDR bandwidth. Unexplained; the candidates this sitting cannot
separate are a DSP/DDR clock or governor difference between sessions and
a skel difference between #94's build and `a8642bd1` (this PR is
test-only for the *model* path, but the skel was rebuilt). **This is the
one number the next sitting should re-read first**, because the H0
verdict does not depend on it but the feed verdict does.

Reference (other sittings, so context only, not a verdict; rules 9 and
20):

* NPU now: 27.74 / 26.88 / 24.83 decode tok/s at G 64 / 512 / 1024.
  This is #105 A, the GEMV-on path; the #88 B GEMV-off path read
  24.50 / 23.80 / 23.52.
* NPU prefill means: 498.5 / 453.5 / 380.7.
* CPU now: 52.43 / 49.22 / 48.31.
* Goal: ≥ 50 decode tok/s. Prefill gate: ≥ −5 % of this sitting's A.

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text = A run 1? | `source=default` line | libnntrainer.so md5 (device) |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | 421.05 | 27.887 | n/a (#89) | 4 619 692 | reference | yes (×1) | `cbc308a0ead6d5117d7daa1d08d076ac` |
| A | 64 | 2 | 540.08 | 27.814 | n/a | 5 332 352 | same | yes (×1) | same |
| A | 512 | 1 | 532.23 | 27.334 | n/a | 5 336 904 | reference | yes (×1) | same |
| A | 512 | 2 | 522.98 | 26.854 | n/a | 5 331 284 | same | yes (×1) | same |
| A | 1024 | 1 | 455.11 | 26.429 | n/a | 5 322 988 | reference | yes (×1) | same |
| A | 1024 | 2 | 469.29 | 26.469 | n/a | 5 322 556 | same | yes (×1) | same |
| A0 | 64 | 1 | 506.43 | 24.578 | n/a | 5 329 444 | same | `off (applied=0x0) source=env` | same |
| A | 64 | 3 (after the rest) | 436.12 | **16.710** | n/a | 5 326 928 | same | yes (×1) | same |

R1 (`grep -c 'moe m1 gemv: on (applied=0x1) source=default'`) = **1 in
every A log**, so A is a valid GEMV reference. No A log carries a
`[HTP-PROFILE]` block. `prefill: 512` and `generation: <G>` in all eight.

* **#101 ride-along, A vs A0 at G = 64**: 27.887 vs 24.578 decode tok/s,
  **A is +13.5 %** (the handoff expected A0 ≈ 6 % below A; it is 11.9 %
  below). Not 0 %, so the default took. Prefill A0 506.43 vs A r2 540.08
  (+6.2 % for A) and A r1 421.05 (cold first run, model page-in); A is
  not below −5 % of A0 on the warm runs.
* **Text = A**: the `diff` recipe in step 4/6 reports `DIFFERENT` for
  A0 vs A only because it compares the `[HTP] moe m1 gemv:` banner
  itself, which necessarily differs. Excluding that one line
  (`grep -v '^\[HTP\]'`) the two are **identical**, so the accuracy gate
  passes. r1 = r2 at all three G, and r3 = r1, with the recipe as given.
* **A r3 is a thermal reading, not a regression.** It ran straight after
  A0 and the two level-2 profiles, with `thermal_zone0` at 55 800 m°C:
  16.71 tok/s, −40 % against r1. Text is still identical to r1. The
  sitting reached 59 300 m°C at checkpoint 2; this unit throttles hard.

Profiles (G = 64, µs/call):

| profile | qos_mode | M==1 row (`blocks`, `m1_gemv`, host / dsp / transport / mm / swiglu / rest) | `weight DMA:` line | `staging:` | M>1 row (`m1_gemv`, host / dsp / transport) |
|---|---|---|---|---|---|
| A_L2 (R2) | 2 | `blocks=0 m1_gemv=1408/1408`, 1227.2 / 1041.5 / 185.8 / mm 980.9 / swiglu 5653.8 / rest ≤ 10.1 (**0.8 %** of host) | `n/a (direct arena read inside mm, no ring; swiglu = lane-time, 5.76 lanes busy over mm)` | `act 65536 B out 65536 B ion=y rpc allocs=19 (session) non-ION in-args=6/464 B` | `m1_gemv=0/23`, 18425.0 / 16485.9 / 1939.1 |
| A0_L2 (R4) | 2 | `blocks=5632 m1_gemv=0/1408`, 1469.2 / 1384.6 / 84.6 / mm 783.1 / swiglu 0.0 / rest ≤ 21.5 (1.5 %) | `21504 KB/call, first 1024 KB took 0 us = 8294.4 GB/s; averaged over the call 15.9 GB/s` | `act 65536 B out 65536 B ion=y rpc allocs=19 (session) non-ION in-args=6/464 B` | `m1_gemv=0/23`, 17995.8 / 16414.5 / 1581.3 |

All of the handoff's expectations for R2/R4 hold: `qos_mode=2`,
`blocks=0 m1_gemv=1408/1408` and `swiglu / mm` = 5653.8 / 980.9 = **5.76**
(matching the `5.76 lanes busy` text) for A_L2, `blocks=5632
m1_gemv=0/1408` with a normal `weight DMA:` line for A0_L2, and
`m1_gemv=0/23` on both M>1 rows.

**Transport GEMV / HMX = 185.8 / 84.6 µs/call (Δ +101.2, GEMV costs
2.20 ×).** The `act`/`out` classes are equal (65536 B each) on both, as
expected. The GEMV wins anyway on `dsp`: 1041.5 vs 1384.6 µs/call
(−24.8 %), and on host 1227.2 vs 1469.2 (−16.5 %). A0's M==1 ring is the
46-descriptor list at `engine 20.4..29.1 GB/s`; A's M==1 ring is
`desc=2/call`, confirming the body's "desc 46 → 2".

#100 gtest (`G_replay.log`, `thermal_zone0` = **46 500** m°C before it):

| line | gbs | us_per_call | busy_us lo..hi | depth_max | wait_us | checksum_ok |
|---|---|---|---|---|---|---|
| `DMA_PROBE shape=iii workers=1` | 73.2 | 10983 (us, whole cell) | — | — | — | y |
| traced | 31.3 | 720.2 | 582.6..716.7 | 11 | 710.2 | y |
| traced_f | 28.1 | 802.5 | 631.9..798.9 | 11 | 791.7 | y |
| traced_nowait | 31.2 | 722.9 | 4.5..719.3 | 45 | 0.0 | y |
| traced_gu | 31.4 | 466.9 | 3.2..463.8 | 32 | 0.0 | y |
| traced_dn | 30.9 | 237.4 | 0.9..235.1 | 8 | 0.0 | y |
| iii_chain | 31.5 | 466.8 | 2.8..463.8 | 28 | 0.0 | y |
| iii_dmstart | 30.2 | 486.8 | 0.8..481.9 | 1 | 481.1 | y |
| iii_link1 | 30.2 | 485.4 | 0.8..480.0 | 1 | 479.1 | y |
| iii_strided | 31.1 | 472.2 | 2.8..469.2 | 28 | 0.0 | y |
| **c_star** | **26.3** | 558.5 | 0.8..553.1 | 1 | 552.2 | y |
| f1 (load 0) | 31.5 | 699.8 | 688.4..697.1 | 2 | 692.9 | y |
| f1_load (load 2) | 31.4 | 700.5 | 688.8..697.5 | 2 | 693.1 | y |
| f2 (load 0) | **31.7** | 695.5 | 631.6..693.2 | 2 | 691.7 | y |
| f2_load (load 2) | **31.6** | 697.6 | 633.1..694.9 | 2 | 693.2 | y |
| f3 (load 0) | 31.4 | 700.5 | 667.0..697.6 | 2 | 691.3 | y |
| f3_load (load 2) | 31.4 | 701.8 | 667.9..698.6 | 2 | 692.1 | y |

Every `DMA_REPLAY_X` line is `checksum_ok=y` with `tag=N/N`, including
the five (`traced`, `traced_f`, `traced_nowait`, `traced_gu`,
`iii_chain`) the handoff allowed to fail on overlapping in-flight
writes. So no cell's rate has to be discarded.

Counts, exactly as the handoff's read-out asks:

```
DMA_PROBE  .. checksum_ok=y            16   (want 16)
DMA_REPLAY_X .. checksum_ok=y          16   (want 16)
DMA_REPLAY workers .. checksum_ok=n    11   (want 11, #99)
grep -E 'skipped|stale_skel'            0   (want nothing)
DMA_REPLAY_TRACE dsp_us=1222 desc=46 waits=30 blocked=3 wait_us=81 wait_act_us=77
  busy_us=536..812 depth_max=9 first_ready_us=105 last_issue_us=941
  trace_words=710 plan_shape_ok=y
```

Result lines and failure locations — exactly the expected shape:

```
[       OK ] HvxDmaProbe.DmaProbeShapes (852 ms)
[  FAILED  ] HvxDmaProbe.MoeChunkReplay (1286 ms)
[  PASSED  ] 1 test.   [  FAILED  ] 1 test: HvxDmaProbe.MoeChunkReplay
11 × unittest_hvx_dma_probe.cpp:633   (res[6] = 1467840 vs want_sum = 9461760)
0 × :715   0 × :692
```

All 11 failures are `:633`, i.e. #99 on the old `DMA_REPLAY` lines,
unchanged by this PR. Nothing at `:715` or `:692`, so **no #100 cell
failed**.

The `awk` read-out (plan §3.3 ratios), protocol run:

```
H0: c_star/probe_iii = 26.3/73.2 = 0.36 (fires < 0.80)
H1: iii_dmstart/probe_iii = 0.41 (>= 0.80), iii_chain/iii_dmstart = 1.04 (<= 0.60), iii_link1/iii_dmstart = 1.00
list: iii_chain/iii_dmstart >= 0.80 and traced/iii_chain = 0.99 (<= 0.60); traced/c_star = 1.19
  H3 traced_nowait/traced = 1.00 (>= 1.25)  H2 traced_gu/traced_dn = 1.02 (>= 1.5 either way)  H4 traced_f/traced = 0.90 (15%)  H5 iii_strided/iii_chain = 0.99 (15%)
feed f1: load0/c_star = 1.20 (>= 0.80), load2 = 31.4 GB/s (>= 37.0)
feed f2: load0/c_star = 1.21 (>= 0.80), load2 = 31.6 GB/s (>= 37.0)
feed f3: load0/c_star = 1.19 (>= 0.80), load2 = 31.4 GB/s (>= 37.0)
```

Cooled repeat (`G_replay_cool.log`, 31 800 m°C, same counts 16/16/11 and
the same 11 × `:633`):

```
H0: c_star/probe_iii = 26.4/72.5 = 0.36 (fires < 0.80)
H1: iii_dmstart/probe_iii = 0.42 (>= 0.80), iii_chain/iii_dmstart = 1.04 (<= 0.60), iii_link1/iii_dmstart = 1.00
list: traced/iii_chain = 0.99 (<= 0.60); traced/c_star = 1.18
  H3 traced_nowait/traced = 1.00  H2 traced_gu/traced_dn = 1.02  H4 traced_f/traced = 0.90  H5 iii_strided/iii_chain = 1.00
feed f1: load0/c_star = 1.19 (>= 0.80), load2 = 31.3 GB/s (>= 37.0)
feed f2: load0/c_star = 1.19 (>= 0.80), load2 = 31.3 GB/s (>= 37.0)
feed f3: load0/c_star = 1.18 (>= 0.80), load2 = 31.2 GB/s (>= 37.0)
```

For the record, the full `DMA_PROBE` sweep of the protocol run (probe
iii reads 73.2 / 84.9 / 96.5 / 101.4 GB/s for workers 1–4, against
88.8 hot / 106.9 cool … 105.3 hot for w=4 that #94 s2 read on this same
unit — see "Session drift" above):

```
shape=i    w1..w4  59.2  62.1  70.1  69.9
shape=i1   w1..w4  58.8  60.9  70.2  69.9
shape=ii   w1..w4  71.8  90.2 103.0 104.8
shape=iii  w1..w4  73.2  84.9  96.5 101.4
```

## Notes from the run

**Unit**: `R3CY205ZMND`, SM-S938N (S25 Ultra), SM8750, v79. Reached over
the ADF SSH bridge (`adf.sraisys.com:51281`), not USB, so `adb` is a
shim over `shell`/`push`/`pull` and every device-side one-liner was
pushed as a `/system/bin/sh` script (`t100_therm.sh`, `t100_cfg.sh`,
`t100_prov.sh`, `t100_run.sh`, `t100_gtest.sh`). Charger in the whole
time; battery stayed at level 100.

Thermal checkpoints (`thermal_zone0` m°C / battery °C):

| checkpoint | when | tz0 | batt |
|---|---|---|---|
| 0 | before install | 28 300 | 26.2 |
| 1 | after A G=64 r1, before the rest of A | 34 900 | 26.9 |
| 2 | after the six A cells | 59 300 | 33.6 |
| (gtest) | printed by `t100_gtest.sh` | 46 500 | — |
| 3 | after the gtest, before A0 | 38 400 | 33.8 |
| 4 | after the profiles and A r3 | 55 800 | 34.7 |
| (cool) | after a 300 s device-side cool-down | 31 800 | 27.6 |

Provenance, read back from the device (`t100_prov.sh`); the three
non-built files match the handoff table exactly, the five rebuilt ones
do not and cannot (rule 14 / another build path):

```
ca117552f4e3d0ff38b1c6c099542706  libnntr_hvx_skel.so        (handoff 25f85189…, rebuilt)
e5695f7d17325687989cd9618c56077d  nntrainer_causallm         (handoff 9ed4e439…, rebuilt)
cbc308a0ead6d5117d7daa1d08d076ac  libnntrainer.so            (handoff dcd57fc1…, rebuilt)
3ba0e0d286edae9eaf0d40b50979d16f  libcausallm_core.so        (handoff 7c84f335…, rebuilt)
1e6412e69dbfbfb3688eac97b55ca32c  libccapi-nntrainer.so      (handoff c57dd47f…, rebuilt)
c3c961bd11e23045366637c8d67f1907  unittest_hvx_dma_probe     (handoff 0f6d5341…, rebuilt)
b1586b9b512712800fd36a24abac1c0a  libc++_shared.so           == handoff
0ad4e22a70e4f135bce38ad8fd1e001b  libsdkl.so                 == handoff
fc65c1588dc66dd764c7013fe96cbb75  prompt512.txt              == handoff
7b8067a580173d3eb1697afae3b456f5  models/…/tokenizer.json    == handoff
7b7867fab51845664c0050c0a837073e  models/…/nntr_lfm2_8b_a1b_q40_arm.bin (4316133120 B) == handoff
```

The skel and the six binaries were rebuilt from `a8642bd1` with the
recipe in §"Rebuild recipe": `HEXKL_SDK_VER=6.4.0.1`, HexKL
`~/Downloads/hexkl_addon`, Hexagon SDK 6.4.0.1, NDK r30. `test/htp/build.sh`
printed `UNDEFINED SYMBOLS OK (46 runtime imports)`. The workstation
sanity checks pass on the rebuilt set: `per-layer-type totals` = 0 in
`nntrainer_causallm`, `source=%s` = **2** in `libnntrainer.so`,
`readelf -d` lists `libsdkl.so` and `libcdsprpc.so`, and no
`libcdsprpc*` was staged. `git diff --stat db7c6eb2 a8642bd1 --
nntrainer/ Applications/ test/htp/nntr_hvx.idl` is empty, so there is no
model binary change. New md5 manifest: `$W/md5.txt` on this workstation.
The model was already on the device from the #105 sitting and matched,
so nothing was pushed for it.

Nothing stale: no `FARF`, no `AEE`, no `0x8000…`, no `ERROR` in any log
of the sitting.

**Deviations from the handoff.**

1. `$W=/local/mnt/workspace/htp_moe/100/` did not exist on this
   workstation, so the artifacts were rebuilt from the commit (rule 22)
   instead of `md5sum -c md5.txt`. Consequence: six of the eleven md5s
   differ from the table, as the table itself says a skel rebuild will.
2. The first A cell (`A_G64_r1`) ran between checkpoints 0 and 1 rather
   than after checkpoint 1. Its prefill (421.05) is the cold-page-in
   number; the warm A prefills are 455–540.
3. One extra cell: the gtest was repeated after a 300 s device-side
   cool-down, because the protocol run sat at 46 500 m°C in a sitting
   that had already touched 59 300 and the feed verdict is a
   single-run decision. It reproduces within 1 % and is reported above
   as `G_replay_cool.log`.
4. `sleep` is blocked in the harness that drove this sitting, so all
   cool-downs were `adb shell "sleep N"` on the device.
5. The step 4/6 text diff compares the `[HTP] moe m1 gemv:` banner,
   which differs between A and A0 by construction. Suggested fix for the
   next handoff: pipe both sides through `grep -v '^\[HTP\]'`.

Logs: `$W/logs/{A_G64_r1,A_G64_r2,A_G512_r1,A_G512_r2,A_G1024_r1,
A_G1024_r2,A0_G64_r1,A_G64_r3,prof_A_L2,prof_A0_L2,G_replay,
G_replay_cool}.log`.
