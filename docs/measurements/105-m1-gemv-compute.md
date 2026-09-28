# Measurement 105: the M=1 GEMV's compute side — one-row loop and l2fetch lead (LEDGER ㉒)

Branch `htp/105-m1-gemv-compute` (code head `a0ae8b29`, this file on the
same branch) — estimated device time: **≈ 50 min** (≈ 40 min if the
short form in step 3 is taken)

Plan: `docs/plans/105-m1-gemv-compute.md` §4 step 5. Issue #105 (tracker
#76; LEDGER ㉒ compute half, ⑯). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`.

**Run right after the handoff-88 sitting, same phone: the NPU model it
installed is reused** (step 1 checks its md5 and skips the 4.3 GB push).
Only the NPU model `q40-qs4cx-wh` runs here; no CPU `q40` cell.

## Why

On the M=1 GEMV path (#94 sitting 2 variant C) the level-2 M==1 row's
`mm` is 975 µs/call: 22.0 MB of arena weight at 22.6 GB/s, with the pool's
lanes busy ≈ 96 % of it. At m = 1 the kernel ran the four-row loop — 32
`vrmpy` per k-tile for 8 that are stored, over four rotating accumulators
(the V79 HVX PRM §5.6 accumulator stall) — and each column's `l2fetch`
went out right before its own loads. **Decision that hangs on this
sitting:** B's `mm` below A's by more than 5 % with decode ≥ A at every G
= pass; `mm` ≤ 600 µs closes ㉒; the microbench says how much of what is
left is feed (㉒'s DMA/VTCM half, #100) versus compute.

Variants (4; one app set, **only the skel changes**):

* **A** — skel from `htp_moe` @ `48fd2420` (the head this branch forks
  from; its DSP sources = the #94 C path). Run first.
* **B1** — one-row loop (`gemm_row1`), per-column `l2fetch` as before
  (`-DHVX_GEMV_PF_LEAD_KB=0u`).
* **B2** — B1 + the next block's `l2fetch` box issued a block ahead, 64 KB
  lead (the branch default).
* **B3** — B1 + 192 KB lead (where L2 eviction would show, as B3 < B2).

`NNTR_MOE_HTP_M1_GEMV=1` is set **explicitly in every cell, A included**
(it resolves to on whether or not #101 has landed). Every log must print
`[HTP] moe m1 gemv: on (applied=0x1)` (a `source=…` suffix is fine) and
every level-2 M==1 row `blocks=0 m1_gemv=1408/1408`, or the run is void.

The ARM side is untouched: the branch's diff against `htp_moe` is
`nntrainer/tensor/htp_backend/{hvx/hvx_gemm_u8i4_wh.{c,h},
hmx/hexkl_mm_u8i4_moe.{c,h}}` (DSP only; `htp_backend/meson.build`
compiles neither into `libnntrainer.so`), `test/htp/host/*` and the
gtest. So one app set serves all four variants, and the generated text
must be byte-identical across A/B1/B2/B3 per G and run (the int32 is
unchanged by construction; see Gates). `NNTR_L2_DIFF` n/a for
`QS4CX_WH`; text = CPU `q40` n/a (different weights) — the accuracy
column is "= A".

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/105/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. The B skels and the
app were built in `/home/j2z0-lee/nntrainer-88-A` on this branch (skels
@ `a0ae8b29`), A's skel in the detached worktree
`/home/j2z0-lee/nntrainer-105-A` @ `48fd2420`. Skel builds are not
bit-reproducible here (two builds of B2's source gave two md5s), so the
**staged file's md5 is the identity**; the variant flag was confirmed in
each binary by disassembly (B1: workers call `hvx_gemm_u8i4_wh_col`; B2/B3:
`_col_nopf` + `_prefetch`, lead constant `0x10000` vs `0x30000`).

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/libnntr_hvx_skel.A.so` | `9bcb81e496800215a2c73a7f865328b8` | `htp_moe` @ `48fd2420`, `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)` |
| `$W/libnntr_hvx_skel.B1.so` | `a3f5d43175177eb19abf70a19d5a52b2` | `a0ae8b29`, `HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=0u`; clean, `UNDEFINED SYMBOLS OK (46)` |
| `$W/libnntr_hvx_skel.B2.so` | `18eeb1c84fe8d068368a623372195dea` | `a0ae8b29`, default (64 KB); clean, `UNDEFINED SYMBOLS OK (46)` |
| `$W/libnntr_hvx_skel.B3.so` | `1e08fd1032e6998c94b02e1c3f548738` | `a0ae8b29`, `HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=192u`; clean, `UNDEFINED SYMBOLS OK (46)` |
| `$W/app/nntrainer_causallm` | `01bd38f6af041e7d3d028b485bf40193` | this branch, `build_android.sh --htp` (+ `ninja -C builddir install`, then `--htp --cache`); `jni/libs/arm64-v8a/`. Same md5 as #88's A file (unchanged sources, deterministic ndk-build) |
| `$W/app/libcausallm_core.so` | `0a700c028c3fede5990a95e245716b51` | same; == #88 A |
| `$W/app/libnntrainer.so` | `ddc61f2945d4e331376f3e991bac67f0` | same (`jni/obj/local/arm64-v8a/`); `readelf -d` NEEDED `libsdkl.so`, `libcdsprpc.so`. Carries PR #103 (it is `htp_moe` post-merge), so it differs from #88's A `341d5762…` and, by build path (rule 14), from #88's B `4f6c90dd…` |
| `$W/app/libccapi-nntrainer.so` | `571c0ff2168dc6b82381a1e393895981` | same; == #88 A |
| `$W/app/libc++_shared.so`, `$W/gtest/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 / #88 / #94) |
| `$W/app/libsdkl.so`, `$W/gtest/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 / #88 / #94) |
| `$W/gtest/unittest_hvx_mm_u8i4` | `e498d6c2a283b7bb2f570bb2c32d10dc` | `ndk-build … unittest_hvx_mm_u8i4` on this branch (`test/jni/obj/local/arm64-v8a/`); has `MoeLayerM1GemvMatchesHmx` and the new `MoeM1GemvFeedVsCompute` |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | copied from `/local/mnt/workspace/htp_moe/94/prompt512.txt` (the #88 / #94 prompt) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; on the phone since the #88 sitting) |
| `tokenizer.json` (`q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/105
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')           # 0
strings $W/app/nntrainer_causallm | grep -c 'per-layer-type totals'  # 0 (not a profile binary)
find $W -name 'libcdsprpc*' | wc -l                                 # 0
```

### If `$W` is not on the workstation you measure from

The #88 sitting ran on a workstation without the staging directory. Rebuild
the set from the commits instead; skel builds are not byte-reproducible, so
the md5s will differ from the table. Record the ones you push under "Notes
from the run" (rules 14 and 21). Build from a clean checkout, NDK r30,
SDK 6.4.0.1, HexKL 6.4.0.1 (`source tools/htp/env.sh` or the same exports).

```
W=/local/mnt/workspace/htp_moe/105; mkdir -p $W/app $W/gtest
git fetch origin && git worktree add /tmp/n105A 48fd2420 && git worktree add /tmp/n105B a0ae8b29
(cd /tmp/n105A && HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh && cp test/htp/build/libnntr_hvx_skel.so $W/libnntr_hvx_skel.A.so)
for v in B1:0u B2:64u B3:192u; do
  (cd /tmp/n105B && HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=${v#*:} HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh \
     && cp test/htp/build/libnntr_hvx_skel.so $W/libnntr_hvx_skel.${v%%:*}.so)
done                                                    # each build prints UNDEFINED SYMBOLS OK (46 runtime imports)
# One app set serves A and every B (this PR changes no ARM source). Build it from a0ae8b29:
(cd /tmp/n105B/Applications/CausalLM && ./build_android.sh --htp)   # fresh builddir: see the hexagon-gates skill, rung 3
cp /tmp/n105B/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} $W/app/
cp /tmp/n105B/Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so} $W/app/
strings $W/app/libnntrainer.so | grep -c 'staging: act'           # 1: PR #103 is in the app (else stale --cache: ninja install, rebuild)
# gtest: ndk-build unittest_hvx_mm_u8i4 in /tmp/n105B/test/jni (rung 3), copy it and libc++_shared.so to $W/gtest/
# libsdkl.so into both dirs from $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/; prompt512.txt = docs/measurements/77-prompt512.txt
(cd $W && md5sum libnntr_hvx_skel.*.so app/* gtest/* prompt512.txt > md5.txt)
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer-88-A && git fetch && git checkout htp/105-m1-gemv-compute && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/105; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
sk() { # sk <A|B1|B2|B3>: the variant's skel into both dirs, device md5 logged
  adb push $W/libnntr_hvx_skel.$1.so $D/libnntr_hvx_skel.so >/dev/null && \
  adb push $W/libnntr_hvx_skel.$1.so $T/libnntr_hvx_skel.so >/dev/null && \
  adb shell "md5sum $D/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so" | sed "s/^/skel $1: /" | tee -a $W/logs/skel_push.log
}
```

### 0. Device state (1 min)

```
adb devices          # exactly one device; record the serial under Notes (any S25 Ultra, contract §4.2)
therm                # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
```

Screen off, charger in. If the phone is warm from the #88 sitting, wait
until `thermal_zone0` is back near the value #88's checkpoint 1 read.

### 1. Install (≈ 3 min) — model reused from #88

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin     # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin    # 7b7867fab51845664c0050c0a837073e -> no push
# only if either line differs:  adb push /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models $T
adb push $W/app/nntrainer_causallm $W/app/libcausallm_core.so $W/app/libnntrainer.so $W/app/libccapi-nntrainer.so $W/app/libc++_shared.so $W/app/libsdkl.so $W/prompt512.txt $D/
adb push $W/gtest/unittest_hvx_mm_u8i4 $W/gtest/libc++_shared.so $W/gtest/libsdkl.so $T/
adb shell "chmod 755 $D/nntrainer_causallm $T/unittest_hvx_mm_u8i4"
sk A
```

(#88 leaves its own A or B app on the phone: the push above replaces all
four files, which is why the md5 line below matters.)

Config edits, re-applied unconditionally (greedy, exact count, `moe_engine: htp`):

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
adb shell "md5sum $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so $T/unittest_hvx_mm_u8i4 $D/prompt512.txt $D/models/q40-qs4cx-wh/tokenizer.json"
```

### 2. Proof + microbench per skel (≈ 12 min) — one level-2 run and the two gtests, A first

One helper per variant: skel swap, a level-2 profile run at G = 64 (never
read for tok/s), then the gtests with the thermal zone around them (rule
18).

```
prof() { # prof <A|B1|B2|B3>
  sk $1
  adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
    NNTR_MOE_HTP_M1_GEMV=1 NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/prof_$1.log | grep -E 'moe m1 gemv|level=|K=2048 N=2048' | cut -c1-420
  adb shell cat /sys/class/thermal/thermal_zone0/temp | sed "s/^/therm before gtest $1: /" | tee -a $W/logs/therm.log
  adb shell "cd $T && md5sum libnntr_hvx_skel.so unittest_hvx_mm_u8i4 && NNTR_MOE_HTP_M1_GEMV=1 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'" 2>&1 | tee $W/logs/gtest_$1.log | grep -E 'libnntr_hvx_skel|U8I4_FIELD|PASSED|FAILED|SKIPPED'
  adb shell cat /sys/class/thermal/thermal_zone0/temp | sed "s/^/therm after gtest $1: /" | tee -a $W/logs/therm.log
}
therm                                                          # checkpoint 1
prof A; prof B1; prof B2; prof B3
therm                                                          # checkpoint 2
```

Expected per variant:

* `prof_<v>.log`: one `[HTP] moe m1 gemv: on (applied=0x1)`; the header
  `[HTP-PROFILE] level=2 qos_mode=2 …` (**`qos_mode=2` in all four**; a
  `1` voids that profile); the M==1 row `K=2048 N=2048 M==1 calls=1408 …
  mm <µs> … blocks=0 m1_gemv=1408/1408]`; the M>1 row with
  `m1_gemv=0/<calls>` (prefill stays on the HMX loop).
* `gtest_<v>.log`: the md5 line of the skel equals the table's row for
  `<v>`; `U8I4_FIELD path=moe_m1_gemv field=bad_elems_M1 value=0 of 2048`,
  `field=bad_elems_M4 value=0 of 8192`, `field=bit_identical value=yes`;
  three `U8I4_FIELD path=m1_bench cell=arena|heap|hot mm_us=… mm_min_us=…
  lane_us=… lanes=… tiles=… ns_per_tile=… gbps=…` lines (no `INVALID`);
  `[  PASSED  ] 2 tests.`

A `bad_elems` ≠ 0 under any B skel is a failed accuracy gate: stop, note
it, skip that variant's E2E cells. A `MoeM1GemvFeedVsCompute` failure on
the `hot` cell alone (it routes one weight pair to all four experts, a
first for this call) does not void the variant: note the message and
continue.

### 3. E2E cells (≈ 30 min) — mirrored order per G

```
run() { # run <A|B1|B2|B3> <G> <run#>   (the variant's skel must already be on the device)
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    NNTR_MOE_HTP_M1_GEMV=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv|libnntr_hvx_skel'
}
cell() { sk $1 >/dev/null; run $1 $2 $3; }
for g in 64 512 1024; do
  cell A $g 1; cell B1 $g 1; cell B2 $g 1; cell B3 $g 1
  cell B3 $g 2; cell B2 $g 2; cell B1 $g 2; cell A $g 2
  therm                                                        # checkpoints 3, 4, 5
done
```

Each log carries the device md5 of the skel it ran (the `md5sum` in
`run`); a cell whose md5 is not its variant's table row is void (rule
21). Expected lines in every log: `prefill: 512 tokens, … TPS`,
`generation: <G> tokens, … TPS`, `total: … ms`, `peak memory: … KB`, one
`[HTP] moe m1 gemv: on (applied=0x1)`, and **no** `[HTP-PROFILE]` block.
`prefill: N` with N ≠ 512 voids the prefill column; `generation: N` ≠ G
is a failed config edit.

**Short form (≈ 40 min total), if time runs out:** keep A and B2 at all
three G; B1 and B3 at G = 64 only (the profile + gtest from step 2 stay).

### 4. Checks on the workstation (1 min)

```
grep -c 'moe m1 gemv: on (applied=0x1)' $W/logs/*_G*_r*.log $W/logs/prof_*.log   # 1 each
grep -l 'HTP-PROFILE' $W/logs/*_G*_r*.log                                        # nothing
for v in A B1 B2 B3; do printf '%s: ' $v; grep 'M==1' $W/logs/prof_$v.log | grep -o 'blocks=[0-9]* m1_gemv=[0-9/]*'; done   # blocks=0 m1_gemv=1408/1408 x4
for g in 64 512 1024; do for r in 1 2; do for v in B1 B2 B3; do
  printf '%s vs A G=%s r%s: ' $v $g $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') \
       <(sed -n '/^=====/q;p' $W/logs/${v}_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') >/dev/null && echo same || echo DIFFERENT
done; done; done
cat $W/logs/skel_push.log $W/logs/therm.log
```

Then fill the tables below, commit this file on the same branch, push,
set #105 to `state:measured`.

## Results (2026-09-22, unit `R3CY205ZMND`)

Unit (serial from `adb devices`): **`R3CY205ZMND`** (SM-S938N, `ro.board.platform=sun`, v79).
Battery 100 % on charger, screen off, all through.
Battery °C / `thermal_zone0` m°C at the checkpoints: **0** 27.9 / 31000 ·
**1** 28.1 / 30200 · **3** (after the G = 64 block) 34.8 / 55400 · **4**
(after G = 512) 37.9 / 64300 · **5** (after G = 1024) 41.4 / 66300.
Checkpoint 2 is replaced by the second profile pass below.

The sitting ran through the ADF SSH bridge (`adf.sraisys.com`, no local
USB), so every `adb` line is a shim over that bridge and the device-side one-liners were pushed as
`/system/bin/sh` scripts. `$W` did not exist on this workstation: the whole
set was rebuilt from the commits per the recipe above, so **every md5 below
is this workstation's build, not the table's** (rules 14 and 21). Staged
md5 = device md5 for all 14 files, and every log carries the md5 of the
skel it ran (0 mismatches over 24 E2E cells, 8 profile runs and 4 gtest
runs).

Rebuilt artifact md5s (`$W/md5.txt`, all verified on the device):

| file | md5 (this workstation) | handoff table said |
|---|---|---|
| `libnntr_hvx_skel.A.so` | `a1af011b79934faac684b1e1a54032c2` | `9bcb81e4…` |
| `libnntr_hvx_skel.B1.so` | `f6bc4014c1da87306bf0d53bec55c3a7` | `a3f5d431…` |
| `libnntr_hvx_skel.B2.so` | `81d7f48acc55fcda695e97f471099789` | `18eeb1c8…` |
| `libnntr_hvx_skel.B3.so` | `9aa17b57c9b26250d4aeadcba74b6029` | `1e08fd10…` |
| `app/nntrainer_causallm` | `30cf2472920f5320b4bc66b0a2d3ae6b` | `01bd38f6…` |
| `app/libnntrainer.so` | `b2a8016758ec8b269e01eb4ec84b2c4b` | `ddc61f29…` |
| `app/libcausallm_core.so` | `94d3fd94c6064f13e5b56366965bada5` | `0a700c02…` |
| `app/libccapi-nntrainer.so` | `160e20dd4c1521b02722d71854d19d62` | `571c0ff2…` |
| `gtest/unittest_hvx_mm_u8i4` | `f74e8a0b5f0676cc201d9ff5598cdf0d` | `e498d6c2…` |
| `app/libc++_shared.so`, `gtest/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | **=** |
| `app/libsdkl.so`, `gtest/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | **=** |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | **=** |
| `tokenizer.json` (device) | `7b8067a580173d3eb1697afae3b456f5` | **=** |
| model `nntr_lfm2_8b_a1b_q40_arm.bin` (device, 4316133120 B) | `7b7867fab51845664c0050c0a837073e` | **=** (reused, no push) |

Since the skels are not byte-reproducible, the variant flag was re-proved
in each rebuilt binary by disassembly, the same way the handoff did it:
`A` exports only `hvx_gemm_u8i4_wh_col`; B1/B2/B3 all export
`_col_nopf` + `_prefetch`; the lead constant appears as **B1** none
(baseline 16 × `#0x10000` only), **B2** 8 extra `#0x10000` (64 KB),
**B3** 8 × `#0x30000` (192 KB). App checks: `staging: act` = 1 (PR #103
present), `per-layer-type totals` = 0 (not a profile binary), no
`libcdsprpc*` staged.

### Proof and kernel columns

`[HTP-PROFILE]` level 2, G = 64, µs/call. **Two passes**, because in pass 1
(the handoff's order A → B1 → B2 → B3, back to back) the four runs did not
start from the same thermal state — A started at 30–32 °C and the B runs at
50–58 °C. Pass 2 re-read all four in the **reverse** order, each from a
34.5 °C cold start (`cool.sh` polls `thermal_zone0` device-side). The two
passes agree to **≤ 0.4 %** on every `mm`, so the column is per-variant and
the thermal worry is closed; the tables below give pass 1 / pass 2.

| variant | skel md5 (device) | qos_mode | M==1 `mm` | M==1 `dsp` | M==1 `swiglu` | M==1 host / transport | blocks / m1_gemv | M>1 `dsp` | M>1 m1_gemv |
|---|---|---|---|---|---|---|---|---|---|
| A | `a1af011b…` | 2 / 2 | 973.1 / 972.9 | 1032.6 / 1032.5 | 5612.3 / 5610.2 | 1214.7 / 1216.6 · 182.0 / 184.2 | 0 / 1408/1408 (both) | 16661.8 / 16416.0 | 0/23 |
| B1 | `f6bc4014…` | 2 / 2 | 1044.2 / 1045.0 | 1104.5 / 1105.2 | 6202.8 / 6207.3 | 1286.9 / 1291.4 · 182.5 / 186.3 | 0 / 1408/1408 (both) | 16447.4 / 16324.0 | 0/23 |
| B2 | `81d7f48a…` | 2 / 2 | 1006.7 / 1003.0 | 1065.8 / 1062.0 | 5981.8 / 5959.5 | 1249.2 / 1247.5 · 183.4 / 185.5 | 0 / 1408/1408 (both) | 16613.0 / 16527.1 | 0/23 |
| B3 | `9aa17b57…` | 2 / 2 | 929.5 / 932.2 | 989.4 / 992.7 | 5452.7 / 5465.4 | 1175.3 / 1183.9 · 186.0 / 191.2 | 0 / 1408/1408 (both) | 16618.2 / 16818.0 | 0/23 |

Means and deltas against A (`mm`): A **973.0** · B1 **1044.6 (+7.4 %)** ·
B2 **1004.9 (+3.3 %)** · B3 **930.9 (−4.3 %)**. `dsp`: 1032.6 / 1104.9 /
1063.9 / 991.1. `swiglu` is still the mis-attributed bucket (#102) and is
not read. M>1 `dsp` spans 16324–16818 over all eight runs, i.e. every
variant within **2.5 %** of A — the M>1 path is untouched, as designed.

### Gtests (per skel; `thermal_zone0` before → after, pass 1)

| variant | bad_elems_M1 | bad_elems_M4 | bit_identical | therm (m°C) | arena mm / min / lane_us / ns_per_tile / gbps | heap mm / min / lane_us / ns_per_tile / gbps | hot mm / min / lane_us / ns_per_tile / gbps |
|---|---|---|---|---|---|---|---|
| A | 0 of 2048 | 0 of 8192 | yes | 32500 → 34900 | 830 / 748 / 4805 / **111.72** / 26.53 | 850 / 838 / 4977 / 115.72 / 25.91 | 31 / 29 / 154 / **70.77** / 35.94 |
| B1 | 0 of 2048 | 0 of 8192 | yes | 56200 → 47700 | 1052 / 1045 / 6246 / **145.23** / 20.93 | 1017 / 1012 / 6026 / 140.11 / 21.65 | 20 / 19 / 93 / **42.74** / 55.71 |
| B2 | 0 of 2048 | 0 of 8192 | yes | 57700 → 51200 | 997 / 979 / 5912 / **137.46** / 22.09 | 1058 / 1049 / 6255 / 145.44 / 20.81 | 19 / 17 / 86 / **39.52** / 58.64 |
| B3 | 0 of 2048 | 0 of 8192 | yes | 57700 → 52300 | 880 / 875 / 5171 / **120.23** / 25.02 | 976 / 963 / 5729 / 133.21 / 22.56 | 16 / 14 / 68 / **31.25** / 69.63 |

`[  PASSED  ] 2 tests.` in all four; no `INVALID`, no `SKIPPED`, and the
`hot` cell never failed. `lanes` reads 5.8–5.9 on arena/heap and 4.3–5.0 on
the (much shorter) hot cell.

### E2E (NPU model, `NNTR_MOE_HTP_M1_GEMV=1`, prompt 512, 8 threads)

Every cell: `prefill: 512 tokens`, `generation: <G> tokens`, exactly one
`[HTP] moe m1 gemv: on (applied=0x1)`, no `[HTP-PROFILE]` block, and the
in-log skel md5 equal to its variant's row (24/24).

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text = A (same G, run)? | skel md5 (device, from the log) |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | 539.5 | 27.960 | n/a (#89) | 4790420 | reference | `a1af011b…` |
| B1 | 64 | 1 | 506.4 | 26.745 | n/a | 4961708 | yes | `f6bc4014…` |
| B2 | 64 | 1 | 487.6 | 27.223 | n/a | 4906256 | yes | `81d7f48a…` |
| B3 | 64 | 1 | 475.0 | 28.508 | n/a | 4856292 | yes | `9aa17b57…` |
| B3 | 64 | 2 | 493.3 | 28.432 | n/a | 4951828 | yes | `9aa17b57…` |
| B2 | 64 | 2 | 482.1 | 27.211 | n/a | 5029992 | yes | `81d7f48a…` |
| B1 | 64 | 2 | 471.5 | 26.490 | n/a | 5262300 | yes | `f6bc4014…` |
| A | 64 | 2 | 457.6 | 27.527 | n/a | 5074368 | reference | `a1af011b…` |
| A | 512 | 1 | 516.1 | 27.089 | n/a | 5251436 | reference | `a1af011b…` |
| B1 | 512 | 1 | 427.4 | 25.907 | n/a | 5197960 | yes | `f6bc4014…` |
| B2 | 512 | 1 | 433.9 | 25.233 | n/a | 5236136 | yes | `81d7f48a…` |
| B3 | 512 | 1 | 372.6 | 27.533 | n/a | 4798148 | yes | `9aa17b57…` |
| B3 | 512 | 2 | 396.9 | 27.431 | n/a | 5057704 | yes | `9aa17b57…` |
| B2 | 512 | 2 | 398.4 | 26.197 | n/a | 5312920 | yes | `81d7f48a…` |
| B1 | 512 | 2 | 403.2 | 25.684 | n/a | 5322848 | yes | `f6bc4014…` |
| A | 512 | 2 | 390.8 | 26.669 | n/a | 4967492 | reference | `a1af011b…` |
| A | 1024 | 1 | 401.6 | 25.674 | n/a | 5322572 | reference | `a1af011b…` |
| B1 | 1024 | 1 | 435.7 | 23.381 | n/a | 5154852 | yes | `f6bc4014…` |
| B2 | 1024 | 1 | 331.0 | 23.749 | n/a | 4802660 | yes | `81d7f48a…` |
| B3 | 1024 | 1 | 358.3 | 24.327 | n/a | 5141020 | yes | `9aa17b57…` |
| B3 | 1024 | 2 | 356.3 | 24.506 | n/a | 5315796 | yes | `9aa17b57…` |
| B2 | 1024 | 2 | 358.3 | 23.005 | n/a | 5320812 | yes | `81d7f48a…` |
| B1 | 1024 | 2 | 364.2 | 22.121 | n/a | 5322896 | yes | `f6bc4014…` |
| A | 1024 | 2 | 359.8 | 23.978 | n/a | 5328960 | reference | `a1af011b…` |

Decode means (of the two mirrored runs) and deltas against A:

| G | A | B1 | B2 | B3 |
|---|---|---|---|---|
| 64 | 27.743 | 26.617 (−4.06 %) | 27.217 (−1.90 %) | **28.470 (+2.62 %)** |
| 512 | 26.879 | 25.795 (−4.03 %) | 25.715 (−4.33 %) | **27.482 (+2.24 %)** |
| 1024 | 24.826 | 22.751 (−8.36 %) | 23.377 (−5.84 %) | 24.416 (−1.65 %) |

Prefill means: A 498.5 / 453.5 / 380.7; B1 488.9 / 415.3 / 399.9; B2
484.9 / 416.2 / 344.6; B3 484.1 / 384.8 / 357.3 (G = 64 / 512 / 1024).
Several B cells fall below −5 %, but A's own two runs span 539.5 → 457.6
at G = 64 and 516.1 → 390.8 at G = 512 (−8 to −14 % purely by position in
the block: the prefill column tracks the block's thermal ramp, checkpoint
3 → 5 is 55.4 → 66.3 °C), and the M>1 `dsp` of every variant is within
2.5 % of A's. **By the handoff's own noise rule the prefill column is
order drift, and the prefill gate is not failed.**

## Verdict: no variant passes; the compute half alone does not close ㉒

* **Gate** = M==1 `mm` < A × 0.95 (= 924.4 µs) **and** decode mean ≥ A at
  G = 64, 512 **and** 1024.
* **B1 (one-row loop, no lead) — fail.** `mm` **+7.4 %** and decode below A
  at all three G. The one-row loop *by itself* is a regression on the
  arena path.
* **B2 (+64 KB lead) — fail.** `mm` +3.3 %, decode below A at all three G.
* **B3 (+192 KB lead) — fail, but the best cell.** `mm` −4.3 %, inside the
  ±5 % band, so the `mm` half of the gate misses; decode is +2.6 / +2.2 %
  at G = 64 / 512 but −1.65 % at G = 1024 (where A's own spread is 25.674
  → 23.978, 7 %), so the decode half is a wash rather than a pass.
* **`mm` ≤ 600 µs is not reached** — best is 930.9 µs — so **㉒ stays open**
  and its remainder keeps waiting on the feed half (#100), exactly as the
  issue's fallback branch says.
* **Accuracy: pass, everywhere.** `bit_identical value=yes` and
  `bad_elems_M1/M4 = 0` under all four skels, generated text byte-identical
  to A in 18/18 B cells, `blocks=0 m1_gemv=1408/1408` in all eight profile
  runs, `qos_mode=2` in all eight, M>1 `m1_gemv=0/23` everywhere.
* **Prefill: pass** (M>1 `dsp` within 2.5 % of A; the tok/s spread is the
  block's thermal ramp, see above).

### What the microbench says (the split this sitting was for)

`ns_per_tile`, lower is better:

| loop | arena | heap | hot | hot / arena |
|---|---|---|---|---|
| four-row (A) | 111.72 | 115.72 | 70.77 | 0.63 |
| one-row, lead 0 (B1) | 145.23 | 140.11 | 42.74 | 0.29 |
| one-row, lead 64 KB (B2) | 137.46 | 145.44 | 39.52 | 0.29 |
| one-row, lead 192 KB (B3) | 120.23 | 133.21 | 31.25 | 0.26 |

Three readings, and they all point the same way:

1. **`hot` ≪ `arena` in every variant**, and the gap widens under the
   one-row loop (0.63 → 0.26). So the arena path is **not** issue-bound:
   plan §3.4's first branch is rejected and the remainder is **feed**.
   Under B3, `arena − hot` = 89.0 of 120.2 ns/tile, i.e. **≈ 74 % of the
   GEMV's `mm`** — of B3's 930.9 µs about **690 µs is feed and about 240 µs
   is compute**. That 240 is the number ㉒'s feed half (#100, VTCM staging)
   would expose, and it is well under the 600 µs target; under A's four-row
   loop the same arithmetic leaves ≈ 590 µs, right at the target.
   **So the one-row loop is what turns ≤ 600 from borderline into
   comfortable — but only after the feed half lands.**
2. **The one-row loop is exactly the 2× win it was designed to be when the
   feed is not in the way**: `hot` 70.77 → 31.25 ns/tile, **2.26×**, close
   to the 4 → 1 `vrmpy` accounting. On silicon today that win is invisible
   because the arena feed dominates, and it even inverts: with 2 `vrmpy`
   per quarter-tile instead of 8 there is less in-flight work to hide the
   DDR read latency, which is why B1's `arena` is 30 % *worse* than A's.
   The `l2fetch` lead buys that latency tolerance back — monotonically,
   145.23 → 137.46 → 120.23 as the lead goes 0 → 64 → 192 KB — but at
   192 KB it has still not caught A.
3. **`heap` ≈ `arena`** (within 11 % either way, no consistent sign), so
   the **ION mapping itself is not the feed problem** — it is plain
   uncached DDR read bandwidth (21–27 GB/s, the same band #59 measured).
   Plan §3.4's third branch is rejected too.

### Follow-ups this sitting earns

* **㉒'s feed half (#100, VTCM staging) is now measured, not assumed: it is
  ≈ 74 % of the M=1 GEMV's `mm`.** It is the only lever left that reaches
  ≤ 600 µs.
* **The lead is not saturated at 192 KB** (the trend is still monotonic and
  B3 is the best cell in every single column — `mm`, decode at two of three
  G, and all three microbench cells). A lead sweep past 192 KB (384 KB, and
  an L2-budget check) is the cheap next probe, and it belongs in the same
  place the feed work goes.
* **PR default after read-back:** the handoff's rule ("the better of
  B2/B3") selects **B3, i.e. `HVX_GEMV_PF_LEAD_KB=192`**, not the branch's
  current 64. B1 does not beat B2, so 0 is out.
* Whether PR #107 should land at all despite failing today's gate is the
  supervisor's call, and the case for it is reading 2 above: the kernel is
  correct (bit-identical, text-identical), it is 2.26× faster on the L2-hot
  path, and it is the half of the ≤ 600 target that cannot be built after
  the feed half without redoing this work.

### Reference the handoff wrote before the run (kept for the record)

#94 sitting 2, same unit `R3CY205ZMND`, `htp_moe` @ `2a75f7d9`, variant C =
this A's path: level-2 M==1 `mm` 974.7, `dsp` 1044.0, host 1792.2,
transport 748.2 µs/call, `blocks=0 m1_gemv=1408/1408`; M>1 `dsp` 16554.3;
decode 18.83 / 18.33 / 17.59 tok/s (G 64 / 512 / 1024, means of two),
prefill 389–527. Goal ≥ 50 decode tok/s, prefill ≥ −5 % of A. Read against
this sitting: `mm` and M>1 `dsp` reproduce (973.0 and 16324–16818), host,
transport and decode do not, because the app here carries PR #103.

## Gates passed on the workstation (host evidence, not device)

* `bash test/htp/host/run_host_checks.sh` (after `source tools/htp/env.sh`):
  `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4; tiny+real)` and
  `M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN (lanes=6; lead 0|64|192 KB)`
  with `ALL CHECKS PASS` for each of the three leads; `WORKER POOL LANES
  OK`; `HVX GEMV NATIVE cases=1152 bad=0` / `HVX GEMV NATIVE BIT-IDENTICAL
  (libnative; m=1..16 rows1+rows4)` — the real kernel on x86 against the
  SDK's HVX emulation; mutants caught: `gemm_row1` shift `bad=9216` (only
  the lone-row cases), `gemm_rows4` shift `bad=82944`.
* `ninja -C build`; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'`
  6/6 PASSED (fixture generated); `*qs4cx*` 2/2; `tools/htp_syntax_check.sh`
  exit 0.

## Notes from the run

* **Unit / bridge.** `R3CY205ZMND` (SM-S938N, `sun`, v79) over the ADF SSH
  bridge — no local USB on this workstation. `getprop ro.serialno` stands in
  for `adb devices`; the bridge takes only `shell` / `push` / `pull`, and it
  mangles quoting, so `therm`, the config edits, `prof`, `gt`, `run` and a
  `cool` helper were pushed as `/system/bin/sh` scripts under
  `/data/local/tmp/` and invoked as `shell sh <path>`. Host key checked
  against the fingerprint the Terminal panel showed
  (`SHA256:gmhiquYuizssrdAW9El92totPqZS/fid3iv28oThjoU`).
* **`$W` was not on this workstation**, so the whole set was rebuilt per the
  recipe. Three things the recipe does not mention and that cost time:
  * A fresh worktree has **empty `subprojects/`** (`iniparser`, `googletest`,
    `benchmark`, `CLBlast`, `OpenBLAS`, `ruy`) and `build_android.sh` dies
    with `fatal error: 'iniparser.h' file not found` after ~15 min of
    compiling. The wrap files are identical to the #88 worktree's, so the
    populated directories were copied from there.
  * `Applications/CausalLM/lib/libtokenizers_android_c.a` is likewise
    per-checkout (Rust); copied from the #88 worktree rather than rebuilt.
  * `libc++_shared.so` is **not** under `jni/obj/local/arm64-v8a/` on this
    box; it was taken from the NDK r30 sysroot
    (`toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/`)
    and its md5 `b1586b9b…` matches the table and #88's staged copy.
  * Env used: `HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1`,
    `HEXKL_ROOT=$HOME/Downloads/hexkl_addon`, `HEXKL_SDK_VER=6.4.0.1`,
    `ANDROID_NDK=$HOME/android-ndk-r30` — note `HEXKL_ROOT` differs from
    `env.sh`'s default on this box and must be set explicitly, or the link
    silently picks another HexKL version.
* **Model reused**, as planned: the device copy is 4316133120 B with md5
  `7b7867fab51845664c0050c0a837073e`, so the 4.3 GB push was skipped. The
  phone had cooled from the earlier sitting on its own (battery log shows
  38.5 → 27.7 °C); checkpoint 0 read 31000 m°C.
* **Second profile pass, and why.** In pass 1 the four profile runs went
  back to back in the handoff's order, and the `thermal_zone0` readings
  taken around the gtests showed A's block at 30–35 °C and the B blocks at
  50–58 °C. Rather than read a possibly order-biased `mm` column, all four
  were re-run in reverse order, each from a 34.5 °C cold start. The two
  passes agree to ≤ 0.4 % on every `mm` (A 973.1/972.9, B1 1044.2/1045.0,
  B2 1006.7/1003.0, B3 929.5/932.2), so **the `mm` column is per-variant,
  not per-position** — the opposite of the #53 outcome, and the reason the
  verdict above is stated as a fail rather than as "one sitting only".
  `thermal_zone0` here recovers fast (57 → 34.5 °C in ≤ 60 s idle).
* **The E2E prefill column does drift with position**, though: A's own two
  runs read 539.5 / 457.6 at G = 64 and 516.1 / 390.8 at G = 512, and the
  ramp follows checkpoints 3 → 5 (55.4 → 66.3 °C). Decode is far steadier
  (A's worst pair is 25.674 / 23.978 at G = 1024). The mirrored order per G
  is what makes the decode means usable.
* **A reads 27.7 decode tok/s here against #94 sitting 2's 18.8** for the
  same DSP path, because the app carries PR #103 (`staging: act` present).
  So this sitting's A is the right and only baseline for these B numbers;
  the #94 reference row is not comparable on decode. The M==1 `mm`, by
  contrast, reproduces #94 almost exactly (973.0 here vs 974.7 there),
  which is the cross-check that the A skel really is that path.
* **No errors of the kinds the handoff asks about:** no `qos_mode=1`
  header (all eight profile runs read `qos_mode=2`), no `AEE_EBADPARM`
  (`0x8000040E`), no `0x80000406` open failure, no
  `nntr_hvx_moe_set_opts failed`, no `INVALID` microbench cell, no
  `SKIPPED` gtest. The `hot` cell (the one the handoff warns may fail)
  passed under all four skels.
* Logs, both profile passes and `md5.txt` are under
  `/local/mnt/workspace/htp_moe/105/` (`logs/prof_*.log`,
  `logs/prof2_*.log`, `logs/gtest_*.log`, `logs/<v>_G<g>_r<n>.log`,
  `logs/therm.log`, `logs/skel_push.log`, `logs/provenance.log`).
