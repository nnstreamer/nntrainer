# Measurement 120: `htp_moe` + upstream PR #4327 head `4ae1ebd7`, A/B in one sitting

Branch `htp/120-upstream-4327-sync` (merge commit `84fb7119` onto
`htp_moe` @ `d79c0efe`, four small commits on top, code head `0e98036d`;
this file on the same branch) — estimated device time: **≈ 40 min**

Issue #120 (tracker #76; contract §5 Q16, the merge is the user's decision
of 2026-09-23). Contract `docs/plans/0001-htp-moe-decode-agent-system.md`.
Only the NPU model `q40-qs4cx-wh` runs here; no CPU `q40` cell. The model
is the one the #100 / #113 / #117 sittings installed — step 1 checks its
md5 and skips the 4.3 GB push.

## Why

`htp_moe` now carries every one of upstream PR #4327's 40 commits (docs
50–53, the conv block as one call, the dense FFN through the MoE kernel,
the fused qkv layer, the down matmul one block behind the gate_up, the
FC profile row, the pooled FC epilogue, `NNTR_PPL`). The M=1 GEMV path
that `htp_moe`'s decode number rests on — one-row loop, 192 KB lead and,
since PR #118, the VTCM feed — was re-derived on top of upstream's
rewritten block loop, and the merge is host-verified only:
`run_host_checks.sh` `ALL CHECKS PASS` (M1 GEMV bit-identical to the HMX
stand-in with `feed=arena,vtcm`, the in-situ chunk plan matching the new
push order), the skel `UNDEFINED SYMBOLS OK`, the six `*Lfm2Moe*` gtests.

**Decision that hangs on it:** whether the merged tree is the new
`htp_moe` (PR #121 into `htp_moe`, `state:review`). Gate (issue #120):
decode ≥ −2 % of A at each G (mean of two runs), prefill ≥ −5 %, text
identical to A at every G. A default-on upstream change that moves
decode is named in the read-out. If B's text differs from A's, the
suspect is upstream `1f538c20` (q/k/v and their norms as one CPU layer;
"the same three GEMMs and the same norm kernel" is its claim) — file it,
do not average it in.

**Rides along (cycle 15, coordinator):** the A/A0 confirmation of PR
#118's VTCM feed default, which #117's handoff scheduled for "the next
sitting". A0 is A with `NNTR_MOE_HTP_GEMV_FEED=0` (the D192 arena read,
`applied=0x103e1`), G = 64 × 2 only. Read A vs A0 as #117 §"Why" says:
A ≥ A0 confirms the default; A < A0 by more than 2 % is a LEDGER entry,
not a verdict on this merge.

### Variants (3; contract §4.2)

| | tree | app set | skel | env | `applied` | cells |
|---|---|---|---|---|---|---|
| **A** (unchanged reference, runs first) | `htp_moe` @ `d79c0efe` (PR #118: D192 + VTCM feed default) | `$W/A/` | `$W/A/libnntr_hvx_skel.so` | none | `0x303e1 … feed=vtcm source=default` | G = 64 / 512 / 1024 × 2, mirrored |
| **B** | merge @ `0e98036d` | `$W/B/` | `$W/B/libnntr_hvx_skel.so` | none | `0x303e1 … feed=vtcm source=default` | same |
| **A0** | = A | `$W/A/` | `$W/A/libnntr_hvx_skel.so` | `NNTR_MOE_HTP_GEMV_FEED=0` | `0x103e1 … feed=arena source=default` | G = 64 × 2 |

**Each variant runs with its own tree's skel.** The IDL differs between the
two trees (upstream added `mm_u8i4_conv_block`), so A's app with B's
skel — or the reverse — fails with `AEE_EBADPARM (0x8000040E)` or
`0x80000406`. `use` below copies the whole set into place and prints the
skel md5 before every cell.

Nothing is set in the environment beyond `NNTR_NUM_THREADS=8` (A0 adds
its one variable). Every A and B log must print exactly one
`[HTP] moe m1 gemv: on (applied=0x303e1) lead=192KB rows1=1 feed=vtcm source=default`
(PR #118's default; B carries it unchanged), every A0 log
`… (applied=0x103e1) lead=192KB rows1=1 feed=arena source=default`.
Every level-2 M==1 row must read `blocks=0 m1_gemv=1408/1408
feed=1408/1408` (A0: `feed=0/1408`). In B, every upstream engine knob is
at its default (`cpu` / unset), so the per-token forward path is A's plus
the qkv fusion on the CPU side: no `dense`, `conv` or FC row may appear
in B's profile (`M==1 dense` / `M==1 conv` / `M==1 FC`, or their `M>1`
forms); if one does, the config on the device was edited, and the cell is
void.

`NNTR_L2_DIFF` and "text = CPU q40" are n/a for `QS4CX_WH` (different
weights); the accuracy column is **"= A"**.

### Ride-along (free, ≈ 1 min): the device gtest under B's skel

`MoeLayerM1GemvMatchesHmx` (`unittest_hvx_mm_u8i4`) proves the M=1 GEMV
path byte-equal to the HMX loop on silicon, on B's kernel with upstream's
block loop under it. `bit_identical value=yes`, `bad_elems_M1 value=0` or
the merge is wrong regardless of the tok/s columns.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/120/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. A was built in
`/home/j2z0-lee/nntrainer-120A` (a detached worktree at `d79c0efe`), B in
`/home/j2z0-lee/nntrainer-120` at `0e98036d`. Skel builds are **not**
byte-reproducible on this machine, so the *staged file's* md5 is the
identity and the device `md5sum` in every log is what ties a cell to it.

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/A/libnntr_hvx_skel.so` | `df496672acd5fbbd33249f17cb40f965` | `d79c0efe`, `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`, no `HEX_EXTRA_CFLAGS`; `UNDEFINED SYMBOLS OK (46 runtime imports)` |
| `$W/A/nntrainer_causallm` | `79128b9396f54d9e7f73831c8052912a` | `d79c0efe`, `build_android.sh --htp` (`jni/libs/arm64-v8a/`) |
| `$W/A/libcausallm_core.so` | `cd858c808523fdeaedb81f1b1c8be93e` | same |
| `$W/A/libnntrainer.so` | `90e0bbd9a7257bf47f0e7681e8e184ac` | same (`jni/obj/local/arm64-v8a/`); `readelf -d` NEEDED `libsdkl.so`, `libcdsprpc.so`; carries the `moe m1 gemv: %s (applied=0x%x)` banner |
| `$W/A/libccapi-nntrainer.so` | `fa9b2785c83153dfdc98df8690cc0230` | same |
| `$W/B/libnntr_hvx_skel.so` | `c65d6dfe852cee6767aad38741c9e3e6` | `0e98036d`, same build line; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)` |
| `$W/B/nntrainer_causallm` | `94c243922fc87e6e024c8640d1010aae` | `0e98036d`, `build_android.sh --htp` |
| `$W/B/libcausallm_core.so` | `7a6e9e99bd1886bbf78a26e0507944ce` | same |
| `$W/B/libnntrainer.so` | `a37ddb22ecdf710c6103cc2fd8c8b45c` | same; NEEDED `libsdkl.so`, `libcdsprpc.so`; the banner string present |
| `$W/B/libccapi-nntrainer.so` | `e0b50a159389c918a26f1497490bd53e` | same |
| `$W/gtest/unittest_hvx_mm_u8i4` | `35331a3985679cdf65129a1529a59a10` | `0e98036d`, `ndk-build … unittest_hvx_mm_u8i4` (`test/jni/obj/local/arm64-v8a/`) |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 … #117) |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 … #117) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; already on the phone). Its file order is q, q_norm, k, k_norm, v per layer, which is the fused qkv layer's weight order, so B reads it unchanged |
| `tokenizer.json` (`q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')                 # 0
for v in A B; do strings $W/$v/nntrainer_causallm | grep -c 'per-layer-type totals'; done   # 0 0 (not profile binaries)
for v in A B; do strings $W/$v/libnntrainer.so | grep -c 'feed=%s source=%s'; done          # 1 1 (#118's banner in both)
find $W -name 'libcdsprpc*' | wc -l                                      # 0
```

### If `$W` is not on the workstation you measure from (rule 22)

Rebuild both sets from their commits (A from `d79c0efe`, B from
`0e98036d`) with the recipe below; record the md5s you push under "Notes
from the run". Clean checkout, NDK r30, SDK 6.4.0.1, HexKL 6.4.0.1
(`source tools/htp/env.sh`, and export `HEXKL_ROOT` explicitly — its
location differs between workstations).

```
W=/local/mnt/workspace/htp_moe/120; mkdir -p $W/A $W/B $W/gtest
for v in A B; do case $v in A) sha=d79c0efe;; B) sha=0e98036d;; esac
  git fetch origin && git worktree add /tmp/n120$v $sha && cd /tmp/n120$v
  git submodule update --init --depth 1                     # a fresh worktree starts with subprojects/ empty
  cp <another worktree>/Applications/CausalLM/lib/libtokenizers_android_c.a Applications/CausalLM/lib/ \
    || ./Applications/CausalLM/build_tokenizer_android.sh   # per-checkout, needs Rust
  HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh && cp test/htp/build/libnntr_hvx_skel.so $W/$v/
                                                            # prints UNDEFINED SYMBOLS OK (46 runtime imports)
  (cd Applications/CausalLM && ./build_android.sh --htp)    # fresh builddir fails in `ninja install` on /usr/local:
  (cd builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install)
  (cd Applications/CausalLM && ./build_android.sh --htp --cache)
  cp Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} \
     Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so} $W/$v/
done
# B only: the device gtest
cd /tmp/n120B && ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT unittest_hvx_mm_u8i4 -j8)
cp test/jni/obj/local/arm64-v8a/unittest_hvx_mm_u8i4 $W/gtest/
cp $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $W/
cp $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/libsdkl.so $W/
cp docs/measurements/77-prompt512.txt $W/prompt512.txt
(cd $W && md5sum A/* B/* gtest/* libc++_shared.so libsdkl.so prompt512.txt > md5.txt)
```

`--cache` skips `builddir` wholesale once
`builddir/android_build_result/lib/arm64-v8a/libnntrainer.so` exists:
after any edit under `nntrainer/`, run `ninja -C builddir && ninja -C
builddir install` first, or `libnntrainer.so` stays stale (the banner
check above is what catches it).

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer-120 && git fetch && git checkout htp/120-upstream-4327-sync && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/120; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
set_of() { case $1 in A0) echo A;; *) echo $1;; esac; }          # A0 runs A's binaries
env_of() { case $1 in A0) echo "NNTR_MOE_HTP_GEMV_FEED=0";; *) echo "";; esac; }
use() { adb shell "cd $D && cp $(set_of $1)/nntrainer_causallm $(set_of $1)/*.so . && md5sum libnntr_hvx_skel.so nntrainer_causallm"; }
```

### 0. Device state (1 min)

```
adb devices          # exactly one device; record the serial under Notes (any S25 Ultra, contract §4.2)
therm                # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 5 min) — model reused

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin     # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin    # 7b7867fab5... -> no push
# only if either line differs:  adb push /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models $D/A $D/B $T
adb push $W/A/. $D/A/ && adb push $W/B/. $D/B/
adb push $W/libc++_shared.so $W/libsdkl.so $W/prompt512.txt $D/
adb push $W/gtest/unittest_hvx_mm_u8i4 $W/libc++_shared.so $W/libsdkl.so $W/B/libnntr_hvx_skel.so $T/
adb shell "chmod 755 $D/A/nntrainer_causallm $D/B/nntrainer_causallm $T/unittest_hvx_mm_u8i4"
adb shell "md5sum $D/A/* $D/B/* $T/unittest_hvx_mm_u8i4 $T/libnntr_hvx_skel.so $D/prompt512.txt \
  $D/models/q40-qs4cx-wh/tokenizer.json" | tee $W/logs/md5.log            # must equal the table
```

Config edits, re-applied unconditionally (greedy, exact count, `moe_engine: htp`):

```
adb shell "cd $D/models/q40-qs4cx-wh && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`, and
**no other `_engine` key** (`attn_proj_engine`, `conv_block_engine`,
`dense_ffn_engine`, `conv_in/out_proj_engine` absent = `cpu`).

### 2. Ride-along, once (≈ 1 min) — B's kernel bit-identical on silicon

```
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*'" \
  2>&1 | tee $W/logs/gtest_B.log | grep -E 'U8I4_FIELD|PASSED|FAILED|SKIPPED'
```

Expected: `U8I4_FIELD path=moe_m1_gemv field=bad_elems_M1 value=0 …`,
`field=bit_identical value=yes`, `[  PASSED  ] 1 test.`

### 3. Profiles (≈ 4 min) — one level-2 run per variant at G = 64, never read for tok/s

```
prof() { # prof <A|B|A0>
  use $1
  adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
    $(env_of $1) NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/prof_$1.log | grep -E 'moe m1 gemv|level=|K=2048 N=2048|staging:|M==1 (dense|conv|FC)|M>1 (dense|conv|FC)' | cut -c1-420
}
therm                                  # checkpoint 1
prof A; prof B; prof A0
therm                                  # checkpoint 2
```

Expected per variant: the banner with the variant's word from the table
above; `[HTP-PROFILE] level=2 qos_mode=2 …` (`qos_mode=2` in all three; a
`1` voids that profile); the M==1 row `K=2048 N=2048 M==1 calls=1408 …
swiglu … mm <µs> … blocks=0 m1_gemv=1408/1408 feed=1408/1408]` (A0:
`feed=0/1408`; B prints `swiglu(hidden)` where A prints `swiglu` — same
meaning, upstream `c64be9dd` reconciled with PR #108's row); the M>1 row
with `m1_gemv=0/23`; the `staging:` line; **no** dense / conv / FC rows.

### 4. E2E cells (≈ 24 min) — mirrored order per G, A0 at G = 64 only

```
run() { # run <A|B|A0> <G> <run#>
  use $1
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $1) NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv|libnntr_hvx_skel'
}
run A 64 1; run B 64 1; run A0 64 1
run A0 64 2; run B 64 2; run A 64 2
therm                                  # checkpoint 3
for g in 512 1024; do
  run A $g 1; run B $g 1
  run B $g 2; run A $g 2
  therm                                # checkpoints 4, 5
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `total: … ms`, `peak memory: … KB`, exactly one
`[HTP] moe m1 gemv: on (applied=0x…)` with the variant's word, and **no**
`[HTP-PROFILE]` block. `prefill: N` with N ≠ 512 voids the prefill
column; `generation: N` ≠ G is a failed config edit. The skel md5 printed
before each cell must be the variant's tree's own.

### 5. Checks on the workstation (1 min)

```
grep -c 'applied=0x303e1) lead=192KB rows1=1 feed=vtcm source=default' $W/logs/[AB]_G*_r*.log $W/logs/prof_[AB].log   # 1 each
grep -c 'applied=0x103e1) lead=192KB rows1=1 feed=arena source=default' $W/logs/A0_G*_r*.log $W/logs/prof_A0.log     # 1 each
grep -l 'HTP-PROFILE' $W/logs/*_G*_r*.log                                 # nothing
for v in A B A0; do printf '%s: ' $v; grep 'M==1' $W/logs/prof_$v.log | grep -o 'blocks=[0-9]* m1_gemv=[0-9/]* feed=[0-9/]*'; done
grep -c -E 'M==1 (dense|conv|FC)|M>1 (dense|conv|FC)' $W/logs/prof_B.log  # 0
for g in 64 512 1024; do for r in 1 2; do
  printf 'B vs A G=%s r%s: ' $g $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm') \
       <(sed -n '/^=====/q;p' $W/logs/B_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm') >/dev/null && echo same || echo DIFFERENT
done; done
for r in 1 2; do printf 'A0 vs A G=64 r%s: ' $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G64_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm') \
       <(sed -n '/^=====/q;p' $W/logs/A0_G64_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm') >/dev/null && echo same || echo DIFFERENT; done
cat $W/logs/md5.log $W/logs/therm.log
```

Then fill the tables below, commit this file on the same branch, push,
set the issue to `state:measured`.

## Results

Run 2026-09-23 17:55–18:08 by the agent on the user's request. Unit
`R3CY10WM83Y` (SM-S938N, SM8750). `decode tok/s (last 64)` is not
printed by this app and is left n/a.

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text = A? (first differing token) | skel md5 (device) |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | 473.636 | 35.4178 | n/a | 5328660 | (reference) | `df496672…` |
| B | 64 | 1 | 551.724 | 37.6471 | n/a | 5310776 | yes | `c65d6dfe…` |
| A0 | 64 | 1 | 514.056 | 28.5714 | n/a | 5327996 | yes | `df496672…` |
| A0 | 64 | 2 | 507.433 | 28.6738 | n/a | 5319772 | yes | `df496672…` |
| B | 64 | 2 | 544.681 | 37.7136 | n/a | 5313520 | yes | `c65d6dfe…` |
| A | 64 | 2 | 501.469 | 36.5297 | n/a | 5319656 | (reference) | `df496672…` |
| A | 512 | 1 | 446.382 | 36.2298 | n/a | 5315840 | (reference) | `df496672…` |
| B | 512 | 1 | 511.489 | 36.1837 | n/a | 5307520 | yes | `c65d6dfe…` |
| B | 512 | 2 | 500.489 | 36.4024 | n/a | 5312264 | yes | `c65d6dfe…` |
| A | 512 | 2 | 471.889 | 35.6919 | n/a | 5316216 | (reference) | `df496672…` |
| A | 1024 | 1 | 459.193 | 35.2338 | n/a | 5326508 | (reference) | `df496672…` |
| B | 1024 | 1 | 476.723 | 35.2787 | n/a | 5316680 | yes | `c65d6dfe…` |
| B | 1024 | 2 | 476.723 | 34.9273 | n/a | 5323728 | yes | `c65d6dfe…` |
| A | 1024 | 2 | 438.356 | 35.1058 | n/a | 5322960 | (reference) | `df496672…` |

Text: the pre-`=====` output of every log (banner, md5 and
`num_to_generate` lines excluded) hashes identically across all variants
and runs of each G (`55f577d9` / `d7e89826` / `db08b4e1`).

Means of the two runs and deltas:

| G | decode A | decode B | B vs A | prefill A | prefill B | B vs A | A0 decode / prefill | A vs A0 decode |
|---|---|---|---|---|---|---|---|---|
| 64 | 35.974 | **37.680** | **+4.74 %** | 487.55 | **548.20** | **+12.44 %** | 28.623 / 510.74 | **+25.68 %** |
| 512 | 35.961 | **36.293** | **+0.92 %** | 459.14 | **505.99** | **+10.20 %** | — | — |
| 1024 | 35.170 | **35.103** | **−0.19 %** | 448.77 | **476.72** | **+6.23 %** | — | — |

Reference (not a gate; rule 23): NPU "now" **27.85 / 27.09 / 26.45**
decode tok/s at G 64 / 512 / 1024 (#100 A, the pre-D192 arena default)
and prefill 430–506; D192 read +4.11 / +3.29 / +4.26 % in #113; the VTCM
feed (this A) read 37.38 / 36.49 / 35.01 in #117's cooled sitting on the
same unit. Goal ≥ 50, prefill ≥ 497.

### Level-2 profile, M==1 row (G = 64)

| variant | `dsp` µs/call | `host` | transport | `quant` | `swiglu` | `requant` | `scatter` | `stage` | `mm` | `rest<=` | `feed=` | `staging:` act/out/ion | `qos_mode` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 711.7 | 793.9 | 82.2 | 7.7 | 1306.7 | 9.4 | 7.5 | 5.7 | **677.2** | 4.2 | 1408/1408 | 4194304 / 4194304 / y (and 65536 / 65536 / y) | 2 |
| B | 711.3 | 799.0 | 87.7 | 8.4 | 1296.3 (`swiglu(hidden)`) | 9.3 | 7.3 | 5.9 | **676.6** | 3.7 | 1408/1408 | same | 2 |
| A0 | 997.4 | 1184.1 | 186.7 | 16.6 | 5464.0 | 12.3 | 12.6 | 10.8 | **934.3** | 10.7 | 0/1408 | same | 2 |

All three M==1 rows `blocks=0 m1_gemv=1408/1408`. A and B `DMA ring:
desc=10/call … engine 32.3..34.4 GB/s depth max=4`. B's profile has no
`dense` / `conv` / `FC` row (count 0).

M>1 row (prefill, `m1_gemv=0/23 feed=0/23`): `dsp` A **16986.0** B
**14807.7** (−12.8 %; A0 17007.2). The drop is in `requant` 1099.0 →
70.0, `dequant` 929.0 → 275.4, `mm` 10222.0 → 9671.6 and transport
1978.5 → 1082.4 — upstream's default-on M>1 changes (the down matmul one
block behind the gate_up, the pooled epilogue), which is the prefill gain
in the E2E column.

### Ride-along

`unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*'` under
B's skel: `bit_identical` **yes**, `bad_elems_M1` **0 of 40960**
(`bad_elems_M4` 0 of 163840), `PASSED` **y** (`[  PASSED  ] 1 test.`).

## Verdict

* **Gate** (decode ≥ −2 % of A at each G, prefill ≥ −5 %, text = A):
  **pass** — decode +4.74 / +0.92 / −0.19 %, prefill +12.44 / +10.20 /
  +6.23 %, text identical at every G. The merge moves decode by nothing
  outside run noise (M==1 `dsp` 711.7 vs 711.3); it moves **prefill up**
  by +6–12 % through upstream's default-on M>1 changes (M>1 `dsp`
  −12.8 %). G = 64's +4.7 % decode is mostly A's run 1 (35.42, first
  after the profiles) against its run 2 (36.53).
* **Ride-along gtest**: pass.
* **A vs A0** (PR #118's VTCM feed default): decode **+25.68 %** at G = 64
  (35.97 vs 28.62), text identical, M==1 `mm` 677.2 vs 934.3 → the
  default is confirmed.

## Notes from the run

* Serial `R3CY10WM83Y` (a second phone, `R3CN80CW3FY`, was attached; all
  commands used `ANDROID_SERIAL`). Battery 92 → 88 %, USB, screen off.
  Checkpoints (zone0 m°C / battery 0.1 °C): ckpt0 26600 / 255; before the
  profiles 29700–35100 / 255–268; ckpt2 58700 / 271; ckpt3 58700 / 303;
  after G 512 60600 / 314; after G 1024 60200 / 325.
* One change to the steps: a cool-down before each profile and each G
  block (battery ≤ 33.0 °C and zone0 < 38 °C, capped at 5 min; waits 0–45
  s). zone0 still reads 53–61 °C during every run.
* Device `md5sum` of every pushed file equals `$W/md5.txt` (0
  mismatches, `logs/md5.log`); skel md5 before each cell: A/A0
  `df496672…` (8 logs), B `c65d6dfe…` (6 logs).
* The model directory `models/q40-qs4cx-wh` on this phone was created in
  #117's sitting: the workstation's config files, with the bin (md5
  `7b7867fab5…`) and `tokenizer.json` symlinked from
  `models/lfm2.5-8b-a1b-q40-qs4cx-wh`. Config echo: `do_sample false`,
  `bad_word_ids [124900]`, `init_seq_len 512`, `moe_engine htp`,
  `moe_htp_layers ""`, no other `_engine` key.
* Every log has exactly one banner with its variant's word; no
  `HTP-PROFILE` in any E2E log; no cell voided; no FARF/AEE errors seen.
