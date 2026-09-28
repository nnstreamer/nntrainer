# 101 ride-along: extra cells for the next sitting (M=1 GEMV default, #101 + #102)

Branch `htp/101-m1-gemv-default` (PR into `htp_moe`; code head `8018c88a`).
Plan `docs/plans/101-m1-gemv-default.md` §4 step 6. **Not a sitting of its
own**: these cells are appended to the next sitting's handoff (the first
one whose variant A is built from a tree that contains this PR). Extra
device time: **≈ 5 min** (≈ 7 if that sitting has no A level-2 profile).

## What changed on the device side

Only `libnntrainer.so` (`htp_compute_ops.cpp`). No skel, IDL or DSP source
change: `git diff f3e99176 8018c88a -- test/htp ':!test/htp/host'
nntrainer/tensor/htp_backend/{hmx,hvx}` is empty, so #88's device skel
`2bd7311f…` (or any skel `test/htp/build.sh` makes from `f3e99176`..head)
is the one to push.

* Env unset now means **GEMV on**. `NNTR_MOE_HTP_M1_GEMV=0` is the opt-out.
  Proof line, once per run:
  `[HTP] moe m1 gemv: on (applied=0x1) source=default` (the old
  `moe m1 gemv: on (applied=0x1)` grep still matches).
* **Inverted rule:** from this PR on, an A log that reads `off` or has no
  `source=default` line is **not** the reference. It means A was built
  without this PR (it is the HMX path), and #101's cells move to the
  following sitting.
* A skel older than `moe_set_opts` now throws at the first MoE call
  unless `=0` is set. That is intended (LEDGER rule 17).
* Level-2 M==1 row under the GEMV path: `swiglu` is Σ lane-time (≈ 5.7 ×
  `mm`). It is no longer subtracted, so `rest` is ≥ 0. Where the `weight
  DMA:` line used to be, the block now prints
  `weight DMA: n/a (direct arena read inside mm, no ring; swiglu = lane-time, N.NN lanes busy over mm)`.

## Cells to append (all NPU model `q40-qs4cx-wh`, `NNTR_NUM_THREADS=8`, A's binaries)

The helpers `run` and `prof` are the ones in `88-moe-call-marshalling.md` §2 and §5.

| # | cell | command | min | expected / read |
|---|---|---|---|---|
| R1 | every A TPS log of the sitting | `grep -c 'moe m1 gemv: on (applied=0x1) source=default' $W/logs/A_G*_r*.log` | 0 | `1` per file. `0` voids that A as a GEMV reference (see the inverted rule) |
| R2 | A level-2 profile, G = 64 (skip if the sitting already runs one) | `prof A_L2 "NNTR_HTP_PROFILE=2"` | 2 | M==1 row `blocks=0 m1_gemv=1408/1408`; `rest` ≥ 0 and ≤ 5 % of host; `swiglu / mm` ≈ 5.7; next line `weight DMA: n/a (direct arena read …, N.NN lanes busy over mm)` with N.NN ≈ 5.7; `staging:` line present; M>1 row `m1_gemv=0/23`; header `qos_mode=2` |
| R3 | **A0**: opt-out TPS, G = 64, one run | `run A0 64 1 NNTR_MOE_HTP_M1_GEMV=0` | 1.5 | `[HTP] moe m1 gemv: off (applied=0x0) source=env`; text identical to `A_G64_r1` (`diff <(sed -n '/^=====/q;p' …)`); decode expected ≈ 6 % **below** A (sitting 2's C/A inverted). A 0 % delta means the default did not take, which voids the sitting's A as the GEMV reference. Prefill: A within −5 % of A0 |
| R4 | **A0** level-2 profile, G = 64 | `prof A0_L2 "NNTR_HTP_PROFILE=2 NNTR_MOE_HTP_M1_GEMV=0"` | 2 | M==1 row `blocks=5632 m1_gemv=0/1408` (HMX loop), a normal `weight DMA:` line, `rest` unchanged in meaning |

**Transport question (supervisor).** Sitting 2 measured GEMV transport
+100 µs/call over the HMX loop (649 → 748, before #103's staging). Does
that survive #103? Read the M==1 `transport=` of **R2 minus R4**. Both
cells come from one binary in one sitting, so this is the only valid
comparison (rule 23: transport is not a cross-sitting column; #88 B's 87.9
is context only). Report it as `transport GEMV / HMX = x / y µs/call (Δ)`
together with the two `staging:` lines. Expected: equal `act`/`out`
classes (65536 B), because the call's arguments do not depend on the path.

Paste into that sitting's results: R1's counts, R2/R4's M==1 and M>1 rows
plus their `weight DMA:` / `staging:` lines, R3's prefill and decode tok/s
and text-identity verdict, and the transport Δ.

## Rebuild recipe (LEDGER rule 22; the staging path below is a convenience, not a deliverable)

```
git fetch origin && git checkout 8018c88a      # or htp_moe once the PR is merged
source tools/htp/env.sh                        # HEXKL_SDK_VER=6.4.0.1, NDK r30, v79
./test/htp/build.sh                            # only if no skel from f3e99176..head is at hand; must print UNDEFINED SYMBOLS OK (46 runtime imports)
(cd builddir && ninja install)                 # existing builddir: refresh android_build_result first, or --cache links a stale libnntrainer.so
# fresh builddir instead: ./build_android.sh --htp; cd builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install
(cd Applications/CausalLM && ./build_android.sh --htp --cache)
strings Applications/CausalLM/jni/obj/local/arm64-v8a/libnntrainer.so | grep -c 'source=%s'          # 2
strings Applications/CausalLM/jni/obj/local/arm64-v8a/libnntrainer.so | grep -c 'direct arena read'  # 1
```

Env var names: `NNTR_MOE_HTP_M1_GEMV` (unset/`1` = on, `0` = off),
`NNTR_HTP_PROFILE`, `NNTR_NUM_THREADS=8`.

## Staged set

`W=/local/mnt/workspace/htp_moe/101/`, built from `cc485637`. Its source
equals `8018c88a` except `test/htp/host/run_host_checks.sh`, which is
host-only. `$W/md5.txt`:

| file | md5 | note |
|---|---|---|
| `libnntr_hvx_skel.so` | `46f34d5f6b04c1f75ab9f450ac026e2d` | `test/htp/build.sh` at this head. Skel builds are not byte-reproducible here (3 builds, 3 md5s), so #88's device skel `2bd7311f…` is equally valid (same DSP source) |
| `A/nntrainer_causallm` | `5a977ff86addc774b8cb7caa15f54391` | == #88 B staged (unchanged by this PR) |
| `A/libcausallm_core.so` | `db004ee3e3eb9cbbdbb0c5dfb88b3410` | == #88 B staged |
| `A/libnntrainer.so` | `8d14e3508601ad29118ebb1be55dee23` | **the one file this PR changes** (`jni/obj/local/arm64-v8a/`) |
| `A/libccapi-nntrainer.so` | `df7cc241def803d997fd8a0581064936` | == #88 B staged |
| `gtest/unittest_hvx_mm_u8i4` | `a03625e046e33214827df68d672177de` | == #88 staged (does not link `libnntrainer.so`; `MoeLayerM1GemvMatchesHmx` sets the flag itself) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77/#88) |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77/#88) |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | == #77/#88 |

The model (`7b7867fa…`) and `tokenizer.json` (`7b8067a5…`) are unchanged
from #88. `readelf -d libnntrainer.so` lists `libsdkl.so` and
`libcdsprpc.so`. A0 is not a separate binary: it is A with the env var.
