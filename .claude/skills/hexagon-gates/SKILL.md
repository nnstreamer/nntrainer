---
name: hexagon-gates
description: The verification ladder for any change on the htp_moe branch (format → host build and host checks → DSP skel → Android app and device gtest → device handoff), with the exact workstation commands and what "pass" means at each rung. Use before claiming an HTP change works.
---

All commands run natively on the Ubuntu workstation from the repo root
after `source tools/htp/env.sh` (SDK 6.4.0.1, HexKL beta.2 `lib/6.4.0.1`,
NDK r30, model dir; contract `docs/plans/0001-htp-moe-decode-agent-system.md`
§4.1). Never report a rung as passed without its pass line in the output.
Rungs are cumulative: a PR needs 0–3; a step inside a plan needs the rung
the plan names. There is no simulator rung in this tree.

**Budget.** The host build is minutes, the skel seconds, the Android app
build ≈ 10 min from scratch (`--cache` reuses `builddir`). Run rung 2 only
when a file under `test/htp/` or `nntrainer/tensor/htp_backend/` (the DSP
sources and the IDL) changed; run rung 3 once before a PR or a handoff,
not per commit.

## 0. Format (every commit)

```
clang-format-14 -i <changed .c .cpp .h>
```
Pass: `git diff --stat` after formatting shows only intended files.
Static-check rules (CI): commit body ≥ 8 words, new `.c/.h/.cpp/.py` files
carry `@file` / `@brief`, no merge commits.

## 1. Host (no SDK; minutes)

```
ninja -C build
./build/test/unittest/unittest_nntrainer_cpu_backend --gtest_filter='*qs4cx*'
NNTR_QUANTIZE_BIN=$PWD/build/Applications/CausalLM/nntr_quantize \
NNTR_QUANTIZE_STREAM_BIN=$PWD/build/Applications/CausalLM/nntr_quantize_stream \
  ./build/Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2Moe*'
bash test/htp/host/run_host_checks.sh
bash tools/htp_syntax_check.sh
bash test/htp/host/run_inproc_e2e.sh      # SDK headers + qaic sourced; no device
```
Pass: every gtest `[  PASSED  ]` — 6 for `*Lfm2Moe*` (3 differential + 3
tiny-model), none skipped; `run_host_checks.sh` prints `ALL CHECKS PASS`
and `WORKER POOL LANES OK`; the syntax check exits 0; `run_inproc_e2e.sh`
prints `E2E eval golden … bit_identical=1`, `E2E eval hmx-loop …
bit_identical=1`, `E2E tokens htp==cpu 8/8`, `E2E eval cpu … min_snr_db=`
(≥ 60), `E2E eval self-test ok` and `INPROC E2E PASS` (it configures
`build_htp_host/` itself; tens of minutes the first time, ≈ 1 min after;
needs `source tools/htp/env.sh` for the SDK headers and `qaic`, nothing
else — HexKL and the phone are not involved, and no number it prints is
a device number). Since #130 it also prints
`E2E fwd tiny … calls/token=11`, `E2E eval fwd-tiny … min_snr_db=` (≥ 30),
`E2E fwd tiny-all-kinds refused: AEE_ESCHEMENOTSUPPORTED`, `E2E eval
golden-hd64 … bit_identical=1`, `E2E fwd hd64 … calls/token=12`, `E2E eval
fwd-hd64 … min_snr_db=` (≥ 30) and `E2E tokens fwd==off 8/8
expected_mismatch=0`. The two fixtures'
weight files are gitignored and generated once per checkout:
`python3 test/unittest/models/causallm_reference/generators/generate_lfm2_moe_reference.py`
then `git checkout -- test/unittest/models/causallm_reference/lfm2_moe_tiny/`
(the generator also rewrites `meta.json` / `nntr_config.json`; the
reference logits are deterministic and must not change), and the hd64 one
with `--dim 128 --n-heads 2 --n-kv-heads 1 --head-dim 64 --max-pos 32
--out test/unittest/models/causallm_reference/lfm2_moe_tiny_hd64` then the
same `git checkout --` of that directory. Without the first the
three differential tests are `SKIPPED`, which is not a pass. The `build/` directory is configured with
`meson setup build -Denable-transformer=true -Denable-tflite-backbone=false
-Denable-tflite-interpreter=false` plus
`-Dc_args=-Wno-error=missing-include-dirs -Dcpp_args=-Wno-error=missing-include-dirs`
(no `flatc` on this machine; the cmake subproject include dirs do not
exist here). Re-run `meson setup --reconfigure` with the same flags if the
directory is missing.

**New DSP kernel rule (doc 45 §3.3/§3.4).** Any op that feeds a quantizer
(RMSNorm, gating, SwiGLU, RoPE, q/k norm) or any M=1 kernel ships with a
scalar spec in `test/htp/host/` and a host check proving bit-identity
against it, in the same PR. The HVX GEMV must produce the same int32
accumulator as the HMX path; that comparison is a host check first and
a device gtest second.

## 2. DSP skel (SDK; seconds)

```
./test/htp/build.sh                       # v79; HEXKL_ROOT / HEXKL_SDK_VER from env.sh
md5sum test/htp/build/libnntr_hvx_skel.so
```
Pass: `test/htp/build/libnntr_hvx_skel.so` exists, `-Wall -Werror` clean,
and `build.sh` printed `UNDEFINED SYMBOLS OK (<n> runtime imports)` (a
project symbol left out of `SRCS` links fine and fails on the device with
`0x80000406`, #97; the guard prints the offending names and exits 1).
`build.sh` regenerates the FastRPC stub/skel from `test/htp/nntr_hvx.idl`;
when the IDL changed, the host side must be rebuilt too (rung 1 and 3),
or the device fails with `AEE_EBADPARM (0x8000040E)`. Variants:
`HEX_EXTRA_CFLAGS=-D...` if the plan defines them; copy each skel to
`test/htp/build/libnntr_hvx_skel.<variant>.so` before the next build.

## 3. Android app and device gtest (SDK + NDK; ≈ 10 min)

```
(cd Applications/CausalLM && ./build_android.sh --htp)        # add --cache to reuse builddir
readelf -d builddir/android_build_result/lib/arm64-v8a/libnntrainer.so | grep -E 'libsdkl|libcdsprpc'
md5sum Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} \
       Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so}
ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest     # once per checkout; in .git/info/exclude
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
   unittest_hvx_mm_u8i4 unittest_hvx_softmax unittest_hvx_attn unittest_hvx_fc -j8)
md5sum test/jni/obj/local/arm64-v8a/unittest_hvx_{mm_u8i4,softmax,attn,fc}
```
With NDK r30 the shared libraries `libnntrainer.so`, `libccapi-nntrainer.so`
and `libc++_shared.so` stay under `jni/obj/local/arm64-v8a/`, while
`install_android.sh` pushes them from `jni/libs/arm64-v8a/`; a handoff
lists the `obj/local` paths (or copies them into `libs/` first) so the
install step does not push stale files.
Pass: both `NEEDED` lines present (`libsdkl.so`, `libcdsprpc.so`); the
binaries exist; md5s recorded for the handoff. Never `--clean` a builddir
that was configured with `--htp` unless you re-run with `--htp` (the
option lives only in `builddir` and is dropped silently). A `--profile`
build is a separate builddir and is never the TPS binary.

Workstation specifics (2026-09-21, first build): `tools/package_android.sh`
ends in `ninja install`, and the googletest subproject installs to the
meson `prefix` (`/usr/local`, not writable here) — on a fresh `builddir`
run `cd builddir && meson configure -Dprefix=$PWD/android_build_result &&
ninja install`, then `build_android.sh --htp --cache` for the rest
(nntrainer's own Android outputs ignore `prefix`). The tokenizer library
(`Applications/CausalLM/lib/libtokenizers_android_c.a`) is built once by
`build_tokenizer_android.sh` with Rust (`~/.cargo`, target
`aarch64-linux-android`; env.sh puts cargo on PATH).

Fresh checkout or new `git worktree` (the #105 sitting's rebuild lost
≈ 15 min to each of these; LEDGER §3a):
* `subprojects/` starts empty; run `git submodule update --init --depth 1`
  first, or `build_android.sh` fails late on `iniparser.h`.
* `Applications/CausalLM/lib/libtokenizers_android_c.a` is per-checkout:
  copy it from another worktree or run `build_tokenizer_android.sh`.
* If `libc++_shared.so` is missing under `jni/obj/local/arm64-v8a/`, take
  it from `$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/`
  (md5 `b1586b9b…` for r30).
* Export `HEXKL_ROOT` explicitly (next to `HEXKL_SDK_VER=6.4.0.1`);
  its location differs between workstations and `env.sh`'s default may
  point at another package.
A handoff's rebuild recipe lists these four lines.

## 4. Device (user only)

Never run here. Write a handoff (`hexagon-handoff` skill), set
`state:needs-measurement`, stop. Performance conclusions are drawn only
from filled handoff tables read as an A/B inside one sitting.

## Kernel review list (apply to any `test/htp/*.c`, `htp_backend/**` change)

* Everything before a quantizer is `_det` (one spec, scalar/NEON/HVX,
  bit-compare gate); the CPU reference path uses the same `_det`.
* Weight DMA is hidden behind compute in every kernel (contract, doc 45
  §3.2), and the prefill path keeps its double buffer.
* int32 accumulation order is documented; the HVX GEMV equals the HMX
  accumulator bit for bit.
* The `NNTR_HTP_PROFILE` stage timers stay paired (a stage without an end
  timer is a bug the report tool cannot see) and `hmx_busy_us` /
  `hvx_busy_us` are kept when stages overlap.
* No new DSP heap allocation without a note on the 32-bit address budget
  (3840 MiB arena + ≈ 182 MiB heap, doc 46 §41).
* `QS4CX_WH` has no CPU fallback: a layout change bumps the format tag
  the quantizer writes and the loader checks, in the same PR.
