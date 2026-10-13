#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the isolated QS2 layout/GEMV check, after test/htp/build.sh.
set -eu
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
: "${ANDROID_NDK:?set ANDROID_NDK}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$HERE/../../nntrainer/tensor/htp_backend"
CLANG="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"
"$CLANG" -std=gnu11 -O2 -Wall -Wextra -DNNTR_LUT_DEVICE \
  -I "$HERE/generated" -I "$BACKEND/hvx" \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/host/qs2_lut_check.c" "$BACKEND/hvx/hvx_gemv_u8i2_lut.c" \
  "$HERE/generated/nntr_hvx_stub.c" \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
  -lcdsprpc -ldl -llog -Wl,-z,max-page-size=16384 \
  -o "$HERE/build/qs2_lut_device_check"
