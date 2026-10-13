#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the device checks after test/htp/build.sh generates the stub.
set -eu
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
: "${ANDROID_NDK:?set ANDROID_NDK}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$HERE/../../nntrainer/tensor/htp_backend"
CLANG="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"
for name in tl2_lut_check vlut_device_probe; do
  "$CLANG" -std=gnu11 -O2 -Wall -Wextra -Werror -DNNTR_TL2_DEVICE \
    -I "$HERE/generated" -I "$BACKEND/hvx" \
    -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    "$HERE/host/$name.c" "$BACKEND/hvx/hvx_gemv_u8tl2.c" \
    "$HERE/generated/nntr_hvx_stub.c" \
    -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" \
    -lcdsprpc -ldl -llog -Wl,-z,max-page-size=16384 \
    -o "$HERE/build/${name}_device"
done
