#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Queue packet/cache validation on the host; no DSP device or Hexagon compiler.
# Needs the SDK's dspqueue.h. Arithmetic and DMA are tested on the device.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$HERE/../../../nntrainer/tensor/htp_backend" && pwd)"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT for dspqueue.h}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
python3 "$HERE/gen_nntr_hvx_h.py" "$HERE/../nntr_hvx.idl" "$OUT/nntr_hvx.h"
# Two SDK error names absent from the host stub, in unreachable queue code.
"${CC:-gcc}" -std=gnu11 -O1 -g -Wall -Wextra -Wno-unused-parameter \
  -Wno-sign-compare -ffunction-sections -fdata-sections \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DAEE_EWOULDBLOCK=516 -DAEE_EQURTTHREADCREATE=331 \
  -I "$OUT" -I "$HERE/stub" -I "$HERE/standin" -I "$HERE/.." \
  -I "$BACKEND/.." -I "$BACKEND" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/dspq_conv_cache_host_check.c" -Wl,--gc-sections -pthread \
  -o "$OUT/check"
"$OUT/check"
