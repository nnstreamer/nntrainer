#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Validate expert-update packets and failure ordering under ASan/UBSan.
# Needs the SDK's dspqueue.h; actual DMA is covered by expert_update_check.cpp.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$HERE/../../../nntrainer/tensor/htp_backend" && pwd)"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT for dspqueue.h}"
CHECK_OUT="$(mktemp -d)"
trap 'rm -rf "$CHECK_OUT"' EXIT
python3 "$HERE/gen_nntr_hvx_h.py" "$HERE/../nntr_hvx.idl" "$CHECK_OUT/nntr_hvx.h"
"${CC:-gcc}" -std=gnu11 -O1 -g -Wall -Wextra -Wno-unused-parameter \
  -Wno-sign-compare -ffunction-sections -fdata-sections \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DAEE_EWOULDBLOCK=516 -DAEE_EQURTTHREADCREATE=331 \
  -I "$CHECK_OUT" -I "$HERE/stub" -I "$HERE/standin" -I "$HERE/.." \
  -I "$BACKEND/.." -I "$BACKEND" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/dspq_expert_update_host_check.c" -Wl,--gc-sections -pthread \
  -o "$CHECK_OUT/check"
"$CHECK_OUT/check"
