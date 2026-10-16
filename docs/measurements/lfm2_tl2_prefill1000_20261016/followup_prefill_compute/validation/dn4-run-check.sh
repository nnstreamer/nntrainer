#!/bin/bash
set -eu
ROOT="${1:?pass repository root}"
OUT="${2:?pass output directory}"
BACKEND="$ROOT/nntrainer/tensor/htp_backend"
HERE="$ROOT/test/htp/host"
mkdir -p "$OUT"
cc=${CC:-gcc}
"$cc" -std=gnu11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter \
  -Wno-format-truncation -ffp-contract=off -ffunction-sections -fdata-sections \
  -include malloc.h -I "$HERE/hvx_emu" -I "$HERE/stub" \
  -I "$BACKEND/.." -I "$BACKEND" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  "$HERE/tl2_dn_dequant_check.c" "$BACKEND/hvx/hvx_dequant_i32.c" \
  "$BACKEND/hvx/hvx_gemv_u8tl2.c" -Wl,--gc-sections -lm \
  -o "$OUT/tl2_dn_dequant_check"
"$OUT/tl2_dn_dequant_check"
