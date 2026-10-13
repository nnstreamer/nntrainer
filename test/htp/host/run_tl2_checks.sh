#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs the HTP kernels' host checks: no device, no Hexagon SDK, no HMX.
#
# What these are and are not. The HMX and HVX primitives are replaced by
# scalar stand-ins in stub/, standin/ and the check itself, so this does NOT
# verify the hardware's arithmetic -- the device tests do that, and this
# would give false confidence if it were read as doing so. What it does
# verify is everything around the arithmetic: which rows each expert gets,
# which weights it uses, that the buffer reuse in the weight DMA pipeline
# does not clobber a live buffer, that the scatter lands on the right
# output row with the right routing weight, that empty and
# multi-block experts work, and that the VTCM layout matches the hand
# arithmetic in docs/htp_attention/46_moe_resident_kernel_design.md.
#
# The DMA stand-in completes immediately, which is the point: a transfer
# issued into a buffer that is still being read shows up as a wrong result
# here rather than as a rare corruption on device.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$HERE/../../../nntrainer/tensor/htp_backend" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cc=${CC:-gcc}
"$cc" -std=gnu11 -O2 -Wall -Wextra -I "$BACKEND/hvx" \
  "$HERE/tl2_lut_check.c" "$BACKEND/hvx/hvx_gemv_u8tl2.c" \
  -o "$OUT/tl2_lut_check"
"$OUT/tl2_lut_check"

g++ -std=c++17 -O2 -Wall -Wextra -I "$BACKEND/.." -I "$BACKEND/hvx" \
  "$HERE/tl2_native_packing_check.cpp" "$BACKEND/hvx/hvx_gemv_u8tl2.c" \
  -o "$OUT/tl2_native_packing_check"
"$OUT/tl2_native_packing_check"

g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$BACKEND" -I "$BACKEND/.." \
  "$HERE/../../../tools/prepare_tl2.cpp" -o "$OUT/prepare_tl2"
g++ -std=c++17 -O2 -Wall -Wextra -pthread -I "$BACKEND" -I "$BACKEND/hvx" \
  "$HERE/tl2_prepared_check.cpp" "$BACKEND/hvx/hvx_gemv_u8tl2.c" \
  -o "$OUT/tl2_prepared_check"
"$OUT/tl2_prepared_check" --fixture "$OUT/tl2-source.bin"
echo '0 256 256' > "$OUT/tl2-weights.tsv"
"$OUT/prepare_tl2" "$OUT/tl2-source.bin" "$OUT/tl2-weights.tsv" "$OUT/tl2-prepared.bin"
"$OUT/tl2_prepared_check" "$OUT/tl2-source.bin" "$OUT/tl2-prepared.bin"

"$cc" -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -DMOE_TAIL_MAX_ROWS=16u \
  -I "$HERE/stub" -I "$HERE/standin" -I "$HERE/.." -I "$BACKEND/.." \
  -I "$BACKEND/hmx" -I "$BACKEND/hvx" -o "$OUT/tl2_moe_host_check" \
  "$HERE/tl2_moe_host_check.c" "$HERE/standin/hvx_scalar.c" \
  "$BACKEND/hmx/hexkl_mm_u8i4_moe.c" "$BACKEND/hmx/hexkl_dma_trace.c" \
  "$BACKEND/hvx/hvx_expand_i2i4.c" "$BACKEND/hvx/hvx_gemv_u8tl2.c" -lm
"$OUT/tl2_moe_host_check"
