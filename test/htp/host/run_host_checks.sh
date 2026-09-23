#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs the HTP kernels' host checks: no device, no Hexagon SDK, no HMX.
#
# What these are and are not. The HMX and HVX primitives are replaced by
# scalar stand-ins in stub/ and in the check itself, so this does NOT
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
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -DMOE_TAIL_MAX_ROWS=16u \
  -I "$HERE/stub" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/moe_layer_host_check" \
  "$HERE/moe_layer_host_check.c" "$HERE/hvx_scalar_stubs.c" \
  "$BACKEND/hmx/hexkl_mm_u8i4_moe.c" "$BACKEND/hvx/hvx_expand_i2i4.c" -lm

"$OUT/moe_layer_host_check"

# The conv block kernel (doc 51 section 2) on the same stand-ins. It is
# built on the MoE kernel's exported helpers, so that file links in too.
# -ffp-contract=off: the conv gate's reference is a separate multiply and
# add per tap, as the HVX computes it, and the host compiler must not fuse
# them into an FMA the device does not have.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off \
  -I "$HERE/stub" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/conv_block_host_check" \
  "$HERE/conv_block_host_check.c" "$HERE/hvx_scalar_stubs.c" \
  "$BACKEND/hmx/hexkl_conv_block.c" "$BACKEND/hmx/hexkl_mm_u8i4_moe.c" "$BACKEND/hvx/hvx_expand_i2i4.c" -lm

"$OUT/conv_block_host_check"

# The FC / projection call (hexkl_mm_u8i4_layer_run) on the same stand-ins:
# several handles against one activation, the pooled epilogue's batching.
"$cc" -std=c99 -O1 -Wall -Wextra -Wno-unused-parameter \
  -I "$HERE/stub" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
  -o "$OUT/fc_layer_host_check" \
  "$HERE/fc_layer_host_check.c" "$HERE/hvx_scalar_stubs.c" \
  "$BACKEND/hmx/hexkl_mm_u8i4_dma.c" -lm

"$OUT/fc_layer_host_check"

# The worker pool's two lanes on pthreads (stub/qurt.h). Concurrency is
# exercised for real here -- 3 workers, a caller that helps -- but a
# desktop scheduler is not QuRT's; the device is still where the timing
# and the HVX-context sharing are checked.
"$cc" -std=c11 -O1 -Wall -Wextra -Wno-unused-parameter -pthread \
  -I "$HERE/stub" -I "$BACKEND/hvx" \
  -o "$OUT/worker_pool_host_check" \
  "$HERE/worker_pool_host_check.c" "$BACKEND/hvx/hvx_worker_pool.c"

"$OUT/worker_pool_host_check"

# whPack2 (the quantizer, C++) against hvx_expand_i2i4 (the kernel, C).
# Nothing else ties those two together -- different toolchains, no shared
# code -- and if they drift a QS2CX_WH model holds different weights than
# the QS4CX_WH model whose perplexity justified it. C++ because whPack2 is
# a C++ header; the pool is real pthreads, as above.
# The kernel halves stay C: hvx_worker_pool.c is C11 stdatomic, which a
# C++ compiler will not take.
cxx=${CXX:-g++}
for k in hvx_expand_i2i4 hvx_worker_pool; do
  "$cc" -std=c11 -O1 -Wall -Wextra -Wno-unused-parameter -pthread -c \
    -I "$HERE/stub" -I "$BACKEND/hvx" -o "$OUT/$k.o" "$BACKEND/hvx/$k.c"
done
"$cxx" -std=c++17 -O1 -Wall -Wextra -Wno-unused-parameter -pthread \
  -I "$HERE/stub" -I "$BACKEND/hvx" -I "$BACKEND/.." \
  -o "$OUT/expand_i2i4_host_check" \
  "$HERE/expand_i2i4_host_check.cc" "$OUT/hvx_expand_i2i4.o" \
  "$OUT/hvx_worker_pool.o"

"$OUT/expand_i2i4_host_check"
