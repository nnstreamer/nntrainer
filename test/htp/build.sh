#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FastRPC stub/skel from nntr_hvx.idl and builds the DSP skel.
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
#   The SDK must be 6.1.1.0 or newer: HexKL ships libhexkl_micro.a for v79
#   only from lib/6.1.1.0 up. 6.4.0.1 is the verified combination.
# Override the target with: HEX_ARCH=v75 ./build.sh
# Override HexKL with:      HEXKL_ROOT=/path/to/hexkl_addon ./build.sh
# Extra defines with:       HEXKL_EXTRA_CFLAGS=-DNNTR_HEXKL_HW_INIT_2ARG ./build.sh
#   (this HexKL drop's hw_init takes two arguments; see test/htp/hvx_add_f32.c)

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
# Unset HEXKL_ROOT picks the first install that is actually there. The two
# candidates are the beta2 drop the device script (run_u8i4_layer_on_device.sh)
# defaults to and the older unpack-in-Downloads location; a stale default cost a
# bisect step, because this script failing is easy to miss when the caller does
# not stop on it.
if [ -z "${HEXKL_ROOT:-}" ]; then
    HEXKL_ROOT="$HOME/workspace/hxkl-beta2/hexkl_addon"
    if [ ! -d "$HEXKL_ROOT" ] && [ -d "$HOME/Downloads/hexkl_addon" ]; then
        HEXKL_ROOT="$HOME/Downloads/hexkl_addon"
    fi
fi
# Likewise the SDK version: fall back to the newest one this install ships.
if [ -z "${HEXKL_SDK_VER:-}" ]; then
    HEXKL_SDK_VER="$(ls -1 "$HEXKL_ROOT/lib" 2>/dev/null | sort -V | tail -1 || true)"
    HEXKL_SDK_VER="${HEXKL_SDK_VER:-6.4.0.2}"
fi
HEXKL_TOOLS_VARIANT="${HEXKL_TOOLS_VARIANT:-toolv19}"
HEXKL_LIB="$HEXKL_ROOT/lib/$HEXKL_SDK_VER/hexagon_${HEXKL_TOOLS_VARIANT}_${HEX_ARCH}/libhexkl_micro.a"

if [ ! -f "$HEXKL_LIB" ]; then
    echo "Error: HexKL static library not found:" >&2
    echo "  $HEXKL_LIB" >&2
    # Say which of the three parts of that path is actually wrong. The old
    # message here named the version policy no matter what was missing,
    # which points at the wrong thing when HEXKL_ROOT is simply not where
    # this script guesses -- the common case, since the default below is one
    # developer's download directory and run_u8i4_layer_on_device.sh
    # overrides it while a direct ./build.sh does not.
    if [ ! -d "$HEXKL_ROOT" ]; then
        echo "HEXKL_ROOT does not exist: $HEXKL_ROOT" >&2
        echo "Pass the real one, e.g." >&2
        echo "  HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon HEXKL_SDK_VER=6.4.0.2 ./build.sh" >&2
    elif [ ! -d "$HEXKL_ROOT/lib/$HEXKL_SDK_VER" ]; then
        echo "HEXKL_SDK_VER=$HEXKL_SDK_VER is not under $HEXKL_ROOT/lib. Available:" >&2
        ls -1 "$HEXKL_ROOT/lib" 2>/dev/null | sed 's/^/  /' >&2
    else
        echo "That version exists but has no ${HEXKL_TOOLS_VARIANT}_${HEX_ARCH} build. Available:" >&2
        ls -1 "$HEXKL_ROOT/lib/$HEXKL_SDK_VER" 2>/dev/null | sed 's/^/  /' >&2
        echo "HexKL provides v79 only for lib/6.1.1.0 and newer." >&2
    fi
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND="$REPO_ROOT/nntrainer/tensor/htp_backend"
# $BACKEND/.. is nntrainer/tensor, where the headers shared with the host live
# -- swiglu_det.h is the DSP's SwiGLU specification AND what the ARM side and
# the offline quantizer compile against, so it cannot sit behind the HTP-only
# include path.
cd "$SCRIPT_DIR"

mkdir -p generated build

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o generated nntr_hvx.idl

SRCS="hvx_add_f32.c nntr_hvx_mm_u8i4.c nntr_hvx_mm_u8i8.c nntr_hvx_softmax.c nntr_hvx_attn.c nntr_hvx_dma_probe.c nntr_hvx_graph.c nntr_hvx_small_ops.c nntr_hvx_attn_m1.c generated/nntr_hvx_skel.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i4.c $BACKEND/hmx/hexkl_mm_u8i4_dma.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i4_moe.c $BACKEND/hmx/hexkl_graph.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_conv_block.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_mm_u8i8_dma.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_dma_ring.c $BACKEND/hmx/hexkl_dma_trace.c $BACKEND/hmx/hexkl_kv_quant.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_probe.c $BACKEND/hmx/hexkl_acc_tile.c"
SRCS="$SRCS $BACKEND/hmx/hexkl_attn_dtype.c $BACKEND/hmx/hexkl_attn_u8.c"
SRCS="$SRCS $BACKEND/hvx/hvx_quant_u8.c $BACKEND/hvx/hvx_dequant_i32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_swiglu_f32.c $BACKEND/hvx/hvx_conv_gate_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_m1_ops_f32.c $BACKEND/hvx/hvx_attn_m1_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_scale_add_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_gather_ah_u8.c"
SRCS="$SRCS $BACKEND/hvx/hvx_softmax_f32.c $BACKEND/hvx/hvx_softmax_blocked_f32.c"
SRCS="$SRCS $BACKEND/hvx/hvx_worker_pool.c $BACKEND/hvx/hvx_gemm_u8i4_wh.c"
SRCS="$SRCS $BACKEND/hvx/hvx_expand_i2i4.c"
SRCS="$SRCS $SCRIPT_DIR/nntr_stack_chk.c"

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -G0 -O3 -fPIC -shared \
    -Wall -Werror \
    ${HEX_EXTRA_CFLAGS:-} \
    -I generated \
    -I "$HEXKL_ROOT/include" \
    -I "$BACKEND/.." \
    -I "$BACKEND" \
    -I "$BACKEND/hvx" \
    -I "$BACKEND/hmx" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
    ${HEXKL_EXTRA_CFLAGS:-} \
    $SRCS \
    "$HEXKL_LIB" \
    -o build/libnntr_hvx_skel.so

READELF="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-readelf"
# Runtime imports the DSP image provides: FastRPC HAP_*, compute_resource_*,
# QuRT, compiler-rt/CRT (__hexagon_*, __extendhfsf2, __cxa_finalize,
# __register_frame_info_bases) and libc. Anything else -- in particular a
# hexkl_*/hvx_*/nntr_* function -- is a project file missing from SRCS; the
# linker accepts it, the on-device loader does not (#97: 0x80000406).
UND=$("$READELF" --dyn-syms build/libnntr_hvx_skel.so | awk '$7=="UND" && $8!="" {print $8}')
# A real skel always imports HAP_*/qurt_*: an empty list means readelf is
# missing or its column layout changed, and the check below would pass
# vacuously.
if [ -z "$UND" ]; then
    echo "Error: $READELF returned no undefined symbols; guard cannot run" >&2
    exit 1
fi
BAD=$(echo "$UND" | grep -Ev '^(HAP_|compute_resource_|qurt_|__hexagon_|__extendhfsf2$|__cxa_finalize$|__register_frame_info_bases$|malloc$|free$|calloc$|memalign$|memcpy$|memset$|lroundf$|nearbyintf$|snprintf$|vsnprintf$|strlcpy$)' || true)
if [ -n "$BAD" ]; then
    echo "Error: skel has undefined symbols the DSP image will not provide:" >&2
    echo "$BAD" | sed 's/^/  /' >&2
    echo "Add the defining .c to SRCS in $0 (see #97)." >&2
    exit 1
fi
echo "UNDEFINED SYMBOLS OK ($(echo "$UND" | wc -l) runtime imports)"

echo "built: $SCRIPT_DIR/build/libnntr_hvx_skel.so ($HEX_ARCH, hexkl $HEXKL_SDK_VER)"
echo "NOTE: this is the DSP skel only. If nntr_hvx.idl changed, the ARM client"
echo "      stub needs regenerating too, or the build fails on the new symbols"
echo "      (or worse, an old client meets this skel and the call returns"
echo "      EBADPARM):"
echo "        bash $REPO_ROOT/nntrainer/tensor/htp_backend/generate_stub.sh"
