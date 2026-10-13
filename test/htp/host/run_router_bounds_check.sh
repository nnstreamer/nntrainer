#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Quick exact ranking/weight parity by default; no device or SDK required.
# Full finite binary32 enumeration is explicitly opt-in:
#   CLANG_FORMAT_14=/path/to/clang-format-14 \
#     bash test/htp/host/run_router_bounds_check.sh --regenerate /tmp/bounds
set -eu

ROUTER_BOUNDS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTER_BOUNDS_TENSOR="$(cd "$ROUTER_BOUNDS_HERE/../../../nntrainer/tensor" && pwd)"
ROUTER_BOUNDS_BUILD="$(mktemp -d)"
trap 'rm -rf "$ROUTER_BOUNDS_BUILD"' EXIT
ROUTER_BOUNDS_CC="${CC:-cc}"

case "${1:-}" in
  ""|--sanitize)
    if [ "$#" -gt 1 ]; then
      echo "usage: $0 [--sanitize | --regenerate OUTPUT_DIRECTORY]" >&2
      exit 1
    fi
    ROUTER_BOUNDS_FLAGS=(-O2)
    if [ "${1:-}" = --sanitize ]; then
      ROUTER_BOUNDS_FLAGS=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    fi
    "$ROUTER_BOUNDS_CC" -std=c11 -ffp-contract=off -Wall -Wextra -Werror \
      "${ROUTER_BOUNDS_FLAGS[@]}" \
      -I "$ROUTER_BOUNDS_HERE/.." -I "$ROUTER_BOUNDS_TENSOR" \
      "$ROUTER_BOUNDS_HERE/router_bounds_host_check.c" -lm \
      -o "$ROUTER_BOUNDS_BUILD/router_bounds_host_check"
    "$ROUTER_BOUNDS_BUILD/router_bounds_host_check"
    ;;
  --regenerate)
    if [ "$#" -ne 2 ]; then
      echo "usage: $0 --regenerate OUTPUT_DIRECTORY" >&2
      exit 1
    fi
    ROUTER_BOUNDS_FORMAT="${CLANG_FORMAT_14:-clang-format-14}"
    case "$("$ROUTER_BOUNDS_FORMAT" --version)" in
      *"version 14."*) ;;
      *) echo "CLANG_FORMAT_14 must identify clang-format version 14" >&2; exit 1 ;;
    esac
    mkdir -p "$2"
    for ROUTER_BOUNDS_NAME in router_sigmoid_bounds router_sigmoid_outer_bounds; do
      "$ROUTER_BOUNDS_CC" -std=c11 -O3 -ffp-contract=off -fopenmp \
        -Wall -Wextra -Werror -I "$ROUTER_BOUNDS_TENSOR" \
        "$ROUTER_BOUNDS_HERE/generate_$ROUTER_BOUNDS_NAME.c" -lm \
        -o "$ROUTER_BOUNDS_BUILD/generate_$ROUTER_BOUNDS_NAME"
      "$ROUTER_BOUNDS_BUILD/generate_$ROUTER_BOUNDS_NAME" \
        "$ROUTER_BOUNDS_BUILD/$ROUTER_BOUNDS_NAME.h"
      "$ROUTER_BOUNDS_FORMAT" --style=file \
        --assume-filename="$ROUTER_BOUNDS_HERE/../$ROUTER_BOUNDS_NAME.h" \
        < "$ROUTER_BOUNDS_BUILD/$ROUTER_BOUNDS_NAME.h" > "$2/$ROUTER_BOUNDS_NAME.h"
      cmp "$ROUTER_BOUNDS_HERE/../$ROUTER_BOUNDS_NAME.h" "$2/$ROUTER_BOUNDS_NAME.h"
    done
    echo "ROUTER_BOUND_HEADERS_REPRODUCED inputs=4278190080"
    ;;
  *)
    echo "usage: $0 [--sanitize | --regenerate OUTPUT_DIRECTORY]" >&2
    exit 1
    ;;
esac
