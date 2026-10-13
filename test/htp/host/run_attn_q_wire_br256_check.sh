#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Host-only VTCM bounds and row/mask geometry checks; no SDK or device.
set -eu

ATTN_Q_WIDE_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTN_Q_WIDE_HMX="$(cd "$ATTN_Q_WIDE_HERE/../../../nntrainer/tensor/htp_backend/hmx" && pwd)"
ATTN_Q_WIDE_BUILD="$(mktemp -d)"
trap 'rm -rf "$ATTN_Q_WIDE_BUILD"' EXIT
ATTN_Q_WIDE_CC="${CC:-cc}"

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --sanitize ]; }; then
  echo "usage: $0 [--sanitize]" >&2
  exit 1
fi
ATTN_Q_WIDE_FLAGS=(-O2)
if [ "${1:-}" = --sanitize ]; then
  ATTN_Q_WIDE_FLAGS=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"$ATTN_Q_WIDE_CC" -std=c11 -Wall -Wextra -Werror \
  "${ATTN_Q_WIDE_FLAGS[@]}" -I "$ATTN_Q_WIDE_HMX" \
  "$ATTN_Q_WIDE_HERE/attn_q_wire_br256_host_check.c" \
  -o "$ATTN_Q_WIDE_BUILD/attn_q_wire_br256_host_check"
"$ATTN_Q_WIDE_BUILD/attn_q_wire_br256_host_check"
