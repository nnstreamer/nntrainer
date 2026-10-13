#!/system/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Gemma 4 26B-A4B QS2CX_WH/QS4CX profile for 2048 input + 128 output tokens.
set -eu

fast=0
if [ "${1:-}" = "--decode-fast" ]; then
  fast=1
  shift
fi
if [ "$#" -ne 4 ] || [ "$1" != "--page-cache" ]; then
  echo "Usage: sh $0 [--decode-fast] --page-cache off|on EXECUTABLE MODEL_DIR" >&2
  exit 2
fi
case "$2" in
  off) NNTR_HTP_PAGE_CACHE=0 ;;
  on) NNTR_HTP_PAGE_CACHE=1 ;;
  *) echo "--page-cache must be off or on" >&2; exit 2 ;;
esac
shift 2

if [ "${NNTR_HTP_KEEP_ARM_WEIGHTS+x}" = x ]; then
  echo "Unset NNTR_HTP_KEEP_ARM_WEIGHTS to release duplicate host weights" >&2
  exit 2
fi

export NNTR_HTP_PAGE_CACHE
export NNTR_HTP_E2E=1
export NNTR_MOE_CACHE_EXPERTS=12
export NNTR_HTP_FC_IN_ARENA=1
export NNTR_HTP_E2E_FREE_KVQ=1
export NNTR_HTP_KV_DROP_CPU=1
export NNTR_HTP_ATTN_M1_Q8=2
export NNTR_NUM_THREADS="${NNTR_NUM_THREADS:-8}"
if [ "$fast" = 1 ]; then
  # Measured on S26 Ultra: C12 for prefill, then C34 for decode. Hints only
  # schedule reads; the DSP still computes every actual router decision.
  export NNTR_HTP_RELEASE_PREFILL_SCRATCH=1
  export NNTR_MOE_CACHE_GROW=22
  export NNTR_HTP_PREDICT=2
  export NNTR_MOE_PREFETCH_DIRECT=1
  export NNTR_MOE_PREFETCH=4
  export NNTR_MOE_PREFETCH_READERS=4
  export NNTR_MOE_PREFETCH_TOP=8
  export NNTR_MOE_PREFETCH_SLOTS=32
  export NNTR_MOE_PREFETCH_RESIDUAL=0.5
  export NNTR_MOE_DMA_QUEUES=4
fi
exec "$@"
