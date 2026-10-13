#!/system/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Run on Android, beside the matching nntrainer libraries and DSP skel:
#   sh run_lfm2_strict.sh /absolute/path/to/model [prompt]
# Requires QS4CX_WH FCs, QS2CX_WH/QS4CX_WH experts and a tied Q4_0 head.
# Model arithmetic except LM head runs on HTP; unsupported paths fail.
# NNTR_HTP_LMHEAD_CPU=0 restores the all-HTP head for comparisons.
# NNTR_HTP_ADD_NORM_FUSION=0 disables residual/FFN-norm fusion.
# NNTR_HTP_BLOCK_NORM_FUSION=0 disables final residual/next-block norm fusion.
# NNTR_HTP_NORM_CONSUMER_FUSION=0 disables decode norm+conv/router/QKV fusion.
# NNTR_HTP_QKV_ROPE_FUSION=0 restores three separate RoPE/KV conversion calls.
# NNTR_HTP_RETAIN_ROUTER_INPUT=0 disables DSP router-to-MoE buffer reuse.
# Shared activations avoid large prefill copies; decode uses small RPC buffers.
# Measured LFM2 32/8x64 attention remains faster on HMX at context 1024.
# NNTR_HTP_ATTN_DECODE_HMX=0 selects the optimized pure-HVX comparison.
# Decode math/router use the DSP queue; set NNTR_HTP_DSPQ_MATH=0 or
# NNTR_HTP_DSPQ_ROUTER=0 to compare the original FastRPC transport.
# A short spin keeps small adjacent math calls responsive without occupying
# a DSP execution slot through the following HMX call.
# QS2 experts stay compressed in DRAM and expand in VTCM on the DSP.
# NNTR_HTP_QS2_VTCM=0 selects ARM expansion for diagnostic comparisons.
# Embedding, sampling, host scheduling and expert file I/O remain on the host. This switch does not claim zero CPU usage.
# NNTR_HTP_STRICT_FUSION=0 disables router/QKV fusion for A/B comparisons.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: sh run_lfm2_strict.sh MODEL_DIR [PROMPT]" >&2
  exit 2
fi
export NNTR_HTP_STRICT=1
export NNTR_HTP_LMHEAD_CPU="${NNTR_HTP_LMHEAD_CPU:-1}"
export NNTR_HTP_SHARED_ACTIVATIONS="${NNTR_HTP_SHARED_ACTIVATIONS:-1}"
export NNTR_HTP_ATTN_DECODE_HMX="${NNTR_HTP_ATTN_DECODE_HMX:-1}"
export NNTR_HTP_DSPQ_SPIN_US="${NNTR_HTP_DSPQ_SPIN_US:-20}"
export NNTR_MOE_CACHE_EXPERTS="${NNTR_MOE_CACHE_EXPERTS:-8}"
export NNTR_MOE_PREFETCH_EARLY="${NNTR_MOE_PREFETCH_EARLY:-1}"
export NNTR_MOE_PREFETCH_READERS="${NNTR_MOE_PREFETCH_READERS:-6}"
export NNTR_NUM_THREADS="${NNTR_NUM_THREADS:-8}"
export LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH=".;${ADSP_LIBRARY_PATH:-}"
exec ./nntrainer_causallm "$@"
