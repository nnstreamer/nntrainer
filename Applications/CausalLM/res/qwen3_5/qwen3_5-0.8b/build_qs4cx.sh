#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Build the Qwen3.5-0.8B QS4CX-FP16 model pack
# (embedding Q6_K, FC QS4CX, lm_head Q6_K, FP16 activations).
#
# Pipeline:
#   1. weight_converter_layergraph.py : HF Qwen3.5-0.8B -> FP32 weights / FP16 norms .bin
#   2. nntr_quantize                  : --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype Q6_K
#                                      => model_tensor_type QS4CX-FP16
#
# With SKIP_QUANTIZE=1 the quantize step is skipped and the FP32 pack is
# left in <out_dir> (for the x86 FP32 reference check against HF).
#
# Usage:
#   build_qs4cx.sh <hf_qwen3_5_0.8b_dir> <out_dir> [nntr_quantize]
set -euo pipefail

HF="${1:?usage: build_qs4cx.sh <hf_qwen3_5_0.8b_dir> <out_dir> [nntr_quantize]}"
OUT="${2:?usage: build_qs4cx.sh <hf_qwen3_5_0.8b_dir> <out_dir> [nntr_quantize]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"

NNTR_QUANTIZE="${3:-}"
if [ -z "$NNTR_QUANTIZE" ] && [ "${SKIP_QUANTIZE:-0}" != "1" ]; then
  NNTR_QUANTIZE="$(ls "$REPO"/build*/Applications/CausalLM/nntr_quantize 2>/dev/null | head -1 || true)"
fi
if [ "${SKIP_QUANTIZE:-0}" != "1" ]; then
  [ -x "$NNTR_QUANTIZE" ] || { echo "nntr_quantize not found; pass it as arg 3" >&2; exit 1; }
fi

STAGE="$(mktemp -d)/stage"
mkdir -p "$STAGE" "$OUT"

# 1) HF -> FP32 weights + FP16 norms layer-graph .bin (quantize source)
python3 "$HERE/weight_converter_layergraph.py" \
  --model_path "$HF" \
  --output_name "$STAGE/nntr_qwen3_5_0.8b_mixed.bin" \
  --data_type float32 --norm_fp16

# 2) assemble the pack dir. config.json stays as-is: the multimodal wrapper's
# text_config is flattened at load time by Qwen35CausalLM::sanitizeConfig.
cp "$HF/config.json" "$STAGE/"
cp "$HF"/tokenizer*.json "$STAGE/" 2>/dev/null || true
cp "$HF"/merges.txt "$STAGE/" 2>/dev/null || true
cp "$HF"/vocab.json "$STAGE/" 2>/dev/null || true
cp "$HF/special_tokens_map.json" "$STAGE/" 2>/dev/null || true
# nntr_quantize reads generation_config.json unconditionally; a HF dir that
# ships without a usable one gets a minimal greedy config (eos from the
# text config so decode can stop early).
if [ ! -s "$STAGE/generation_config.json" ]; then
  EOS_ID="$(python3 -c "
import json
c = json.load(open('$STAGE/config.json'))
t = c.get('text_config', c)
e = t.get('eos_token_id', c.get('eos_token_id', 0))
print(e[0] if isinstance(e, list) else e)
")"
  printf '{ "eos_token_id": %s, "bos_token_id": null, "do_sample": false }\n' \
    "$EOS_ID" > "$STAGE/generation_config.json"
fi

if [ "${SKIP_QUANTIZE:-0}" = "1" ]; then
  mv "$STAGE/nntr_qwen3_5_0.8b_mixed.bin" "$OUT/"
  # The tokenizer / config files were staged above; the FP32 pack needs them
  # in OUT (the quantize branch moves the whole stage via nntr_quantize -o).
  cp "$STAGE"/tokenizer*.json "$OUT/" 2>/dev/null || true
  cp "$STAGE"/merges.txt "$STAGE"/vocab.json "$STAGE"/special_tokens_map.json \
     "$STAGE"/config.json "$OUT/" 2>/dev/null || true
  cat > "$OUT/nntr_config.json" <<EOF
{
  "model_type": "CausalLM",
  "model_tensor_type": "FP32-FP16",
  "model_file_name": "nntr_qwen3_5_0.8b_mixed.bin",
  "fc_layer_dtype": "FP32",
  "embedding_dtype": "FP32",
  "lora_rank": 0, "lora_alpha": 0, "lora_target": [], "bad_word_ids": [],
  "fsu": false, "fsu_lookahead": 2,
  "num_to_generate": 32, "init_seq_len": 128, "max_seq_len": 512, "batch_size": 1,
  "tokenizer_file": "$OUT/tokenizer.json",
  "sample_input": "The capital of France is"
}
EOF
  echo "Qwen3.5-0.8B FP32 pack written to: $OUT"
  exit 0
fi

cat > "$STAGE/nntr_config.json" <<EOF
{
  "model_type": "CausalLM",
  "model_tensor_type": "FP32-FP32",
  "model_file_name": "nntr_qwen3_5_0.8b_mixed.bin",
  "fc_layer_dtype": "FP32",
  "embedding_dtype": "FP32",
  "lora_rank": 0, "lora_alpha": 0, "lora_target": [], "bad_word_ids": [],
  "fsu": false, "fsu_lookahead": 2,
  "num_to_generate": 32, "init_seq_len": 1024, "max_seq_len": 2048, "batch_size": 1,
  "tokenizer_file": "$STAGE/tokenizer.json",
  "sample_input": "The capital of France is"
}
EOF

# 3) quantize: FC QS4CX + embedding Q6_K + lm_head Q6_K. The linear-attn
# b/a projections and the conv / gdelta core weights are absent from the
# quantize dtype map on purpose -> they stay FP32.
"$NNTR_QUANTIZE" "$STAGE" \
  --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype Q6_K \
  -o "$OUT"

rm -rf "$(dirname "$STAGE")"
echo "Qwen3.5-0.8B QS4CX-FP16 (Q6_K embed/lm_head) written to: $OUT"
echo "Update nntr_config.json:tokenizer_file to the deployed tokenizer.json path before running."
