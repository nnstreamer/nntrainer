// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen35_causallm.h
 * @brief  Qwen3.5 causal language model implementation (hybrid linear
 *         attention / full attention).
 * @date   2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @note   Reference: transformers modeling_qwen3_5.py (Qwen3_5ForConditional
 *         Generation text stack).
 */

#ifndef __QWEN35_CAUSALLM_H__
#define __QWEN35_CAUSALLM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Qwen35Transformer - gated full attention + gated-delta linear
 *        attention blocks per config layer_types.
 */
class Qwen35Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3_5TextForConditionalGeneration";

  Qwen35Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen35Transformer() = default;

  /** Full-attention sub-graph: q+gate split, partial rope, gated output. */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  /** Dispatch: linear_attention -> createLinearAttnBlock, else the base
   *  attention+ffn block. */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /** Linear-attention (gated delta net) block. */
  Tensor createLinearAttnBlock(const int layer_id, Tensor input);

  void registerCustomLayers() override;

  /** @copydoc Transformer::getLayerSlidingWindow(int) — one uniform window
   *  for every layer (no sliding interleave), so the base pattern rule does
   *  not apply. */
  unsigned int getLayerSlidingWindow(int layer_id) const override {
    (void)layer_id;
    return SLIDING_WINDOW;
  }

  /** @copydoc Transformer::getModelFeatures() — qwen3.5: q/k-norm, SwiGLU,
   *  pre-norm, head_dim=256, decode RoPE on the device, tied lm_head. */
  nntrainer::ModelFeatures getModelFeatures() const override {
    nntrainer::ModelFeatures f;
    f.has_qk_norm = true;
    f.mlp_kind = nntrainer::MlpKind::SWIGLU;
    f.norm_style = nntrainer::NormStyle::PRE;
    f.lmhead_kind = nntrainer::LmHeadKind::TIED;
    f.decode_gpu = true;
    f.decode_rope_gpu = true;
    f.head_dim = 256;
    return f;
  }

  /** linear-attention block geometry (text_config; 0.8B: 16 x 128, conv 4) */
  int LINEAR_NUM_HEADS = 16;
  int LINEAR_HEAD_DIM = 128;
  int CONV_KERNEL_SIZE = 4;
  float PARTIAL_ROPE_FACTOR = 0.25f;

  /** per-layer type, "linear_attention" | "full_attention" */
  std::vector<std::string> layer_types_;
};

/**
 * @brief Qwen35CausalLM class (architectures key "Qwen3_5ForConditionalGeneration")
 */
class Qwen35CausalLM : public CausalLM, public Qwen35Transformer {

public:
  static constexpr const char *architectures = "Qwen3_5ForConditionalGeneration";

  Qwen35CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  virtual ~Qwen35CausalLM() = default;

  /**
   * @brief Flatten the multimodal wrapper's text_config into the top-level
   *        config before any base-class parsing runs.
   */
  static json &sanitizeConfig(json &cfg);

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __QWEN35_CAUSALLM_H__ */
