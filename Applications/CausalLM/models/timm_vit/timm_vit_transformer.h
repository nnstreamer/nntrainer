// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.h
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief   This timm_vit_transformer.h constructs a class for timm ViT model
 * compatible with the PyTorch timm library.
 */

#ifndef __TIMM_VIT_TRANSFORMER_H__
#define __TIMM_VIT_TRANSFORMER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief TimmViTTransformer class
 */
class TimmViTTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "TimmViT";

  /**
   * @brief Construct a TimmViTTransformer object.
   */
  TimmViTTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the TimmViTTransformer object.
   */
  virtual ~TimmViTTransformer() = default;

public:
  /**
   * @brief Create patch embedding layers for an input image tensor.
   */
  Tensor createPatchEmbed(Tensor input);

  // The 2-arg ViT variants below take a single input tensor, while the base
  // Transformer::createAttention / createMlp expose the (seq_len, n_heads,
  // ..., ...) causal-LM signatures. Bring the base overloads into scope so
  // the ViT-specific 2-arg versions overload them instead of hiding them
  // (which -Werror=overloaded-virtual rejects).
  using Transformer::createAttention;
  using Transformer::createMlp;

  /**
   * @brief Create a ViT self-attention block for a transformer layer.
   */
  Tensor createAttention(const int layer_id, Tensor input);

  /**
   * @brief Create a ViT MLP block for a transformer layer.
   */
  Tensor createMlp(const int layer_id, Tensor input);

protected:
  /**
   * @brief Construct the ViT graph and return input/output tensors.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Set model parameters from HuggingFace and nntrainer configs.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create a ViT transformer encoder block.
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Register custom layers required by the base transformer.
   */
  void registerCustomLayers() override;

  /**
   * @brief Run the model (override for ViT specific behavior)
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  unsigned int IMG_SIZE = 224;    /**< Image height/width */
  unsigned int PATCH_SIZE = 16;   /**< Patch height/width */
  unsigned int NUM_PATCHES = 196; /**< Number of patches */
  unsigned int IMG_CHANNELS = 3;  /**< Image channels (RGB) */
};

} // namespace causallm

#endif /* __TIMM_VIT_TRANSFORMER_H__ */
