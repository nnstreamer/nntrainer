// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   ced_transformer.h
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This ced_transformer.h constructs a class for the CED audio tagging
 * model (https://huggingface.co/mispeech/ced-tiny).
 */

#ifndef __CED_TRANSFORMER_H__
#define __CED_TRANSFORMER_H__

#include "timm_vit/timm_vit_transformer.h"

namespace causallm {

/**
 * @brief CedTransformer class
 *
 * CED is a ViT over mel-spectrogram patches, so the encoder trunk is exactly
 * the one TimmViTTransformer already builds. This class only supplies the
 * pieces that differ, all of them driven from the upstream HuggingFace config:
 *
 *  - a non-square single-channel input (n_mels x target_length),
 *  - a per-frequency BatchNorm on that input (CED's `init_bn`),
 *  - a mean-pool + LayerNorm + linear + sigmoid classification head,
 *  - a mel-spectrogram runner instead of the image runner.
 *
 * CED's two separable positional embedding tables (time_pos_embed and
 * freq_pos_embed) are pre-summed into the single flattened table the base
 * patch-embed builds; that is exact, since both are constants. The summing is
 * done by res/ced/weight_converter.py.
 *
 * The model input is `input_values`, the mel-dB spectrogram -- the same tensor
 * HuggingFace's CedModel.forward() takes. The STFT / mel / dB front-end sits
 * outside the model in both implementations.
 */
class CedTransformer : virtual public Transformer, public TimmViTTransformer {

public:
  static constexpr const char *architectures = "CedForAudioClassification";

  /**
   * @brief Construct a CedTransformer object.
   */
  CedTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL),
    TimmViTTransformer(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the CedTransformer object.
   */
  virtual ~CedTransformer() = default;

protected:
  /**
   * @brief Map the HuggingFace CED config onto the generic ViT parameters.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Run inference on a mel-spectrogram file.
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  std::vector<std::string> LABELS; /**< id2label from the config, if present */
};

} // namespace causallm

#endif /* __CED_TRANSFORMER_H__ */
