// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen3_causallm.h
 * @date   10 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen3/modeling_qwen3.py
 */

#ifndef __QWEN_CAUSAL_LM_H__
#define __QWEN_CAUSAL_LM_H__ __QWEN_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

class Qwen3CausalLM : public CausalLM {

public:
  static constexpr const char *architectures = "Qwen3ForCausalLM";

  Qwen3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg) {}

  ~Qwen3CausalLM() = default;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  void registerCustomLayers() override;

private:
};
} // namespace causallm

#endif /* __QWEN3_CAUSAL_LM_H__ */
