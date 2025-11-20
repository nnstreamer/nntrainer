//
// Created by donghak on 25. 11. 13..
//

#ifndef __ERNIE_CAUSALLM_H__
#define __ERNIE_CAUSALLM_H__
#include <causal_lm.h>

namespace causallm {

class Ernie4_5_MoeForCausalLM : public CausalLM {
public:
  static constexpr const char *architecture = "Ernie4_5_MoeForCausalLM";
  Ernie4_5_MoeForCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Ernie4_5_MoeForCausalLM() = default;

  /**
   * @brief MoE layer
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  /**
   * @brief setupParameters
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief registerCutomLayers
   */
  void registerCustomLayers() override;

private:
  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
  unsigned int NUM_SHARED_EXPERTS;
  unsigned int MOE_INTERMEDIATE_SIZE;

  std::vector<std::string> LAYER_TYPES;
  float ATTENTION_ROPE_SCALING_FACTOR;

};

} // namespace causallm

#endif // __ERNIE_CAUSALLM_H__
