// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_causallm.cpp
 * @date   06 July 2026
 * @brief  This defines the LFM2-8B-A1B Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <lfm2_moe_causallm.h>
#include <lfm2_moe_layer.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_graph_desc.h>
#endif

namespace causallm {

void Lfm2MoeCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  // Parse the LFM2 backbone parameters (dims, layer_types, conv, ...).
  Lfm2CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  // MoE-specific parameters.
  try {
    NUM_EXPERTS = cfg["num_experts"];
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"];
  } catch (const std::exception &e) {
    throw std::runtime_error(
      "Lfm2Moe: num_experts, num_experts_per_tok and moe_intermediate_size "
      "must be specified in the config file");
  }
  // Layers [0, num_dense_layers) keep the dense SwiGLU FFN. Optional (default 0).
  NUM_DENSE_LAYERS = cfg.value("num_dense_layers", 0);

  // MoE expert FFN weight dtype. Defaults to FC_LAYER_DTYPE (set by
  // Lfm2CausalLM::setupParameters above) so an unquantized or fc_dtype==
  // moe_dtype model is unaffected; nntr_quantize's --moe_dtype writes
  // moe_layer_dtype when the two are meant to differ (see quantize.cpp's
  // buildLayerDtypeMap docstring).
  MOE_LAYER_DTYPE = nntr_cfg.value("moe_layer_dtype", FC_LAYER_DTYPE);

  // MoE FFN engine. Defaults to "cpu" so an existing run is unaffected
  // unless nntr_config.json opts in; MOE_HTP_LAYERS (empty = every MoE
  // layer) bounds a real checkpoint's device run to what the HTP weight
  // registry can hold at once -- see the two fields' docs in the header.
  MOE_ENGINE = nntr_cfg.value("moe_engine", std::string("cpu"));
  MOE_HTP_LAYERS =
    parseLayerIdList(nntr_cfg.value("moe_htp_layers", std::string("")));

#ifdef ENABLE_HEXKL
  // [#85] NNTR_HTP_FORWARD=1: describe this model's decode step to the HTP
  // backend once, so its MoE calls at M == 1 can go through the per-token
  // entry. The words are the header's LFM2 builder over this config; the
  // backend validates them with the same validator the skel runs. Off by
  // default, and only meaningful with the MoE FFN on the HTP.
  const char *fwd = std::getenv("NNTR_HTP_FORWARD");
  if (fwd != nullptr && std::atoi(fwd) != 0 && MOE_ENGINE == "htp") {
    htp_graph_lfm2_shape shape;
    shape.n_layers = static_cast<uint32_t>(NUM_LAYERS);
    shape.n_dense_layers = NUM_DENSE_LAYERS;
    shape.hidden = static_cast<uint32_t>(DIM);
    shape.inter_dense = static_cast<uint32_t>(INTERMEDIATE_SIZE);
    shape.inter_moe = MOE_INTERMEDIATE_SIZE;
    shape.n_experts = NUM_EXPERTS;
    shape.top_k = NUM_EXPERTS_PER_TOK;
    shape.n_heads = static_cast<uint32_t>(NUM_HEADS);
    shape.n_kv_heads = static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
    shape.head_dim = static_cast<uint32_t>(HEAD_DIM);
    shape.vocab = NUM_VOCAB;
    shape.max_seq = MAX_SEQ_LEN;
    shape.eps = NORM_EPS;
    std::vector<uint8_t> attn(layer_types_.size());
    for (size_t l = 0; l < layer_types_.size(); ++l)
      attn[l] = layer_types_[l] != "conv";
    // [#130] NNTR_HTP_FORWARD_KINDS (comma-separated kind names) chooses
    // which kinds carry the resident bit; the default is every kind the
    // skel and the ARM hooks run. The bit is the one source of truth for
    // both sides (plan 130 section 3.2).
    const char *kinds = std::getenv("NNTR_HTP_FORWARD_KINDS");
    const uint32_t mask =
      kinds != nullptr
        ? htp_graph_kinds_parse(kinds)
        : (HTP_GRAPH_KIND_BIT(HTP_OP_MOE) | HTP_GRAPH_KIND_BIT(HTP_OP_RMSNORM) |
           HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) |
           HTP_GRAPH_KIND_BIT(HTP_OP_ROPE) |
           HTP_GRAPH_KIND_BIT(HTP_OP_CONV1D_GATE) |
           HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1));
    if (mask == 0u)
      throw std::runtime_error(
        std::string("Lfm2Moe: NNTR_HTP_FORWARD_KINDS: unknown kind in '") +
        kinds + "'");
    std::vector<uint32_t> words(
      htp_graph_words_for(shape.n_layers, HTP_GRAPH_MAX_OPS));
    const uint32_t n =
      htp_graph_lfm2_build(words.data(), static_cast<uint32_t>(words.size()),
                           &shape, attn.data(), mask);
    if (n == 0u)
      throw std::runtime_error("Lfm2Moe: NNTR_HTP_FORWARD: the decode op list "
                               "does not fit HTP_GRAPH_MAX_OPS");
    words.resize(n);
    if (!nntrainer::get_htp_ops()->set_decode_graph_desc(words))
      throw std::runtime_error(
        "Lfm2Moe: NNTR_HTP_FORWARD: the HTP backend has no per-token entry");
  }
#endif
}

Tensor Lfm2MoeCausalLM::createMoeLayer(const int layer_id, Tensor input) {
  const std::string engine =
    (MOE_HTP_LAYERS.empty() || MOE_HTP_LAYERS.count(layer_id)) ? MOE_ENGINE
                                                               : "cpu";
  LayerHandle moe(createLayer(
    "lfm2_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", MOE_INTERMEDIATE_SIZE),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish"),
     withKey("weight_dtype", MOE_LAYER_DTYPE), withKey("engine", engine)}));
  return moe(input);
}

Tensor Lfm2MoeCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {
  // Dense SwiGLU FFN for the first NUM_DENSE_LAYERS layers.
  if (layer_id < static_cast<int>(NUM_DENSE_LAYERS))
    return Transformer::createMlp(layer_id, dim, hidden_dim, input);

  // MoE FFN for the remaining layers.
  return createMoeLayer(layer_id, input);
}

void Lfm2MoeCausalLM::registerCustomLayers() {

  Lfm2CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::Lfm2MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register Lfm2MoELayer factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
