// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	qwen35_causallm.cpp
 * @date	2026
 * @brief	This defines a Qwen3.5 causal language model (hybrid gated delta
 *          linear attention / gated full attention).
 * @see		https://github.com/nnstreamer/
 * @bug		No known bugs except for NYI items
 * @note   Reference: transformers modeling_qwen3_5.py
 */
#include <qwen35_causallm.h>

#include <llm_util.hpp>

#include <app_context.h>
#include <causal_conv1d_layer.h>
#include <custom_multiply.h>
#include <engine.h>
#include <gated_delta_core.h>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Qwen35CausalLM::sanitizeConfig(json &cfg) {
  // Qwen3.5 ships as a multimodal wrapper (Qwen3_5ForConditionalGeneration):
  // flatten text_config to the top level so the common parser sees the text
  // fields. The vision / mtp sections are ignored (text-only deployment).
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    const auto &text_cfg = cfg["text_config"];
    for (auto it = text_cfg.begin(); it != text_cfg.end(); ++it) {
      if (!cfg.contains(it.key()))
        cfg[it.key()] = it.value();
    }
  }

  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = true;

  return cfg;
}

Qwen35CausalLM::Qwen35CausalLM(json &cfg, json &generation_cfg,
                               json &nntr_cfg) :
  Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
              ModelType::CAUSALLM),
  CausalLM(sanitizeConfig(cfg), generation_cfg, nntr_cfg),
  Qwen35Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg) {
  // Re-run the (virtual) parameter setup now that the final type is complete,
  // so the qwen3.5-specific keys below are parsed (mirrors Lfm2CausalLM).
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void Qwen35CausalLM::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  NNTR_THROW_IF(!cfg.contains("layer_types"), std::invalid_argument)
    << "Qwen3.5 requires layer_types in config";
  layer_types_ = cfg["layer_types"].get<std::vector<std::string>>();
  NNTR_THROW_IF(layer_types_.size() != static_cast<size_t>(NUM_LAYERS),
                std::invalid_argument)
    << "layer_types size must match num_hidden_layers ("
    << layer_types_.size() << " vs " << NUM_LAYERS << ")";
  for (const auto &lt : layer_types_) {
    NNTR_THROW_IF(lt != "linear_attention" && lt != "full_attention",
                  std::invalid_argument)
      << "Unsupported Qwen3.5 layer type: " << lt;
  }

  LINEAR_NUM_HEADS = cfg.contains("linear_num_value_heads")
                       ? cfg["linear_num_value_heads"].get<int>()
                       : 16;
  LINEAR_HEAD_DIM = cfg.contains("linear_key_head_dim")
                      ? cfg["linear_key_head_dim"].get<int>()
                      : 128;
  CONV_KERNEL_SIZE = cfg.contains("linear_conv_kernel_dim")
                       ? cfg["linear_conv_kernel_dim"].get<int>()
                       : 4;
  NNTR_THROW_IF(CONV_KERNEL_SIZE != 3 && CONV_KERNEL_SIZE != 4,
                std::invalid_argument)
    << "Qwen3.5 linear conv kernel must be 3 or 4, got " << CONV_KERNEL_SIZE;

  // rope_parameters carries theta (already parsed by the base from the same
  // key) plus the partial rotary factor.
  if (cfg.contains("rope_parameters") &&
      cfg["rope_parameters"].contains("partial_rotary_factor")) {
    PARTIAL_ROPE_FACTOR =
      cfg["rope_parameters"]["partial_rotary_factor"].get<float>();
  } else if (cfg.contains("partial_rotary_factor")) {
    PARTIAL_ROPE_FACTOR = cfg["partial_rotary_factor"].get<float>();
  }
}

Tensor Qwen35Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {
  if (layer_types_.at(layer_id) == "linear_attention")
    return createLinearAttnBlock(layer_id, input);
  return Transformer::createTransformerDecoderBlock(layer_id, input);
}

Tensor Qwen35Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {
  (void)seq_len;
  auto prefix = "layer" + std::to_string(layer_id);

  // The HF q_proj carries q and the attention output gate interleaved
  // per 512-block; the converter pre-splits it into two plain FCs (wq,
  // wgate), so no split layer is needed here and both stay LoRA targets.
  std::vector<std::string> wq_props = {
    withKey("name", prefix + "_wq"), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wq", layer_id))
    appendLoRAProps(wq_props);
  else if (LORA_RANK > 0)
    wq_props.push_back(withKey("trainable", "false"));
  LayerHandle wq(createLayer("fully_connected", wq_props));
  Tensor q = wq(query);

  std::vector<std::string> q_norm_props = {
    withKey("name", prefix + "_q_norm"), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", layerEngine())};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    q_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_props));
  Tensor q_normed = q_norm(q);

  // K side (GQA)
  std::vector<std::string> wk_props = {
    withKey("name", prefix + "_wk"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wk", layer_id))
    appendLoRAProps(wk_props);
  else if (LORA_RANK > 0)
    wk_props.push_back(withKey("trainable", "false"));
  LayerHandle wk(createLayer("fully_connected", wk_props));
  Tensor k = wk(key);

  std::vector<std::string> k_norm_props = {
    withKey("name", prefix + "_k_norm"), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", layerEngine())};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    k_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_props));
  Tensor k_normed = k_norm(k);

  std::vector<std::string> wv_props = {
    withKey("name", prefix + "_wv"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wv", layer_id))
    appendLoRAProps(wv_props);
  else if (LORA_RANK > 0)
    wv_props.push_back(withKey("trainable", "false"));
  LayerHandle wv(createLayer("fully_connected", wv_props));
  Tensor v = wv(value);

  // The gate branch is created AFTER the v branch so the creation order
  // matches the graph's dataflow order: the positional weight file is read
  // in graph order, and wgate lands between wv and wo either way.
  std::vector<std::string> wgate_props = {
    withKey("name", prefix + "_wgate"), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wgate", layer_id))
    appendLoRAProps(wgate_props);
  else if (LORA_RANK > 0)
    wgate_props.push_back(withKey("trainable", "false"));
  LayerHandle wgate(createLayer("fully_connected", wgate_props));
  Tensor gate = wgate(query);

  // sigmoid(gate): host layer (the gate multiplies the attention output on
  // the host in every current path).
  LayerHandle gate_act(createLayer(
    "activation", {withKey("name", prefix + "_attn_gate_act"),
                   withKey("activation", "sigmoid"), withKey("engine", "cpu")}));
  Tensor gate_sig = gate_act(gate);

  static const bool _kv_int8_setup = std::getenv("NNTR_KV_INT8") != nullptr;

  LayerHandle mha(createLayer(
    "mha_core",
    {
      withKey("name", prefix + "_attention"),
      withKey("num_heads", n_heads),
      withKey("num_heads_kv", n_heads / GQA_SIZE),
      withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
      withKey("init_seq_len", std::to_string(INIT_SEQ_LEN)),
      withKey("sliding_window", getLayerSlidingWindow(layer_id)),
      withKey("rope_theta", ROPE_THETA),
      withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
      withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
      withKey("is_causal", IS_CAUSAL ? "true" : "false"),
      // Qwen3.5 partial RoPE: theta 1e7 with only the first
      // partial_rotary_factor * head_dim channels rotated (the text-only
      // 1D position makes the interleaved mrope re-indexing a no-op, so a
      // plain partial rope is exact).
      withKey("rope_scaling_type", "default"),
      withKey("rope_partial_rotary_factor", PARTIAL_ROPE_FACTOR),
      withKey("use_gemm_attention",
              causallm_gemm_attention(USE_FLASH_ATTENTION) ? "true" : "false"),
      withKey("gpu_decode_attn",
              getModelFeatures().decode_gpu ? "true" : "false"),
      withKey("gpu_decode_rope",
              getModelFeatures().decode_rope_gpu ? "true" : "false"),
      withKey("gpu_ohwi_rope",
              getModelFeatures().decode_gpu ? "true" : "false"),
    }));

  Tensor a;
  if (FOR_TRAINING) {
    a = mha({q_normed, k_normed, v});
  } else {
    a = wireAttentionKVCache(layer_id, n_heads, mha, q_normed, k_normed, v,
                             _kv_int8_setup);
  }

  // Gated output: a * sigmoid(gate). Training uses the core multiply layer
  // (backwarding-capable); inference uses custom_multiply, whose incremental
  // path is the one exercised by the LFM2 decode (the generic binary-op
  // incremental makes a height-slicing assumption that does not hold here).
  LayerHandle gate_mul(createLayer(
    FOR_TRAINING ? "multiply" : "custom_multiply",
    {withKey("name", prefix + "_attn_gate_mul"), withKey("engine", "cpu")}));
  Tensor gated = gate_mul({a, gate_sig});

  std::vector<std::string> wo_props = {
    withKey("name", prefix + "_attention_out"), withKey("unit", DIM),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", layerEngine())};
  if (hasLoRA("wo", layer_id))
    appendLoRAProps(wo_props);
  else if (LORA_RANK > 0)
    wo_props.push_back(withKey("trainable", "false"));
  LayerHandle wo(createLayer("fully_connected", wo_props));
  return wo(gated);
}

Tensor Qwen35Transformer::createLinearAttnBlock(const int layer_id,
                                                Tensor input) {
  auto prefix = "layer" + std::to_string(layer_id);
  const unsigned int NH = static_cast<unsigned int>(LINEAR_NUM_HEADS);
  const unsigned int HD = static_cast<unsigned int>(LINEAR_HEAD_DIM);
  const unsigned int KV_DIM = NH * HD;            // 2048 (key == value width)
  const unsigned int QKV_W = KV_DIM * 3; // q | k | v, each KV_DIM (6144)

  // Pre-attention norm (same naming convention as the base block).
  std::vector<std::string> attn_norm_props = {
    withKey("name", prefix + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
    withKey("engine", "cpu")};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    attn_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  // in_proj_qkv: [B,1,T,DIM] -> [B,1,T,6144] (LoRA target; QS4CX-quantized)
  std::vector<std::string> qkv_props = {
    withKey("name", prefix + "_linattn_in_qkv"), withKey("unit", QKV_W),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", "cpu")};
  if (hasLoRA("linattn_in_qkv", layer_id))
    appendLoRAProps(qkv_props);
  else if (LORA_RANK > 0)
    qkv_props.push_back(withKey("trainable", "false"));
  LayerHandle in_qkv(createLayer("fully_connected", qkv_props));
  Tensor qkv = in_qkv(normed);

  // Depthwise causal conv (kernel 4, frozen FP32 weight) — HF fuses silu
  // into the conv op; here it is a separate node so the conv core stays a
  // pure linear op for backwarding. The conv weight itself never trains
  // (the layer only propagates input gradients).
  LayerHandle conv(createLayer(
    "causal_conv1d",
    {withKey("name", prefix + "_linattn_conv"),
     withKey("weight_dtype", "FP32"), withKey("engine", "cpu"),
     withKey("kernel_size", std::to_string(CONV_KERNEL_SIZE)),
     withKey("trainable", "false")}));
  Tensor conv_out = conv(qkv);

  LayerHandle conv_act(createLayer(
    "activation", {withKey("name", prefix + "_linattn_conv_act"),
                   withKey("activation", "swish"), withKey("engine", "cpu")}));
  Tensor act = conv_act(conv_out);

  // Split the conv output into q | k | v along the width axis (the conv is
  // depthwise, so conv-then-split == split-then-conv).
  LayerHandle qkv_split(createLayer(
    "split", {withKey("name", prefix + "_linattn_qkv_split"),
              withKey("axis", 3), withKey("split_number", 3),
              withKey("engine", "cpu")}));
  Tensor qkv3 = qkv_split(act);
  Tensor q = qkv3.output(0);
  Tensor k = qkv3.output(1);
  Tensor v = qkv3.output(2);

  // z: gated-norm gate projection (LoRA target)
  std::vector<std::string> z_props = {
    withKey("name", prefix + "_linattn_in_z"), withKey("unit", KV_DIM),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", "cpu")};
  if (hasLoRA("linattn_in_z", layer_id))
    appendLoRAProps(z_props);
  else if (LORA_RANK > 0)
    z_props.push_back(withKey("trainable", "false"));
  LayerHandle in_z(createLayer("fully_connected", z_props));
  Tensor z = in_z(normed);

  // b / a: 16-unit scalar projections, frozen by design (they feed
  // sigmoid/softplus where 16-bit precision is not enough) and never
  // quantized (absent from the quantize dtype map). weight_dtype=FP32 is
  // REQUIRED here: without it the runtime graph derives these FC weights'
  // dtype from the pack's fc_layer_dtype (QS4CX) while the quantizer saved
  // them FP32 (unmapped), and every record after them in the positional file
  // is then read at a shifted offset.
  LayerHandle in_b(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_linattn_in_b"), withKey("unit", NH),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", "cpu"), withKey("trainable", "false"),
     withKey("weight_dtype", "FP32")}));
  Tensor b = in_b(normed);

  LayerHandle in_a(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_linattn_in_a"), withKey("unit", NH),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", "cpu"), withKey("trainable", "false"),
     withKey("weight_dtype", "FP32")}));
  Tensor a = in_a(normed);

  // Gated delta rule core (frozen A_log / dt_bias / gated-norm weight).
  std::vector<std::string> core_props = {
    withKey("name", prefix + "_linattn_core"), withKey("engine", "cpu"),
    withKey("gdelta_num_heads", NH), withKey("gdelta_head_dim", HD)};
  if (FOR_TRAINING)
    core_props.push_back(withKey("gdelta_backward_cache", "true"));
  LayerHandle core(createLayer("gated_delta_core", core_props));
  Tensor core_out = core({q, k, v, z, b, a});

  // out projection (LoRA target)
  std::vector<std::string> out_props = {
    withKey("name", prefix + "_linattn_out"), withKey("unit", DIM),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("engine", "cpu")};
  if (hasLoRA("linattn_out", layer_id))
    appendLoRAProps(out_props);
  else if (LORA_RANK > 0)
    out_props.push_back(withKey("trainable", "false"));
  LayerHandle out_proj(createLayer("fully_connected", out_props));
  Tensor attn_out = out_proj(core_out);

  // Residual + FFN, the same structure as the base block (HF applies both to
  // EVERY layer, linear attention included) and the same weight order the
  // converter writes for every layer: post-norm, gate, up, down.
  LayerHandle block_add(createLayer(
    "addition", {withKey("name", prefix + "_decoder_add"),
                 withKey("engine", "cpu")}));
  Tensor residual = block_add({input, attn_out});

  std::vector<std::string> ffn_norm_props = {
    withKey("name", prefix + "_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
    withKey("engine", "cpu")};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    ffn_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_norm(createLayer("rms_norm", ffn_norm_props));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out =
    Transformer::createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle block_out(createLayer(
    "addition", {withKey("name", prefix + "_decoder_output"),
                 withKey("engine", "cpu")}));
  return block_out({residual, ffn_out});
}

void Qwen35Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  // Each factory in its own try/catch so an "already taken key" from a
  // model sharing the global AppContext (qwen3 registers reshaped_rms_norm
  // too) does not abort the remaining qwen3.5-specific layers. Matches the
  // LFM2 / Gemma4 pattern.
  auto tryRegister = [&](auto factory_fn) {
    try {
      app_context->registerFactory(factory_fn);
    } catch (std::invalid_argument &e) {
      std::cerr << "failed to register factory, reason: " << e.what()
                << std::endl;
    }
  };

  tryRegister(nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  tryRegister(nntrainer::createLayer<causallm::CustomMultiplyLayer>);
  tryRegister(nntrainer::createLayer<causallm::CausalConv1DLayer>);
  tryRegister(nntrainer::createLayer<causallm::GatedDeltaCoreLayer>);

  // GPU-context registration of ReshapedRMSNormLayer (qwen3 pattern): the
  // full-attention q/k norms build with engine=layerEngine() and resolve on
  // the gpu context in an OpenCL build.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context (CPU-only build) or already registered — both benign.
  }
}

void Qwen35CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen35Transformer::registerCustomLayers();
}

} // namespace causallm
