// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   model_base.h
 * @date   10 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Base class for all models (CausalLM, Vision, etc.) in the application.
 */

#ifndef __MODEL_BASE_H__
#define __MODEL_BASE_H__

#pragma once

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::string
#define WCHAR_P std::string &
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <layer.h>
#include <layer_context.h>
#include <map>
#include <model.h>
#include <random>
#include <stdexcept>
#include <tensor_api.h>
#include <utility>
#include <vector>

#include <limits.h>

#include "json.hpp"
#include "performance_metrics.h"
#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

namespace causallm {

/*** ALIAS ****/
using LayerHandle = ml::train::LayerHandle;
using Tensor = ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief {data, size} pointer pair produced/consumed by multimodal vision
 *        models.
 */
using multimodal_pointer = std::pair<void *, size_t>;

WIN_EXPORT class Model {
public:
  virtual ~Model() = default;

  virtual void initialize() = 0;

  inline ml::train::ModelFormat formatFromExtension(const std::string &weight_path) {
    const auto dot = weight_path.find_last_of('.');
    if (dot != std::string::npos) {
      const std::string ext = weight_path.substr(dot + 1);
      if (ext == "safetensors")
        return ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS;
    }
    return ml::train::ModelFormat::MODEL_FORMAT_BIN;
  }

  virtual void load_weight(const std::string &weight_path) {
    if (!is_initialized) {
      throw std::runtime_error(
        "Model is not initialized. Please call initialize() before load_weight().");
    }
    try {
      model->load(weight_path, formatFromExtension(weight_path));
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to load model weights: " + std::string(e.what()));
    }
  }

  virtual void repack_weight() {
    if (!is_initialized) {
      throw std::runtime_error(
        "Model is not initialized. Please call initialize() before repack_weight().");
    }
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
      fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
        auto weights = context.getWeights();
        for (auto &w : weights) {
          if (w->getVariableRef().getDataType() ==
              ml::train::TensorDim::DataType::QS4CX) {
            w->getVariableRef().pack();
          }
        }
      };
    model->forEachLayer(fn, nullptr);
  }

  virtual void save_weight(const std::string &weight_path) {
    if (!is_initialized) {
      throw std::runtime_error(
        "Model is not initialized. Please call initialize() before save_weight().");
    }
    try {
      model->save(weight_path, formatFromExtension(weight_path));
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to save model weights: " + std::string(e.what()));
    }
  }

  virtual void
  save_weight(const std::string &weight_path,
              ml::train::TensorDim::DataType dtype,
              const std::map<std::string, ml::train::TensorDim::DataType>
                &layer_dtype_map = {},
              ml::train::ISA target_isa = ml::train::ISA::DEFAULT) {
    if (!is_initialized) {
      throw std::runtime_error(
        "Model is not initialized. Please call initialize() before save_weight().");
    }
    try {
      model->save(weight_path, formatFromExtension(weight_path), dtype,
                  layer_dtype_map, target_isa);
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to save model weights with dtype: " + std::string(e.what()));
    }
  }

  virtual std::vector<std::string> getQuantizableLayerNames() const {
    return {};
  }

  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = WSTR(),
                   const WSTR tail_prompt = WSTR(), bool log_output = true) = 0;

  virtual void registerCustomLayers() {}
  virtual void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg) {
    (void)cfg;
    (void)generation_cfg;
    (void)nntr_cfg;
  }

  // Multimodal and Tokenizer interface defaults
  virtual size_t embeddingBytesPerToken() const { return 0; }
  virtual const void *lookupEmbedding(int token_id) const {
    (void)token_id;
    return nullptr;
  }
  virtual std::pair<float, int> get_embedding_info() { return {1.0f, 0}; }
  virtual void run_with_embeddings(const void *prefill_embeds, size_t n_tokens,
                                   std::vector<int> seed_tokens, bool do_sample,
                                   bool log_output) {
    (void)prefill_embeds;
    (void)n_tokens;
    (void)seed_tokens;
    (void)do_sample;
    (void)log_output;
    throw std::runtime_error("run_with_embeddings not supported");
  }
  virtual void set_quant_param(float scale, int offset) {
    (void)scale;
    (void)offset;
  }
  virtual multimodal_pointer
  run_image(const WSTR prompt, multimodal_pointer image, int image_height,
            int image_width, bool do_sample = false,
            const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
            bool log_output = true) {
    (void)prompt;
    (void)image;
    (void)image_height;
    (void)image_width;
    (void)do_sample;
    (void)system_prompt;
    (void)tail_prompt;
    (void)log_output;
    return {nullptr, 0};
  }
  virtual int getKvLen() const { return 0; }
  virtual TransformerPerformanceMetrics getPerformanceMetrics() const {
    return {};
  }
  virtual bool hasRun() const { return false; }
  virtual unsigned int getVocabSize() const { return 0; }
  virtual tokenizers::Tokenizer *getTokenizer() { return nullptr; }
  virtual void setLogitsProcessor(class LogitsProcessor *) {}
  virtual void resetLogitsProcessor() {}

protected:
  ModelHandle model;
  bool is_initialized = false;
};

} // namespace causallm

#endif /* __MODEL_BASE_H__ */
