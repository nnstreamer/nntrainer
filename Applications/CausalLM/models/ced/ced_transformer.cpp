// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   ced_transformer.cpp
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This ced_transformer.cpp constructs a class for the CED audio
 * tagging model (https://huggingface.co/mispeech/ced-tiny).
 */

#include "ced_transformer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <factory.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Map the HuggingFace CED config onto the generic ViT parameters.
 *
 * Every value below comes from the upstream config.json of the CED checkpoint
 * (https://huggingface.co/mispeech/ced-tiny/blob/main/config.json); the two
 * epsilons are the PyTorch defaults implied by modeling_ced.py rather than
 * config entries, and are annotated as such.
 */
void CedTransformer::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  const unsigned int embed_dim = cfg.value("embed_dim", 192u);
  const unsigned int num_heads = cfg.value("num_heads", 3u);
  const float mlp_ratio = cfg.value("mlp_ratio", 4.0f);
  const unsigned int patch_size = cfg.value("patch_size", 16u);
  const std::string pooling = cfg.value("pooling", std::string("mean"));

  if (pooling != "mean" && pooling != "logit") {
    throw std::invalid_argument(
      "CED pooling mode '" + pooling +
      "' is not supported yet (only 'mean' and 'logit')");
  }

  json vit_cfg = cfg;

  // -- encoder trunk -----------------------------------------------------
  vit_cfg["hidden_size"] = embed_dim;
  vit_cfg["num_hidden_layers"] = cfg.value("depth", 12u);
  vit_cfg["num_attention_heads"] = num_heads;
  vit_cfg["num_key_value_heads"] = num_heads;
  vit_cfg["head_dim"] = embed_dim / num_heads;
  vit_cfg["intermediate_size"] =
    static_cast<unsigned int>(embed_dim * mlp_ratio);
  // modeling_ced.py: norm_layer = partial(nn.LayerNorm, eps=1e-6)
  vit_cfg["norm_eps"] = 1e-6;
  vit_cfg["is_causal"] = false;
  vit_cfg["rope_theta"] = 0;
  vit_cfg["tie_word_embeddings"] = false;
  vit_cfg["vocab_size"] = 1;

  // -- input geometry ----------------------------------------------------
  vit_cfg["input_height"] = cfg.value("n_mels", 64u);
  vit_cfg["input_width"] = cfg.value("target_length", 1012u);
  vit_cfg["in_chans"] = 1;
  vit_cfg["patch_size"] = patch_size;
  vit_cfg["patch_stride"] = cfg.value("patch_stride", patch_size);
  vit_cfg.erase("num_patches"); // derive from the grid, never guess

  // -- init_bn -----------------------------------------------------------
  vit_cfg["use_input_norm"] = true;
  // nn.BatchNorm2d default eps; modeling_ced.py only overrides momentum.
  vit_cfg["input_norm_eps"] = 1e-5;

  // -- classification head ----------------------------------------------
  // outputlayer = Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim,
  // outputdim)); forward_head applies sigmoid for every pooling mode except
  // "logit".
  vit_cfg["num_classes"] = cfg.value("outputdim", 527u);
  vit_cfg["pooling"] = std::string("mean");
  vit_cfg["head_sigmoid"] = (pooling != "logit");
  vit_cfg["head_norm_eps"] = 1e-5; // plain nn.LayerNorm default

  TimmViTTransformer::setupParameters(vit_cfg, generation_cfg, nntr_cfg);

  // Class names, for a readable top-k report.
  LABELS.clear();
  if (cfg.contains("id2label") && cfg["id2label"].is_object()) {
    LABELS.assign(NUM_CLASSES, std::string());
    for (auto it = cfg["id2label"].begin(); it != cfg["id2label"].end(); ++it) {
      const unsigned long idx = std::strtoul(it.key().c_str(), nullptr, 10);
      if (idx < LABELS.size() && it.value().is_string()) {
        LABELS[idx] = it.value().get<std::string>();
      }
    }
  }

  std::cout << "[CED] input " << IMG_CHANNELS << "x" << INPUT_HEIGHT << "x"
            << INPUT_WIDTH << ", patch " << PATCH_SIZE << "/" << PATCH_STRIDE
            << " -> grid " << GRID_H << "x" << GRID_W << " = " << NUM_PATCHES
            << " tokens, dim " << DIM << ", layers " << NUM_LAYERS << ", heads "
            << NUM_HEADS << ", classes " << NUM_CLASSES
            << (HEAD_SIGMOID ? " (sigmoid)" : " (logit)") << std::endl;
}

/**
 * @brief Read a raw FP32 mel spectrogram of exactly `count` values.
 */
static std::vector<float> loadMelSpectrogram(const std::string &path,
                                             size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Failed to open mel spectrogram: " + path);
  }
  const size_t got = static_cast<size_t>(f.tellg()) / sizeof(float);
  if (got != count) {
    throw std::runtime_error(
      "Mel spectrogram " + path + " holds " + std::to_string(got) +
      " floats, expected " + std::to_string(count) +
      " (n_mels * target_length). The file must be FP32 [n_mels, frames], "
      "frame-major rows, i.e. the HuggingFace `input_values` tensor.");
  }
  f.seekg(0);
  std::vector<float> data(count);
  f.read(reinterpret_cast<char *>(data.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  return data;
}

/**
 * @brief Run CED inference on a mel-spectrogram file.
 */
void CedTransformer::run(const WSTR prompt, bool do_sample,
                         const WSTR system_prompt, const WSTR tail_prompt,
                         bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("CED model is not initialized. Please call "
                             "initialize() before run().");
  }

  const std::string mel_path(prompt);
  const size_t mel_count =
    static_cast<size_t>(INPUT_HEIGHT) * static_cast<size_t>(INPUT_WIDTH);
  std::vector<float> mel = loadMelSpectrogram(mel_path, mel_count);

  std::vector<float *> input{mel.data()};
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  const size_t out_count = NUM_CLASSES;

  // Top-5 report.
  std::vector<size_t> order(out_count);
  std::iota(order.begin(), order.end(), 0);
  const size_t topk = std::min<size_t>(5, out_count);
  std::partial_sort(
    order.begin(), order.begin() + topk, order.end(),
    [&](size_t a, size_t b) { return output[0][a] > output[0][b]; });
  std::cout << std::setprecision(6) << std::fixed;
  for (size_t i = 0; i < topk; ++i) {
    const size_t idx = order[i];
    std::cout << "[top" << (i + 1) << "] " << idx << " "
              << (idx < LABELS.size() ? LABELS[idx] : std::string("?")) << " = "
              << output[0][idx] << std::endl;
  }
  std::cout.unsetf(std::ios::floatfield);

  if (const char *dump_path = std::getenv("CED_OUT_BIN")) {
    std::ofstream dump(dump_path, std::ios::binary);
    if (!dump) {
      std::cerr << "[CED_OUT_BIN] cannot open " << dump_path << std::endl;
    } else {
      dump.write(reinterpret_cast<const char *>(output[0]),
                 static_cast<std::streamsize>(out_count * sizeof(float)));
      std::cout << "[CED_OUT_BIN] wrote " << out_count << " floats to "
                << dump_path << std::endl;
    }
  }

  if (const char *ref_path = std::getenv("CED_REF_BIN")) {
    std::ifstream ref(ref_path, std::ios::binary | std::ios::ate);
    if (!ref) {
      std::cerr << "[CED_REF_BIN] cannot open " << ref_path << std::endl;
      return;
    }
    const size_t ref_count = static_cast<size_t>(ref.tellg()) / sizeof(float);
    if (ref_count != out_count) {
      std::cerr << "[CED_REF_BIN] size mismatch: reference has " << ref_count
                << " floats, output has " << out_count << std::endl;
      return;
    }
    ref.seekg(0);
    std::vector<float> ref_data(ref_count);
    ref.read(reinterpret_cast<char *>(ref_data.data()),
             static_cast<std::streamsize>(ref_count * sizeof(float)));

    double max_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    size_t max_idx = 0;
    size_t nan_count = 0;
    for (size_t i = 0; i < out_count; ++i) {
      const float got = output[0][i];
      // Bit-pattern NaN test: -ffast-math folds std::isnan() to false.
      uint32_t bits;
      std::memcpy(&bits, &got, sizeof(bits));
      if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        ++nan_count;
        continue;
      }
      const double diff = std::fabs(static_cast<double>(got) - ref_data[i]);
      sum_sq_diff += diff * diff;
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
        max_idx = i;
      }
    }
    std::cout << "[CED_REF_BIN] elements=" << out_count << " nan=" << nan_count
              << " max_abs_diff=" << max_abs_diff << " (at class " << max_idx
              << ") rms_diff=" << std::sqrt(sum_sq_diff / out_count)
              << std::endl;
    std::cout << "[CED_REF_BIN] "
              << ((nan_count == 0 && max_abs_diff < 1e-4) ? "PASS" : "FAIL")
              << std::endl;
  }
}

} // namespace causallm
