// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   7 July 2026
 * @brief  FastViTKeyword inference example on nntrainer.
 *
 * Builds the full model (FastViT-S12 backbone + MLP head), loads converted
 * weights, runs one forward pass, and outputs logits (507-dim) + features
 * (512-dim). When YOLO_VERIFY=1 (or KEYWORD_VERIFY=1), compares outputs
 * against PyTorch reference .bin files.
 *
 * Usage: fastvit_keyword_infer [RES_DIR] [INPUT_BIN]
 *   RES_DIR   dir with weights/ and input bins
 *             (default: Applications/CausalLM/models/FastViTKeyword/res)
 *   INPUT_BIN [1,3,320,320] float32 NCHW (default: RES_DIR/input.bin)
 *
 * Env vars:
 *   KEYWORD_IMGSZ  Input image size (square, default 320).
 *   KEYWORD_VERIFY If set, compare outputs to PyTorch references.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "fastvit_attention_layer.h"
#include "fastvit_keyword_graph.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace {

std::string RES_DIR = "Applications/CausalLM/models/FastViTKeyword/res";

/**
 * @brief Load binary file as float vector
 */
std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open: " + path);
  }
  f.seekg(0, std::ios::end);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

/** @brief Register the FastViT custom layers with the global AppContext. */
void registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(
    nntrainer::createLayer<fastvit_keyword::FastViTAttentionLayer>);
}

/** @brief Optionally compare a tensor to a PyTorch reference .bin. */
void verifyAgainst(const std::string &ref_name, const float *out, size_t n) {
  std::ifstream f(RES_DIR + "/" + ref_name, std::ios::binary);
  if (!f) {
    std::cout << "  [verify] " << ref_name << " not found, skipped"
              << std::endl;
    return;
  }
  auto ref = loadBin(RES_DIR + "/" + ref_name);
  float max_diff = 0.0f;
  for (size_t i = 0; i < n && i < ref.size(); ++i)
    max_diff = std::max(max_diff, std::abs(out[i] - ref[i]));
  std::cout << "  [verify] " << ref_name << ": max_abs_diff=" << max_diff
            << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc > 1)
      RES_DIR = argv[1];

    const int imgsz = std::getenv("KEYWORD_IMGSZ")
                        ? std::max(32, std::atoi(std::getenv("KEYWORD_IMGSZ")))
                        : 320;

    const int proj_target_dim = 507;
    const int proj_feature_dim = 512;

    const std::string input_path =
      (argc > 2) ? argv[2] : (RES_DIR + "/input.bin");
    const bool verify = std::getenv("KEYWORD_VERIFY") != nullptr ||
                        std::getenv("YOLO_VERIFY") != nullptr;

    std::cout << "[FastViTKeyword] imgsz=" << imgsz
              << " proj_target=" << proj_target_dim
              << " proj_feature=" << proj_feature_dim << std::endl;

    registerCustomLayers();

    // Build the full model: input -> backbone -> head -> {logits, features}
    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    auto x = Tensor(ml::train::TensorDim(1, 3, imgsz, imgsz,
                                         ml::train::TensorDim::Format::NCHW,
                                         ml::train::TensorDim::DataType::FP32),
                    "input0");

    auto backbone_out = fastvit_keyword::buildBackbone(x);
    auto outputs = fastvit_keyword::buildHead(backbone_out, proj_target_dim,
                                              proj_feature_dim);

    if (int ret =
          model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("compile failed: " + std::to_string(ret));

    // Load weights
    std::string weights_path = RES_DIR + "/fastvit_keyword.safetensors";
    if (const char *wenv = std::getenv("KEYWORD_WEIGHTS")) {
      weights_path =
        (wenv[0] == '/') ? std::string(wenv) : RES_DIR + "/" + wenv;
    }
    model->load(weights_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Model built and weights loaded (" << weights_path << ")."
              << std::endl;

    // Diagnostic print of first 5 weights of stem0/conv
    try {
      std::shared_ptr<ml::train::Layer> l_stem0;
      if (model->getLayer("stem0/conv", &l_stem0) == 0 && l_stem0 != nullptr) {
        auto w_stem0 = l_stem0->getWeights();
        if (!w_stem0.empty() && w_stem0[0] != nullptr) {
          std::cout << "=== C++ stem0/conv:filter weights ===" << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "w[" << i << "] = " << w_stem0[0][i] << std::endl;
          }
        } else {
          std::cout << "stem0/conv weights are empty or null!" << std::endl;
        }
      } else {
        std::cout << "stem0/conv layer not found!" << std::endl;
      }
    } catch (const std::exception &e) {
      std::cout << "Failed to print stem0 weights: " << e.what() << std::endl;
    }

    // Load input
    auto input = loadBin(input_path);
    std::cout << "Input loaded from: " << input_path
              << " (size=" << input.size() << ")" << std::endl;
    if (!input.empty()) {
      std::cout << "=== C++ Input values ===" << std::endl;
      for (int i = 0; i < 5; ++i) {
        std::cout << "input[" << i << "] = " << input[i] << std::endl;
      }
    }
    std::vector<float *> in_ptr = {input.data()};

    // Inference timing
    int bench_iters =
      std::getenv("KEYWORD_BENCH_ITERS")
        ? std::max(1, std::atoi(std::getenv("KEYWORD_BENCH_ITERS")))
        : 1;
    std::vector<float *> outs;
    double total_ms = 0.0;
    for (int it = 0; it < bench_iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      outs = model->inference(1, in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "Inference done (" << outs.size() << " outputs)." << std::endl;
    std::cout << "Inference time: " << (total_ms / bench_iters)
              << " ms (avg over " << bench_iters << " iters)" << std::endl;

    // Output: outs[0] = logits [1, 507, 1, 1], outs[1] = features [1, 512, 1,
    // 1] Print top-k logits
    const float *logits = outs[0];
    const float *features = outs[1];

    // Sigmoid for probabilities
    std::vector<float> probs(proj_target_dim);
    for (int i = 0; i < proj_target_dim; ++i)
      probs[i] = 1.0f / (1.0f + std::exp(-logits[i]));

    // Print top-20
    std::vector<int> indices(proj_target_dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 20, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::cout << "\n[TOP-20 Keywords]" << std::endl;
    for (int i = 0; i < 20 && i < proj_target_dim; ++i) {
      int idx = indices[i];
      std::printf("  %2d. idx=%3d  prob=%.4f\n", i + 1, idx, probs[idx]);
    }

    // Print threshold-based predictions
    float threshold = 0.5f;
    if (const char *t = std::getenv("KEYWORD_THRESHOLD"))
      threshold = std::stof(t);

    std::cout << "\n[Predicted threshold >= " << threshold << "]" << std::endl;
    bool any = false;
    for (int i = 0; i < proj_target_dim; ++i) {
      if (probs[i] >= threshold) {
        std::printf("  idx=%3d  prob=%.4f\n", i, probs[i]);
        any = true;
      }
    }
    if (!any)
      std::cout << "  (no keywords above threshold)" << std::endl;

    // Verification
    if (verify) {
      std::cout << "\nVerification vs PyTorch references:" << std::endl;
      verifyAgainst("ref_logits.bin", logits, proj_target_dim);
      verifyAgainst("ref_features.bin", features, proj_feature_dim);
      verifyAgainst("ref_sigmoid.bin", probs.data(), proj_target_dim);
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
