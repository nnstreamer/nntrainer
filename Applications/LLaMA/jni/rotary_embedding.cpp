// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rotary_embedding.cpp
 * @date   31 July 2023
 * @brief  Implementation of Rotary Positional Embedding
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "rotary_embedding.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

std::vector<std::vector<float>> *precompute_freqs_cos(int dim, int seq_len,
                                                      float theta = 10000.0) {
  seq_len = 1024; // forcing value for temporal

  std::vector<float> freqs(dim / 2);
  for (int i = 0; i < dim / 2; ++i) {
    freqs[i] = 1.0 / (std::pow(theta, (2 * i) / static_cast<float>(dim)));
  }

  auto cos = new std::vector<std::vector<float>>();
  cos->assign(seq_len, std::vector<float>(dim, 0));

  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < dim / 2; ++j) {
      float angle = i * freqs[j];
      (*cos)[i][j] = std::cos(angle);
      (*cos)[i][j + int(dim / 2)] = std::cos(angle); // repeated 2 times
    }
  }

  return cos;
}

std::vector<std::vector<float>> *precompute_freqs_sin(int dim, int seq_len,
                                                      float theta = 10000.0) {
  seq_len = 1024; // forcing value for temporal

  std::vector<float> freqs(dim / 2);
  for (int i = 0; i < dim / 2; ++i) {
    freqs[i] = 1.0 / (std::pow(theta, (2 * i) / static_cast<float>(dim)));
  }

  auto sin = new std::vector<std::vector<float>>();
  sin->assign(seq_len, std::vector<float>(dim, 0));

  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < dim / 2; ++j) {
      float angle = i * freqs[j];
      (*sin)[i][j] = std::sin(angle);
      (*sin)[i][j + int(dim / 2)] = std::sin(angle); // repeated 2 times
    }
  }

  return sin;
}

void RotaryEmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height());
      dim[i].width(dim[i].width());
    }
  }

  context.setOutputDimensions(dim);

  int seq_len = dim[0].height();
  int dimension = dim[0].width();
  freqs_cos = precompute_freqs_cos(dimension, seq_len);
  freqs_sin = precompute_freqs_sin(dimension, seq_len);
}

void RotaryEmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  float value = 0;
  float transformed_value = 0;
  int dim = in.width();

  for (int b = 0; b < (int)in.batch(); b++) {
    for (int c = 0; c < (int)in.channel(); c++) {
      for (int h = 0; h < (int)in.height(); h++) {
        for (int w = 0; w < (int)in.width(); w++) {
          value = in.getValue(b, c, h, w);

          if (w < dim / 2) {
            transformed_value = -1 * in.getValue(b, c, h, dim / 2 + w);
          } else {
            transformed_value = in.getValue(b, c, h, w - dim / 2);
          }

          out.setValue(b, c, h, w,
                       value * (*freqs_cos)[h][w] +
                         transformed_value * (*freqs_sin)[h][w]);
        }
      }
    }
  }
}

void RotaryEmbeddingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rotary_embedding_layer() {
  auto layer = new RotaryEmbeddingLayer();
  std::cout << "rotary_positional_embedding created\n";
  return layer;
}

void destroy_rotary_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "rotary_positional_embedding deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_rotary_embedding_layer, destroy_rotary_embedding_layer};
}

#endif

} // namespace custom
