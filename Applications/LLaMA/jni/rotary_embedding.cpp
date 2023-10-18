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

std::vector<std::vector<std::complex<float>>> *
precompute_freqs_cis(int dim, int seq_len, float theta = 10000.0) {
  std::vector<float> freqs(dim / 2);
  for (int i = 0; i < dim / 2; ++i) {
    freqs[i] = 1.0 / (std::pow(theta, (2 * i) / static_cast<float>(dim)));
  }

  auto cis = new std::vector<std::vector<std::complex<float>>>();
  cis->assign(seq_len, std::vector<std::complex<float>>(dim / 2, 0));

  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < dim / 2; ++j) {
      float angle = i * freqs[j];
      (*cis)[i][j] = std::polar(1.0f, angle);
    }
  }

  return cis;
}

template <typename T = float>
std::tuple<T, T>
apply_rotary_emb(T real, T imag,
                 std::vector<std::vector<std::complex<float>>> *freqs, int i,
                 int j) {
  std::complex<float> input_complex(static_cast<float>(real),
                                    static_cast<float>(imag));
  std::complex<float> output_complex = input_complex * (*freqs)[i][(int)j / 2];
  return std::make_tuple(static_cast<T>(output_complex.real()),
                         static_cast<T>(output_complex.imag()));

} // namespace custom

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
  freqs_cis = precompute_freqs_cis(dimension, seq_len);
}

void RotaryEmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  for (int b = 0; b < (int)in.batch(); b++) {
    for (int c = 0; c < (int)in.channel(); c++) {
      for (int h = 0; h < (int)in.height(); h++) {
        for (int w = 0; w < (int)in.width(); w = w + 2) {
          float real = in.getValue(b, c, h, w);
          float imag = in.getValue(b, c, h, w + 1);
          std::tie(real, imag) = apply_rotary_emb(real, imag, freqs_cis, h, w);
          out.setValue(b, c, h, w, real);
          out.setValue(b, c, h, w + 1, imag);
        }
      }
    }
  }
}

void RotaryEmbeddingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (int b = 0; b < (int)in.batch(); b++) {
      for (int c = 0; c < (int)in.channel(); c++) {
        for (int h = 0; h < (int)in.height(); h++) {
          for (int w = 0; w < (int)in.width(); w = w + 2) {
            float *data = in.getAddress(b, c, h, w);
            float real = data[0];
            float imag = data[1];
            std::tie(real, imag) =
              apply_rotary_emb(real, imag, freqs_cis, h, w);
            out.setValue(b, c, h, w, real);
            out.setValue(b, c, h, w + 1, imag);
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    for (int b = 0; b < (int)in.batch(); b++) {
      for (int c = 0; c < (int)in.channel(); c++) {
        for (int h = 0; h < (int)in.height(); h++) {
          for (int w = 0; w < (int)in.width(); w = w + 2) {
#ifdef ENABLE_FP16
            _FP16 *data = in.getAddress<_FP16>(b, c, h, w);
            float real = static_cast<float>(data[0]);
            float imag = static_cast<float>(data[1]);
            std::tie(real, imag) =
              apply_rotary_emb(real, imag, freqs_cis, h, w);
            out.setValue(b, c, h, w, static_cast<_FP16>(real));
            out.setValue(b, c, h, w + 1, static_cast<_FP16>(imag));
#else
            throw std::invalid_argument("enable-fp16 is not set");

#endif
          }
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
