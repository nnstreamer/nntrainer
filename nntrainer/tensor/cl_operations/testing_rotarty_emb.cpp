// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	testing_rotary_emb.cpp
 * @date	28 August 2024
 * @brief	Rotary Embedding CPU code
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "tensor.h"
#include <string>

/**
 * @brief     compute frequency for rotary embedding
 * @param[in] dim hidden dim size
 * @param[in] seq_len sequency length
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[out] freqs base frequencies array to be used in computation of cos and
 * sin values for each position in sequence
 * @param[in] theta rotary angle
 */
void precompute_freqs(int dim, unsigned int seq_len,
                      std::vector<std::vector<float>> &freqs_cos,
                      std::vector<std::vector<float>> &freqs_sin,
                      std::vector<float> &freqs, float theta = 10000.0) {
  if (freqs_cos.empty()) {
    unsigned int half_ = dim / 2;
    for (unsigned int i = 0; i < half_; ++i) {
      freqs.push_back(1.0 /
                      (std::pow(theta, (2 * i) / static_cast<float>(dim))));
    }

    auto cos = std::vector<std::vector<float>>();
    cos.assign(seq_len, std::vector<float>(dim, 0));

    auto sin = std::vector<std::vector<float>>();
    sin.assign(seq_len, std::vector<float>(dim, 0));

    for (unsigned int i = 0; i < seq_len; ++i) {
      for (unsigned int j = 0; j < half_; ++j) {
        float angle = i * freqs[j];
        cos[i][j] = std::cos(angle);
        cos[i][j + half_] = std::cos(angle); // repeated 2 times

        sin[i][j] = std::sin(angle);
        sin[i][j + half_] = std::sin(angle); // repeated 2 times
      }
    }
    freqs_cos = cos;
    freqs_sin = sin;
  }
}

/**
 * @brief     apply rotary embedding
 * @param[in] in input tensor
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep maximum timestep
 */
void apply_rotary_emb_tensor(nntrainer::Tensor &in, unsigned int dim,
                             unsigned int from, unsigned int max_timestep) {
  nntrainer::Tensor out(in.getDim());
  float value = 0;
  float transformed_value = 0.0;
  unsigned int half_ = dim / 2;

  std::vector<std::vector<float>> freqs_cos = {};
  std::vector<std::vector<float>> freqs_sin = {};
  std::vector<float> freqs;

  precompute_freqs(dim, max_timestep, freqs_cos, freqs_sin, freqs);

  std::vector<float> cos_;
  std::vector<float> sin_;

  if (from >= max_timestep) {
    cos_ = std::vector<float>(dim);
    sin_ = std::vector<float>(dim);
    for (unsigned int i = 0; i < half_; ++i) {
      float angle = from * freqs[i];
      cos_[i] = std::cos(angle);
      cos_[i + half_] = std::cos(angle); // repeated 2 times

      sin_[i] = std::sin(angle);
      sin_[i + half_] = std::sin(angle); // repeated 2 times
    }
  } else {
    cos_.resize(max_timestep);
    sin_.resize(max_timestep);
  }

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {

    unsigned int input_batch_size, input_height, input_width, input_channels;
    input_batch_size = in.batch();
    input_height = in.height();
    input_width = in.width();
    input_channels = in.channel();

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from + h < max_timestep) {
            cos_ = freqs_cos[from + h];
            sin_ = freqs_sin[from + h];
          }

          for (unsigned int w = 0; w < in.width(); w = w + dim) {
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              if (span < in.width()) {
                value = in.getValue<float>(b, c, h, span);
                if (k < half_) {
                  transformed_value =
                    -1.0 * in.getValue<float>(b, c, h, span + half_);
                } else {
                  transformed_value = in.getValue<float>(b, c, h, span - half_);
                }
                value = value * cos_[k] + transformed_value * sin_[k];
                out.setValue(b, c, h, span, value);
              }
            }
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from + h < max_timestep) {
            cos_ = freqs_cos[from + h];
            sin_ = freqs_sin[from + h];
          }
          for (unsigned int w = 0; w < in.width(); w = w + dim) {
#ifdef USE_NEON
            nntrainer::compute_rotary_embedding_value(
              dim, half_, w, in.getData<_FP16>() + in.getIndex(b, c, h, 0),
              out.getData<_FP16>() + out.getIndex(b, c, h, 0), cos_.data(),
              sin_.data());
#else
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              value = static_cast<float>(in.getValue<_FP16>(b, c, h, span));

              if (k < half_) {
                transformed_value =
                  -1.0 *
                  static_cast<float>(in.getValue<_FP16>(b, c, h, half_ + span));
              } else {
                transformed_value =
                  static_cast<float>(in.getValue<_FP16>(b, c, h, span - half_));
              }
              out.setValue(b, c, h, span,
                           static_cast<_FP16>(value * cos_[k] +
                                              transformed_value * sin_[k]));
            }
#endif
          }
        }
      }
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  if (from >= max_timestep) {
    cos_.clear();
    sin_.clear();
  }

  in.copy(out);
}
