// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernel_interface.cpp
 * @date	28 August 2024
 * @brief	Interface for attention OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <attention_kernel_interface.h>
#include <attention_kernels.h>

namespace nntrainer {
/**
 * @brief      compute frequency for rotary embedding
 * @param[in]  dim hidden dim size
 * @param[in]  seq_len sequency length
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[out] freqs base frequencies array to be used in the future computation
 * @param[in]  theta rotary angle
 */
void precompute_freqs(unsigned int dim, unsigned int seq_len,
                      std::vector<std::vector<float>> &freqs_cos,
                      std::vector<std::vector<float>> &freqs_sin,
                      std::vector<float> &freqs, float theta = 10000.0) {
  unsigned int half_ = dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    freqs.push_back(1.0 / (std::pow(theta, (2 * i) / static_cast<float>(dim))));
  }

  auto cos_vec = std::vector<std::vector<float>>();
  cos_vec.assign(seq_len, std::vector<float>(dim, 0));

  auto sin_vec = std::vector<std::vector<float>>();
  sin_vec.assign(seq_len, std::vector<float>(dim, 0));

  for (unsigned int i = 0; i < seq_len; ++i) {
    for (unsigned int j = 0; j < half_; ++j) {
      float angle = i * freqs[j];
      cos_vec[i][j] = std::cos(angle);
      cos_vec[i][j + half_] = std::cos(angle); // repeated 2 times

      sin_vec[i][j] = std::sin(angle);
      sin_vec[i][j + half_] = std::sin(angle); // repeated 2 times
    }
  }
  freqs_cos = cos_vec;
  freqs_sin = sin_vec;
}

/**
 * @brief     apply rotary embedding
 * @param[in] in input tensor
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep maximum timestep
 *
 * @todo      Calling precompute_freqs in finalize to reduce code redundancy.
 */
void apply_rotary_emb_cl(Tensor &in, unsigned int dim, unsigned int from,
                         unsigned int max_timestep) {
  nntrainer::Tensor out(in.getDim());
  float value = 0.0f;
  float transformed_value = 0.0f;
  unsigned int half_ = dim / 2;

  std::vector<std::vector<float>> freqs_cos = {};
  std::vector<std::vector<float>> freqs_sin = {};
  std::vector<float> freqs;

  precompute_freqs(dim, max_timestep, freqs_cos, freqs_sin, freqs);

  std::vector<float> cos_;
  std::vector<float> sin_;

  if (from >= max_timestep) {
    cos_.resize(dim);
    sin_.resize(dim);

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

  unsigned int input_batch_size, input_height, input_width, input_channels;
  input_batch_size = in.batch();
  input_height = in.height();
  input_width = in.width();
  input_channels = in.channel();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {

    unsigned int in_size = in.size();
    unsigned int out_size = out.size();
    float *data = in.getData();
    float *rdata = out.getData();

    rotary_emb_cl(data, rdata, freqs_cos, freqs_sin, cos_, sin_,
                  input_batch_size, input_channels, input_height, input_width,
                  dim, from, max_timestep, in_size, out_size);

  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16

    unsigned int in_size = in.size();
    unsigned int out_size = out.size();
    _FP16 *data = in.getData<_FP16>();
    _FP16 *rdata = out.getData<_FP16>();

    rotary_emb_cl(data, rdata, freqs_cos, freqs_sin, cos_, sin_,
                  input_batch_size, input_channels, input_height, input_width,
                  dim, from, max_timestep, in_size, out_size);
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
} // namespace nntrainer
