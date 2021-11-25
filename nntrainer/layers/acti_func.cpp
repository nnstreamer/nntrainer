// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   acti_func.cpp
 * @date   22 March 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <acti_func.h>
#include <blas_interface.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {
ActiFunc::ActiFunc(ActivationType at, bool in_place_) : in_place(in_place_) {
  setActiFunc(at);
}

ActiFunc::~ActiFunc() {}

int ActiFunc::setActivation(
  std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
  std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> const
    &activation_prime_fn) {
  if (in_place)
    return ML_ERROR_INVALID_PARAMETER;

  _act_fn = activation_fn;
  _act_prime_fn = activation_prime_fn;

  return ML_ERROR_NONE;
}

int ActiFunc::setActivation(
  std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
  std::function<Tensor &(Tensor &, Tensor &)> const &activation_prime_fn) {
  _act_fn = activation_fn;
  if (!in_place) {
    _act_prime_fn =
      [activation_prime_fn](Tensor &x, Tensor &ret_derivative,
                            Tensor const &derivative) -> Tensor & {
      /** @todo update this based on supportInPlace */
      ret_derivative = activation_prime_fn(x, ret_derivative);
      ret_derivative.multiply_i_strided(derivative);

      return ret_derivative;
    };
  } else {
    _act_prime_fn =
      [activation_prime_fn](Tensor &x, Tensor &ret_derivative,
                            Tensor const &derivative) -> Tensor & {
      x = activation_prime_fn(x, x);
      ret_derivative = derivative.multiply_strided(x, ret_derivative);

      return ret_derivative;
    };
  }

  return ML_ERROR_NONE;
}

int ActiFunc::setActivation(
  std::function<float(float const)> const &activation_fn,
  std::function<float(float const)> const &activation_prime_fn) {
  _act_fn = [activation_fn](Tensor const &x, Tensor &hidden) -> Tensor & {
    return x.apply(activation_fn, hidden);
  };
  if (!in_place) {
    _act_prime_fn =
      [activation_prime_fn](Tensor &x, Tensor &ret_derivative,
                            Tensor const &derivative) -> Tensor & {
      /** @todo update this based on supportInPlace */
      x.apply(activation_prime_fn, ret_derivative);
      ret_derivative.multiply_i_strided(derivative);

      return ret_derivative;
    };
  } else {
    _act_prime_fn =
      [activation_prime_fn](Tensor &x, Tensor &ret_derivative,
                            Tensor const &derivative) -> Tensor & {
      x = x.apply(activation_prime_fn, x);
      ret_derivative = derivative.multiply_strided(x, ret_derivative);

      return ret_derivative;
    };
  }

  return ML_ERROR_NONE;
}

/**
 * @brief setActiFunc by preset ActivationType
 *
 * @param[in] ActivationType ActivationType ActivationType to be set
 */
void ActiFunc::setActiFunc(ActivationType acti_type) {
  activation_type = acti_type;

  switch (acti_type) {
  case ActivationType::ACT_TANH:
    this->setActivation(tanhFloat, tanhPrime);
    break;
  case ActivationType::ACT_SIGMOID:
    this->setActivation(sigmoid, sigmoidPrime);
    break;
  case ActivationType::ACT_SOFTMAX:
    in_place = false;
    this->setActivation(softmax, softmaxPrime);
    break;
  case ActivationType::ACT_RELU:
    this->setActivation(relu, reluPrime);
    break;
  case ActivationType::ACT_LEAKY_RELU:
    this->setActivation(leakyRelu, leakyReluPrime);
    break;
  case ActivationType::ACT_NONE:
    this->setActivation(no_op, no_op_prime);
    break;
  case ActivationType::ACT_UNKNOWN:
  default:
    throw std::runtime_error("Error: Not Supported Activation Type");
  }
}

void ActiFunc::run_fn(Tensor const &x, Tensor &output) { _act_fn(x, output); }

Tensor &ActiFunc::run_prime_fn(Tensor &in, Tensor &ret, Tensor const &deriv) {
  return _act_prime_fn(in, ret, deriv);
}

bool ActiFunc::supportInPlace() const {
  bool support_in_place = in_place;
  if (activation_type == ActivationType::ACT_SOFTMAX)
    support_in_place = false;

  return support_in_place;
}

Tensor &ActiFunc::softmax(Tensor const &t, Tensor &output) {
  /**
   * shiftx_logit = logit - max_batch(logit)
   * softmax = exp(shiftx_logit) / (sum(exp(shiftx_logit)))
   *
   * @note softmax is applied on the last dimension
   */
  unsigned int fixed_dim = t.getDim().getDataLen() / t.width();
  float *dp;
  float *rp;

  Tensor divisor = t.clone();

  dp = divisor.getData();
  unsigned int feat_len = t.width();

  for (unsigned int k = 0; k < fixed_dim; k++) {
    int index = k * feat_len;
    // find max and subtract it
    float m = *std::max_element(dp + index, dp + index + feat_len);

    Tensor tmp = Tensor(1, 1, 1, feat_len);
    tmp.setValue(m);
    saxpy(feat_len, -1, tmp.getData(), 1, dp + index, 1);
  }

  // take exp
  output = divisor.apply(exp_util, output);
  rp = output.getData();
  // take sum over the last dimension
  Tensor sum = output.sum(3);

  for (unsigned int k = 0; k < fixed_dim; k++) {
    int index = k * feat_len;
    std::transform(
      rp + index, rp + index + feat_len, rp + index,
      std::bind(std::divides<float>(), std::placeholders::_1, sum.getValue(k)));
  }

  return output;
}

Tensor &ActiFunc::softmaxPrime(Tensor const &x, Tensor &output,
                               Tensor const &derivative) {
  unsigned int batch = x.batch();
  unsigned int channel = x.channel();
  unsigned int height = x.height();
  unsigned int width = x.width();

  if (output.empty())
    output = Tensor(x.getDim());

  const float *xp = x.getData();
  const float *d = derivative.getData();
  float *pp = output.getData();

  for (unsigned int k = 0; k < batch; ++k) {
    int K = k * channel * height * width;
    for (unsigned int c = 0; c < channel; ++c) {
      int C = K + c * height * width;
      for (unsigned int i = 0; i < height; ++i) {
        int I = C + i * width;
        for (unsigned int j = 0; j < width; ++j) {
          float sum = 0.0f;
          for (unsigned int l = 0; l < width; ++l) {
            float val;
            if (j == l) {
              val = xp[I + l] * (1.0f - xp[I + j]);
            } else {
              val = -xp[I + l] * xp[I + j];
            }
            if (!derivative.empty())
              val *= d[I + l];
            sum += val;
          }
          pp[I + j] = sum;
        }
      }
    }
  }
  return output;
}

float ActiFunc::sigmoid(float x) { return 1.0f / (1.0f + exp_util(-x)); }

float ActiFunc::sigmoidPrime(float x) {
  // float sprime = sigmoid(x);
  return x * (1.0f - x);
}

float ActiFunc::tanhFloat(float x) { return (float)tanh(x); }

float ActiFunc::tanhPrime(float x) {
  // float th = (float)tanh(x);
  return 1.0f - x * x;
}

float ActiFunc::relu(float x) {
  if (x <= 0.0f) {
    return 0.0f;
  } else {
    return x;
  }
}

float ActiFunc::reluPrime(float x) {
  if (x <= 0.0f) {
    return 0.0f;
  } else {
    return 1.0f;
  }
}

float ActiFunc::no_op(float x) { return x; }

float ActiFunc::no_op_prime(float x) { return 1.0f; }

constexpr static inline float NEGATIVE_SLOPE = 0.01f;

float ActiFunc::leakyRelu(float x) {
  return x >= 0.0f ? x : NEGATIVE_SLOPE * x;
}

float ActiFunc::leakyReluPrime(float x) {
  return x >= 0.0f ? 1.0f : NEGATIVE_SLOPE;
}

void ActiFunc::executeInPlace(bool val) {
  if (val && !supportInPlace())
    throw std::runtime_error("Error setting activation layer to work in-place");

  in_place = val;
}
}; // namespace nntrainer
