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
      activation_prime_fn(x, ret_derivative);
      ret_derivative.multiply_i_strided(derivative);

      return ret_derivative;
    };
  } else {
    _act_prime_fn =
      [activation_prime_fn](Tensor &x, Tensor &ret_derivative,
                            Tensor const &derivative) -> Tensor & {
      activation_prime_fn(x, x);
      derivative.multiply_strided(x, ret_derivative);

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
      x.apply(activation_prime_fn, x);
      derivative.multiply_strided(x, ret_derivative);

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

void ActiFunc::run_fn(Tensor const &input, Tensor &output) {
  _act_fn(input, output);
}

Tensor &ActiFunc::run_prime_fn(Tensor &output, Tensor &outgoing_derivative,
                               Tensor const &incoming_derivative) {
  return _act_prime_fn(output, outgoing_derivative, incoming_derivative);
}

bool ActiFunc::supportInPlace() const { return in_place; }

Tensor &ActiFunc::softmax(Tensor const &input, Tensor &output) {
  /**
   * shiftx_logit = logit - max_batch(logit)
   * softmax = exp(shiftx_logit) / (sum(exp(shiftx_logit)))
   *
   * @note softmax is applied on the last dimension
   */
  /** TODO: support strided operations */
  if (input.size() == output.size() &&
      input.getStrides() != output.getStrides())
    throw std::invalid_argument(
      "Softmax does not support operating on strided tensors");

  unsigned int width = input.width();
  unsigned int bch_size = input.getDim().getDataLen() / width;

  // copy will not executed in inplace case
  output.copy(input);

  float *output_data = output.getData();

  // prevent overflow
  Tensor tmp(width);
  for (unsigned int i = 0; i < bch_size; i++) {
    float *ptr = output_data + i * width;

    // find max value and subtract it
    float max_value = *std::max_element(ptr, ptr + width);

    tmp.setValue(max_value);
    saxpy(width, -1, tmp.getData(), 1, ptr, 1);
  }

  // take exp
  output.apply(exp_util, output);

  // take sum over the last dimension
  Tensor sum = output.sum(3);

  for (unsigned int i = 0; i < bch_size; i++) {
    float *ptr = output_data + i * width;
    std::transform(
      ptr, ptr + width, ptr,
      std::bind(std::divides<float>(), std::placeholders::_1, sum.getValue(i)));
  }

  return output;
}

Tensor &ActiFunc::softmaxPrime(Tensor const &output,
                               Tensor &outgoing_derivative,
                               Tensor const &incoming_derivative) {
  /** TODO: support strided operations */
  if ((output.size() == outgoing_derivative.size() &&
       output.getStrides() != outgoing_derivative.getStrides()) ||
      (output.size() == incoming_derivative.size() &&
       output.getStrides() != incoming_derivative.getStrides()))
    throw std::invalid_argument(
      "SoftmaxPrime does not support operating on strided tensors");

  unsigned int batch = output.batch();
  unsigned int channel = output.channel();
  unsigned int height = output.height();
  unsigned int width = output.width();

  if (outgoing_derivative.empty())
    outgoing_derivative = Tensor(output.getDim());

  const float *output_data = output.getData();
  const float *incoming_derivative_data = incoming_derivative.getData();
  float *outgoing_derivative_data = outgoing_derivative.getData();

  Tensor tmp = Tensor(width);
  float *tmp_data = tmp.getData();
  unsigned int output_width_stride = output.getStrides()[3];
  for (unsigned int b = 0; b < batch; ++b) {
    int b_offset = b * channel * height * width;
    for (unsigned int c = 0; c < channel; ++c) {
      int bc_offset = b_offset + c * height * width;
      for (unsigned int h = 0; h < height; ++h) {
        int bch_offset = bc_offset + h * width;
        for (unsigned int w1 = 0; w1 < width; ++w1) {
          float sum = 0.0f;
          for (unsigned int w2 = 0; w2 < width; ++w2) {
            float val;
            if (w1 == w2) {
              val = output_data[bch_offset + w2] *
                    (1.0f - output_data[bch_offset + w1]);
            } else {
              val =
                -output_data[bch_offset + w2] * output_data[bch_offset + w1];
            }
            if (!incoming_derivative.empty())
              val *= incoming_derivative_data[bch_offset + w2];
            sum += val;
          }
          tmp.setValue(0, 0, 0, w1, sum);
        }
        scopy(width, tmp_data, 1, outgoing_derivative_data + bch_offset,
              output_width_stride);
      }
    }
  }
  return outgoing_derivative;
}

float ActiFunc::sigmoid(float x) { return 1.0f / (1.0f + exp_util(-x)); }

float ActiFunc::sigmoidPrime(float x) {
  // float sprime = sigmoid(x);
  return x * (1.0f - x);
}

float ActiFunc::tanhFloat(float x) {
  // return (float)tanh(x); Using sigmoid implementaion for latency reason.
  return 2.0 * sigmoid(2.0 * x) - 1.0;
}

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
