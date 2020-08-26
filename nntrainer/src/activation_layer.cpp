// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <activation_layer.h>
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <parse_util.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief     Constructor of Activation Layer
 */
ActivationLayer::ActivationLayer() : Layer() {
  setType(LAYER_ACTIVATION);
  setActivation(ACT_NONE);
}

/**
 * @brief     Initialize the layer
 *
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int ActivationLayer::initialize() {

  output_dim = input_dim;

  return ML_ERROR_NONE;
}

sharedConstTensor ActivationLayer::forwarding(sharedConstTensor in) {
  input = *in;
  /// @note @a _act_fn is expected to work out of place and not modify @a input
  hidden = _act_fn(input);

  return MAKE_SHARED_TENSOR(hidden);
}

sharedConstTensor ActivationLayer::backwarding(sharedConstTensor derivative,
                                               int iteration) {
  Tensor deriv = *derivative;
  Tensor ret;
  if (activation_type == ActiType::ACT_SOFTMAX)
    ret = _act_prime_fn(hidden, deriv);
  else
    ret = _act_prime_fn(input, deriv);

  return MAKE_SHARED_TENSOR(std::move(ret));
}

/**
 * @brief     copy layer
 * @param[in] l layer to copy
 */
void ActivationLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<ActivationLayer> from =
    std::static_pointer_cast<ActivationLayer>(l);
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->activation_type = from->activation_type;
};

int ActivationLayer::setActivation(
  std::function<Tensor(Tensor const &)> const &activation_fn,
  std::function<Tensor(Tensor const &, Tensor const &)> const
    &activation_prime_fn) {
  _act_fn = activation_fn;
  _act_prime_fn = activation_prime_fn;

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<Tensor(Tensor const &)> const &activation_fn,
  std::function<Tensor(Tensor const &)> const &activation_prime_fn) {
  _act_fn = activation_fn;
  _act_prime_fn = [activation_prime_fn](Tensor const &x,
                                        Tensor const &derivative) {
    return derivative.multiply(activation_prime_fn(x));
  };

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<float(float const)> const &activation_fn,
  std::function<float(float const)> const &activation_prime_fn) {
  _act_fn = [activation_fn](Tensor const &x) { return x.apply(activation_fn); };
  _act_prime_fn = [activation_prime_fn](Tensor const &x,
                                        Tensor const &derivative) {
    return derivative.multiply(x.apply(activation_prime_fn));
  };

  return ML_ERROR_NONE;
}

/**
 * @brief setActivation by preset actiType
 *
 * @param[in] ActiType actiType actiType to be set
 */
void ActivationLayer::setActivation(ActiType acti_type) {
  switch (acti_type) {
  case ActiType::ACT_TANH:
    this->setActivation(tanhFloat, tanhPrime);
    break;
  case ActiType::ACT_SIGMOID:
    this->setActivation(sigmoid, sigmoidPrime);
    break;
  case ActiType::ACT_SOFTMAX:
    this->setActivation(softmax, softmaxPrime);
    break;
  case ActiType::ACT_RELU:
    this->setActivation(relu, reluPrime);
    break;
  case ActiType::ACT_NONE:
    this->setActivation(no_op, no_op_prime);
    break;
  case ActiType::ACT_UNKNOWN:
  default:
    throw std::runtime_error("Error: Not Supported Activation Type");
  }
  this->activation_type = acti_type;
}

Tensor ActivationLayer::softmax(Tensor const &t) {
  /**
   * shiftx_logit = logit - max_batch(logit)
   * softmax = exp(shiftx_logit) / (sum(exp(shiftx_logit)))
   */
  unsigned int batch = t.batch();
  float *dp;
  float *rp;

  Tensor divisor = t.clone();

  dp = divisor.getData();
  unsigned int feat_len = t.getDim().getFeatureLen();

  for (unsigned int k = 0; k < batch; k++) {
    int index = k * feat_len;
    // find max and subtract it
    float m = *std::max_element(dp + index, dp + index + feat_len);

    Tensor tmp = Tensor(1, 1, 1, feat_len);
    tmp.setValue(m);
    Tensor::saxpy(feat_len, -1, tmp.getData(), 1, dp + index, 1);
  }

  // take exp
  Tensor result = divisor.apply(exp_util);
  rp = result.getData();
  // take sum over batch
  Tensor sum = result.sum_by_batch();

  for (unsigned int k = 0; k < batch; k++) {
    int index = k * feat_len;
    std::transform(rp + index, rp + index + feat_len, rp + index,
                   std::bind(std::divides<float>(), std::placeholders::_1,
                             sum.getValue(k, 0, 0, 0)));
  }

  return result;
}

Tensor ActivationLayer::softmaxPrime(Tensor const &x,
                                     Tensor const &derivative) {
  unsigned int batch = x.batch();
  unsigned int channel = x.channel();
  unsigned int height = x.height();
  unsigned int width = x.width();
  bool is_derivative = true;

  Tensor PI = Tensor(x.getDim());

  const float *xp = x.getData();
  const float *d = derivative.getData();
  float *pp = PI.getData();

  /** @todo update default tensorDim to be 0 and not 1 */
  if (derivative.getDim() == TensorDim()) {
    is_derivative = false;
  }

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
            if (is_derivative)
              val *= d[I + l];
            sum += val;
          }
          pp[I + j] = sum;
        }
      }
    }
  }
  return PI;
}

float ActivationLayer::sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

float ActivationLayer::sigmoidPrime(float x) {
  float sprime = sigmoid(x);
  return sprime * (1.0f - sprime);
}

float ActivationLayer::tanhFloat(float x) { return (float)tanh(x); }

float ActivationLayer::tanhPrime(float x) {
  float th = (float)tanh(x);
  return 1.0f - th * th;
}

float ActivationLayer::relu(float x) {
  if (x <= 0.0f) {
    return 0.0f;
  } else {
    return x;
  }
}

float ActivationLayer::reluPrime(float x) {
  if (x <= 0.0f) {
    return 0.0f;
  } else {
    return 1.0f;
  }
}

float ActivationLayer::no_op(float x) { return x; }

float ActivationLayer::no_op_prime(float x) { return 1.0f; }
}; // namespace nntrainer
