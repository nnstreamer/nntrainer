/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <activation_layer.h>
#include <fstream>
#include <iostream>
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <parse_util.h>
#include <tensor.h>
#include <util_func.h>
#include <vector>

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
    this->setActivation(no_op, no_op);
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
  int batch = t.getBatch();
  int channel = t.getChannel();
  int height = t.getHeight();
  int width = t.getWidth();
  float *dp;
  float *rp;
  const float *tp;

  Tensor result(t.getDim());
  Tensor divisor(t.getWidth());

  dp = divisor.getData();
  rp = result.getData();
  tp = t.getData();

  divisor.setZero();

  for (int k = 0; k < batch; k++) {
    for (int c = 0; c < channel; c++) {
      for (int i = 0; i < height; i++) {
        int index =
          k * channel * height * width + c * height * width + i * width;
        float m = std::numeric_limits<float>::lowest();

        // find max
        for (int j = 0; j < width; j++) {
          if (tp[index + j] > m)
            m = tp[index + j];
        }

        // shiftx
        float sum = 0.0f;
        for (int j = 0; j < width; j++) {
          dp[j] = exp(tp[index + j] - m);
          sum += dp[j];
        }

        for (int j = 0; j < width; j++) {
          rp[index + j] = dp[j] / sum;
        }
      }
    }
  }
  return result;
}

Tensor ActivationLayer::softmaxPrime(Tensor const &x,
                                     Tensor const &derivative) {
  int batch = x.getBatch();
  int channel = x.getChannel();
  int width = x.getWidth();
  int height = x.getHeight();
  bool is_derivative = true;

  Tensor PI = Tensor(x.getDim());

  const float *xp = x.getData();
  const float *d = derivative.getData();
  float *pp = PI.getData();

  /** @todo update default tensorDim to be 0 and not 1 */
  if (derivative.getDim() == TensorDim()) {
    is_derivative = false;
  }

  for (int k = 0; k < batch; ++k) {
    int K = k * channel * height * width;
    for (int c = 0; c < channel; ++c) {
      int C = K + c * height * width;
      for (int i = 0; i < height; ++i) {
        int I = C + i * width;
        for (int j = 0; j < width; ++j) {
          float sum = 0.0f;
          for (int l = 0; l < width; ++l) {
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
}; // namespace nntrainer
