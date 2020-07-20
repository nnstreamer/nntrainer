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
 * @param[in] last last layer
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int ActivationLayer::initialize(bool last) {

  this->last_layer = last;

  dim = input_dim;
  output_dim = dim;

  return ML_ERROR_NONE;
}

Tensor ActivationLayer::forwarding(Tensor in, int &status) {
  status = ML_ERROR_NONE;

  input = in;
  hidden = _act_fn(in);

  return hidden;
}

Tensor ActivationLayer::backwarding(Tensor derivative, int iteration) {
  if (activation_type == ActiType::ACT_SOFTMAX) {
    /**
     * This is a hotfix for now
     * @todo Update activationLayer semantics usages to take another argument
     */
    Tensor x = hidden;
    int batch = x.getBatch();
    int channel = x.getChannel();
    int width = x.getWidth();
    int height = x.getHeight();

    Tensor PI = Tensor(x.getDim());

    float *xp = x.getData();
    float *pp = PI.getData();
    float *d = derivative.getData();

    for (int k = 0; k < batch; ++k) {
      int K = k * channel * height * width;
      for (int c = 0; c < channel; ++c) {
        int C = K + c * height * width;
        for (int i = 0; i < height; ++i) {
          int I = C + i * width;
          for (int j = 0; j < width; ++j) {
            float sum = 0.0;
            for (int l = 0; l < width; ++l) {
              if (j == l) {
                sum += d[I + l] * xp[I + l] * (1.0 - xp[I + j]);
              } else {
                sum -= d[I + l] * xp[I + l] * xp[I + j];
              }
            }
            pp[I + j] = sum;
          }
        }
      }
    }
    return PI;
  } else
    return derivative.multiply(_act_prime_fn(input));
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
  std::function<Tensor(Tensor const &)> const &activation_prime_fn) {
  _act_fn = activation_fn;
  _act_prime_fn = activation_prime_fn;

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<float(float const)> const &activation_fn,
  std::function<float(float const)> const &activation_prime_fn) {
  _act_fn = [activation_fn](Tensor const &t) { return t.apply(activation_fn); };
  _act_prime_fn = [activation_prime_fn](Tensor const &t) {
    return t.apply(activation_prime_fn);
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
    this->setActivation(sigmoid, sigmoidePrime);
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

Tensor ActivationLayer::softmaxPrime(Tensor const &x) {
  /** @todo Fix this to use derivative */
  int batch = x.getBatch();
  int channel = x.getChannel();
  int width = x.getWidth();
  int height = x.getHeight();

  Tensor PI = Tensor(x.getDim());

  const float *xp = x.getData();
  float *pp = PI.getData();

  for (int k = 0; k < batch; ++k) {
    int K = k * channel * height * width;
    for (int c = 0; c < channel; ++c) {
      int C = K + c * height * width;
      for (int i = 0; i < height; ++i) {
        int I = C + i * width;
        for (int j = 0; j < width; ++j) {
          float sum = 0.0;
          for (int l = 0; l < width; ++l) {
            if (j == l) {
              sum += xp[I + l] * (1.0 - xp[I + j]);
            } else {
              sum += xp[I + l] * xp[I + j] * -1.0;
            }
          }
          pp[I + j] = sum;
        }
      }
    }
  }
  return PI;
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
  Tensor divisor(t.getDim());

  dp = divisor.getData();
  rp = result.getData();
  tp = t.getData();

  divisor.setZero();

  for (int k = 0; k < batch; k++) {
    int index = k * channel * height * width;
    float m = std::numeric_limits<float>::lowest();
    // find max
    for (int c = 0; c < channel; c++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          if (tp[index + i * width + j] > m)
            m = tp[index + c * height * width + i * width + j];
        }
      }
    }

    // shiftx
    float sum = 0.0;
    for (int c = 0; c < channel; c++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          dp[index + width * i + j] = exp(tp[index + i * width + j] - m);
          sum += dp[index + c * height * width + width * i + j];
        }
      }
    }

    for (int c = 0; c < channel; c++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          rp[index + c * height * width + width * i + j] =
            dp[index + c * height * width + width * i + j] / sum;
        }
      }
    }
  }
  return result;
}

float ActivationLayer::sigmoid(float x) { return 1 / (1 + exp(-x)); }

float ActivationLayer::sigmoidePrime(float x) {
  float sprime = sigmoid(x);
  return sprime * (1 - sprime);
}

float ActivationLayer::tanhFloat(float x) { return (float)tanh(x); }

float ActivationLayer::tanhPrime(float x) {
  float th = (float)tanh(x);
  return 1.0 - th * th;
}

float ActivationLayer::relu(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return x;
  }
}

float ActivationLayer::reluPrime(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return 1.0;
  }
}

float ActivationLayer::no_op(float x) { return x; }
}; // namespace nntrainer
