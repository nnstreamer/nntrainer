/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	loss_layer.cpp
 * @date	12 June 2020
 * @brief	This is Loss Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <layer.h>
#include <lazy_tensor.h>
#include <loss_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int LossLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;

  if (!last) {
    ml_loge("Error: Loss layer, if exists, must be the layer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->last_layer = last;
  output_dim = input_dim;
  return status;
}

Tensor LossLayer::forwarding(Tensor output, Tensor label, int &status) {
  input = output;
  Tensor y2 = label;
  Tensor y = output;
  Tensor l;

  switch (cost) {
  case COST_MSR: {
    // y2 <- y2 - y;
    y2.subtract_i(y);

    l = y2.chain().multiply_i(y2).average().run();
  } break;
  case COST_ENTROPY_SIGMOID: {
    // @todo: change this to apply_i
    // @note: the output should be logit before applying sigmoid
    // log(1 + exp(-abs(y))) + max(y, 0)
    Tensor mid_term = y.apply(static_cast<float (*)(float)>(&std::fabs))
                        .multiply(-1.0)
                        .apply(static_cast<float (*)(float)>(&std::exp))
                        .add(1.0)
                        .apply(logFloat);
    mid_term = mid_term.add(y.apply(relu));

    // y * y2
    Tensor end_term = y2.chain().multiply_i(y).run();

    // loss = log(1 + exp(-abs(y))) + max(y, 0) - (y * y2)
    l = mid_term.subtract(end_term).average();
    y = y.apply(sigmoid);
  } break;
  case COST_ENTROPY_SOFTMAX: {
    y = y.apply(softmax);
    l = y2.chain().multiply_i(y.apply(logFloat)).run().sum_by_batch();

  } break;
  case COST_ENTROPY: {
    status = ML_ERROR_NOT_SUPPORTED;
    ml_loge("Error: Cross Entropy not supported without softmax or sigmoid.");
    return y;
  }
  case COST_UNKNOWN:
    /** intended */
  default: {
    status = ML_ERROR_NOT_SUPPORTED;
    ml_loge("Error: Unknown cost.");
    return y;
  }
  }

  updateLoss(l);
  status = ML_ERROR_NONE;
  return y;
}

void LossLayer::updateLoss(const Tensor &l) {
  float loss_sum = 0.0;
  const float *data = l.getData();

  for (int i = 0; i < l.getBatch(); i++) {
    loss_sum += data[i];
  }
  loss = loss_sum / (float)l.getBatch();
}

void LossLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<LossLayer> from = std::static_pointer_cast<LossLayer>(l);
  this->last_layer = from->last_layer;
  this->input.copy(from->input);
  this->cost = from->cost;
  this->loss = from->loss;
}

Tensor LossLayer::backwarding(Tensor derivative, int iteration) {
  Tensor ret_derivative;
  Tensor y2 = derivative;
  Tensor y = input;

  switch (cost) {
  case COST_MSR:
    ret_derivative = y.subtract(y2);
    break;
  case COST_ENTROPY_SIGMOID:
    y = y.apply(sigmoid);
    ret_derivative = y.subtract(y2).multiply(1.0 / y.getWidth());
    break;
  case COST_ENTROPY_SOFTMAX:
    y = y.apply(softmax);
    ret_derivative = y.subtract(y2).multiply(1.0 / y.getWidth());
    break;
  case COST_ENTROPY:
    throw std::runtime_error(
      "Error: Cross Entropy not supported without softmax or sigmoid.");
  case COST_UNKNOWN:
    /** intended */
  default:
    throw std::runtime_error("Unknown cost.");
  }

  return ret_derivative;
}

Tensor LossLayer::forwarding(Tensor in, int &status) {
  status = ML_ERROR_NOT_SUPPORTED;
  return in;
}

void LossLayer::setProperty(const PropertyType type, const std::string &value) {
  throw std::invalid_argument("[Loss Layer] setProperty not supported");
}

} /* namespace nntrainer */
