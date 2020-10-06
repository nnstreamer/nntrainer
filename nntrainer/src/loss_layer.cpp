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

#include <activation_layer.h>
#include <cmath>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <loss_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int LossLayer::initialize() {
  int status = ML_ERROR_NONE;

  output_dim = input_dim;
  return status;
}

sharedConstTensor LossLayer::forwarding(sharedConstTensor in,
                                        sharedConstTensor label) {
  input = *in;
  Tensor y2 = *label;
  Tensor y = input;
  Tensor l;

  switch (loss_type) {
  case LossType::LOSS_MSE: {
    // y2 <- y2 - y;
    Tensor residual = y2.subtract(y);
    l = residual.chain().multiply_i(residual).average().run();
  } break;
  case LossType::LOSS_ENTROPY_SIGMOID: {
    // @todo: change this to apply_i
    // @note: the output should be logit before applying sigmoid
    // log(1 + exp(-abs(y))) + max(y, 0)
    Tensor mid_term = y.apply(static_cast<float (*)(float)>(&std::fabs))
                        .multiply(-1.0)
                        .apply(static_cast<float (*)(float)>(&std::exp))
                        .add(1.0)
                        .apply(logFloat);
    mid_term = mid_term.add(y.apply(ActivationLayer::relu));

    // y * y2
    Tensor end_term = y2.chain().multiply_i(y).run();

    // loss = log(1 + exp(-abs(y))) + max(y, 0) - (y * y2)
    l = mid_term.subtract(end_term).average();
    y = y.apply(ActivationLayer::sigmoid);
  } break;
  case LossType::LOSS_ENTROPY_SOFTMAX: {
    y = y.apply(ActivationLayer::softmax);
    l = y2.chain()
          .multiply_i(y.apply(logFloat))
          .run()
          .sum_by_batch()
          .multiply(-1);

  } break;
  case LossType::LOSS_ENTROPY: {
    throw std::runtime_error(
      "Error: Cross Entropy not supported without softmax or sigmoid.");
  }
  case LossType::LOSS_UNKNOWN:
    /** intended */
  default: { throw std::runtime_error("Error: Unknown loss_type."); }
  }

  updateLoss(l);
  return MAKE_SHARED_TENSOR(std::move(y));
}

sharedConstTensor LossLayer::forwarding(sharedConstTensor in) {
  Tensor ret;

  switch (loss_type) {
  case LossType::LOSS_MSE:
    return in;
  case LossType::LOSS_ENTROPY_SIGMOID:
    ret = in->apply(ActivationLayer::sigmoid);
    return MAKE_SHARED_TENSOR(std::move(ret));
  case LossType::LOSS_ENTROPY_SOFTMAX:
    ret = in->apply(ActivationLayer::softmax);
    return MAKE_SHARED_TENSOR(std::move(ret));
  case LossType::LOSS_ENTROPY:
    throw std::runtime_error(
      "Error: Cross Entropy not supported without softmax or sigmoid.");
  case LossType::LOSS_UNKNOWN:
    /** intended */
  default:
    throw std::runtime_error("Error: Unknown loss_type.");
  }
}

void LossLayer::updateLoss(const Tensor &l) {
  float loss_sum = 0.0f;
  const float *data = l.getData();

  for (unsigned int i = 0; i < l.batch(); i++) {
    loss_sum += data[i];
  }
  loss = loss_sum / (float)l.batch();
}

void LossLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<LossLayer> from = std::static_pointer_cast<LossLayer>(l);
  this->loss_type = from->loss_type;
}

sharedConstTensor LossLayer::backwarding(sharedConstTensor derivative,
                                         int iteration) {
  Tensor ret_derivative;
  Tensor y2 = *derivative;
  Tensor y = input;

  switch (loss_type) {
  case LossType::LOSS_MSE:
    ret_derivative = y.subtract(y2).multiply(2).divide(y.getDim().getDataLen());
    break;
  case LossType::LOSS_ENTROPY_SIGMOID:
    y = y.apply(ActivationLayer::sigmoid);
    ret_derivative = y.subtract(y2).divide(y.getDim().getDataLen());
    break;
  case LossType::LOSS_ENTROPY_SOFTMAX:
    y = y.apply(ActivationLayer::softmax);
    ret_derivative = y.subtract(y2).divide(y.batch());
    break;
  case LossType::LOSS_ENTROPY:
    throw std::runtime_error(
      "Error: Cross Entropy not supported without softmax or sigmoid.");
  case LossType::LOSS_UNKNOWN:
    /** intended */
  default:
    throw std::runtime_error("Unknown loss_type.");
  }

  return MAKE_SHARED_TENSOR(std::move(ret_derivative));
}

int LossLayer::setLoss(LossType l) {
  int status = ML_ERROR_NONE;
  if (l == LossType::LOSS_UNKNOWN) {
    ml_loge("Error: Unknown loss type");
    return ML_ERROR_INVALID_PARAMETER;
  }
  loss_type = l;
  return status;
}

} /* namespace nntrainer */
