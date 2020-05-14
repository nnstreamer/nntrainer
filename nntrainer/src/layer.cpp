/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	layer.cpp
 * @date	04 December 2019
 * @brief	This is Layers Classes for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

Layer::Layer() {
  type = LAYER_UNKNOWN;
  activation_type = ACT_UNKNOWN;
  last_layer = false;
  init_zero = false;
  activation = NULL;
  activation_prime = NULL;
  bn_fallow = false;
  weight_decay.type = WeightDecayType::unknown;
  weight_decay.lambda = 0.0;
  weight_ini_type = WEIGHT_UNKNOWN;
}

int Layer::setActivation(ActiType acti) {
  int status = ML_ERROR_NONE;
  if (acti == ACT_UNKNOWN) {
    ml_loge("Error:have to specify activation function");
    return ML_ERROR_INVALID_PARAMETER;
  }
  activation_type = acti;
  switch (acti) {
  case ACT_TANH:
    activation = tanhFloat;
    activation_prime = tanhPrime;
    break;
  case ACT_SIGMOID:
    activation = sigmoid;
    activation_prime = sigmoidePrime;
    break;
  case ACT_RELU:
    activation = relu;
    activation_prime = reluPrime;
    break;
  default:
    break;
  }
  return status;
}

int Layer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  return this->opt.initialize(dim.height(), dim.width(), true);
}

int Layer::checkValidation() {
  int status = ML_ERROR_NONE;
  if (type == LAYER_UNKNOWN) {
    ml_loge("Error: Layer type is unknown");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (activation_type == ACT_UNKNOWN) {
    ml_loge("Error: Have to set activation for this layer");
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (dim.batch() == 0 || dim.width() == 0 || dim.height() == 0) {
    ml_loge("Error: Tensor Dimension must be set before initialization");
    return ML_ERROR_INVALID_PARAMETER;
  }
  return status;
}
} /* namespace nntrainer */
