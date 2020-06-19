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

int Layer::setActivation(ActiType acti) {
  int status = ML_ERROR_NONE;
  if (acti == ACT_UNKNOWN) {
    ml_loge("Error:have to specify activation function");
    return ML_ERROR_INVALID_PARAMETER;
  }
  activation_type = acti;

  return status;
}

int Layer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  return this->opt.initialize(dim, false);
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
  if (dim.batch() == 0 || dim.width() == 0 || dim.height() == 0 ||
      dim.channel() == 0) {
    ml_loge("Error: Tensor Dimension must be set before initialization");
    return ML_ERROR_INVALID_PARAMETER;
  }
  return status;
}

Tensor Layer::initializeWeight(TensorDim w_dim, WeightIniType init_type,
                               int &status) {

  Tensor w = Tensor(w_dim);

  if (init_type == WEIGHT_UNKNOWN) {
    ml_logw("Warning: Weight Initalization Type is not set. "
            "WEIGHT_XAVIER_NORMAL is used by default");
    init_type = WEIGHT_XAVIER_NORMAL;
  }

  switch (init_type) {
  case WEIGHT_LECUN_NORMAL:
    w.setRandNormal(0, sqrt(1.0 / dim.height()));
    break;
  case WEIGHT_XAVIER_NORMAL:
    w.setRandNormal(0, sqrt(2.0 / (dim.width() + dim.height())));
    break;
  case WEIGHT_HE_NORMAL:
    w.setRandNormal(0, sqrt(2.0 / (dim.height())));
    break;
  case WEIGHT_LECUN_UNIFORM:
    w.setRandUniform(-1.0 * sqrt(1.0 / dim.height()), sqrt(1.0 / dim.height()));
    break;
  case WEIGHT_XAVIER_UNIFORM:
    w.setRandUniform(-1.0 * sqrt(6.0 / (dim.height() + dim.width())),
                     sqrt(6.0 / (dim.height() + dim.width())));
    break;
  case WEIGHT_HE_UNIFORM:
    w.setRandUniform(-1.0 * sqrt(6.0 / (dim.height())),
                     sqrt(6.0 / (dim.height())));
    break;
  default:
    break;
  }
  return w;
}

int Layer::setCost(CostType c) {
  int status = ML_ERROR_NONE;
  if (c == COST_UNKNOWN) {
    ml_loge("Error: Unknown cost fucntion");
    return ML_ERROR_INVALID_PARAMETER;
  }
  cost = c;
  return status;
}

} /* namespace nntrainer */
