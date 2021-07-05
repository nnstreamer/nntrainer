// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dropout_layer.cpp
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Dropout Layer Class for Neural Network
 *
 */

#include <dropout.h>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int DropOutLayer::initialize(Manager &manager) {
  output_dim = input_dim;
  // TODO : asking Tensor to manager
  for (auto &t : input_dim) {
    mask.push_back(std::make_shared<Tensor>(t, true));
  }

  return ML_ERROR_NONE;
}

void DropOutLayer::forwarding(bool training) {
  auto &rate_ = std::get<props::DropOutSpec>(dropout_rate).get();
  // Assume it is in-place calculation. It means input and output share mem
  // buffer. So if the training is false, the output is the same with input. In
  // orther words, there is nothing happen during inference.

  if (training && rate_ > 0.0) {
    for (unsigned int i = 0; i < input_dim.size(); ++i) {
      Tensor &input = net_input[i]->getVariableRef();
      Tensor &mask_ = *mask[i].get();

      mask_ = input.dropout_mask(rate_);
      input.multiply_i(mask_);
    }
  }
}

void DropOutLayer::calcDerivative() {
  // Assume it is in-place calculation
  auto &rate_ = std::get<props::DropOutSpec>(dropout_rate).get();
  if (rate_ > 0.0) {
    for (unsigned int i = 0; i < input_dim.size(); ++i) {
      Tensor deriv = net_hidden[i]->getGradientRef();
      deriv.multiply_i(*mask[i].get());
    }
  }
}

int DropOutLayer::setProperty(std::vector<std::string> values) {
  try {
    values = loadProperties(values, dropout_rate);
  } catch (std::invalid_argument &e) {
    ml_loge("parsing property failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return LayerV1::setProperty(values);
}
} /* namespace nntrainer */
