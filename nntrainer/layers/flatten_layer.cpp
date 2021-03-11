// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	flatten_layer.cpp
 * @date	16 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Flatten Layer Class for Neural Network
 *
 */

#include <flatten_layer.h>
#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string FlattenLayer::type = "flatten";

int FlattenLayer::initialize(Manager &manager) {
  if (getNumInputs() != 1) {
    throw std::invalid_argument("input_shape keyword is only for one input");
  }

  TensorDim &out_dim = output_dim[0];
  int status = ML_ERROR_NONE;
  if (input_dim[0].getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  out_dim.batch(input_dim[0].batch());
  out_dim.channel(1);
  out_dim.height(1);
  out_dim.width(input_dim[0].getFeatureLen());

  return status;
}

void FlattenLayer::forwarding(bool training) {
  Tensor temp = net_input[0]->getVariableRef();
  temp.reshape(net_hidden[0]->getDim());
  net_hidden[0]->getVariableRef() = temp;
}

void FlattenLayer::calcDerivative() {
  Tensor temp = net_hidden[0]->getGradientRef();
  temp.reshape(net_input[0]->getDim());
  net_input[0]->getGradientRef() = temp;
}

} /* namespace nntrainer */
