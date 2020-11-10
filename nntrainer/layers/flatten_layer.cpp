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

int FlattenLayer::initialize() {
  if (num_inputs != 1) {
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

sharedConstTensors FlattenLayer::forwarding(sharedConstTensors in) {
  input = *in[0];
  hidden = input;

  hidden.reshape(output_dim[0]);

  return {MAKE_SHARED_TENSOR(hidden)};
}

sharedConstTensors FlattenLayer::backwarding(sharedConstTensors in,
                                             int iteration) {
  Tensor temp = *in[0];
  temp.reshape(input_dim[0]);

  return {MAKE_SHARED_TENSOR(std::move(temp))};
}

} /* namespace nntrainer */
