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

int FlattenLayer::initialize() {
  int status = ML_ERROR_NONE;
  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  output_dim.batch(input_dim.batch());
  output_dim.channel(1);
  output_dim.height(1);
  output_dim.width(input_dim.getFeatureLen());

  return status;
}

sharedConstTensors FlattenLayer::forwarding(sharedConstTensors in) {
  input = *in[0];
  hidden = input;

  hidden.reshape(output_dim);

  return {MAKE_SHARED_TENSOR(hidden)};
}

sharedConstTensors FlattenLayer::backwarding(sharedConstTensors in,
                                             int iteration) {
  Tensor temp = *in[0];
  temp.reshape(input_dim);

  return {MAKE_SHARED_TENSOR(std::move(temp))};
}

} /* namespace nntrainer */
