/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
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
#include <layer.h>
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

sharedConstTensor FlattenLayer::forwarding(sharedConstTensor in) {
  input = *in;
  hidden = input;

  /// @note in->batch can be different from input_dim.batch();
  hidden.reshape({in->batch(), output_dim.channel(), output_dim.height(),
                  output_dim.width()});

  return MAKE_SHARED_TENSOR(hidden);
}

sharedConstTensor FlattenLayer::backwarding(sharedConstTensor in,
                                            int iteration) {
  Tensor temp = *in;
  temp.reshape(input_dim);

  return MAKE_SHARED_TENSOR(std::move(temp));
}

void FlattenLayer::setProperty(const PropertyType type,
                               const std::string &value) {
  throw exception::not_supported("[Flatten Layer] setProperty not supported");
}

void FlattenLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<FlattenLayer> from =
    std::static_pointer_cast<FlattenLayer>(l);
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
}

} /* namespace nntrainer */
