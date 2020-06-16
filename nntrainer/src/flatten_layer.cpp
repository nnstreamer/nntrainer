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

int FlattenLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }
  if (input_dim.batch() <= 0 || input_dim.height() <= 0 ||
      input_dim.width() <= 0 || input_dim.channel() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->last_layer = last;

  output_dim.batch(input_dim.batch());
  output_dim.channel(1);
  output_dim.height(1);
  output_dim.width(input_dim.getFeatureLen());

  hidden = Tensor(output_dim);

  return status;
}

Tensor FlattenLayer::forwarding(Tensor in, int &status) {
  memcpy(hidden.getData(), in.getData(),
         in.getDim().getDataLen() * sizeof(float));

  return hidden;
}

Tensor FlattenLayer::forwarding(Tensor in, Tensor output, int &status) {
  return forwarding(in, status);
}

Tensor FlattenLayer::backwarding(Tensor in, int iteration) {
  // NYI
  return in;
}

void FlattenLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<FlattenLayer> from =
    std::static_pointer_cast<FlattenLayer>(l);
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->dim = from->dim;
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
  this->last_layer = from->last_layer;
}

} /* namespace nntrainer */
