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
  // NYI
  return status;
}

Tensor FlattenLayer::forwarding(Tensor in, int &status) {
  // NYI
  return in;
}

Tensor FlattenLayer::forwarding(Tensor in, Tensor output, int &status) {
  return forwarding(in, status);
}

Tensor FlattenLayer::backwarding(Tensor in, int iteration) {
  // NYI
  return in;
}

void FlattenLayer::copy(std::shared_ptr<Layer> l) {
  // NYI
}

} /* namespace nntrainer */
