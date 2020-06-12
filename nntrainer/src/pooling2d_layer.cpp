/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	pooling2d_layer.h
 * @date	12 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is 2 Dimensional Pooling Layer Class for Neural Network
 *
 */

#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <pooling2d_layer.h>
#include <util_func.h>

namespace nntrainer {

int Pooling2DLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  // NYI
  return status;
}

Tensor Pooling2DLayer::forwarding(Tensor in, int &status) {
  // NYI
  return in;
}

Tensor Pooling2DLayer::forwarding(Tensor in, Tensor output, int &status) {
  return forwarding(in, status);
}

Tensor Pooling2DLayer::backwarding(Tensor in, int iteration) {
  // NYI
  return in;
}

void Pooling2DLayer::copy(std::shared_ptr<Layer> l) {
  // NYI
}

int Pooling2DLayer::setSize(int *size,
                            nntrainer::Pooling2DLayer::PropertyType type) {
  int status = ML_ERROR_NONE;
  // NYI
  return status;
}

int Pooling2DLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;
  // NYI
  return status;
}

Tensor Pooling2DLayer::zero_pad(int batch, Tensor in,
                                unsigned int const *padding) { // NYI
  return in;
}

Tensor Pooling2DLayer::pooling2d(Tensor in, int &status) {
  // NYI
  return in;
}

} /* namespace nntrainer */
