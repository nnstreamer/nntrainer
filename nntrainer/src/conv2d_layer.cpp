/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	conv2d_layer.h
 * @date	02 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */

#include <conv2d_layer.h>
#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <random>
#include <util_func.h>

namespace nntrainer {

int Conv2DLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  // NYI
  return status;
}

void Conv2DLayer::read(std::ifstream &file) {
  // NYI
}

void Conv2DLayer::save(std::ofstream &file) {
  // NYI
}

Tensor Conv2DLayer::forwarding(Tensor in, int &status) {
  // NYI
  return in;
};

Tensor Conv2DLayer::backwarding(Tensor in, int iteration) {
  // NYI
  return in;
}

void Conv2DLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<Conv2DLayer> from = std::static_pointer_cast<Conv2DLayer>(l);
  this->filter_size = from->filter_size;
  for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
    this->kernel_size[i] = from->kernel_size[i];
    this->stride[i] = from->stride[i];
  }
  this->padding = from->padding;

  for (int i = 0; from->filters.size(); ++i) {
    this->filters.push_back(from->filters[i]);
    this->bias.push_back(from->bias[i]);
  }
}

int Conv2DLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;
  // NYI
  return status;
}

} /* namespace nntrainer */
