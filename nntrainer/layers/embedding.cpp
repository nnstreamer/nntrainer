// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	embedding.cpp
 * @date	04 March 2021
 * @brief	This is Embedding Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <embedding.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string EmbeddingLayer::type = "embedding";

int EmbeddingLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;

  // NYI

  return status;
}

void EmbeddingLayer::setProperty(const PropertyType type,
                                 const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::in_dim: {
    if (!value.empty()) {
      status = setUint(in_dim, value);
      throw_status(status);
      input_dim[0].width(in_dim);
    }
  } break;
  case PropertyType::out_dim: {
    if (!value.empty()) {
      status = setUint(out_dim, value);
      throw_status(status);
      output_dim[0].width(out_dim);
    }
  } break;
  case PropertyType::in_length: {
    if (!value.empty()) {
      status = setUint(in_length, value);
      throw_status(status);
      output_dim[0].height(in_length);
      input_dim[0].height(in_length);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void EmbeddingLayer::forwarding(bool training) {
  // NYI
}

void EmbeddingLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<EmbeddingLayer> from =
    std::static_pointer_cast<EmbeddingLayer>(l);
  this->in_dim = from->in_dim;
  this->out_dim = from->out_dim;
  this->in_length = from->in_length;
}

void EmbeddingLayer::calcDerivative() {
  // NYI
}
void EmbeddingLayer::calcGradient() {
  // NYI
}

} // namespace nntrainer
