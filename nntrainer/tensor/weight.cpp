// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	weight.cpp
 * @date	22 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Weight Class for Neural Network
 *
 */

#include <util_func.h>
#include <weight.h>

namespace nntrainer {

Weight::Weight(const Weight &rhs) :
  initializer(rhs.initializer),
  trainable(rhs.trainable),
  name(rhs.name) {
  var = rhs.var.clone();
  grad = rhs.grad.clone();
}

Weight &Weight::operator=(const Weight &rhs) {
  Weight temp(rhs);
  swap(temp, *this);
  return *this;
}

Weight::Weight(const TensorDim &dim, const WeightInitializer init, bool train,
               std::string name) :
  initializer(init),
  trainable(train),
  name(name) {
  if (initializer == WeightInitializer::WEIGHT_UNKNOWN)
    throw std::invalid_argument("Weight initializer unknown");

  initializeVar(dim);
  if (trainable) {
    grad = Tensor(dim);
    grad.setZero();
  } else
    grad = Tensor();
}

void Weight::initializeVar(const TensorDim &dim) {
  var = Tensor(dim);
  switch (initializer) {
  case WeightInitializer::WEIGHT_ZEROS:
    var.setZero();
    break;
  case WeightInitializer::WEIGHT_ONES:
    var.setValue(1.0f);
    break;
  case WeightInitializer::WEIGHT_LECUN_NORMAL:
    var.setRandNormal(0.0f, sqrtFloat(1.0f / dim.height()));
    break;
  case WeightInitializer::WEIGHT_XAVIER_NORMAL:
    var.setRandNormal(0.0f, sqrtFloat(2.0f / (dim.width() + dim.height())));
    break;
  case WeightInitializer::WEIGHT_HE_NORMAL:
    var.setRandNormal(0.0f, sqrtFloat(2.0f / (dim.height())));
    break;
  case WeightInitializer::WEIGHT_LECUN_UNIFORM:
    var.setRandUniform(-1.0f * sqrtFloat(1.0f / dim.height()),
                       sqrtFloat(1.0f / dim.height()));
    break;
  case WeightInitializer::WEIGHT_XAVIER_UNIFORM:
    var.setRandUniform(-1.0f * sqrtFloat(6.0f / (dim.height() + dim.width())),
                       sqrtFloat(6.0 / (dim.height() + dim.width())));
    break;
  case WeightInitializer::WEIGHT_HE_UNIFORM:
    var.setRandUniform(-1.0f * sqrtFloat(6.0f / (dim.height())),
                       sqrtFloat(6.0 / (dim.height())));
    break;
  default:
    break;
  }
}

} // namespace nntrainer
