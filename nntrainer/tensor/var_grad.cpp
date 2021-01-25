// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	var_grad.cpp
 * @date	13 November 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Var_Grad Class for Neural Network
 *
 */

#include <util_func.h>
#include <var_grad.h>

namespace nntrainer {

Var_Grad::Var_Grad(const TensorDim &dim, bool train, const std::string &name) :
  dim(dim),
  trainable(train),
  name(name) {
  var = std::make_shared<Tensor>();
  grad = std::make_shared<Tensor>();
}

void Var_Grad::initializeWeight(const Tensor &preallocated) {
  if (!preallocated.uninitialized()) {
    var = std::make_shared<Tensor>(preallocated);
  } else {
    var = std::make_shared<Tensor>(dim);
  }
}

void Var_Grad::initializeGrad(const Tensor &preallocated, bool gtrain) {
  if (!preallocated.uninitialized() && gtrain) {
    /**
     * Making a new tensor is intentional here as this tensor is not shared
     * with other layers but the internal memory is.
     */
    grad = std::make_shared<Tensor>(preallocated);
  } else {
    grad = std::make_shared<Tensor>();
    if (trainable && gtrain) {
      grad = std::make_shared<Tensor>(dim);
    }
    resetGradient();
  }
}

void Var_Grad::initializeShared() {
  var = std::make_shared<Tensor>(dim);
  grad = std::make_shared<Tensor>(*var.get());
}

void Var_Grad::setTrainable(bool train) {
  trainable = train;
  if (trainable && grad->uninitialized()) {
    grad = std::make_shared<Tensor>(var->getDim());
  }
}

} // namespace nntrainer
