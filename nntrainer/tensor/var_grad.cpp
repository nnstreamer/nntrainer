// SPDX-License-Identifier: Apache-2.0-only
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

Var_Grad::Var_Grad(const Var_Grad &rhs) :
  trainable(rhs.trainable),
  name(rhs.name) {
  var = rhs.var.clone();
  grad = rhs.grad.clone();
}

Var_Grad &Var_Grad::operator=(const Var_Grad &rhs) {
  Var_Grad temp(rhs);
  swap(temp, *this);
  return *this;
}

Var_Grad::Var_Grad(const TensorDim &dim, bool train, const std::string &name) :
  trainable(train),
  name(name) {
  var = Tensor(dim);

  grad = Tensor();
  if (trainable) {
    grad = Tensor(dim);
  }
  resetGradient();
}

} // namespace nntrainer
