// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   var_grad.cpp
 * @date   13 November 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Var_Grad Class for Neural Network
 *
 */

#include <util_func.h>
#include <var_grad.h>

#include <nntrainer_error.h>

namespace nntrainer {

Var_Grad::Var_Grad(const TensorDim &dim, bool train, bool alloc_now_,
                   const std::string &name) :
  dim(dim),
  trainable(train),
  alloc_now(alloc_now_),
  name(name) {
  var = std::make_shared<Tensor>(dim, alloc_now);
  if (trainable)
    grad = std::make_shared<Tensor>(dim, alloc_now);
  else
    grad = std::make_shared<Tensor>();
}

void Var_Grad::initializeVariable(const Tensor &preallocated) {
  if (!preallocated.uninitialized()) {
    var->makeSharedDataTensor(preallocated);
  }
}

void Var_Grad::initializeGradient(const Tensor &preallocated) {
  if (!preallocated.uninitialized()) {
    /**
     * Making a new tensor is intentional here as this tensor is not shared
     * with other layers but the internal memory is.
     */
    grad->makeSharedDataTensor(preallocated);
  }
  if (alloc_now)
    resetGradient();
}

void Var_Grad::initializeShared() { grad->makeSharedDataTensor(*var.get()); }

void Var_Grad::setTrainable(bool train) {
  trainable = train;
  if (trainable && grad->uninitialized()) {
    bool alloc_now_ = var->isAllocated();
    grad = std::make_shared<Tensor>(var->getDim(), alloc_now_);
  }
}

Var_Grad
Var_Grad::cloneTransposeVariableOnly(const std::string &direction) const {
  Var_Grad vg(*this);
  /// @todo: make this clonable even when not allocated
  NNTR_THROW_IF(var->isAllocated() == false, std::invalid_argument)
    << "transpose clone is only allowed when variable is allocated, var name: "
    << getName();

  vg.var = std::make_shared<Tensor>(var->transpose(direction));
  vg.dim = vg.var->getDim();
  vg.grad.reset();

  return vg;
}

} // namespace nntrainer
