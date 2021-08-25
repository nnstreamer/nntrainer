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

Var_Grad::Var_Grad(const TensorDim &dim, const Tensor::Initializer init,
                   bool need_gradient, bool alloc_now,
                   const std::string &name) {
  var = std::make_shared<Tensor>(dim, alloc_now, init, name);

  std::string grad_name = name + grad_suffix;
  if (need_gradient)
    /**
     * @todo gradient initializer should be none, and then they should be set
     * zero right before using by the user itself.
     */
    grad = std::make_shared<Tensor>(dim, alloc_now, Tensor::Initializer::ZEROS,
                                    grad_name);
  else
    grad = std::make_shared<Tensor>(grad_name);
}

void Var_Grad::initializeVariable(const Tensor &preallocated) {
  if (!preallocated.empty()) {
    var = std::make_shared<Tensor>(preallocated);
    /** intentionally not initialized tensor memory for shared tensors */
  }
}

void Var_Grad::initializeGradient(const Tensor &preallocated) {
  if (!preallocated.empty()) {
    /**
     * Making a new tensor is intentional here as this tensor is not shared
     * with other layers but the internal memory is.
     */
    grad = std::make_shared<Tensor>(preallocated);
    /** intentionally not initialized tensor memory for shared tensors */
  }
  /**
   * No need to reset gradient here. With shared memory, each gradient setting
   * their own memory to zero is inefficient. Rather, the preallocated memory
   * must be created with zero initializer.
   */
}

void Var_Grad::initializeShared() { grad->makeSharedDataTensor(*var.get()); }

void Var_Grad::needsGradient(bool need_gradient) {
  if (need_gradient && grad->empty()) {
    grad =
      std::make_shared<Tensor>(var->getDim(), var->isAllocated(),
                               Tensor::Initializer::ZEROS, grad->getName());
  }
}

} // namespace nntrainer
