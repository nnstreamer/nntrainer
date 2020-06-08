/* SPDX-License-Identifier: Apache-2.0-only
 *
 * Copyright (C) 2020 Jihoon Lee <jihoon.it.lee@samsung.com>
 *
 * @file	lazy_tensor.cpp
 * @date	05 Jun 2020
 * @brief	A lazy evaluation calculator for tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jihoon Lee <jihoon.it.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief Wrapper method of add_i (immediate version of add)
 * @retval this
 */
LazyTensor LazyTensor::add_i(float const &value) {
  call_chain.push_back([value](Tensor &t) mutable -> int {
    t.add_i(value);
    return ML_ERROR_NONE;
  });
  return *this;
}

/**
 * @brief execute the call_chain to evaluate
 * @retval calculated tensor
 */
Tensor &LazyTensor::run() {
  int status;
  for (auto item : call_chain) {
    status = item(target);
    if (status != ML_ERROR_NONE) {
      throw std::runtime_error("Error: evaluation failed");
    }
  }
  return target;
}

} /* namespace nntrainer */
