// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.cpp
 * @date   11 December 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#include <layer.h>
#include <tensor.h>
#include <tuple>
#include <var_grad.h>

namespace ml {
namespace train {

Tensor::Tensor(const TensorDim &dim, const iTensor::Initializer init, bool ng,
               std::string name) :
  Var_Grad(dim, init, ng, false, name),
  src_layer(nullptr) {}

} // namespace train
} // namespace ml
