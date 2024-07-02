// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   sgd.cpp
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the SGD optimizer.
 */

#include <sgd.h>

namespace nntrainer {

void SGD::applyGradient(RunOptimizerContext &context) {
  // @todo This could go inside the context.
  Tensor empty_tensor;

  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty()) {
    x_grad = context.getGradient().clone(ml::train::TensorDim::DataType::FP32);
    context.applyLossScale(x_grad);
  }

  context.applyGradient(context.getLearningRate(), x_grad);
}

} // namespace nntrainer
