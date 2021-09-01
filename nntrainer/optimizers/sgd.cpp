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

SGD::SGD() { setProperty({"learning_rate=0.0001"}); }

void SGD::applyGradient(RunOptimizerContext &context) {
  context.applyGradient(getLearningRate(context.getIteration()));
}

} // namespace nntrainer
