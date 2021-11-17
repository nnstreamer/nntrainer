// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file multiout_realizer.h
 * @date 17 November 2021
 * @brief NNTrainer graph realizer which realizes multiout to actual node
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <compiler_fwd.h>
#include <multiout_realizer.h>

namespace nntrainer {
MultioutRealizer::~MultioutRealizer() {}

GraphRepresentation
MultioutRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation ret;
  return GraphRepresentation();
}

} // namespace nntrainer
