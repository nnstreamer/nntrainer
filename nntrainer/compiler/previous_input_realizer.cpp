// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file previous_input_realizer.cpp
 * @date 18 November 2021
 * @brief NNTrainer graph realizer which connects input to previous one if empty
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <compiler_fwd.h>
#include <memory>
#include <vector>

#include <previous_input_realizer.h>

namespace nntrainer {

PreviousInputRealizer::PreviousInputRealizer(
  const std::vector<std::string> &identified_input) {}

PreviousInputRealizer::~PreviousInputRealizer() {}

GraphRepresentation
PreviousInputRealizer::realize(const GraphRepresentation &reference) {
  return GraphRepresentation();
}

} // namespace nntrainer
