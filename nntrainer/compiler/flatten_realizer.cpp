// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file flatten_realizer.h
 * @date 09 October 2021
 * @brief NNTrainer graph realizer which realizes flatten=true to actual node
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <flatten_realizer.h>

#include <layer_node.h>
#include <flatten_layer.h>

namespace nntrainer {

FlattenRealizer::~FlattenRealizer() {}

GraphRepresentation
FlattenRealizer::realize(const GraphRepresentation &reference) {
  /// NYI
  return reference;
}
} // namespace nntrainer
