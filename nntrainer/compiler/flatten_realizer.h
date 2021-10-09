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
#ifndef __FLATTEN_REALIZER_H__
#define __FLATTEN_REALIZER_H__

#include <memory>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class
 *
 */
class FlattenRealizer final : public GraphRealizer {
public:
  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~FlattenRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __FLATTEN_REALIZER_H__
