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
#ifndef __MULTIOUT_REALIZER_H__
#define __MULTIOUT_REALIZER_H__

#include <memory>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Add multiout layer when a certain input is referenced multiple times
 * @note after multiout realizer, it is guaranteed that input_layer only refers
 * to a single connection
 *
 */
class MultioutRealizer final : public GraphRealizer {
public:
  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~MultioutRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __MULTIOUT_REALIZER_H__
