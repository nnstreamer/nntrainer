// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file activation_realizer.h
 * @date 23 November 2021
 * @brief NNTrainer graph realizer which realizes activation!=none to actual
 * activation node
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __ACTIVATION_REALIZER_H__
#define __ACTIVATION_REALIZER_H__

#include <memory>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer which realizes activation
 *
 */
class ActivationRealizer final : public GraphRealizer {
public:
  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~ActivationRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __ACTIVATION_REALIZER_H__
