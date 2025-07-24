// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file realizer.h
 * @date 09 October 2021
 * @brief NNTrainer graph realizer which preprocess graph representation as a
 * lowering process of compile
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __REALIZER_H__
#define __REALIZER_H__

#include <memory>
#include <vector>

#include <compiler_fwd.h>

namespace nntrainer {

/**
 * @brief Graph realizer class
 *
 */
class GraphRealizer {
public:
  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  virtual ~GraphRealizer() {}

  /**
   * @brief graph realizer creates a new graph based on the reference
   * @todo consider void GraphRepresentation &
   */
  virtual GraphRepresentation realize(const GraphRepresentation &reference) = 0;
};

} // namespace nntrainer

#endif // __REALIZER_H__
