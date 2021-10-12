// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file remap_realizer.h
 * @date 12 October 2021
 * @brief NNTrainer graph realizer which realizes identifer to a new identifier
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __REMAP_REALIZER_H__
#define __REMAP_REALIZER_H__

#include <functional>
#include <memory>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which remaps identifiers inside the graph
 * representation, remap_function will be applied for all the layers identifier
 * visible
 *
 */
class RemapRealizer final : public GraphRealizer {
public:
  RemapRealizer(std::function<void(std::string &)> remap_function);
  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~RemapRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

private:
  std::function<void(std::string &)> remap_fn;
};

} // namespace nntrainer

#endif // __REMAP_REALIZER_H__
