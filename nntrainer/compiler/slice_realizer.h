// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file slice_realizer.h
 * @date 14 October 2021
 * @brief NNTrainer graph realizer which slice the graph representation
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __SLICE_REALIZER_H__
#define __SLICE_REALIZER_H__

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <realizer.h>

namespace nntrainer {

class Connection;

/**
 * @brief Graph realizer class which slice graph representation
 *
 */
class SliceRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Slice Realizer object
   *
   * @param start_connections start layers
   * @param end_connections end layers
   */
  SliceRealizer(const std::vector<Connection> &start_connections,
                const std::vector<Connection> &end_connections);

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~SliceRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   * @note for each layer in start_layers, start dfs, if traversal meets an end
   * layers, node is added to an ordered set.
   * @throw std::invalid_argument if created GraphRepresentation is empty
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

private:
  std::vector<std::string> start_layers;
  std::unordered_set<std::string> end_layers;
};

} // namespace nntrainer

#endif // __SLICE_REALIZER_H__
