// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file input_realizer.h
 * @date 14 October 2021
 * @brief NNTrainer graph realizer which remaps input to the external graph
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __INPUT_REALIZER_H__
#define __INPUT_REALIZER_H__

#include <memory>
#include <string>
#include <vector>

#include <connection.h>
#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which remaps input from start -> input layers
 * @note This class overwrites input conns to the location of start conns.
 * This requires each start location have the slot for input connection.
 * @note When number of input connection == 0 for a start connection of index
 * zero, this pushes input connection to the slot
 *
 */
class InputRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Input Realizer object
   *
   * @param start_conns start layers
   * @param input_conns input layers
   */
  InputRealizer(const std::vector<Connection> &start_conns,
                const std::vector<Connection> &input_conns);

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~InputRealizer();

  /**
   * @brief graph realizer creates a shallow copied graph based on the reference
   * @note input realizer resets input_layers of start_layers so that it can be
   * connected to the external network
   * @throw std::invalid_argument if graph is ill formed
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

private:
  std::vector<Connection> start_conns;
  std::vector<Connection> input_conns;
};

} // namespace nntrainer

#endif // __INPUT_REALIZER_H__
