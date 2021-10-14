// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file inputremap_realizer.h
 * @date 14 October 2021
 * @brief NNTrainer graph realizer which remaps input to the external graph
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __INPUTREMAP_REALIZER_H__
#define __INPUTREMAP_REALIZER_H__

#include <memory>
#include <string>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which remaps input from start -> input layers
 * @note This class find orphaned identifer in order from start_layers and
 * change the identifier to input_layers. If start_layers does not have any
 * input layers, push single input identifier, if start_layers have
 * input_layers, check if the given input layer exists starting from the first
 * input layers, if not exist, change to the given input layer in order. In case
 * of start_layer contains n input_layers to be replaced.
 *
 */
class InputRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Input Realizer object
   *
   * @param start_layers start layers
   * @param input_layers input layers
   */
  InputRealizer(const std::vector<std::string> &start_layers,
                const std::vector<std::string> &input_layers);

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
  std::vector<std::string> start_layers;
  std::vector<std::string> input_layers;
};

} // namespace nntrainer

#endif // __INPUTREMAP_REALIZER_H__
