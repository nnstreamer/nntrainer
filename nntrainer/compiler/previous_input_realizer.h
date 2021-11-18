// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file previous_input_realizer.h
 * @date 18 November 2021
 * @brief NNTrainer graph realizer which connects input to previous one if empty
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __PREVIOUS_INPUT_REALIZER_H__
#define __PREVIOUS_INPUT_REALIZER_H__

#include <memory>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Add default inputs if input connection is empty.
 * if a node is identified as input with @a identified_input by user or a node
 * has input_shape property, adding input behavior is skipped
 *
 */
class PreviousInputRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Previous Input Realizer object
   *
   * @param identified_input node that is identified as an input, this must not
   * connect to other nodes automatically
   */
  PreviousInputRealizer(const std::vector<std::string> &identified_input);

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~PreviousInputRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __PREVIOUS_INPUT_REALIZER_H__
