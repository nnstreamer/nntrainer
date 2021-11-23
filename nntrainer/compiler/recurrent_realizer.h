// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file recurrent_realizer.h
 * @date 12 October 2021
 * @brief NNTrainer graph realizer to create unrolled graph from a graph
 * realizer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __RECURRENT_REALIZER_H__
#define __RECURRENT_REALIZER_H__

#include <realizer.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace nntrainer {

namespace props {
class UnrollFor;
class ReturnSequences;
class OutputLayer;
class RecurrentInput;
class RecurrentOutput;
} // namespace props

/**
 * @brief Recurrent Realizer which unrolls graph from given graph
 * representation
 *
 */
class RecurrentRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Recurrent Realizer object
   * @note
   * There are three types of input_layers in recurrent realizer
   * 1. input_layers: input layers
   * 2. external_input_layers: input_layers being renamed to
   * 3. recurrent_input: Override it's input layers to recurrent output for the
   * steps, where steps > 0
   *
   * @param properties
   *        unroll_for = <int> // define timestep of unrolloing
   *        return_sequences = <bool> // return sequences
   *        recurrent_inputs = <vector<std::string>> // start of the loop
   *        recurrent_ouptuts = <vector<std::string>> // end of the loop
   * @param input_layers input layer from outer side
   * @param end_layers end layers (output of the internal graph)
   */
  RecurrentRealizer(const std::vector<std::string> &properties,
                    const std::vector<std::string> &input_layers,
                    const std::vector<std::string> &end_layers);

  /**
   * @brief Construct a new Recurrent Realizer object
   *
   * @param ini ini to load recurrent properties from
   * @param external_input_layers external input layers to map input layers
   */
  RecurrentRealizer(const char *ini,
                    const std::vector<std::string> &external_input_layers);

  /**
   * @brief Destroy the Recurrent Realizer object
   *
   */
  ~RecurrentRealizer();

  /**
   * @brief realized graph
   *
   * @param reference reference to realize graph
   * @return GraphRepresentation realized graph
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

private:
  using PropTypes = std::tuple<props::RecurrentInput, props::RecurrentOutput,
                               props::ReturnSequences, props::UnrollFor>;

  std::unordered_set<std::string> input_layers; /**< external input layers */
  std::vector<std::string> end_layers;          /**< final output layers id */
  std::unique_ptr<PropTypes> recurrent_props;   /**< recurrent properties */
};

} // namespace nntrainer

#endif // __RECURRENT_REALIZER_H__
