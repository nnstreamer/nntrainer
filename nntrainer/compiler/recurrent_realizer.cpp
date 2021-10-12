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

#include <recurrent_realizer.h>

#include <common_properties.h>

#include <nntrainer_error.h>
#include <node_exporter.h>

namespace nntrainer {

namespace props {

/**
 * @brief Property check unroll_for
 *
 */
class UnrollFor final : public PositiveIntegerProperty {
public:
  UnrollFor(const unsigned &value = 1);
  static constexpr const char *key = "unroll_for";
  using prop_tag = uint_prop_tag;
};

UnrollFor::UnrollFor(const unsigned &value) { set(value); }

/**
 * @brief Property for recurrent inputs
 *
 */
class RecurrentInput final : public Name {
public:
  RecurrentInput();
  RecurrentInput(const std::string &name);
  static constexpr const char *key = "recurrent_input";
  using prop_tag = str_prop_tag;
};

RecurrentInput::RecurrentInput() {}
RecurrentInput::RecurrentInput(const std::string &name) { set(name); };

/**
 * @brief Property for recurrent outputs
 *
 */
class RecurrentOutput final : public Name {
public:
  RecurrentOutput();
  RecurrentOutput(const std::string &name);
  static constexpr const char *key = "recurrent_output";
  using prop_tag = str_prop_tag;
};

RecurrentOutput::RecurrentOutput() {}
RecurrentOutput::RecurrentOutput(const std::string &name) { set(name); };
} // namespace props

RecurrentRealizer::RecurrentRealizer(
  const std::vector<std::string> &properties,
  const std::vector<std::string> &external_input_layers) :
  recurrent_props(new PropTypes({}, {}, {}, {}, props::ReturnSequences(false),
                                props::UnrollFor(1))) {
  auto left = loadProperties(properties, *recurrent_props);

  auto throw_if_empty = [](auto &&prop) {
    if (prop.empty()) {
      throw std::invalid_argument(
        "there is unfilled property for recurrent realizer, key: " +
        std::string(getPropKey(prop)));
    }
  };

  throw_if_empty(std::get<0>(*recurrent_props));
  throw_if_empty(std::get<1>(*recurrent_props));
  throw_if_empty(std::get<2>(*recurrent_props));
  throw_if_empty(std::get<3>(*recurrent_props));
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There is unparesed properties";

  auto &input_layers =
    std::get<std::vector<props::InputLayer>>(*recurrent_props);
  auto external_layers = std::vector<props::Name>(external_input_layers.begin(),
                                                  external_input_layers.end());
  NNTR_THROW_IF(input_layers.size() != external_layers.size(),
                std::invalid_argument)
    << "input_layers and external input_layers size does not match: "
    << to_string(input_layers) << " vs " << to_string(external_layers);

  std::transform(input_layers.begin(), input_layers.end(),
                 external_layers.begin(), std::inserter(id_map, id_map.end()),
                 [](const std::string &key, const std::string &val) {
                   return std::pair<std::string, std::string>(key, val);
                 });
}

RecurrentRealizer::RecurrentRealizer(
  const char *ini_path, const std::vector<std::string> &external_input_layers) {
  /// NYI!
}

RecurrentRealizer::~RecurrentRealizer() {}

GraphRepresentation
RecurrentRealizer::realize(const GraphRepresentation &reference) {
  /// @todo remap identifier input_layers -> external_input_layers

  /// @todo copy the layers to loop and remap with numbers
  ///       1. define layer node copy in this context
  ///       2. copy and remap layers to be looped

  /// @todo if return sequence is true, remap identifier and concat output
  /// layers

  /// NYI!
  return reference;
}

} // namespace nntrainer
