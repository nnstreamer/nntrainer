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
#include <input_layer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <remap_realizer.h>
#include <util_func.h>
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
  /**
   * @brief Construct a new Recurrent Input object
   *
   */
  RecurrentInput();

  /**
   * @brief Construct a new Recurrent Input object
   *
   * @param name name
   */
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
  /**
   * @brief Construct a new Recurrent Output object
   *
   */
  RecurrentOutput();

  /**
   * @brief Construct a new Recurrent Output object
   *
   * @param name name
   */
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
    NNTR_THROW_IF(prop.empty(), std::invalid_argument)
      << "there is unfilled property for recurrent realizer, key: "
      << getPropKey(prop);
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
  /// @todo delegate to RecurrentRealizer(
  // const std::vector<std::string> &properties,
  // const std::vector<std::string> &external_input_layers)
  /// NYI!
}

RecurrentRealizer::~RecurrentRealizer() {}

GraphRepresentation
RecurrentRealizer::realize(const GraphRepresentation &reference) {
  auto step0_verify_and_prepare = [this, &reference]() {
    for (auto &node : reference) {
      NNTR_THROW_IF(node->getNumInputConnections() == 0, std::invalid_argument)
        << "every node must have input connection defined";

      auto &input = std::get<props::RecurrentInput>(*recurrent_props);
      if (node->getName() == input.get()) {
        NNTR_THROW_IF(node->getNumInputConnections() != 1,
                      std::invalid_argument)
          << "recurrent input must have single connection: " << input.get();
      }
    }
  };

  auto step1_connect_external_input =
    [this](const GraphRepresentation &reference_) {
      RemapRealizer input_mapper([this](std::string &id) {
        if (auto iter = id_map.find(id); iter != id_map.end()) {
          id = iter->second;
        } else {
          id += "/0";
        }
      });

      return input_mapper.realize(reference_);
    };

  auto create_step = [this](const GraphRepresentation &reference_,
                            unsigned idx) {
    GraphRepresentation step;
    auto &input = std::get<props::RecurrentInput>(*recurrent_props);
    auto &output = std::get<props::RecurrentOutput>(*recurrent_props);
    step.reserve(reference_.size());

    auto replace_idx = [](std::string &name, unsigned idx) {
      auto pos = name.find_last_of('/');
      if (pos != std::string::npos) {
        name.replace(pos + 1, std::string::npos, std::to_string(idx));
      }
    };
    for (auto &node : reference_) {
      auto new_node = node->cloneConfiguration();

      /// 1. remap identifiers to $name/$idx
      new_node->remapIdentifiers([this, idx, replace_idx](std::string &id) {
        if (auto iter = id_map.find(id); iter == id_map.end()) {
          replace_idx(id, idx);
        }
      });

      /// 2. override first output name to $name/$idx - 1
      if (node->getName() == input.get() + "/0") {
        std::string output_name = output.get() + "/" + std::to_string(idx - 1);
        new_node->setProperty({"input_layers=" + output_name});
      }

      /// 3. set shared_from
      new_node->setProperty({"shared_from=" + node->getName()});

      step.push_back(std::move(new_node));
    }
    return step;
  };

  auto step2_unroll = [this, create_step](const GraphRepresentation &reference_,
                                          unsigned unroll_for_) {
    GraphRepresentation processed(reference_.begin(), reference_.end());
    processed.reserve(reference_.size() * unroll_for_);

    for (unsigned int i = 1; i < unroll_for_; ++i) {
      auto step = create_step(reference_, i);
      processed.insert(processed.end(), step.begin(), step.end());
    }

    return processed;
  };

  auto naive_output = [this](const GraphRepresentation &reference_,
                             unsigned unroll_for) {
    /// last output's index is removed so that it can be directly an output
    auto suffix = "/" + std::to_string(unroll_for - 1);
    RemapRealizer r([suffix](std::string &name) {
      if (endswith(name, suffix)) {
        auto pos = name.find_last_of('/');
        if (pos != std::string::npos) {
          name = name.substr(0, pos);
        }
      }
    });

    return r.realize(reference_);
  };

  auto concat_output = [this](const GraphRepresentation &reference_,
                              unsigned unroll_for) {
    GraphRepresentation processed(reference_.begin(), reference_.end());
    auto output_layers =
      std::get<std::vector<props::OutputLayer>>(*recurrent_props);

    for (auto &output : output_layers) {
      std::vector<props::Name> names;
      for (unsigned int i = 0; i < unroll_for; ++i) {
        names.push_back(output.get() + "/" + std::to_string(i));
      }
      /// @todo have axis in concat layer
      auto node = createLayerNode(
        "concat", {"name=" + output.get(), "input_layers=" + to_string(names)});
      processed.push_back(std::move(node));
    }

    return processed;
  };

  auto step3_connect_output =
    [this, naive_output, concat_output](const GraphRepresentation &reference_,
                                        unsigned unroll_for) {
      bool return_sequence =
        std::get<props::ReturnSequences>(*recurrent_props).get();
      return return_sequence ? concat_output(reference_, unroll_for)
                             : naive_output(reference_, unroll_for);
    };

  /// @todo if return sequence is true, remap identifier and concat output
  /// layers else remap last loop identifier

  auto unroll_for = std::get<props::UnrollFor>(*recurrent_props).get();
  step0_verify_and_prepare();
  auto processed = step1_connect_external_input(reference);
  processed = step2_unroll(processed, unroll_for);
  return step3_connect_output(processed, unroll_for);
}

} // namespace nntrainer
