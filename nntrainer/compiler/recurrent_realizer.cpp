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
#include <grucell.h>
#include <input_layer.h>
#include <layer_node.h>
#include <lstm.h>
#include <lstmcell.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <remap_realizer.h>
#include <rnncell.h>
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
  const std::vector<std::string> &input_layers,
  const std::vector<std::string> &end_layers) :
  input_layers(input_layers.begin(), input_layers.end()),
  end_layers(end_layers),
  recurrent_props(
    new PropTypes(props::RecurrentInput(), props::RecurrentOutput(),
                  props::ReturnSequences(false), props::UnrollFor(1))) {
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
}

/**
 * @brief if node is of recurrent type, set time step and max time step
 *
 * @param node node
 * @param time_step time step
 * @param max_time_step max time step
 */
static void propagateTimestep(LayerNode *node, unsigned int time_step,
                              unsigned int max_time_step) {

  /** @todo add an interface to check if a layer supports a property */
  auto is_recurrent_type = [](LayerNode *node) {
    return node->getType() == RNNCellLayer::type ||
           node->getType() == LSTMLayer::type ||
           node->getType() == LSTMCellLayer::type ||
           node->getType() == GRUCellLayer::type;
  };

  if (is_recurrent_type(node)) {
    node->setProperty({"max_timestep=" + std::to_string(max_time_step),
                       "timestep=" + std::to_string(time_step)});
  }

  return;
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
      auto &input = std::get<props::RecurrentInput>(*recurrent_props);
      if (node->getName() == input.get()) {
        NNTR_THROW_IF(node->getNumInputConnections() != 1,
                      std::invalid_argument)
          << "recurrent input must have single connection: " << input.get();
      }
    }
  };

  /**
   * @brief maps input place holder to given name otherwise scopped to suffix
   * "/0"
   *
   */
  auto step1_connect_external_input =
    [this](const GraphRepresentation &reference_, unsigned max_idx) {
      RemapRealizer input_mapper([this](std::string &id) {
        if (input_layers.count(id) == 0) {
          id += "/0";
        }
      });

      auto nodes = input_mapper.realize(reference_);
      for (auto &node : nodes) {
        propagateTimestep(node.get(), 0, max_idx);
      }

      return nodes;
    };

  /**
   * @brief Create a single time step. Used inside step2_unroll.
   *
   */
  auto create_step = [this](const GraphRepresentation &reference_, unsigned idx,
                            unsigned max_idx) {
    GraphRepresentation step;
    auto &input = std::get<props::RecurrentInput>(*recurrent_props);
    auto &output = std::get<props::RecurrentOutput>(*recurrent_props);
    step.reserve(reference_.size());

    auto replace_idx = [this](std::string &name, unsigned idx) {
      auto pos = name.find_last_of('/');
      if (pos != std::string::npos && input_layers.count(name) == 0) {
        name.replace(pos + 1, std::string::npos, std::to_string(idx));
      }
    };
    for (auto &node : reference_) {
      auto new_node = node->cloneConfiguration();

      /// 1. remap identifiers to $name/$idx
      new_node->remapIdentifiers([this, idx, replace_idx](std::string &id) {
        if (input_layers.count(id) == 0) {
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
      /// 4. if recurrent layer type set timestep property
      propagateTimestep(new_node.get(), idx, max_idx);

      step.push_back(std::move(new_node));
    }
    return step;
  };

  /**
   * @brief unroll the graph by calling create_step()
   *
   */
  auto step2_unroll = [this, create_step](const GraphRepresentation &reference_,
                                          unsigned unroll_for_) {
    GraphRepresentation processed(reference_.begin(), reference_.end());
    processed.reserve(reference_.size() * unroll_for_);

    for (unsigned int i = 1; i < unroll_for_; ++i) {
      auto step = create_step(reference_, i, unroll_for_);
      processed.insert(processed.end(), step.begin(), step.end());
    }

    return processed;
  };

  /**
   * @brief case when return sequence is not true, only last output is renamed
   *
   */
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

  /**
   * @brief case when return sequence is true, concat layer is added to
   * aggregate all the output
   *
   */
  auto concat_output = [this](const GraphRepresentation &reference_,
                              unsigned unroll_for) {
    GraphRepresentation processed(reference_.begin(), reference_.end());

    for (auto &end : end_layers) {
      std::vector<props::Name> names;
      for (unsigned int i = 0; i < unroll_for; ++i) {
        names.push_back(end + "/" + std::to_string(i));
      }
      /// @todo have axis in concat layer
      auto node = createLayerNode(
        "concat", {"name=" + end, "input_layers=" + to_string(names)});
      processed.push_back(std::move(node));
    }

    return processed;
  };

  /**
   * @brief set output name
   *
   */
  auto step3_connect_output =
    [this, naive_output, concat_output](const GraphRepresentation &reference_,
                                        unsigned unroll_for) {
      bool return_sequence =
        std::get<props::ReturnSequences>(*recurrent_props).get();
      return return_sequence ? concat_output(reference_, unroll_for)
                             : naive_output(reference_, unroll_for);
    };

  auto unroll_for = std::get<props::UnrollFor>(*recurrent_props).get();
  step0_verify_and_prepare();
  auto processed = step1_connect_external_input(reference, unroll_for);
  processed = step2_unroll(processed, unroll_for);
  return step3_connect_output(processed, unroll_for);
}

} // namespace nntrainer
