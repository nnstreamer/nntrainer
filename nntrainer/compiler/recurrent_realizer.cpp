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
#include <algorithm>
#include <stdexcept>

#include <common_properties.h>
#include <connection.h>
#include <grucell.h>
#include <input_layer.h>
#include <layer_node.h>
#include <lstm.h>
#include <lstmcell.h>
#include <lstmcell_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <recurrent_realizer.h>
#include <remap_realizer.h>
#include <rnncell.h>
#include <util_func.h>
#include <zoneout_lstmcell.h>

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
class RecurrentInput final : public Property<Connection> {
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
  RecurrentInput(const Connection &name);
  static constexpr const char *key = "recurrent_input";
  using prop_tag = connection_prop_tag;
};

RecurrentInput::RecurrentInput() {}
RecurrentInput::RecurrentInput(const Connection &con) { set(con); };

/**
 * @brief Property for recurrent outputs
 *
 */
class RecurrentOutput final : public Property<Connection> {
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
  RecurrentOutput(const Connection &name);
  static constexpr const char *key = "recurrent_output";
  using prop_tag = connection_prop_tag;
};

RecurrentOutput::RecurrentOutput() {}
RecurrentOutput::RecurrentOutput(const Connection &con) { set(con); };
} // namespace props

RecurrentRealizer::RecurrentRealizer(
  const std::vector<std::string> &properties,
  const std::vector<std::string> &input_layers,
  const std::vector<std::string> &end_layers) :
  input_layers(input_layers.begin(), input_layers.end()),
  end_layers(end_layers),
  sequenced_return_layers(),
  recurrent_props(new PropTypes(
    std::vector<props::RecurrentInput>(), std::vector<props::RecurrentOutput>(),
    std::vector<props::AsSequence>(), props::UnrollFor(1))) {
  auto left = loadProperties(properties, *recurrent_props);

  /// @note AsSequence must be identifier based (not connection based) for now
  /// consider A(layer) outputs a0, a1 connection and a0 needs return seq
  /// Then it is impossible to locate a0 and a1 with the same name unless we
  /// have some kind of multi,multiout identity layer. Until this is supported,
  /// AsSequenced stays as identifier based

  auto &[inputs, outputs, as_sequence, unroll_for] = *recurrent_props;

  NNTR_THROW_IF(inputs.empty() || inputs.size() != outputs.size(),
                std::invalid_argument)
    << "recurrent inputs and outputs must not be empty and 1:1 map but given "
       "different size. input: "
    << inputs.size() << " output: " << outputs.size();

  NNTR_THROW_IF(!std::all_of(as_sequence.begin(), as_sequence.end(),
                             [&end_layers](const std::string &seq) {
                               return std::find(end_layers.begin(),
                                                end_layers.end(),
                                                seq) != end_layers.end();
                             }),
                std::invalid_argument)
    << "as_sequence property must be subset of end_layers";

  std::unordered_set<std::string> check_seqs;
  for (auto &name : as_sequence) {
    sequenced_return_layers.emplace(name.get());
  };

  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There is unparesed properties";

  for (unsigned i = 0, sz = inputs.size(); i < sz; ++i) {
    recurrent_info.emplace(inputs.at(i).get(), outputs.at(i).get());
  }
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
           node->getType() == LSTMCellCoreLayer::type ||
           node->getType() == ZoneoutLSTMCellLayer::type ||
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

  auto step0_verify_and_prepare = []() {
    /// empty intended
  };

  /**
   * @brief maps input place holder to given name otherwise scopped to suffix
   * "/0"
   *
   */
  auto step1_connect_external_input =
    [this](const GraphRepresentation &reference_, unsigned max_time_idx) {
      RemapRealizer input_mapper([this](std::string &id) {
        if (input_layers.count(id) == 0) {
          id += "/0";
        }
      });

      auto nodes = input_mapper.realize(reference_);
      for (auto &node : nodes) {
        propagateTimestep(node.get(), 0, max_time_idx);
      }

      return nodes;
    };

  /**
   * @brief Create a single time step. Used inside step2_unroll.
   *
   */
  auto create_step = [this](const GraphRepresentation &reference_,
                            unsigned time_idx, unsigned max_time_idx) {
    GraphRepresentation step;
    step.reserve(reference_.size());

    auto replace_time_idx = [](std::string &name, unsigned time_idx) {
      auto pos = name.find_last_of('/');
      if (pos != std::string::npos) {
        name.replace(pos + 1, std::string::npos, std::to_string(time_idx));
      }
    };
    for (auto &node : reference_) {
      auto new_node = node->cloneConfiguration();

      /// 1. remap identifiers to $name/$idx
      new_node->remapIdentifiers(
        [this, time_idx, replace_time_idx](std::string &id) {
          if (input_layers.count(id) == 0) {
            replace_time_idx(id, time_idx);
          }
        });

      /// 2. override first output name to $name/$idx - 1
      for (auto &[recurrent_input, recurrent_output] : recurrent_info) {
        if (node->getName() != recurrent_input.getName() + "/0") {
          continue;
        }
        new_node->setInputConnectionIndex(recurrent_input.getIndex(),
                                          recurrent_output.getIndex());
        new_node->setInputConnectionName(recurrent_input.getIndex(),
                                         recurrent_output.getName() + "/" +
                                           std::to_string(time_idx - 1));
      }

      /// 3. set shared_from
      new_node->setProperty({"shared_from=" + node->getName()});
      /// 4. if recurrent layer type set timestep property
      propagateTimestep(new_node.get(), time_idx, max_time_idx);

      step.push_back(std::move(new_node));
    }
    return step;
  };

  /**
   * @brief unroll the graph by calling create_step()
   *
   */
  auto step2_unroll = [create_step](const GraphRepresentation &reference_,
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
   * @todo support connection using node->remapConnection
   */
  auto naive_output = [](const GraphRepresentation &reference_,
                         const std::string &con, unsigned unroll_for) {
    auto target = con + "/" + std::to_string(unroll_for - 1);
    RemapRealizer r([target, con](std::string &name) {
      if (name == target) {
        name = con;
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
                              const std::string &con, unsigned unroll_for) {
    GraphRepresentation processed(reference_.begin(), reference_.end());

    std::vector<props::Name> names;
    for (unsigned int i = 0; i < unroll_for; ++i) {
      names.push_back(con + "/" + std::to_string(i));
    }
    /// @todo have axis in concat layer
    auto node = createLayerNode(
      "concat", {"name=" + con, "input_layers=" + to_string(names)});
    processed.push_back(std::move(node));

    return processed;
  };

  /**
   * @brief set output name
   *
   */
  auto step3_connect_output =
    [this, naive_output, concat_output](const GraphRepresentation &reference_,
                                        unsigned unroll_for) {
      /// @note below is inefficient way of processing nodes. consider optimize
      /// below as needed by calling remap realizer only once
      auto processed = reference_;
      for (auto &name : end_layers) {
        processed = sequenced_return_layers.count(name)
                      ? concat_output(processed, name, unroll_for)
                      : naive_output(processed, name, unroll_for);
      }

      return processed;
    };

  auto unroll_for = std::get<props::UnrollFor>(*recurrent_props).get();
  step0_verify_and_prepare();
  auto processed = step1_connect_external_input(reference, unroll_for);
  processed = step2_unroll(processed, unroll_for);
  return step3_connect_output(processed, unroll_for);
}

} // namespace nntrainer
