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
#include <iterator>
#include <stdexcept>
#include <string>

#include <base_properties.h>
#include <common_properties.h>
#include <connection.h>
#include <input_layer.h>
#include <layer_node.h>
#include <lstm.h>
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

RecurrentRealizer::RecurrentRealizer(const std::vector<std::string> &properties,
                                     const std::vector<Connection> &input_conns,
                                     const std::vector<Connection> &end_conns) :
  input_layers(),
  end_info(),
  sequenced_return_conns(),
  recurrent_props(new PropTypes(
    std::vector<props::RecurrentInput>(), std::vector<props::RecurrentOutput>(),
    std::vector<props::AsSequence>(), props::UnrollFor(1),
    std::vector<props::InputIsSequence>())) {
  auto left = loadProperties(properties, *recurrent_props);

  std::transform(input_conns.begin(), input_conns.end(),
                 std::inserter(this->input_layers, this->input_layers.begin()),
                 [](const Connection &c) { return c.getName(); });

  /// build end info.
  /// eg)
  /// end_layers: a(0), a(3), b(0) becomes
  /// end_info: {{a, 3}, {b, 0}}
  /// end_layers: a(1), b(3), c(0) becomes
  /// end_info: {{a, 1}, {b, 3}, {c, 0}}
  for (unsigned i = 0u, sz = end_conns.size(); i < sz; ++i) {
    const auto &name = end_conns[i].getName();
    const auto &idx = end_conns[i].getIndex();
    auto iter =
      std::find_if(end_info.begin(), end_info.end(),
                   [&name](auto &info) { return info.first == name; });
    if (iter == end_info.end()) {
      end_info.emplace_back(name, idx);
    } else {
      iter->second = std::max(iter->second, idx);
    }
  }

  auto &[inputs, outputs, as_sequence, unroll_for, input_is_seq] =
    *recurrent_props;

  NNTR_THROW_IF(inputs.empty() || inputs.size() != outputs.size(),
                std::invalid_argument)
    << "recurrent inputs and outputs must not be empty and 1:1 map but given "
       "different size. input: "
    << inputs.size() << " output: " << outputs.size();

  /// @todo Deal as sequence as proper connection with identity layer
  NNTR_THROW_IF(!std::all_of(as_sequence.begin(), as_sequence.end(),
                             [&end_conns](const Connection &seq) {
                               return std::find(end_conns.begin(),
                                                end_conns.end(),
                                                seq) != end_conns.end();
                             }),
                std::invalid_argument)
    << "as_sequence property must be subset of end_layers";

  for (auto &name : as_sequence) {
    sequenced_return_conns.emplace(name.get());
  };

  sequenced_input =
    std::unordered_set<std::string>(input_is_seq.begin(), input_is_seq.end());

  for (auto &seq_input : sequenced_input) {
    NNTR_THROW_IF(input_layers.count(seq_input) == 0, std::invalid_argument)
      << seq_input
      << " is not found inside input_layers, inputIsSequence argument must be "
         "subset of inputs";
  }

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
           node->getType() == ZoneoutLSTMCellLayer::type;
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
        } else if (sequenced_input.count(id) != 0) {
          id += "/0";
        }
      });

      auto nodes = input_mapper.realize(reference_);
      for (auto &node : nodes) {
        propagateTimestep(node.get(), 0, max_time_idx);
        /// #1744, quick fix, add shared_from to every node
        node->setProperty({"shared_from=" + node->getName()});
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
   * @brief case when return sequence is true, concat layer is added to
   * aggregate all the output
   *
   */
  auto concat_output = [](const GraphRepresentation &reference_,
                          const Connection &con, unsigned unroll_for,
                          const std::string &new_layer_name) {
    GraphRepresentation processed(reference_.begin(), reference_.end());

    std::vector<props::RecurrentInput> conns;
    for (unsigned int i = 0; i < unroll_for; ++i) {
      conns.emplace_back(Connection{
        con.getName() + "/" + std::to_string(i),
        con.getIndex(),
      });
    }
    /// @todo have axis in concat layer
    /// @todo this has to be wrapped with identity layer as #1793
    auto node = createLayerNode(
      "concat", {"name=" + new_layer_name, "input_layers=" + to_string(conns)});
    processed.push_back(std::move(node));

    return processed;
  };

  /**
   * @brief create identity layer with output name by either creating concat
   * layer or directly using the connection, the number of inputs connection
   * have is depending on the end_conns max.
   *
   * eg)
   * layer A outputs a, b, c, d
   *
   * if end_layers=A(0),A(2)
   *    as_sequence=A(0)
   * realizer cannot know there is d so this is ignored. It is okay because user
   * didn't specify to use it anyway
   *
   * [A]
   * type=identity
   * input_layers=A_concat_0, A(1), A(2)
   *
   */
  auto step3_connect_output = [this, concat_output](
                                const GraphRepresentation &reference_,
                                unsigned unroll_for) {
    /// @note below is inefficient way of processing nodes. consider optimize
    /// below as needed by calling remap realizer only once
    auto processed = reference_;
    for (auto [name, max_idx] : end_info) {

      std::vector<props::InputConnection> out_node_inputs;

      for (auto i = 0u; i <= max_idx; ++i) {

        if (auto con = Connection(name, i); sequenced_return_conns.count(con)) {
          auto concat_name = name + "/concat_" + std::to_string(i);
          processed = concat_output(processed, con, unroll_for, concat_name);
          // create concat connection name,
          out_node_inputs.emplace_back(Connection(concat_name, 0));
        } else {
          auto last_layer_name = name + "/" + std::to_string(unroll_for - 1);
          out_node_inputs.emplace_back(Connection(last_layer_name, i));
        }
      }

      auto alias_layer = createLayerNode(
        "identity",
        {"name=" + name, "input_layers=" + to_string(out_node_inputs)});
      processed.push_back(std::move(alias_layer));
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
