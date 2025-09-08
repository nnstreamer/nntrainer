#include <computational_graph.h>

#include <base_properties.h>
#include <iostream>
#include <nlohmann_json.hpp>

namespace nntrainer {

void ComputationalGraph::initialize(const NetworkGraph &network_graph) {
  nodes_.resize(network_graph.size());

  for (size_t i = 0; i < network_graph.size(); ++i) {
    const auto &network_graph_node = network_graph.getSortedLayerNode(i);

    nodes_[i].node = network_graph_node;
    nodes_map_[network_graph_node->getName()] = &nodes_[i];

    if (network_graph_node->getType() == "input") {
      input_nodes_.push_back(&nodes_[i]);
    }
  }

  for (const auto node_ptr : input_nodes_) {
    evaluateNode(node_ptr, nullptr);
  }

  for (auto &node : nodes_) {
    if (node.outputs.empty()) {
      output_nodes_.push_back(&node);
    }
  }
}

void ComputationalGraph::serialize(const std::string &file_name) {
  nlohmann::json graph_json;

  graph_json["class_name"] = "Functional";

  nlohmann::json config_json;

  config_json["name"] = "functional";

  for (const auto &node : nodes_) {
    nlohmann::json layer_json;

    std::string class_name = node.node->getType();
    if (class_name == "input") {
      class_name = "InputLayer";
    }

    layer_json["module"] = "keras.layers";
    layer_json["class_name"] = class_name;
    layer_json["name"] = node.node->getName();
    layer_json["registered_name"] = nullptr;

    nlohmann::json layer_config_json;

    layer_config_json["dtype"] =
      str_converter<enum_class_prop_tag, nntrainer::TensorDataTypeInfo>::
        to_string(node.node->getWeightDataType());
    layer_config_json["name"] = node.node->getName();
    layer_config_json["activation"] =
      str_converter<enum_class_prop_tag, nntrainer::props::ActivationTypeInfo>::
        to_string(node.node->getActivationType());

    if (node.inputs.empty()) {
      for (const auto &dim : node.node->getInputDimensions()) {
        nlohmann::json layer_shape;

        layer_shape.push_back(dim.batch());
        layer_shape.push_back(dim.channel());
        layer_shape.push_back(dim.height());
        layer_shape.push_back(dim.width());

        layer_config_json["batch_input_shape"].push_back(layer_shape);
      }
    } else {
      layer_config_json["shape"].push_back("Inputs");
      for (const auto &dim : node.node->getInputDimensions()) {
        nlohmann::json layer_shape;

        layer_shape.push_back(dim.batch());
        layer_shape.push_back(dim.channel());
        layer_shape.push_back(dim.height());
        layer_shape.push_back(dim.width());

        if (node.inputs.empty()) {
          layer_config_json["batch_input_shape"].push_back(layer_shape);
        } else {

          layer_config_json["shape"].push_back(layer_shape);
        }
      }

      layer_config_json["shape"].push_back("Outputs");
      for (const auto &dim : node.node->getOutputDimensions()) {
        nlohmann::json layer_shape;

        layer_shape.push_back(dim.batch());
        layer_shape.push_back(dim.channel());
        layer_shape.push_back(dim.height());
        layer_shape.push_back(dim.width());

        if (node.inputs.empty()) {
          layer_config_json["batch_input_shape"].push_back(layer_shape);
        } else {

          layer_config_json["shape"].push_back(layer_shape);
        }
      }
    }

    layer_json["config"] = layer_config_json;

    if (node.inputs.empty()) {
      layer_json["inbound_nodes"] = nlohmann::json::array();
    } else {
      nlohmann::json input_node_json;

      for (const auto &input_node : node.inputs) {
        nlohmann::json args_json;
        nlohmann::json args_config_json;
        nlohmann::json args_keras_history_json;

        args_keras_history_json.push_back(input_node->node->getName());
        args_keras_history_json.push_back(0);
        args_keras_history_json.push_back(0);

        args_config_json["keras_history"] = args_keras_history_json;
        args_config_json["dtype"] =
          str_converter<enum_class_prop_tag, nntrainer::TensorDataTypeInfo>::
            to_string(input_node->node->getWeightDataType());

        args_json["config"] = args_config_json;
        args_json["class_name"] = "__keras_tensor__";

        input_node_json["args"].push_back(args_json);
      }

      layer_json["inbound_nodes"].push_back(input_node_json);
    }

    config_json["layers"].push_back(layer_json);
  }

  nlohmann::json input_layers_json;

  for (const auto input_node : input_nodes_) {
    nlohmann::json input_layer_json;
    input_layer_json.push_back(input_node->node->getName());
    input_layer_json.push_back(0);
    input_layer_json.push_back(0);
    input_layers_json.push_back(input_layer_json);
  }

  config_json["input_layers"] = input_layers_json;

  nlohmann::json output_layers_json;
  for (const auto &node : nodes_) {
    if (node.outputs.empty()) {
      nlohmann::json output_layer_json;
      output_layer_json.push_back(node.node->getName());
      output_layer_json.push_back(0);
      output_layer_json.push_back(0);
      output_layers_json.push_back(output_layer_json);
    }
  }

  config_json["output_layers"] = output_layers_json;

  graph_json["config"] = config_json;

  std::ofstream output_frames(file_name);
  output_frames << std::setw(4) << graph_json << std::endl;

  std::cout << "Graph saved in: " << file_name << std::endl;
}

void ComputationalGraph::evaluateNode(ComputationalGraphNode *node,
                                      ComputationalGraphNode *input_node) {

  if (input_node) {
    node->inputs.insert(input_node);
  }

  if (node->evaluated) {
    return;
  }

  node->evaluated = true;

  for (const auto &output_name : node->node->getOutputConnections()) {
    auto iter = nodes_map_.find(output_name);

    if (iter != nodes_map_.end()) {
      node->outputs.insert(iter->second);
    }
  }

  for (const auto output_node : node->outputs) {
    evaluateNode(output_node, node);
  }
}

void ComputationalGraph::topologicalSort() {
  std::cout << "Topological sort:" << std::endl;

  sorted_nodes_.clear();
  std::deque<ComputationalGraphNode *> node_set;

  for (auto node : input_nodes_) {
    node->order = 0;
    node_set.push_back(node);
  }

  while (!node_set.empty()) {
    auto node = node_set.front();
    node_set.pop_front();
    sorted_nodes_.push_back(node);

    for (auto out_node : node->outputs) {
      out_node->in_orders.push_back(node->order);
      out_node->inputs.erase(node);

      if (out_node->inputs.empty()) {
        int max_order = 0;
        for (auto order : out_node->in_orders) {
          max_order = std::max(max_order, order);
        }
        out_node->order = max_order + 1;

        node_set.push_back(out_node);
      }
    }
  }

  for (const auto node : sorted_nodes_) {
    std::cout << node->node->getName() << ", " << node->node->getType()
              << ", order: " << node->order << ", execution order: "
              << std::get<0>(node->node->getExecutionOrder())
              << ", isGPU: " << node->node->getLayer()->isGPU() << std::endl;
  }
}

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

sharedConstTensors ComputationalGraph::forwarding(
  bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool, SynchronizationInfo *)>
    forwarding_op) {

  bool inside_gpu_block = false;

  for (int i = 0; i < sorted_nodes_.size(); ++i) {
    auto node = sorted_nodes_[i];

    if (node->node->getLayer()->isGPU() && !inside_gpu_block) {
      inside_gpu_block = true;
      // std::cout << "Gpu block begin, " << node->node->getName() << std::endl;
    }

    if (!node->node->getLayer()->isGPU() && inside_gpu_block) {
      inside_gpu_block = false;
      // std::cout << "Gpu block end, " << node->node->getName() << std::endl;
    }

    std::vector<ComputationalGraphNode *> paralell_run;
    paralell_run.push_back(node);
    auto i_next = i + 1;
    while ((i_next < sorted_nodes_.size()) &&
           (sorted_nodes_[i_next]->order == node->order) &&
           sorted_nodes_[i_next]->node->getLayer()->isGPU()) {
      paralell_run.push_back(sorted_nodes_[i_next]);
      ++i_next;
    }

    if (paralell_run.size() > 1) {
      auto cl_context =
        static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

      std::vector<std::unique_ptr<SynchronizationInfo>>
        synchronization_info_vec(paralell_run.size());

      std::vector<cl_event> events;

      for (int p = 0; p < paralell_run.size(); ++p) {

        synchronization_info_vec[p] =
          std::move(std::make_unique<SynchronizationInfo>());
        // std::cout << "|ENQ->" << paralell_run.at(p)->node->getName() << ","
        //           << paralell_run.at(p)->node->getType() << "|";
        forwarding_op(paralell_run.at(p)->node, training,
                      synchronization_info_vec.at(p).get());

        if (synchronization_info_vec.at(p)->wait_for_event) {
          events.push_back(synchronization_info_vec.at(p)->event);
        }
      }

      if (!events.empty()) {
        cl_context->command_queue_inst_.waitForEvent(events.size(),
                                                     events.data());

        for (int p = 0; p < events.size(); ++p) {
          cl_context->command_queue_inst_.releaseEvent(events.at(p));
        }
      }

      i += paralell_run.size() - 1;
    } else {
      // std::cout << "|RUN->" << node->node->getName() << ","
      //           << node->node->getType() << "|";
      forwarding_op(node->node, training, nullptr);
    }
  }

  sharedConstTensors out;

  for (auto node : output_nodes_) {
    for (unsigned int j = 0; j < node->node->getNumOutputs(); ++j) {
      out.push_back(MAKE_SHARED_TENSOR(node->node->getOutput(j)));
    }
  }

  return out;
}

} // namespace nntrainer
