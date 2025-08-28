#include <computational_graph.h>

#include <base_properties.h>
#include <iostream>
#include <nlohmann_json.hpp>

namespace nntrainer {

void ComputationalGraph::initialize(const NetworkGraph &network_graph) {
  nodes_.resize(network_graph.size());

  for (size_t i = 0; i < network_graph.size(); ++i) {
    const auto &network_graph_node = network_graph.getSortedLayerNode(i);

    nodes_[i].node = network_graph_node.get();
    nodes_map_[network_graph_node->getName()] = &nodes_[i];

    if (network_graph_node->getType() == "input") {
      input_nodes_.push_back(&nodes_[i]);
    }
  }

  for (const auto node_ptr : input_nodes_) {
    evaluateNode(node_ptr, nullptr);
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
    node->inputs.push_back(input_node);
  }

  if (node->evaluated) {
    return;
  }

  node->evaluated = true;

  for (const auto &output_name : node->node->getOutputConnections()) {
    auto iter = nodes_map_.find(output_name);

    if (iter != nodes_map_.end()) {
      node->outputs.push_back(iter->second);
    }
  }

  for (const auto output_node : node->outputs) {
    evaluateNode(output_node, node);
  }
}

} // namespace nntrainer
