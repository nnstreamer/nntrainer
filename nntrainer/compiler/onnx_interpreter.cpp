// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   onnx_interpreter.cpp
 * @date   12 February 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#include "layer_node.h"
#include <onnx_interpreter.h>
#include <string>
#include <unordered_map>

namespace nntrainer {

void ONNXInterpreter::handleUnaryOp(const onnx::NodeProto &node,
                                    GraphRepresentation &representation,
                                    const std::string &op_type,
                                    std::vector<std::string> &props) {
  std::vector<std::string> inputNames = createOutputRemap(node);
  props.push_back("name=" + cleanName(node.name()));
  if (op_type == "activation") {
    props.push_back("activation=" + activationKeyMap[node.op_type()]);
  }
  props.push_back("input_layers=" + inputNames[0]);
  representation.push_back(
    createLayerNode(op_type, {props.begin(), props.end()}));
}

void ONNXInterpreter::handleBinaryOp(const onnx::NodeProto &node,
                                     GraphRepresentation &representation,
                                     const std::string &op_type,
                                     std::vector<std::string> &props) {
  std::vector<std::string> inputNames = createOutputRemap(node);
  props.push_back("name=" + cleanName(node.name()));
  props.push_back("input_layers=" + inputNames[0] + "," + inputNames[1]);
  representation.push_back(
    createLayerNode(op_type, {props.begin(), props.end()}));
}

void ONNXInterpreter::registerBasicUnaryOp(const std::string &op_type) {
  NodeHandlers[op_type] = [this](const onnx::NodeProto &node,
                                 GraphRepresentation &rep) {
    std::vector<std::string> props;
    handleUnaryOp(node, rep, layerKeyMap[node.op_type()], props);
  };
}

void ONNXInterpreter::registerBasicBinaryOp(const std::string &op_type) {
  NodeHandlers[op_type] = [this](const onnx::NodeProto &node,
                                 GraphRepresentation &rep) {
    std::vector<std::string> props;
    handleBinaryOp(node, rep, layerKeyMap[node.op_type()], props);
  };
}

void ONNXInterpreter::registerNodeHandlers() {
  registerBasicBinaryOp("Add");
  registerBasicBinaryOp("Sub");
  registerBasicBinaryOp("Mul");
  registerBasicBinaryOp("Div");
  registerBasicBinaryOp("MatMul");
  registerBasicUnaryOp("Softmax");

  NodeHandlers["Reshape"] = [this](const onnx::NodeProto &node,
                                   GraphRepresentation &rep) {
    std::vector<std::string> props;
    const std::string &input_name = node.input(0); // input tensor name
    const std::string &shape_name = node.input(1);
    const auto &shape_node = constantTensors.at(shape_name);
    for (const auto &attr : shape_node.attribute()) {
      if (attr.name() == "value") {
        const auto &shape_tensor = attr.t().raw_data();
        const int64_t *vals =
          reinterpret_cast<const int64_t *>(shape_tensor.data());
        int size = shape_tensor.size() / sizeof(int64_t);
        std::ostringstream oss;
        oss << "target_shape=";
        for (int i = 1; i < size; ++i) {
          oss << vals[i];
          if (i < size - 1)
            oss << ":";
        }
        props.push_back(oss.str());
      }
    }
    handleUnaryOp(node, rep, layerKeyMap[node.op_type()], props);
  };

  NodeHandlers["Transpose"] = [this](const onnx::NodeProto &node,
                                     GraphRepresentation &rep) {
    std::vector<std::string> props;
    for (const auto &attr : node.attribute()) {
      if (attr.name() == "perm") {
        std::ostringstream oss;
        oss << "direction=";
        for (int i = 1; i < attr.ints_size(); ++i) {
          oss << attr.ints(i);
          if (i < attr.ints_size() - 1)
            oss << ",";
        }
        props.push_back(oss.str());
      }
    }

    handleUnaryOp(node, rep, layerKeyMap[node.op_type()], props);
  };

  NodeHandlers["Cast"] = [this](const onnx::NodeProto &node,
                                GraphRepresentation &rep) {
    std::vector<std::string> props;
    for (const auto &attr : node.attribute()) {
      if (attr.name() == "to") {
        props.push_back("tensor_dtype=" + getDataTypeFromONNX(attr.i()));
      }
    }
    handleUnaryOp(node, rep, layerKeyMap[node.op_type()], props);
  };
};

std::string ONNXInterpreter::getDataTypeFromONNX(int onnx_type) {
  switch (onnx_type) {
  case onnx::TensorProto::FLOAT:
    return "FP32";
  case onnx::TensorProto::FLOAT16:
    return "FP16";
  default:
    throw std::runtime_error("Unsupported ONNX tensor data type: " +
                             std::to_string(onnx_type));
  }
}

void ONNXInterpreter::loadInputsAndWeights(
  GraphRepresentation &representation) {
  // Create initializer(weight) unordered map and create weight layer
  for (auto &initializer : onnx_model.graph().initializer()) {
    // initializers are used to identify weights in the model
    initializers.insert({cleanName(initializer.name()), initializer});
    std::string dim = transformDimString(initializer);

    // weight layer should be modified not to use input_shape as a parameter
    representation.push_back(createLayerNode(
      "weight",
      {withKey("name", cleanName(initializer.name())), withKey("dim", dim),
       withKey("input_shape", dim),
       withKey("tensor_dtype", getDataTypeFromONNX(initializer.data_type())),
       withKey("weight_name", cleanName(initializer.name()))}));
  }

  // Create input & constant tensor layer
  for (const auto &input : onnx_model.graph().input()) {
    auto shape = input.type().tensor_type().shape();
    if (shape.dim_size() >= 4 || shape.dim_size() == 0) {
      throw std::runtime_error(
        "Tensors with batch dimensions of 5 or more, or zero_dimensional "
        "tensors are not supported.");
    }
    inputDims.insert({cleanName(input.name()), shape});
    std::string dim = transformDimString(shape);
    if (input.name().find("input") != std::string::npos) { // Create input layer
      representation.push_back(
        createLayerNode("input", {withKey("name", cleanName(input.name())),
                                  withKey("input_shape", dim)}));
    } else {
      throw std::runtime_error(input.name() + " is not supported yet.");
    }
  }
}

std::vector<std::string>
ONNXInterpreter::createOutputRemap(const onnx::NodeProto &node) {
  /**
  * @brief While NNTrainer represents graphs as connections between
  * operations, ONNX represents graphs as connections between
  operations and tensors, requiring remapping of the names of output
tensors from operations.
  */
  std::vector<std::string> inputNames;
  auto outputRemap = [this](std::string &input_layer_name) {
    if (layerOutputMap.find(input_layer_name) != layerOutputMap.end()) {
      input_layer_name = layerOutputMap.find(input_layer_name)->second;
    }
  };
  for (auto &input : node.input()) {
    std::string inputName = cleanName(input);
    outputRemap(inputName);
    inputNames.push_back(inputName);
  }
  layerOutputMap.insert({cleanName(node.output()[0]), cleanName(node.name())});

  return inputNames;
};

void ONNXInterpreter::loadOperations(GraphRepresentation &representation) {

  // Create graph
  for (const auto &node : onnx_model.graph().node()) {
    if (node.op_type() == "Constant") {
      constantTensors.insert({node.name() + "_output_0", node});
      continue;
    }
    std::vector<std::string> inputNames = createOutputRemap(node);
    NodeHandlers[node.op_type()](node, representation);
  }
};

void ONNXInterpreter::loadONNXModel(const std::string &file_path) {
  // Load and parse onnx file with protobuf
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("File does not exist: " + file_path);
  }

  onnx_model.ParseFromIstream(&file);
}

void ONNXInterpreter::serialize(const GraphRepresentation &representation,
                                const std::string &out){};

GraphRepresentation ONNXInterpreter::deserialize(const std::string &in) {

  // Create nntrainer model instance
  GraphRepresentation graph;
  // Load ONNX model file
  loadONNXModel(in);
  // Load inputs and weights from ONNX model files and add them to graph
  loadInputsAndWeights(graph);
  // Load operations from ONNX model files and add them to graph
  loadOperations(graph);

  return graph;
};

std::string ONNXInterpreter::cleanName(std::string name) {
  if (!name.empty() && name[0] == '/') {
    name.erase(0, 1);
  }
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '.', '_');
  std::replace(name.begin(), name.end(), ':', '_');
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return name;
}

std::string ONNXInterpreter::transformDimString(onnx::TensorShapeProto shape) {
  std::string dim = "";
  for (int i = 0; i < shape.dim_size(); ++i) {
    if (shape.dim()[i].has_dim_param()) {
      throw std::runtime_error("Dynamic dimensions are not supported");
    }
    dim += std::to_string(shape.dim()[i].dim_value());
    if (i < shape.dim_size() - 1) {
      dim += ":";
    }
  }

  if (shape.dim_size() == 1) {
    dim = "1:1:" + dim;
  } else if (shape.dim_size() == 2) {
    dim = "1:" + dim;
  }

  return dim;
}

std::string ONNXInterpreter::transformDimString(onnx::TensorProto initializer) {
  std::string dim = "";
  for (int i = 0; i < initializer.dims_size(); ++i) {
    dim += std::to_string(initializer.dims()[i]);
    if (i < initializer.dims_size() - 1) {
      dim += ":";
    }
  }

  if (initializer.dims_size() == 1) {
    dim = "1:1:" + dim;
  } else if (initializer.dims_size() == 2) {
    dim = "1:" + dim;
  }

  return dim;
};
}; // namespace nntrainer
