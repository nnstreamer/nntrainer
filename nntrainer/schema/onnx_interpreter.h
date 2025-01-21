// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   onnx_interpreter.h
 * @date   12 February 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#ifndef __ONNX_INTERPRETER_H__
#define __ONNX_INTERPRETER_H__
#ifdef ENABLE_ONNX_INTERPRETER

#include <app_context.h>
#include <interpreter.h>
#include <layer.h>
#include <layer_node.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <onnx.pb.h>
#include <string>

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

/**
 * @brief make "key=value1,value2,...valueN" from key and multiple values
 *
 * @tparam T type of a value
 * @param key key
 * @param value list of values
 * @return std::string with "key=value1,value2,...valueN"
 */
template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }
  std::stringstream ss;
  ss << key << "=";
  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;
  return ss.str();
}

namespace nntrainer {
/**
 * @brief ONNX Interpreter class for converting onnx model to nntrainer model.
 *
 */
class ONNXInterpreter {
public:
  /**
   * @brief Construct a new ONNXInterpreter object
   *
   */
  ONNXInterpreter(){};

  /**
   * @brief Destroy the ONNXInterpreter object
   *
   */
  ~ONNXInterpreter(){};

  /**
   * @brief Load onnx model from given path and convert to nntrainer model.
   *
   * @param path path of onnx model file.
   */
  std::unique_ptr<ml::train::Model> load(std::string path) {
    // Load and parse onnx file with protobuf
    std::ifstream file(path, std::ios::binary);
    onnx_model.ParseFromIstream(&file);

    // Create nntrainer model instance
    nntrainer_model = ml::train::createModel();
    std::vector<std::shared_ptr<ml::train::Layer>> layers;

    // Create initializer(weight) unordered map and create weight layer
    for (auto &initializer : onnx_model.graph().initializer()) {
      // initializers are used to identify weights in the model
      initializers.insert({cleanName(initializer.name()), initializer});
      std::string dim = transformDimString(initializer);

      // weight layer should be modified not to use input_shape as a parameter
      layers.push_back(ml::train::createLayer(
        "weight", {withKey("name", cleanName(initializer.name())),
                   withKey("dim", dim), withKey("input_shape", dim)}));
    }

    // Create input & constant tensor layer
    for (const auto &input : onnx_model.graph().input()) {
      auto shape = input.type().tensor_type().shape();
      if (shape.dim_size() >= 4 || shape.dim_size() == 0) {
        throw std::runtime_error(
          "Tensors with batch dimensions of 5 or more, or zero_dimensional "
          "tensors are not supported.");
      }

      std::string dim = transformDimString(shape);
      if (input.name().find("input") !=
          std::string::npos) { // Create input layer
        layers.push_back(ml::train::createLayer(
          "input", {withKey("name", cleanName(input.name())),
                    withKey("input_shape", dim)}));
      } else { // Create constant tensor layer
        throw std::runtime_error("Constant tensors are not supported yet.");
      }
    }

    // Create graph
    for (const auto &node : onnx_model.graph().node()) {
      /**
       * @brief While NNTrainer represents graphs as connections between
       * operations, ONNX represents graphs as connections between operations
       * and tensors, requiring remapping of the names of output tensors from
       * operations.
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

      if (node.op_type() == "Add") {
        layerOutputMap.insert(
          {cleanName(node.output()[0]), cleanName(node.name())});

        layers.push_back(ml::train::createLayer(
          "add",
          {"name=" + cleanName(node.name()),
           withKey("input_layers", inputNames[0] + "," + inputNames[1])}));
      } else {
        throw std::runtime_error("Unsupported operation type: " +
                                 node.op_type());
      }
    }

    for (auto &layer : layers) {
      nntrainer_model->addLayer(layer);
    }

    return std::move(nntrainer_model);
  };

  /**
   * @brief Clean the name of the layer to be used in nntrainer model
   *
   * @param name name of the layer
   */
  std::string cleanName(std::string name) {
    if (!name.empty() && name[0] == '/') {
      name.erase(0, 1);
    }
    std::replace(name.begin(), name.end(), '/', '_');
    std::replace(name.begin(), name.end(), '.', '_');
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
  }

  /**
   * @brief Transform dimension string to nntrainer's format.
   *
   * @param shape ONNX TensorShapeProto
   */
  std::string transformDimString(onnx::TensorShapeProto shape) {
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

  // /**
  //  * @brief Transform dimension string to nntrainer's format.
  //  *
  //  * @param initializer ONNX TensorProto
  //  */
  std::string transformDimString(onnx::TensorProto initializer) {
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

private:
  onnx::ModelProto onnx_model; // parsed onnx model
  std::unique_ptr<ml::train::Model>
    nntrainer_model; // converted nntrainer model
  std::unordered_map<std::string, std::string>
    layerOutputMap; // key: name of output, value: name of layer
  std::unordered_map<std::string, onnx::TensorProto>
    initializers; // initializers are used to identify weights
};

} // namespace nntrainer

#endif // ENABLE_ONNX_INTERPRETER
#endif // __ONNX_INTERPRETER_H__
