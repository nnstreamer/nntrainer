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
#include <iostream>
#include <layer.h>
#include <layer_node.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <onnx.pb.h>
#include <string>
#include <util_func.h>

namespace nntrainer {
/**
 * @brief ONNX Interpreter class for converting onnx model to nntrainer model.
 *
 */
class ONNXInterpreter : public GraphInterpreter {
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

  std::string getDataType(int onnx_type);

  /**
   * @copydoc GraphInterpreter::serialize(const GraphRepresentation
   * representation, const std::string &out)
   */
  void serialize(const GraphRepresentation &representation,
                 const std::string &out) override;

  /**
   * @copydoc GraphInterpreter::deserialize(const std::string &in)
   */
  GraphRepresentation deserialize(const std::string &in) override;

  /**
   * @brief Clean the name of the layer to be used in nntrainer model
   *
   * @param name name of the layer
   */
  std::string cleanName(std::string name);

  /**
   * @brief Transform dimension string to nntrainer's format.
   *
   * @param shape ONNX TensorShapeProto
   */
  std::string transformDimString(onnx::TensorShapeProto shape);

  // /**
  //  * @brief Transform dimension string to nntrainer's format.
  //  *
  //  * @param initializer ONNX TensorProto
  //  */
  std::string transformDimString(onnx::TensorProto initializer);

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
