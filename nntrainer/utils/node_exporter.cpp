// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file node_exporter.cpp
 * @date 09 April 2021
 * @brief NNTrainer Node exporter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <node_exporter.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <common_properties.h>
#include <fc_layer.h>
#include <tf_schema_generated.h>
#include <tflite_opnode.h>
#endif

namespace nntrainer {

/**
 * @brief Construct a new Exporter object
 *
 */
Exporter::Exporter() : stored_result(nullptr), is_exported(false){};

/**
 * @brief Destroy the Exporter object
 *
 */
Exporter::~Exporter() = default;

template <>
std::unique_ptr<std::vector<std::pair<std::string, std::string>>>
Exporter::getResult<ExportMethods::METHOD_STRINGVECTOR>() noexcept {
  return std::move(stored_result);
}

#ifdef ENABLE_TFLITE_INTERPRETER
template <>
std::unique_ptr<TfOpNode>
Exporter::getResult<ExportMethods::METHOD_TFLITE>() noexcept {
  return std::move(tf_node);
}

template <>
void Exporter::saveTflResult(const std::tuple<> &props, const LayerV1 *self) {
  createIfNull(tf_node);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Name, props::Flatten, props::Distribute> &props,
  const LayerNode *self) {
  createIfNull(tf_node);
  tf_node->setInOut(*self);
  tf_node->setInputs(self->getObject()->getInputRef());
  tf_node->setOutputs(self->getObject()->getOutputRef());
}

template <>
void Exporter::saveTflResult(const std::tuple<props::Unit> &props,
                             const FullyConnectedLayer *self) {
  createIfNull(tf_node);

  auto &weights = self->getWeightsRef();

  /// transpose weight [h, w] -> [w, h]
  std::unique_ptr<Var_Grad> weight_vg =
    std::make_unique<Var_Grad>(weights[0].cloneTransposeVariableOnly("0:2:1"));

  tf_node->appendInput(std::move(weight_vg), true);
  tf_node->appendInput(&weights[1], true);

  tf_node->setOpType(tflite::BuiltinOperator_FULLY_CONNECTED);
  /// we probably going to need flatbuffer inside exporter regarding this
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_FullyConnectedOptions,
                             flatbuffers::Offset<void>());
}
#endif

} // namespace nntrainer
