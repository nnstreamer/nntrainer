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
Exporter::Exporter() = default;

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
void Exporter::saveTflResult(const std::tuple<props::Name> &props,
                             const Layer *self) {
  createIfNull(tf_node);
  tf_node->setInOut(*self);
  tf_node->setInputs(self->getInputRef());
  tf_node->setOutputs(self->getOutputRef());
}

template <>
void Exporter::saveTflResult(const std::tuple<props::Unit> &props,
                             const FullyConnectedLayer *self) {
  createIfNull(tf_node);

  tf_node->setWeights(self->getWeightsRef());

  tf_node->setOpType(tflite::BuiltinOperator_FULLY_CONNECTED);
  /// we probably going to need flatbuffer inside exporter regarding this
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_FullyConnectedOptions,
                             flatbuffers::Offset<void>());
}
#endif

} // namespace nntrainer
