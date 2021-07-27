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
Exporter::Exporter() : stored_result(nullptr), is_exported(false) {}

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
void Exporter::saveTflResult(const std::tuple<> &props,
                             const nntrainer::Layer *self) {
  createIfNull(tf_node);
}

static void saveTflWeights(TfOpNode *tf_node, const RunLayerContext &context,
                           const std::string &transpose_direction = "0:2:1") {
  for (unsigned int idx = 0; idx < context.getNumWeights(); idx++) {
    const Tensor &w = context.getWeight(idx);
    const Tensor &g = context.getWeightGrad(idx);
    const std::string name = context.getWeightName(idx);
    std::unique_ptr<Var_Grad> vg = std::make_unique<Var_Grad>(
      w.getDim(), Tensor::Initializer::NONE, false, false, name);
    if (w.getDim().rank() > 1) {
      Tensor w_trans = w.transpose(transpose_direction);
      vg->initialize(w_trans, g, false);
    } else {
      vg->initialize(w, g, false);
    }

    tf_node->appendInput(std::move(vg), true);
  }
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Name, props::Flatten, props::Distribute,
                   props::Trainable> &props,
  const LayerNode *self) {
  createIfNull(tf_node);
  tf_node->setInOut(*self);
  /** TODO: update to use run_context format for set inputs/outputs */
  // tf_node->setInputs(self->getObject()->getInputRef());
  // tf_node->setOutputs(self->getObject()->getOutputRef());

  saveTflWeights(tf_node.get(), self->getRunContext(), "0:2:1");
}

template <>
void Exporter::saveTflResult(const std::tuple<props::Unit> &props,
                             const FullyConnectedLayer *self) {
  createIfNull(tf_node);

  tf_node->setOpType(tflite::BuiltinOperator_FULLY_CONNECTED);
  /// we probably going to need flatbuffer inside exporter regarding this
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_FullyConnectedOptions,
                             flatbuffers::Offset<void>());
}
#endif

} // namespace nntrainer
