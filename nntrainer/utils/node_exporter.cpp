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
#include <node_exporter.h>
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
Exporter::getResult<ExportMethods::METHOD_STRINGVECTOR>() {
  return std::move(stored_result);
}

#ifdef ENABLE_TFLITE_INTERPRETER
template <>
std::unique_ptr<TfOpNode> Exporter::getResult<ExportMethods::METHOD_TFLITE>() {
  tf_node->finalize();
  return std::move(tf_node);
}

template <>
void Exporter::saveTflResult(const std::tuple<> &props,
                             const nntrainer::Layer *self) {
  createIfNull(tf_node);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Name, props::Distribute, props::Trainable,
                   std::vector<props::InputConnection>,
                   std::vector<props::InputShape>, props::SharedFrom,
                   props::ClipGradByGlobalNorm> &props,
  const LayerNode *self) {
  createIfNull(tf_node);
  tf_node->setLayerNode(*self);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::WeightRegularizer, props::WeightRegularizerConstant,
                   props::WeightInitializer, props::WeightDecay,
                   props::BiasDecay, props::BiasInitializer, props::DisableBias>
    &props,
  const LayerImpl *self) { /// layer impl has nothing to serialize so do nothing
}

template <>
void Exporter::saveTflResult(const std::tuple<props::Unit> &props,
                             const FullyConnectedLayer *self) {
  createIfNull(tf_node);

  auto weight_transform = [](std::vector<const Tensor *> &weights) {
    std::vector<Tensor> new_weights;
    new_weights.reserve(weights.size());

    // std::cerr << "weights! " << weights.size() << ' ' <<
    // new_weights.capacity() << std::endl; std::transform(weights.begin(),
    // weights.end(),
    //                std::back_inserter(new_weights),
    //                [](const Tensor *t) { return t->clone(); });
    // std::cerr << "22\n";
    new_weights.push_back(weights[0]->transpose("0:2:1"));
    new_weights.push_back(*weights[1]);
    // std::cerr << "33\n";
    return new_weights;
  };
  tf_node->setWeightTransformFn(weight_transform);

  tf_node->setOpType(tflite::BuiltinOperator_FULLY_CONNECTED);
  /// we probably going to need flatbuffer inside exporter regarding this
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_FullyConnectedOptions,
                             flatbuffers::Offset<void>());
}
#endif

} // namespace nntrainer
