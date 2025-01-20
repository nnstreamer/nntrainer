// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   weight_layer.cpp
 * @date   2 August 2024
 * @brief  This is a layer that simply stores a weight tensor without any
 * operation.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>
#include <weight_layer.h>

#include <iostream>

namespace nntrainer {

WeightLayer::WeightLayer() : LayerImpl(), weight_props({}, {}, {}) {}

void WeightLayer::finalize(InitLayerContext &context) {
  auto &dims = std::get<std::vector<props::TensorDimension>>(weight_props);
  std::vector<TensorDim> t_dims(dims.begin(), dims.end());

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  auto &t_dtype = std::get<std::vector<props::TensorDataType>>(weight_props);
  auto &t_name = std::get<std::vector<props::WeightName>>(weight_props);

  NNTR_THROW_IF(!t_dims.size(), std::invalid_argument)
    << "Weight dimension is not provided.";
  n_weight = t_dims.size();

  if (!t_dtype.size()) {

    ml_logi("Set Weight Data Type provided by network");
    t_dtype.reserve(t_dims.size());
    for (auto t : t_dims)
      t_dtype.push_back(context.getWeightDataType());
  }
  auto engine = context.getComputeEngineType();

  NNTR_THROW_IF(t_dims.size() != t_dtype.size(), std::invalid_argument)
    << "Size of Dimension, Types must be same!";

  weight_idx.reserve(t_dims.size());

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);
    std::string name = context.getName() + "_w" + std::to_string(i);

    if (!t_name.empty())
      name = t_name[i];

    weight_idx.push_back(context.requestWeight(
      t_dims[i], weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, name, true));
  }

  context.setOutputDimensions(t_dims);
}

void WeightLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(weight_props, method, this);
}

void WeightLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, weight_props);
  LayerImpl::setProperty(remain_props);
}

void WeightLayer::forwarding(RunLayerContext &context, bool training) {
  if (!context.getInPlace()) {
    for (unsigned int i = 0; i < n_weight; ++i) {
      Tensor &input_ = context.getWeight(i);
      Tensor &output_ = context.getOutput(i);
      output_.copy(input_);
    }
  }
}

void WeightLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for weight layer is not supported");
}

void WeightLayer::calcGradient(RunLayerContext &context) {
  for (unsigned int i = 0; i < n_weight; ++i) {
    Tensor &djdw = context.getWeightGrad(i);
    const Tensor &derivative_ = context.getIncomingDerivative(i);
    djdw.copy(derivative_);
  }
}

} /* namespace nntrainer */
