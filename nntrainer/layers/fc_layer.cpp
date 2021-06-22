/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <fc_layer.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

enum FCParams { weight, bias };

void FullyConnectedLayer::finalize(InitLayerContext &context) {
  auto unit = std::get<props::Unit>(fc_props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("Fully connected layer takes only one input");
  }

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  output_dims[0].width(unit);
  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  TensorDim bias_dim(1, 1, 1, unit, 0b0001);
  TensorDim weight_dim(1, 1, in_dim.width(), unit, 0b0011);

  weight_idx[FCParams::weight] =
    context.requestWeight(weight_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "FC:weight", true);

  weight_idx[FCParams::bias] = context.requestWeight(
    bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, "FC:bias", true);

  // if (weights.empty()) {
  //   weights.reserve(2);
  //   weights.emplace_back(weight_dim, weight_initializer, weight_regularizer,
  //                        weight_regularizer_constant, true, false,
  //                        "FC:weight");
  //   weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
  //                        1.0f, true, false, "FC:bias");
  //   manager.trackWeights(weights);
  // } else {
  //   weights[FCParams::weight].reset(weight_dim, weight_initializer,
  //                                   weight_regularizer,
  //                                   weight_regularizer_constant, true);
  //   weights[FCParams::bias].reset(bias_dim, bias_initializer,
  //                                 WeightRegularizer::NONE, 1.0f, true);
  // }
}

void FullyConnectedLayer::exportTo(Exporter &exporter,
                                   const ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);

  Tensor &hidden_ = context.getOutput(0);
  Tensor &input_ = context.getInput(0);

  input_.dot(weight, hidden_);
  hidden_.add_i(bias);
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  Tensor &derivative_ = context.getIncomingDerivative(0);
  Tensor &ret_ = context.getOutgoingDerivative(0);

  ret_ = derivative_.dot(weight, ret_, false, true);
}

void FullyConnectedLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
  Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);

  Tensor &derivative_ = context.getIncomingDerivative(0);
  Tensor &input_ = context.getInput(0);

  derivative_.sum({0, 1, 2}, djdb);
  input_.dot(derivative_, djdw, true, false);
}

} /* namespace nntrainer */
