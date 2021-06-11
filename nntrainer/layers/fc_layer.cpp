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

int FullyConnectedLayer::initialize(Manager &manager) {
  auto unit = std::get<props::Unit>(fc_props).get();
  int status = ML_ERROR_NONE;

  if (getNumInputs() != 1) {
    throw std::invalid_argument("Fully connected layer takes only one input");
  }

  auto &in_dim = input_dim[0];
  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  in_dim.setEffDimFlag(0b1001);
  in_dim.setDynDimFlag(0b1000);

  output_dim[0] = in_dim;
  output_dim[0].width(unit);

  TensorDim bias_dim(1, 1, 1, unit, 0b0001);
  TensorDim weight_dim(1, 1, in_dim.width(), unit, 0b0011);

  if (weights.empty()) {
    weights.reserve(2);
    weights.emplace_back(weight_dim, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, false, "FC:weight");
    weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
                         1.0f, true, false, "FC:bias");
    manager.trackWeights(weights);
  } else {
    weights[FCParams::weight].reset(weight_dim, weight_initializer,
                                    weight_regularizer,
                                    weight_regularizer_constant, true);
    weights[FCParams::bias].reset(bias_dim, bias_initializer,
                                  WeightRegularizer::NONE, 1.0f, true);
  }

  return status;
}

void FullyConnectedLayer::export_to(Exporter &exporter,
                                    ExportMethods method) const {
  LayerV1::export_to(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const PropertyType type,
                                      const std::string &value) {
  switch (type) {
  case PropertyType::unit: {
    from_string(value, std::get<props::Unit>(fc_props));
  } break;
  default:
    LayerV1::setProperty(type, value);
    break;
  }
}

void FullyConnectedLayer::forwarding(bool training) {
  Tensor &weight =
    weightAt(static_cast<int>(FCParams::weight)).getVariableRef();
  Tensor &bias = weightAt(static_cast<int>(FCParams::bias)).getVariableRef();

  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();
  input_.dot(weight, hidden_);
  hidden_.add_i(bias);

  loss = weightAt(static_cast<int>(FCParams::weight)).getRegularizationLoss();
}

void FullyConnectedLayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

  std::shared_ptr<FullyConnectedLayer> from =
    std::static_pointer_cast<FullyConnectedLayer>(l);

  std::get<props::Unit>(fc_props) = std::get<props::Unit>(from->fc_props);
}

void FullyConnectedLayer::calcDerivative() {
  unsigned int weight_idx = static_cast<int>(FCParams::weight);
  Tensor &weight = weightAt(weight_idx).getVariableRef();
  Tensor &derivative_ = net_hidden[0]->getGradientRef();
  Tensor &ret_ = net_input[0]->getGradientRef();

  ret_ = derivative_.dot(weight, ret_, false, true);
}

void FullyConnectedLayer::calcGradient() {
  unsigned int weight_idx = static_cast<int>(FCParams::weight);
  unsigned int bias_idx = static_cast<int>(FCParams::bias);
  Tensor &djdw = weightAt(weight_idx).getGradientRef();
  Tensor &djdb = weightAt(bias_idx).getGradientRef();

  Tensor &derivative_ = net_hidden[0]->getGradientRef();

  derivative_.sum({0, 1, 2}, djdb);
  net_input[0]->getVariableRef().dot(derivative_, djdw, true, false);
}

void FullyConnectedLayer::scaleSize(float scalesize) noexcept {
  auto &unit = std::get<props::Unit>(fc_props).get();
  unit = (unsigned int)(scalesize * (float)unit);
  unit = std::max(unit, 1u);
}

} /* namespace nntrainer */
