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

const std::string FullyConnectedLayer::type = "fully_connected";

enum class FCParams { weight, bias };

int FullyConnectedLayer::initialize() {
  int status = ML_ERROR_NONE;

  if (num_inputs != 1) {
    throw std::invalid_argument("Fully connected layer takes only one input");
  }

  output_dim[0] = input_dim[0];
  output_dim[0].width(unit);

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit);

  TensorDim dim = output_dim[0];
  dim.height(input_dim[0].width());
  dim.batch(1);

  setNumWeights(2);
  weightAt(static_cast<int>(FCParams::weight)) =
    std::move(Weight(dim, weight_initializer, true, "FC:weight"));
  weightAt(static_cast<int>(FCParams::bias)) =
    std::move(Weight(bias_dim, bias_initializer, true, "FC:bias"));

  return status;
}

void FullyConnectedLayer::setProperty(const PropertyType type,
                                      const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::unit: {
    if (!value.empty()) {
      status = setUint(unit, value);
      throw_status(status);
      output_dim[0].width(unit);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void FullyConnectedLayer::forwarding(sharedConstTensors in) {
  Tensor &weight =
    weightAt(static_cast<int>(FCParams::weight)).getVariableRef();
  Tensor &bias = weightAt(static_cast<int>(FCParams::bias)).getVariableRef();

  Tensor &hidden_ = net_hidden[0]->var;
  Tensor &input_ = net_input[0]->var;
  hidden_ = input_.dot(weight, hidden_);
  hidden_.add_i(bias);

  if (weight_regularizer == WeightRegularizerType::l2norm) {
    loss = weight_regularizer_constant * 0.5f * (weight.l2norm());
  }
}

void FullyConnectedLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<FullyConnectedLayer> from =
    std::static_pointer_cast<FullyConnectedLayer>(l);
  this->unit = from->unit;
}

void FullyConnectedLayer::calcDerivative(sharedConstTensors derivative) {
  unsigned int weight_idx = static_cast<int>(FCParams::weight);
  Tensor &weight = weightAt(weight_idx).getVariableRef();
  Tensor &derivative_ = net_hidden[0]->grad;
  Tensor &ret_ = net_input[0]->grad;

  ret_ = derivative_.dot(weight, ret_, false, true);
}

void FullyConnectedLayer::calcGradient(sharedConstTensors derivative) {
  unsigned int weight_idx = static_cast<int>(FCParams::weight);
  unsigned int bias_idx = static_cast<int>(FCParams::bias);
  Tensor &weight = weightAt(weight_idx).getVariableRef();
  Tensor &djdw = weightAt(weight_idx).getGradientRef();
  Tensor &djdb = weightAt(bias_idx).getGradientRef();

  Tensor &derivative_ = net_hidden[0]->grad;

  djdb = derivative_.sum(0);
  djdw = net_input[0]->var.dot(derivative_, djdw, true, false);

  if (isWeightRegularizerL2Norm())
    djdw.add_i(weight, weight_regularizer_constant);
}

void FullyConnectedLayer::scaleSize(float scalesize) noexcept {
  unit = (unsigned int)(scalesize * (float)unit);
  unit = std::max(unit, 1u);
}

} /* namespace nntrainer */
