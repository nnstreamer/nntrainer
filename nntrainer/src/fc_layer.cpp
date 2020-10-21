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

enum class FCParams { weight, bias };

int FullyConnectedLayer::initialize() {
  int status = ML_ERROR_NONE;

  output_dim = input_dim;
  output_dim.width(unit);

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit);

  TensorDim dim = output_dim;
  dim.height(input_dim.width());
  dim.batch(1);

  setNumWeights(2);
  weightAt(0) = std::move(Weight(dim, weight_initializer, true, "FC:weight"));
  weightAt(1) = std::move(Weight(bias_dim, bias_initializer, true, "FC:bias"));

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
      output_dim.width(unit);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

sharedConstTensors FullyConnectedLayer::forwarding(sharedConstTensors in) {
  Tensor &weight =
    weightAt(static_cast<int>(FCParams::weight)).getVariableRef();
  Tensor &bias = weightAt(static_cast<int>(FCParams::bias)).getVariableRef();

  input = *in[0];
  hidden = input.dot(weight);
  hidden.add_i(bias);

  if (weight_regularizer == WeightRegularizerType::l2norm) {
    loss = weight_regularizer_constant * 0.5f * (weight.l2norm());
  }

  return {MAKE_SHARED_TENSOR(hidden)};
}

void FullyConnectedLayer::read(std::ifstream &file) {
  Layer::read(file);
  if (opt)
    opt->read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  Layer::save(file);
  if (opt)
    opt->save(file);
}

void FullyConnectedLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<FullyConnectedLayer> from =
    std::static_pointer_cast<FullyConnectedLayer>(l);
  this->unit = from->unit;
}

sharedConstTensors
FullyConnectedLayer::backwarding(sharedConstTensors derivative, int iteration) {
  unsigned int weight_idx = static_cast<int>(FCParams::weight);
  unsigned int bias_idx = static_cast<int>(FCParams::bias);
  Tensor &weight = weightAt(weight_idx).getVariableRef();
  Tensor &djdw = weightAt(weight_idx).getGradientRef();
  Tensor &djdb = weightAt(bias_idx).getGradientRef();

  Tensor ret = derivative[0]->dot(weight, false, true);
  djdb = derivative[0]->sum(0);

  djdw = input.dot(*derivative[0], true, false);
  if (isWeightRegularizerL2Norm())
    djdw.add_i(weight, weight_regularizer_constant);
  djdw = djdw.sum(0);

  if (trainable) {
    opt->apply_gradients(weight_list, num_weights, iteration);
  }

  return {MAKE_SHARED_TENSOR(std::move(ret))};
}
} /* namespace nntrainer */
