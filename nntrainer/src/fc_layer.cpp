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
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

enum class FCParams { weight, bias };

int FullyConnectedLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;

  this->last_layer = last;

  Tensor bias = Tensor(1, unit);
  dim = input_dim;
  dim.width(unit);
  dim.height(input_dim.width());
  dim.batch(1);
  Tensor weight = initializeWeight(dim, weight_ini_type, status);
  NN_RETURN_STATUS();

  output_dim = input_dim;
  output_dim.width(unit);

  if (bias_init_zero) {
    bias.setZero();
  } else {
    bias.setRandUniform(-0.5, 0.5);
  }

  setParamSize(2);
  paramsAt(0) = {std::move(weight), Tensor(weight.getDim()), "FC:weight"};
  paramsAt(1) = {std::move(bias), Tensor(bias.getDim()), "FC:bias"};

  return status;
}

void FullyConnectedLayer::setProperty(const PropertyType type,
                                      const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::unit: {
    if (!value.empty()) {
      int width;
      status = setInt(width, value);
      throw_status(status);
      unit = width;
      output_dim.width(unit);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

Tensor FullyConnectedLayer::forwarding(Tensor in, int &status) {
  Tensor &weight = paramsAt(static_cast<int>(FCParams::weight)).weight;
  Tensor &bias = paramsAt(static_cast<int>(FCParams::bias)).weight;

  input = in;
  hidden = input.chain().dot(weight).add_i(bias).run();
  status = ML_ERROR_NONE;

  if (weight_decay.type == WeightDecayType::l2norm) {
    loss = weight_decay.lambda * 0.5f * (weight.l2norm());
  }

  return hidden;
}

void FullyConnectedLayer::read(std::ifstream &file) {
  Layer::read(file);
  opt.read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  Layer::save(file);
  opt.save(file);
}

void FullyConnectedLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<FullyConnectedLayer> from =
    std::static_pointer_cast<FullyConnectedLayer>(l);
  this->opt = from->opt;
  this->last_layer = from->last_layer;
  this->dim = from->dim;
  this->unit = from->unit;
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->loss = from->loss;
  this->cost = from->cost;
}

Tensor FullyConnectedLayer::backwarding(Tensor derivative, int iteration) {
  unsigned int weight_idx = static_cast<int>(FCParams::weight);
  unsigned int bias_idx = static_cast<int>(FCParams::bias);
  Tensor &weight = paramsAt(weight_idx).weight;
  Tensor &djdw = paramsAt(weight_idx).grad;
  Tensor &djdb = paramsAt(bias_idx).grad;

  Tensor ret = derivative.dot(weight.transpose("0:2:1"));
  djdb = derivative.sum(0);
  djdw = input.chain()
           .transpose("0:2:1")
           .dot(derivative)
           .applyIf(this->isWeightDecayL2Norm(), _LIFT(add_i), weight,
                    weight_decay.lambda)
           .run()
           .sum(0);

  if (trainable) {
    opt.apply_gradients(params, param_size, iteration);
  }

  return ret;
}
} /* namespace nntrainer */
