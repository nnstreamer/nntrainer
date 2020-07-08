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

int FullyConnectedLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;

  this->last_layer = last;

  bias = Tensor(1, unit);
  dim = input_dim;
  dim.width(unit);
  dim.height(input_dim.width());
  dim.batch(1);
  weight = initializeWeight(dim, weight_ini_type, status);
  NN_RETURN_STATUS();

  output_dim = input_dim;
  output_dim.width(unit);

  if (bias_init_zero) {
    bias.setZero();
  } else {
    bias.setRandUniform(-0.5, 0.5);
  }

  weights.clear();
  weights.push_back(weight);
  weights.push_back(bias);

  return status;
}

int FullyConnectedLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::unit: {
      int width;
      status = setInt(width, value);
      NN_RETURN_STATUS();
      unit = width;
      output_dim.width(unit);
    } break;
    default:
      status = Layer::setProperty({values[i]});
      NN_RETURN_STATUS();
      break;
    }
  }
  return status;
}

int FullyConnectedLayer::setOptimizer(Optimizer &opt) {
  int status = Layer::setOptimizer(opt);
  if (status != ML_ERROR_NONE)
    return status;

  return this->opt.initialize(dim, true);
}

Tensor FullyConnectedLayer::forwarding(Tensor in, int &status) {
  input = in;
  hidden = input.chain().dot(weight).add_i(bias).run();
  status = ML_ERROR_NONE;

  if (weight_decay.type == WeightDecayType::l2norm) {
    loss = weight_decay.lambda * 0.5f * (weight.l2norm());
  }

  return hidden;
}

void FullyConnectedLayer::read(std::ifstream &file) {
  weight.read(file);
  bias.read(file);
  opt.read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  weight.save(file);
  bias.save(file);
  opt.save(file);
}

void FullyConnectedLayer::copy(std::shared_ptr<Layer> l) {
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
  this->weight.copy(from->weight);
  this->bias.copy(from->bias);
  this->loss = from->loss;
  this->cost = from->cost;
}

Tensor FullyConnectedLayer::backwarding(Tensor derivative, int iteration) {
  Tensor ret = derivative.dot(weight.transpose("0:2:1"));
  Tensor djdb = derivative;
  Tensor djdw = input.chain()
                  .transpose("0:2:1")
                  .dot(derivative)
                  .applyIf(this->isWeightDecayL2Norm(), _LIFT(add_i), weight,
                           weight_decay.lambda)
                  .run();

  if (trainable) {
    gradients.clear();
    gradients.push_back(djdw);
    gradients.push_back(djdb);

    opt.apply_gradients(weights, gradients, iteration);
  }

  return ret;
}

void FullyConnectedLayer::print(std::ostream &out) const {
  Layer::print(out);

  out << "unit: " << unit << std::endl << "= weight info =" << std::endl;

  weight.print(out);
  bias.print(out);
}
} /* namespace nntrainer */
