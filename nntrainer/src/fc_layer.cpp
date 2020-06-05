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
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int FullyConnectedLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  if (input_dim.batch() <= 0 || input_dim.height() <= 0 ||
      input_dim.width() <= 0 || input_dim.channel() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->last_layer = last;

  bias = Tensor(1, unit);
  dim = input_dim;
  dim.width(unit);
  dim.height(input_dim.width());
  weight = initializeWeight(dim, weight_ini_type, status);
  NN_RETURN_STATUS();

  output_dim.batch(input_dim.batch());
  output_dim.width(unit);

  if (init_zero) {
    bias.setZero();
  } else {
    bias = bias.apply(random);
  }
  return status;
}

int FullyConnectedLayer::initialize(int b, int c, int h, int w, bool last,
                                    bool init_zero) {
  int status = ML_ERROR_NONE;

  this->input_dim.batch(b);
  this->input_dim.channel(c);
  this->input_dim.width(w);
  this->input_dim.height(h);

  this->init_zero = init_zero;

  status = initialize(last);

  return status;
}

int FullyConnectedLayer::setCost(CostType c) {
  int status = ML_ERROR_NONE;
  if (c == COST_UNKNOWN) {
    ml_loge("Error: Unknown cost fucntion");
    return ML_ERROR_INVALID_PARAMETER;
  }
  cost = c;
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
    case PropertyType::bias_zero: {
      status = setBoolean(init_zero, value);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::activation:
      status = setActivation((ActiType)parseType(value, TOKEN_ACTI));
      NN_RETURN_STATUS();
      break;
    case PropertyType::weight_decay:
      weight_decay.type = (WeightDecayType)parseType(value, TOKEN_WEIGHT_DECAY);
      if (weight_decay.type == WeightDecayType::unknown) {
        ml_loge("Error: Unknown Weight Decay");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case PropertyType::weight_decay_lambda:
      status = setFloat(weight_decay.lambda, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::weight_init:
      weight_ini_type = (WeightIniType)parseType(value, TOKEN_WEIGHTINI);
      break;
    default:
      ml_loge("Error: Unknown Layer Property Key : %s", key.c_str());
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

Tensor FullyConnectedLayer::forwarding(Tensor in, int &status) {
  input = in;
  hidden = input.dot(weight).add(bias);

  if (this->bn_follow)
    return hidden;

  if (activation_type == ACT_SOFTMAX) {
    return hidden.apply(softmax);
  } else {
    return hidden.apply(activation);
  }
}

Tensor FullyConnectedLayer::forwarding(Tensor in, Tensor output, int &status) {
  if (!this->last_layer) {
    ml_loge("Error: Cannot update cost. This is not last layer of network");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  input = in;
  hidden = input.dot(weight).add(bias);
  Tensor y2 = output;
  Tensor y = hidden;

  if (activation_type == ACT_SOFTMAX) {
    y = y.apply(softmax);
  } else {
    y = y.apply(activation);
  }

  switch (cost) {
  case COST_MSR: {
    Tensor sub = y2.subtract(y);
    Tensor l = (sub.multiply(sub)).sum().multiply(0.5);

    updateLoss(l);
  } break;
  case COST_ENTROPY: {
    Tensor l;
    if (activation_type == ACT_SIGMOID) {
      l = (y2.multiply(y.apply(logFloat))
             .add((y2.multiply(-1.0).add(1.0))
                    .multiply((y.multiply(-1.0).add(1.0)).apply(logFloat))))
            .multiply(-1.0 / (y2.getWidth()))
            .sum();
    } else if (activation_type == ACT_SOFTMAX) {
      l =
        (y2.multiply(y.apply(logFloat))).multiply(-1.0 / (y2.getWidth())).sum();
    } else {
      ml_loge("Only support sigmoid & softmax for cross entropy loss");
      exit(0);
    }

    updateLoss(l);
  } break;
  case COST_UNKNOWN:
  default:
    break;
  }
  return y;
}

void FullyConnectedLayer::updateLoss(Tensor l) {
  float loss_sum = 0.0;
  std::vector<float> t = l.mat2vec();

  for (int i = 0; i < l.getBatch(); i++) {
    loss_sum += t[i];
  }
  loss = loss_sum / (float)l.getBatch();

  if (weight_decay.type == WeightDecayType::l2norm) {
    loss += weight_decay.lambda * 0.5 * (weight.l2norm());
  }
}

void FullyConnectedLayer::read(std::ifstream &file) {
  weight.read(file);
  bias.read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  weight.save(file);
  bias.save(file);
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
}

Tensor FullyConnectedLayer::backwarding(Tensor derivative, int iteration) {
  Tensor djdb;
  Tensor y2 = derivative;
  Tensor y;

  if (!last_layer) {
    djdb = derivative.multiply(hidden.apply(activation_prime));
  } else {
    if (activation_type == ACT_SOFTMAX)
      y = hidden.apply(softmax);
    else
      y = hidden.apply(activation);
    float ll = opt.getLearningRate();
    if (opt.getDecaySteps() != -1) {
      ll = ll * pow(opt.getDecayRate(), (iteration / opt.getDecaySteps()));
    }

    switch (cost) {
    case COST_MSR: {
      Tensor sub = y2.subtract(y);
      Tensor l = (sub.multiply(sub)).sum().multiply(0.5);

      updateLoss(l);

      if (activation_type == ACT_SOFTMAX) {
        djdb = y.subtract(y2).multiply(y.apply(softmaxPrime));
      } else {
        djdb = y.subtract(y2).multiply(hidden.apply(activation_prime));
      }
    } break;
    case COST_ENTROPY: {
      Tensor l;
      if (activation_type == ACT_SIGMOID) {
        djdb = y.subtract(y2).multiply(1.0 / y.getWidth());
        l = (y2.multiply(y.apply(logFloat))
               .add((y2.multiply(-1.0).add(1.0))
                      .multiply((y.multiply(-1.0).add(1.0)).apply(logFloat))))
              .multiply(-1.0 / (y2.getWidth()))
              .sum();
      } else if (activation_type == ACT_SOFTMAX) {
        djdb = y.subtract(y2).multiply(1.0 / y.getWidth());
        l = (y2.multiply(y.apply(logFloat)))
              .multiply(-1.0 / (y2.getWidth()))
              .sum();
      } else {
        ml_loge("Only support sigmoid & softmax for cross entropy loss");
        exit(0);
      }

      updateLoss(l);
    } break;
    case COST_UNKNOWN:
    default:
      break;
    }
  }

  Tensor ret = djdb.dot(weight.transpose("0:2:1"));

  Tensor djdw = input.transpose("0:2:1").dot(djdb);

  opt.calculate(djdw, djdb, weight, bias, iteration, this->init_zero,
                weight_decay);

  return ret;
}
} /* namespace nntrainer */
