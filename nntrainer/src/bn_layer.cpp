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
 * @file	bn_layer.cpp
 * @date	14 May 2020
 * @brief	This is Batch Normalization Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <assert.h>
#include <bn_layer.h>
#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int BatchNormalizationLayer::initialize(bool last) {
  int status = ML_ERROR_NONE;
  if (dim.batch() <= 0 || dim.height() <= 0 || dim.width() <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->gamma = Tensor(dim.batch(), dim.width());
  this->beta = Tensor(dim.batch(), dim.width());
  beta.setZero();
  gamma.setZero();

  return status;
}

int BatchNormalizationLayer::initialize(int b, int h, int w, bool last,
                                        bool init_zero) {
  int status = ML_ERROR_NONE;

  this->dim.batch(b);
  this->dim.width(w);
  this->dim.height(h);

  this->init_zero = init_zero;

  status = initialize(last);

  return status;
}

int BatchNormalizationLayer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  this->epsilon = 0.0;
  return this->opt.initialize(dim.height(), dim.width(), false);
}

int BatchNormalizationLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::input_shape:
      status = dim.setTensorDim(values[0].c_str());
      break;
    case PropertyType::bias_zero: {
      status = setBoolean(init_zero, value);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::epsilon:
      status = setFloat(epsilon, value);
      NN_RETURN_STATUS();
      break;
    default:
      ml_loge("Error: Unknown Layer Property Key: %s", key.c_str());
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

Tensor BatchNormalizationLayer::forwarding(Tensor in, int &status) {
  Tensor temp;
  assert(dim.batch() > 0);
  hidden = in;

  mu = in.sum(0).multiply(1.0 / dim.batch());

  temp = in.subtract(mu);

  var = temp.multiply(temp).sum(0).multiply(1.0 / dim.batch());

  Tensor hath = temp.divide(var.add(0.001).apply(sqrtFloat));

  hidden = hath;

  Tensor ret = hath.multiply(gamma).add(beta).apply(activation);

  return ret;
}

Tensor BatchNormalizationLayer::backwarding(Tensor derivative, int iteration) {
  Tensor dbeta;
  Tensor dgamma;
  assert(dim.batch() > 0);

  Tensor hath = hidden;
  Tensor dy =
    derivative.multiply(hath.multiply(gamma).add(beta).apply(activation_prime));

  dbeta = dy.sum(0);
  dgamma = (input.subtract(mu)
              .divide(var.add(0.001).apply(sqrtFloat))
              .multiply(dy)
              .sum(0));

  Tensor Temp =
    (dy.multiply(dim.batch()).subtract(dy.sum(0)))
      .subtract(input.subtract(mu)
                  .divide(var.add(0.001))
                  .multiply(dy.multiply(input.subtract(mu)).sum(0)));
  Tensor dh = Temp.multiply(1.0 / dim.batch())
                .multiply(var.add(0.001).apply(sqrtFloat))
                .multiply(gamma);

  float ll = opt.getLearningRate();
  if (opt.getDecaySteps() != -1) {
    ll = ll * pow(opt.getDecayRate(), (iteration / opt.getDecaySteps()));
  }

  gamma = gamma.subtract(dgamma.multiply(ll));
  beta = beta.subtract(dbeta.multiply(ll));

  return dh;
}

void BatchNormalizationLayer::read(std::ifstream &file) {
  file.read((char *)&mu, sizeof(float));
  file.read((char *)&var, sizeof(float));
  gamma.read(file);
  beta.read(file);
}

void BatchNormalizationLayer::save(std::ofstream &file) {
  file.write((char *)&mu, sizeof(float));
  file.write((char *)&var, sizeof(float));
  gamma.save(file);
  beta.save(file);
}

void BatchNormalizationLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<BatchNormalizationLayer> from =
    std::static_pointer_cast<BatchNormalizationLayer>(l);
  this->opt = from->opt;
  this->last_layer = from->last_layer;
  this->dim = from->dim;
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->weight.copy(from->weight);
  this->bias.copy(from->bias);
  this->mu = from->mu;
  this->var = from->var;
  this->gamma.copy(from->gamma);
  this->beta.copy(from->beta);
}
} /* namespace nntrainer */
