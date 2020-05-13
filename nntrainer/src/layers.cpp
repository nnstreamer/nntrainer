/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	layers.cpp
 * @date	04 December 2019
 * @brief	This is Layers Classes for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <assert.h>
#include <cstring>
#include <layers.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <random>
#include <util_func.h>

namespace nntrainer {

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  return rng;
}();

template <typename... Args> static void RandNormal(Tensor &w, Args &&... args) {
  std::normal_distribution<float> dist(std::forward<Args>(args)...);
  unsigned int width = w.getWidth();
  unsigned int height = w.getHeight();

  for (unsigned int i = 0; i < width; ++i) {
    for (unsigned int j = 0; j < height; ++j) {
      w.setValue(0, j, i, dist(rng));
    }
  }
}

template <typename... Args>
static void RandUniform(Tensor &w, Args &&... args) {
  std::uniform_real_distribution<float> dist(std::forward<Args>(args)...);
  unsigned int width = w.getWidth();
  unsigned int height = w.getHeight();

  for (unsigned int i = 0; i < width; ++i) {
    for (unsigned int j = 0; j < height; ++j) {
      w.setValue(0, j, i, dist(rng));
    }
  }
}

static Tensor weightInitialization(unsigned int width, unsigned int height,
                                   WeightIniType init_type, int &status) {

  Tensor w = Tensor(height, width);

  if (init_type == WEIGHT_UNKNOWN) {
    ml_logw("Warning: Weight Initalization Type is not set. "
            "WEIGHT_XAVIER_NORMAL is used by default");
    init_type = WEIGHT_XAVIER_NORMAL;
  }

  switch (init_type) {
  case WEIGHT_LECUN_NORMAL:
    RandNormal(w, 0, sqrt(1.0 / height));
    break;
  case WEIGHT_XAVIER_NORMAL:
    RandNormal(w, 0, sqrt(2.0 / (width + height)));
    break;
  case WEIGHT_HE_NORMAL:
    RandNormal(w, 0, sqrt(2.0 / (height)));
    break;
  case WEIGHT_LECUN_UNIFORM:
    RandUniform(w, -1.0 * sqrt(1.0 / height), sqrt(1.0 / height));
    break;
  case WEIGHT_XAVIER_UNIFORM:
    RandUniform(w, -1.0 * sqrt(6.0 / (height + width)),
                sqrt(6.0 / (height + width)));
    break;
  case WEIGHT_HE_UNIFORM:
    RandUniform(w, -1.0 * sqrt(6.0 / (height)), sqrt(6.0 / (height)));
    break;
  default:
    break;
  }
  return w;
}

Layer::Layer() {
  type = LAYER_UNKNOWN;
  activation_type = ACT_UNKNOWN;
  last_layer = false;
  dim.batch(0);
  dim.channel(0);
  dim.width(0);
  dim.height(0);
  init_zero = false;
  activation = NULL;
  activation_prime = NULL;
  bn_fallow = false;
  weight_decay.type = WeightDecayType::unknown;
  weight_decay.lambda = 0.0;
}

int Layer::setActivation(ActiType acti) {
  int status = ML_ERROR_NONE;
  if (acti == ACT_UNKNOWN) {
    ml_loge("Error:have to specify activation function");
    return ML_ERROR_INVALID_PARAMETER;
  }
  activation_type = acti;
  switch (acti) {
  case ACT_TANH:
    activation = tanhFloat;
    activation_prime = tanhPrime;
    break;
  case ACT_SIGMOID:
    activation = sigmoid;
    activation_prime = sigmoidePrime;
    break;
  case ACT_RELU:
    activation = relu;
    activation_prime = reluPrime;
    break;
  default:
    break;
  }
  return status;
}

int Layer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  return this->opt.initialize(dim.height(), dim.width(), true);
}

int Layer::checkValidation() {
  int status = ML_ERROR_NONE;
  if (type == LAYER_UNKNOWN) {
    ml_loge("Error: Layer type is unknown");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (activation_type == ACT_UNKNOWN) {
    ml_loge("Error: Have to set activation for this layer");
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (dim.batch() == 0 || dim.width() == 0 || dim.height() == 0) {
    ml_loge("Error: Tensor Dimension must be set before initialization");
    return ML_ERROR_INVALID_PARAMETER;
  }
  return status;
}

int InputLayer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  return this->opt.initialize(dim.height(), dim.width(), false);
}

int InputLayer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key.c_str());

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::input_shape:
      status = dim.setTensorDim(value.c_str());
      NN_RETURN_STATUS();
      break;
    case PropertyType::bias_zero:
      status = setBoolean(init_zero, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::normalization:
      status = setBoolean(normalization, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::standardization:
      status = setBoolean(standardization, value);
      NN_RETURN_STATUS();
      break;
    default:
      ml_loge("Error: Unknown Layer Property Key");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

void InputLayer::copy(std::shared_ptr<Layer> l) {
  std::shared_ptr<InputLayer> from = std::static_pointer_cast<InputLayer>(l);
  this->opt = from->opt;
  this->last_layer = from->last_layer;
  this->dim = from->dim;
  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
}

Tensor InputLayer::forwarding(Tensor in, int &status) {
  input = in;
  if (normalization)
    input = input.normalization();
  return input;
}

int InputLayer::initialize(int b, int h, int w, bool last, bool init_zero,
                           WeightIniType wini) {
  int status = ML_ERROR_NONE;
  if (b <= 0 || h <= 0 || w <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->dim.batch(b);
  this->dim.width(w);
  this->dim.height(h);
  this->last_layer = last;
  this->bn_fallow = false;
  return status;
}

int FullyConnectedLayer::initialize(int b, int h, int w, bool last,
                                    bool init_zero, WeightIniType wini) {
  int status = ML_ERROR_NONE;
  if (b <= 0 || h <= 0 || w <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->dim.batch(b);
  this->dim.width(w);
  this->dim.height(h);

  this->last_layer = last;
  this->init_zero = init_zero;
  this->bn_fallow = false;

  bias = Tensor(1, w);
  weight = weightInitialization(w, h, wini, status);

  if (status != ML_ERROR_NONE)
    return status;

  if (init_zero) {
    bias.setZero();
  } else {
    bias = bias.apply(random);
  }
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
    case PropertyType::input_shape:
      status = dim.setTensorDim(value);
      break;
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
    default:
      ml_loge("Error: Unknown Layer Property Key");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  return status;
}

Tensor FullyConnectedLayer::forwarding(Tensor in, int &status) {
  input = in;
  hidden = input.dot(weight).add(bias);

  if (this->bn_fallow)
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

  Tensor ret = djdb.dot(weight.transpose());

  Tensor djdw = input.transpose().dot(djdb);

  opt.calculate(djdw, djdb, weight, bias, iteration, this->init_zero,
                weight_decay);

  return ret;
}

int BatchNormalizationLayer::initialize(int b, int h, int w, bool last,
                                        bool init_zero, WeightIniType wini) {
  int status = ML_ERROR_NONE;
  if (b <= 0 || h <= 0 || w <= 0) {
    ml_loge("Error: Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->dim.batch(b);
  this->dim.width(w);
  this->dim.height(h);

  this->last_layer = last;
  this->init_zero = init_zero;

  this->gamma = Tensor(b, w);
  this->beta = Tensor(b, w);
  beta.setZero();
  gamma.setZero();
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
      ml_loge("Error: Unknown Layer Property Key");
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
