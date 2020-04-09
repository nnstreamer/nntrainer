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

#include "layers.h"
#include <assert.h>
#include <nntrainer_log.h>
#include <random>
#include "util_func.h"

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  return rng;
}();

template <typename... Args>
static void RandNormal(Tensors::Tensor &W, Args &&... args) {
  std::normal_distribution<float> dist(std::forward<Args>(args)...);
  unsigned int width = W.getWidth();
  unsigned int height = W.getHeight();

  for (unsigned int i = 0; i < width; ++i) {
    for (unsigned int j = 0; j < height; ++j) {
      W.setValue(0, j, i, dist(rng));
    }
  }
}

template <typename... Args>
static void RandUniform(Tensors::Tensor &W, Args &&... args) {
  std::uniform_real_distribution<float> dist(std::forward<Args>(args)...);
  unsigned int width = W.getWidth();
  unsigned int height = W.getHeight();

  for (unsigned int i = 0; i < width; ++i) {
    for (unsigned int j = 0; j < height; ++j) {
      W.setValue(0, j, i, dist(rng));
    }
  }
}

static Tensors::Tensor WeightInitialization(unsigned int width, unsigned int height, Layers::weightIni_type init_type) {
  Tensors::Tensor W = Tensors::Tensor(height, width);

  switch (init_type) {
    case Layers::WEIGHT_LECUN_NORMAL:
      RandNormal(W, 0, sqrt(1 / height));
      break;
    case Layers::WEIGHT_XAVIER_NORMAL:
      RandNormal(W, 0, sqrt(2.0 / (width + height)));
      break;
    case Layers::WEIGHT_HE_NORMAL:
      RandNormal(W, 0, sqrt(2.0 / (height)));
      break;
    case Layers::WEIGHT_LECUN_UNIFORM:
      RandUniform(W, -1.0 * sqrt(1.0 / height), sqrt(1.0 / height));
      break;
    case Layers::WEIGHT_XAVIER_UNIFORM:
      RandUniform(W, -1.0 * sqrt(6.0 / (height + width)), sqrt(6.0 / (height + width)));
      break;
    case Layers::WEIGHT_HE_UNIFORM:
      RandUniform(W, -1.0 * sqrt(6.0 / (height)), sqrt(6.0 / (height)));
      break;
    default:
      break;
  }
  return W;
}

namespace Layers {

void Layer::setActivation(acti_type acti) {
  if (acti == ACT_UNKNOWN) {
    ml_loge("have to specify activation function");
    exit(0);
  }
  activation_type = acti;
  switch (acti) {
    case ACT_TANH:
      activation = tanh_float;
      activationPrime = tanhPrime;
      break;
    case ACT_SIGMOID:
      activation = sigmoid;
      activationPrime = sigmoidePrime;
      break;
    case ACT_RELU:
      activation = Relu;
      activationPrime = ReluPrime;
      break;
    default:
      break;
  }
}

void Layer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  this->opt.initialize(height, width, true);
}

void InputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  this->opt.initialize(height, width, false);
}

void InputLayer::copy(Layer *l) {
  InputLayer *from = static_cast<InputLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
}

Tensors::Tensor InputLayer::forwarding(Tensors::Tensor input) {
  Input = input;
  if (normalization)
    Input = Input.normalization();
  return Input;
}

void InputLayer::initialize(int b, int h, int w, int id, bool init_zero, weightIni_type wini) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = 0;
  this->bnfallow = false;
}

void FullyConnectedLayer::initialize(int b, int h, int w, int id, bool init_zero, weightIni_type wini) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;
  this->bnfallow = false;

  Bias = Tensors::Tensor(1, w);
  Weight = WeightInitialization(w, h, wini);

  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.apply(random);
  }
}

Tensors::Tensor FullyConnectedLayer::forwarding(Tensors::Tensor input) {
  Input = input;
  hidden = Input.dot(Weight).add(Bias);

  if (this->bnfallow)
    return hidden;

  return hidden.apply(activation);
  ;
}

void FullyConnectedLayer::read(std::ifstream &file) {
  Weight.read(file);
  Bias.read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  Weight.save(file);
  Bias.save(file);
}

void FullyConnectedLayer::copy(Layer *l) {
  FullyConnectedLayer *from = static_cast<FullyConnectedLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
}

Tensors::Tensor FullyConnectedLayer::backwarding(Tensors::Tensor derivative, int iteration) {
  Tensors::Tensor dJdB;

  dJdB = derivative.multiply(hidden.apply(activationPrime));

  Tensors::Tensor ret = dJdB.dot(Weight.transpose());

  Tensors::Tensor dJdW = Input.transpose().dot(dJdB);

  opt.calculate(dJdW, dJdB, Weight, Bias, iteration, this->init_zero);

  return ret;
}

void OutputLayer::initialize(int b, int h, int w, int id, bool init_zero, weightIni_type wini) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;

  Bias = Tensors::Tensor(1, w);
  this->cost = cost;
  this->bnfallow = false;

  Weight = WeightInitialization(w, h, wini);

  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.apply(random);
  }
}

Tensors::Tensor OutputLayer::forwarding(Tensors::Tensor input) {
  Input = input;
  hidden = input.dot(Weight).add(Bias);
  if (activation_type == ACT_SOFTMAX) {
    return hidden.apply(softmax);
  } else {
    return hidden.apply(activation);
  }
}

Tensors::Tensor OutputLayer::forwarding(Tensors::Tensor input, Tensors::Tensor output) {
  Input = input;
  hidden = input.dot(Weight).add(Bias);
  Tensors::Tensor Y2 = output;
  Tensors::Tensor Y = hidden;

  if (activation_type == ACT_SOFTMAX) {
    Y = Y.apply(softmax);
  } else {
    Y = Y.apply(activation);
  }

  float lossSum = 0.0;

  switch (cost) {
    case COST_MSR: {
      Tensors::Tensor sub = Y2.subtract(Y);
      Tensors::Tensor l = (sub.multiply(sub)).sum().multiply(0.5);
      std::vector<float> t = l.Mat2Vec();
      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }

      loss = lossSum / (float)l.getBatch();
    } break;
    case COST_ENTROPY: {
      Tensors::Tensor l;
      if (activation_type == ACT_SIGMOID) {
        l = (Y2.multiply(Y.apply(log_float))
                 .add((Y2.multiply(-1.0).add(1.0)).multiply((Y.multiply(-1.0).add(1.0)).apply(log_float))))
                .multiply(-1.0 / (Y2.getWidth()))
                .sum();
      } else if (activation_type == ACT_SOFTMAX) {
        l = (Y2.multiply(Y.apply(log_float))).multiply(-1.0 / (Y2.getWidth())).sum();
      } else {
        ml_loge("Only support sigmoid & softmax for cross entropy loss");
        exit(0);
      }

      std::vector<float> t = l.Mat2Vec();

      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }
      loss = lossSum / (float)l.getBatch();

      if (opt.getWeightDecayType() == WeightDecayType::l2norm) {
        loss += opt.getWeightDecayLambda() * 0.5 * (Weight.l2norm());
      }

    } break;
    case COST_UNKNOWN:
    default:
      break;
  }
  return Y;
}

void OutputLayer::read(std::ifstream &file) {
  Weight.read(file);
  Bias.read(file);
}

void OutputLayer::save(std::ofstream &file) {
  Weight.save(file);
  Bias.save(file);
}

void OutputLayer::copy(Layer *l) {
  OutputLayer *from = static_cast<OutputLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
  this->loss = from->loss;
}

Tensors::Tensor OutputLayer::backwarding(Tensors::Tensor label, int iteration) {
  float lossSum = 0.0;
  Tensors::Tensor Y2 = label;
  Tensors::Tensor Y;
  if (activation_type == ACT_SOFTMAX)
    Y = hidden.apply(softmax);
  else
    Y = hidden.apply(activation);

  Tensors::Tensor ret;
  Tensors::Tensor dJdB;

  float ll = opt.getLearningRate();
  if (opt.getDecaySteps() != -1) {
    ll = ll * pow(opt.getDecayRate(), (iteration / opt.getDecaySteps()));
  }

  switch (cost) {
    case COST_MSR: {
      Tensors::Tensor sub = Y2.subtract(Y);
      Tensors::Tensor l = (sub.multiply(sub)).sum().multiply(0.5);
      std::vector<float> t = l.Mat2Vec();
      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }

      loss = lossSum / (float)l.getBatch();
      if (opt.getWeightDecayType() == WeightDecayType::l2norm) {
        loss += opt.getWeightDecayLambda() * 0.5 * (Weight.l2norm());
      }
      if (activation_type == ACT_SOFTMAX) {
        dJdB = Y.subtract(Y2).multiply(Y.apply(softmaxPrime));
      } else {
        dJdB = Y.subtract(Y2).multiply(hidden.apply(activationPrime));
      }
    } break;
    case COST_ENTROPY: {
      Tensors::Tensor l;
      if (activation_type == ACT_SIGMOID) {
        dJdB = Y.subtract(Y2).multiply(1.0 / Y.getWidth());
        l = (Y2.multiply(Y.apply(log_float))
                 .add((Y2.multiply(-1.0).add(1.0)).multiply((Y.multiply(-1.0).add(1.0)).apply(log_float))))
                .multiply(-1.0 / (Y2.getWidth()))
                .sum();
      } else if (activation_type == ACT_SOFTMAX) {
        dJdB = Y.subtract(Y2).multiply(1.0 / Y.getWidth());
        l = (Y2.multiply(Y.apply(log_float))).multiply(-1.0 / (Y2.getWidth())).sum();
      } else {
        ml_loge("Only support sigmoid & softmax for cross entropy loss");
        exit(0);
      }

      std::vector<float> t = l.Mat2Vec();

      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }
      loss = lossSum / (float)l.getBatch();

      if (opt.getWeightDecayType() == WeightDecayType::l2norm) {
        loss += opt.getWeightDecayLambda() * 0.5 * (Weight.l2norm());
      }

    } break;
    case COST_UNKNOWN:
    default:
      break;
  }

  Tensors::Tensor dJdW = Input.transpose().dot(dJdB);

  ret = dJdB.dot(Weight.transpose());

  opt.calculate(dJdW, dJdB, Weight, Bias, iteration, this->init_zero);

  return ret;
}

void BatchNormalizationLayer::initialize(int b, int h, int w, int id, bool init_zero, weightIni_type wini) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;

  this->gamma = Tensors::Tensor(batch, w);
  this->beta = Tensors::Tensor(batch, w);
  beta.setZero();
  gamma.setZero();
}

void BatchNormalizationLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  this->opt.initialize(height, width, false);
}

Tensors::Tensor BatchNormalizationLayer::forwarding(Tensors::Tensor input) {
  Tensors::Tensor temp;

  hidden = input;

  mu = input.sum(0).multiply(1.0 / batch);

  temp = input.subtract(mu);

  var = temp.multiply(temp).sum(0).multiply(1.0 / batch);

  Tensors::Tensor hath = temp.divide(var.add(0.001).apply(sqrt_float));

  hidden = hath;

  Tensors::Tensor ret = hath.multiply(gamma).add(beta).apply(activation);

  return ret;
}

Tensors::Tensor BatchNormalizationLayer::backwarding(Tensors::Tensor derivative, int iteration) {
  Tensors::Tensor dbeta;
  Tensors::Tensor dgamma;
  Tensors::Tensor hath = hidden;
  Tensors::Tensor dy = derivative.multiply(hath.multiply(gamma).add(beta).apply(activationPrime));

  dbeta = dy.sum(0);
  dgamma = (Input.subtract(mu).divide(var.add(0.001).apply(sqrt_float)).multiply(dy).sum(0));

  Tensors::Tensor Temp =
      (dy.multiply(batch).subtract(dy.sum(0)))
          .subtract(Input.subtract(mu).divide(var.add(0.001)).multiply(dy.multiply(Input.subtract(mu)).sum(0)));
  Tensors::Tensor dh = Temp.multiply(1.0 / batch).multiply(var.add(0.001).apply(sqrt_float)).multiply(gamma);

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

void BatchNormalizationLayer::copy(Layer *l) {
  BatchNormalizationLayer *from = static_cast<BatchNormalizationLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
  this->mu = from->mu;
  this->var = from->var;
  this->gamma.copy(from->gamma);
  this->beta.copy(from->beta);
}
}
