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
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "include/layers.h"
#include <assert.h>
#include <random>

/**
 * @brief     random function
 * @param[in] x float
 */
float random(float x) { return (float)(rand() % 10000 + 1) / 10000 - 0.5; }

/**
 * @brief     sqrt function for float type
 * @param[in] x float
 */
float sqrt_float(float x) { return (float)(sqrt(x)); }

/**
 * @brief     log function for float type
 * @param[in] x float
 */
float log_float(float x) { return (float)(log(x)); }

/**
 * @brief     sigmoid activation function
 * @param[in] x input
 */
float sigmoid(float x) { return 1 / (1 + exp(-x)); }

/**
 * @brief     derivative sigmoid function
 * @param[in] x input
 */
float sigmoidePrime(float x) { return (float)(1.0 / ((1 + exp(-x)) * (1.0 + 1.0 / (exp(-x) + 0.0000001)))); }

/**
 * @brief     tanh function for float type
 * @param[in] x input
 */
float tanh_float(float x) { return (float)tanh(x); }

/**
 * @brief     derivative tanh function
 * @param[in] x input
 */
float tanhPrime(float x) {
  float th = (float)tanh(x);
  return 1.0 - th * th;
}

/**
 * @brief     relu activation function
 * @param[in] x input
 */
float Relu(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return x;
  }
}

/**
 * @brief     derivative relu function
 * @param[in] x input
 */
float ReluPrime(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return 1.0;
  }
}

Tensor softmaxPrime(Tensor x) {
  int batch = x.getBatch();
  int width = x.getWidth();
  int height = x.getHeight();

  assert(height == 1);

  Tensor PI = Tensor(batch, height, width);

  for (int k = 0; k < batch; ++k) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        float sum = 0.0;
        for (int l = 0; l < width; ++l) {
          if (j == l) {
            sum += x.getValue(k, i, l) * (1.0 - x.getValue(k, i, j));
          } else {
            sum += x.getValue(k, i, l) * x.getValue(k, i, j) * -1.0;
          }
        }
        PI.setValue(k, i, j, sum);
      }
    }
  }
  return PI;
}

static Tensor WeightInitialization(unsigned int width, unsigned int height, Layers::weightIni_type init_type) {
  std::random_device rd;
  std::mt19937 gen(rd());
  Tensor W = Tensor(height, width);

  if(init_type == Layers::WEIGHT_UNKNOWN)
    init_type = Layers::WEIGHT_XAVIER_NORMAL;
  switch (init_type) {
    case Layers::WEIGHT_LECUN_NORMAL: {
      std::normal_distribution<float> dist(0, sqrt(1 / height));
      for (unsigned int i = 0; i < width; ++i)
        for (unsigned int j = 0; j < height; ++j) {
          float f = dist(gen);
          W.setValue(0, j, i, f);
        }
    } break;
    case Layers::WEIGHT_LECUN_UNIFORM: {
      std::uniform_real_distribution<float> dist(-1.0 * sqrt(1.0 / height), sqrt(1.0 / height));
      for (unsigned int i = 0; i < width; ++i)
        for (unsigned int j = 0; j < height; ++j) {
          float f = dist(gen);
          W.setValue(0, j, i, f);
        }
    } break;
    case Layers::WEIGHT_XAVIER_NORMAL: {
      std::normal_distribution<float> dist(0, sqrt(2.0 / (width + height)));
      for (unsigned int i = 0; i < width; ++i)
        for (unsigned int j = 0; j < height; ++j) {
          float f = dist(gen);
          W.setValue(0, j, i, f);
        }
    } break;
    case Layers::WEIGHT_XAVIER_UNIFORM: {
      std::uniform_real_distribution<float> dist(-1.0 * sqrt(6.0 / (height + width)), sqrt(6.0 / (height + width)));
      for (unsigned int i = 0; i < width; ++i)
        for (unsigned int j = 0; j < height; ++j) {
          float f = dist(gen);
          W.setValue(0, j, i, f);
        }
    } break;
    case Layers::WEIGHT_HE_NORMAL: {
      std::normal_distribution<float> dist(0, sqrt(2.0 / (height)));
      for (unsigned int i = 0; i < width; ++i)
        for (unsigned int j = 0; j < height; ++j) {
          float f = dist(gen);
          W.setValue(0, j, i, f);
        }
    } break;
    case Layers::WEIGHT_HE_UNIFORM: {
      std::uniform_real_distribution<float> dist(-1.0 * sqrt(6.0 / (height)), sqrt(6.0 / (height)));
      for (unsigned int i = 0; i < width; ++i)
        for (unsigned int j = 0; j < height; ++j) {
          float f = dist(gen);
          W.setValue(0, j, i, f);
        }
    } break;
    default:
      W.setZero();
      break;
  }
  return W;
}

namespace Layers {

void Layer::setActivation(acti_type acti) {
  if(acti == ACT_UNKNOWN){
    std::cout << "have to specify activation function" << std::endl;
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

void InputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
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

Tensor InputLayer::forwarding(Tensor input) {
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

  Bias = Tensor(1, w);
  Weight = WeightInitialization(w, h, wini);

  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.applyFunction(random);
  }
}

void FullyConnectedLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  if (opt.type == OPT_ADAM) {
    WM = Tensor(height, width);
    WV = Tensor(height, width);
    WM.setZero();
    WV.setZero();
    BM = Tensor(1, width);
    BV = Tensor(1, width);
    BM.setZero();
    BV.setZero();
  }
}

Tensor FullyConnectedLayer::forwarding(Tensor input) {
  Input = input;
  hidden = Input.dot(Weight).add(Bias);

  if (this->bnfallow)
    return hidden;

  return hidden.applyFunction(activation);
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

Tensor FullyConnectedLayer::backwarding(Tensor derivative, int iteration) {
  Tensor dJdB;

  dJdB = derivative.multiply(hidden.applyFunction(activationPrime));

  Tensor dJdW = Input.transpose().dot(dJdB);

  if (opt.weight_decay.type == WEIGHT_DECAY_L2NORM) {
    dJdW = dJdW.add(Weight.multiply(opt.weight_decay.lambda));
  }

  Tensor ret = dJdB.dot(Weight.transpose());

  float ll = opt.learning_rate;
  if (opt.decay_steps != -1) {
    ll = opt.learning_rate * pow(opt.decay_rate, (iteration / opt.decay_steps));
  }

  switch (opt.type) {
    case OPT_SGD:
      Weight = Weight.subtract(dJdW.average().multiply(ll));
      break;
    case OPT_ADAM:
      WM = WM.multiply(opt.beta1).add(dJdW.average().multiply(1 - opt.beta1));
      WV = WV.multiply(opt.beta2).add((dJdW.average().multiply(dJdW.average())).multiply(1 - opt.beta2));
      WM.divide(1 - pow(opt.beta1, iteration + 1));
      WV.divide(1 - pow(opt.beta2, iteration + 1));
      Weight = Weight.subtract((WM.divide(WV.applyFunction(sqrt_float).add(opt.epsilon))).multiply(ll));
      BM = BM.multiply(opt.beta1).add(dJdB.average().multiply(1 - opt.beta1));
      BV = BV.multiply(opt.beta2).add((dJdB.average().multiply(dJdB.average())).multiply(1 - opt.beta2));
      BM.divide(1 - pow(opt.beta1, iteration + 1));
      BV.divide(1 - pow(opt.beta2, iteration + 1));
      Bias = Bias.subtract((BM.divide(BV.applyFunction(sqrt_float).add(opt.epsilon))).multiply(ll));
      break;
    default:
      break;
  }

  if (!this->init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(ll));
  }

  return ret;
}

void OutputLayer::initialize(int b, int h, int w, int id, bool init_zero, weightIni_type wini) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;

  Bias = Tensor(1, w);
  this->cost = cost;
  this->bnfallow = false;

  Weight = WeightInitialization(w, h, wini);

  if (cost == COST_CATEGORICAL)
    init_zero = true;

  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.applyFunction(random);
  }
}

Tensor OutputLayer::forwarding(Tensor input) {
  Input = input;
  hidden = input.dot(Weight).add(Bias);
  if (activation_type == ACT_SOFTMAX) {
    return hidden.softmax();
  } else {
    return hidden.applyFunction(activation);
  }
}

Tensor OutputLayer::forwarding(Tensor input, Tensor output) {
  Input = input;
  hidden = input.dot(Weight).add(Bias);
  Tensor Y2 = output;
  Tensor Y = hidden;

  if (activation_type == ACT_SOFTMAX) {
    Y = Y.softmax();
  } else {
    Y = Y.applyFunction(activation);
  }

  float lossSum = 0.0;

  switch (cost) {
    case COST_CATEGORICAL: {
      Tensor temp = ((Y2.multiply(-1.0).transpose().dot(Y.add(opt.epsilon).applyFunction(log_float)))
                         .subtract(Y2.multiply(-1.0).add(1.0).transpose().dot(
                             Y.multiply(-1.0).add(1.0).add(opt.epsilon).applyFunction(log_float))));
      loss = (1.0 / Y.Mat2Vec().size()) * temp.Mat2Vec()[0];
    } break;
    case COST_MSR: {
      Tensor sub = Y2.subtract(Y);
      Tensor l = (sub.multiply(sub)).sum().multiply(0.5);
      std::vector<float> t = l.Mat2Vec();
      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }

      loss = lossSum / (float)l.getBatch();
    } break;
    case COST_ENTROPY: {
      Tensor l;
      if (activation_type == ACT_SIGMOID) {
        l = (Y2.multiply(Y.applyFunction(log_float))
                 .add((Y2.multiply(-1.0).add(1.0)).multiply((Y.multiply(-1.0).add(1.0)).applyFunction(log_float))))
                .multiply(-1.0 / (Y2.getWidth()))
                .sum();
      } else if (activation_type == ACT_SOFTMAX) {
        l = (Y2.multiply(Y.applyFunction(log_float))).multiply(-1.0 / (Y2.getWidth())).sum();
      } else {
        std::cout << "Only support sigmoid & softmax for cross entropy loss" << std::endl;
        exit(0);
      }

      std::vector<float> t = l.Mat2Vec();

      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }
      loss = lossSum / (float)l.getBatch();

      if (opt.weight_decay.type == WEIGHT_DECAY_L2NORM) {
        loss += opt.weight_decay.lambda * 0.5 * (Weight.l2norm());
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

void OutputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  if (opt.type == OPT_ADAM) {
    WM = Tensor(height, width);
    WV = Tensor(height, width);
    WM.setZero();
    WV.setZero();
    BM = Tensor(1, width);
    BV = Tensor(1, width);
    BM.setZero();
    BV.setZero();
  }
}

Tensor OutputLayer::backwarding(Tensor label, int iteration) {
  float lossSum = 0.0;
  Tensor Y2 = label;
  Tensor Y;
  if (activation_type == ACT_SOFTMAX)
    Y = hidden.softmax();
  else
    Y = hidden.applyFunction(activation);

  Tensor ret;
  Tensor dJdB;

  float ll = opt.learning_rate;
  if (opt.decay_steps != -1) {
    ll = opt.learning_rate * pow(opt.decay_rate, (iteration / opt.decay_steps));
  }

  switch (cost) {
    case COST_CATEGORICAL: {
      dJdB = Y.subtract(Y2);
      Tensor temp = ((Y2.multiply(-1.0).transpose().dot(Y.add(opt.epsilon).applyFunction(log_float)))
                         .subtract(Y2.multiply(-1.0).add(1.0).transpose().dot(
                             Y.multiply(-1.0).add(1.0).add(opt.epsilon).applyFunction(log_float))));
      loss = (1.0 / Y.Mat2Vec().size()) * temp.Mat2Vec()[0];
      if (opt.weight_decay.type == WEIGHT_DECAY_L2NORM) {
        loss += opt.weight_decay.lambda * 0.5 * (Weight.l2norm());
      }
    } break;
    case COST_MSR: {
      Tensor sub = Y2.subtract(Y);
      Tensor l = (sub.multiply(sub)).sum().multiply(0.5);
      std::vector<float> t = l.Mat2Vec();
      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }

      loss = lossSum / (float)l.getBatch();
      if (opt.weight_decay.type == WEIGHT_DECAY_L2NORM) {
        loss += opt.weight_decay.lambda * 0.5 * (Weight.l2norm());
      }
      if (activation_type == ACT_SOFTMAX) {
        dJdB = Y.subtract(Y2).multiply(softmaxPrime(Y));
      } else {
        dJdB = Y.subtract(Y2).multiply(hidden.applyFunction(activationPrime));
      }
    } break;
    case COST_ENTROPY: {
      Tensor l;
      if (activation_type == ACT_SIGMOID) {
        dJdB = Y.subtract(Y2).multiply(1.0 / Y.getWidth());
        l = (Y2.multiply(Y.applyFunction(log_float))
                 .add((Y2.multiply(-1.0).add(1.0)).multiply((Y.multiply(-1.0).add(1.0)).applyFunction(log_float))))
                .multiply(-1.0 / (Y2.getWidth()))
                .sum();
      } else if (activation_type == ACT_SOFTMAX) {
        dJdB = Y.subtract(Y2).multiply(1.0 / Y.getWidth());
        l = (Y2.multiply(Y.applyFunction(log_float))).multiply(-1.0 / (Y2.getWidth())).sum();
      } else {
        std::cout << "Only support sigmoid & softmax for cross entropy loss" << std::endl;
        exit(0);
      }

      std::vector<float> t = l.Mat2Vec();

      for (int i = 0; i < l.getBatch(); i++) {
        lossSum += t[i];
      }
      loss = lossSum / (float)l.getBatch();

      if (opt.weight_decay.type == WEIGHT_DECAY_L2NORM) {
        loss += opt.weight_decay.lambda * 0.5 * (Weight.l2norm());
      }

    } break;
    case COST_UNKNOWN:
    default:
      break;
  }

  Tensor dJdW = Input.transpose().dot(dJdB);

  if (opt.weight_decay.type == WEIGHT_DECAY_L2NORM) {
    dJdW = dJdW.add(Weight.multiply(opt.weight_decay.lambda));
  }

  ret = dJdB.dot(Weight.transpose());

  switch (opt.type) {
    case Layers::OPT_SGD:
      Weight = Weight.subtract(dJdW.average().multiply(ll));
      break;
    case Layers::OPT_ADAM:
      WM = WM.multiply(opt.beta1).add(dJdW.average().multiply(1 - opt.beta1));
      WV = WV.multiply(opt.beta2).add((dJdW.average().multiply(dJdW.average())).multiply(1 - opt.beta2));
      WM.divide(1 - pow(opt.beta1, iteration + 1));
      WV.divide(1 - pow(opt.beta2, iteration + 1));
      Weight = Weight.subtract((WM.divide(WV.applyFunction(sqrt_float).add(opt.epsilon))).multiply(ll));
      BM = BM.multiply(opt.beta1).add(dJdB.average().multiply(1 - opt.beta1));
      BV = BV.multiply(opt.beta2).add((dJdB.average().multiply(dJdB.average())).multiply(1 - opt.beta2));
      BM.divide(1 - pow(opt.beta1, iteration + 1));
      BV.divide(1 - pow(opt.beta2, iteration + 1));
      Bias = Bias.subtract((BM.divide(BV.applyFunction(sqrt_float).add(opt.epsilon))).multiply(ll));
      break;
    default:
      break;
  }

  if (!this->init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(ll));
  }

  return ret;
}

void BatchNormalizationLayer::initialize(int b, int h, int w, int id, bool init_zero, weightIni_type wini) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;

  this->gamma = Tensor(batch, w);
  this->beta = Tensor(batch, w);
  beta.setZero();
  gamma.setZero();
}

void BatchNormalizationLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
}

Tensor BatchNormalizationLayer::forwarding(Tensor input) {
  Tensor temp;

  hidden = input;

  mu = input.sum(0).multiply(1.0 / batch);

  temp = input.subtract(mu);

  var = temp.multiply(temp).sum(0).multiply(1.0 / batch);

  Tensor hath = temp.divide(var.add(0.001).applyFunction(sqrt_float));

  hidden = hath;

  Tensor ret = hath.multiply(gamma).add(beta).applyFunction(activation);

  return ret;
}

Tensor BatchNormalizationLayer::backwarding(Tensor derivative, int iteration) {
  Tensor dbeta;
  Tensor dgamma;
  Tensor hath = hidden;
  Tensor dy = derivative.multiply(hath.multiply(gamma).add(beta).applyFunction(activationPrime));

  dbeta = dy.sum(0);
  dgamma = (Input.subtract(mu).divide(var.add(0.001).applyFunction(sqrt_float)).multiply(dy).sum(0));

  Tensor Temp =
      (dy.multiply(batch).subtract(dy.sum(0)))
          .subtract(Input.subtract(mu).divide(var.add(0.001)).multiply(dy.multiply(Input.subtract(mu)).sum(0)));
  Tensor dh = Temp.multiply(1.0 / batch).multiply(var.add(0.001).applyFunction(sqrt_float)).multiply(gamma);

  float ll = opt.learning_rate;
  if (opt.decay_steps != -1) {
    ll = opt.learning_rate * pow(opt.decay_rate, (iteration / opt.decay_steps));
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
