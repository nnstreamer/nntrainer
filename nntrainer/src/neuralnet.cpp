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
 * @file	neuralnet.cpp
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "include/neuralnet.h"
#include <assert.h>
#include <stdio.h>
#include <array>
#include <cmath>
#include <sstream>
#include "include/nntrainer_log.h"
#include "iniparser.h"

/**
 * @brief     compare character to remove case sensitivity
 * @param[in] c1 characer #1
 * @param[in] c2 characer #2
 * @retval    boolean true if they are same
 */
bool compareChar(char &c1, char &c2) {
  if (c1 == c2)
    return true;
  else if (std::toupper(c1) == std::toupper(c2))
    return true;
  return false;
}

/**
 * @brief     compare string with case insensitive
 * @param[in] str1 string #1
 * @param[in] str2 string #2
 * @retval    boolean true if they are same
 */
bool caseInSensitiveCompare(std::string &str1, std::string &str2) {
  return ((str1.size() == str2.size()) && std::equal(str1.begin(), str1.end(), str2.begin(), &compareChar));
}

namespace Network {

/**
 * @brief     Check Existance of File
 * @param[in] filename file path to check
 * @retval    boolean true if exists
 */
static bool is_file_exist(std::string filename) {
  std::ifstream infile(filename);
  return infile.good();
}

/**
 * @brief     Parsing Layer Name
 * @param[in] string layer name
 * @retval    vector stored layer name
 */
std::vector<std::string> parseLayerName(std::string ll) {
  std::vector<std::string> ret;
  std::istringstream ss(ll);
  do {
    std::string word;
    ss >> word;
    if (word.compare("") != 0)
      ret.push_back(word);
  } while (ss);

  return ret;
}

/**
 * @brief     Parsing Configuration Token
 * @param[in] ll string to be parsed
 * @param[in] t  Token type
 * @retval    int enumerated type
 */
unsigned int parseType(std::string ll, input_type t) {
  int ret;
  unsigned int i;

  /**
   * @brief     Optimizer String from configure file
   *            "sgd"  : Stochestic Gradient Descent
   *            "adam" : Adaptive Moment Estimation
   */
  std::array<std::string, 150> Optimizer_string = {"sgd", "adam", "unknown"};

  /**
   * @brief     Cost Function String from configure file
   *            "msr"  : Mean Squared Roots
   *            "caterogical" : Categorical Cross Entropy
   */
  std::array<std::string, 150> Cost_string = {"msr", "cross", "unknown"};

  /**
   * @brief     Network Type String from configure file
   *            "knn"  : K Neearest Neighbor
   *            "regression" : Logistic Regression
   *            "neuralnet" : Neural Network
   */
  std::array<std::string, 150> NetworkType_string = {"knn", "regression", "neuralnet", "unknown"};

  /**
   * @brief     Activation Type String from configure file
   *            "tanh"  : tanh
   *            "sigmoid" : sigmoid
   *            "relu" : relu
   *            "softmax" : softmax
   */
  std::array<std::string, 150> activation_string = {"tanh", "sigmoid", "relu", "softmax", "unknown"};

  /**
   * @brief     Layer Type String from configure file
   *            "InputLayer"  : InputLayer Object
   *            "FullyConnectedLayer" : Fully Connected Layer Object
   *            "OutputLayer" : Output Layer Object
   *            "BatchNormalizationLayer" : Batch Normalization Layer
   */
  std::array<std::string, 150> layer_string = {"InputLayer", "FullyConnectedLayer", "OutputLayer",
                                               "BatchNormalizationLayer", "Unknown"};

  /**
   * @brief     Weight Initialization Type String from configure file
   *            "lecun_normal"  : LeCun Normal Initialization
   *            "lecun_uniform"  : LeCun Uniform Initialization
   *            "xavier_normal"  : Xavier Normal Initialization
   *            "xavier_uniform"  : Xavier Uniform Initialization
   *            "he_normal"  : He Normal Initialization
   *            "he_uniform"  : He Uniform Initialization
   */
  std::array<std::string, 150> weightini_string = {"lecun_normal", "lecun_uniform", "xavier_normal", "xavier_uniform",
                                                   "he_normal",    "he_uniform",    "unknown"};

  /**
   * @brief     Weight Decay String from configure file
   *            "L2Norm"  : squared norm regularization
   *            "Regression" : Regression
   */
  std::array<std::string, 150> weight_decay_string = {"l2norm", "regression", "unknown"};

  switch (t) {
    case TOKEN_OPT:
      for (i = 0; i < Optimizer_string.size(); i++) {
        if (caseInSensitiveCompare(Optimizer_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_COST:
      for (i = 0; i < Cost_string.size(); i++) {
        if (caseInSensitiveCompare(Cost_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_NET:
      for (i = 0; i < NetworkType_string.size(); i++) {
        if (caseInSensitiveCompare(NetworkType_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_ACTI:
      for (i = 0; i < activation_string.size(); i++) {
        if (caseInSensitiveCompare(activation_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_LAYER:
      for (i = 0; i < layer_string.size(); i++) {
        if (caseInSensitiveCompare(layer_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_WEIGHTINI:
      for (i = 0; i < weightini_string.size(); i++) {
        if (caseInSensitiveCompare(weightini_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_WEIGHT_DECAY:
      for (i = 0; i < weight_decay_string.size(); i++) {
        if (caseInSensitiveCompare(weight_decay_string[i], ll)) {
          return (i);
        }
      }
      ret = i - 1;
      break;
    case TOKEN_UNKNOWN:
    default:
      ret = 3;
      break;
  }
  return ret;
}

NeuralNetwork::NeuralNetwork(std::string config) { this->config = config; }

void NeuralNetwork::setConfig(std::string config) { this->config = config; }

void NeuralNetwork::init() {
  int id;
  bool b_zero;
  std::string l_type;
  Layers::layer_type t;
  std::string inifile = config;
  dictionary *ini = iniparser_load(inifile.c_str());
  std::vector<int> HiddenSize;

  char unknown[] = "Unknown";
  char model_name[] = "model.bin";

  if (ini == NULL) {
    fprintf(stderr, "cannot parse file: %s\n", inifile.c_str());
  }

  nettype = (Network::net_type)parseType(iniparser_getstring(ini, "Network:Type", unknown), TOKEN_NET);
  std::vector<std::string> layers_name = parseLayerName(iniparser_getstring(ini, "Network:Layers", unknown));
  learning_rate = iniparser_getdouble(ini, "Network:Learning_rate", 0.0);
  decay_rate = iniparser_getdouble(ini, "Network:Decay_rate", 0.0);
  decay_steps = iniparser_getint(ini, "Network:Decay_steps", -1);

  opt.learning_rate = learning_rate;
  opt.decay_steps = decay_steps;
  opt.decay_rate = decay_rate;
  epoch = iniparser_getint(ini, "Network:Epoch", 100);
  opt.type = (Layers::opt_type)parseType(iniparser_getstring(ini, "Network:Optimizer", unknown), TOKEN_OPT);
  cost = (Layers::cost_type)parseType(iniparser_getstring(ini, "Network:Cost", unknown), TOKEN_COST);
  weightini =
      (Layers::weightIni_type)parseType(iniparser_getstring(ini, "Network:WeightIni", unknown), TOKEN_WEIGHTINI);

  opt.weight_decay.type = (Layers::weight_decay_type)parseType(
      iniparser_getstring(ini, "Network:Weight_Decay", unknown), TOKEN_WEIGHT_DECAY);

  opt.weight_decay.lambda = 0.0;
  if (opt.weight_decay.type == Layers::WEIGHT_DECAY_L2NORM) {
    opt.weight_decay.lambda = iniparser_getdouble(ini, "Network:Weight_Decay_Lambda", 0.0);
  }

  model = iniparser_getstring(ini, "Network:Model", model_name);
  batchsize = iniparser_getint(ini, "Network:minibatch", 1);

  opt.beta1 = iniparser_getdouble(ini, "Network:beta1", 0.0);
  opt.beta2 = iniparser_getdouble(ini, "Network:beta2", 0.0);
  opt.epsilon = iniparser_getdouble(ini, "Network:epsilon", 0.0);

  for (unsigned int i = 0; i < layers_name.size(); i++)
    ml_logi("%s", layers_name[i].c_str());

  loss = 100000.0;

  for (unsigned int i = 0; i < layers_name.size(); ++i) {
    unsigned int hidden_size;
    l_type = iniparser_getstring(ini, (layers_name[i] + ":Type").c_str(), unknown);
    t = (Layers::layer_type)parseType(l_type, TOKEN_LAYER);

    switch (t) {
      case Layers::LAYER_BN: {
        if (i == 0)
          std::runtime_error("Error: Batch NormalizationLayer should be after InputLayer.");
        hidden_size = HiddenSize[i - 1];
      }

      break;
      case Layers::LAYER_IN:
      case Layers::LAYER_FC:
      case Layers::LAYER_OUT:
      default:
        hidden_size = iniparser_getint(ini, (layers_name[i] + ":HiddenSize").c_str(), 1);
        break;
    }
    HiddenSize.push_back(hidden_size);
  }

  for (unsigned int i = 0; i < layers_name.size(); i++) {
    l_type = iniparser_getstring(ini, (layers_name[i] + ":Type").c_str(), unknown);
    t = (Layers::layer_type)parseType(l_type, TOKEN_LAYER);
    id = iniparser_getint(ini, (layers_name[i] + ":Id").c_str(), 0);
    b_zero = iniparser_getboolean(ini, (layers_name[i] + ":Bias_zero").c_str(), true);
    std::stringstream ss;
    ml_logi("%d : %s %d %d", id, l_type.c_str(), HiddenSize[i], b_zero);

    switch (t) {
      case Layers::LAYER_IN: {
        Layers::InputLayer *inputlayer = new (Layers::InputLayer);
        inputlayer->setType(t);
        inputlayer->initialize(batchsize, 1, HiddenSize[i], id, b_zero, weightini);
        inputlayer->setOptimizer(opt);
        inputlayer->setNormalization(iniparser_getboolean(ini, (layers_name[i] + ":Normalization").c_str(), false));
        inputlayer->setStandardization(iniparser_getboolean(ini, (layers_name[i] + ":Standardization").c_str(), false));
        inputlayer->setActivation((Layers::acti_type)parseType(
            iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(), unknown), TOKEN_ACTI));
        layers.push_back(inputlayer);
      } break;
      case Layers::LAYER_FC: {
        Layers::FullyConnectedLayer *fclayer = new (Layers::FullyConnectedLayer);
        fclayer->setType(t);
        fclayer->initialize(batchsize, HiddenSize[i - 1], HiddenSize[i], id, b_zero, weightini);
        fclayer->setOptimizer(opt);
        fclayer->setActivation((Layers::acti_type)parseType(
            iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(), unknown), TOKEN_ACTI));
        layers.push_back(fclayer);
      } break;
      case Layers::LAYER_OUT: {
        Layers::OutputLayer *outputlayer = new (Layers::OutputLayer);
        outputlayer->setType(t);
        outputlayer->setCost(cost);
        outputlayer->initialize(batchsize, HiddenSize[i - 1], HiddenSize[i], id, b_zero, weightini);
        outputlayer->setOptimizer(opt);
        outputlayer->setActivation((Layers::acti_type)parseType(
            iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(), unknown), TOKEN_ACTI));
        layers.push_back(outputlayer);
      } break;
      case Layers::LAYER_BN: {
        Layers::BatchNormalizationLayer *bnlayer = new (Layers::BatchNormalizationLayer);
        bnlayer->setType(t);
        bnlayer->setOptimizer(opt);
        bnlayer->initialize(batchsize, 1, HiddenSize[i], id, b_zero, weightini);
        layers.push_back(bnlayer);
        layers[i - 1]->setBNfallow(true);
        bnlayer->setActivation((Layers::acti_type)parseType(
            iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(), unknown), TOKEN_ACTI));
      } break;
      case Layers::LAYER_UNKNOWN:
        break;
      default:
        break;
    }
  }

  iniparser_freedict(ini);
}

/**
 * @brief     free layers
 */
void NeuralNetwork::finalize() {
  for (unsigned int i = 0; i < layers.size(); i++) {
    delete layers[i];
  }
}

/**
 * @brief     forward propagation using layers object which has layer
 */
Tensors::Tensor NeuralNetwork::forwarding(Tensors::Tensor input) {
  Tensors::Tensor X = input;
  for (unsigned int i = 0; i < layers.size(); i++) {
    X = layers[i]->forwarding(X);
  }
  return X;
}

/**
 * @brief     forward propagation using layers object which has layer
 */
Tensors::Tensor NeuralNetwork::forwarding(Tensors::Tensor input, Tensors::Tensor output) {
  Tensors::Tensor X = input;
  Tensors::Tensor Y2 = output;
  for (unsigned int i = 0; i < layers.size(); i++) {
    X = layers[i]->forwarding(X, Y2);
  }
  return X;
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(Tensors::Tensor input, Tensors::Tensor expected_output, int iteration) {
  Tensors::Tensor Y2 = expected_output;
  Tensors::Tensor X = input;
  Tensors::Tensor Y = forwarding(X);

  for (unsigned int i = layers.size() - 1; i > 0; i--) {
    Y2 = layers[i]->backwarding(Y2, iteration);
  }
}

float NeuralNetwork::getLoss() {
  Layers::OutputLayer *out = static_cast<Layers::OutputLayer *>((layers[layers.size() - 1]));
  return out->getLoss();
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    batchsize = from.batchsize;
    learning_rate = from.learning_rate;
    loss = from.loss;
    opt = from.opt;

    for (unsigned int i = 0; i < layers.size(); i++)
      layers[i]->copy(from.layers[i]);
  }
  return *this;
}

/**
 * @brief     save model
 *            save Weight & Bias Data into file by calling save from layer
 */
void NeuralNetwork::saveModel() {
  std::ofstream modelFile(model, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(modelFile);
  modelFile.close();
}

/**
 * @brief     read model
 *            read Weight & Bias Data into file by calling save from layer
 */
void NeuralNetwork::readModel() {
  if (!is_file_exist(model))
    return;
  std::ifstream modelFile(model, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->read(modelFile);
  modelFile.close();
  ml_logi("read modelfile: %s", model.c_str());
}
}
