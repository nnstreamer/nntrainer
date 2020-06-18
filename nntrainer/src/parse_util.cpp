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
 * @file	parse_util.cpp
 * @date	08 April 2020
 * @brief	This is collection of math functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <array>
#include <assert.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <regex>
#include <string.h>

namespace nntrainer {

int getKeyValue(std::string input_str, std::string &key, std::string &value) {
  int status = ML_ERROR_NONE;
  std::vector<std::string> list;
  std::regex words_regex("[^\\s=]+");
  input_str.erase(std::remove(input_str.begin(), input_str.end(), ' '),
                  input_str.end());
  auto words_begin =
    std::sregex_iterator(input_str.begin(), input_str.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int nwords = std::distance(words_begin, words_end);
  if (nwords != 2) {
    ml_loge("Error: input string must be 'key = value' format");
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    list.push_back((*i).str());
  }

  key = list[0];
  value = list[1];
  return status;
}

unsigned int parseType(std::string ll, InputType t) {
  int ret;
  unsigned int i;
  /**
   * @brief     Optimizer String from configure file
   *            "sgd"  : Stochestic Gradient Descent
   *            "adam" : Adaptive Moment Estimation
   */
  std::array<std::string, 3> optimizer_string = {"sgd", "adam", "unknown"};

  /**
   * @brief     Cost Function String from configure file
   *            "msr"  : Mean Squared Roots
   *            "cross" : Categorical Cross Entropy, alias for cross_softmax
   *            "cross_logit " : Cross Entropy with sigmoid
   *            "cross_softmax " : Cross Entropy with softmax
   */
  std::array<std::string, 5> cost_string = {"msr", "cross", "cross_logit", "cross_softmax", "unknown"};

  /**
   * @brief     Network Type String from configure file
   *            "knn"  : K Neearest Neighbor
   *            "regression" : Logistic Regression
   *            "neuralnet" : Neural Network
   */
  std::array<std::string, 4> network_type_string = {"knn", "regression",
                                                    "neuralnet", "unknown"};

  /**
   * @brief     Activation Type String from configure file
   *            "tanh"  : tanh
   *            "sigmoid" : sigmoid
   *            "relu" : relu
   *            "softmax" : softmax
   */
  std::array<std::string, 5> activation_string = {"tanh", "sigmoid", "relu",
                                                  "softmax", "unknown"};

  /**
   * @brief     Layer Type String from configure file
   *            "input"  : Input Layer Object
   *            "fully_conntected" : Fully Connected Layer Object
   *            "batch_normalization" : Batch Normalization Layer Object
   *            "conv2d" : Convolution 2D Layer Object
   *            "pooling2d" : Pooling 2D Layer Object
   *            "flatten" : Flatten Layer Object
   *            "unknown" :
   */
  std::array<std::string, 7> layer_string = {
    "input",   "fully_connected", "batch_normalization", "conv2d", "pooling2d",
    "flatten", "unknown"};

  /**
   * @brief     Weight Initialization Type String from configure file
   *            "lecun_normal"  : LeCun Normal Initialization
   *            "lecun_uniform"  : LeCun Uniform Initialization
   *            "xavier_normal"  : Xavier Normal Initialization
   *            "xavier_uniform"  : Xavier Uniform Initialization
   *            "he_normal"  : He Normal Initialization
   *            "he_uniform"  : He Uniform Initialization
   */
  std::array<std::string, 7> weight_ini_string = {
    "lecun_normal", "lecun_uniform", "xavier_normal", "xavier_uniform",
    "he_normal",    "he_uniform",    "unknown"};

  /**
   * @brief     Weight Decay String from configure file
   *            "L2Norm"  : squared norm regularization
   *            "Regression" : Regression
   */
  std::array<std::string, 3> weight_decay_string = {"l2norm", "regression",
                                                    "unknown"};

  /**
   * @brief     Weight Decay String from configure file
   *            "L2Norm"  : squared norm regularization
   *            "Regression" : Regression
   */
  std::array<std::string, 4> padding_string = {"full", "same", "valid",
                                               "unknown"};

  /**
   * @brief     Pooling String from configure file
   *            "max"  : Max Pooling
   *            "average" : Average Pooling
   *            "global_max" : Global Max Pooling
   *            "global_average" : Global Average Pooling
   */
  std::array<std::string, 5> pooling_string = {"max", "average", "global_max",
                                               "global_average", "unknown"};

  switch (t) {
  case TOKEN_OPT:
    for (i = 0; i < optimizer_string.size(); i++) {
      if (!strncasecmp(optimizer_string[i].c_str(), ll.c_str(),
                       optimizer_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_COST:
    for (i = 0; i < cost_string.size(); i++) {
      if (!strncasecmp(cost_string[i].c_str(), ll.c_str(),
                       cost_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_NET:
    for (i = 0; i < network_type_string.size(); i++) {
      if (!strncasecmp(network_type_string[i].c_str(), ll.c_str(),
                       network_type_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_ACTI:
    for (i = 0; i < activation_string.size(); i++) {
      if (!strncasecmp(activation_string[i].c_str(), ll.c_str(),
                       activation_string[i].size())) {

        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_LAYER:
    for (i = 0; i < layer_string.size(); i++) {
      if (!strncasecmp(layer_string[i].c_str(), ll.c_str(),
                       layer_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_WEIGHTINI:
    for (i = 0; i < weight_ini_string.size(); i++) {
      if (!strncasecmp(weight_ini_string[i].c_str(), ll.c_str(),
                       weight_ini_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_WEIGHT_DECAY:
    for (i = 0; i < weight_decay_string.size(); i++) {
      if (!strncasecmp(weight_decay_string[i].c_str(), ll.c_str(),
                       weight_decay_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_PADDING:
    for (i = 0; i < padding_string.size(); i++) {
      if (!strncasecmp(padding_string[i].c_str(), ll.c_str(),
                       padding_string[i].size())) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_POOLING:
    for (i = 0; i < pooling_string.size(); i++) {
      if (!strncasecmp(pooling_string[i].c_str(), ll.c_str(),
                       pooling_string[i].size())) {
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

unsigned int parseLayerProperty(std::string property) {
  int ret;
  unsigned int i;

  /**
   * @brief     Layer Properties
   * input_shape = 0,
   * bias_zero = 1,
   * normalization = 2,
   * standardization = 3,
   * activation = 4,
   * epsilon = 5
   * weight_decay = 6
   * weight_decay_lambda = 7
   * unit = 8
   * weight_ini = 9
   * filter = 10
   * kernel_size = 11
   * stride = 12
   * padding = 13
   * pooling_size = 14
   * pooling = 15
   *
   * InputLayer has 0, 1, 2, 3 properties.
   * FullyConnectedLayer has 1, 4, 6, 7, 8, 9 properties.
   * Conv2DLayer has 0, 1, 4, 6, 7, 9, 10, 11, 12, 13 properties.
   * Pooling2DLayer has 12, 13, 14, 15 properties.
   * BatchNormalizationLayer has 0, 1, 5, 6, 7 properties.
   */
  std::array<std::string, 17> property_string = {
    "input_shape", "bias_zero",  "normalization", "standardization",
    "activation",  "epsilon",    "weight_decay",  "weight_decay_lambda",
    "unit",        "weight_ini", "filter",        "kernel_size",
    "stride",      "padding",    "pooling_size",  "pooling",
    "unknown"};

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }
  ret = i - 1;

  return ret;
}

unsigned int parseOptProperty(std::string property) {
  int ret;
  unsigned int i;

  /**
   * @brief     Layer Properties
   * learning_rate = 0,
   * decay_rate = 1,
   * decay_steps = 2
   * beta1 = 3,
   * beta2 = 4,
   * epsilon = 5,
   * continue_train = 6,
   */
  std::array<std::string, 8> property_string = {
    "learning_rate", "decay_rate", "decay_steps", "beta1", "beta2", "epsilon",
    "continue_train", "unknown"};

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }
  ret = i - 1;

  return ret;
}

unsigned int parseNetProperty(std::string property) {
  int ret;
  unsigned int i;

  /**
   * @brief     Network Properties
   * loss = 0,
   * cost = 1,
   * train_data = 2,
   * val_data = 3,
   * test_data = 4,
   * label_data = 5,
   * buffer_size = 6,
   * batch_size = 7,
   * epochs = 8,
   * model_file = 9
   */
  std::array<std::string, 11> property_string = {
    "loss",      "cost",       "train_data",  "val_data",
    "test_data", "label_data", "buffer_size", "batch_size",
    "epochs",    "model_file", "unknown"};

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }
  ret = i - 1;

  return ret;
}

int setInt(int &val, std::string str) {
  int status = ML_ERROR_NONE;
  try {
    val = std::stoi(str.c_str());
  } catch (...) {
    ml_loge("Error: Wrong Type. Must be int");
    status = ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

int setFloat(float &val, std::string str) {
  int status = ML_ERROR_NONE;
  try {
    val = std::stof(str.c_str());
  } catch (...) {
    ml_loge("Error: Wrong Type. Must be float");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  return status;
}

int setDouble(double &val, std::string str) {
  int status = ML_ERROR_NONE;
  try {
    val = std::stod(str.c_str());
  } catch (...) {
    ml_loge("Error: Wrong Type. Must be double");
    status = ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

int setBoolean(bool &val, std::string str) {
  int status = ML_ERROR_NONE;
  std::string t = "true";
  std::string f = "false";

  if (!strncasecmp(str.c_str(), t.c_str(), t.size())) {
    val = true;
  } else if (!strncasecmp(str.c_str(), f.c_str(), f.size())) {
    val = false;
  } else {
    status = ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

int getValues(int n_str, std::string str, int *value) {
  int status = ML_ERROR_NONE;
  std::regex words_regex("[^\\s.,:;!?]+");
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  auto words_begin = std::sregex_iterator(str.begin(), str.end(), words_regex);
  auto words_end = std::sregex_iterator();

  int num = std::distance(words_begin, words_end);
  if (num != n_str) {
    ml_loge("Number of Data is not match");
    return ML_ERROR_INVALID_PARAMETER;
  }
  int cn = 0;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    value[cn] = std::stoi((*i).str());
    cn++;
  }
  return status;
}

} /* namespace nntrainer */
