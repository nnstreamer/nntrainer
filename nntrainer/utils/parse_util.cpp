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
#include <cstring>
#include <databuffer.h>
#include <iostream>
#include <sstream>
#include <string>

#include <acti_func.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_devel.h>
#include <parse_util.h>
#include <pooling2d_layer.h>

#define NUM_SKIP_CHAR 3

namespace nntrainer {

int getKeyValue(const std::string &input_str, std::string &key,
                std::string &value) {
  int status = ML_ERROR_NONE;
  auto input_trimmed = input_str;

  std::vector<std::string> list;
  static const std::regex words_regex("[^\\s=]+");
  input_trimmed.erase(
    std::remove(input_trimmed.begin(), input_trimmed.end(), ' '),
    input_trimmed.end());
  auto words_begin = std::sregex_iterator(input_trimmed.begin(),
                                          input_trimmed.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int nwords = std::distance(words_begin, words_end);
  if (nwords != 2) {
    ml_loge("Error: input string must be 'key = value' format, %s given",
            input_trimmed.c_str());
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
  unsigned int ret;
  unsigned int i;

  /**
   * @brief     Model Type String from configure file
   *            "knn"  : K Neearest Neighbor
   *            "regression" : Logistic Regression
   *            "neuralnet" : Neural Network
   */
  std::array<std::string, 3> model_type_string = {"knn", "regression",
                                                  "neuralnet"};

  /**
   * @brief     Weight Initialization Type String from configure file
   *            "zeros" : Zero Initialization
   *            "ones" : One Initialization
   *            "lecun_normal"  : LeCun Normal Initialization
   *            "lecun_uniform"  : LeCun Uniform Initialization
   *            "xavier_normal"  : Xavier Normal Initialization
   *            "xavier_uniform"  : Xavier Uniform Initialization
   *            "he_normal"  : He Normal Initialization
   *            "he_uniform"  : He Uniform Initialization
   */
  std::array<std::string, 8> weight_ini_string = {
    "zeros",         "ones",           "lecun_normal", "lecun_uniform",
    "xavier_normal", "xavier_uniform", "he_normal",    "he_uniform"};

  /**
   * @brief     Weight Regularization String from configure file
   *            "L2Norm"  : squared norm regularization
   *            "None" : none
   */
  std::array<std::string, 2> weight_regularizer_string = {"l2norm", "none"};

  /**
   * @brief     Weight Decay String from configure file
   *            "L2Norm"  : squared norm regularization
   *            "Regression" : Regression
   */
  std::array<std::string, 3> padding_string = {"full", "same", "valid"};

  /**
   * @brief     Pooling String from configure file
   *            "max"  : Max Pooling
   *            "average" : Average Pooling
   *            "global_max" : Global Max Pooling
   *            "global_average" : Global Average Pooling
   */
  std::array<std::string, 4> pooling_string = {"max", "average", "global_max",
                                               "global_average"};

  switch (t) {
  case TOKEN_MODEL:
    for (i = 0; i < model_type_string.size(); i++) {
      if (!strncasecmp(model_type_string[i].c_str(), ll.c_str(),
                       model_type_string[i].size())) {
        return (i);
      }
    }
    ret = (unsigned int)NetType::UNKNOWN;
    break;
  case TOKEN_ACTI:
    for (i = 0; i < ActivationTypeStr.size(); i++) {
      if (!strncasecmp(ActivationTypeStr[i].c_str(), ll.c_str(),
                       ActivationTypeStr[i].size())) {

        return (i);
      }
    }
    ml_logw("Input activation %s cannot be identified. "
            "Moved to NO activation layer by default.",
            ll.c_str());
    ret = (unsigned int)ActivationType::ACT_UNKNOWN;
    break;
  case TOKEN_WEIGHT_INIT:
    for (i = 0; i < weight_ini_string.size(); i++) {
      if (!strncasecmp(weight_ini_string[i].c_str(), ll.c_str(),
                       weight_ini_string[i].size())) {
        return (i);
      }
    }
    ret = (unsigned int)TensorInitializer::NONE;
    break;
  case TOKEN_WEIGHT_REGULARIZER:
    for (i = 0; i < weight_regularizer_string.size(); i++) {
      if (!strncasecmp(weight_regularizer_string[i].c_str(), ll.c_str(),
                       weight_regularizer_string[i].size())) {
        return (i);
      }
    }
    ret = (unsigned int)WeightRegularizer::UNKNOWN;
    break;
  case TOKEN_POOLING:
    for (i = 0; i < pooling_string.size(); i++) {
      if (!strncasecmp(pooling_string[i].c_str(), ll.c_str(),
                       pooling_string[i].size())) {
        return (i);
      }
    }
    ret = (unsigned int)Pooling2DLayer::PoolingType::unknown;
    break;
  case TOKEN_UNKNOWN:
  default:
    ml_loge("Error: unknown token cannot be parsed.");
    ret = 0;
    break;
  }
  return ret;
}

/**
 * @brief     Layer Properties
 * input_shape = 0
 * normalization = 1
 * standardization = 2
 * activation = 3
 * epsilon = 4
 * weight_regularizer = 5
 * weight_regularizer_constant = 6
 * unit = 7
 * weight_initializer = 8
 * bias_initializer = 9
 * filters = 10
 * kernel_size = 11
 * stride = 12
 * padding = 13
 * pool_size = 14
 * pooling = 15
 * flatten = 16
 * name = 17
 * momentum = 18
 * moving_mean_initializer = 19
 * moving_variance_initializer = 20
 * gamma_initializer = 21
 * beta_initializer = 22
 * modelfile = 23
 * input_layers = 24
 * output_layers = 25
 * trainable = 26
 * flip_direction = 27
 * random_translate = 28
 * in_dim = 29
 * out_dim = 30
 * recurrent_activation = 31
 * distribute = 32
 * split_dimension = 33
 * return_sequences = 34
 * hidden_state_activation = 35
 * dropout = 36
 *
 * InputLayer has 0, 1, 2, 3 properties.
 * FullyConnectedLayer has 1, 4, 6, 7, 8, 9 properties.
 * Conv2DLayer has 0, 1, 4, 6, 7, 9, 10, 11, 12, 13 properties.
 * Pooling2DLayer has 12, 13, 14, 15 properties.
 * BatchNormalizationLayer has 0, 1, 5, 6, 7 properties.
 */
static std::array<std::string, 38> property_string = {
  "input_shape",
  "normalization",
  "standardization",
  "activation",
  "epsilon",
  "weight_regularizer",
  "weight_regularizer_constant",
  "unit",
  "weight_initializer",
  "bias_initializer",
  "filters",
  "kernel_size",
  "stride",
  "padding",
  "pool_size",
  "pooling",
  "flatten",
  "name",
  "momentum",
  "moving_mean_initializer",
  "moving_variance_initializer",
  "gamma_initializer",
  "beta_initializer",
  "modelfile",
  "input_layers",
  "output_layers",
  "trainable",
  "flip_direction",
  "random_translate",
  "in_dim",
  "out_dim",
  "recurrent_activation",
  "distribute",
  "split_dimension",
  "return_sequences",
  "hidden_state_activation",
  "dropout",
  "unknown"};

unsigned int parseLayerProperty(std::string property) {
  unsigned int i;

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }

  return (unsigned int)Layer::PropertyType::unknown;
}

std::string propToStr(unsigned int type) { return property_string[type]; }

unsigned int parseOptProperty(std::string property) {
  unsigned int i;

  /**
   * @brief     Optimizer Properties
   * learning_rate = 0,
   * decay_rate = 1,
   * decay_steps = 2
   * beta1 = 3,
   * beta2 = 4,
   * epsilon = 5,
   * continue_train = 6,
   */
  std::array<std::string, 7> property_string = {
    "learning_rate", "decay_rate", "decay_steps",   "beta1",
    "beta2",         "epsilon",    "continue_train"};

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }

  return (unsigned int)Optimizer::PropertyType::unknown;
}

unsigned int parseNetProperty(std::string property) {
  unsigned int i;

  /**
   * @brief     Network Properties
   * loss_val = 0,
   * loss = 1,
   * batch_size = 2,
   * epochs = 3,
   * save_path = 4
   */
  std::array<std::string, 5> property_string = {
    "loss_val", "loss", "batch_size", "epochs", "save_path"};

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }

  return (unsigned int)NeuralNetwork::PropertyType::unknown;
}

unsigned int parseDataProperty(std::string property) {
  unsigned int i;

  /**
   * @brief     Data Properties
   * train_data = 0,
   * val_data = 1,
   * test_data = 2,
   * buffer_size = 3
   */
  std::array<std::string, 5> property_string = {"train_data", "val_data",
                                                "test_data", "buffer_size"};

  for (i = 0; i < property_string.size(); i++) {
    unsigned int size = (property_string[i].size() > property.size())
                          ? property_string[i].size()
                          : property.size();

    if (!strncasecmp(property_string[i].c_str(), property.c_str(), size)) {
      return (i);
    }
  }

  return (unsigned int)DataBuffer::PropertyType::unknown;
}

int setUint(unsigned int &val, const std::string &str) {
  int status = ML_ERROR_NONE;
  try {
    val = (unsigned int)std::stoul(str.c_str());
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
  static const std::regex words_regex("[^\\s.,:;!?]+");
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

const char *getValues(std::vector<int> values, const char *delimiter) {
  std::stringstream vec_str;

  if (values.empty())
    return "unknown";

  std::copy(values.begin(), values.end() - 1,
            std::ostream_iterator<int>(vec_str, delimiter));
  vec_str << values.back();

  return std::move(vec_str.str().c_str());
}

std::vector<std::string> split(const std::string &s, const std::regex &reg) {
  std::vector<std::string> out;
  char char_to_remove[NUM_SKIP_CHAR] = {' ', '[', ']'};
  std::string str = s;
  for (unsigned int i = 0; i < NUM_SKIP_CHAR; ++i) {
    str.erase(std::remove(str.begin(), str.end(), char_to_remove[i]),
              str.end());
  }
  std::regex_token_iterator<std::string::iterator> end;
  std::regex_token_iterator<std::string::iterator> iter(str.begin(), str.end(),
                                                        reg, -1);

  while (iter != end) {
    out.push_back(*iter);
    ++iter;
  }
  return out;
}

bool istrequal(const std::string &a, const std::string &b) {
  if (a.size() != b.size())
    return false;

  return std::equal(a.begin(), a.end(), b.begin(), [](char a_, char b_) {
    return tolower(a_) == tolower(b_);
  });
}

} /* namespace nntrainer */
