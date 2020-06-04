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
 * @file	nntrainer_test_util.h
 * @date	28 April 2020
 * @brief	This is util functions for test
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_TEST_UTIL_H__
#define __NNTRAINER_TEST_UTIL_H__
#ifdef __cplusplus

#include "nntrainer_log.h"
#include <fstream>
#include <gtest/gtest.h>

#define tolerance 10e-5

const std::string config_str = "[Network]"
                               "\n"
                               "Type = NeuralNetwork"
                               "\n"
                               "Layers = inputlayer outputlayer"
                               "\n"
                               "Learning_rate = 0.0001"
                               "\n"
                               "Decay_rate = 0.96"
                               "\n"
                               "Decay_steps = 1000"
                               "\n"
                               "Epoch = 1"
                               "\n"
                               "Optimizer = adam"
                               "\n"
                               "Cost = cross"
                               "\n"
                               "Weight_Decay = l2norm"
                               "\n"
                               "weight_Decay_Lambda = 0.005"
                               "\n"
                               "Model = 'model.bin'"
                               "\n"
                               "minibatch = 32"
                               "\n"
                               "beta1 = 0.9"
                               "\n"
                               "beta2 = 0.9999"
                               "\n"
                               "epsilon = 1e-7"
                               "\n"
                               "[DataSet]"
                               "\n"
                               "BufferSize=100"
                               "\n"
                               "TrainData = trainingSet.dat"
                               "\n"
                               "ValidData = valSet.dat"
                               "\n"
                               "LabelData = label.dat"
                               "\n"
                               "[inputlayer]"
                               "\n"
                               "Type = input"
                               "\n"
                               "HiddenSize = 62720"
                               "\n"
                               "Bias_zero = true"
                               "\n"
                               "Normalization = true"
                               "\n"
                               "Activation = sigmoid"
                               "\n"
                               "[outputlayer]"
                               "\n"
                               "Type = fully_connected"
                               "\n"
                               "HiddenSize = 10"
                               "\n"
                               "Bias_zero = true"
                               "\n"
                               "Activation = softmax"
                               "\n";

#define GEN_TEST_INPUT(input, eqation_i_j_k_l) \
  do {                                         \
    for (int i = 0; i < batch; ++i) {          \
      for (int j = 0; j < channel; ++j) {      \
        for (int k = 0; k < height; ++k) {     \
          for (int l = 0; l < width; ++l) {    \
            float val = eqation_i_j_k_l;       \
            input.setValue(i, j, k, l, val);   \
          }                                    \
        }                                      \
      }                                        \
    }                                          \
  } while (0)

#define ASSERT_EXCEPTION(TRY_BLOCK, EXCEPTION_TYPE, MESSAGE)                  \
  try {                                                                       \
    TRY_BLOCK                                                                 \
    FAIL() << "exception '" << MESSAGE << "' not thrown at all!";             \
  } catch (const EXCEPTION_TYPE &e) {                                         \
    EXPECT_EQ(std::string(MESSAGE), e.what())                                 \
      << " exception message is incorrect. Expected the following "           \
         "message:\n\n"                                                       \
      << MESSAGE << "\n";                                                     \
  } catch (...) {                                                             \
    FAIL() << "exception '" << MESSAGE << "' not thrown with expected type '" \
           << #EXCEPTION_TYPE << "'!";                                        \
  }

#define RESET_CONFIG(conf_name)                              \
  do {                                                       \
    std::ifstream file_stream(conf_name, std::ifstream::in); \
    if (file_stream.good()) {                                \
      if (std::remove(conf_name) != 0)                       \
        ml_loge("Error: Cannot delete file: %s", conf_name); \
      else                                                   \
        ml_logi("Info: deleteing file: %s", conf_name);      \
    }                                                        \
  } while (0)

/**
 * @brief replace string and save in file
 * @param[in] from string to be replaced
 * @param[in] to string to repalce with
 * @param[in] n file name to save
 * @retval void
 */
void replaceString(const std::string &from, const std::string &to,
                   const std::string n);

/**
 * @brief      get data which size is mini batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false
 */
bool getMiniBatch_train(float *outVec, float *outLabel, int *status);

/**
 * @brief      get data which size is mini batch for val
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false
 */
bool getMiniBatch_val(float *outVec, float *outLabel, int *status);

#endif /* __cplusplus */
#endif /* __NNTRAINER_TEST_UTIL_H__ */
