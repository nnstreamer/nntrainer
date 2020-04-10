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
 */

/**
 * @file        unittest_nntrainer_interanl.cpp
 * @date        10 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <nntrainer_error.h>
#include "neuralnet.h"

/**
 * @brief Neural Network Model Configuration with ini file (possitive test )
 */
TEST(nntrainer_NeuralNetwork, setConfig_01_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "../test/tizen_capi/test_conf.ini";
  Network::NeuralNetwork NN;
  status = NN.setConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Configuration with ini file (negative test )
 */
TEST(nntrainer_NeuralNetwork, setConfig_02_n) {
  int status = ML_ERROR_NONE;
  std::string config_file = "../test/not_found.ini";
  Network::NeuralNetwork NN;
  status = NN.setConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  testing::InitGoogleTest(&argc, argv);

  result = RUN_ALL_TESTS();

  return result;
}
