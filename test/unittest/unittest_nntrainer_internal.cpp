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
 * @file        unittest_nntrainer_internal.cpp
 * @date        10 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <fstream>

#include <databuffer_file.h>
#include <databuffer_func.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <optimizer_factory.h>
#include <util_func.h>

#include <nntrainer_test_util.h>

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_01_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file, config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_01_n) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("[Model]", "", "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_02_n) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("adam", "aaaadam", "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_03_n) {
  RESET_CONFIG("./test.ini");
  replaceString("Input_Shape = 1:1:62720", "Input_Shape = 1:1:0", "./test.ini",
                config_str);

  nntrainer::NeuralNetwork NN;

  int status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization, there is warning when no
 * input_shape is provided
 */
TEST(nntrainer_NeuralNetwork, load_config_04_p) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("Input_Shape = 1:1:62720", "", "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_05_n) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("Learning_rate = 0.0001", "Learning_rate = -0.0001",
                "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_06_p) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("TrainData = trainingSet.dat", "", "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/*
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_07_p) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("bias_initializer = zeros", "Bias_Initializer = he_normal",
                "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Configuration with ini file (negative test)
 */
TEST(nntrainer_NeuralNetwork, load_config_08_n) {
  int status = ML_ERROR_NONE;
  std::string config_file = "../test/not_found.ini";
  nntrainer::NeuralNetwork NN;
  status = NN.loadFromConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_02_p) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("TestData = testSet.dat", "", "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, load_config_07_n) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("LabelData = label.dat", "", "./test.ini", config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_03_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("ValidData = valSet.dat", "", config_file, config_str);
  nntrainer::NeuralNetwork NN;

  status = NN.loadFromConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_01_p) {
  std::shared_ptr<nntrainer::Optimizer> op;
  EXPECT_NO_THROW(op = nntrainer::createOptimizer("adam"));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, setType_02_p) {
  std::shared_ptr<nntrainer::Optimizer> op;
  EXPECT_NO_THROW(op = nntrainer::createOptimizer("sgd"));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, setType_03_n) {
  std::shared_ptr<nntrainer::Optimizer> op;
  EXPECT_THROW(op = nntrainer::createOptimizer("non-existing type"),
               std::invalid_argument);
}

/**
 * @brief Convolution 2D Layer
 */
TEST(nntrainer_Conv2DLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("ValidData = valSet.dat", "", config_file, config_str2);
  nntrainer::NeuralNetwork NN;
  status = NN.loadFromConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Conv2D Layer
 */
TEST(nntrainer_Conv2DLayer, initialize_02_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("flatten = false", "flatten = true", config_file, config_str2);
  nntrainer::NeuralNetwork NN;
  status = NN.loadFromConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Flatten Layer
 */
TEST(nntrainer_Layer, initialize_03_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("flatten = false", "flatten = true", config_file, config_str2);
  nntrainer::NeuralNetwork NN;
  status = NN.loadFromConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_throw_if, throw_invalid_arg_p) {
  try {
    NNTR_THROW_IF(1 == 1, std::invalid_argument) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
  }

  try {
    NNTR_THROW_IF(true, std::invalid_argument) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
  }

  bool hit = false;
  auto cleanup = [&hit] { hit = true; };
  try {
    NNTR_THROW_IF_CLEANUP(true, std::invalid_argument, cleanup) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
    EXPECT_TRUE(hit);
  }
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
