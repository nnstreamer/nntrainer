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
#include "neuralnet.h"
#include "util_func.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nntrainer_error.h>

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
                               "Epoch = 30000"
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
                               "Id = 0"
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
                               "Type = output"
                               "\n"
                               "Id = 1"
                               "\n"
                               "HiddenSize = 10"
                               "\n"
                               "Bias_zero = true"
                               "\n"
                               "Activation = softmax"
                               "\n";

#define GEN_TEST_INPUT(input, eqation_i_j_k) \
  do {                                       \
    for (int i = 0; i < batch; ++i) {        \
      for (int j = 0; j < height; ++j) {     \
        for (int k = 0; k < width; ++k) {    \
          float val = eqation_i_j_k;         \
          input.setValue(i, j, k, val);      \
        }                                    \
      }                                      \
    }                                        \
  } while (0)

static void replaceString(const std::string &from, const std::string &to,
                          const std::string n) {
  size_t start_pos = 0;
  std::string s = config_str;
  while ((start_pos = s.find(from, start_pos)) != std::string::npos) {
    s.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  std::ofstream data_file(n.c_str());
  data_file << s;
  data_file.close();
}

/**
 * @brief Neural Network Model Configuration with ini file (possitive test )
 */
TEST(nntrainer_NeuralNetwork, setConfig_01_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file);
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Configuration with ini file (negative test )
 */
TEST(nntrainer_NeuralNetwork, setConfig_02_n) {
  int status = ML_ERROR_NONE;
  std::string config_file = "../test/not_found.ini";
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_01_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file);
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_02_n) {
  int status = ML_ERROR_NONE;
  replaceString("Layers = inputlayer outputlayer", "", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_03_n) {
  int status = ML_ERROR_NONE;
  replaceString("adam", "aaaadam", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_04_n) {
  int status = ML_ERROR_NONE;
  replaceString("HiddenSize = 62720", "HiddenSize = 0", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_05_n) {
  int status = ML_ERROR_NONE;
  replaceString("HiddenSize = 62720", "", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_06_n) {
  int status = ML_ERROR_NONE;
  replaceString("Learning_rate = 0.0001", "Learning_rate = -0.0001",
                "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_07_n) {
  int status = ML_ERROR_NONE;
  replaceString("TrainData = trainingSet.dat", "", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_08_n) {
  int status = ML_ERROR_NONE;
  replaceString("TestData = testSet.dat", "", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_09_n) {
  int status = ML_ERROR_NONE;
  replaceString("HiddenSize = 10", "HiddenSize = 9", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_10_n) {
  int status = ML_ERROR_NONE;
  replaceString("LabelData = label.dat", "", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_011_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  replaceString("ValidData = valSet.dat", "", config_file);
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig(config_file);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer set number of Class
 */
TEST(nntrainer_DataBuffer, setClassNum_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(3);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setClassNum(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer set number of Class
 */
TEST(nntrainer_DataBuffer, setClassNum_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer set train Data file
 */
TEST(nntrainer_DataBuffer, setDataFile_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile("./trainingSet.dat", nntrainer::DATA_TRAIN);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer set train Data file
 */
TEST(nntrainer_DataBuffer, setDataFile_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile("./no_exist.dat", nntrainer::DATA_TRAIN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer set train Data file
 */
TEST(nntrainer_DataBuffer, setDataFile_03_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("./label.dat", nntrainer::DATA_LABEL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer set train Data file
 */
TEST(nntrainer_DataBuffer, setDataFile_04_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(3);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("./label.dat", nntrainer::DATA_LABEL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Optimizer set type
 */
TEST(nntrainer_Optimizer, setType_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Optimizer set type
 */
TEST(nntrainer_Optimizer, setType_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::sgd;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Optimizer set type
 */
TEST(nntrainer_Optimizer, setType_03_n) {
  int status = ML_ERROR_NONE;
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::unknown;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Optimizer set Opt Param
 */
TEST(nntrainer_Optimizer, setOptParam_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = -0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
  p.weight_decay.type = nntrainer::WeightDecayType::l2norm;
  p.weight_decay.lambda = 0.001;
  status = op.setOptParam(p);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status = layer.initialize(1, 0, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
  p.weight_decay.type = nntrainer::WeightDecayType::l2norm;
  p.weight_decay.lambda = 0.001;
  status = op.setOptParam(p);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setOptimizer(op);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, setActivation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status = layer.setActivation(nntrainer::ACT_TANH);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, setActivation_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.initialize(1, 0, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_03_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
  p.weight_decay.type = nntrainer::WeightDecayType::l2norm;
  p.weight_decay.lambda = 0.001;
  status = op.setOptParam(p);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setOptimizer(op);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setActivation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.setActivation(nntrainer::ACT_TANH);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setActivation_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.initialize(1, 0, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, initialize_03_n) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
  p.weight_decay.type = nntrainer::WeightDecayType::l2norm;
  p.weight_decay.lambda = 0.001;
  status = op.setOptParam(p);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setOptimizer(op);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, setActivation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.setActivation(nntrainer::ACT_SIGMOID);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, setActivation_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, setCost_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.setCost(nntrainer::COST_ENTROPY);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Output Layer
 */
TEST(nntrainer_OutputLayer, setCost_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::OutputLayer layer;
  status = layer.setCost(nntrainer::COST_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status = layer.initialize(1, 0, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status = layer.initialize(1, 1, 1, 0, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
  p.weight_decay.type = nntrainer::WeightDecayType::l2norm;
  p.weight_decay.lambda = 0.001;
  status = op.setOptParam(p);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setOptimizer(op);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, setActivation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status = layer.setActivation(nntrainer::ACT_SIGMOID);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, setActivation_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setFeatureSize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("./trainingSet.dat", nntrainer::DATA_TRAIN);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setFeatureSize(62720);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setFeatureSize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("./trainingSet.dat", nntrainer::DATA_TRAIN);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setFeatureSize(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setMiniBatch_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setMiniBatch(32);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setMiniBatch_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setMiniBatch(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, init_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setMiniBatch(32);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("trainingSet.dat", nntrainer::DATA_TRAIN);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("valSet.dat", nntrainer::DATA_VAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("testSet.dat", nntrainer::DATA_TEST);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("label.dat", nntrainer::DATA_LABEL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setFeatureSize(62720);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.init();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, init_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setMiniBatch(32);
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("trainingSet.dat", nntrainer::DATA_TRAIN);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("valSet.dat", nntrainer::DATA_VAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("testSet.dat", nntrainer::DATA_TEST);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile("label.dat", nntrainer::DATA_LABEL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.init();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_util_func, softmax_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 1;
  int width = 10;
  float results[10] = {7.80134161e-05, 2.12062451e-04, 5.76445508e-04,
                       1.56694135e-03, 4.25938820e-03, 1.15782175e-02,
                       3.14728583e-02, 8.55520989e-02, 2.32554716e-01,
                       6.32149258e-01};

  nntrainer::Tensor T(batch, height, width);
  nntrainer::Tensor Results(batch, height, width);

  GEN_TEST_INPUT(T, (i * (width) + k + 1));

  Results = T.apply(nntrainer::softmax);
  float *data = Results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - results[i % width]) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, random_01_p) {
  int status = ML_ERROR_INVALID_PARAMETER;
  srand(time(NULL));
  float x = nntrainer::random(0.0);
  if (-1.0 < x && x < 1.0)
    status = ML_ERROR_NONE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, sqrtFloat_01_p) {
  int status = ML_ERROR_INVALID_PARAMETER;

  float x = 9871.0;
  float sx = nntrainer::sqrtFloat(x);

  if ((sx * sx - x) * (sx * sx - x) < tolerance)
    status = ML_ERROR_NONE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, logFloat_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (width) + k + 1);

  nntrainer::Tensor Results = input.apply(nntrainer::logFloat);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - (float)log(indata[i])) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, sigmoid_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, (k - 4) * 0.1);

  nntrainer::Tensor Results = input.apply(nntrainer::sigmoid);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - (float)(1 / (1 + exp(-indata[i])))) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, tanhFloat_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, (k - 4) * 0.1);

  nntrainer::Tensor Results = input.apply(nntrainer::tanhFloat);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - (float)(tanh(indata[i]))) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, relu_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, (k - 4) * 0.1);

  nntrainer::Tensor Results = input.apply(nntrainer::relu);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    float r = (indata[i] <= 0.0) ? 0.0 : indata[i];
    if ((data[i] - r) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
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
