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
#include "databuffer_file.h"
#include "databuffer_func.h"
#include "neuralnet.h"
#include "nntrainer_test_util.h"
#include "util_func.h"
#include <fstream>
#include <nntrainer_error.h>

/**
 * @brief Neural Network Model Configuration with ini file (possitive test )
 */
TEST(nntrainer_NeuralNetwork, setConfig_01_p) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test.ini";
  RESET_CONFIG(config_file.c_str());
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
  RESET_CONFIG(config_file.c_str());
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
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG("./test.ini");
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
TEST(nntrainer_NeuralNetwork, init_07_p) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
  replaceString("TrainData = trainingSet.dat", "", "./test.ini");
  nntrainer::NeuralNetwork NN;
  status = NN.setConfig("./test.ini");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = NN.init();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model initialization
 */
TEST(nntrainer_NeuralNetwork, init_08_n) {
  int status = ML_ERROR_NONE;
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG("./test.ini");
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
  RESET_CONFIG(config_file.c_str());
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
  status =
    layer.initialize(1, 1, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status =
    layer.initialize(1, 0, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  status =
    layer.initialize(1, 1, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
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
  status =
    layer.initialize(1, 1, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status =
    layer.initialize(1, 0, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_03_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.initialize(1, 1, 1, false, true, nntrainer::WEIGHT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_04_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status =
    layer.initialize(1, 1, 1, true, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_05_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status =
    layer.initialize(1, 0, 1, true, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_06_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.initialize(1, 1, 1, true, true, nntrainer::WEIGHT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status =
    layer.initialize(1, 1, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
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
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setOptimizer_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status =
    layer.initialize(1, 1, 1, true, true, nntrainer::WEIGHT_XAVIER_NORMAL);
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
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setCost_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.setCost(nntrainer::COST_ENTROPY);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setCost_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  status = layer.setCost(nntrainer::COST_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status =
    layer.initialize(1, 1, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  status =
    layer.initialize(1, 0, 1, false, true, nntrainer::WEIGHT_XAVIER_NORMAL);
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

TEST(nntrainer_TensorDim, setTensorDim_01_p) {
  int status = ML_ERROR_NONE;

  nntrainer::TensorDim tensor_dim;
  status = tensor_dim.setTensorDim("1:2:3:4");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_TensorDim, setTensorDim_02_n) {
  int status = ML_ERROR_NONE;

  nntrainer::TensorDim tensor_dim;
  status = tensor_dim.setTensorDim("1:2:3:4:5");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, Tensor_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(1, 2, 3);
  ASSERT_NE(nullptr, tensor.getData());
  if (tensor.getValue(0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_p) {
  int status = ML_ERROR_NONE;
  int height = 3;
  int width = 10;
  std::vector<std::vector<float>> in;
  for (int i = 0; i < height; ++i) {
    std::vector<float> tv;
    for (int j = 0; j < width; ++j) {
      tv.push_back(i * 2.0 + j);
    }
    in.push_back(tv);
  }

  nntrainer::Tensor tensor = nntrainer::Tensor(in);
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue(0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_03_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<float>>> in;
  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<float>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<float> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::Tensor tensor = nntrainer::Tensor(in);
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue(0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor result = input.multiply(0.0);
  if (result.getValue(0, 1, 1) != 0.0)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.multiply(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] * indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.multiply(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, divide_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.divide(1.0);
  if (result.getValue(0, 1, 1) != input.getValue(0, 1, 1))
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, divide_02_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  ASSERT_EXCEPTION({ input.divide(0.0); }, std::runtime_error,
                   "Error: Divide by zero");
}

TEST(nntrainer_Tensor, divide_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.divide(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, add_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] + 1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] + indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.add(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, subtract_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] - 1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, subtract_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != 0.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, subtract_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.subtract(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, sum_01_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  ASSERT_EXCEPTION({ input.sum(3); }, std::out_of_range,
                   "Error: Cannot exceede 2");
}

TEST(nntrainer_Tensor, sum_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 2;
  int width = 10;

  float ans0[1][2][10] = {{{21, 24, 27, 30, 33, 36, 39, 42, 45, 48},
                           {51, 54, 57, 60, 63, 66, 69, 72, 75, 78}}};

  float ans1[3][1][10] = {{{18, 20, 22, 24, 26, 28, 30, 32, 34, 36}},
                          {{24, 26, 28, 30, 32, 34, 36, 38, 40, 42}},
                          {{30, 32, 34, 36, 38, 40, 42, 44, 46, 48}}};
  float ans2[3][2][1] = {{{154}, {164}}, {{160}, {170}}, {{166}, {176}}};

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result1 = input.sum(1);
  nntrainer::Tensor result2 = input.sum(2);

  for (int i = 0; i < result0.getBatch(); ++i) {
    for (int j = 0; j < result0.getHeight(); ++j) {
      for (int k = 0; k < result0.getWidth(); ++k) {
        if (ans0[i][j][k] != result0.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_test;
        }
      }
    }
  }

  for (int i = 0; i < result1.getBatch(); ++i) {
    for (int j = 0; j < result1.getHeight(); ++j) {
      for (int k = 0; k < result1.getWidth(); ++k) {
        if (ans1[i][j][k] != result1.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_test;
        }
      }
    }
  }

  for (int i = 0; i < result2.getBatch(); ++i) {
    for (int j = 0; j < result2.getHeight(); ++j) {
      for (int k = 0; k < result2.getWidth(); ++k) {
        if (ans2[i][j][k] != result2.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_test;
        }
      }
    }
  }

end_test:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, sum_03_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 2;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.sum();
  if (result.getValue(0, 0, 0) != 210 || result.getValue(1, 0, 0) != 330 ||
      result.getValue(2, 0, 0) != 450)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, dot_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 3;
  float ans[3][3][3] = {
    {{30, 36, 42}, {66, 81, 96}, {102, 126, 150}},
    {{435, 468, 501}, {552, 594, 636}, {669, 720, 771}},
    {{1326, 1386, 1446}, {1524, 1593, 1662}, {1722, 1800, 1878}}};

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.dot(input);

  for (int i = 0; i < result.getBatch(); ++i) {
    for (int j = 0; j < result.getHeight(); ++j) {
      for (int k = 0; k < result.getWidth(); ++k) {
        if (ans[i][j][k] != result.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_dot_01_p;
        }
      }
    }
  }
end_dot_01_p:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, transpose_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 3;
  float ans[3][3][3] = {{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}},
                        {{10, 13, 16}, {11, 14, 17}, {12, 15, 18}},
                        {{19, 22, 25}, {20, 23, 26}, {21, 24, 27}}};
  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.transpose();

  for (int i = 0; i < result.getBatch(); ++i) {
    for (int j = 0; j < result.getHeight(); ++j) {
      for (int k = 0; k < result.getWidth(); ++k) {
        if (ans[i][j][k] != result.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_transpose_01_p;
        }
      }
    }
  }
end_transpose_01_p:
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
