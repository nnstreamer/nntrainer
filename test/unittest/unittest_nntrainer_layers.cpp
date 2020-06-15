/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file        unittest_nntrainer_layers.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for layers.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <fstream>
#include <input_layer.h>
#include <loss_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>
#include <pooling2d_layer.h>
#include <util_func.h>

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:3:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:0:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
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
 * @brief Input Layer
 */
TEST(nntrainer_InputLayer, checkValidation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::InputLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  layer.setActivation(nntrainer::ACT_TANH);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:0:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_03_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_04_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_05_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:0:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer, initialize_06_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST(nntrainer_FullyConnectedLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  layer.setUnit(1);
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
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
  layer.setUnit(1);
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(true);
  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
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
TEST(nntrainer_FullyConnectedLayer, checkValidation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer;
  layer.setUnit(1);
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);

  layer.setActivation(nntrainer::ACT_RELU);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);

  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, initialize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:0:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, setOptimizer_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:0:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);

  nntrainer::Optimizer op;
  nntrainer::OptType t = nntrainer::OptType::adam;
  nntrainer::OptParam p;
  status = op.setType(t);
  EXPECT_EQ(status, ML_ERROR_NONE);
  p.learning_rate = 0.001;
  p.beta1 = 0.9;
  p.beta2 = 0.9999;
  p.epsilon = 1e-7;
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
 * @brief Batch Normalization Layer
 */
TEST(nntrainer_BatchNormalizationLayer, checkValidation_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::BatchNormalizationLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:1:28");
  layer.setInputDimension(d);
  status = layer.initialize(false);
  layer.setActivation(nntrainer::ACT_RELU);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Convolution 2D Layer
 */
TEST(nntrainer_Conv2DLayer, setProperty_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Conv2DLayer layer;
  std::vector<std::string> input_str;

  input_str.push_back("input_shape=32:3:28:28");
  input_str.push_back("bias_zero=true");
  input_str.push_back("activation=sigmoid");
  input_str.push_back("weight_decay=l2norm");
  input_str.push_back("weight_decay_lambda = 0.005");
  input_str.push_back("weight_ini=xavier_uniform");
  input_str.push_back("filter=12");
  input_str.push_back("kernel_size= 5,5");
  input_str.push_back("stride=3, 3");
  input_str.push_back("padding=1,1");
  input_str.push_back("normalization=true");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Convolution 2D Layer
 */
TEST(nntrainer_Conv2DLayer, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Conv2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("32:3:28:28");

  input_str.push_back("input_shape=32:3:28:28");
  input_str.push_back("bias_zero=true");
  input_str.push_back("activation=sigmoid");
  input_str.push_back("weight_decay=l2norm");
  input_str.push_back("weight_decay_lambda = 0.005");
  input_str.push_back("weight_ini=xavier_uniform");
  input_str.push_back("filter=6");
  input_str.push_back("kernel_size= 5,5");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer.setInputDimension(previous_dim);

  status = layer.initialize(true);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Convolution 2D Layer
 */
TEST(nntrainer_Conv2DLayer, save_read_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Conv2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("32:3:28:28");

  input_str.push_back("input_shape=32:3:28:28");
  input_str.push_back("bias_zero=true");
  input_str.push_back("activation=sigmoid");
  input_str.push_back("weight_decay=l2norm");
  input_str.push_back("weight_decay_lambda = 0.005");
  input_str.push_back("weight_ini=xavier_uniform");
  input_str.push_back("filter=6");
  input_str.push_back("kernel_size= 5,5");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer.setInputDimension(previous_dim);

  status = layer.initialize(true);
  EXPECT_EQ(status, ML_ERROR_NONE);
  std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
  layer.save(save_file);
  save_file.close();

  std::ifstream read_file("save.bin");
  layer.read(read_file);
  read_file.seekg(0, std::ios::beg);

  std::ofstream save_file2("save1.bin", std::ios::out | std::ios::binary);
  layer.save(save_file2);
  save_file2.close();

  std::ifstream read_file2("save1.bin");
  float d1, d2;

  for (int i = 0; i < (5 * 5 * 3 * 6) + 6; ++i) {
    read_file.read((char *)&d1, sizeof(float));
    read_file2.read((char *)&d2, sizeof(float));
    EXPECT_EQ(d1, d2);
  }
}

/**
 * @brief Convolution 2D Layer
 */
TEST(nntrainer_Conv2DLayer, forwarding_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Conv2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("1:3:7:7");

  input_str.push_back("input_shape=1:3:7:7");
  input_str.push_back("bias_zero=true");
  input_str.push_back("activation=sigmoid");
  input_str.push_back("weight_decay=l2norm");
  input_str.push_back("weight_decay_lambda = 0.005");
  input_str.push_back("weight_ini=xavier_uniform");
  input_str.push_back("filter=2");
  input_str.push_back("kernel_size= 3,3");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer.setInputDimension(previous_dim);

  status = layer.initialize(true);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Tensor in(1, 3, 7, 7);
  nntrainer::Tensor result(1, 2, 5, 5);
  nntrainer::Tensor out;
  float *out_ptr, *golden;
  std::ifstream file("conv2dLayer.in");
  in.read(file);
  file.close();
  std::ifstream kfile("conv2dKernel.in");
  layer.read(kfile);
  kfile.close();
  std::ifstream rfile("goldenConv2DResult.out");
  result.read(rfile);
  rfile.close();
  out = layer.forwarding(in, status);

  golden = result.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 1 * 2 * 5 * 5; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Pooling 2D Layer
 */
TEST(nntrainer_Pooling2D, setProperty_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Pooling2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("1:2:5:5");
  layer.setInputDimension(previous_dim);

  input_str.push_back("pooling_size= 2,2");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");
  input_str.push_back("pooling = average");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Pooling 2D Layer
 */
TEST(nntrainer_Pooling2D, initialize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Pooling2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("1:2:5:5");
  layer.setInputDimension(previous_dim);

  input_str.push_back("pooling_size= 2,2");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");
  input_str.push_back("pooling = average");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Pooling 2D Layer
 */
TEST(nntrainer_Pooling2D, forwarding_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Pooling2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("1:2:5:5");
  layer.setInputDimension(previous_dim);

  input_str.push_back("pooling_size= 2,2");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");
  input_str.push_back("pooling = max");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
  nntrainer::Tensor in(1, 2, 5, 5);
  nntrainer::Tensor out, result(1, 2, 4, 4);
  std::ifstream file("goldenConv2DResult.out");
  in.read(file);
  file.close();
  out = layer.forwarding(in, status);
  EXPECT_EQ(status, ML_ERROR_NONE);
  std::ifstream rfile("goldenPooling2DResult.out");
  result.read(rfile);
  rfile.close();
  float *out_ptr, *golden;

  golden = result.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 1 * 2 * 4 * 4; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setCost_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setCost(nntrainer::COST_ENTROPY);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setCost_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setCost(nntrainer::COST_UNKNOWN);
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
