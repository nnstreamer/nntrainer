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
#include <activation_layer.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
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
  std::ifstream file("test_1_conv2DLayer.in");
  in.read(file);
  file.close();
  std::ifstream kfile("test_1_conv2DKernel.in");
  layer.read(kfile);
  kfile.close();
  std::ifstream rfile("test_1_goldenConv2DResult.out");
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
 * @brief Convolution 2D Layer
 */
TEST(nntrainer_Conv2DLayer, forwarding_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Conv2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("2:3:7:7");

  input_str.push_back("input_shape=2:3:7:7");
  input_str.push_back("bias_zero=true");
  input_str.push_back("activation=sigmoid");
  input_str.push_back("weight_decay=l2norm");
  input_str.push_back("weight_decay_lambda = 0.005");
  input_str.push_back("weight_ini=xavier_uniform");
  input_str.push_back("filter=3");
  input_str.push_back("kernel_size= 3,3");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer.setInputDimension(previous_dim);

  status = layer.initialize(true);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Tensor in(2, 3, 7, 7);
  nntrainer::Tensor result(2, 3, 5, 5);
  nntrainer::Tensor out;
  float *out_ptr, *golden;
  std::ifstream file("test_2_conv2DLayer.in");
  in.read(file);
  file.close();
  std::ifstream kfile("test_2_conv2DKernel.in");
  layer.read(kfile);
  kfile.close();
  std::ifstream rfile("test_2_goldenConv2DResult.out");
  result.read(rfile);
  rfile.close();
  out = layer.forwarding(in, status);

  golden = result.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 2 * 3 * 5 * 5; ++i) {
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
  previous_dim.setTensorDim("2:3:5:5");
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
  previous_dim.setTensorDim("2:3:5:5");
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
  std::ifstream file("test_1_goldenConv2DResult.out");
  in.read(file);
  file.close();
  out = layer.forwarding(in, status);
  EXPECT_EQ(status, ML_ERROR_NONE);
  std::ifstream rfile("test_1_goldenPooling2DResult.out");
  result.read(rfile);
  rfile.close();
  float *out_ptr, *golden;

  golden = result.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 32; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Pooling 2D Layer
 */
TEST(nntrainer_Pooling2D, forwarding_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Pooling2DLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("2:3:5:5");
  layer.setInputDimension(previous_dim);

  input_str.push_back("pooling_size= 2,2");
  input_str.push_back("stride=1, 1");
  input_str.push_back("padding=0,0");
  input_str.push_back("pooling = max");

  status = layer.setProperty(input_str);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
  nntrainer::Tensor in(2, 3, 5, 5);
  nntrainer::Tensor out, result(2, 3, 4, 4);
  std::ifstream file("test_2_goldenConv2DResult.out");
  in.read(file);
  file.close();
  out = layer.forwarding(in, status);
  EXPECT_EQ(status, ML_ERROR_NONE);
  std::ifstream rfile("test_2_goldenPooling2DResult.out");
  result.read(rfile);
  rfile.close();
  float *out_ptr, *golden;

  golden = result.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 2 * 3 * 4 * 4; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Pooling 2D Layer
 */
TEST(nntrainer_Pooling2D, backwarding_01_p) {
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

  nntrainer::Tensor in(1, 2, 4, 4);
  nntrainer::Tensor out, result(1, 2, 5, 5);
  std::ifstream file("test_1_goldenConv2DResult.out");
  result.read(file);
  file.close();
  out = layer.forwarding(result, status);
  EXPECT_EQ(status, ML_ERROR_NONE);

  for (unsigned int i = 0; i < in.getDim().getDataLen(); ++i) {
    in.getData()[i] = 1.0;
  }

  out = layer.backwarding(in, 0);

  float golden[50] = {0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 0, 1, 4, 0, 0, 0, 1,
                      2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 4, 0,
                      1, 0, 3, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 1, 0};

  float *out_ptr = out.getData();

  for (int i = 0; i < 50; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Pooling 2D Layer
 */
TEST(nntrainer_Pooling2D, backwarding_02_p) {
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

  nntrainer::Tensor in(1, 2, 4, 4);
  nntrainer::Tensor out, result(1, 2, 5, 5);
  std::ifstream file("test_1_goldenConv2DResult.out");
  result.read(file);
  file.close();
  out = layer.forwarding(result, status);
  EXPECT_EQ(status, ML_ERROR_NONE);

  for (unsigned int i = 0; i < in.getDim().getDataLen(); ++i) {
    in.getData()[i] = 1.0;
  }

  out = layer.backwarding(in, 0);

  float golden[50] = {0.25, 0.5, 0.5, 0.5, 0.25, 0.5,  1.0, 1.0, 1.0, 0.5,
                      0.5,  1.0, 1.0, 1.0, 0.5,  0.5,  1.0, 1.0, 1.0, 0.5,
                      0.25, 0.5, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.25,
                      0.5,  1.0, 1.0, 1.0, 0.5,  0.5,  1.0, 1.0, 1.0, 0.5,
                      0.5,  1.0, 1.0, 1.0, 0.5,  0.25, 0.5, 0.5, 0.5, 0.25};
  float *out_ptr = out.getData();

  for (int i = 0; i < 50; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Flatten Layer
 */
TEST(nntrainer_Flatten, forwarding_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FlattenLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("1:2:4:4");
  layer.setInputDimension(previous_dim);

  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Tensor in(1, 2, 4, 4);
  nntrainer::Tensor out, result(1, 1, 1, 32);
  std::ifstream file("test_1_goldenPooling2DResult.out");
  in.read(file);
  file.close();
  out = layer.forwarding(in, status);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(out.getDim().batch(), 1);
  EXPECT_EQ(out.getDim().channel(), 1);
  EXPECT_EQ(out.getDim().height(), 1);
  EXPECT_EQ(out.getDim().width(), 32);

  float *out_ptr, *golden;
  golden = in.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 32; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Flatten Layer
 */
TEST(nntrainer_Flatten, forwarding_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FlattenLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("2:3:4:4");
  layer.setInputDimension(previous_dim);

  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Tensor in(2, 3, 4, 4);
  nntrainer::Tensor out, result(2, 1, 1, 48);
  std::ifstream file("test_2_goldenPooling2DResult.out");
  in.read(file);
  file.close();
  out = layer.forwarding(in, status);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(out.getDim().batch(), 2);
  EXPECT_EQ(out.getDim().channel(), 1);
  EXPECT_EQ(out.getDim().height(), 1);
  EXPECT_EQ(out.getDim().width(), 48);

  float *out_ptr, *golden;
  golden = in.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 48 * 2; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Flatten Layer
 */
TEST(nntrainer_Flatten, backwarding_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FlattenLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("1:2:4:4");
  layer.setInputDimension(previous_dim);

  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Tensor in(1, 1, 1, 32);
  nntrainer::Tensor out, result(1, 1, 1, 32);
  std::ifstream file("test_1_goldenPooling2DResult.out");
  in.read(file);
  file.close();

  out = layer.backwarding(in, 0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(out.getDim().batch(), 1);
  EXPECT_EQ(out.getDim().channel(), 2);
  EXPECT_EQ(out.getDim().height(), 4);
  EXPECT_EQ(out.getDim().width(), 4);

  float *out_ptr, *golden;
  golden = in.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 32; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], golden[i]);
  }
}

/**
 * @brief Flatten Layer
 */
TEST(nntrainer_Flatten, backwarding_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::FlattenLayer layer;
  std::vector<std::string> input_str;
  nntrainer::TensorDim previous_dim;
  previous_dim.setTensorDim("2:3:4:4");
  layer.setInputDimension(previous_dim);

  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Tensor in(2, 1, 1, 48);
  nntrainer::Tensor out, result(2, 1, 1, 48);
  std::ifstream file("goldenPooling2DResult.out");
  in.read(file);
  file.close();

  out = layer.backwarding(in, 0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(out.getDim().batch(), 2);
  EXPECT_EQ(out.getDim().channel(), 3);
  EXPECT_EQ(out.getDim().height(), 4);
  EXPECT_EQ(out.getDim().width(), 4);

  float *out_ptr, *golden;
  golden = in.getData();
  out_ptr = out.getData();

  for (int i = 0; i < 48 * 2; ++i) {
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

TEST(nntrainer_ActivationLayer, init_01_1) {
  int status = ML_ERROR_NONE;
  nntrainer::ActivationLayer layer;
  status = layer.initialize(false);

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_ActivationLayer, setType_01_p) {
  nntrainer::ActivationLayer layer;
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_RELU));
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_SOFTMAX));
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_SIGMOID));
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_TANH));
}

TEST(nntrainer_ActivationLayer, setType_02_n) {
  nntrainer::ActivationLayer layer;
  EXPECT_THROW(layer.setActivation(nntrainer::ACT_UNKNOWN), std::runtime_error);
}

TEST(nntrainer_ActivationLayer, forward_backward_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::ActivationLayer layer;
  layer.setActivation(nntrainer::ACT_RELU);

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));
  nntrainer::Tensor expected(batch, channel, height, width);
  GEN_TEST_INPUT(expected, nntrainer::relu((l - 4) * 0.1 * (i + 1)));
  nntrainer::Tensor result = layer.forwarding(input, status);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_TRUE(result == expected);

  expected.copy(input);
  result = layer.backwarding(constant(1.0, 3, 1, 1, 10), 1);
  GEN_TEST_INPUT(expected, nntrainer::reluPrime(nntrainer::relu((l - 4) * 0.1 * (i + 1))));;
  EXPECT_TRUE(result == expected);
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
