// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   integration_test_fsu.cpp
 * @date   20 Dec 2024
 * @brief  Unit Test for Asynch FSU
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <app_context.h>
#include <gtest/gtest.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

void MakeWightFile(size_t size, std::string file_path) {
  std::ofstream outFile(file_path, std::ios::out | std::ios::binary);
  char *random_data = static_cast<char *>(calloc(size, 1));
  for (size_t i = 0; i < size; i++) {
    random_data[i] = 0xAA;
  }
  outFile.write(reinterpret_cast<const char *>(random_data), size);
  free(random_data);
  outFile.close();
}

void RemoveWeightFile(std::string file_path) { remove(file_path.c_str()); }

template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

void MakeAndRunModel(unsigned int feature_size, unsigned int layer_num,
                     unsigned int look_ahead, std::string file_path ) {

  ModelHandle _model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                              {withKey("loss", "mse")});
  _model->addLayer(ml::train::createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(feature_size))}));

  for (unsigned int i = 0; i < layer_num; i++) {
    _model->addLayer(ml::train::createLayer(
      "fully_connected", {withKey("unit", std::to_string(feature_size)),
                          withKey("weight_initializer", "xavier_uniform"),
                          withKey("bias_initializer", "zeros")}));
  }

  _model->setProperty(
    {withKey("batch_size", 1), withKey("epochs", 1),
     withKey("memory_swap", "true"),
     withKey("memory_swap_lookahead", std::to_string(look_ahead)),
     withKey("model_tensor_type", "FP16-FP16")});

  int status = _model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = _model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);
  _model->load(file_path);

  float *input = new float[feature_size];
  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = static_cast<float>(j);

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  answer = _model->inference(1, in);

  in.clear();
  delete[] input;
}

class LookAheadParm : public ::testing::TestWithParam<unsigned int> {};

TEST_P(LookAheadParm, simple_fc_FP16_FP16) {
  unsigned int look_ahead_parm = GetParam();
  unsigned int feature_size = 2048;
  unsigned int layer_num = 28;
  std::string file_path = "weight_file" + std::to_string(look_ahead_parm) + ".bin";

  MakeWightFile((feature_size * feature_size * 2 + (feature_size * 2)) *
                layer_num, file_path);
  EXPECT_NO_THROW(MakeAndRunModel(feature_size, layer_num, look_ahead_parm, file_path));
  RemoveWeightFile(file_path);
}

INSTANTIATE_TEST_SUITE_P(LookAheadParmTest, LookAheadParm,
                         ::testing::Values(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                           13, 14, 15, 16, 17, 18, 19, 20, 21,
                                           22, 23, 24, 25, 26, 27, 28));
