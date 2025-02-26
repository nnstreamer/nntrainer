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

TEST(fsu, simple_fc) {
  std::ofstream outFile("./simple_fc_test.bin", std::ios::binary);
  size_t size = (4096 * 4096 * 2 + 8192) * 3;
  char *random_data = static_cast<char *>(calloc(size, 1));
  for (size_t i = 0; i < size; i++) {
    random_data[i] = 0xAA;
  }
  outFile.write(reinterpret_cast<const char *>(random_data), size);
  free(random_data);
  outFile.close();

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "mse")});

  model->addLayer(ml::train::createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "1:1:4096")}));

  for (int i = 0; i < 3; i++) {
    model->addLayer(ml::train::createLayer(
      "fully_connected",
      {withKey("unit", 4096), withKey("weight_initializer", "xavier_uniform"),
       withKey("bias_initializer", "zeros")}));
  }

  model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                      withKey("memory_swap", "true"),
                      withKey("memory_swap_lookahead", "3"),
                      withKey("model_tensor_type", "FP16-FP16")});

  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  model->load("./simple_fc_test.bin");

  unsigned int feature_size = 4096;
  float input[4096];

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = j;

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  answer = model->inference(1, in);

  in.clear();
  remove("./simple_fc_test.bin");
}
