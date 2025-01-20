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
#include <array>
#include <chrono>
#include <ctime>
#include <gtest/gtest.h>
#include <iostream>
#include <layer.h>
#include <memory>
#include <model.h>
#include <optimizer.h>
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

  std::unique_ptr<ml::train::Model> model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model->addLayer(ml::train::createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "1:1:320")}));
  for (int i = 0; i < 6; i++) {
    model->addLayer(ml::train::createLayer(
      "fully_connected",
      {withKey("unit", 1000), withKey("weight_initializer", "xavier_uniform"),
       withKey("bias_initializer", "zeros")}));
  }
  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {withKey("unit", 100), withKey("weight_initializer", "xavier_uniform"),
     withKey("bias_initializer", "zeros")}));

  model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                      withKey("memory_swap", "true"),
                      withKey("memory_swap_lookahead", "1"),
                      withKey("model_tensor_type", "FP16-FP16")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  model->save("simplefc_weight_fp16_fp16_100.bin",
              ml::train::ModelFormat::MODEL_FORMAT_BIN);
  model->load("./simplefc_weight_fp16_fp16_100.bin");

  uint feature_size = 320;

  float input[320];

  for (uint j = 0; j < feature_size; ++j)
    input[j] = j;

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  answer = model->inference(1, in);

  in.clear();
}
