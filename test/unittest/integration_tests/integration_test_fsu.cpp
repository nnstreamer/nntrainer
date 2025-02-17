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
#include <util_func.h>
#include <vector>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

TEST(fsu, simple_fc) {

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});

  model->addLayer(ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "1:1:320")}));
  for (int i = 0; i < 6; i++) {
    model->addLayer(ml::train::createLayer(
      "fully_connected",
      {nntrainer::withKey("unit", 1000),
       nntrainer::withKey("weight_initializer", "xavier_uniform"),
       nntrainer::withKey("bias_initializer", "zeros")}));
  }

  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {nntrainer::withKey("unit", 100),
     nntrainer::withKey("weight_initializer", "xavier_uniform"),
     nntrainer::withKey("bias_initializer", "zeros")}));

  model->setProperty({nntrainer::withKey("batch_size", 1),
                      nntrainer::withKey("epochs", 1),
                      nntrainer::withKey("memory_swap", "true"),
                      nntrainer::withKey("memory_swap_lookahead", "1"),
                      nntrainer::withKey("model_tensor_type", "FP16-FP16")});

  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  model->save("simplefc_weight_fp16_fp16_100.bin",
              ml::train::ModelFormat::MODEL_FORMAT_BIN);
  model->load("./simplefc_weight_fp16_fp16_100.bin");

  unsigned int feature_size = 320;
  float input[320];

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = j;

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  answer = model->inference(1, in);

  in.clear();
}
