// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file integration_test_mixed_precision.cpp
 * @date 26 Nov 2024
 * @brief Mixed Precision integration test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tuple>

#include <gtest/gtest.h>

#include <app_context.h>
#include <layer.h>
#include <lite/core/c/common.h>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>
using namespace nntrainer;

TEST(mixed_precision, input_only_model_test) {

  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate = 0.1"}));

  EXPECT_EQ(nn->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(), ML_ERROR_NONE);
  EXPECT_EQ(nn->reinitialize(), ML_ERROR_NONE);
}

TEST(mixed_precision, loss_scale_test) {
  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

  EXPECT_THROW(nn->setProperty({"batch_size=1", "model_tensor_type=FP16-FP16",
                                "loss_scale=0"}),
               std::invalid_argument);

  EXPECT_NO_THROW(
    nn->setProperty(
      {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"}),
    std::invalid_argument);
}

TEST(mixed_precision, model_tensor_type_test) {
  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

  std::string positive_type_list[] = {"QINT4-FP16", "QINT4-FP32", "QINT8-FP16",
                                      "QINT8-FP32", "FP16-FP16",  "FP16-FP32",
                                      "FP32-FP16",  "FP32-FP32"};
  std::string negative_type_list[] = {"FP16-XXX", "XXX-XXX", "", "ttkt",
                                      "UINT8-UINT8"};

  for (auto type_item : positive_type_list) {
    EXPECT_NO_THROW(nn->setProperty(
      {"batch_size=1", "model_tensor_type=" + type_item, "loss_scale=65536"}));
  }

  for (auto type_item : negative_type_list) {
    EXPECT_THROW(
      nn->setProperty(
        {"batch_size=1", "model_tensor_type=" + type_item, "loss_scale=65536"}),
      std::invalid_argument);
  }
}
