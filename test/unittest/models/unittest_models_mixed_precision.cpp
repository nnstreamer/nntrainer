// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_models_mixed_precision.cpp
 * @date 3 May 2024
 * @brief unittest models to cover mixed precision
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static std::unique_ptr<NeuralNetwork> fc_mixed_training() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=2", "model_tensor_type=FP16-FP16", "loss_scale=128"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3"}},
    {"Fully_connected", {"name=fc", "input_layers=in", "unit=10"}},
    {"mse", {"name=loss", "input_layers=fc"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate = 0.1"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> multi_fc_mixed_training() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=2", "model_tensor_type=FP16-FP16", "loss_scale=128"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:28:28"}},
    {"Fully_connected", {"name=fc1", "input_layers=in", "unit=100"}},
    {"Fully_connected", {"name=fc2", "input_layers=fc1", "unit=50"}},
    {"Fully_connected", {"name=fc3", "input_layers=fc2", "unit=10"}},
    {"mse", {"name=loss", "input_layers=fc3"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }
  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate=0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> flatten_fc_mixed_training() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=2", "model_tensor_type=FP16-FP16", "loss_scale=128"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:28:28"}},
    {"flatten", {"name=flatten", "input_layers=in"}},
    {"Fully_connected", {"name=fc", "input_layers=flatten", "unit=10"}},
    {"mse", {"name=loss", "input_layers=fc"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }
  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate=0.1"}));
  return nn;
}

GTEST_PARAMETER_TEST(
  MixedPrecision, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelTc_V2(fc_mixed_training, "fc_mixed_training",
                 ModelTestOption::NO_THROW_RUN_V2),
    mkModelTc_V2(multi_fc_mixed_training, "multi_fc_mixed_training",
                 ModelTestOption::NO_THROW_RUN_V2),
    mkModelTc_V2(flatten_fc_mixed_training, "flatten_fc_mixed_training",
                 ModelTestOption::NO_THROW_RUN_V2),
    /** ModelTestOption::ALL_V2),
     * Disabled for now to check
     */
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info)
    -> const auto & { return std::get<1>(info.param); });
