// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_loss_crossentropy.cpp
 * @date 16 Oct 2024
 * @brief CrossEntropy loss Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tuple>

#include <gtest/gtest.h>

#include <app_context.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <util_func.h>

TEST(crossentropy_loss, model_pass_test) {

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  std::shared_ptr<ml::train::Layer> input_layer = ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "3:32:32")});

  std::shared_ptr<ml::train::Layer> fc_layer = ml::train::createLayer(
    "fully_connected", {nntrainer::withKey("unit", 100),
                        nntrainer::withKey("activation", "softmax")});

  model->addLayer(input_layer);
  model->addLayer(fc_layer);

  model->setProperty(
    {nntrainer::withKey("batch_size", 16), nntrainer::withKey("epochs", 10)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  int status = model->compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = model->initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(crossentropy_loss, model_fail_test) {

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  std::shared_ptr<ml::train::Layer> input_layer = ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "3:32:32")});
  std::shared_ptr<ml::train::Layer> fc_layer = ml::train::createLayer(
    "fully_connected", {nntrainer::withKey("unit", 100),
                        nntrainer::withKey("activation", "relu")});

  model->addLayer(input_layer);
  model->addLayer(fc_layer);

  model->setProperty(
    {nntrainer::withKey("batch_size", 16), nntrainer::withKey("epochs", 10)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  int status = model->compile();
  EXPECT_FALSE(status == ML_ERROR_NONE);
}

TEST(kld_loss, compile_test) {

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "kld")});

  std::shared_ptr<ml::train::Layer> input_layer = ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "3:32:32")});
  std::shared_ptr<ml::train::Layer> fc_layer = ml::train::createLayer(
    "fully_connected", {nntrainer::withKey("unit", 100),
                        nntrainer::withKey("activation", "softmax")});

  model->addLayer(input_layer);
  model->addLayer(fc_layer);

  model->setProperty(
    {nntrainer::withKey("batch_size", 16), nntrainer::withKey("epochs", 1)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  int status = model->compile();
  EXPECT_FALSE(status == ML_ERROR_NONE);

  status = model->initialize();
  EXPECT_FALSE(status == ML_ERROR_NONE);
}
