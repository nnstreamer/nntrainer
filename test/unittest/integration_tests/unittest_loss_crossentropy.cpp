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
#include <lite/core/c/common.h>
#include <model.h>
#include <optimizer.h>

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

TEST(crossentropy_loss, model_pass_test) {

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "cross")});

  std::shared_ptr<ml::train::Layer> input_layer = ml::train::createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "3:32:32")});

  std::shared_ptr<ml::train::Layer> fc_layer = ml::train::createLayer(
    "fully_connected",
    {withKey("unit", 100), withKey("activation", "softmax")});

  model->addLayer(input_layer);
  model->addLayer(fc_layer);

  model->setProperty({withKey("batch_size", 16), withKey("epochs", 10)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  int status = model->compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = model->initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(crossentropy_loss, model_fail_test) {

  std::unique_ptr<ml::train::Model> model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "cross")});

  std::shared_ptr<ml::train::Layer> input_layer = ml::train::createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "3:32:32")});
  std::shared_ptr<ml::train::Layer> fc_layer = ml::train::createLayer(
    "fully_connected", {withKey("unit", 100), withKey("activation", "relu")});

  model->addLayer(input_layer);
  model->addLayer(fc_layer);

  model->setProperty({withKey("batch_size", 16), withKey("epochs", 10)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));
  int status = model->compile();
  EXPECT_FALSE(status == ML_ERROR_NONE);
}
