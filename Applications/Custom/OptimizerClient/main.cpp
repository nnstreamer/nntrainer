// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   10 December 2021
 * @brief  This file contains the execution part of Optimizer Client example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <iostream>

#include <app_context.h>
#include <model.h>
#include <momentum.h>
#include <optimizer.h>
#include <string>

#define BATCH_SIZE 10
#define FEATURE_SIZE 100
#define NUM_CLASS 10

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int constant_generator_cb(float **outVec, float **outLabel, bool *last,
                          void *user_data) {
  static int count = 0;
  unsigned int i;
  unsigned int data_size = FEATURE_SIZE;

  for (i = 0; i < data_size; ++i) {
    outVec[0][i] = 2.0f;
  }

  for (i = 0; i < NUM_CLASS - 1; ++i) {
    outLabel[0][i] = 0.0f;
  }
  outLabel[0][0] = 1.0f;

  count++;
  if (count == 10) {
    *last = true;
    count = 0;
  } else {
    *last = false;
  }

  return ML_ERROR_NONE;
}

int main() {
  try {
    auto &app_context = nntrainer::AppContext::Global();
    /// registering custom optimizer for nntrainer to understand here
    /// @see app_context::registerFactory for the detail
    app_context.registerFactory(ml::train::createOptimizer<custom::Momentum>,
                                custom::Momentum::type);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  auto model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                           {"batch_size=" + std::to_string(BATCH_SIZE)});

  std::shared_ptr<ml::train::Dataset> dataset;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::vector<std::shared_ptr<ml::train::Layer>> layers;

  try {
    dataset = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                       constant_generator_cb);
  } catch (...) {
    std::cerr << "creating dataset failed";
    return 1;
  }

  try {
    if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset) !=
        0) {
      std::cerr << "failed to set datatset";
      return 1;
    }
  } catch (...) {
    std::cerr << "failed setting dataset";
    return 1;
  }

  try {
    ///////// attaching custom momentum
    optimizer = ml::train::createOptimizer("custom_momentum");
  } catch (...) {
    std::cerr << "creating optimizer failed";
    return 1;
  }

  try {
    if (model->setOptimizer(optimizer) != 0) {
      std::cerr << "failed to set optimizer";
      return 1;
    }
  } catch (...) {
    std::cerr << "failed setting dataset";
    return 1;
  }

  try {
    /// creating a simple model
    layers = std::vector<std::shared_ptr<ml::train::Layer>>{
      ml::train::layer::Input({"name=inputlayer", "input_shape=1:1:100"}),
      ml::train::layer::FullyConnected(
        {"name=outputlayer", "input_layers=inputlayer",
         "unit=" + std::to_string(NUM_CLASS), "bias_initializer=zeros",
         "activation=softmax"}),
      ml::train::createLayer("mse", {"name=mse_loss"})};
  } catch (nntrainer::exception::not_supported &e) {
    std::cerr << "creating model failed " << e.what();
    return 1;
  }

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  if (model->compile() != 0) {
    std::cerr << "model compilation failed";
    return 1;
  }

  if (model->initialize() != 0) {
    std::cerr << "model initiation failed";
    return 1;
  }

  if (model->train() != 0) {
    std::cerr << "train failed";
    return 1;
  }

  std::cout << "successfully ran";
  return 0;
}
