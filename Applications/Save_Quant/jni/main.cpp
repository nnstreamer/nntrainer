// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   21 Mar 2025
 * @brief  Test Application for quantized save
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <util_func.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
std::string filePath;

/**
 * @brief Create network
 *
 * @return vector of layers that contain full graph of asynch
 */
std::vector<LayerHandle> createGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "1:1:1024:1024")}));

  for (int i = 0; i < 56; i++) {
    layers.push_back(
      createLayer("fully_connected",
                  {nntrainer::withKey("unit", 1024),
                   nntrainer::withKey("weight_initializer", "xavier_uniform"),
                   nntrainer::withKey("disable_bias", "true")}));
  }

  return layers;
}

ModelHandle create() {
  ModelHandle model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});

  for (auto &layer : createGraph()) {
    model->addLayer(layer);
  }

  return model;
}

ModelHandle saveBin(unsigned int epochs, unsigned int batch_size) {
  ModelHandle model = create();
  model->setProperty({nntrainer::withKey("batch_size", batch_size),
                      nntrainer::withKey("epochs", epochs),
                      nntrainer::withKey("model_tensor_type", "FP32-FP32")});
  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  int status = model->setOptimizer(std::move(optimizer));
  if (status) {
    throw std::invalid_argument("failed to set optimizer!");
  }

  status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  std::cout << "SAVE FP32 model to QINT8 type" << std::endl;
  model->save(filePath, ml::train::TensorDim::DataType::QINT8);

  return model;
}

void createAndRun(unsigned int epochs = 1, unsigned int batch_size = 1) {

  auto model_fp32 = saveBin(epochs, batch_size);

  // setup model
  ModelHandle model = create();
  model->setProperty({nntrainer::withKey("batch_size", batch_size),
                      nntrainer::withKey("epochs", epochs),
                      nntrainer::withKey("model_tensor_type", "QINT8-FP32")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  int status = model->setOptimizer(std::move(optimizer));
  if (status) {
    throw std::invalid_argument("failed to set optimizer!");
  }

  status = model->compile(ml::train::ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  const unsigned int feature_size = 1 * 1024 * 1024;

  float input[feature_size];

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = (j / (float)feature_size);

  std::vector<float *> in;
  std::vector<float *> answer_qint;
  std::vector<float *> answer_fp32;

  in.push_back(input);

  // run qint8-fp32 model
  model->load(filePath);
  answer_qint = model->inference(1, in);

  // run fp32-fp32 model
  answer_fp32 = model_fp32->inference(1, in);

  // compare the value
  std::cout << "\n\n[QINT8 output]" << std::endl;
  for (unsigned int i = 0; i < 10; ++i)
    std::cout << answer_qint[0][i] << " ";

  std::cout << "\n\n[FP32 output]" << std::endl;
  for (unsigned int i = 0; i < 10; ++i)
    std::cout << answer_fp32[0][i] << " ";

  std::cout << "\n=========================\n";
  in.clear();
}

int main(int argc, char *argv[]) {

  unsigned int batch_size = 1;
  unsigned int epoch = 1;
  filePath = "model_qint8.bin";

  try {
    createAndRun();
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}