// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   10 Dec 2024
 * @brief  Test Application for Asynch FSU
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
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

#ifdef PROFILE
#include <profiler.h>
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

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
                          nntrainer::withKey("input_shape", "1:1:1024")}));

  for (int i = 0; i < 28; i++) {
    layers.push_back(
      createLayer("fully_connected",
                  {nntrainer::withKey("unit", 1024),
                   nntrainer::withKey("weight_initializer", "xavier_uniform"),
                   nntrainer::withKey("bias_initializer", "zeros")}));
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

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  std::string swap_on_off, std::string look_ahaed) {

  // setup model
  ModelHandle model = create();
  model->setProperty({nntrainer::withKey("batch_size", batch_size),
                      nntrainer::withKey("epochs", epochs),
                      nntrainer::withKey("model_tensor_type", "FP16-FP16")});

  model->setProperty({nntrainer::withKey("memory_swap", swap_on_off)});
  if (swap_on_off == "true") {
    model->setProperty(
      {nntrainer::withKey("memory_swap_lookahead", look_ahaed)});
  }

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

  unsigned int feature_size = 1 * 1 * 1024;

  float input[1 * 1024];

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = (j / (float)feature_size);

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

  // to test asynch fsu, we do need save the model weight data in file
  std::string filePath = "simplefc_weight_fp16_fp16_100.bin";
  if (std::filesystem::exists(filePath)) {
    model->load(filePath);

    auto load_end = std::chrono::system_clock::now();
    std::chrono::duration<double> load_elapsed_seconds = load_end - start;
    std::time_t load_end_time = std::chrono::system_clock::to_time_t(load_end);
    std::cout << "Load finished computation at " << std::ctime(&load_end_time)
              << "elapsed time: " << load_elapsed_seconds.count() << "s\n";
  } else {
    model->save(filePath, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    model->load(filePath);
  }

  answer = model->inference(1, in);

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  in.clear();
}

int main(int argc, char *argv[]) {

#ifdef PROFILE
  auto listener =
    std::make_shared<nntrainer::profile::GenericProfileListener>();
  nntrainer::profile::Profiler::Global().subscribe(listener);
#endif

  std::string swap_on = "true";
  std::string look_ahead = "1";
  if (argc < 3) {
    std::cerr << "need more argc, executable swap_on look_ahead" << std::endl;
  }

  swap_on = argv[1];    // true or false
  look_ahead = argv[2]; // int

  std::cout << "swap_on : " << swap_on << std::endl;
  std::cout << "look_ahead : " << look_ahead << std::endl;

  unsigned int batch_size = 1;
  unsigned int epoch = 1;

  try {
    createAndRun(epoch, batch_size, swap_on, look_ahead);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

#ifdef PROFILE
  std::cout << *listener;
#endif

  int status = EXIT_SUCCESS;
  return status;
}
