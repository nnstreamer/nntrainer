// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   10 Dec 2024
 * @brief  Test Application for Asynch FSU
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
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
                  {nntrainer::withKey("unit", 2048),
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

void saveBin(std::string file_path) {
  ModelHandle model = create();
  model->setProperty({nntrainer::withKey("batch_size", 1),
                      nntrainer::withKey("epochs", 1),
                      nntrainer::withKey("model_tensor_type", "FP16-FP16")});
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

  model->save(file_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
}

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  std::string look_ahaed, std::string file_path) {

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});

  // Model with FSU option
  ModelHandle model_fsu = create();
  model_fsu->setProperty(
    {nntrainer::withKey("batch_size", batch_size),
     nntrainer::withKey("epochs", epochs),
     nntrainer::withKey("model_tensor_type", "FP16-FP16")});
  model_fsu->setProperty({nntrainer::withKey("fsu", "true")});
  model_fsu->setProperty({nntrainer::withKey("fsu_lookahead", look_ahaed)});
  model_fsu->setOptimizer(std::move(optimizer));
  model_fsu->compile(ml::train::ExecutionMode::INFERENCE);
  model_fsu->initialize(ml::train::ExecutionMode::INFERENCE);

  // Model without FSU option
  ModelHandle model_no_fsu = create();
  model_no_fsu->setProperty(
    {nntrainer::withKey("batch_size", batch_size),
     nntrainer::withKey("epochs", epochs),
     nntrainer::withKey("model_tensor_type", "FP16-FP16")});
  model_no_fsu->setOptimizer(std::move(optimizer));
  model_no_fsu->compile(ml::train::ExecutionMode::INFERENCE);
  model_no_fsu->initialize(ml::train::ExecutionMode::INFERENCE);

  const unsigned int feature_size = 1 * 1024 * 1024;

  std::vector<float> input(feature_size);

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = (j / (float)feature_size);

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input.data());

  if (std::filesystem::exists(file_path)) {
    model_fsu->load(file_path);
    model_no_fsu->load(file_path);
  } else {
    saveBin(file_path);
    model_fsu->load(file_path);
    model_no_fsu->load(file_path);
  }
  std::cout << "==============================================================="
               "==========================\n"
            << "| Operation             | FSU        | NO_FSU          | "
               "DIFF(ABS)       | NO_FSU/FSU(%) "
            << std::endl;

  double total_fsu_time = 0.0;
  double total_no_fsu_time = 0.0;

  for (int iteration = 1; iteration <= 5; iteration++) {
    auto fsu_iter_start = std::chrono::system_clock::now();
    answer = model_fsu->inference(1, in);
    auto fsu_iter_end = std::chrono::system_clock::now();

    auto no_fsu_iter_start = std::chrono::system_clock::now();
    answer = model_no_fsu->inference(1, in);
    auto no_fsu_iter_end = std::chrono::system_clock::now();

    std::chrono::duration<double> fsu_time = fsu_iter_end - fsu_iter_start;
    std::chrono::duration<double> no_fsu_time =
      no_fsu_iter_end - no_fsu_iter_start;

    std::cout << "-------------------------------------------------------------"
                 "----------------------------\n"
              << "| " << iteration << "-infer Time          | "
              << fsu_time.count() << "   | " << no_fsu_time.count()
              << "        | "
              << std::abs(fsu_time.count() - no_fsu_time.count())
              << "        | " << (no_fsu_time.count() / fsu_time.count()) * 100
              << " % \n";
    total_fsu_time += fsu_time.count();
    total_no_fsu_time += no_fsu_time.count();
  }

  std::cout << "---------------------------------------------------------------"
               "--------------------------\n"
            << "| total Infer Time      | " << total_fsu_time << "   | "
            << total_no_fsu_time << "        | " <<std::abs(total_fsu_time - total_no_fsu_time)
            << "        | " << (total_no_fsu_time / total_fsu_time) * 100
            << " % \n"
            << "==============================================================="
               "==========================\n";
  in.clear();
}

int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "need more argc, executable fsu_on look_ahead Weight_file_path"
              << std::endl;
  }

  std::string fsu_on = argv[1];     // true or false
  std::string look_ahead = argv[2]; // int
  std::string weight_file_path = argv[3];

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "==============================================================="
               "==========================\n"
            << "| LOOK_AHEAD ==>   " << look_ahead << "\n";

  unsigned int batch_size = 1;
  unsigned int epoch = 1;

  try {
    createAndRun(epoch, batch_size, look_ahead, weight_file_path);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }
  int status = EXIT_SUCCESS;
  return status;
}
