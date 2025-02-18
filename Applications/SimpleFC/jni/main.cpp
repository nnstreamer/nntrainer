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
#include <base_properties.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <bits/fs_fwd.h>
#include <bits/fs_path.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <unistd.h>

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

/**
 * @brief Create network
 *
 * @return vector of layers that contain full graph of asynch
 */
std::vector<LayerHandle> createGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "1:1024:1024")}));

  for (int i = 0; i < 52; i++) {
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("unit", 2048),
        withKey("weight_initializer", "xavier_uniform"),
        withKey("bias_initializer", "zeros"),
        // withKey("disable_bias", "true")
        }));
  }

  return layers;
}

ModelHandle create() {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});

  for (auto &layer : createGraph()) {
    model->addLayer(layer);
  }

  return model;
}

double createAndRun(unsigned int epochs, unsigned int batch_size,
                    std::string swap_on_off, std::string look_ahaed) {

  // setup model
  ModelHandle model = create();
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("model_tensor_type", "FP16-FP16")});

  if (swap_on_off == "true") {
    model->setProperty({withKey("memory_swap", swap_on_off)});
    model->setProperty({withKey("memory_swap_lookahead", look_ahaed)});
  }

  model->compile(ml::train::ExecutionMode::INFERENCE);
  std::cout << "\033[31;5;32m" << "model Compiled" << "\033[0m"<<std::endl;

  model->initialize(ml::train::ExecutionMode::INFERENCE);
  std::cout << "\033[31;5;32m" << "model initialized" << "\033[0m"<<std::endl;

  unsigned int feature_size = 1 * 1024 * 1024;

  float input[1024 * 1024];

  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = (j / (float)feature_size);

  std::vector<float *> in;
  std::vector<float *> answer;

  in.push_back(input);

  auto start = std::chrono::system_clock::now();

  // to test asynch fsu, we do need save the model weight data in file
  std::string filePath = "./TEST2.bin";

  // model->save(filePath, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  model->load(filePath);
  std::cout << "\033[31;5;32m" << "model loaded" << "\033[0m"<<std::endl;

  answer = model->inference(1, in);

  std::cout << "\033[31;5;32m" << "model Inference END" << "\033[0m"<<std::endl;
  std::cout << "ANSWER : ";
  for (int i = 0; i < 4096; i++) {
    std::cout << (float)static_cast<_FP16>(answer[0][i]) << ", ";
  }
  std::cout << std::endl;

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  in.clear();
  return elapsed_seconds.count();
}

int main(int argc, char *argv[]) {

  std::string swap_on = "true";
  std::string look_ahead = "1";
  if (argc < 3) {
    std::cerr << "need more argc, executable swap_on look_ahead" << std::endl;
  }

  swap_on = argv[1];    // true or false
  look_ahead = argv[2]; // int

  unsigned int batch_size = 1;
  unsigned int epoch = 1;

  swap_on = "true";
  look_ahead = "2";

  try {

    auto ret = createAndRun(epoch, batch_size, swap_on, look_ahead);
    std::cout << "swap : " << swap_on << " look ahead : " << look_ahead << " "
              << ret << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }
  int status = EXIT_SUCCESS;
  return status;
}
