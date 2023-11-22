// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   5 July 2023
 * @brief  Example of multi-input dataloader
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <multi_loader.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

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

ModelHandle createMultiInputModel() {
  using ml::train::createLayer;
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "1:2:2")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "input1"), withKey("input_shape", "1:2:2")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "input2"), withKey("input_shape", "1:2:2")}));

  layers.push_back(
    createLayer("concat", {withKey("name", "concat0"), withKey("axis", "3"),
                           withKey("input_layers", "input0, input1, input2")}));

  layers.push_back(createLayer(
    "fully_connected", {withKey("unit", 5), withKey("activation", "softmax")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);
  data->next(input, label, last);
  return 0;
}

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data) {

  ModelHandle model = createMultiInputModel();

  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", "resnet_full.bin")});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));

  model->summarize(std::cout, ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);
  model->train();
}

std::array<UserDataType, 1>
createFakeMultiDataGenerator(unsigned int batch_size,
                             unsigned int simulated_data_size) {
  UserDataType train_data(new nntrainer::util::MultiDataLoader(
    {{batch_size, 1, 2, 2}, {batch_size, 1, 2, 2}, {batch_size, 1, 2, 2}},
    {{batch_size, 1, 1, 5}}, simulated_data_size));

  return {std::move(train_data)};
}

int main(int argc, char *argv[]) {
  unsigned int total_data_size = 16;
  unsigned int batch_size = 2;
  unsigned int epoch = 2;

  std::array<UserDataType, 1> user_datas;

  try {
    user_datas = createFakeMultiDataGenerator(batch_size, total_data_size);
    auto &[train_user_data] = user_datas;
    createAndRun(epoch, batch_size, train_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
