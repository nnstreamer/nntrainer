// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   main.cpp
 * @date   01 Jul 2024
 * @brief  Resnet Mixed Precision example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
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

#include <cifar_dataloader.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
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

std::vector<LayerHandle> createGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "3:32:32")}));

  layers.push_back(createLayer(
    "conv2d", {withKey("name", "conv0"), withKey("filters", 64),
               withKey("kernel_size", {3, 3}), withKey("stride", {1, 1}),
               withKey("padding", "same"), withKey("bias_initializer", "zeros"),
               withKey("weight_initializer", "xavier_uniform")}));

  layers.push_back(createLayer(
    "batch_normalization",
    {withKey("name", "first_bn_relu"), withKey("activation", "relu"),
     withKey("momentum", "0.9"), withKey("epsilon", "0.00001")}));

  layers.push_back(createLayer(
    "pooling2d", {withKey("name", "last_p1"), withKey("pooling", "average"),
                  withKey("pool_size", {4, 4}), withKey("stride", "4,4")}));

  layers.push_back(createLayer("flatten", {withKey("name", "last_f1")}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("unit", 100), withKey("activation", "softmax")}));

  return layers;
}

ModelHandle createModel() {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});
  for (auto &layer : createGraph()) {
    model->addLayer(layer);
  }
  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data,
                  UserDataType &valid_user_data) {
  // setup model
  ModelHandle model = createModel();
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", "mixed_model.bin")});

#ifdef ENABLE_FP16
  model->setProperty({"model_tensor_type=FP16-FP16"});
  model->setProperty({"loss_scale=17768"});
#endif

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
  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                    std::move(dataset_valid));
  model->train();
}

std::array<UserDataType, 2>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int simulated_data_size,
                        unsigned int data_split) {
  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));
  UserDataType valid_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));

  return {std::move(train_data), std::move(valid_data)};
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage: ./main [batchsize] [data_split] [epoch] \n"
              << "data size is assumed 512 for both train and validation\n";
    return EXIT_FAILURE;
  }

  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

  unsigned int batch_size = std::stoul(argv[1]);
  unsigned int data_split = std::stoul(argv[2]);
  unsigned int epoch = std::stoul(argv[3]);

  std::cout << "batch_size: " << batch_size << " data_split: " << data_split
            << " epoch: " << epoch << std::endl;

  std::array<UserDataType, 2> user_datas;

  try {
    user_datas = createFakeDataGenerator(batch_size, 512, data_split);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto &[train_user_data, valid_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data, valid_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  int status = EXIT_SUCCESS;
  return status;
}
