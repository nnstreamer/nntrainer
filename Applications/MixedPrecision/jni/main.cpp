// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   25 Jul 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is AlexNet Example with CIFAR100
 *
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <cifar_dataloader.h>
#include <model.h>

/**
 * @brief     Data size for each category
 */
const unsigned int num_class = 100;

const unsigned int num_train = 100;

const unsigned int num_val = 20;

const unsigned int batch_size = 128;

const unsigned int feature_size = 3072;

unsigned int train_count = 0;
unsigned int val_count = 0;

int width = 32;

int height = 32;

int channel = 3;

unsigned int seed;

std::string resource;

float training_loss = 0.0;
float validation_loss = 0.0;
float last_batch_loss = 0.0;

using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

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

std::array<UserDataType, 2>
createRealDataGenerator(const std::string &directory, unsigned int batch_size,
                        unsigned int data_split) {

  UserDataType train_data(new nntrainer::util::Cifar100DataLoader(
    directory + "/alex_trainingSet.dat", batch_size, 1));
  UserDataType valid_data(new nntrainer::util::Cifar100DataLoader(
    directory + "/alex_valSet.dat", batch_size, 1));

  return {std::move(train_data), std::move(valid_data)};
}

int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cout << "./nntrainer_alex alex.ini resource\n";
    exit(-1);
  }

  seed = time(NULL);
  srand(seed);

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];
  resource = args[1];

  std::array<UserDataType, 2> user_datas;

  try {
    user_datas = createRealDataGenerator(resource, batch_size, 1);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return 1;
  }

  auto &[train_user_data, valid_user_data] = user_datas;

  std::unique_ptr<ml::train::Dataset> dataset_train;
  try {
    dataset_train = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  } catch (const std::exception &e) {
    std::cerr << "Error during create train dataset: " << e.what() << std::endl;
    return 1;
  }

  std::unique_ptr<ml::train::Dataset> dataset_valid;
  try {
    dataset_valid = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());
  } catch (const std::exception &e) {
    std::cerr << "Error during create valid dataset: " << e.what() << std::endl;
    return 1;
  }

  /**
   * @brief     Neural Network Create & Initialization
   */

  ModelHandle model;
  try {
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  } catch (const std::exception &e) {
    std::cerr << "Error during create model: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI);
  } catch (const std::exception &e) {
    std::cerr << "Error during loadFromConfig: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during ininitialize: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                      std::move(dataset_train));
  } catch (const std::exception &e) {
    std::cerr << "Error during set train dataset: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                      std::move(dataset_valid));
  } catch (const std::exception &e) {
    std::cerr << "Error during set valid dataset: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->train();
    training_loss = model->getTrainingLoss();
    validation_loss = model->getValidationLoss();
    last_batch_loss = model->getLoss();
  } catch (const std::exception &e) {
    std::cerr << "Error during train: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
