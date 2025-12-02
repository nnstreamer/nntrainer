// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
 *
 * @file   main.cpp
 * @date   26 Jan 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Layer Example with ini file
 *
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <cifar_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

unsigned int DATA_SIZE;
unsigned int BATCH_SIZE;
unsigned int INPUT_SHAPE[3];
unsigned int OUTPUT_SHAPE[3];
unsigned int seed;

float training_loss = 0.0;
float last_batch_loss = 0.0;

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);
  data->next(input, label, last);
  return 0;
}

std::array<UserDataType, 1> createFakeDataGenerator(unsigned int batch_size) {

  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]}},
    {{batch_size, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]}},
    DATA_SIZE));

  return {std::move(train_data)};
}

int main(int argc, char *argv[]) {
  int status = 0;
  seed = time(NULL);
  srand(seed);

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <model_config>" << std::endl;
    return -1;
  }

  auto config = argv[1];

  std::unique_ptr<ml::train::Model> model;

  try {
    model = createModel(ml::train::ModelType::NEURAL_NET);
  } catch (std::exception &e) {
    std::cerr << "Error while creating model! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI);
  } catch (std::exception &e) {
    std::cerr << "Error while loading model! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->compile();
  } catch (std::exception &e) {
    std::cerr << "Error while compiling model! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->initialize();
  } catch (std::exception &e) {
    std::cerr << "Error while initializing model! details: " << e.what()
              << std::endl;
    return 1;
  }

  auto input_dim = model->getInputDimension();
  auto output_dim = model->getOutputDimension();

  INPUT_SHAPE[0] = input_dim[0].channel();
  INPUT_SHAPE[1] = input_dim[0].height();
  INPUT_SHAPE[2] = input_dim[0].width();
  OUTPUT_SHAPE[0] = output_dim[0].channel();
  OUTPUT_SHAPE[1] = output_dim[0].height();
  OUTPUT_SHAPE[2] = output_dim[0].width();
  DATA_SIZE = input_dim[0].batch();
  BATCH_SIZE = input_dim[0].batch();

  std::array<UserDataType, 1> user_datas;

  try {
    user_datas = createFakeDataGenerator(DATA_SIZE);
  } catch (std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return 1;
  }

  auto &[train_user_data] = user_datas;

  std::unique_ptr<ml::train::Dataset> dataset_train;
  try {
    dataset_train = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  } catch (std::exception &e) {
    std::cerr << "uncaught error while creating dataset! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                      std::move(dataset_train));
  } catch (std::exception &e) {
    std::cerr << "uncaught error while setting dataset! details: " << e.what()
              << std::endl;
    return 1;
  }
  try {
    model->train();
    training_loss = model->getTrainingLoss();
    last_batch_loss = model->getLoss();

  } catch (std::exception &e) {
    std::cerr << "uncaught error while training! details: " << e.what()
              << std::endl;
    return 1;
  }

  std::cout << "Training is finished" << std::endl;

  return status;
}
