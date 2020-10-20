// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	factory.cpp
 * @date	14 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is implementaion for factory builder interface for c++ API
 */

#include <memory>
#include <string>
#include <vector>

#include <databuffer.h>
#include <databuffer_factory.h>
#include <layer.h>
#include <layer_factory.h>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <optimizer.h>
#include <optimizer_factory.h>

namespace ml {
namespace train {

/**
 * @brief Factory creator with constructor for layer
 */
std::unique_ptr<Layer> createLayer(LayerType type,
                                   const std::vector<std::string> &properties) {
  std::unique_ptr<Layer> layer = nntrainer::createLayer(type);

  if (layer->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Set properties failed for layer");

  return layer;
}

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Optimizer>
createOptimizer(OptimizerType type,
                const std::vector<std::string> &properties) {
  std::unique_ptr<Optimizer> optimizer = nntrainer::createOptimizer(type);

  if (optimizer->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Set properties failed for optimizer");

  return optimizer;
}

/**
 * @brief Factory creator with constructor for model
 */
std::unique_ptr<Model> createModel(ModelType type,
                                   const std::vector<std::string> &properties) {
  std::unique_ptr<Model> model;
  switch (type) {
  case ModelType::NEURAL_NET:
    model = std::make_unique<nntrainer::NeuralNetwork>();
    break;
  default:
    throw std::invalid_argument("This type of model is not yet supported");
  }

  if (model->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Set properties failed for model");

  return model;
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset>
createDataset(DatasetType type, const std::vector<std::string> &properties) {
  std::unique_ptr<Dataset> dataset = nntrainer::createDataBuffer(type);

  if (dataset->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Set properties failed for dataset");

  return dataset;
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset> createDataset(DatasetType type, const char *train_file,
                                       const char *valid_file,
                                       const char *test_file) {
  std::unique_ptr<Dataset> dataset = createDataset(type);

  if (train_file && dataset->setDataFile(DatasetDataType::DATA_TRAIN,
                                         train_file) != ML_ERROR_NONE)
    throw std::invalid_argument("Invalid train file");

  if (valid_file && dataset->setDataFile(DatasetDataType::DATA_VAL,
                                         valid_file) != ML_ERROR_NONE)
    throw std::invalid_argument("Invalid valid file");

  if (test_file && dataset->setDataFile(DatasetDataType::DATA_TEST,
                                        test_file) != ML_ERROR_NONE)
    throw std::invalid_argument("Invalid test file");

  return dataset;
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset> createDataset(DatasetType type, datagen_cb train,
                                       datagen_cb valid, datagen_cb test) {
  std::unique_ptr<Dataset> dataset = createDataset(type);

  if (train && dataset->setGeneratorFunc(DatasetDataType::DATA_TRAIN, train) !=
                 ML_ERROR_NONE)
    throw std::invalid_argument("Invalid train data generator");

  if (valid && dataset->setGeneratorFunc(DatasetDataType::DATA_VAL, valid) !=
                 ML_ERROR_NONE)
    throw std::invalid_argument("Invalid valid data generator");

  if (test && dataset->setGeneratorFunc(DatasetDataType::DATA_TEST, test) !=
                ML_ERROR_NONE)
    throw std::invalid_argument("Invalid test data generator");

  return dataset;
}

} // namespace train
} // namespace ml
