// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   factory.cpp
 * @date   14 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is implementaion for factory builder interface for c++ API
 */

#include <memory>
#include <string>
#include <vector>

#include <app_context.h>
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

std::unique_ptr<Layer> createLayer(const LayerType &type,
                                   const std::vector<std::string> &properties) {
  const std::string &t = nntrainer::layerGetStrType(type);

  return createLayer(t, properties);
}

/**
 * @brief Factory creator with constructor for layer
 */
std::unique_ptr<Layer> createLayer(const std::string &type,
                                   const std::vector<std::string> &properties) {
  std::unique_ptr<nntrainer::LayerNode> layer =
    nntrainer::createLayerNode(type, properties);

  return layer;
}

std::unique_ptr<Optimizer>
createOptimizer(const OptimizerType &type,
                const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  const std::string &t = nntrainer::optimizerIntToStrType(type);
  return ac.createObject<Optimizer>(t, properties);
}

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Optimizer>
createOptimizer(const std::string &type,
                const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  return ac.createObject<Optimizer>(type, properties);
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
std::unique_ptr<Dataset> createDataset(DatasetType type, const char *file) {
  return nntrainer::createDataBuffer(type, file);
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset> createDataset(DatasetType type, datagen_cb cb,
                                       void *user_data) {
  return nntrainer::createDataBuffer(type, cb, user_data);
}

} // namespace train
} // namespace ml
