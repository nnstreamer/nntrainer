// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   main.cpp
 * @date   27 Dec 2024
 * @brief  Test Application for shared_from
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cifar_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <model_util.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

/**
 * @brief tain data callback
 */
int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

/**
 * @brief Create subgraph
 * @return vector of layers that contain subgraph
 */
std::vector<LayerHandle> createSubGraph(const std::string &scope,
                                        int subgraph_idx) {

  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", scope + "/fc_in" + std::to_string(subgraph_idx)),
     withKey("unit", 320),
     withKey("input_layers", "input/" + std::to_string(subgraph_idx)),
     withKey("shared_from", scope + "/fc_in0")}));
  layers.push_back(createLayer(
    "fully_connected",
    {
      withKey("name", scope + "/fc_out" + std::to_string(subgraph_idx)),
      withKey("unit", 320),
      withKey("input_layers", scope + "/fc_in" + std::to_string(subgraph_idx)),
      withKey("shared_from", scope + "/fc_out0"),
    }));
  layers.push_back(createLayer(
    "identity",
    {withKey("name", "input/" + std::to_string(subgraph_idx + 1))}));

  return layers;
}

int main(int argc, char *argv[]) {

  /** model */
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  /** number of subgraphs */
  const int n_sg = 3;

  /** add input layer */
  model->addLayer(
    ml::train::createLayer("input", {"name=input/0", "input_shape=1:1:320"}));

  /** add subgraphs with shared_from */
  for (auto idx_sg = 0; idx_sg < n_sg; ++idx_sg) {
    for (auto &layer : createSubGraph(std::string("subgraph"), idx_sg))
      model->addLayer(layer);
  }

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  /** model compilation */
  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("model compilation failed!");
  }

  /** model initialization */
  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("model initialization failed!");
  }

  /** check weight sharing from summary */
  model->summarize(std::cout, ML_TRAIN_SUMMARY_TENSOR);
}