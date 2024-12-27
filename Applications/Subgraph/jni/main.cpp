// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   main.cpp
 * @date   27 Dec 2024
 * @brief  Test Application for subgraph weight sharing
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
 * @brief Create subgraph
 * @return vector of layers that contain subgraph
 */
std::vector<LayerHandle> createSubGraph(const std::string &scope) {

  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer("fully_connected", {
                                                    withKey("name", "fc_in"),
                                                    withKey("unit", 320),
                                                  }));
  layers.push_back(createLayer("fully_connected", {
                                                    withKey("name", "fc_out"),
                                                    withKey("unit", 320),
                                                  }));

  return layers;
}

int main(int argc, char *argv[]) {

  /** model */
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  /** number of subgraphs */
  const int n_sg = 3;

  /** add input layer */
  model->addLayer(
    ml::train::createLayer("input", {"name=input", "input_shape=1:1:320"}));

  /** create a subgraph structure */
  auto subgraph = createSubGraph("subgraph");

  for (unsigned int idx_sg = 0; idx_sg < n_sg; ++idx_sg) {
    model->addWithReferenceLayers(
      subgraph, "subgraph", {}, {"fc_in"}, {"fc_out"},
      ml::train::ReferenceLayersType::SUBGRAPH,
      {withKey("subgraph_idx", idx_sg), withKey("is_shared_subgraph", "true")});
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
