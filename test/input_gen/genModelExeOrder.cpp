/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file genModelExeOrder.cpp
 * @date 14 February 2023
 * @brief Generate execution order golden data for models
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstdlib>

#include <layer.h>
#include <model.h>
#include <vector>

#include "neuralnet.h"
#include "nntrainer_test_util.h"
#include <set>

nntrainer::NeuralNetwork
genModel(const std::vector<LayerRepresentation> &layers) {
  auto model = nntrainer::NeuralNetwork();
  auto graph = makeGraph(layers);

  for (auto &layer : graph) {
    model.addLayer(layer);
  }

  model.compile();
  model.initialize();

  return model;
}

void exportToFile(std::string name, nntrainer::NeuralNetwork &model) {
  std::string file_name = name + ".exegolden";
  std::ofstream file(file_name);

  auto graph = model.getNetworkGraph();
  for (unsigned int i = 0; i < graph.size(); ++i) {
    auto layer = graph.getSortedLayerNode(i);
    auto orders = graph.getLayerExecutionOrders(layer);
    for (auto &[name, ords] : orders) {
      file << name;
      std::set<unsigned int> set_ords(ords.begin(), ords.end());
      for (auto &o : set_ords) {
        file << ", " << o;
      }
      file << std::endl;
    }
  }

  file.close();
}

std::vector<std::tuple<std::string, nntrainer::NeuralNetwork>> test_models = {
  {"fc3_mse", genModel({{"input", {"name=in", "input_shape=1:1:3"}},
                        {"fully_connected",
                         {"name=fc1", "input_layers=in", "unit=10",
                          "activation=relu", "trainable=true"}},
                        {"fully_connected",
                         {"name=fc2", "input_layers=fc1", "unit=10",
                          "activation=relu", "trainable=true"}},
                        {"fully_connected",
                         {"name=fc3", "input_layers=fc2", "unit=2",
                          "activation=sigmoid", "trainable=true"}},
                        {"mse", {"name=mse", "input_layers=fc3"}}})},
  {"fc3_mse_nt1", genModel({{"input", {"name=in", "input_shape=1:1:3"}},
                            {"fully_connected",
                             {"name=fc1", "input_layers=in", "unit=10",
                              "activation=relu", "trainable=false"}},
                            {"fully_connected",
                             {"name=fc2", "input_layers=fc1", "unit=10",
                              "activation=relu", "trainable=true"}},
                            {"fully_connected",
                             {"name=fc3", "input_layers=fc2", "unit=2",
                              "activation=sigmoid", "trainable=true"}},
                            {"mse", {"name=mse", "input_layers=fc3"}}})},
  {"fc3_mse_nt2", genModel({{"input", {"name=in", "input_shape=1:1:3"}},
                            {"fully_connected",
                             {"name=fc1", "input_layers=in", "unit=10",
                              "activation=relu", "trainable=true"}},
                            {"fully_connected",
                             {"name=fc2", "input_layers=fc1", "unit=10",
                              "activation=relu", "trainable=false"}},
                            {"fully_connected",
                             {"name=fc3", "input_layers=fc2", "unit=2",
                              "activation=sigmoid", "trainable=true"}},
                            {"mse", {"name=mse", "input_layers=fc3"}}})},
  {"fc3_mse_nt3", genModel({{"input", {"name=in", "input_shape=1:1:3"}},
                            {"fully_connected",
                             {"name=fc1", "input_layers=in", "unit=10",
                              "activation=relu", "trainable=true"}},
                            {"fully_connected",
                             {"name=fc2", "input_layers=fc1", "unit=10",
                              "activation=relu", "trainable=true"}},
                            {"fully_connected",
                             {"name=fc3", "input_layers=fc2", "unit=2",
                              "activation=sigmoid", "trainable=false"}},
                            {"mse", {"name=mse", "input_layers=fc3"}}})},
};

int main(int argc, char **argv) {
  for (auto &[name, model] : test_models) {
    try {
      exportToFile(name, model);
    } catch (const std::exception &e) {
      ml_loge("Got error while export file. %s, %s", typeid(e).name(),
              e.what());
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
