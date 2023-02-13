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
 * @file        unittest_nntrainer_exe_order.cpp
 * @date        10 February 2023
 * @brief       execution order test
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jiho Chu <jiho.chu@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <memory>
#include <nntrainer_error.h>
#include <tuple>
#include <util_func.h>

#include <nntrainer_test_util.h>

/**
 * @brief nntrainerExeOrderTest for parametrized test
 *
 * @param std::string name of the model
 * @param nntrainer::NeuralNetwork model
 */
class nntrainerExeOrderTest
  : public ::testing::TestWithParam<
      std::tuple<std::string, nntrainer::NeuralNetwork>> {
public:
  virtual void SetUp() {
    name = std::get<std::string>(GetParam());
    model = std::get<nntrainer::NeuralNetwork>(GetParam());
  }

  virtual void TearDown() {}

  std::string getGoldenData(std::string name) {
    std::string file_name =
      getResPath(name, {"test", "unittest_models"}) + ".exegolden";
    std::ifstream file(file_name);
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  }

  bool compare(std::string golden_name, nntrainer::NeuralNetwork &model) {
    auto golden_data = getGoldenData(golden_name);
    std::stringstream generated;

    auto graph = model.getNetworkGraph();
    for (unsigned int i = 0; i < graph.size(); ++i) {
      auto layer = graph.getSortedLayerNode(i);
      auto orders = graph.getLayerExecutionOrders(layer);
      for (auto &[name, ords] : orders) {
        generated << name;
        std::set<unsigned int> set_ords(ords.begin(), ords.end());
        for (auto &o : set_ords) {
          generated << ", " << o;
        }
        generated << std::endl;
      }
    }

    EXPECT_STREQ(golden_data.c_str(), generated.str().c_str());

    return true;
  }

  std::string name;
  nntrainer::NeuralNetwork model;
};

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

TEST_P(nntrainerExeOrderTest, compare_p) { compare(name, model); }

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

GTEST_PARAMETER_TEST(ExecutionOrder, nntrainerExeOrderTest,
                     ::testing::ValuesIn(test_models));

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
