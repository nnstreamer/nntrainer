// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_interpreter.cpp
 * @date 02 April 2021
 * @brief interpreter test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <functional>
#include <gtest/gtest.h>
#include <memory>

#include <app_context.h>
#include <ini_interpreter.h>
#include <interpreter.h>
#include <layer.h>

#include <nntrainer_test_util.h>

using LayerReprentation = std::pair<std::string, std::vector<std::string>>;

auto &ac = nntrainer::AppContext::Global();

static std::shared_ptr<nntrainer::GraphRepresentation>
makeGraph(const std::vector<LayerReprentation> &layer_reps) {
  auto graph = std::make_shared<nntrainer::GraphRepresentation>();

  for (const auto &layer_representation : layer_reps) {
    std::shared_ptr<ml::train::Layer> layer = ac.createObject<ml::train::Layer>(
      layer_representation.first, layer_representation.second);
    graph->addLayer(std::static_pointer_cast<nntrainer::Layer>(layer));
  }

  return graph;
}

const std::string pathResolver(const std::string &path) {
  return getResPath(path, {"test", "test_models", "models"});
}

auto ini_interpreter =
  std::make_shared<nntrainer::IniGraphInterpreter>(ac, pathResolver);

/**
 * @brief nntrainer Interpreter Test setup
 *
 * @note Proposing an evolutional path of current test
 * 1. A reference graph vs given paramter
 * 2. A reference graph vs list of models
 * 3. A reference graph vs (pick two models) a -> b -> a graph, b -> a -> b
 * graph
 */
class nntrainerInterpreterTest
  : public ::testing::TestWithParam<
      std::tuple<std::shared_ptr<nntrainer::GraphRepresentation>, const char *,
                 std::shared_ptr<nntrainer::GraphInterpreter>>> {

protected:
  virtual void SetUp() {
    auto params = GetParam();

    reference = std::get<0>(params);
    file_path = pathResolver(std::get<1>(params));
    interpreter = std::move(std::get<2>(params));
  }

  std::shared_ptr<nntrainer::GraphRepresentation> reference;
  std::shared_ptr<nntrainer::GraphInterpreter> interpreter;
  std::string file_path;
};

/**
 * @brief Check two compiled graph is equal
 * @note later this will be more complicated (getting N graph and compare each
 * other)
 *
 */
TEST_P(nntrainerInterpreterTest, graphEqual) {
  std::cerr << "testing " << file_path << '\n';

  int status = reference->compile(nntrainer::LossType::LOSS_NONE);
  EXPECT_EQ(status, ML_ERROR_NONE);
  auto g = interpreter->deserialize(file_path);

  /// @todo: change this to something like graph::finalize
  status = g->compile(nntrainer::LossType::LOSS_NONE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /// @todo: make a graph equal
  /// 1. having same number of nodes
  /// 2. layer name is identical (this is too strict though)
  /// 3. attributes of layer is identical
  // EXPECT_EQ(*graph, *interpreter->deserialize(file_path));

  auto layers = g->getLayers();
  auto ref_layers = reference->getLayers();
  EXPECT_EQ(layers.size(), ref_layers.size());

  if (layers.size() == ref_layers.size()) {
    for (auto &layer : layers) {
      std::shared_ptr<nntrainer::Layer> ref_layer;
      EXPECT_NO_THROW(ref_layer = reference->getLayer(layer->getName()));

      /// @todo: layer->getProperties() and do check on each properties
    }
  }
}

auto fc0 = LayerReprentation("fully_connected",
                             {"name=fc0", "unit=1", "input_shape=1:1:100"});

auto flatten = LayerReprentation("flatten", {"name=flat"});

/**
 * @brief make ini test case from given parameter
 */
static std::tuple<std::shared_ptr<nntrainer::GraphRepresentation>, const char *,
                  std::shared_ptr<nntrainer::GraphInterpreter>>
mkTc(std::shared_ptr<nntrainer::GraphRepresentation> graph, const char *file,
     std::shared_ptr<nntrainer::GraphInterpreter> interpreter) {
  return std::make_tuple(graph, file, interpreter);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(nntrainerAutoInterpreterTest, nntrainerInterpreterTest,
                        ::testing::Values(
  mkTc(makeGraph({fc0, flatten}), "simple_fc.ini", ini_interpreter),
  mkTc(makeGraph({fc0, flatten}), "simple_fc_backbone.ini", ini_interpreter)
));
// clang-format on
