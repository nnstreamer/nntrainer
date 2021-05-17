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

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tflite_interpreter.h>
#endif

#include <nntrainer_test_util.h>

using LayerReprentation = std::pair<std::string, std::vector<std::string>>;

auto &ac = nntrainer::AppContext::Global();

static std::shared_ptr<nntrainer::GraphRepresentation>
makeGraph(const std::vector<LayerReprentation> &layer_reps) {
  auto graph = std::make_shared<nntrainer::GraphRepresentation>();

  for (const auto &layer_representation : layer_reps) {
    /// @todo Use unique_ptr here
    std::shared_ptr<nntrainer::Layer> nntr_layer =
      ac.createObject<nntrainer::Layer>(layer_representation.first,
                                        layer_representation.second);
    std::shared_ptr<nntrainer::LayerNode> layer =
      std::make_unique<nntrainer::LayerNode>(nntr_layer);
    graph->addLayer(layer);
  }

  return graph;
}

const std::string pathResolver(const std::string &path) {
  return getResPath(path, {"test", "test_models", "models"});
}

auto ini_interpreter =
  std::make_shared<nntrainer::IniGraphInterpreter>(ac, pathResolver);

/**
 * @brief prototypical version of checking graph is equal
 *
 * @param lhs compiled(later, finalized) graph to be compared
 * @param rhs compiled(later, finalized) graph to be compared
 * @return true graph is equal
 * @return false graph is not equal
 */
static void graphEqual(const nntrainer::GraphRepresentation &lhs,
                       const nntrainer::GraphRepresentation &rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());

  auto is_node_equal = [](const nntrainer::Layer &l,
                          const nntrainer::Layer &r) {
    nntrainer::Exporter lhs_export;
    nntrainer::Exporter rhs_export;

    l.export_to(lhs_export);
    r.export_to(rhs_export);

    /*** fixme, there is one caveat that order matters in this form */
    EXPECT_EQ(
      lhs_export.get_result<nntrainer::ExportMethods::METHOD_STRINGVECTOR>(),
      rhs_export.get_result<nntrainer::ExportMethods::METHOD_STRINGVECTOR>());
  };

  if (lhs.size() == rhs.size()) {
    auto lhs_iter = lhs.cbegin();
    auto rhs_iter = rhs.cbegin();
    for (; lhs_iter != lhs.cend(), rhs_iter != rhs.cend();
         lhs_iter++, rhs_iter++) {
      is_node_equal(*lhs_iter->getObject(), *rhs_iter->getObject());
    }
  }
}

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

  /// @todo: make a proper graph equal
  /// 1. having same number of nodes
  /// 2. layer name is identical (this is too strict though)
  /// 3. attributes of layer is identical
  graphEqual(*g, *reference);
}

/**
 * @brief graph serialize after deserialize, compare if they are the same
 *
 */
TEST_P(nntrainerInterpreterTest, graphSerializeAfterDeserialize) {
  auto g = interpreter->deserialize(file_path);

  auto out_file_path = file_path + ".out";

  /// @todo: change this to something like graph::finalize
  int status = g->compile(nntrainer::LossType::LOSS_NONE);
  EXPECT_EQ(status, ML_ERROR_NONE);
  interpreter->serialize(g, out_file_path);
  auto new_g = interpreter->deserialize(out_file_path);

  graphEqual(*g, *new_g);

  EXPECT_EQ(remove(out_file_path.c_str()), 0) << strerror(errno);
}

auto fc0 = LayerReprentation("fully_connected",
                             {"name=fc0", "unit=2", "input_shape=1:1:10"});
auto fc1 = LayerReprentation("fully_connected", {"name=fc1", "unit=2"});

auto flatten = LayerReprentation("flatten", {"name=flat"});

#ifdef ENABLE_TFLITE_INTERPRETER
TEST(flatbuffer, playground) {

  auto manager = std::make_shared<nntrainer::Manager>();

  nntrainer::TfliteInterpreter interpreter;
  auto g = makeGraph({fc0, fc1});
  EXPECT_EQ(g->compile(nntrainer::LossType::LOSS_NONE), ML_ERROR_NONE);
  EXPECT_EQ(g->initialize(manager), ML_ERROR_NONE);

  manager->initializeWeights();
  manager->allocateWeights();

  interpreter.serialize(g, "test.tflite");

  manager->deallocateWeights();
}
#endif
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
