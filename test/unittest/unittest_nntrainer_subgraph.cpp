// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file        unittest_subgraph.cpp
 * @date        22 Jan 2025
 * @brief       Unit test utility for subgraph.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Eunju Yang <ej.yang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <memory>
#include <nntrainer_error.h>

#define NNPTR(x) std::static_pointer_cast<nntrainer::NeuralNetwork>(x)

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

void graphEqual(const nntrainer::GraphLayerNodeRepresentation &lhs,
                const nntrainer::GraphLayerNodeRepresentation &rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());

  auto is_node_equal = [](const nntrainer::LayerNode &l,
                          const nntrainer::LayerNode &r) {
    nntrainer::Exporter lhs_export;
    nntrainer::Exporter rhs_export;

    l.exportTo(lhs_export, ml::train::ExportMethods::METHOD_STRINGVECTOR);
    r.exportTo(rhs_export, ml::train::ExportMethods::METHOD_STRINGVECTOR);

    /*** fixme, there is one caveat that order matters in this form */
    EXPECT_EQ(
      *lhs_export.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>(),
      *rhs_export.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>());
  };

  if (lhs.size() == rhs.size()) {
    auto lhs_iter = lhs.cbegin();
    auto rhs_iter = rhs.cbegin();
    while (lhs_iter != lhs.cend() && rhs_iter != rhs.cend()) {
      auto lhs = *lhs_iter;
      auto rhs = *rhs_iter;
      is_node_equal(*lhs.get(), *rhs.get());
      lhs_iter++;
      rhs_iter++;
    }
  }
}

/**
 * @brief Internal API test :
 * unittest for createSubGraph with subgraph_name property
 * `nntrainer::createSubGraph`
 */
TEST(nntrainer_SubGraph, prop_subgraph_name_01_p) {
  auto sg = nntrainer::createSubGraph();
  EXPECT_EQ(sg->getName(), "default");
}

TEST(nntrainer_SubGraph, prop_subgraph_name_02_p) {
  const std::string &n = "cpu_graph";
  auto sg = nntrainer::createSubGraph({"subgraph_name=" + n});
  EXPECT_EQ(sg->getName(), n);
}

TEST(nntrainer_SubGraph, prop_subgraph_name_03_p) {
  const std::string &n = "cpu_graph";
  auto sg = nntrainer::createSubGraph();
  sg->setName(n);
  EXPECT_EQ(sg->getName(), n);
}

TEST(nntrainer_SubGraph, prop_subgraph_name_01_n) {
  const std::string &n = "cpu_graph";
  auto sg = nntrainer::createSubGraph({"name=" + n}); // keyword : subgraph_name
  EXPECT_NE(sg->getName(), n);
}

/**
 * @brief External API test :
 * unittest for createSubGraph with subgraph_name property
 * `ml::train::createSubGraph`
 */
TEST(nntrainer_SubGraph, create_subgraph_01_p) {
  auto sg = ml::train::createSubGraph("subgraph");
  EXPECT_EQ(sg->getName(), "default");
}

TEST(nntrainer_SubGraph, create_subgraph_02_p) {
  auto sg = ml::train::createSubGraph("subgraph", {"subgraph_name=cpu_graph"});
  EXPECT_EQ(sg->getName(), "cpu_graph");
}

/**
 * @brief Unittest to create a subgraph with layers
 * create a graph
 */
TEST(nntrainer_SubGraph, create_subgraph_03_p) {
  static auto &ac = nntrainer::AppContext::Global();

  // subgraph named `default`
  std::shared_ptr<ml::train::Model> model1 =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model1->addSubGraph(ml::train::createSubGraph("subgraph"));
  model1->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc0"), withKey("unit", 2)}));

  // subgraph named `default`
  std::shared_ptr<ml::train::Model> model2 =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model2->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc0"), withKey("unit", 2)}));

  graphEqual(NNPTR(model1)->getFlatGraph(), NNPTR(model2)->getFlatGraph());
}

/**
 * create a graph
 */
TEST(nntrainer_SubGraph, create_subgraph_04_p) {
  static auto &ac = nntrainer::AppContext::Global();

  // 1. create a default subgraph
  std::shared_ptr<ml::train::Model> model1 =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  std::shared_ptr<ml::train::SubGraph> sg =
    ml::train::createSubGraph("subgraph");
  // 2. add three layers to subgraph
  sg->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc0"), withKey("unit", 2)}));
  sg->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc1"), withKey("unit", 2)}));
  sg->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc2"), withKey("unit", 2)}));
  // 3. add a subgraph to model
  model1->addSubGraph(sg);

  // add three layers to model directly
  // It implies a default subgraph creation and adding layers to the defulat
  // subgraph
  std::shared_ptr<ml::train::Model> model2 =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model2->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc0"), withKey("unit", 2)}));
  model2->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc1"), withKey("unit", 2)}));
  model2->addLayer(ml::train::createLayer(
    "fully_connected", {withKey("name", "fc2"), withKey("unit", 2)}));

  graphEqual(NNPTR(model1)->getFlatGraph(), NNPTR(model2)->getFlatGraph());
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
