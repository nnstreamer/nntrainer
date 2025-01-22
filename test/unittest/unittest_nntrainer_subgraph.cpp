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
#include <nntrainer_error.h>
#include <subgraph.h>

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
