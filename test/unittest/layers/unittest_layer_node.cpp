// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file        unittest_nntrainer_layers.cpp
 * @date        02 July 2021
 * @brief       Unit test utility for layer node
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

/**
 * @brief test distribute property
 */
TEST(nntrainer_LayerNode, setDistribute_01_p) {
  int status = ML_ERROR_NONE;

  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);

  EXPECT_EQ(false, lnode->getDistribute());

  status = lnode->setProperty({"distribute=true"});
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_EQ(true, lnode->getDistribute());
}

/**
 * @brief test flatten property
 */
TEST(nntrainer_LayerNode, setFlatten_01_p) {
  int status = ML_ERROR_NONE;

  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  status = lnode->setProperty({"flatten=true"});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief test finalize with wrong context
 */
TEST(nntrainer_LayerNode, finalize_01_n) {
  int status = ML_ERROR_NONE;

  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  EXPECT_THROW(lnode->finalize(), std::runtime_error);
}

/**
 * @brief test finalize with right context
 */
TEST(nntrainer_LayerNode, finalize_02_p) {
  int status = ML_ERROR_NONE;

  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  lnode->setProperty("input_shape=1:1:1");
  EXPECT_NO_THROW(lnode->finalize());
}

/**
 * @brief test double finalize
 */
TEST(nntrainer_LayerNode, finalize_01_n) {
  int status = ML_ERROR_NONE;

  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  lnode->setProperty("input_shape=1:1:1");
  EXPECT_NO_THROW(lnode->finalize());
  EXPECT_THROW(lnode->finalize(), std::runtime_error);
}

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
