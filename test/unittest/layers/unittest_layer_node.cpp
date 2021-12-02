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

#include <fc_layer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <var_grad.h>

/**
 * @brief test distribute property
 */
TEST(nntrainer_LayerNode, setDistribute_01_p) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);

  EXPECT_EQ(false, lnode->getDistribute());

  EXPECT_NO_THROW(lnode->setProperty({"distribute=true"}));

  EXPECT_EQ(true, lnode->getDistribute());
}

/**
 * @brief test flatten property
 */
TEST(nntrainer_LayerNode, setFlatten_01_p) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  EXPECT_NO_THROW(lnode->setProperty({"flatten=true"}));
}

/**
 * @brief test finalize with wrong context
 */
TEST(nntrainer_LayerNode, finalize_01_n) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  EXPECT_ANY_THROW(lnode->finalize());
}

/**
 * @brief test finalize with wrong context
 */
TEST(nntrainer_LayerNode, finalize_02_n) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  lnode->setProperty({"name=abc"});
  EXPECT_ANY_THROW(lnode->finalize());
}

/**
 * @brief test finalize with wrong context
 */
TEST(nntrainer_LayerNode, finalize_03_n) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  lnode->setProperty({"input_shape=1:1:1"});
  EXPECT_ANY_THROW(lnode->finalize());
}

/**
 * @brief test finalize with right context
 */
TEST(nntrainer_LayerNode, finalize_04_p) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  lnode->setProperty({"input_shape=1:1:1", "name=abc", "unit=4"});
  EXPECT_NO_THROW(lnode->finalize());
}

/**
 * @brief test double finalize
 */
TEST(nntrainer_LayerNode, finalize_05_n) {
  auto lnode = nntrainer::createLayerNode(nntrainer::FullyConnectedLayer::type);
  lnode->setProperty({"input_shape=1:1:1", "name=abc", "unit=4"});
  EXPECT_NO_THROW(lnode->finalize());
  nntrainer::Var_Grad input = nntrainer::Var_Grad(
    nntrainer::TensorDim({1, 1, 1, 1}), nntrainer::Tensor::Initializer::NONE,
    true, false, "dummy");
  lnode->configureRunContext({}, {&input}, {}, {});
  EXPECT_ANY_THROW(lnode->finalize());
}
