// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   simpleshot_centroid_knn.cpp
 * @date   08 Jan 2021
 * @brief  test for simpleshot centering layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <app_context.h>
#include <layer_node.h>
#include <manager.h>
#include <nntrainer_test_util.h>

#include <layers/centroid_knn.h>

namespace simpleshot {
namespace layers {

TEST(centroid_knn, simple_functions) {
  auto &app_context = nntrainer::AppContext::Global();
  app_context.registerFactory(nntrainer::createLayer<CentroidKNN>);

  auto lnode =
    nntrainer::createLayerNode(app_context.createObject<nntrainer::Layer>(
      "centroid_knn", {"num_class=5", "input_shape=1:1:3"}));
  auto &c = lnode->getObject();

  std::shared_ptr<CentroidKNN> layer = std::static_pointer_cast<CentroidKNN>(c);

  nntrainer::Manager manager{true, true, true, false};
  layer->initialize(manager);
  layer->setInputBuffers(manager.trackLayerInputs(
    lnode->getType(), lnode->getName(), layer->getInputDimension()));
  layer->setOutputBuffers(manager.trackLayerOutputs(
    lnode->getType(), lnode->getName(), layer->getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();

  auto &map = layer->getWeightsRef()[0].getVariableRef();
  auto &num_samples = layer->getWeightsRef()[1].getVariableRef();

  EXPECT_EQ(2U, layer->getNumWeights());
  EXPECT_EQ(nntrainer::TensorDim({5, 3}), map.getDim());
  EXPECT_EQ(nntrainer::TensorDim({5}), num_samples.getDim());

  // feature 1: 0, 1, 2
  auto feature1 = MAKE_SHARED_TENSOR(ranged(1, 1, 1, 3));
  auto label1 = MAKE_SHARED_TENSOR(constant(0, 1, 1, 1, 5));
  label1->setValue(0, 0, 0, 0, 1.0);

  // feature 2: 3, 2, 1
  auto feature2 =
    MAKE_SHARED_TENSOR(constant(3, 1, 1, 1, 3).subtract(ranged(1, 1, 1, 3)));
  auto label2 = MAKE_SHARED_TENSOR(constant(0, 1, 1, 1, 5));
  label2->setValue(0, 0, 0, 1, 1.0);

  // test feature 3: 5, 5, 5
  auto feature_test = MAKE_SHARED_TENSOR(constant(5, 1, 1, 1, 3));

  // forward feature 1
  layer->forwarding_with_val({feature1}, {label1});
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(map.getData()[i], i);
  }
  EXPECT_FLOAT_EQ(num_samples.getValue(0, 0, 0, 0), 1);

  // forward feature 1 again
  layer->forwarding_with_val({feature1}, {label1});
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(map.getData()[i], i);
  }
  EXPECT_FLOAT_EQ(num_samples.getValue(0, 0, 0, 0), 2);

  // forward feature 2
  layer->forwarding_with_val({feature2}, {label2});
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(map.getData()[i + 3], 3 - i);
  }
  EXPECT_FLOAT_EQ(num_samples.getValue(0, 0, 0, 0), 2);

  auto old_map = map.clone();
  auto old_num_samples = num_samples.clone();

  auto out = layer->forwarding_with_val({feature_test}, {}, false);

  // after inference map doesn't change
  EXPECT_EQ(map, old_map);
  EXPECT_EQ(num_samples, old_num_samples);

  EXPECT_NEAR(out[0]->getValue(0, 0, 0, 0), -7.0710, 1e-4);
  EXPECT_NEAR(out[0]->getValue(0, 0, 0, 1), -5.38516, 1e-4);
}

} // namespace layers
} // namespace simpleshot
