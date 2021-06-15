// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   simpleshot_centering_test.cpp
 * @date   08 Jan 2021
 * @brief  test for simpleshot centering layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <fstream>
#include <memory>

#include <app_context.h>
#include <layer_node.h>
#include <manager.h>
#include <nntrainer_test_util.h>

#include <layers/centering.h>

namespace simpleshot {
namespace layers {

TEST(centering, simple_functions) {
  std::ofstream file("feature.bin", std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.good());

  nntrainer::Tensor feature(4);
  feature.setRandNormal();
  feature.save(file);
  file.close();

  auto &app_context = nntrainer::AppContext::Global();
  app_context.registerFactory(nntrainer::createLayer<CenteringLayer>);

  auto lnode =
    nntrainer::createLayerNode(app_context.createObject<nntrainer::LayerV1>(
      "centering", {"feature_path=feature.bin", "input_shape=1:1:4"}));
  auto &c = lnode->getObject();

  std::shared_ptr<CenteringLayer> layer =
    std::static_pointer_cast<CenteringLayer>(c);

  nntrainer::Manager manager;

  manager.setInferenceInOutMemoryOptimization(false);
  std::ifstream stub;
  layer->initialize(manager);
  layer->read(stub);
  layer->setInputBuffers(manager.trackLayerInputs(
    lnode->getType(), lnode->getName(), layer->getInputDimension()));
  layer->setOutputBuffers(manager.trackLayerOutputs(
    lnode->getType(), lnode->getName(), layer->getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();
  auto t = MAKE_SHARED_TENSOR(randUniform(1, 1, 1, 4));

  {
    auto actual = layer->forwarding_with_val({t});
    EXPECT_EQ(*actual[0], t->subtract(feature));
  }

  int status = remove("feature.bin");
  ASSERT_EQ(status, 0);
}

} // namespace layers
} // namespace simpleshot
