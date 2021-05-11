// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   simpleshot_l2norm_test.cpp
 * @date   08 Jan 2021
 * @brief  test for simpleshot l2norm layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <app_context.h>
#include <layer_internal.h>
#include <manager.h>
#include <nntrainer_test_util.h>

#include <layers/l2norm.h>

namespace simpleshot {
namespace layers {

TEST(l2norm, simple_functions) {
  auto &app_context = nntrainer::AppContext::Global();
  app_context.registerFactory(nntrainer::createLayer<L2NormLayer>);

  auto c =
    app_context.createObject<nntrainer::Layer>("l2norm", {"input_shape=1:1:4"});

  std::unique_ptr<L2NormLayer> layer(static_cast<L2NormLayer *>(c.release()));

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer->setInputBuffers(manager.trackLayerInputs(
    layer->getType(), layer->getName(), layer->getInputDimension()));
  layer->setOutputBuffers(manager.trackLayerOutputs(
    layer->getType(), layer->getName(), layer->getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();
  auto t = MAKE_SHARED_TENSOR(randUniform(1, 1, 1, 4));

  {
    auto actual = layer->forwarding_with_val({t});
    EXPECT_EQ(*actual[0], t->divide(t->l2norm()));
  }
}

} // namespace layers
} // namespace simpleshot
