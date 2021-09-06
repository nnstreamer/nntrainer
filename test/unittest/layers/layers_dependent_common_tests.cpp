// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file layer_common_tests.cpp
 * @date 02 July 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <layers_common_tests.h>

#include <app_context.h>
#include <layer_devel.h>
#include <layer_node.h>

constexpr unsigned SAMPLE_TRIES = 10;

TEST_P(LayerSemantics, createFromAppContext_pn) {
  auto &ac = nntrainer::AppContext::Global();
  if (!(options & LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT)) {
    try {
      ac.createObject<nntrainer::Layer>(expected_type);
    } catch (...) {
      ac.registerFactory<nntrainer::Layer>(std::get<0>(GetParam()));
    }
  }

  EXPECT_EQ(ac.createObject<nntrainer::Layer>(expected_type)->getType(),
            expected_type);
}

TEST_P(LayerSemantics, setPropertiesInvalid_n) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(LayerSemantics, finalizeValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  lnode->setProperty({"input_shape=1:1:1", "name=test"});
  // /** purpose is to set number of outputs to 1 */
  // lnode->setOutputLayers({"dummy"});
  EXPECT_NO_THROW(lnode->setProperty(valid_properties));

  if (!must_fail) {
    nntrainer::InitLayerContext init_context = lnode->finalize();
    EXPECT_EQ(init_context.getOutputDimensions().size(),
              init_context.getNumOutputs());

    for (auto const &dim : init_context.getOutputDimensions())
      EXPECT_GT(dim.getDataLen(), size_t(0));
    for (auto const &ws : init_context.getWeightsSpec())
      EXPECT_GT(std::get<0>(ws).getDataLen(), size_t(0));
    for (auto const &ts : init_context.getTensorsSpec())
      EXPECT_GT(std::get<0>(ts).getDataLen(), size_t(0));
  } else {
    EXPECT_THROW(lnode->finalize(), nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemantics, getTypeValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  std::string type;

  EXPECT_NO_THROW(type = lnode->getType());
  EXPECT_GT(type.size(), size_t(0));
}

TEST_P(LayerSemantics, gettersValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);

  EXPECT_NO_THROW(lnode->supportInPlace());
  EXPECT_NO_THROW(lnode->requireLabel());
  EXPECT_NO_THROW(lnode->supportBackwarding());
}

TEST_P(LayerSemantics, setBatchValidateLayerNode_p) {
  auto lnode = nntrainer::createLayerNode(expected_type);
  lnode->setProperty({"input_shape=1:1:1", "name=test"});
  EXPECT_NO_THROW(lnode->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(lnode->finalize());
  } else {
    EXPECT_THROW(lnode->finalize(), nntrainer::exception::not_supported);
  }
}
