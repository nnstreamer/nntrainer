// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_common_tests.cpp
 * @date 15 June 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <layers_common_tests.h>

#include <app_context.h>
#include <layer_devel.h>

constexpr unsigned SAMPLE_TRIES = 10;

LayerSemantics::~LayerSemantics() {}

void LayerSemantics::SetUp() {
  auto f = std::get<0>(GetParam());
  layer = std::move(f({}));
  std::tie(std::ignore, expected_type, valid_properties, invalid_properties,
           options) = GetParam();
}

void LayerSemantics::TearDown() {}

TEST_P(LayerSemantics, createFromAppContext_pn) {
  auto ac = nntrainer::AppContext::Global(); /// copy intended
  if (!(options & LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT)) {
    EXPECT_THROW(ac.createObject<nntrainer::Layer>(expected_type),
                 std::invalid_argument);
    ac.registerFactory<nntrainer::Layer>(std::get<0>(GetParam()));
  }
  EXPECT_EQ(ac.createObject<nntrainer::Layer>(expected_type)->getType(),
            expected_type);
}

TEST_P(LayerSemantics, setProperties_p) {
  /// @todo check if setProperties does not collide with layerNode designated
  /// properties
}

TEST_P(LayerSemantics, setPropertiesValidWithInvalid_n) {}

TEST_P(LayerSemantics, setPropertiesValidInvalidOnly_n) {}

TEST_P(LayerSemantics, finalizeTwice_p) {}

TEST_P(LayerGoldenTest, HelloWorld) { EXPECT_TRUE(true); }
