// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_reshape.cpp
 * @date 19 October 2021
 * @brief Reshape Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <reshape_layer.h>

auto semantic_reshape = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ReshapeLayer>,
  nntrainer::ReshapeLayer::type, {"target_shape=-1"}, 0, false, 1);

GTEST_PARAMETER_TEST(Reshape, LayerSemantics,
                     ::testing::Values(semantic_reshape));
