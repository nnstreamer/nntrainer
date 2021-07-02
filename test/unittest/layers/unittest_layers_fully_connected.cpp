// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_layers_fully_connected.cpp
 * @date 15 June 2021
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <fc_layer.h>
#include <layers_common_tests.h>

auto semantic_fc = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>,
  nntrainer::FullyConnectedLayer::type, {"unit=1"}, 0, false);

INSTANTIATE_TEST_CASE_P(FullyConnected, LayerSemantics,
                        ::testing::Values(semantic_fc));

INSTANTIATE_TEST_CASE_P(
  FullyConnected, LayerGoldenTest,
  ::testing::Values(
    "golden1") /**< format of type, properties, num_batch, golden file name */);
